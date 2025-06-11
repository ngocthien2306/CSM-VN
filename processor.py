import random
from typing import Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizer, ProcessorMixin


class CSMProcessor(ProcessorMixin):
    def __init__(self, tokenizer: PreTrainedTokenizer, audio_tokenizer):
        """
        Initialize with text and multi-codebook audio tokenizers.

        Args:
            tokenizer: Tokenizer for text (e.g., Llama tokenizer).
            audio_tokenizer: Tokenizer that converts audio into multiple codebooks.
        """
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.sample_rate = getattr(self.audio_tokenizer, "sample_rate", 24000)
        
        # Default amortization ratio
        self.amortization_ratio = 16

    def __call__(
        self,
        messages=None,
        text=None,
        audios=None,
        speaker_id=None,
        return_tensors="pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 2048,
        amortize_decoder_training: bool = True,
        amortization_ratio: int = 16,
        messages_training_mask: Optional[
            Union[List[int], List[bool], List[List[int]], List[List[bool]]]
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert conversation messages (and optional audio) into input tensors.

        Produces:
          - "input_ids": [B, S, 33] token ids.
          - "attention_mask": [B, S, 33] binary mask.
          - "labels": [B, S, 33] targets with -100 for positions to ignore.

        Args:
            messages: Conversation messages (dicts) with text/audio content.
            text: Single text string (used when messages is None).
            audios: Audio tensors corresponding to messages.
            speaker_id: Used in single-text mode to identify the speaker.
            return_tensors: Output tensor format.
            padding: Pad sequences to the same length.
            truncation: Truncate sequences longer than max_length.
            max_length: Maximum allowed sequence length.
            amortize_decoder_training: If True, only a subset of frames get decoder labels.
            amortization_ratio: Determines fraction of frames used for decoder training.
            messages_training_mask: Boolean or int mask to disable training for certain messages.

        Returns:
            A dict with keys "input_ids", "attention_mask", and "labels".
        """
        # Update amortization ratio
        self.amortization_ratio = amortization_ratio
        
        if messages is not None:
            # Ensure messages and audios are batched.
            is_batched = isinstance(messages[0], list) if messages else False

            if not is_batched:
                messages = [messages]
                audios = [audios] if audios is not None else [None]
            elif audios is not None and not isinstance(audios[0], list):
                audios = [audios]

            if messages_training_mask is not None:
                if not is_batched:
                    if isinstance(messages_training_mask[0], list):
                        raise ValueError(
                            "`messages_training_mask` is nested but expected flat for a single conversation."
                        )
                    messages_training_mask = [messages_training_mask]

            batch_outputs = []
            for i, convo_messages in enumerate(messages):
                convo_audios = audios[i] if i < len(audios) else None
                convo_mask = None
                if messages_training_mask is not None:
                    if i >= len(messages_training_mask):
                        raise ValueError(
                            f"messages_training_mask has {len(messages_training_mask)} entries but {len(messages)} conversations were provided."
                        )
                    convo_mask = messages_training_mask[i]

                batch_outputs.append(
                    self._process_messages(
                        convo_messages,
                        convo_audios,
                        return_tensors,
                        padding,
                        truncation,
                        max_length,
                        amortize_decoder_training,
                        amortization_ratio,
                        convo_mask,
                    )
                )

            if return_tensors == "pt":
                if batch_outputs:
                    max_seq_len = max(
                        output["input_ids"].size(0) for output in batch_outputs
                    )
                    padded_inputs, padded_masks, padded_labels = [], [], []

                    for output in batch_outputs:
                        seq_len = output["input_ids"].size(0)
                        if seq_len < max_seq_len and padding:
                            padded_input = torch.zeros(max_seq_len, 33).long()
                            padded_mask = torch.zeros(max_seq_len, 33)
                            padded_label = torch.full((max_seq_len, 33), -100).long()

                            padded_input[max_seq_len - seq_len :] = output["input_ids"]
                            padded_mask[max_seq_len - seq_len :] = output[
                                "attention_mask"
                            ]
                            padded_label[max_seq_len - seq_len :] = output["labels"]
                        else:
                            padded_input = output["input_ids"]
                            padded_mask = output["attention_mask"]
                            padded_label = output["labels"]

                        padded_inputs.append(padded_input.unsqueeze(0))
                        padded_masks.append(padded_mask.unsqueeze(0))
                        padded_labels.append(padded_label.unsqueeze(0))

                    return {
                        "input_ids": torch.cat(padded_inputs, dim=0),
                        "attention_mask": torch.cat(padded_masks, dim=0),
                        "labels": torch.cat(padded_labels, dim=0),
                    }
                else:
                    return {
                        "input_ids": torch.zeros(0, 0, 33),
                        "attention_mask": torch.zeros(0, 0, 33),
                        "labels": torch.zeros(0, 0, 33, dtype=torch.long),
                    }
            else:
                raise ValueError(f"Unsupported return format: {return_tensors}")

        elif text is not None and speaker_id is not None:
            # Wrap single text input as a conversation message.
            message = {
                "role": f"speaker_{speaker_id}",
                "content": [{"type": "text", "text": text}],
            }
            return self.__call__(
                [message],
                audios,
                return_tensors,
                padding,
                truncation,
                max_length,
                amortize_decoder_training,
                amortization_ratio,
            )
        else:
            raise ValueError(
                "Must provide either 'messages' or both 'text' and 'speaker_id'."
            )

    def _process_messages(
        self,
        messages: List[Dict],
        audios: List[torch.Tensor],
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
        amortize_decoder_training: bool = True,
        amortization_ratio: int = 16,
        messages_training_mask: Optional[Union[List[int], List[bool]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert a single conversation into token, mask, and label tensors.

        For each message:
          - Text: Encode with a speaker tag; output zeros in columns 0–31 and place the text token in column 32.
          - Audio: Encode audio into codebook tokens for columns 0–31; set column 32 to 0.

        Also applies:
          - Message-level masking via messages_training_mask.
          - Decoder label amortization (subsample frames for decoder training).

        Returns:
            Dict with "input_ids", "attention_mask", and "labels" tensors of shape [S, 33].
        """
        device = next(self.audio_tokenizer.parameters()).device

        all_tokens = []
        all_masks = []
        audio_index = 0
        message_boundaries = []

        for msg_idx, message in enumerate(messages):
            speaker_id = int(message["role"].split("_")[-1])
            keep_message = (
                True
                if messages_training_mask is None
                else bool(messages_training_mask[msg_idx])
            )

            text_content = []
            has_audio_content = False

            for item in message["content"]:
                if item["type"] == "text" and item.get("text", ""):
                    text_content.append(item["text"])
                elif item["type"] == "audio":
                    has_audio_content = True

            text = " ".join(text_content)
            start_idx = sum(chunk.size(0) for chunk in all_tokens)

            # Process text: encode and place token in column 32.
            if text:
                # Encode text with explicit BOS/EOS tokens
                text_tokens = self.tokenizer.encode(
                    f"[{speaker_id}]{text}", add_special_tokens=True
                )
                # Create frames for text tokens (zeros in audio columns, text in last column)
                text_frame = torch.zeros(len(text_tokens), 33).long()
                text_frame_mask = torch.zeros(len(text_tokens), 33, dtype=torch.int)

                text_frame[:, -1] = torch.tensor(text_tokens)
                text_frame_mask[:, -1] = 1

                all_tokens.append(text_frame)
                all_masks.append(text_frame_mask)

            # Process audio: encode into codebook tokens for columns 0–31.
            if (
                has_audio_content
                and audios
                and audio_index < len(audios)
                and audios[audio_index] is not None
            ):
                audio_tensor = audios[audio_index]
                audio_index += 1

                if not isinstance(audio_tensor, torch.Tensor):
                    raise ValueError(
                        f"Audio must be torch.Tensor, got {type(audio_tensor)}"
                    )

                with torch.no_grad():
                    # Ensure correct format for audio tokenizer
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
                    elif audio_tensor.dim() == 2:
                        audio_tensor = audio_tensor.unsqueeze(0)  # [1, C, T]
                    
                    audio_tokens = self.audio_tokenizer.encode(audio_tensor.to(device))[0]

                # Append EOS as an extra column.
                eos_frame = torch.zeros(audio_tokens.size(0), 1, device=device)
                audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

                audio_frame = torch.zeros(audio_tokens.size(1), 33).long()
                audio_frame_mask = torch.zeros(
                    audio_tokens.size(1), 33, dtype=torch.int
                )
                audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
                audio_frame_mask[:, :-1] = True

                all_tokens.append(audio_frame)
                all_masks.append(audio_frame_mask)
            elif has_audio_content:
                message_id = message.get("role", "unknown")
                print(
                    f"Warning: Audio content declared but no audio tensor provided for message with {message_id}"
                )

            end_idx = sum(chunk.size(0) for chunk in all_tokens)
            message_boundaries.append((start_idx, end_idx, keep_message))

        if audios and audio_index < len(audios):
            print(f"Warning: {len(audios) - audio_index} audio tensors were not used")

        # Merge tokens and masks; truncate if sequence exceeds max_length.
        if all_tokens:
            tokens = torch.cat(all_tokens, dim=0)
            tokens_mask = torch.cat(all_masks, dim=0)
            if truncation and tokens.size(0) > max_length:
                tokens = tokens[-max_length:]
                tokens_mask = tokens_mask[-max_length:]
        else:
            tokens = torch.zeros(0, 33).long()
            tokens_mask = torch.zeros(0, 33)

        # Create labels: mask positions where attention_mask is 0 and in the text column.
        labels = tokens.clone()
        labels = labels.masked_fill(tokens_mask == 0, -100)
        labels[:, -1] = -100

        # Apply message-level masking.
        for start_idx, end_idx, keep_msg in message_boundaries:
            if start_idx >= labels.size(0):
                break
            if end_idx > labels.size(0):
                end_idx = labels.size(0)
            if not keep_msg:
                labels[start_idx:end_idx, :] = -100

        # Amortize decoder training: retain decoder labels for only a subset of frames.
        if amortize_decoder_training:
            seq_len = labels.shape[0]
            valid_frames = torch.any(labels[:, :-1] != -100, dim=-1)
            valid_indices = torch.where(valid_frames)[0]

            if len(valid_indices) > 0:
                num_to_select = max(1, len(valid_indices) // amortization_ratio)
                selected_indices = random.sample(valid_indices.tolist(), num_to_select)
                frame_mask = torch.zeros(seq_len, dtype=torch.bool)
                frame_mask[selected_indices] = True
            else:
                frame_mask = torch.zeros(seq_len, dtype=torch.bool)

            # Always keep labels for codebook0 and the text token.
            codebook_mask = torch.zeros_like(labels, dtype=torch.bool)
            codebook_mask[:, -1] = True
            valid_frames_mask = torch.any(labels != -100, dim=-1, keepdim=True).expand(
                -1, 1
            )
            codebook_mask[:, 0:1] = valid_frames_mask

            # For selected frames, keep labels for codebooks 1..(N-1).
            for s in range(seq_len):
                if frame_mask[s]:
                    codebook_mask[s, 1:-1] = True

            new_labels = torch.where(
                (labels != -100) & ~codebook_mask, torch.full_like(labels, -100), labels
            )
            labels = new_labels

        if return_tensors == "pt":
            return {
                "input_ids": tokens,
                "attention_mask": tokens_mask,
                "labels": labels,
            }
        else:
            raise ValueError(f"Unsupported return format: {return_tensors}")

    def batch_decode(self, sequences, skip_special_tokens=True):
        """Decode sequences using text tokenizer"""
        return self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)

    def decode(self, sequence, skip_special_tokens=True):
        """Decode single sequence using text tokenizer"""
        return self.tokenizer.decode(sequence, skip_special_tokens=skip_special_tokens)

    def save_pretrained(self, save_directory):
        """Save processor components"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save text tokenizer
        self.tokenizer.save_pretrained(os.path.join(save_directory, "text_tokenizer"))
        
        # Save processor config
        import json
        config = {
            "sample_rate": self.sample_rate,
            "amortization_ratio": self.amortization_ratio
        }
        with open(os.path.join(save_directory, "processor_config.json"), 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod 
    def from_pretrained(cls, save_directory, audio_tokenizer=None):
        """Load processor from saved directory"""
        from transformers import AutoTokenizer
        import json
        import os
        
        # Load text tokenizer
        text_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(save_directory, "text_tokenizer")
        )
        
        # Load processor config
        config_path = os.path.join(save_directory, "processor_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Create processor
        processor = cls(text_tokenizer, audio_tokenizer)
        processor.sample_rate = config.get("sample_rate", 24000)
        processor.amortization_ratio = config.get("amortization_ratio", 16)
        
        return processor


# Utility functions for Vietnamese text processing
def normalize_vietnamese_text(text: str) -> str:
    """Normalize Vietnamese text for better tokenization"""
    import re
    
    if not text:
        return ""
    
    # Common Vietnamese text normalization
    text = text.strip()
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[`']", "'", text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Ensure proper sentence ending
    if text and text[-1] not in '.!?':
        text += '.'
    
    return text


def extract_speaker_id(role: str) -> int:
    """Extract speaker ID from role string like 'speaker_0'"""
    try:
        return int(role.split('_')[-1])
    except (ValueError, IndexError):
        return 0


# Vietnamese-specific processor
class VietnameseCSMProcessor(CSMProcessor):
    """
    Vietnamese-optimized CSM Processor
    
    Includes Vietnamese text normalization and language-specific optimizations.
    """
    
    def __init__(self, tokenizer, audio_tokenizer):
        super().__init__(tokenizer, audio_tokenizer)
        
        # Vietnamese-specific settings
        self.vietnamese_patterns = [
            (r'\s+', ' '),  # Multiple spaces
            (r'["""]', '"'),  # Normalize quotes
            (r'\.{2,}', '...'),  # Multiple dots
            (r'[?!]{2,}', '?!'),  # Multiple punctuation
        ]
    
    def _normalize_vietnamese_text(self, text: str) -> str:
        """Apply Vietnamese-specific text normalization"""
        if not text:
            return ""
        
        text = text.strip()
        
        # Apply Vietnamese normalization patterns
        for pattern, replacement in self.vietnamese_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Ensure proper ending
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def _process_messages(self, messages, audios, *args, **kwargs):
        """Override to add Vietnamese text normalization"""
        
        # Normalize Vietnamese text in messages
        normalized_messages = []
        for message in messages:
            normalized_message = message.copy()
            normalized_content = []
            
            for content in message["content"]:
                if content["type"] == "text":
                    normalized_content.append({
                        "type": "text",
                        "text": self._normalize_vietnamese_text(content["text"])
                    })
                else:
                    normalized_content.append(content)
            
            normalized_message["content"] = normalized_content
            normalized_messages.append(normalized_message)
        
        # Call parent method with normalized messages
        return super()._process_messages(normalized_messages, audios, *args, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    print("CSM Processor implementation loaded successfully!")
    
    # Test Vietnamese text normalization
    test_text = "Xin   chào!!!   Tôi là  trợ lý AI.."
    normalized = normalize_vietnamese_text(test_text)
    print(f"Original: '{test_text}'")
    print(f"Normalized: '{normalized}'")
    
    # Test speaker ID extraction
    test_roles = ["speaker_0", "speaker_5", "invalid_role"]
    for role in test_roles:
        speaker_id = extract_speaker_id(role)
        print(f"Role: '{role}' -> Speaker ID: {speaker_id}")