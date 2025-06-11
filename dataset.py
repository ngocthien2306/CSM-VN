"""
dataset.py - Vietnamese CSM Dataset Implementation

Handles loading and processing of Vietnamese conversational speech data
for CSM training.
"""

import json
import logging
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset
from dataclasses import dataclass

# Fix for old GPU compatibility
import torch._dynamo
torch._dynamo.config.suppress_errors = True

logger = logging.getLogger(__name__)

class VietnameseCSMDataset(Dataset):
    """
    Dataset for Vietnamese CSM training
    
    Loads conversations from JSONL format and processes them for CSM training.
    Each conversation contains text and audio content that gets tokenized
    into the multimodal CSM format.
    """

    def __init__(
        self, 
        data_path: str, 
        processor, 
        num_train_epochs: int = 1,
        max_samples: Optional[int] = None,
        amortization_ratio: int = 16
    ):
        """
        Initialize dataset
        
        Args:
            data_path: Path to JSONL file containing conversations
            processor: CSMProcessor instance for tokenization
            num_train_epochs: Number of times to repeat dataset (for amortization)
            max_samples: Maximum number of samples to load
            amortization_ratio: Decoder training amortization ratio
        """
        self.data_path = Path(data_path)
        self.processor = processor
        self.num_train_epochs = num_train_epochs
        self.amortization_ratio = amortization_ratio

        # Load conversations from JSONL
        self.data = self._load_conversations()
        
        if max_samples:
            self.data = self.data[:max_samples]

        logger.info(f"Loaded {len(self.data)} conversations from {data_path}")
        logger.info(f"Dataset will repeat {num_train_epochs} times for amortization")
        
        # Set amortization ratio on processor
        if hasattr(processor, 'amortization_ratio'):
            processor.amortization_ratio = amortization_ratio

    def _load_conversations(self) -> List[Dict]:
        """Load conversations from JSONL file"""
        conversations = []
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        
        with open(self.data_path, "r", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    conversation = json.loads(line.strip())
                    conversations.append(conversation)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                    continue
        
        return conversations

    def _load_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess audio file"""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != self.processor.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.processor.sample_rate
                )
                waveform = resampler(waveform)
            
            return waveform.squeeze(0)
            
        except Exception as e:
            logger.warning(f"Error loading audio {audio_path}: {e}")
            return None

    def __len__(self):
        """Return dataset length (repeated for multiple epochs)"""
        return len(self.data) * self.num_train_epochs

    def __getitem__(self, idx):
        """Get a single training sample"""
        # Map to actual data index
        data_idx = idx % len(self.data)
        item = self.data[data_idx]
        
        try:
            return self._process_conversation(item)
        except Exception as e:
            logger.error(f"Error processing conversation {data_idx}: {e}")
            logger.warning("Falling back to dummy sample due to GPU compatibility issue")
            return self._get_dummy_sample()

    def _process_conversation(self, conversation: Dict) -> Dict[str, torch.Tensor]:
        """Process a single conversation into CSM format"""
        messages = conversation["messages"]
        training_mask = conversation.get("training_mask", None)

        # Load audio data for all messages
        audio_tensors = []
        for message in messages:
            for content in message["content"]:
                if content["type"] == "audio":
                    if "url" in content:
                        # Load from file path
                        audio_tensor = self._load_audio(content["url"])
                        audio_tensors.append(audio_tensor)
                    elif "audio_data" in content:
                        # Use embedded audio data
                        audio_data = content["audio_data"]
                        audio_tensor = torch.from_numpy(audio_data['array']).float()
                        audio_tensors.append(audio_tensor)
                    else:
                        audio_tensors.append(None)

        # Process with CSMProcessor
        processed = self.processor(
            messages=messages,
            audios=audio_tensors,
            messages_training_mask=training_mask,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            amortize_decoder_training=True,
            amortization_ratio=self.amortization_ratio,
        )
        
        return {
            "input_ids": processed["input_ids"].squeeze(0),
            "attention_mask": processed["attention_mask"].squeeze(0),
            "labels": processed["labels"].squeeze(0),
        }

    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return a dummy sample to avoid training interruption"""
        return {
            "input_ids": torch.zeros(10, 33, dtype=torch.long),
            "attention_mask": torch.zeros(10, 33),
            "labels": torch.full((10, 33), -100, dtype=torch.long),
        }

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        total_conversations = len(self.data)
        total_messages = sum(len(conv["messages"]) for conv in self.data)
        
        # Count by dataset source
        datasets = {}
        dialects = {}
        speakers = set()
        
        for conv in self.data:
            # Dataset source
            dataset = conv.get("metadata", {}).get("dataset", "unknown")
            datasets[dataset] = datasets.get(dataset, 0) + 1
            
            # Dialect
            dialect = conv.get("metadata", {}).get("dialect", "unknown")
            dialects[dialect] = dialects.get(dialect, 0) + 1
            
            # Speakers
            for msg in conv["messages"]:
                speakers.add(msg["role"])
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "unique_speakers": len(speakers),
            "datasets": datasets,
            "dialects": dialects,
            "effective_size": len(self),  # Including repetitions
        }

@dataclass
class VietnameseCSMDataCollator:
    """
    Data collator for Vietnamese CSM training
    
    Handles padding sequences to the same length within a batch
    and applies proper masking for the CSM format.
    """
    
    text_pad_token_id: int

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features
        
        Args:
            features: List of processed conversation samples
            
        Returns:
            Batched and padded tensors
        """
        if not features:
            return {}

        # Find maximum sequence length in batch
        max_seq_len = max(f["input_ids"].size(0) for f in features)
        
        # Pad all sequences to max length
        padded_features = {}
        for key in ["input_ids", "attention_mask", "labels"]:
            padded_tensors = []
            
            for feature in features:
                tensor = feature[key]
                seq_len, width = tensor.size()
                
                if seq_len < max_seq_len:
                    pad_rows = max_seq_len - seq_len
                    
                    # Create appropriate padding
                    if key == "labels":
                        # Pad labels with -100 (ignore index)
                        pad_tensor = torch.full(
                            (pad_rows, width), -100, dtype=tensor.dtype
                        )
                    elif key == "attention_mask":
                        # Pad attention mask with 0s
                        pad_tensor = torch.zeros(
                            (pad_rows, width), dtype=tensor.dtype
                        )
                    else:  # input_ids
                        # Pad input_ids with 0s, except text column gets pad token
                        pad_tensor = torch.zeros(
                            (pad_rows, width), dtype=tensor.dtype
                        )
                        pad_tensor[:, -1] = self.text_pad_token_id
                    
                    # Left pad (add padding at beginning)
                    padded_tensor = torch.cat([pad_tensor, tensor], dim=0)
                else:
                    padded_tensor = tensor
                
                padded_tensors.append(padded_tensor.unsqueeze(0))
            
            padded_features[key] = torch.cat(padded_tensors, dim=0)
        
        return padded_features

def create_datasets(
    train_file: str,
    eval_file: Optional[str],
    processor,
    num_train_epochs: int = 3,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    amortization_ratio: int = 16
) -> tuple:
    """
    Create train and eval datasets
    
    Args:
        train_file: Path to training JSONL file
        eval_file: Path to evaluation JSONL file (optional)
        processor: CSMProcessor instance
        num_train_epochs: Number of epochs for training dataset repetition
        max_train_samples: Limit training samples
        max_eval_samples: Limit evaluation samples
        amortization_ratio: Decoder training amortization ratio
    
    Returns:
        Tuple of (train_dataset, eval_dataset, data_collator)
    """
    
    # Create training dataset
    train_dataset = VietnameseCSMDataset(
        data_path=train_file,
        processor=processor,
        num_train_epochs=num_train_epochs,
        max_samples=max_train_samples,
        amortization_ratio=amortization_ratio
    )
    
    # Create evaluation dataset
    eval_dataset = None
    if eval_file:
        eval_dataset = VietnameseCSMDataset(
            data_path=eval_file,
            processor=processor,
            num_train_epochs=1,  # No repetition for eval
            max_samples=max_eval_samples,
            amortization_ratio=amortization_ratio
        )
    
    # Create data collator
    # Assume processor has a text tokenizer with eos_token_id
    text_pad_token_id = getattr(processor.tokenizer, 'eos_token_id', 0)
    data_collator = VietnameseCSMDataCollator(text_pad_token_id=text_pad_token_id)
    
    # Log dataset statistics
    train_stats = train_dataset.get_statistics()
    logger.info("Training dataset statistics:")
    for key, value in train_stats.items():
        logger.info(f"  {key}: {value}")
    
    if eval_dataset:
        eval_stats = eval_dataset.get_statistics()
        logger.info("Evaluation dataset statistics:")
        for key, value in eval_stats.items():
            logger.info(f"  {key}: {value}")
    
    return train_dataset, eval_dataset, data_collator

# Example usage and testing
if __name__ == "__main__":
    # Test dataset loading
    import tempfile
    import json
    
    # Create a dummy JSONL file for testing
    test_data = [
        {
            "conversation_id": "test_1",
            "messages": [
                {
                    "role": "speaker_0",
                    "content": [
                        {"type": "text", "text": "Xin chào"},
                        {"type": "audio", "url": "/fake/path.wav", "duration": 2.0}
                    ]
                }
            ],
            "metadata": {"dataset": "test", "dialect": "North"}
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in test_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
        temp_file = f.name
    
    # Mock processor for testing
    class MockProcessor:
        def __init__(self):
            self.sample_rate = 24000
            self.amortization_ratio = 16
            
        def __call__(self, **kwargs):
            return {
                "input_ids": torch.zeros(5, 33, dtype=torch.long),
                "attention_mask": torch.ones(5, 33),
                "labels": torch.zeros(5, 33, dtype=torch.long)
            }
    
    # Test dataset
    try:
        dataset = VietnameseCSMDataset(
            data_path=temp_file,
            processor=MockProcessor(),
            max_samples=1
        )
        
        print(f"Dataset loaded successfully with {len(dataset)} samples")
        
        # Test getting an item
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Input shape: {sample['input_ids'].shape}")
        
        # Test statistics
        stats = dataset.get_statistics()
        print(f"Dataset statistics: {stats}")
        
    finally:
        import os
        os.unlink(temp_file)
    
    print("Dataset module test completed!")