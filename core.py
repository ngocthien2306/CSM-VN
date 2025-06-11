import json
import os
import random
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datasets import Dataset, DatasetDict
import numpy as np
from underthesea import word_tokenize, sent_tokenize
import re
from tqdm import tqdm

class VietnameseCSMDatasetProcessor:
    """
    Process ViMD and VietBud500 datasets for Vietnamese CSM training
    Creates JSONL files compatible with CSM training pipeline
    """
    
    def __init__(self, output_dir: str = "./vietnamese_csm_prepared"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Vietnamese text normalization patterns
        self.text_patterns = [
            (r'\s+', ' '),  # Multiple spaces to single space
            (r'["""]', '"'),  # Normalize quotes
            (r'\.{2,}', '...'),  # Multiple dots to ellipsis
            (r'[,]{2,}', ','),  # Multiple commas to single
            (r'[?!]{2,}', '?!'),  # Multiple punctuation
            (r'([.!?])\s*([.!?])', r'\1 \2'),  # Space between punctuation
        ]
        
        # Vietnamese dialect mapping based on provinces
        self.dialect_regions = {
            'North': ['HaNoi', 'HaiPhong', 'QuangNinh', 'CaoBang', 'LangSon', 
                     'ThaiBinh', 'NamDinh', 'NinhBinh', 'HaGiang', 'TuyenQuang'],
            'Central': ['ThuaThienHue', 'DaNang', 'QuangNam', 'QuangNgai', 'BinhDinh', 
                       'PhuYen', 'KhanhHoa', 'NinhThuan', 'BinhThuan', 'DakLak'],
            'South': ['HoChiMinh', 'DongNai', 'BinhDuong', 'LongAn', 'TienGiang', 
                     'BenTre', 'CanTho', 'AnGiang', 'KienGiang', 'CaMau']
        }
    
    def normalize_vietnamese_text(self, text: str) -> str:
        """Normalize Vietnamese text for consistency"""
        if not text or not isinstance(text, str):
            return ""
        
        text = text.strip()
        
        # Apply normalization patterns
        for pattern, replacement in self.text_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Remove excessive whitespace and normalize
        text = ' '.join(text.split())
        
        # Ensure text ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def get_dialect_from_province(self, province_name: str) -> str:
        """Map province name to dialect region"""
        for region, provinces in self.dialect_regions.items():
            if province_name in provinces:
                return region
        return 'North'  # Default to North if unknown
    
    def validate_audio_data(self, audio_data: Dict, 
                           min_duration: float = 0.5, 
                           max_duration: float = 30) -> Tuple[bool, str]:
        """
        Validate audio data quality
        Returns: (is_valid, reason)
        """
        try:
            audio_array = audio_data.get('array')
            sample_rate = audio_data.get('sampling_rate')
            
            if audio_array is None:
                return False, "No audio array"
            
            if len(audio_array) == 0:
                return False, "Empty audio array"
            
            if sample_rate is None or sample_rate <= 0:
                return False, "Invalid sample rate"
            
            duration = len(audio_array) / sample_rate
            
            # Check duration constraints
            if duration < min_duration:
                return False, f"Too short: {duration:.2f}s < {min_duration}s"
            
            if duration > max_duration:
                return False, f"Too long: {duration:.2f}s > {max_duration}s"
            
            # Check for silent audio (very low amplitude)
            max_amplitude = np.max(np.abs(audio_array))
            if max_amplitude < 0.001:
                return False, f"Too quiet: max amplitude {max_amplitude}"
            
            # Check for clipped audio (too loud)
            if max_amplitude > 0.95:
                return False, f"Likely clipped: max amplitude {max_amplitude}"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def resample_audio(self, audio_data: Dict, target_sample_rate: int = 24000) -> Dict:
        """Resample audio to target sample rate"""
        try:
            current_sr = audio_data['sampling_rate']
            audio_array = audio_data['array']
            
            if current_sr == target_sample_rate:
                return audio_data
            
            # Convert to tensor for resampling
            audio_tensor = torch.from_numpy(audio_array).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
            
            # Resample
            resampler = torchaudio.transforms.Resample(current_sr, target_sample_rate)
            resampled = resampler(audio_tensor)
            
            return {
                'array': resampled.squeeze(0).numpy(),
                'sampling_rate': target_sample_rate,
                'path': audio_data.get('path')
            }
            
        except Exception as e:
            print(f"Error resampling audio: {e}")
            return audio_data
    
    def process_vimd_dataset(self, ds_vimd: DatasetDict, 
                           target_sample_rate: int = 24000,
                           max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process ViMD dataset to CSM conversation format"""
        conversations = []
        
        # Get train split
        if isinstance(ds_vimd, DatasetDict):
            train_data = ds_vimd['train'] if 'train' in ds_vimd else ds_vimd[list(ds_vimd.keys())[0]]
        else:
            train_data = ds_vimd
        
        if max_samples:
            train_data = train_data.select(range(min(max_samples, len(train_data))))
        
        print(f"Processing ViMD dataset: {len(train_data)} samples")
        
        speaker_counter = {}  # Track speakers by region
        valid_count = 0
        invalid_count = 0
        
        for idx, item in enumerate(tqdm(train_data, desc="Processing ViMD")):
            try:
                # Extract data with error handling
                region = item.get('region', 'Unknown')
                province_name = item.get('province_name', 'Unknown')
                text = item.get('text', '')
                speaker_id = item.get('speakerID', f'vimd_spk_{idx}')
                gender = item.get('gender', 0)
                audio_data = item.get('audio', {})
                
                # Validate and normalize text
                normalized_text = self.normalize_vietnamese_text(text)
                if not normalized_text or len(normalized_text) < 10:
                    invalid_count += 1
                    continue
                
                # Validate audio
                is_valid, reason = self.validate_audio_data(audio_data)
                if not is_valid:
                    invalid_count += 1
                    if idx % 1000 == 0:
                        print(f"Skipped audio {idx}: {reason}")
                    continue
                
                # Resample audio
                resampled_audio = self.resample_audio(audio_data, target_sample_rate)
                
                # Assign consistent speaker number based on region
                if region not in speaker_counter:
                    speaker_counter[region] = 0
                
                speaker_num = speaker_counter[region] % 10  # Cycle through 10 speakers per region
                speaker_role = f"speaker_{speaker_num}"
                
                # Calculate audio duration
                duration = len(resampled_audio['array']) / resampled_audio['sampling_rate']
                
                # Create conversation in CSM format
                conversation = {
                    "conversation_id": f"vimd_{region}_{idx}",
                    "messages": [
                        {
                            "role": speaker_role,
                            "content": [
                                {
                                    "type": "text",
                                    "text": normalized_text
                                },
                                {
                                    "type": "audio",
                                    "audio_data": resampled_audio,
                                    "duration": duration
                                }
                            ]
                        }
                    ],
                    "metadata": {
                        "dataset": "vimd",
                        "region": region,
                        "province": province_name,
                        "speaker_id": speaker_id,
                        "gender": "male" if gender == 1 else "female",
                        "dialect": self.get_dialect_from_province(province_name),
                        "duration": duration,
                        "text_length": len(normalized_text)
                    }
                }
                
                conversations.append(conversation)
                speaker_counter[region] += 1
                valid_count += 1
                
            except Exception as e:
                invalid_count += 1
                if idx % 1000 == 0:
                    print(f"Error processing ViMD item {idx}: {e}")
                continue
        
        print(f"ViMD processing complete: {valid_count} valid, {invalid_count} invalid")
        return conversations
    
    def process_vietbud500_dataset(self, ds_viet_bud500: DatasetDict, 
                                 target_sample_rate: int = 24000,
                                 max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process VietBud500 dataset to CSM conversation format"""
        conversations = []
        
        # Get train split
        if isinstance(ds_viet_bud500, DatasetDict):
            train_data = ds_viet_bud500['train'] if 'train' in ds_viet_bud500 else ds_viet_bud500[list(ds_viet_bud500.keys())[0]]
        else:
            train_data = ds_viet_bud500
        
        if max_samples:
            train_data = train_data.select(range(min(max_samples, len(train_data))))
        
        print(f"Processing VietBud500 dataset: {len(train_data)} samples")
        
        valid_count = 0
        invalid_count = 0
        
        for idx, item in enumerate(tqdm(train_data, desc="Processing VietBud500")):
            try:
                # Extract data
                transcription = item.get('transcription', '')
                audio_data = item.get('audio', {})
                
                # Validate and normalize text
                normalized_text = self.normalize_vietnamese_text(transcription)
                if not normalized_text or len(normalized_text) < 10:
                    invalid_count += 1
                    continue
                
                # Validate audio
                is_valid, reason = self.validate_audio_data(audio_data)
                if not is_valid:
                    invalid_count += 1
                    if idx % 1000 == 0:
                        print(f"Skipped VietBud500 audio {idx}: {reason}")
                    continue
                
                # Resample audio
                resampled_audio = self.resample_audio(audio_data, target_sample_rate)
                
                # Assign speaker (cycle through speakers)
                speaker_num = idx % 20  # Use 20 different speakers
                speaker_role = f"speaker_{speaker_num}"
                
                # Calculate duration
                duration = len(resampled_audio['array']) / resampled_audio['sampling_rate']
                
                # Create conversation
                conversation = {
                    "conversation_id": f"vietbud500_{idx}",
                    "messages": [
                        {
                            "role": speaker_role,
                            "content": [
                                {
                                    "type": "text",
                                    "text": normalized_text
                                },
                                {
                                    "type": "audio",
                                    "audio_data": resampled_audio,
                                    "duration": duration
                                }
                            ]
                        }
                    ],
                    "metadata": {
                        "dataset": "vietbud500",
                        "speaker_id": f"vietbud_spk_{speaker_num}",
                        "dialect": "mixed",  # VietBud500 contains mixed dialects
                        "duration": duration,
                        "text_length": len(normalized_text)
                    }
                }
                
                conversations.append(conversation)
                valid_count += 1
                
            except Exception as e:
                invalid_count += 1
                if idx % 1000 == 0:
                    print(f"Error processing VietBud500 item {idx}: {e}")
                continue
        
        print(f"VietBud500 processing complete: {valid_count} valid, {invalid_count} invalid")
        return conversations
    
    def create_multi_turn_conversations(self, single_turns: List[Dict[str, Any]], 
                                      min_turns: int = 2, max_turns: int = 4) -> List[Dict[str, Any]]:
        """Create multi-turn conversations from single-turn data"""
        print("Creating multi-turn conversations...")
        
        multi_conversations = []
        
        # Group by dialect for realistic conversations
        dialect_groups = {}
        for conv in single_turns:
            dialect = conv["metadata"].get("dialect", "mixed")
            if dialect not in dialect_groups:
                dialect_groups[dialect] = []
            dialect_groups[dialect].append(conv)
        
        # Create multi-turn conversations within each dialect group
        for dialect, conversations in dialect_groups.items():
            random.shuffle(conversations)
            
            i = 0
            while i < len(conversations) - 1:
                num_turns = random.randint(min_turns, min(max_turns, len(conversations) - i))
                
                # Select conversations for this multi-turn
                selected_convs = conversations[i:i + num_turns]
                
                if len(selected_convs) < min_turns:
                    break
                
                # Combine messages with alternating speakers
                combined_messages = []
                for j, conv in enumerate(selected_convs):
                    original_message = conv["messages"][0]
                    new_role = f"speaker_{j % 2}"  # Alternate between 2 speakers
                    
                    new_message = {
                        "role": new_role,
                        "content": original_message["content"]
                    }
                    combined_messages.append(new_message)
                
                # Calculate total duration
                total_duration = sum(
                    content.get("duration", 0) 
                    for msg in combined_messages 
                    for content in msg["content"] 
                    if content["type"] == "audio"
                )
                
                # Create multi-turn conversation
                multi_conversation = {
                    "conversation_id": f"multi_{dialect}_{len(multi_conversations)}",
                    "messages": combined_messages,
                    "metadata": {
                        "type": "multi_turn",
                        "dialect": dialect,
                        "num_turns": len(combined_messages),
                        "total_duration": total_duration,
                        "source_datasets": list(set(conv["metadata"]["dataset"] for conv in selected_convs))
                    }
                }
                
                multi_conversations.append(multi_conversation)
                i += num_turns
        
        print(f"Created {len(multi_conversations)} multi-turn conversations")
        return multi_conversations
    
    def save_audio_separately(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Save audio data to separate files and replace with file paths"""
        audio_dir = self.output_dir / "audio_files"
        audio_dir.mkdir(exist_ok=True)
        
        print("Saving audio files separately...")
        
        updated_conversations = []
        
        for conv in tqdm(conversations, desc="Saving audio files"):
            try:
                updated_conv = conv.copy()
                updated_messages = []
                
                for msg_idx, message in enumerate(conv["messages"]):
                    updated_message = message.copy()
                    updated_content = []
                    
                    for content_idx, content_item in enumerate(message["content"]):
                        if content_item["type"] == "audio" and "audio_data" in content_item:
                            # Save audio to file
                            audio_data = content_item["audio_data"]
                            audio_filename = f"{conv['conversation_id']}_{message['role']}_{content_idx}.wav"
                            audio_path = audio_dir / audio_filename
                            
                            # Convert to tensor and save
                            audio_tensor = torch.from_numpy(audio_data['array']).float()
                            if audio_tensor.dim() == 1:
                                audio_tensor = audio_tensor.unsqueeze(0)
                            
                            torchaudio.save(
                                str(audio_path),
                                audio_tensor,
                                audio_data['sampling_rate']
                            )
                            
                            # Replace audio_data with file path
                            updated_content_item = {
                                "type": "audio",
                                "url": str(audio_path),
                                "duration": content_item.get("duration", 0)
                            }
                            updated_content.append(updated_content_item)
                        else:
                            updated_content.append(content_item)
                    
                    updated_message["content"] = updated_content
                    updated_messages.append(updated_message)
                
                updated_conv["messages"] = updated_messages
                updated_conversations.append(updated_conv)
                
            except Exception as e:
                print(f"Error saving audio for conversation {conv.get('conversation_id', 'unknown')}: {e}")
                # Keep original conversation without audio file
                updated_conversations.append(conv)
        
        print(f"Saved {len(updated_conversations)} conversations with audio files")
        return updated_conversations
    
    def split_dataset(self, conversations: List[Dict[str, Any]], 
                     train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[Dict[str, Any]]]:
        """Split dataset into train/val/test sets"""
        random.seed(42)  # For reproducible splits
        random.shuffle(conversations)
        
        total = len(conversations)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        splits = {
            'train': conversations[:train_end],
            'val': conversations[train_end:val_end],
            'test': conversations[val_end:]
        }
        
        print(f"Dataset split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
        return splits
    
    def save_to_jsonl(self, conversations: List[Dict[str, Any]], filename: str) -> str:
        """Save conversations to JSONL format"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for conversation in conversations:
                json.dump(conversation, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Saved {len(conversations)} conversations to {output_path}")
        return str(output_path)
    
    def generate_summary(self, all_conversations: List[Dict[str, Any]]) -> Dict:
        """Generate comprehensive dataset summary"""
        print("Generating dataset summary...")
        
        summary = {
            "total_conversations": len(all_conversations),
            "datasets": {},
            "dialects": {},
            "speakers": set(),
            "conversation_types": {"single_turn": 0, "multi_turn": 0},
            "total_audio_duration": 0,
            "total_text_chars": 0,
            "duration_stats": [],
            "text_length_stats": []
        }
        
        for conv in all_conversations:
            # Count by dataset
            dataset = conv["metadata"].get("dataset", "unknown")
            summary["datasets"][dataset] = summary["datasets"].get(dataset, 0) + 1
            
            # Count by dialect
            dialect = conv["metadata"].get("dialect", "unknown")
            summary["dialects"][dialect] = summary["dialects"].get(dialect, 0) + 1
            
            # Count conversation types
            conv_type = conv["metadata"].get("type", "single_turn")
            summary["conversation_types"][conv_type] += 1
            
            # Collect statistics
            conv_duration = 0
            conv_text_length = 0
            
            for message in conv["messages"]:
                summary["speakers"].add(message["role"])
                
                for content in message["content"]:
                    if content["type"] == "text":
                        text_len = len(content["text"])
                        conv_text_length += text_len
                        summary["total_text_chars"] += text_len
                    elif content["type"] == "audio":
                        duration = content.get("duration", 0)
                        conv_duration += duration
                        summary["total_audio_duration"] += duration
            
            summary["duration_stats"].append(conv_duration)
            summary["text_length_stats"].append(conv_text_length)
        
        # Calculate final statistics
        summary["num_speakers"] = len(summary["speakers"])
        summary["speakers"] = sorted(list(summary["speakers"]))
        summary["total_audio_hours"] = summary["total_audio_duration"] / 3600
        summary["average_text_length"] = summary["total_text_chars"] / len(all_conversations) if all_conversations else 0
        summary["average_audio_duration"] = summary["total_audio_duration"] / len(all_conversations) if all_conversations else 0
        
        # Duration statistics
        if summary["duration_stats"]:
            summary["duration_stats"] = {
                "min": min(summary["duration_stats"]),
                "max": max(summary["duration_stats"]),
                "mean": np.mean(summary["duration_stats"]),
                "median": np.median(summary["duration_stats"]),
                "std": np.std(summary["duration_stats"])
            }
        
        # Save summary
        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print(f"\n=== Dataset Summary ===")
        print(f"Total conversations: {summary['total_conversations']}")
        print(f"Unique speakers: {summary['num_speakers']}")
        print(f"Datasets: {summary['datasets']}")
        print(f"Dialects: {summary['dialects']}")
        print(f"Conversation types: {summary['conversation_types']}")
        print(f"Total audio: {summary['total_audio_hours']:.2f} hours")
        print(f"Average audio per conversation: {summary['average_audio_duration']:.1f} seconds")
        print(f"Average text length: {summary['average_text_length']:.1f} characters")
        print(f"Summary saved to: {summary_path}")
        
        return summary
    
    def process_complete_dataset(self, 
                               ds_viet_bud500: DatasetDict,
                               ds_vimd: DatasetDict, 
                               create_multi_turn: bool = True,
                               save_audio_files: bool = True,
                               max_vimd_samples: Optional[int] = None,
                               max_vietbud_samples: Optional[int] = None) -> Dict[str, str]:
        """Complete processing pipeline for Vietnamese datasets"""
        
        print("=== Starting Vietnamese CSM Dataset Processing ===")
        
        # Process both datasets
        vimd_conversations = self.process_vimd_dataset(ds_vimd, max_samples=max_vimd_samples)
        vietbud_conversations = self.process_vietbud500_dataset(ds_viet_bud500, max_samples=max_vietbud_samples)
        
        # Combine all conversations
        all_conversations = vimd_conversations + vietbud_conversations
        print(f"Combined total: {len(all_conversations)} conversations")
        
        # Create multi-turn conversations if requested
        if create_multi_turn and len(all_conversations) > 4:
            multi_turn_conversations = self.create_multi_turn_conversations(
                all_conversations[:len(all_conversations)//2]  # Use half for multi-turn
            )
            all_conversations.extend(multi_turn_conversations)
        
        # Save audio separately if requested
        if save_audio_files:
            all_conversations = self.save_audio_separately(all_conversations)
        
        # Split dataset
        splits = self.split_dataset(all_conversations)
        
        # Save splits to JSONL
        output_files = {}
        for split_name, split_data in splits.items():
            filename = f"vietnamese_csm_{split_name}.jsonl"
            output_files[split_name] = self.save_to_jsonl(split_data, filename)
        
        # Generate summary
        summary = self.generate_summary(all_conversations)
        
        print(f"\n=== Dataset Processing Complete ===")
        print(f"Output directory: {self.output_dir}")
        print(f"Files created: {list(output_files.values())}")
        
        return output_files

# Main function for easy usage
def prepare_vietnamese_datasets(ds_vimd, ds_viet_bud500, 
                              output_dir: str = "./vietnamese_csm_data",
                              max_vimd_samples: Optional[int] = None,
                              max_vietbud_samples: Optional[int] = None):
    """
    Main function to prepare Vietnamese datasets for CSM training
    
    Args:
        ds_vimd: ViMD dataset from HuggingFace
        ds_viet_bud500: VietBud500 dataset from HuggingFace
        output_dir: Output directory for processed data
        max_vimd_samples: Maximum samples from ViMD (None = all)
        max_vietbud_samples: Maximum samples from VietBud500 (None = all)
    
    Returns:
        Dictionary with paths to train/val/test JSONL files
    """
    processor = VietnameseCSMDatasetProcessor(output_dir)
    
    return processor.process_complete_dataset(
        ds_vimd, ds_viet_bud500,
        create_multi_turn=True,
        save_audio_files=True,
        max_vimd_samples=max_vimd_samples,
        max_vietbud_samples=max_vietbud_samples
    )

