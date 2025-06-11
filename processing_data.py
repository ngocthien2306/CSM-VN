#!/usr/bin/env python3
"""
Run Vietnamese Dataset Preprocessing for CSM Training

This script processes ViMD and VietBud500 datasets and prepares them 
for Vietnamese CSM training.

Usage:
    python run_preprocessing.py --output_dir ./vietnamese_csm_data --max_samples 10000
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Preprocess Vietnamese datasets for CSM training")
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./vietnamese_csm_data",
        help="Output directory for processed data"
    )
    
    parser.add_argument(
        "--max_vimd_samples", 
        type=int, 
        default=None,
        help="Maximum samples from ViMD dataset (None = all)"
    )
    
    parser.add_argument(
        "--max_vietbud_samples", 
        type=int, 
        default=100000,
        help="Maximum samples from VietBud500 dataset (None = all)"
    )
    
    parser.add_argument(
        "--create_multi_turn", 
        action="store_true", 
        default=True,
        help="Create multi-turn conversations"
    )
    
    parser.add_argument(
        "--save_audio_files", 
        action="store_true", 
        default=True,
        help="Save audio as separate files"
    )
    
    parser.add_argument(
        "--vimd_dataset_name", 
        type=str, 
        default="None",  # You'll need to replace with actual dataset name
        help="HuggingFace dataset name for ViMD"
    )
    
    parser.add_argument(
        "--vietbud_dataset_name", 
        type=str, 
        default="None",  # You'll need to replace with actual dataset name
        help="HuggingFace dataset name for VietBud500"
    )
    
    args = parser.parse_args()
    
    print("=== Vietnamese CSM Dataset Preprocessing ===")
    print(f"Output directory: {args.output_dir}")
    print(f"Max ViMD samples: {args.max_vimd_samples}")
    print(f"Max VietBud samples: {args.max_vietbud_samples}")
    print(f"Create multi-turn: {args.create_multi_turn}")
    print(f"Save audio files: {args.save_audio_files}")
    
    try:
        # Import required libraries
        print("\nLoading required libraries...")
        from datasets import load_dataset
        import torch
        import torchaudio
        
        # Import our processing class
        from core import VietnameseCSMDatasetProcessor
        
        print("Libraries loaded successfully!")
        

        # Create processor
        processor = VietnameseCSMDatasetProcessor(args.output_dir)
        
        
        def process_with_your_datasets(ds_vimd, ds_viet_bud500):
            """
            Process your datasets once you have them loaded
            """
            return processor.process_complete_dataset(
                ds_viet_bud500, ds_vimd,
                create_multi_turn=args.create_multi_turn,
                save_audio_files=args.save_audio_files,
                max_vimd_samples=args.max_vimd_samples,
                max_vietbud_samples=args.max_vietbud_samples
            )
        
        print("\n=== Ready for Processing ===")
        print("The preprocessing system is ready.")
        print("Please load your datasets and call process_with_your_datasets(ds_vimd, ds_viet_bud500)")
        
        return process_with_your_datasets
        
    except ImportError as e:
        print(f"Error importing required libraries: {e}")
        print("Please install required packages:")
        print("pip install datasets torch torchaudio transformers underthesea tqdm numpy")
        sys.exit(1)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    process_function = main()
    
    from datasets import load_dataset
    print("Loading dataset ...")
    ds_viMD = load_dataset("nguyendv02/ViMD_Dataset")
    ds_viet_bud500 = load_dataset("linhtran92/viet_bud500")
    print("Loaded data successfully!")

    output_files = process_function(ds_viMD, ds_viet_bud500)
