#!/usr/bin/env python3
"""
train.py - Vietnamese CSM Training Script

Main training script for Vietnamese Conversational Speech Model.
Handles model training, evaluation, and checkpointing.

Usage:
    python train.py --config configs/training_config.json
    python train.py --help  # for all options
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Fix for old GPU compatibility
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch.nn as nn
from transformers import (
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback,
    set_seed
)
import wandb

# Import our modules
from dataset import create_datasets
from model import (
    load_vietnamese_csm_model,
    create_vietnamese_processor,
    save_vietnamese_csm_model,
    freeze_backbone,
    get_model_memory_usage
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class VietnameseCSMTrainer(Trainer):
    """
    Custom trainer for Vietnamese CSM with enhanced logging and monitoring
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
        self.step_times = []

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Compute loss with separate backbone and decoder loss logging
        """
        outputs = model(**inputs)
        loss = outputs.loss

        # Debug: Log tensor shapes to understand the issue
        if hasattr(outputs, 'backbone_loss') and outputs.backbone_loss is not None:
            logger.debug(f"backbone_loss shape: {outputs.backbone_loss.shape}")
        if hasattr(outputs, 'decoder_loss') and outputs.decoder_loss is not None:
            logger.debug(f"decoder_loss shape: {outputs.decoder_loss.shape}")
        if loss is not None:
            logger.debug(f"total_loss shape: {loss.shape}")

        # Log separate losses if available with proper tensor handling
        logs = {}
        
        if hasattr(outputs, 'backbone_loss') and outputs.backbone_loss is not None:
            backbone_loss = outputs.backbone_loss
            # Handle tensor with multiple elements
            if backbone_loss.numel() > 1:
                backbone_loss_scalar = backbone_loss.mean().detach().float().item()
                logger.debug(f"backbone_loss reduced from {backbone_loss.numel()} elements")
            else:
                backbone_loss_scalar = backbone_loss.detach().float().item()
            logs["train/backbone_loss"] = backbone_loss_scalar
        
        if hasattr(outputs, 'decoder_loss') and outputs.decoder_loss is not None:
            decoder_loss = outputs.decoder_loss
            # Handle tensor with multiple elements
            if decoder_loss.numel() > 1:
                decoder_loss_scalar = decoder_loss.mean().detach().float().item()
                logger.debug(f"decoder_loss reduced from {decoder_loss.numel()} elements")
            else:
                decoder_loss_scalar = decoder_loss.detach().float().item()
            logs["train/decoder_loss"] = decoder_loss_scalar
        
        if loss is not None:
            # Handle total loss
            if loss.numel() > 1:
                total_loss_scalar = loss.mean().detach().float().item()
                logger.debug(f"total_loss reduced from {loss.numel()} elements")
            else:
                total_loss_scalar = loss.detach().float().item()
            logs["train/total_loss"] = total_loss_scalar
        
        if logs:
            self.log(logs)

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_time=None) -> None:
        """Enhanced logging with timing information"""
        # Add custom logs if available
        custom_logs = getattr(self, '_custom_logs', {})
        if custom_logs:
            logs.update(custom_logs)
            self._custom_logs = {}  # Clear after logging
        
        # Add timing information
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if len(self.step_times) > 0:
            step_time = current_time - self.step_times[-1]
            logs["train/step_time"] = step_time
        
        logs["train/elapsed_time"] = elapsed_time
        self.step_times.append(current_time)
        
        # Add memory usage if on GPU
        if torch.cuda.is_available():
            logs["train/gpu_memory_gb"] = torch.cuda.memory_allocated() / (1024**3)
            logs["train/gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        
        # Call parent log method with proper signature
        super().log(logs)
        
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with detailed logging"""
        logger.info(f"Starting {description}...")
        
        eval_start_time = time.time()
        outputs = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
        eval_time = time.time() - eval_start_time
        
        # Log evaluation timing
        self.log({f"{metric_key_prefix}/eval_time": eval_time})
        
        logger.info(f"{description} completed in {eval_time:.2f}s")
        return outputs

def load_training_config(config_path: str) -> Dict:
    """Load training configuration from JSON file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    logger.info(f"Loaded training config from {config_path}")
    return config

def setup_logging_and_monitoring(config: Dict):
    """Setup logging and monitoring (Wandb, TensorBoard)"""
    logging_config = config.get("logging", {})
    
    # Setup Wandb if enabled
    if logging_config.get("use_wandb", False):
        wandb_config = {
            "project": logging_config.get("wandb_project", "vietnamese-csm"),
            "name": logging_config.get("wandb_run_name", f"vietnamese-csm-{int(time.time())}"),
            "config": config,
            "tags": ["vietnamese", "csm", "speech-synthesis"]
        }
        
        wandb.init(**wandb_config)
        logger.info("Wandb initialized")
    
    # TensorBoard is automatically handled by transformers

def validate_config(config: Dict):
    """Validate training configuration"""
    required_sections = ["model", "data", "training"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate data files exist
    data_config = config["data"]
    train_file = data_config.get("train_file")
    if not train_file or not Path(train_file).exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    eval_file = data_config.get("eval_file")
    if eval_file and not Path(eval_file).exists():
        logger.warning(f"Evaluation file not found: {eval_file}")
    
    logger.info("Configuration validation passed")

def create_training_arguments(config: Dict) -> TrainingArguments:
    """Create TrainingArguments from config"""
    training_config = config["training"]
    
    args = TrainingArguments(
        # Basic training settings
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        
        # Optimization settings
        learning_rate=training_config.get("learning_rate", 5e-6),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_steps=training_config.get("warmup_steps", 500),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        
        # Logging and saving
        logging_steps=training_config.get("logging_steps", 50),
        save_steps=training_config.get("save_steps", 500),
        eval_steps=training_config.get("eval_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        
        # Evaluation settings
        evaluation_strategy=training_config.get("evaluation_strategy", "steps"),
        eval_delay=training_config.get("eval_delay", 0),
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_config.get("greater_is_better", False),
        
        # Performance optimizations
        dataloader_num_workers=training_config.get("dataloader_num_workers", 0),
        fp16=training_config.get("fp16", False),
        bf16=training_config.get("bf16", False),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        dataloader_pin_memory=training_config.get("dataloader_pin_memory", True),
        
        # Misc settings
        remove_unused_columns=False,  # Important for CSM
        seed=training_config.get("seed", 42),
        
        # Reporting
        report_to=config.get("logging", {}).get("report_to", ["tensorboard"]),
        run_name=config.get("logging", {}).get("wandb_run_name", "vietnamese-csm"),
    )
    
    return args

def main():
    """Main training function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Vietnamese CSM model")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to training configuration JSON file"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (use small dataset)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run - setup everything but don't train"
    )
    
    args = parser.parse_args()
    
    # Load and validate config
    logger.info("=== Vietnamese CSM Training ===")
    config = load_training_config(args.config)
    validate_config(config)
    
    # Set debug mode
    if args.debug:
        logger.info("Debug mode enabled - using small dataset")
        config["data"]["max_train_samples"] = 100
        config["data"]["max_eval_samples"] = 20
        config["training"]["save_steps"] = 10
        config["training"]["eval_steps"] = 10
        config["training"]["logging_steps"] = 5
    
    # Set seed for reproducibility
    seed = config["training"].get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")
    
    # Setup logging and monitoring
    setup_logging_and_monitoring(config)
    
    # Check device and memory
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.warning("CUDA not available - training will be very slow on CPU")
    
    # Create processor (loads tokenizers)
    logger.info("Loading tokenizers and creating processor...")
    processor = create_vietnamese_processor(
        text_tokenizer_path=config["model"].get("text_tokenizer_path"),
        device=device,
        amortization_ratio=config["data"].get("amortization_ratio", 16)
    )
    
    # Load model
    logger.info("Loading Vietnamese CSM model...")
    model = load_vietnamese_csm_model(
        model_path=config["model"].get("model_name_or_path"),
        config_path=config["model"].get("config_path"),
        device=device,
        torch_dtype=torch.bfloat16 if config["training"].get("bf16", False) else torch.float32,
        use_pretrained=config["model"].get("use_pretrained_csm", True),
        enable_gradient_checkpointing=False  # Let trainer handle this
    )
    
    # Freeze backbone if specified
    if config["model"].get("freeze_backbone", False):
        freeze_backbone(model)
    
    # Log model memory usage
    memory_info = get_model_memory_usage(model)
    logger.info("Model memory usage:")
    for key, value in memory_info.items():
        logger.info(f"  {key}: {value:,.0f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, eval_dataset, data_collator = create_datasets(
        train_file=config["data"]["train_file"],
        eval_file=config["data"].get("eval_file"),
        processor=processor,
        num_train_epochs=config["training"]["num_train_epochs"],
        max_train_samples=config["data"].get("max_train_samples"),
        max_eval_samples=config["data"].get("max_eval_samples"),
        amortization_ratio=config["data"].get("amortization_ratio", 16)
    )
    
    # Create training arguments
    training_args = create_training_arguments(config)
    
    # Create trainer
    trainer = VietnameseCSMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset and training_args.load_best_model_at_end else None,
    )
    
    # Log training setup
    logger.info("=== Training Setup Complete ===")
    logger.info(f"Training samples: {len(train_dataset):,}")
    logger.info(f"Evaluation samples: {len(eval_dataset):,}" if eval_dataset else "No evaluation set")
    logger.info(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
    logger.info(f"Warmup steps: {training_args.warmup_steps}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Output directory: {training_args.output_dir}")
    
    # Dry run check
    if args.dry_run:
        logger.info("Dry run completed - exiting without training")
        return
    
    # Start training
    logger.info("=== Starting Training ===")
    start_time = time.time()
    
    try:
        if args.resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/3600:.2f} hours")
        
        # Save final model
        logger.info("Saving final model...")
        save_vietnamese_csm_model(
            model=model,
            save_path=training_args.output_dir,
            save_config=True,
            save_processor=True,
            processor=processor
        )
        
        # Final evaluation
        if eval_dataset:
            logger.info("Running final evaluation...")
            eval_results = trainer.evaluate()
            logger.info("Final evaluation results:")
            for key, value in eval_results.items():
                logger.info(f"  {key}: {value}")
        
        logger.info("=== Training Completed Successfully ===")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
        # Save current state
        logger.info("Saving current state...")
        trainer.save_model(os.path.join(training_args.output_dir, "interrupted_checkpoint"))
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Close wandb if used
        if config.get("logging", {}).get("use_wandb", False):
            wandb.finish()

if __name__ == "__main__":
    main()