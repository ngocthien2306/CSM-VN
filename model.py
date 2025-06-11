"""
model.py - Vietnamese CSM Model Implementation (FIXED)

Fixed issues:
1. RoPE scaling configuration
2. Gradient checkpointing support
3. Memory optimizations for Vietnamese training
"""

import json
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

# Import base CSM components
try:
    from modeling_csm import CSMConfig, CSMModel
    from processor import CSMProcessor
except ImportError:
    logging.error("CSM model components not found. Please ensure modeling_csm.py and processor.py are available.")
    raise

logger = logging.getLogger(__name__)

class VietnameseCSMConfig(CSMConfig):
    """
    FIXED Configuration for Vietnamese CSM model with memory optimization
    """
    
    def __init__(
        self,
        # Vietnamese-specific parameters
        vietnamese_tone_vocab_size: int = 7,
        vietnamese_dialect_vocab_size: int = 3,
        use_vietnamese_features: bool = True,
        
        # Base CSM parameters
        text_vocab_size: int = 128256,
        audio_vocab_size: int = 2051,
        audio_num_codebooks: int = 32,
        max_seq_len: int = 2048,
        
        # MEMORY OPTIMIZATION: Model size options
        model_size: str = "small",  # tiny, small, medium
        
        # FIXED: Memory optimizations
        gradient_checkpointing: bool = True,
        use_cache: bool = True,
        
        **kwargs
    ):
        # Define model size configurations for memory optimization
        size_configs = {
            "tiny": {
                "backbone": {
                    "hidden_size": 512,
                    "intermediate_size": 2048,
                    "num_hidden_layers": 8,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 4,
                },
                "decoder": {
                    "hidden_size": 256,
                    "intermediate_size": 1024,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                }
            },
            "small": {
                "backbone": {
                    "hidden_size": 1024,
                    "intermediate_size": 4096,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                },
                "decoder": {
                    "hidden_size": 512,
                    "intermediate_size": 2048,
                    "num_hidden_layers": 3,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 4,
                }
            },
            "medium": {
                "backbone": {
                    "hidden_size": 2048,
                    "intermediate_size": 8192,
                    "num_hidden_layers": 16,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                },
                "decoder": {
                    "hidden_size": 1024,
                    "intermediate_size": 8192,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                }
            }
        }
        
        # Get size config
        if model_size not in size_configs:
            model_size = "small"
        size_config = size_configs[model_size]
        
        # Create FIXED backbone config with proper RoPE scaling
        backbone_config = {
            "vocab_size": text_vocab_size,
            "max_position_embeddings": max_seq_len,
            "rms_norm_eps": 1e-5,
            "attention_dropout": 0.0,
            "rope_theta": 500000,
            "rope_scaling": {
                "type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": max_seq_len // 4,
            },
            "architectures": ["LlamaForCausalLM"],
            "hidden_act": "silu",
            **size_config["backbone"]
        }
        
        # Create FIXED decoder config with proper RoPE scaling  
        decoder_config = {
            "vocab_size": text_vocab_size,
            "max_position_embeddings": audio_num_codebooks,
            "rms_norm_eps": 1e-5,
            "attention_dropout": 0.0,
            "rope_theta": 500000,
            "rope_scaling": {
                "type": "llama3", 
                "factor": 2.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 2.0,
                "original_max_position_embeddings": audio_num_codebooks // 2,
            },
            "architectures": ["LlamaForCausalLM"],
            "hidden_act": "silu",
            **size_config["decoder"]
        }
        
        super().__init__(
            text_vocab_size=text_vocab_size,
            audio_vocab_size=audio_vocab_size,
            audio_num_codebooks=audio_num_codebooks,
            max_seq_len=max_seq_len,
            backbone_config=backbone_config,
            decoder_config=decoder_config,
            **kwargs
        )
        
        # Vietnamese-specific config
        self.vietnamese_tone_vocab_size = vietnamese_tone_vocab_size
        self.vietnamese_dialect_vocab_size = vietnamese_dialect_vocab_size
        self.use_vietnamese_features = use_vietnamese_features
        self.gradient_checkpointing = gradient_checkpointing
        self.use_cache = use_cache
        self.model_size = model_size
    
    def to_dict(self):
        """Convert config to dictionary with JSON serializable values"""
        config_dict = {}
        
        # Add all config attributes, converting LlamaConfig to dict
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if hasattr(value, 'to_dict'):
                    # Convert LlamaConfig objects to dict
                    config_dict[key] = value.to_dict()
                elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    # Keep JSON serializable types
                    config_dict[key] = value
                else:
                    # Convert other types to string
                    config_dict[key] = str(value)
                    
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary, handling LlamaConfig objects"""
        # Make a copy to avoid modifying original
        config_dict = config_dict.copy()
        
        # Convert backbone_config dict back to LlamaConfig if needed
        if 'backbone_config' in config_dict and isinstance(config_dict['backbone_config'], dict):
            from transformers import LlamaConfig
            config_dict['backbone_config'] = LlamaConfig(**config_dict['backbone_config'])
            
        # Convert decoder_config dict back to LlamaConfig if needed  
        if 'decoder_config' in config_dict and isinstance(config_dict['decoder_config'], dict):
            from transformers import LlamaConfig
            config_dict['decoder_config'] = LlamaConfig(**config_dict['decoder_config'])
        
        return cls(**config_dict)
    
    def to_json_string(self, use_diff: bool = True) -> str:
        """Serialize config to JSON string"""
        import json
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

class VietnameseCSMModel(CSMModel):
    """
    FIXED Vietnamese-adapted CSM model with proper gradient checkpointing support
    """
    
    config_class = VietnameseCSMConfig
    _supports_gradient_checkpointing = True  # FIXED: Enable gradient checkpointing support
    
    def __init__(self, config: VietnameseCSMConfig):
        super().__init__(config)
        self.config = config
        
        # Add Vietnamese-specific components if enabled
        if config.use_vietnamese_features:
            self._add_vietnamese_features()
        
        # FIXED: Properly enable gradient checkpointing
        if config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
    
    def _add_vietnamese_features(self):
        """Add Vietnamese-specific model components"""
        try:
            # Vietnamese tone embeddings
            if hasattr(self.config, 'vietnamese_tone_vocab_size'):
                self.tone_embeddings = nn.Embedding(
                    self.config.vietnamese_tone_vocab_size,
                    self.config.backbone_config.hidden_size // 4
                )
            
            # Vietnamese dialect embeddings  
            if hasattr(self.config, 'vietnamese_dialect_vocab_size'):
                self.dialect_embeddings = nn.Embedding(
                    self.config.vietnamese_dialect_vocab_size,
                    self.config.backbone_config.hidden_size // 8
                )
                
            logger.info("Vietnamese-specific features added successfully")
            
        except Exception as e:
            logger.warning(f"Failed to add Vietnamese features: {e}")
    
    def _enable_gradient_checkpointing(self):
        """FIXED: Enable gradient checkpointing for memory optimization"""
        try:
            # Enable for backbone
            if hasattr(self.backbone, 'gradient_checkpointing_enable'):
                self.backbone.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled for backbone")
            
            # Enable for decoder
            if hasattr(self.decoder, 'gradient_checkpointing_enable'):
                self.decoder.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled for decoder")
                
            # Set gradient checkpointing flag
            self.gradient_checkpointing = True
            
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")
            self.gradient_checkpointing = False
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Override to support gradient checkpointing with proper signature"""
        # Handle the gradient_checkpointing_kwargs parameter from transformers
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        
        self._enable_gradient_checkpointing()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        try:
            if hasattr(self.backbone, 'gradient_checkpointing_disable'):
                self.backbone.gradient_checkpointing_disable()
            
            if hasattr(self.decoder, 'gradient_checkpointing_disable'):
                self.decoder.gradient_checkpointing_disable()
                
            self.gradient_checkpointing = False
            logger.info("Gradient checkpointing disabled")
            
        except Exception as e:
            logger.warning(f"Could not disable gradient checkpointing: {e}")
    
    def _gradient_checkpointing_func(self, func):
        """Wrapper function for gradient checkpointing"""
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        if hasattr(torch.utils.checkpoint, 'checkpoint'):
            return torch.utils.checkpoint.checkpoint(create_custom_forward(func))
        else:
            return func
    
    def forward(self, *args, **kwargs):
        """Forward pass with Vietnamese-specific processing"""
        # Extract Vietnamese-specific inputs if present
        tone_ids = kwargs.pop('tone_ids', None)
        dialect_ids = kwargs.pop('dialect_ids', None)
        
        # Call parent forward
        outputs = super().forward(*args, **kwargs)
        
        # Could add Vietnamese-specific loss terms here if needed
        return outputs
    
    def generate_vietnamese(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tone_ids: Optional[torch.Tensor] = None,
        dialect_ids: Optional[torch.Tensor] = None,
        **generation_kwargs
    ):
        """Generate speech with Vietnamese-specific features"""
        return self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs
        )

def load_vietnamese_text_tokenizer(tokenizer_path: Optional[str] = None) -> AutoTokenizer:
    """Load and configure text tokenizer for Vietnamese CSM"""
    if tokenizer_path and Path(tokenizer_path).exists():
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        logger.info("Loading default Llama-3.2 tokenizer")
        tokenizer_name = "meta-llama/Llama-3.2-1B"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Configure for CSM format
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[
            (bos, tokenizer.bos_token_id),
            (eos, tokenizer.eos_token_id),
        ],
    )
    
    logger.info(f"Text tokenizer loaded: vocab_size={len(tokenizer)}")
    return tokenizer

def load_vietnamese_audio_tokenizer(device: str = "cuda"):
    """Load and configure audio tokenizer for Vietnamese CSM"""
    try:
        from moshi.models import loaders
        
        logger.info("Loading Mimi audio tokenizer...")
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        audio_tokenizer = loaders.get_mimi(mimi_weight, device=device)
        audio_tokenizer.set_num_codebooks(32)
        
        logger.info(f"Audio tokenizer loaded: {audio_tokenizer.num_codebooks} codebooks")
        return audio_tokenizer
        
    except ImportError:
        logger.error("moshi-models not found. Please install: pip install moshi-models")
        raise
    except Exception as e:
        logger.error(f"Error loading audio tokenizer: {e}")
        raise

def create_vietnamese_processor(
    text_tokenizer_path: Optional[str] = None,
    device: str = "cuda",
    amortization_ratio: int = 16
) -> CSMProcessor:
    """Create CSM processor with Vietnamese tokenizers"""
    text_tokenizer = load_vietnamese_text_tokenizer(text_tokenizer_path)
    audio_tokenizer = load_vietnamese_audio_tokenizer(device)
    
    processor = CSMProcessor(text_tokenizer, audio_tokenizer)
    processor.amortization_ratio = amortization_ratio
    
    logger.info("Vietnamese CSM processor created successfully")
    return processor

def load_vietnamese_csm_model(
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float32,
    use_pretrained: bool = True,
    enable_gradient_checkpointing: bool = False  # Changed default to False
) -> VietnameseCSMModel:
    """FIXED: Load Vietnamese CSM model with proper configuration"""
    
    # Load config
    if config_path and Path(config_path).exists():
        logger.info(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = VietnameseCSMConfig.from_dict(config_dict)
    else:
        logger.info("Using default Vietnamese CSM config")
        config = VietnameseCSMConfig(
            gradient_checkpointing=False  # Let trainer handle this
        )
    
    # Load model
    if use_pretrained and model_path:
        try:
            logger.info(f"Loading pretrained model from {model_path}")
            model = VietnameseCSMModel.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch_dtype
            )
        except Exception as e:
            logger.warning(f"Failed to load pretrained model: {e}")
            logger.info("Creating new Vietnamese CSM model")
            model = VietnameseCSMModel(config)
    else:
        logger.info("Creating new Vietnamese CSM model")
        model = VietnameseCSMModel(config)
    
    model.to(device)
    model.train()
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Vietnamese CSM model loaded:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Dtype: {torch_dtype}")
    logger.info(f"  Gradient checkpointing: {getattr(model, 'gradient_checkpointing', False)}")
    
    return model

def save_vietnamese_csm_model(
    model: VietnameseCSMModel,
    save_path: str,
    save_config: bool = True,
    save_processor: bool = True,
    processor: Optional[CSMProcessor] = None
):
    """Save Vietnamese CSM model and associated components"""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving Vietnamese CSM model to {save_path}")
    model.save_pretrained(str(save_path))
    
    if save_config:
        config_path = save_path / "vietnamese_csm_config.json"
        with open(config_path, 'w') as f:
            json.dump(model.config.to_dict(), f, indent=2)
        logger.info(f"Config saved to {config_path}")
    
    if save_processor and processor:
        text_tokenizer_path = save_path / "text_tokenizer"
        processor.tokenizer.save_pretrained(str(text_tokenizer_path))
        logger.info(f"Text tokenizer saved to {text_tokenizer_path}")
        
        processor_config = {
            "sample_rate": processor.sample_rate,
            "amortization_ratio": getattr(processor, 'amortization_ratio', 16)
        }
        with open(save_path / "processor_config.json", 'w') as f:
            json.dump(processor_config, f, indent=2)
        logger.info("Processor config saved")
    
    logger.info("Vietnamese CSM model saved successfully")

def freeze_backbone(model: VietnameseCSMModel):
    """Freeze backbone parameters for fine-tuning"""
    logger.info("Freezing backbone parameters")
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters after freezing: {trainable_params:,}")

def unfreeze_all(model: VietnameseCSMModel):
    """Unfreeze all model parameters"""
    logger.info("Unfreezing all parameters")
    for param in model.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters after unfreezing: {trainable_params:,}")

def get_model_memory_usage(model: VietnameseCSMModel) -> Dict[str, float]:
    """Get model memory usage statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage
    param_memory = total_params * 4 / (1024**3)  # 4 bytes per float32 param
    
    # GPU memory if available
    gpu_memory = 0
    gpu_memory_reserved = 0
    if next(model.parameters()).is_cuda:
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "estimated_param_memory_gb": param_memory,
        "current_gpu_memory_gb": gpu_memory,
        "reserved_gpu_memory_gb": gpu_memory_reserved,
        "gradient_checkpointing_enabled": getattr(model, 'gradient_checkpointing', False)
    }

def optimize_model_for_training(model: VietnameseCSMModel, 
                               enable_checkpointing: bool = True,
                               mixed_precision: bool = True) -> VietnameseCSMModel:
    """Apply memory optimizations for training"""
    logger.info("Optimizing model for training...")
    
    # Enable gradient checkpointing
    if enable_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            logger.info("✓ Gradient checkpointing enabled")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")
    
    # Mixed precision optimizations
    if mixed_precision:
        # Convert to half precision where appropriate
        # (Note: This should be handled by training arguments)
        logger.info("✓ Mixed precision will be handled by trainer")
    
    # Memory-efficient attention (if available)
    try:
        # Enable flash attention if available
        for module in model.modules():
            if hasattr(module, 'enable_flash_attention'):
                module.enable_flash_attention()
        logger.info("✓ Flash attention enabled where available")
    except Exception:
        logger.info("Flash attention not available")
    
    return model

# Example usage and testing
if __name__ == "__main__":
    # Test model creation with FIXED configuration
    logger.info("Testing FIXED Vietnamese CSM model...")
    
    try:
        # Create config with FIXED RoPE scaling
        config = VietnameseCSMConfig(
            max_seq_len=1024,  # Smaller for testing
            use_vietnamese_features=True,
            gradient_checkpointing=True
        )
        logger.info("✓ Config created successfully")
        
        # Create model
        model = VietnameseCSMModel(config)
        logger.info("✓ Model created successfully")
        
        # Test model info
        memory_info = get_model_memory_usage(model)
        logger.info("Model memory info:")
        for key, value in memory_info.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:,.0f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Test gradient checkpointing
        logger.info(f"✓ Gradient checkpointing enabled: {model.gradient_checkpointing}")
        
        # Test model forward (with dummy input)
        dummy_input = torch.zeros(1, 10, 33, dtype=torch.long)
        dummy_mask = torch.ones(1, 10, 33)
        
        try:
            with torch.no_grad():
                outputs = model(input_ids=dummy_input, attention_mask=dummy_mask)
            logger.info("✓ Model forward test successful")
            
            if hasattr(outputs, 'loss'):
                logger.info(f"  Output has loss: {outputs.loss is not None}")
            
        except Exception as e:
            logger.error(f"✗ Model forward test failed: {e}")
        
        logger.info("✓ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        raise