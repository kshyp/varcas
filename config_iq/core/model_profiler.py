"""Model profiling and characteristic extraction from HuggingFace models."""

import json
import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelProfile:
    """Profile of a language model."""
    model_id: str
    parameters_b: float  # Billions of parameters
    hidden_dim: int
    num_layers: int
    num_heads: int
    head_dim: int
    vocab_size: int
    
    # Derived characteristics
    @property
    def kv_cache_per_token_kb(self) -> float:
        """KV cache size per token in KB."""
        # KV cache = 2 (K+V) * num_layers * hidden_dim * bytes_per_param
        # Assuming fp16 = 2 bytes
        bytes_per_param = 2
        kv_size_bytes = 2 * self.num_layers * self.hidden_dim * bytes_per_param
        return kv_size_bytes / 1024
    
    @property
    def model_size_gb(self) -> float:
        """Model size in GB (fp16)."""
        bytes_per_param = 2  # fp16
        return (self.parameters_b * 1e9 * bytes_per_param) / (1024 ** 3)
    
    @property
    def model_size_gb_int8(self) -> float:
        """Model size in GB (int8)."""
        bytes_per_param = 1  # int8
        return (self.parameters_b * 1e9 * bytes_per_param) / (1024 ** 3)
    
    @property
    def model_size_gb_int4(self) -> float:
        """Model size in GB (int4)."""
        bytes_per_param = 0.5  # int4
        return (self.parameters_b * 1e9 * bytes_per_param) / (1024 ** 3)
    
    def get_model_size_for_quantization(self, quantization: str) -> float:
        """Get model size for a specific quantization."""
        sizes = {
            "fp32": self.model_size_gb * 2,
            "fp16": self.model_size_gb,
            "bf16": self.model_size_gb,
            "int8": self.model_size_gb_int8,
            "fp8": self.model_size_gb_int8,
            "int4": self.model_size_gb_int4
        }
        return sizes.get(quantization, self.model_size_gb)
    
    @property
    def flops_per_prefill_token(self) -> float:
        """FLOPs per prefill token (forward pass)."""
        # Roughly 2 * params per token for forward pass
        return 2 * self.parameters_b * 1e9
    
    @property
    def flops_per_decode_token(self) -> float:
        """FLOPs per decode token (autoregressive generation)."""
        # In decode phase, we process one token but attend to all previous
        # Roughly 2 * params (same as prefill but with KV cache reuse)
        return 2 * self.parameters_b * 1e9
    
    def calculate_memory_requirement(
        self,
        batch_size: int = 1,
        seq_length: int = 4096,
        quantization: str = "fp16"
    ) -> Dict[str, float]:
        """Calculate memory requirements for inference.
        
        Returns:
            Dict with model_weights_gb, kv_cache_gb, activations_gb, total_gb
        """
        # Model weights
        model_weights_gb = self.get_model_size_for_quantization(quantization)
        
        # KV cache: 2 (K+V) * num_layers * batch * seq_len * hidden_dim * bytes_per_param
        bytes_per_param = 2 if quantization in ["fp16", "bf16"] else 1
        kv_cache_gb = (
            2 * self.num_layers * batch_size * seq_length * 
            self.hidden_dim * bytes_per_param / (1024 ** 3)
        )
        
        # Activations (rough estimate: proportional to batch * seq_len * hidden_dim)
        activation_gb = (
            batch_size * seq_length * self.hidden_dim * 
            bytes_per_param / (1024 ** 3)
        )
        
        # Overhead for vLLM (20%)
        overhead = 1.2
        
        return {
            "model_weights_gb": model_weights_gb,
            "kv_cache_gb": kv_cache_gb,
            "activations_gb": activation_gb,
            "subtotal_gb": model_weights_gb + kv_cache_gb + activation_gb,
            "total_gb": (model_weights_gb + kv_cache_gb + activation_gb) * overhead,
            "overhead_factor": overhead
        }
    
    def calculate_memory_required_per_gpu(self, tp_size: int, quantization: str = "fp16") -> float:
        """Calculate memory required per GPU with given tensor parallel size.
        
        Args:
            tp_size: Tensor parallel degree (model sharded across this many GPUs)
            quantization: Quantization type
            
        Returns:
            Memory required per GPU in GB
        """
        model_size_gb = self.get_model_size_for_quantization(quantization)
        
        # Model is sharded across TP group
        model_per_gpu = model_size_gb / tp_size
        
        # Overhead: CUDA context + activations + safety margin
        cuda_overhead_gb = 1.5
        activation_buffer_gb = model_per_gpu * 0.10
        safety_margin_gb = 0.5
        
        return model_per_gpu + cuda_overhead_gb + activation_buffer_gb + safety_margin_gb
    
    def calculate_tensor_parallel_degree(self, available_gpus: int, gpu_vram_gb: int = None, quantization: str = "fp16") -> int:
        """Calculate optimal tensor parallel degree.
        
        Args:
            available_gpus: Number of GPUs available
            gpu_vram_gb: VRAM per GPU in GB (if known)
            quantization: Quantization type
        
        Returns:
            Optimal tensor parallel degree
        """
        # Common TP sizes: 1, 2, 4, 8
        valid_tp = [tp for tp in [1, 2, 4, 8] if tp <= available_gpus]
        
        if gpu_vram_gb is not None:
            # Find the minimum TP size that fits in GPU memory
            for tp_size in valid_tp:
                mem_required = self.calculate_memory_required_per_gpu(tp_size, quantization)
                if gpu_vram_gb >= mem_required:
                    return tp_size
            # If none fit, return the largest TP size (will likely OOM but best effort)
            return valid_tp[-1] if valid_tp else 1
        
        # Without VRAM info, use heuristics based on model size
        # For most models, TP=1 is best for throughput
        if self.parameters_b <= 13:
            return 1
        if self.parameters_b <= 30 and available_gpus >= 2:
            return min(2, available_gpus)
        if self.parameters_b <= 70 and available_gpus >= 4:
            return min(4, available_gpus)
        if available_gpus >= 8:
            return 8
        
        return min(available_gpus, 8)


# Known model configurations (fallback when HF is not available)
KNOWN_MODELS = {
    # Llama 2 family
    "meta-llama/Llama-2-7b-hf": {"parameters_b": 6.7, "hidden_dim": 4096, "num_layers": 32, "num_heads": 32, "vocab_size": 32000},
    "meta-llama/Llama-2-13b-hf": {"parameters_b": 13.0, "hidden_dim": 5120, "num_layers": 40, "num_heads": 40, "vocab_size": 32000},
    "meta-llama/Llama-2-70b-hf": {"parameters_b": 70.0, "hidden_dim": 8192, "num_layers": 80, "num_heads": 64, "vocab_size": 32000},
    
    # Llama 3 family
    "meta-llama/Llama-3.1-8B": {"parameters_b": 8.0, "hidden_dim": 4096, "num_layers": 32, "num_heads": 32, "vocab_size": 128000},
    "meta-llama/Llama-3.1-70B": {"parameters_b": 70.0, "hidden_dim": 8192, "num_layers": 80, "num_heads": 64, "vocab_size": 128000},
    "meta-llama/Llama-3.1-405B": {"parameters_b": 405.0, "hidden_dim": 16384, "num_layers": 126, "num_heads": 128, "vocab_size": 128000},
    
    # Mistral family
    "mistralai/Mistral-7B-v0.1": {"parameters_b": 7.3, "hidden_dim": 4096, "num_layers": 32, "num_heads": 32, "vocab_size": 32000},
    "mistralai/Mistral-7B-Instruct-v0.2": {"parameters_b": 7.3, "hidden_dim": 4096, "num_layers": 32, "num_heads": 32, "vocab_size": 32000},
    "mistralai/Mixtral-8x7B-v0.1": {"parameters_b": 47.0, "hidden_dim": 4096, "num_layers": 32, "num_heads": 32, "vocab_size": 32000},
    "mistralai/Mixtral-8x22B-v0.1": {"parameters_b": 141.0, "hidden_dim": 6144, "num_layers": 56, "num_heads": 48, "vocab_size": 32000},
    
    # CodeLlama
    "codellama/CodeLlama-7b-hf": {"parameters_b": 6.7, "hidden_dim": 4096, "num_layers": 32, "num_heads": 32, "vocab_size": 32000},
    "codellama/CodeLlama-13b-hf": {"parameters_b": 13.0, "hidden_dim": 5120, "num_layers": 40, "num_heads": 40, "vocab_size": 32000},
    "codellama/CodeLlama-34b-hf": {"parameters_b": 34.0, "hidden_dim": 8192, "num_layers": 48, "num_heads": 64, "vocab_size": 32000},
    
    # Qwen
    "Qwen/Qwen2.5-7B": {"parameters_b": 7.6, "hidden_dim": 3584, "num_layers": 28, "num_heads": 28, "vocab_size": 152064},
    "Qwen/Qwen2.5-14B": {"parameters_b": 14.2, "hidden_dim": 5120, "num_layers": 48, "num_heads": 40, "vocab_size": 152064},
    "Qwen/Qwen2.5-32B": {"parameters_b": 32.5, "hidden_dim": 5120, "num_layers": 64, "num_heads": 40, "vocab_size": 152064},
    "Qwen/Qwen2.5-72B": {"parameters_b": 72.7, "hidden_dim": 8192, "num_layers": 80, "num_heads": 64, "vocab_size": 152064},
    
    # DeepSeek
    "deepseek-ai/deepseek-llm-7b-base": {"parameters_b": 7.0, "hidden_dim": 4096, "num_layers": 30, "num_heads": 32, "vocab_size": 102400},
    "deepseek-ai/deepseek-llm-67b-base": {"parameters_b": 67.0, "hidden_dim": 8192, "num_layers": 95, "num_heads": 64, "vocab_size": 102400},
    
    # Google Gemma
    "google/gemma-2b": {"parameters_b": 2.5, "hidden_dim": 2048, "num_layers": 18, "num_heads": 8, "vocab_size": 256000},
    "google/gemma-7b": {"parameters_b": 8.5, "hidden_dim": 3072, "num_layers": 28, "num_heads": 16, "vocab_size": 256000},
    "google/gemma-2-9b": {"parameters_b": 9.2, "hidden_dim": 3584, "num_layers": 42, "num_heads": 16, "vocab_size": 256000},
    "google/gemma-2-27b": {"parameters_b": 27.2, "hidden_dim": 4608, "num_layers": 46, "num_heads": 32, "vocab_size": 256000},
}


def get_model_profile(model_id: str) -> ModelProfile:
    """Get model profile for a HuggingFace model ID.
    
    Args:
        model_id: HuggingFace model ID
    
    Returns:
        ModelProfile with model characteristics
    """
    # Try to fetch from known models
    if model_id in KNOWN_MODELS:
        config = KNOWN_MODELS[model_id]
        return ModelProfile(
            model_id=model_id,
            parameters_b=config["parameters_b"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            head_dim=config["hidden_dim"] // config["num_heads"],
            vocab_size=config["vocab_size"]
        )
    
    # Try to parse from model name
    # Pattern: name-XXb or name-XXB
    import re
    match = re.search(r'(\d+)(\.\d+)?[bB]', model_id.split('/')[-1])
    if match:
        params = float(match.group(1))
        if match.group(2):
            params += float(match.group(2))
        
        # Estimate architecture based on size
        if params <= 3:
            hidden_dim, num_layers, num_heads = 2048, 24, 32
        elif params <= 8:
            hidden_dim, num_layers, num_heads = 4096, 32, 32
        elif params <= 15:
            hidden_dim, num_layers, num_heads = 5120, 40, 40
        elif params <= 35:
            hidden_dim, num_layers, num_heads = 8192, 48, 64
        elif params <= 75:
            hidden_dim, num_layers, num_heads = 8192, 80, 64
        else:
            hidden_dim, num_layers, num_heads = 16384, 120, 128
        
        return ModelProfile(
            model_id=model_id,
            parameters_b=params,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=hidden_dim // num_heads,
            vocab_size=32000
        )
    
    # Default fallback
    raise ValueError(f"Unknown model: {model_id}. Please use a known model or specify parameters manually.")


def estimate_model_size(parameters_b: float, quantization: str = "fp16") -> float:
    """Estimate model size in GB.
    
    Args:
        parameters_b: Parameters in billions
        quantization: Quantization type
    
    Returns:
        Size in GB
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "fp8": 1,
        "int4": 0.5
    }.get(quantization, 2)
    
    return (parameters_b * 1e9 * bytes_per_param) / (1024 ** 3)
