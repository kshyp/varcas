"""Static roofline analysis for GPU inference performance prediction."""

import math
from typing import Tuple, Dict
from .types import VMInstance, RooflineMetrics
from .model_profiler import ModelProfile
from .hardware_catalog import HardwareCatalog


# Realistic efficiency factors for vLLM
# These account for kernel launch overhead, memory fragmentation, 
# scheduling inefficiency, and other real-world factors
VLLM_EFFICIENCY_FACTORS = {
    # Attention kernel efficiency (typically 60-80% of theoretical)
    "attention_compute": 0.65,
    
    # FFN/Linear layer efficiency
    "ffn_compute": 0.75,
    
    # Memory bandwidth efficiency (typically 70-85% of peak)
    "memory_bw": 0.75,
    
    # KV cache access efficiency (scatter/gather overhead)
    "kv_cache_access": 0.70,
    
    # Tensor parallel communication overhead
    "tp_overhead_per_gpu": {
        1: 1.0,
        2: 0.95,
        4: 0.90,
        8: 0.85
    },
    
    # Batching efficiency (diminishing returns at high batch sizes)
    "batch_efficiency": {
        1: 1.0,
        2: 0.98,
        4: 0.95,
        8: 0.90,
        16: 0.85,
        32: 0.80,
        64: 0.75
    }
}


class RooflineAnalyzer:
    """Performs static roofline analysis for LLM inference."""
    
    def __init__(self, hardware_catalog: HardwareCatalog):
        self.hardware = hardware_catalog
    
    def analyze_prefill(
        self,
        vm: VMInstance,
        model: ModelProfile,
        batch_size: int,
        seq_length: int,
        quantization: str = "fp16",
        cached_tokens: int = 0
    ) -> RooflineMetrics:
        """Analyze prefill phase (prompt processing).
        
        The prefill phase processes all input tokens in parallel.
        It's typically compute-bound for large batches.
        
        With prefix caching (e.g., in RAG workloads), cached tokens don't need
        to be re-computed, significantly reducing prefill latency.
        
        Args:
            vm: VM instance configuration
            model: Model profile
            batch_size: Number of concurrent sequences
            seq_length: Total input sequence length
            quantization: Quantization type
            cached_tokens: Number of tokens that are cache hits (skip computation)
        
        Returns:
            RooflineMetrics for prefill phase
        """
        # Get hardware specs
        total_compute = self.hardware.calculate_effective_compute(vm, quantization)
        total_memory_bw = sum(g.memory_bw_gbps * g.count for g in vm.gpus)
        
        # Effective tokens that need computation (excluding cached)
        effective_seq_length = max(1, seq_length - cached_tokens)
        cache_hit_ratio = cached_tokens / seq_length if seq_length > 0 else 0
        
        # Calculate FLOPs for prefill
        # Attention: 2 * batch * seq^2 * hidden (simplified)
        # FFN: 2 * batch * seq * params
        # Only compute for non-cached tokens
        flops_per_token = model.flops_per_prefill_token
        total_flops = flops_per_token * batch_size * effective_seq_length
        
        # Calculate memory traffic
        # Weights read (once per batch): model size
        # Activations: batch * seq * hidden * layers * bytes
        memory_per_param = 2 if quantization in ["fp16", "bf16"] else 1
        model_size_bytes = model.parameters_b * 1e9 * memory_per_param
        
        # Activation memory (rough estimate)
        # Use effective_seq_length for activations since cached tokens don't need new activations
        activation_bytes = (
            batch_size * effective_seq_length * model.hidden_dim * 
            model.num_layers * memory_per_param
        )
        
        total_memory_traffic_gb = (model_size_bytes + activation_bytes) / (1024 ** 3)
        
        # Operational intensity (FLOPs per byte)
        operational_intensity = total_flops / (total_memory_traffic_gb * 1e9)
        
        # Determine if compute or memory bound
        # Ridge point: where compute = memory_bw * intensity
        ridge_point = total_compute / total_memory_bw
        compute_bound = operational_intensity >= ridge_point
        
        # Calculate achievable performance
        if compute_bound:
            achievable_tflops = total_compute
            memory_bound_bw_gbps = 0
        else:
            achievable_tflops = total_memory_bw * operational_intensity
            memory_bound_bw = total_memory_bw
        
        # Calculate latency with realistic efficiency factors
        # For prefill, we need to account for both compute AND memory bandwidth
        # Even when compute-bound on paper, memory bandwidth limits small batches
        
        # Apply efficiency factors
        batch_eff = VLLM_EFFICIENCY_FACTORS["batch_efficiency"].get(
            batch_size, 0.75
        )
        compute_eff = VLLM_EFFICIENCY_FACTORS["attention_compute"] if seq_length > 1024 else VLLM_EFFICIENCY_FACTORS["ffn_compute"]
        
        effective_tflops = achievable_tflops * compute_eff * batch_eff
        
        # Compute time
        compute_time_ms = (total_flops / (effective_tflops * 1e12)) * 1000 if effective_tflops > 0 else float('inf')
        
        # Memory time - must load model weights even for prefill
        # Use same conservative 35% efficiency as decode for small batches
        prefill_memory_eff = 0.50 if batch_size > 4 else 0.35
        effective_memory_bw = total_memory_bw * prefill_memory_eff
        model_size_gb = model_size_bytes / (1024 ** 3)
        memory_time_ms = (model_size_gb / effective_memory_bw) * 1000
        
        # Add kernel launch and scheduling overhead (typical 5-10ms for prefill)
        # With prefix caching, overhead is reduced since we skip cache lookup for cached portion
        base_overhead_ms = 8 * (1 - cache_hit_ratio * 0.5)  # Up to 50% overhead reduction
        
        # For prefill, latency is the max of memory and compute times
        # (they can overlap somewhat, but memory often dominates for small batches)
        if batch_size <= 4:
            # Small batches: memory bandwidth limited
            prefill_latency_ms = max(compute_time_ms, memory_time_ms * 0.9) + base_overhead_ms
        else:
            # Larger batches: can better overlap compute with memory
            prefill_latency_ms = max(compute_time_ms, memory_time_ms * 0.7) + base_overhead_ms
        
        # Ensure minimum latency for cache lookup and KV cache retrieval
        if cached_tokens > 0:
            # Small overhead for retrieving cached KV (memory bandwidth bound)
            kv_retrieval_time_ms = 0.5  # Approximate time to load cached KV
            prefill_latency_ms = max(prefill_latency_ms, kv_retrieval_time_ms)
        
        # Throughput metrics
        tokens_per_sec_batch = (batch_size * seq_length) / (prefill_latency_ms / 1000)
        tokens_per_sec_single = seq_length / (prefill_latency_ms / 1000)
        
        # Utilization
        compute_utilization = achievable_tflops / total_compute if total_compute > 0 else 0
        memory_bw_utilization = (
            (achievable_tflops / operational_intensity) / total_memory_bw 
            if not compute_bound and total_memory_bw > 0 else 1.0
        )
        
        return RooflineMetrics(
            compute_bound=compute_bound,
            operational_intensity=operational_intensity,
            achievable_tflops=achievable_tflops,
            memory_bound_bw_gbps=memory_bound_bw_gbps if not compute_bound else 0,
            tokens_per_sec_single=tokens_per_sec_single,
            tokens_per_sec_batch=tokens_per_sec_batch,
            prefill_latency_ms=prefill_latency_ms,
            decode_latency_per_token_ms=0,  # Not applicable for prefill
            compute_utilization=min(1.0, compute_utilization),
            memory_bw_utilization=min(1.0, memory_bw_utilization)
        )
    
    def analyze_decode(
        self,
        vm: VMInstance,
        model: ModelProfile,
        batch_size: int,
        context_length: int,
        quantization: str = "fp16",
        tensor_parallel_size: int = 1
    ) -> RooflineMetrics:
        """Analyze decode phase (token generation).
        
        The decode phase generates one token at a time autoregressively.
        It's typically memory-bound due to KV cache access.
        
        Args:
            vm: VM instance configuration
            model: Model profile
            batch_size: Number of concurrent sequences
            context_length: Current context length (affects KV cache size)
            quantization: Quantization type
            tensor_parallel_size: Tensor parallel degree
        
        Returns:
            RooflineMetrics for decode phase
        """
        # Get hardware specs with tensor parallel considerations
        total_compute = self.hardware.calculate_effective_compute(
            vm, quantization, tensor_parallel_size
        )
        effective_memory_bw = self.hardware.calculate_effective_memory_bw(
            vm, tensor_parallel_size
        )
        
        # Calculate FLOPs for one decode step
        flops_per_token = model.flops_per_decode_token
        total_flops = flops_per_token * batch_size
        
        # Calculate memory traffic for decode
        # Model weights (same as prefill, but amortized over output tokens)
        memory_per_param = 2 if quantization in ["fp16", "bf16"] else 1
        model_size_bytes = model.parameters_b * 1e9 * memory_per_param
        
        # KV cache read: 2 * layers * batch * seq_len * hidden * bytes
        kv_cache_bytes = (
            2 * model.num_layers * batch_size * context_length * 
            model.hidden_dim * memory_per_param
        )
        
        # New KV cache write: 2 * layers * batch * hidden * bytes
        kv_cache_write_bytes = (
            2 * model.num_layers * batch_size * model.hidden_dim * memory_per_param
        )
        
        total_memory_traffic_gb = (
            model_size_bytes + kv_cache_bytes + kv_cache_write_bytes
        ) / (1024 ** 3)
        
        # Operational intensity
        operational_intensity = total_flops / (total_memory_traffic_gb * 1e9)
        
        # Determine bound
        ridge_point = total_compute / effective_memory_bw
        compute_bound = operational_intensity >= ridge_point
        
        # Achievable performance
        if compute_bound:
            achievable_tflops = total_compute
            memory_bound_bw_gbps = 0
        else:
            achievable_tflops = effective_memory_bw * operational_intensity
            memory_bound_bw_gbps = effective_memory_bw
        
        # Decode latency per token with realistic efficiency
        # For small batches, decode is memory-bandwidth-bound, not compute-bound
        # The roofline model incorrectly classifies this as compute-bound when
        # operational intensity > ridge_point, but memory latency dominates.
        
        batch_eff = VLLM_EFFICIENCY_FACTORS["batch_efficiency"].get(
            batch_size, 0.75
        )
        memory_eff = VLLM_EFFICIENCY_FACTORS["memory_bw"]  # 0.75
        kv_cache_eff = VLLM_EFFICIENCY_FACTORS["kv_cache_access"]  # 0.70
        tp_eff = VLLM_EFFICIENCY_FACTORS["tp_overhead_per_gpu"].get(
            tensor_parallel_size, 0.85
        )
        
        # For decode with small batch sizes, calculate latency based on memory bandwidth
        # rather than compute. Each token requires reading the full model weights.
        model_size_gb = model_size_bytes / (1024 ** 3)
        
        # Effective memory bandwidth for decode
        # vLLM typically achieves 35-50% of peak for decode (lower than the 75% general factor)
        # This accounts for memory access patterns, KV cache overhead, scheduling, etc.
        # Lower-end GPUs (T4) are less efficient than high-end (A100/H100)
        decode_memory_efficiency = 0.35  # Conservative real-world vLLM efficiency for decode
        effective_mem_bw = effective_memory_bw * decode_memory_efficiency * tp_eff
        
        # Time to read model weights from memory (dominant cost for decode)
        memory_time_ms = (model_size_gb / effective_mem_bw) * 1000
        
        # Add compute time (small compared to memory for decode)
        # For compute: use efficiency factors
        total_efficiency = memory_eff * kv_cache_eff * batch_eff * tp_eff
        effective_tflops = achievable_tflops * total_efficiency
        compute_time_ms = (total_flops / (effective_tflops * 1e12)) * 1000 if effective_tflops > 0 else 0
        
        # Add overhead for sampling, logits processing, scheduling
        decode_overhead_ms = 8  # Conservative estimate based on observed latency
        
        # For decode, latency is dominated by memory bandwidth for small batches
        # Larger batches can amortize memory reads across more tokens
        if batch_size <= 4:
            # Memory bandwidth dominated - small batches don't help much
            # Some overlap between memory and compute
            batch_scaling = 0.9 + 0.1 * batch_size / 4  # Minimal improvement
            decode_latency_ms = memory_time_ms * batch_scaling + decode_overhead_ms
        elif batch_size <= 16:
            # Medium batches start to help amortize memory reads
            batch_scaling = 0.8 + 0.2 * (batch_size / 16)
            decode_latency_ms = memory_time_ms * batch_scaling + decode_overhead_ms
        else:
            # Larger batches approach compute-bound
            batch_scaling = batch_eff
            decode_latency_ms = max(memory_time_ms * batch_scaling, compute_time_ms) + decode_overhead_ms
        
        # Throughput
        tokens_per_sec_batch = batch_size / (decode_latency_ms / 1000)
        tokens_per_sec_single = 1 / (decode_latency_ms / 1000)
        
        # Utilization
        compute_utilization = achievable_tflops / total_compute if total_compute > 0 else 0
        memory_bw_utilization = (
            (achievable_tflops / operational_intensity) / effective_memory_bw
            if not compute_bound and effective_memory_bw > 0 else 1.0
        )
        
        return RooflineMetrics(
            compute_bound=compute_bound,
            operational_intensity=operational_intensity,
            achievable_tflops=achievable_tflops,
            memory_bound_bw_gbps=memory_bound_bw_gbps if memory_bound_bw_gbps else 0,
            tokens_per_sec_single=tokens_per_sec_single,
            tokens_per_sec_batch=tokens_per_sec_batch,
            prefill_latency_ms=0,  # Not applicable for decode
            decode_latency_per_token_ms=decode_latency_ms,
            compute_utilization=min(1.0, compute_utilization),
            memory_bw_utilization=min(1.0, memory_bw_utilization)
        )
    
    def analyze_full_request(
        self,
        vm: VMInstance,
        model: ModelProfile,
        batch_size: int,
        input_tokens: int,
        output_tokens: int,
        quantization: str = "fp16",
        tensor_parallel_size: int = 1,
        cached_tokens: int = 0
    ) -> Dict:
        """Analyze a full request including prefill and decode.
        
        Args:
            vm: VM instance configuration
            model: Model profile
            batch_size: Number of concurrent sequences
            input_tokens: Total input sequence length (including cached)
            output_tokens: Number of output tokens to generate
            quantization: Quantization type
            tensor_parallel_size: Tensor parallel degree
            cached_tokens: Number of input tokens that are cache hits
        
        Returns:
            Dict with combined metrics
        """
        # Analyze prefill (with cache accounting)
        prefill = self.analyze_prefill(
            vm, model, batch_size, input_tokens, quantization, cached_tokens
        )
        
        # Analyze decode (use avg context length for KV cache estimation)
        avg_context_length = input_tokens + output_tokens // 2
        decode = self.analyze_decode(
            vm, model, batch_size, avg_context_length, quantization, tensor_parallel_size
        )
        
        # Combined metrics
        total_latency_ms = prefill.prefill_latency_ms + (output_tokens * decode.decode_latency_per_token_ms)
        
        # End-to-end throughput
        e2e_tokens_per_sec = (
            batch_size * output_tokens / (total_latency_ms / 1000)
        )
        
        # Average utilization
        avg_compute_util = (prefill.compute_utilization + decode.compute_utilization) / 2
        avg_memory_util = (prefill.memory_bw_utilization + decode.memory_bw_utilization) / 2
        
        return {
            "prefill_latency_ms": prefill.prefill_latency_ms,
            "decode_latency_per_token_ms": decode.decode_latency_per_token_ms,
            "total_latency_ms": total_latency_ms,
            "tokens_per_second": e2e_tokens_per_sec,
            "compute_utilization": avg_compute_util,
            "memory_utilization": avg_memory_util,
            "prefill_compute_bound": prefill.compute_bound,
            "decode_compute_bound": decode.compute_bound,
            "prefill": prefill,
            "decode": decode
        }
    
    def find_max_batch_size(
        self,
        vm: VMInstance,
        model: ModelProfile,
        max_seq_length: int,
        quantization: str = "fp16",
        memory_fraction: float = 0.85
    ) -> int:
        """Find maximum batch size that fits in GPU memory.
        
        Args:
            vm: VM instance
            model: Model profile
            max_seq_length: Maximum sequence length
            quantization: Quantization type
            memory_fraction: Fraction of VRAM to use
        
        Returns:
            Maximum batch size
        """
        total_vram_gb = vm.total_vram_gb * memory_fraction
        
        # Model weights
        model_gb = model.get_model_size_for_quantization(quantization)
        
        # vLLM requires additional memory beyond model weights:
        # 1. CUDA context overhead (~1.0-1.5 GB per GPU)
        # 2. Activation buffers (~5-10% of model size during inference)
        # 3. PagedAttention block tables and metadata
        # 4. Temporary buffers for attention computations
        # 5. Safety margin for memory fragmentation
        
        cuda_overhead_gb = 1.5  # Fixed CUDA context overhead
        activation_buffer_gb = model_gb * 0.10  # 10% for activations
        safety_margin_gb = 0.5  # Additional safety margin
        
        fixed_overhead_gb = cuda_overhead_gb + activation_buffer_gb + safety_margin_gb
        
        # Available for KV cache
        available_gb = total_vram_gb - model_gb - fixed_overhead_gb
        
        if available_gb <= 0:
            return 0  # Model doesn't fit with required overhead
        
        # KV cache per token (convert KB to GB: KB -> MB -> GB)
        kv_per_token_gb = model.kv_cache_per_token_kb / 1024 / 1024
        
        # Max tokens that fit
        max_tokens = int(available_gb / kv_per_token_gb)
        
        # Batch size = max_tokens / max_seq_length
        max_batch = int(max_tokens / max_seq_length)
        
        return max(1, max_batch)
    
    def calculate_optimal_batch_size(
        self,
        vm: VMInstance,
        model: ModelProfile,
        target_throughput: float,
        quantization: str = "fp16"
    ) -> int:
        """Calculate optimal batch size for target throughput.
        
        Uses roofline to find batch size that achieves target throughput
        while staying within memory constraints.
        """
        max_batch = self.find_max_batch_size(vm, model, 4096, quantization)
        
        # Binary search for optimal batch size
        low, high = 1, max_batch
        optimal_batch = 1
        
        while low <= high:
            mid = (low + high) // 2
            
            # Analyze with this batch size (use average tokens)
            metrics = self.analyze_decode(vm, model, mid, 2048, quantization)
            
            if metrics.tokens_per_sec_batch >= target_throughput:
                optimal_batch = mid
                low = mid + 1  # Try larger batch
            else:
                high = mid - 1  # Need smaller batch
        
        return optimal_batch
