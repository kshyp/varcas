#!/usr/bin/env python3
"""
Static Roofline Analysis for vLLM Models

This script performs theoretical roofline analysis without running the model,
based on model architecture and hardware specifications.

The roofline model characterizes performance as:
    Attainable FLOPS = min(Peak FLOPS, AI * Peak Bandwidth)
where AI (Arithmetic Intensity) = FLOPs / Bytes

For transformer inference, we have two distinct phases:
1. Prefill (input processing): Compute-bound
2. Decode (token generation): Memory-bound
"""

import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path


@dataclass
class GPUConfig:
    """GPU hardware specifications."""
    name: str
    compute_capability: Tuple[int, int]
    peak_fp16_flops: float  # TFLOPS
    peak_fp32_flops: float  # TFLOPS
    memory_bw: float  # GB/s
    memory_size: float  # GB
    tensor_cores: int
    
    def mem_roofline_x_intercept(self) -> float:
        """The ridge point - minimum AI to achieve peak FLOPS."""
        return self.peak_fp16_flops * 1000 / self.memory_bw


# GPU Specifications Database
GPU_SPECS = {
    "Tesla T4": GPUConfig(
        name="Tesla T4",
        compute_capability=(7, 5),
        peak_fp16_flops=65.0,  # With tensor cores
        peak_fp32_flops=8.1,
        memory_bw=320.0,
        memory_size=16.0,
        tensor_cores=320
    ),
    "A100-SXM4-40GB": GPUConfig(
        name="A100-SXM4-40GB",
        compute_capability=(8, 0),
        peak_fp16_flops=312.0,  # TF32 tensor core
        peak_fp32_flops=19.5,
        memory_bw=1555.0,
        memory_size=40.0,
        tensor_cores=432
    ),
    "A100-SXM4-80GB": GPUConfig(
        name="A100-SXM4-80GB",
        compute_capability=(8, 0),
        peak_fp16_flops=312.0,
        peak_fp32_flops=19.5,
        memory_bw=2039.0,
        memory_size=80.0,
        tensor_cores=432
    ),
    "L4": GPUConfig(
        name="L4",
        compute_capability=(8, 9),
        peak_fp16_flops=121.0,
        peak_fp32_flops=30.3,
        memory_bw=300.0,
        memory_size=24.0,
        tensor_cores=240
    ),
    "RTX 4090": GPUConfig(
        name="RTX 4090",
        compute_capability=(8, 9),
        peak_fp16_flops=82.6,
        peak_fp32_flops=82.6,
        memory_bw=1008.0,
        memory_size=24.0,
        tensor_cores=512
    ),
}


@dataclass
class ModelConfig:
    """LLM Model architecture specifications."""
    name: str
    params: int  # Total parameters (B)
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int  # For GQA/MQA
    intermediate_size: int
    vocab_size: int
    quantization: Optional[str] = None  # "awq", "gptq", "fp16", "int8"
    bits: int = 16  # Bits per parameter
    
    def effective_params(self) -> float:
        """Parameters after quantization."""
        return self.params * self.bits / 16


def get_llama2_7b_config(quantization: Optional[str] = None, bits: int = 16) -> ModelConfig:
    """Llama-2-7B configuration."""
    return ModelConfig(
        name="Llama-2-7B" + (f"-{quantization.upper()}" if quantization else ""),
        params=6.7,  # Actual ~6.7B parameters
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=32,  # Llama-2 7B uses MHA, not GQA
        intermediate_size=11008,
        vocab_size=32000,
        quantization=quantization,
        bits=bits
    )


class RooflineAnalyzer:
    """
    Static roofline analyzer for transformer inference.
    
    Computes theoretical arithmetic intensity and performance bounds.
    """
    
    def __init__(self, model: ModelConfig, gpu: GPUConfig):
        self.model = model
        self.gpu = gpu
        self.bytes_per_param = model.bits / 8
        
    def compute_prefill_ops(self, batch_size: int, seq_len: int) -> Tuple[float, float]:
        """
        Compute FLOPs and memory bytes for prefill phase.
        
        For prefill (processing input tokens):
        - Self-attention: 2 * batch * seq_len^2 * hidden_size per layer
        - MLP: 2 * batch * seq_len * hidden_size * intermediate_ratio per layer
        
        Returns: (flops, bytes)
        """
        b = batch_size
        s = seq_len
        h = self.model.hidden_size
        i = self.model.intermediate_size
        l = self.model.num_layers
        
        # Attention FLOPs: QKV projection + attention computation + output projection
        # 2 * b * s * h * (3h) for QKV + 2 * b * s^2 * h for attention + 2 * b * s * h^2 for output
        attn_flops = 2 * b * s * h * (3 * h + s + h) * l
        
        # MLP FLOPs: gate/up projection + down projection
        # 2 * b * s * h * i + 2 * b * s * i * h
        mlp_flops = 4 * b * s * h * i * l
        
        # Embedding + LM Head (negligible for long sequences, but include for completeness)
        embed_flops = 2 * b * s * self.model.vocab_size * h
        
        total_flops = attn_flops + mlp_flops + embed_flops
        
        # Memory traffic: reading all weights once (weights dominate for prefill)
        # Plus activations (KV cache writes, intermediate activations)
        weight_bytes = self.model.params * 1e9 * self.bytes_per_param
        
        # Activation memory: Q, K, V, attention scores, MLP intermediates
        # Rough estimate: ~4 * b * s * h * l * 4 bytes (fp32)
        activation_bytes = 4 * b * s * h * l * 4
        
        total_bytes = weight_bytes + activation_bytes
        
        return total_flops, total_bytes
    
    def compute_decode_ops(self, batch_size: int, seq_len: int) -> Tuple[float, float]:
        """
        Compute FLOPs and memory bytes for decode phase (per token).
        
        For decode (generating one token):
        - Much lower compute per token (only one new token)
        - KV cache reads dominate memory traffic
        
        Returns: (flops, bytes)
        """
        b = batch_size
        s = seq_len
        h = self.model.hidden_size
        i = self.model.intermediate_size
        l = self.model.num_layers
        
        # Attention FLOPs for single token: much less than prefill
        # QKV proj: 2 * b * 1 * h * 3h
        # Attention with cached K,V: 2 * b * 1 * s * h (read from cache)
        # Output proj: 2 * b * 1 * h^2
        attn_flops = 2 * b * h * (3 * h + s + h) * l
        
        # MLP FLOPs (same as prefill but for single token)
        mlp_flops = 4 * b * h * i * l
        
        # LM head
        lm_head_flops = 2 * b * h * self.model.vocab_size
        
        total_flops = attn_flops + mlp_flops + lm_head_flops
        
        # Memory traffic: weights + KV cache reads
        weight_bytes = self.model.params * 1e9 * self.bytes_per_param
        
        # KV cache: reading all past K,V for attention
        # 2 (K,V) * b * s * h * l * bytes_per_elem
        kv_cache_bytes = 2 * b * s * h * l * self.bytes_per_param
        
        total_bytes = weight_bytes + kv_cache_bytes
        
        return total_flops, total_bytes
    
    def analyze_prefill(self, batch_sizes: List[int], seq_lens: List[int]) -> List[Dict]:
        """Analyze prefill phase across different configurations."""
        results = []
        
        for batch in batch_sizes:
            for seq_len in seq_lens:
                flops, bytes_accessed = self.compute_prefill_ops(batch, seq_len)
                
                # Arithmetic intensity
                ai = flops / bytes_accessed if bytes_accessed > 0 else 0
                
                # Performance bounds
                mem_bound_perf = ai * self.gpu.memory_bw * 1e9  # FLOPS
                compute_bound_perf = self.gpu.peak_fp16_flops * 1e12
                
                # Attainable performance
                attainable = min(mem_bound_perf, compute_bound_perf)
                utilization = attainable / compute_bound_perf * 100
                
                # Time estimate
                time_sec = flops / attainable if attainable > 0 else float('inf')
                
                results.append({
                    "phase": "prefill",
                    "batch_size": batch,
                    "seq_len": seq_len,
                    "total_tokens": batch * seq_len,
                    "flops": flops,
                    "bytes": bytes_accessed,
                    "arithmetic_intensity": ai,
                    "memory_bound_tflops": mem_bound_perf / 1e12,
                    "compute_bound_tflops": compute_bound_perf / 1e12,
                    "attainable_tflops": attainable / 1e12,
                    "compute_utilization_pct": utilization,
                    "estimated_time_ms": time_sec * 1000,
                    "tokens_per_sec": (batch * seq_len) / time_sec if time_sec > 0 else 0
                })
        
        return results
    
    def analyze_decode(self, batch_sizes: List[int], seq_lens: List[int]) -> List[Dict]:
        """Analyze decode phase (per token) across different configurations."""
        results = []
        
        for batch in batch_sizes:
            for seq_len in seq_lens:
                flops, bytes_accessed = self.compute_decode_ops(batch, seq_len)
                
                ai = flops / bytes_accessed if bytes_accessed > 0 else 0
                
                mem_bound_perf = ai * self.gpu.memory_bw * 1e9
                compute_bound_perf = self.gpu.peak_fp16_flops * 1e12
                
                attainable = min(mem_bound_perf, compute_bound_perf)
                utilization = attainable / compute_bound_perf * 100
                
                time_sec = flops / attainable if attainable > 0 else float('inf')
                
                results.append({
                    "phase": "decode",
                    "batch_size": batch,
                    "context_len": seq_len,
                    "flops": flops,
                    "bytes": bytes_accessed,
                    "arithmetic_intensity": ai,
                    "memory_bound_tflops": mem_bound_perf / 1e12,
                    "compute_bound_tflops": compute_bound_perf / 1e12,
                    "attainable_tflops": attainable / 1e12,
                    "compute_utilization_pct": utilization,
                    "time_per_token_ms": time_sec * 1000,
                    "tokens_per_sec": 1.0 / time_sec if time_sec > 0 else 0
                })
        
        return results
    
    def analyze_workload(self, profile_name: str, input_mean: int, output_mean: int,
                         target_rps: float, duration: int) -> Dict:
        """
        Analyze a complete workload based on varcas load profile.
        
        Args:
            profile_name: Name of the load profile
            input_mean: Mean input tokens per request
            output_mean: Mean output tokens per request
            target_rps: Target requests per second
            duration: Duration in seconds
        """
        total_requests = int(target_rps * duration)
        
        # Average batch size (assuming continuous batching)
        # Rough estimate: if decode takes T seconds per token, batch ~ T * RPS
        # For 20ms per token and 20 RPS, batch ~ 1
        assumed_time_per_token = 0.02  # 20ms estimate
        avg_batch_size = max(1, int(assumed_time_per_token * target_rps))
        
        # Prefill analysis
        prefill_flops, prefill_bytes = self.compute_prefill_ops(avg_batch_size, input_mean)
        prefill_ai = prefill_flops / prefill_bytes
        
        # Decode analysis (for average context length ~ input_mean + output_mean/2)
        avg_context = input_mean + output_mean // 2
        decode_flops, decode_bytes = self.compute_decode_ops(avg_batch_size, avg_context)
        decode_ai = decode_flops / decode_bytes
        
        # Total tokens
        total_input_tokens = total_requests * input_mean
        total_output_tokens = total_requests * output_mean
        
        # Memory required for KV cache
        kv_cache_per_token = (2 * self.model.hidden_size * self.model.num_layers * 
                              self.bytes_per_param)  # K + V
        max_kv_cache = (avg_batch_size * (input_mean + output_mean) * 
                        kv_cache_per_token / 1e9)  # GB
        
        return {
            "profile_name": profile_name,
            "gpu": self.gpu.name,
            "model": self.model.name,
            "workload_params": {
                "target_rps": target_rps,
                "duration_sec": duration,
                "total_requests": total_requests,
                "input_mean": input_mean,
                "output_mean": output_mean,
                "avg_batch_size": avg_batch_size
            },
            "prefill": {
                "arithmetic_intensity": prefill_ai,
                "ridge_point": self.gpu.mem_roofline_x_intercept(),
                "compute_bound": prefill_ai > self.gpu.mem_roofline_x_intercept(),
                "estimated_tflops": min(
                    self.gpu.peak_fp16_flops,
                    prefill_ai * self.gpu.memory_bw / 1000
                )
            },
            "decode": {
                "arithmetic_intensity": decode_ai,
                "ridge_point": self.gpu.mem_roofline_x_intercept(),
                "compute_bound": decode_ai > self.gpu.mem_roofline_x_intercept(),
                "estimated_tflops": min(
                    self.gpu.peak_fp16_flops,
                    decode_ai * self.gpu.memory_bw / 1000
                )
            },
            "memory": {
                "model_size_gb": self.model.params * self.bytes_per_param,
                "kv_cache_max_gb": max_kv_cache,
                "total_memory_required_gb": (self.model.params * self.bytes_per_param + max_kv_cache)
            },
            "bottleneck": "memory" if decode_ai < self.gpu.mem_roofline_x_intercept() else "compute"
        }
    
    def generate_roofline_plot_data(self, min_ai: float = 0.1, max_ai: float = 1000, 
                                     points: int = 1000) -> Dict:
        """Generate data for roofline plot."""
        import math
        
        # Log-spaced AI values
        log_min = math.log10(min_ai)
        log_max = math.log10(max_ai)
        ai_values = [10 ** (log_min + (log_max - log_min) * i / points) for i in range(points)]
        
        roofline = []
        for ai in ai_values:
            mem_bound = ai * self.gpu.memory_bw / 1000  # Convert to TFLOPS
            compute_bound = self.gpu.peak_fp16_flops
            roofline.append({
                "ai": ai,
                "memory_bound_tflops": mem_bound,
                "compute_bound_tflops": compute_bound,
                "roofline_tflops": min(mem_bound, compute_bound)
            })
        
        return {
            "gpu": self.gpu.name,
            "peak_tflops": self.gpu.peak_fp16_flops,
            "memory_bw_gbps": self.gpu.memory_bw,
            "ridge_point": self.gpu.mem_roofline_x_intercept(),
            "roofline": roofline
        }


def detect_gpu() -> Optional[GPUConfig]:
    """Auto-detect GPU from system."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        gpu_name = result.stdout.strip()
        
        # Find matching GPU
        for name, config in GPU_SPECS.items():
            if name in gpu_name:
                return config
        
        print(f"Warning: Unknown GPU '{gpu_name}', using T4 defaults")
        return GPU_SPECS["Tesla T4"]
    except Exception as e:
        print(f"Could not detect GPU: {e}")
        return None


def main():
    """Run static roofline analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Static Roofline Analysis for LLMs")
    parser.add_argument("--model", default="llama2-7b", help="Model name")
    parser.add_argument("--quantization", default="awq", help="Quantization type")
    parser.add_argument("--bits", type=int, default=4, help="Bits per parameter")
    parser.add_argument("--gpu", default=None, help="GPU name (auto-detect if not specified)")
    parser.add_argument("--output", default="varcas/profiles/roofline/static_analysis.json",
                       help="Output file")
    args = parser.parse_args()
    
    # Get GPU config
    if args.gpu and args.gpu in GPU_SPECS:
        gpu = GPU_SPECS[args.gpu]
    else:
        gpu = detect_gpu() or GPU_SPECS["Tesla T4"]
    
    print(f"GPU: {gpu.name}")
    print(f"  Peak FP16: {gpu.peak_fp16_flops} TFLOPS")
    print(f"  Memory BW: {gpu.memory_bw} GB/s")
    print(f"  Ridge Point: {gpu.mem_roofline_x_intercept():.2f} FLOP/Byte")
    
    # Get model config
    if "llama2-7b" in args.model.lower() or "llama-2-7b" in args.model.lower():
        model = get_llama2_7b_config(args.quantization, args.bits)
    else:
        print(f"Unknown model: {args.model}")
        return
    
    print(f"\nModel: {model.name}")
    print(f"  Parameters: {model.params}B")
    print(f"  Quantization: {args.quantization} ({args.bits}-bit)")
    print(f"  Effective Model Size: {model.effective_params():.2f}B params")
    
    # Create analyzer
    analyzer = RooflineAnalyzer(model, gpu)
    
    # Run analyses
    print("\n" + "="*60)
    print("PREFILL PHASE ANALYSIS")
    print("="*60)
    
    prefill_results = analyzer.analyze_prefill(
        batch_sizes=[1, 4, 8, 16],
        seq_lens=[64, 128, 256, 512, 1024]
    )
    
    # Print summary for key configurations
    for r in prefill_results:
        if r["batch_size"] == 1 and r["seq_len"] in [128, 512]:
            print(f"\nBatch={r['batch_size']}, Seq={r['seq_len']}:")
            print(f"  Arithmetic Intensity: {r['arithmetic_intensity']:.2f} FLOP/Byte")
            print(f"  Attainable: {r['attainable_tflops']:.2f} TFLOPS ({r['compute_utilization_pct']:.1f}%)")
            print(f"  Region: {'COMPUTE' if r['compute_utilization_pct'] > 90 else 'MEMORY'}-bound")
    
    print("\n" + "="*60)
    print("DECODE PHASE ANALYSIS")
    print("="*60)
    
    decode_results = analyzer.analyze_decode(
        batch_sizes=[1, 4, 8, 16],
        seq_lens=[128, 512, 1024, 2048]
    )
    
    for r in decode_results:
        if r["batch_size"] == 1 and r["context_len"] in [512, 2048]:
            print(f"\nBatch={r['batch_size']}, Context={r['context_len']}:")
            print(f"  Arithmetic Intensity: {r['arithmetic_intensity']:.2f} FLOP/Byte")
            print(f"  Attainable: {r['attainable_tflops']:.2f} TFLOPS ({r['compute_utilization_pct']:.1f}%)")
            print(f"  Time/Token: {r['time_per_token_ms']:.2f} ms")
            print(f"  Region: {'COMPUTE' if r['compute_utilization_pct'] > 90 else 'MEMORY'}-bound")
    
    # Workload analysis for chat profiles
    print("\n" + "="*60)
    print("WORKLOAD ANALYSIS (Chat Profiles)")
    print("="*60)
    
    profiles = [
        ("chat_low", 5, 50, 150),
        ("chat_medium", 20, 50, 150),
        ("chat_high", 50, 50, 150),
    ]
    
    workload_results = []
    for name, rps, in_mean, out_mean in profiles:
        result = analyzer.analyze_workload(name, in_mean, out_mean, rps, 60)
        workload_results.append(result)
        
        print(f"\n{name} ({rps} RPS):")
        print(f"  Prefill AI: {result['prefill']['arithmetic_intensity']:.2f}")
        print(f"  Decode AI: {result['decode']['arithmetic_intensity']:.2f}")
        print(f"  Dominant bottleneck: {result['bottleneck'].upper()}")
        print(f"  Model + KV Cache: {result['memory']['total_memory_required_gb']:.2f} GB")
    
    # Generate roofline plot data
    roofline_data = analyzer.generate_roofline_plot_data()
    
    # Mark workload points on roofline
    roofline_data["workload_points"] = [
        {
            "name": "prefill_typical",
            "ai": prefill_results[5]["arithmetic_intensity"],  # batch=1, seq=256
            "tflops": prefill_results[5]["attainable_tflops"],
            "label": "Prefill (B=1, S=256)"
        },
        {
            "name": "decode_typical",
            "ai": decode_results[2]["arithmetic_intensity"],  # batch=1, ctx=512
            "tflops": decode_results[2]["attainable_tflops"],
            "label": "Decode (B=1, CTX=512)"
        },
        {
            "name": "decode_long",
            "ai": decode_results[6]["arithmetic_intensity"],  # batch=1, ctx=2048
            "tflops": decode_results[6]["attainable_tflops"],
            "label": "Decode (B=1, CTX=2048)"
        }
    ]
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    full_results = {
        "analysis_type": "static_roofline",
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "gpu": asdict(gpu),
        "model": asdict(model),
        "prefill_results": prefill_results,
        "decode_results": decode_results,
        "workload_analysis": workload_results,
        "roofline_plot_data": roofline_data
    }
    
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print("="*60)
    
    # Summary
    print("\nSUMMARY:")
    print(f"  GPU: {gpu.name}")
    print(f"  Model: {model.name} ({model.effective_params():.2f}B effective params)")
    print(f"  Ridge Point: {gpu.mem_roofline_x_intercept():.2f} FLOP/Byte")
    print(f"\n  Phase Characteristics:")
    print(f"    PREFILL: Compute-bound (high arithmetic intensity)")
    print(f"    DECODE:  Memory-bound (low arithmetic intensity)")
    print(f"\n  Optimization Opportunities:")
    print(f"    - Quantization reduces memory pressure (current: {args.bits}-bit)")
    print(f"    - Larger batch sizes increase AI for decode")
    print(f"    - Continuous batching helps overlap prefill/decode")


if __name__ == "__main__":
    main()
