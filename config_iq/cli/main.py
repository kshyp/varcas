"""CLI interface for the hardware sizing tool."""

import argparse
import json
import sys
from typing import List, Optional

try:
    from ..core.hardware_catalog import HardwareCatalog
    from ..core.workload_patterns import WorkloadPatterns
    from ..core.model_profiler import get_model_profile
    from ..core.roofline_analyzer import RooflineAnalyzer
    from ..core.queuing_model import MG1QueuingModel
    from ..core.sla_calculator import SLACalculator, SLAMetrics
    from ..core.types import (
        SizingInput, WorkloadType, PricingModel, SizingRecommendation, HardwareConfig
    )
except ImportError:
    from config_iq.core.hardware_catalog import HardwareCatalog
    from config_iq.core.workload_patterns import WorkloadPatterns
    from config_iq.core.model_profiler import get_model_profile
    from config_iq.core.roofline_analyzer import RooflineAnalyzer
    from config_iq.core.queuing_model import MG1QueuingModel
    from config_iq.core.sla_calculator import SLACalculator, SLAMetrics
    from config_iq.core.types import (
        SizingInput, WorkloadType, PricingModel, SizingRecommendation, HardwareConfig
    )


def parse_hardware_configs(args, hardware_catalog) -> List[HardwareConfig]:
    """Parse hardware configurations from args or configs file.
    
    Returns a list of HardwareConfig objects to evaluate.
    """
    configs = []
    
    # Check if configs file is provided
    if args.configs_file:
        with open(args.configs_file, 'r') as f:
            data = json.load(f)
            for cfg in data.get('configs', []):
                configs.append(HardwareConfig(
                    vcpus=cfg['vcpus'],
                    memory_gb=cfg['memory_gb'],
                    gpu_type=cfg['gpu_type'],
                    gpu_count=cfg['gpu_count'],
                    gpu_vram_gb=cfg.get('gpu_vram_gb', lookup_gpu_vram(cfg['gpu_type'], hardware_catalog)),
                    name=cfg.get('name', f"{cfg['gpu_type']}-{cfg['gpu_count']}x")
                ))
        return configs
    
    # Check if individual hardware args are provided
    if args.gpu_type and args.gpu_count:
        # Get GPU VRAM from catalog if not specified
        gpu_vram = args.gpu_vram_gb
        if gpu_vram is None:
            gpu_vram = lookup_gpu_vram(args.gpu_type, hardware_catalog)
            if gpu_vram is None:
                print(f"Error: Unknown GPU type '{args.gpu_type}'. Please specify --gpu-vram-gb.")
                return []
        
        # Default vcpus and memory if not specified
        vcpus = args.vcpus if args.vcpus else 4  # Sensible default
        memory_gb = args.memory_gb if args.memory_gb else 15  # Sensible default
        
        configs.append(HardwareConfig(
            vcpus=vcpus,
            memory_gb=memory_gb,
            gpu_type=args.gpu_type,
            gpu_count=args.gpu_count,
            gpu_vram_gb=gpu_vram,
            name=args.config_name
        ))
        return configs
    
    # No hardware config specified - will use catalog mode
    return []


def lookup_gpu_vram(gpu_type: str, hardware_catalog) -> Optional[int]:
    """Look up GPU VRAM from catalog."""
    for vm in hardware_catalog.get_all_instances():
        for gpu in vm.gpus:
            if gpu.type == gpu_type:
                return gpu.vram_gb
    return None


def format_latency(value: float) -> str:
    """Format latency value."""
    if value == float('inf'):
        return "∞"
    return f"{value:.0f}"


def format_cost(value: float) -> str:
    """Format cost value."""
    if value == float('inf'):
        return "∞"
    elif value < 0.001:
        return f"${value:.6f}"
    elif value < 0.01:
        return f"${value:.5f}"
    else:
        return f"${value:.4f}"


def generate_vllm_command(rec: SizingRecommendation, model_id: str, quantization: str, 
                          mode: str = "safe", max_batch_size: int = None) -> str:
    """Generate recommended vLLM server start command based on roofline analysis assumptions.
    
    Args:
        rec: Sizing recommendation
        model_id: HuggingFace model ID
        quantization: Quantization type (fp16, bf16, int8, etc.)
        mode: "safe" or "performance" - safe uses conservative settings,
              performance uses aggressive settings for max throughput
        max_batch_size: Maximum batch size calculated from roofline analysis
    
    Returns:
        Recommended vLLM command string
    """
    gpu_count = rec.vm_instance.gpu_count
    tp_size = rec.tensor_parallel_size
    
    # Calculate memory headroom per GPU to determine if this is a tight fit
    # With tensor parallelism, model is sharded across GPUs
    gpu_vram = rec.vm_instance.max_vram_per_gpu_gb
    from config_iq.core.model_profiler import get_model_profile
    model = get_model_profile(model_id)
    model_size_gb = model.get_model_size_for_quantization(quantization)
    
    # Model size per GPU with TP
    model_per_gpu = model_size_gb / tp_size if tp_size > 0 else model_size_gb
    
    # Calculate actual overhead per GPU (same as compatibility check)
    cuda_overhead_gb = 1.5
    activation_buffer_gb = model_per_gpu * 0.10
    safety_margin_gb = 0.5
    fixed_overhead_gb = cuda_overhead_gb + activation_buffer_gb + safety_margin_gb
    
    # Memory headroom per GPU after model weights + overhead (available for KV cache)
    memory_headroom_gb = gpu_vram - model_per_gpu - fixed_overhead_gb
    is_tight_fit = memory_headroom_gb < 8.0  # Less than 8GB for KV cache is tight
    is_very_tight = memory_headroom_gb < 4.0  # Less than 4GB is very tight
    
    # Mode-specific settings
    if mode == "safe":
        # For tight fits, use more conservative memory settings
        if is_very_tight:
            gpu_memory_util = 0.75  # Very conservative for T4 with 7B models
        elif is_tight_fit:
            gpu_memory_util = 0.80  # Conservative for tight fits
        else:
            gpu_memory_util = 0.85  # Standard safe setting
        
        max_num_seqs = max_batch_size // 2 if max_batch_size else 64  # Conservative
        batched_tokens = 2048
        scheduler_delay_factor = 0.0
    else:  # performance
        if is_very_tight:
            gpu_memory_util = 0.80  # Even performance mode needs to be careful
        elif is_tight_fit:
            gpu_memory_util = 0.85
        else:
            gpu_memory_util = 0.93
        
        max_num_seqs = max_batch_size if max_batch_size else 256  # Aggressive
        batched_tokens = 4096
        scheduler_delay_factor = 0.1  # Slight delay for better batching
    
    # Build the command
    cmd_parts = ["vllm serve", model_id]
    
    # Tensor parallel size
    if tp_size > 1:
        cmd_parts.append(f"--tensor-parallel-size {tp_size}")
    
    # Quantization settings
    if quantization == "int8":
        cmd_parts.append("--quantization int8")
        cmd_parts.append("--kv-cache-dtype int8")
    elif quantization == "int4":
        cmd_parts.append("--quantization awq")  # or gptq depending on model
    elif quantization == "fp8":
        cmd_parts.append("--quantization fp8")
        cmd_parts.append("--kv-cache-dtype fp8")
    elif quantization == "bf16":
        cmd_parts.append("--dtype bfloat16")
    elif quantization in ["fp16", "fp32"]:
        cmd_parts.append("--dtype auto")  # vLLM will select fp16
    
    # GPU memory utilization
    cmd_parts.append(f"--gpu-memory-utilization {gpu_memory_util}")
    
    # Max model length - use 2048 as default (many models don't support 4096)
    # User can increase this if their model supports longer contexts
    cmd_parts.append("--max-model-len 2048")
    
    # Batch size configuration (from roofline analysis)
    if max_num_seqs:
        cmd_parts.append(f"--max-num-seqs {max_num_seqs}")
    
    # Performance optimizations
    cmd_parts.append("--enable-prefix-caching")
    cmd_parts.append("--enable-chunked-prefill")
    
    # Scheduler configuration (performance mode only)
    # Note: --scheduler-delay-factor is not available in all vLLM versions
    if mode == "performance":
        cmd_parts.append(f"--max-num-batched-tokens {batched_tokens}")
    
    # Combine into command
    return " \\\n    ".join(cmd_parts)


def print_vllm_command(rec: SizingRecommendation, model_id: str, quantization: str, 
                       indent: str = "", max_batch_size: int = None):
    """Print both safe and performance vLLM commands for a configuration."""
    # Calculate memory settings for display (accounting for tensor parallelism)
    from config_iq.core.model_profiler import get_model_profile
    model = get_model_profile(model_id)
    model_size_gb = model.get_model_size_for_quantization(quantization)
    gpu_vram = rec.vm_instance.max_vram_per_gpu_gb
    tp_size = rec.tensor_parallel_size
    
    # Model size per GPU with TP
    model_per_gpu = model_size_gb / tp_size if tp_size > 0 else model_size_gb
    
    cuda_overhead_gb = 1.5
    activation_buffer_gb = model_per_gpu * 0.10
    safety_margin_gb = 0.5
    fixed_overhead_gb = cuda_overhead_gb + activation_buffer_gb + safety_margin_gb
    memory_headroom_gb = gpu_vram - model_per_gpu - fixed_overhead_gb
    is_tight_fit = memory_headroom_gb < 8.0
    is_very_tight = memory_headroom_gb < 4.0
    
    # Determine memory util values for display
    if is_very_tight:
        safe_mem = 75
        perf_mem = 80
    elif is_tight_fit:
        safe_mem = 80
        perf_mem = 85
    else:
        safe_mem = 85
        perf_mem = 93
    
    # Safe mode command
    safe_cmd = generate_vllm_command(rec, model_id, quantization, mode="safe", 
                                      max_batch_size=max_batch_size)
    
    # Performance mode command
    perf_cmd = generate_vllm_command(rec, model_id, quantization, mode="performance",
                                      max_batch_size=max_batch_size)
    
    print(f"\n{indent}vLLM Commands:")
    print(f"{indent}{'=' * 70}")
    
    print(f"\n{indent}Option 1: SAFE (Conservative - SLA Pass/Fail based on this)")
    print(f"{indent}{'-' * 70}")
    # Print with backslash continuation for copy-paste
    lines = safe_cmd.split(" \\\n    ")
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            print(f"{indent}{line} \\")
        else:
            print(f"{indent}{line}")
    print(f"{indent}{'-' * 70}")
    print(f"{indent}Settings: {safe_mem}% VRAM, conservative batching, reliable")
    
    print(f"\n{indent}Option 2: HIGH PERFORMANCE (Aggressive)")
    print(f"{indent}{'-' * 70}")
    # Print with backslash continuation for copy-paste
    lines = perf_cmd.split(" \\\n    ")
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            print(f"{indent}{line} \\")
        else:
            print(f"{indent}{line}")
    print(f"{indent}{'-' * 70}")
    print(f"{indent}Settings: {perf_mem}% VRAM, max batching, higher throughput")
    print(f"{indent}⚠ Higher risk of OOM under burst load")
    print()
    print(f"{indent}NOTE: If vLLM reports errors:")
    print(f"{indent}  • For 'max_model_len > derived max' error: Reduce --max-model-len to match")
    print(f"{indent}    the model's max_position_embeddings (check config.json)")
    print(f"{indent}  • Set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 to override (use with caution)")
    print(f"{indent}  • Remove unsupported args if your vLLM version doesn't support them")


def print_sla_table(recommendation: SizingRecommendation, workload_type: WorkloadType):
    """Print the SLA comparison table."""
    calc = SLACalculator(
        HardwareCatalog(),
        WorkloadPatterns(),
        RooflineAnalyzer(HardwareCatalog())
    )
    
    metrics = calc.get_sla_comparison_table(recommendation, workload_type)
    perf = recommendation.performance
    
    print("\n" + "=" * 80)
    print("SLA COMPLIANCE REPORT")
    print("=" * 80)
    print(f"\nConfiguration: {recommendation.vm_instance.name}")
    print(f"GPU: {recommendation.vm_instance.gpu_type} x {recommendation.vm_instance.gpu_count}")
    print(f"Tensor Parallel: {recommendation.tensor_parallel_size}")
    print(f"Instances needed: {recommendation.num_instances_needed}")
    print(f"Headroom: {recommendation.headroom_percent}%")
    print()
    
    # Print table header
    print(f"{'Metric':<20} {'Target':<15} {'Predicted':<15} {'Status':<10} {'Gap':<10}")
    print("-" * 70)
    
    # Print rows
    for m in metrics:
        target_str = format_latency(m.target_value) if "P95" not in m.metric_name and "P99" not in m.metric_name else "—"
        gap_str = f"{m.gap_percent:+.1f}%" if m.gap_percent != 0 else "—"
        print(f"{m.metric_name:<20} {target_str:<15} {format_latency(m.predicted_value):<15} {m.status:<10} {gap_str:<10}")
    
    # Print tail latency multipliers
    print()
    print("Tail Latency Analysis:")
    if perf.ttft_p50_ms > 0:
        p95_mult = perf.ttft_p95_ms / perf.ttft_p50_ms
        p99_mult = perf.ttft_p99_ms / perf.ttft_p50_ms
        
        health = "✓ Healthy" if p95_mult < 3 else "~ Acceptable" if p95_mult < 5 else "⚠ High Variance"
        print(f"  TTFT P95: {format_latency(perf.ttft_p95_ms)} ({p95_mult:.1f}x P50) - {health}")
        
        health = "✓ Healthy" if p99_mult < 4 else "~ Acceptable" if p99_mult < 6 else "⚠ High Variance"
        print(f"  TTFT P99: {format_latency(perf.ttft_p99_ms)} ({p99_mult:.1f}x P50) - {health}")


def print_throughput_analysis(recommendation: SizingRecommendation):
    """Print throughput analysis."""
    perf = recommendation.performance
    
    print("\n" + "=" * 80)
    print("THROUGHPUT ANALYSIS")
    print("=" * 80)
    print()
    print(f"Throughput at SLA point:")
    print(f"  • {perf.throughput_tok_s:.0f} tok/s")
    print(f"  • {perf.throughput_req_s:.1f} req/s")
    print(f"  • {perf.gpu_utilization*100:.0f}% GPU utilization")
    print()
    print(f"Burst capacity ({recommendation.peak_capacity_factor:.1f}x traffic):")
    print(f"  • {perf.burst_throughput_tok_s:.0f} tok/s")
    print(f"  • {perf.burst_gpu_utilization*100:.0f}% GPU utilization")
    print()
    print(f"Requests per hour: {perf.throughput_req_s * 3600:.0f}")
    print(f"Tokens per hour: {perf.throughput_tok_s * 3600:.0f}")


def print_cost_analysis(recommendation: SizingRecommendation):
    """Print cost analysis."""
    print("\n" + "=" * 80)
    print("COST ANALYSIS")
    print("=" * 80)
    print()
    print(f"Hourly cost per instance: {format_cost(recommendation.hourly_cost_usd)}")
    print(f"Total hourly cost ({recommendation.num_instances_needed} instances): {format_cost(recommendation.total_hourly_cost_usd)}")
    print()
    print(f"Cost per 1K tokens: {format_cost(recommendation.cost_per_1k_tokens_usd)}")
    print(f"Cost per request: {format_cost(recommendation.cost_per_request_usd)}")
    print()
    print(f"Monthly estimate (730 hours): {format_cost(recommendation.total_hourly_cost_usd * 730)}")


def calculate_max_batch_size_for_config(rec: SizingRecommendation, model_id: str, quantization: str) -> int:
    """Calculate max batch size for a configuration using roofline analysis.
    
    Returns:
        Maximum batch size that fits in GPU memory
    """
    try:
        from ..core.model_profiler import get_model_profile
        from ..core.roofline_analyzer import RooflineAnalyzer
        from ..core.hardware_catalog import HardwareCatalog
    except ImportError:
        from config_iq.core.model_profiler import get_model_profile
        from config_iq.core.roofline_analyzer import RooflineAnalyzer
        from config_iq.core.hardware_catalog import HardwareCatalog
    
    hardware = HardwareCatalog()
    roofline = RooflineAnalyzer(hardware)
    model = get_model_profile(model_id)
    
    # Use 2048 as max_seq_len to match the vLLM command's --max-model-len
    max_batch = roofline.find_max_batch_size(
        rec.vm_instance, model, 2048, quantization, memory_fraction=0.93
    )
    
    return max(1, max_batch)


def print_recommendation_card(rec: SizingRecommendation, rank: int, title: str = "RECOMMENDATION", 
                               model_id: str = None, quantization: str = None, max_batch_size: int = None):
    """Print a compact recommendation card."""
    status = "✓ MEETS SLA" if rec.meets_sla else "✗ DOES NOT MEET SLA"
    print(f"\n{'='*80}")
    print(f"{title} #{rank}: {rec.vm_instance.name} - {status}")
    print(f"{'='*80}")
    print(f"  GPU: {rec.vm_instance.gpu_type} x {rec.vm_instance.gpu_count}")
    print(f"  Tensor Parallel: {rec.tensor_parallel_size}")
    print(f"  Instances needed: {rec.num_instances_needed}")
    print(f"  Hourly cost: {format_cost(rec.total_hourly_cost_usd)}/hr")
    print(f"  Cost/request: {format_cost(rec.cost_per_request_usd)}")
    print(f"  Throughput: {rec.performance.throughput_tok_s:.0f} tok/s")
    print(f"  TTFT P50: {format_latency(rec.performance.ttft_p50_ms)}ms")
    print(f"  TPOT: {format_latency(rec.performance.tpot_p50_ms)}ms/token")
    
    # Print vLLM command if model_id and quantization are provided
    if model_id and quantization:
        print_vllm_command(rec, model_id, quantization, indent="  ", max_batch_size=max_batch_size)


def calculate_max_sla_gap(rec: SizingRecommendation) -> float:
    """Calculate the maximum SLA gap (worst violation) for a recommendation.
    
    Returns the largest positive gap percentage. Higher value means worse SLA achievement.
    """
    if not rec.sla_gaps:
        return 0.0
    # Get the maximum gap (worst violation)
    return max(rec.sla_gaps.values())


def print_all_configurations(recommendations: List[SizingRecommendation], model_id: str, quantization: str):
    """Print all evaluated configurations in two parts:
    
    1. Passing configurations - sorted by increasing cost/hour
    2. Failing configurations - sorted by worst to better SLA achieved
    """
    # Split into passing and failing
    passing = [r for r in recommendations if r.meets_sla]
    failing = [r for r in recommendations if not r.meets_sla]
    
    # Sort passing by total hourly cost (ascending - cheapest first)
    passing.sort(key=lambda r: r.total_hourly_cost_usd)
    
    # Sort failing by worst SLA gap (largest positive gap first)
    failing.sort(key=lambda r: calculate_max_sla_gap(r), reverse=True)
    
    # Pre-calculate max batch sizes for all unique instances
    print("\nCalculating optimal batch sizes for configurations...")
    max_batch_sizes = {}
    for rec in recommendations:
        instance_key = rec.vm_instance.name
        if instance_key not in max_batch_sizes:
            max_batch_sizes[instance_key] = calculate_max_batch_size_for_config(
                rec, model_id, quantization
            )
    
    print("\n" + "=" * 80)
    print("ALL EVALUATED HARDWARE CONFIGURATIONS")
    print("=" * 80)
    print(f"Total configurations evaluated: {len(recommendations)}")
    print(f"  - Passing (meet SLA): {len(passing)}")
    print(f"  - Failing (miss SLA): {len(failing)}")
    
    # Part 1: Passing configurations
    print("\n" + "-" * 80)
    print("PART 1: PASSING CONFIGURATIONS (sorted by increasing cost/hour)")
    print("-" * 80)
    
    if passing:
        print(f"Showing all {len(passing)} passing configurations:\n")
        for i, rec in enumerate(passing, 1):
            max_batch = max_batch_sizes.get(rec.vm_instance.name, 64)
            print_recommendation_card(rec, i, "PASSING CONFIG", model_id, quantization, max_batch)
    else:
        print("\nNo passing configurations found.")
    
    # Part 2: Failing configurations
    print("\n" + "-" * 80)
    print("PART 2: FAILING CONFIGURATIONS (sorted by worst to better SLA)")
    print("-" * 80)
    
    if failing:
        print(f"Showing all {len(failing)} failing configurations:\n")
        for i, rec in enumerate(failing, 1):
            max_batch = max_batch_sizes.get(rec.vm_instance.name, 64)
            print_recommendation_card(rec, i, "FAILING CONFIG", model_id, quantization, max_batch)
            # Show the worst SLA gap
            max_gap = calculate_max_sla_gap(rec)
            print(f"  Worst SLA gap: {max_gap:+.1f}% over target")
    else:
        print("\nNo failing configurations found.")


def main():
    parser = argparse.ArgumentParser(
        description="Hardware sizing tool for LLM inference on vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Size for a 7B model with 100 concurrent chat users
  python -m varcas.config_iq.cli.main --model meta-llama/Llama-2-7b-hf --workload chat --users 100

  # With 50% headroom for growth
  python -m varcas.config_iq.cli.main --model mistralai/Mistral-7B-v0.1 --workload rag --users 50 --headroom 50

  # Show top 3 recommendations
  python -m varcas.config_iq.cli.main --model Qwen/Qwen2.5-32B --workload code --users 200 --top 3
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="HuggingFace model ID (e.g., meta-llama/Llama-2-7b-hf)"
    )
    parser.add_argument(
        "--workload", "-w",
        choices=["chat", "rag", "code"],
        default="chat",
        help="Workload type (default: chat)"
    )
    parser.add_argument(
        "--users", "-u",
        type=int,
        default=None,
        help="Number of concurrent users to support"
    )
    parser.add_argument(
        "--headroom",
        type=int,
        choices=[0, 25, 50, 100],
        default=25,
        help="Headroom percentage for growth/spikes (default: 25)"
    )
    parser.add_argument(
        "--quantization", "-q",
        choices=["fp16", "bf16", "int8", "fp8", "int4"],
        default="fp16",
        help="Quantization type (default: fp16)"
    )
    parser.add_argument(
        "--pricing",
        choices=["ondemand", "spot"],
        default="ondemand",
        help="Pricing model (default: ondemand)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Show top N recommendations (default: 3)"
    )
    parser.add_argument(
        "--detail",
        type=int,
        default=1,
        help="Show detailed analysis for recommendation #N (default: 1)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--zone",
        default=None,
        help="GCP zone (e.g., us-central1-a). Filters instances by zone availability."
    )
    parser.add_argument(
        "--region",
        default=None,
        help="GCP region (e.g., us-central1). Filters instances by region availability."
    )
    parser.add_argument(
        "--list-zones",
        action="store_true",
        help="List all available zones and exit"
    )
    
    # Hardware configuration options (for DIY/custom configs)
    hw_group = parser.add_argument_group("Hardware Configuration (optional)")
    hw_group.add_argument(
        "--vcpus",
        type=int,
        default=None,
        help="Number of vCPUs (e.g., 4)"
    )
    hw_group.add_argument(
        "--memory-gb",
        type=int,
        default=None,
        help="Host memory in GB (e.g., 15)"
    )
    hw_group.add_argument(
        "--gpu-type",
        default=None,
        help="GPU type (e.g., nvidia-t4, nvidia-l4, nvidia-a100-40gb)"
    )
    hw_group.add_argument(
        "--gpu-count",
        type=int,
        default=None,
        help="Number of GPUs (e.g., 1, 2, 4, 8)"
    )
    hw_group.add_argument(
        "--gpu-vram-gb",
        type=int,
        default=None,
        help="VRAM per GPU in GB (e.g., 16 for T4, 24 for L4). If not specified, will be looked up from catalog."
    )
    hw_group.add_argument(
        "--config-name",
        default="custom",
        help="Name for this hardware configuration (default: custom)"
    )
    hw_group.add_argument(
        "--configs-file",
        default=None,
        help="JSON file with multiple hardware configurations to evaluate"
    )
    
    args = parser.parse_args()
    
    # Initialize components
    hardware = HardwareCatalog()
    
    # Validate required args (unless --list-zones)
    if not args.list_zones:
        if not args.model:
            print("Error: --model is required (unless using --list-zones)")
            return
        if not args.users:
            print("Error: --users is required (unless using --list-zones)")
            return
    
    # Handle --list-zones
    if args.list_zones:
        print("Available GCP Zones:")
        print("=" * 60)
        zones = hardware.get_available_zones()
        regions = hardware.get_available_regions()
        
        print("\nRegions:")
        for r in regions:
            print(f"  {r}")
        
        print("\nZones:")
        for z in zones:
            print(f"  {z}")
        
        print("\nInstance availability by zone/region:")
        for zone in zones[:5]:  # Show first 5 zones
            instances = hardware.get_instances_by_zone(zone)
            print(f"\n  {zone}: {len(instances)} instance types")
            for inst in instances[:3]:  # Show first 3
                print(f"    - {inst.name}")
        return
    
    # Validate zone/region args
    if args.zone and args.region:
        print("Error: Cannot specify both --zone and --region. Use one or the other.")
        return
    
    workloads = WorkloadPatterns()
    roofline = RooflineAnalyzer(hardware)
    calculator = SLACalculator(hardware, workloads, roofline)
    
    # Parse hardware configurations (if specified)
    hardware_configs = parse_hardware_configs(args, hardware)
    use_custom_configs = len(hardware_configs) > 0
    
    # Filter catalog by zone/region if specified (only for catalog mode)
    if not use_custom_configs:
        if args.zone:
            filtered_catalog = hardware.get_instances_by_zone(args.zone)
            location_str = f"Zone: {args.zone}"
        elif args.region:
            filtered_catalog = hardware.get_instances_by_region(args.region)
            location_str = f"Region: {args.region}"
        else:
            filtered_catalog = hardware.get_all_instances()
            location_str = "All zones"
    else:
        filtered_catalog = []
        location_str = "Custom hardware configuration"
    
    # Create sizing input
    sizing_input = SizingInput(
        model_id=args.model,
        workload_type=WorkloadType(args.workload),
        concurrent_users=args.users,
        headroom_percent=args.headroom,
        vm_catalog=filtered_catalog,
        pricing_model=PricingModel(args.pricing),
        quantization=args.quantization
    )
    
    # Print input summary
    model = get_model_profile(args.model)
    print("=" * 80)
    print("HARDWARE SIZING ANALYSIS")
    print("=" * 80)
    print()
    print(f"Model: {args.model}")
    print(f"  Parameters: {model.parameters_b}B")
    print(f"  Model size (fp16): {model.model_size_gb:.1f} GB")
    print(f"  Model size ({args.quantization}): {model.get_model_size_for_quantization(args.quantization):.1f} GB")
    print()
    print(f"Workload: {args.workload}")
    workload = workloads.get_workload(WorkloadType(args.workload))
    print(f"  Description: {workload.description}")
    print(f"  Expected input tokens: {workload.input_tokens.mean:.0f}")
    print(f"  Expected output tokens: {workload.output_tokens.mean:.0f}")
    if args.workload == "rag":
        print(f"  Context window (assumed): {workload.context_tokens.mean:.0f} tokens")
    print()
    print(f"Concurrent users: {args.users}")
    print(f"Headroom: {args.headroom}%")
    print(f"Pricing: {args.pricing}")
    print(f"Location: {location_str}")
    print(f"Quantization: {args.quantization}")
    
    # Generate recommendations
    print("\nAnalyzing hardware configurations...")
    
    if use_custom_configs:
        # Evaluate each custom hardware configuration
        recommendations = []
        for hw_config in hardware_configs:
            try:
                rec = calculator.evaluate_hardware_config(sizing_input, hw_config)
                recommendations.append(rec)
            except Exception as e:
                print(f"Warning: Could not evaluate config '{hw_config.name}': {e}")
    else:
        # Use catalog mode
        recommendations = calculator.generate_all_recommendations(sizing_input)
    
    if not recommendations:
        print("\nNo compatible hardware configurations found!")
        print("The model may be too large for available instances.")
        sys.exit(1)
    
    print(f"Found {len(recommendations)} compatible configurations")
    
    # Print all configurations in two parts (passing/failing)
    print_all_configurations(recommendations, args.model, args.quantization)
    
    # Re-sort recommendations by cost per request for detailed analysis
    # (maintaining original behavior for detailed view)
    recommendations.sort(key=lambda r: r.cost_per_request_usd)
    
    # Print detailed analysis for selected recommendation
    detail_idx = min(args.detail - 1, len(recommendations) - 1)
    selected = recommendations[detail_idx]
    
    # Calculate max batch size for selected recommendation
    selected_max_batch = calculate_max_batch_size_for_config(
        selected, args.model, args.quantization
    )
    
    print_sla_table(selected, sizing_input.workload_type)
    print_throughput_analysis(selected)
    print_cost_analysis(selected)
    
    # Print vLLM command for the selected recommendation
    print_vllm_command(selected, args.model, args.quantization, indent="", max_batch_size=selected_max_batch)
    
    # Check memory situation and warn if tight (per-GPU with tensor parallelism)
    model_size_gb = model.get_model_size_for_quantization(args.quantization)
    gpu_vram = selected.vm_instance.max_vram_per_gpu_gb
    tp_size = selected.tensor_parallel_size
    
    # Model size per GPU with TP
    model_per_gpu = model_size_gb / tp_size if tp_size > 0 else model_size_gb
    
    # Calculate actual overhead (same as compatibility check)
    cuda_overhead_gb = 1.5
    activation_buffer_gb = model_per_gpu * 0.10
    safety_margin_gb = 0.5
    fixed_overhead_gb = cuda_overhead_gb + activation_buffer_gb + safety_margin_gb
    
    # Memory headroom per GPU after accounting for model + overhead
    memory_headroom_gb = gpu_vram - model_per_gpu - fixed_overhead_gb
    
    # Thresholds: < 4GB is very tight, < 8GB is tight
    is_very_tight = memory_headroom_gb < 4.0
    is_tight_fit = memory_headroom_gb < 8.0
    
    if is_very_tight or is_tight_fit:
        raw_headroom = gpu_vram - model_per_gpu
        print("\n" + "=" * 80)
        print("⚠ MEMORY WARNING")
        print("=" * 80)
        print()
        if tp_size > 1:
            print(f"Using Tensor Parallelism: {tp_size} GPUs")
            print(f"Model size per GPU: {model_per_gpu:.1f} GB (total {model_size_gb:.1f} GB / {tp_size})")
        if is_very_tight:
            print(f"CRITICAL: Model size ({model_per_gpu:.1f} GB) is very close to GPU VRAM ({gpu_vram:.0f} GB)")
            print(f"Raw headroom per GPU: {raw_headroom:.1f} GB")
            print(f"Available for KV cache (after overhead): only {memory_headroom_gb:.1f} GB")
            print()
            print("The generated commands use REDUCED memory settings to avoid OOM:")
            print("  • --gpu-memory-utilization 0.75 (very conservative)")
            print("  • --max-model-len 2048 (reduced context window)")
            print()
        else:
            print(f"WARNING: Limited memory headroom for KV cache ({memory_headroom_gb:.1f} GB per GPU)")
            print(f"Model: {model_per_gpu:.1f} GB per GPU on GPU with {gpu_vram:.0f} GB VRAM")
            print(f"Raw headroom: {raw_headroom:.1f} GB, Required overhead: {fixed_overhead_gb:.1f} GB")
            print()
            print("The generated commands use conservative memory settings:")
            print("  • --gpu-memory-utilization 0.80")
        print()
        print("RECOMMENDATIONS to avoid OOM errors:")
        print("  1. Use int8 quantization (50% memory savings):")
        print(f"     --quantization int8 --kv-cache-dtype int8")
        print("  2. Further reduce --max-model-len if you don't need 2K-4K context")
        print("  3. Reduce --max-num-seqs to limit concurrent sequences")
        print("  4. Consider using A100 (40GB/80GB) for more headroom")
        print()
        print("If you still get OOM errors, try:")
        print("  vllm serve <model> --quantization int8 --kv-cache-dtype int8 \\")
        print("    --gpu-memory-utilization 0.75 --max-model-len 2048")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Recommended configuration: {selected.vm_instance.name}")
    print(f"  • {selected.vm_instance.gpu_type} x {selected.vm_instance.gpu_count} GPUs")
    print(f"  • {selected.num_instances_needed} instance(s) needed")
    print(f"  • Total cost: {format_cost(selected.total_hourly_cost_usd)}/hr")
    print(f"  • Cost per request: {format_cost(selected.cost_per_request_usd)}")
    print()
    
    if selected.meets_sla:
        print("✓ This configuration MEETS all SLA targets")
    else:
        print("⚠ This configuration does NOT meet all SLA targets")
        print("  Gaps:")
        for metric, gap in selected.sla_gaps.items():
            if gap > 0:
                print(f"    - {metric}: {gap:+.1f}% over target")
    
    # Print analysis assumptions
    print("\n" + "=" * 80)
    print("ANALYSIS ASSUMPTIONS & VLLM CONFIGURATION")
    print("=" * 80)
    print()
    print("SLA Pass/Fail is based on the SAFE configuration (85% VRAM utilization)")
    print("HIGH PERFORMANCE mode (93% VRAM) offers better throughput but higher OOM risk")
    print()
    print("  SAFE Mode Configuration:")
    print("    • --gpu-memory-utilization 0.80-0.85 (conservative, reliable)")
    print("    • --max-num-seqs <half of max> (conservative batching)")
    print("    • Recommended for production workloads")
    print()
    print("  HIGH PERFORMANCE Mode Configuration:")
    print("    • --gpu-memory-utilization 0.85-0.93 (aggressive)")
    print("    • --max-num-seqs <maximum> (max batching)")
    print("    • --max-num-batched-tokens 4096 (higher throughput)")
    print("    • Higher throughput but increased OOM risk under burst load")
    print()
    print("  Performance Optimizations (both modes):")
    print("    • --enable-prefix-caching (improves TTFT for repeated prefixes)")
    print("    • --enable-chunked-prefill (better throughput)")
    print("    • --max-model-len 4096 (context window)")
    print()
    print("  Efficiency Factors Applied:")
    print("    • Attention compute: 65% of theoretical FLOPs")
    print("    • FFN compute: 75% of theoretical FLOPs")
    print("    • Memory bandwidth: 75% of peak")
    print("    • KV cache access: 70% efficiency")
    print("    • Tensor parallel overhead: 5-15% depending on TP size")
    print()
    print("  Scheduling Assumptions:")
    print("    • Continuous batching with vLLM's default scheduler")
    print("    • Preemption enabled for long sequences")
    print("    • 8ms prefill overhead, 3ms decode overhead per token")
    
    # Save to file if requested
    if args.output:
        results = {
            "input": {
                "model": args.model,
                "workload": args.workload,
                "concurrent_users": args.users,
                "headroom_percent": args.headroom,
                "quantization": args.quantization,
                "pricing": args.pricing
            },
            "analysis_assumptions": {
                "vllm_version": "0.5.0+",
                "safe_mode": {
                    "gpu_memory_utilization": 0.85,
                    "batch_size_factor": 0.5,
                    "description": "Conservative settings, SLA pass/fail based on this"
                },
                "performance_mode": {
                    "gpu_memory_utilization": 0.93,
                    "batch_size_factor": 1.0,
                    "description": "Aggressive settings, higher throughput but OOM risk"
                },
                "max_model_len": 4096,
                "optimizations": ["prefix_caching", "chunked_prefill"],
                "efficiency_factors": {
                    "attention_compute": 0.65,
                    "ffn_compute": 0.75,
                    "memory_bw": 0.75,
                    "kv_cache_access": 0.70,
                    "prefill_overhead_ms": 8,
                    "decode_overhead_ms": 3
                }
            },
            "recommendations": [
                (lambda rec, max_batch: {
                    "instance": rec.vm_instance.name,
                    "gpu_type": rec.vm_instance.gpu_type,
                    "gpu_count": rec.vm_instance.gpu_count,
                    "tensor_parallel": rec.tensor_parallel_size,
                    "instances_needed": rec.num_instances_needed,
                    "meets_sla": rec.meets_sla,
                    "hourly_cost_usd": rec.total_hourly_cost_usd,
                    "cost_per_request_usd": rec.cost_per_request_usd,
                    "cost_per_1k_tokens_usd": rec.cost_per_1k_tokens_usd,
                    "max_batch_size": max_batch,
                    "vllm_commands": {
                        "safe": generate_vllm_command(rec, args.model, args.quantization, mode="safe", max_batch_size=max_batch),
                        "performance": generate_vllm_command(rec, args.model, args.quantization, mode="performance", max_batch_size=max_batch)
                    },
                    "performance": {
                        "ttft_p50_ms": rec.performance.ttft_p50_ms,
                        "ttft_p95_ms": rec.performance.ttft_p95_ms,
                        "ttft_p99_ms": rec.performance.ttft_p99_ms,
                        "tpot_p50_ms": rec.performance.tpot_p50_ms,
                        "throughput_tok_s": rec.performance.throughput_tok_s,
                        "throughput_req_s": rec.performance.throughput_req_s,
                        "gpu_utilization": rec.performance.gpu_utilization
                    }
                })(rec, calculate_max_batch_size_for_config(rec, args.model, args.quantization))
                for rec in recommendations[:args.top]
            ]
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
