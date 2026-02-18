"""Tests for the hardware sizing tool."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from config_iq.core import (
    HardwareCatalog,
    WorkloadPatterns,
    get_model_profile,
    ModelProfile,
    RooflineAnalyzer,
    MG1QueuingModel,
    SLACalculator,
    WorkloadType,
    PricingModel,
    SizingInput,
)


def test_hardware_catalog():
    """Test hardware catalog loading."""
    print("Testing Hardware Catalog...")
    catalog = HardwareCatalog()
    
    instances = catalog.get_all_instances()
    print(f"  Loaded {len(instances)} VM instances")
    
    a100_instances = catalog.get_instances_by_gpu_type("nvidia-a100-40gb")
    print(f"  Found {len(a100_instances)} A100 instances")
    
    # Test interconnect info
    interconnect = catalog.get_interconnect_info("nvidia-a100-40gb")
    print(f"  A100 NVLink BW: {interconnect.get('nvlink_bw_gbps', 'N/A')} GB/s")
    
    print("  ✓ Hardware catalog test passed\n")


def test_model_profiler():
    """Test model profiler."""
    print("Testing Model Profiler...")
    
    # Test known model
    model = get_model_profile("meta-llama/Llama-2-7b-hf")
    print(f"  Model: {model.model_id}")
    print(f"  Parameters: {model.parameters_b}B")
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Layers: {model.num_layers}")
    print(f"  Model size (fp16): {model.model_size_gb:.1f} GB")
    print(f"  KV cache per token: {model.kv_cache_per_token_kb:.2f} KB")
    
    # Test memory calculation
    mem = model.calculate_memory_requirement(batch_size=1, seq_length=4096)
    print(f"  Memory required (1x4096): {mem['total_gb']:.1f} GB")
    
    print("  ✓ Model profiler test passed\n")


def test_workload_patterns():
    """Test workload patterns."""
    print("Testing Workload Patterns...")
    patterns = WorkloadPatterns()
    
    for wt in [WorkloadType.CHAT, WorkloadType.RAG, WorkloadType.CODE]:
        workload = patterns.get_workload(wt)
        sla = patterns.get_sla_targets(wt)
        print(f"  {wt.value}:")
        print(f"    Input tokens: {workload.input_tokens.mean:.0f}")
        print(f"    Output tokens: {workload.output_tokens.mean:.0f}")
        print(f"    TTFT target: {sla.ttft_p50_ms:.0f}ms")
        print(f"    TPOT target: {sla.tpot_p50_ms:.0f}ms")
    
    # Test headroom config
    headroom = patterns.get_headroom_config(50)
    print(f"  50% headroom: {headroom}")
    
    print("  ✓ Workload patterns test passed\n")


def test_roofline_analyzer():
    """Test roofline analyzer."""
    print("Testing Roofline Analyzer...")
    
    catalog = HardwareCatalog()
    analyzer = RooflineAnalyzer(catalog)
    model = get_model_profile("meta-llama/Llama-2-7b-hf")
    
    # Get a test VM
    vm = catalog.get_instances_by_gpu_type("nvidia-l4")[0]
    print(f"  Testing with VM: {vm.name}")
    
    # Analyze prefill
    prefill = analyzer.analyze_prefill(vm, model, batch_size=1, seq_length=512)
    print(f"  Prefill (1x512):")
    print(f"    Latency: {prefill.prefill_latency_ms:.1f} ms")
    print(f"    Compute bound: {prefill.compute_bound}")
    print(f"    Achievable: {prefill.achievable_tflops:.0f} TFLOPS")
    
    # Analyze decode
    decode = analyzer.analyze_decode(vm, model, batch_size=1, context_length=512)
    print(f"  Decode (batch=1):")
    print(f"    Latency/token: {decode.decode_latency_per_token_ms:.2f} ms")
    print(f"    Tokens/sec: {decode.tokens_per_sec_single:.1f}")
    print(f"    Compute bound: {decode.compute_bound}")
    
    # Analyze full request
    full = analyzer.analyze_full_request(
        vm, model, batch_size=1, 
        input_tokens=512, output_tokens=256
    )
    print(f"  Full request (512 in, 256 out):")
    print(f"    Total latency: {full['total_latency_ms']:.1f} ms")
    print(f"    Throughput: {full['tokens_per_second']:.1f} tok/s")
    
    print("  ✓ Roofline analyzer test passed\n")


def test_queuing_model():
    """Test M/G/1 queuing model."""
    print("Testing M/G/1 Queuing Model...")
    
    model = MG1QueuingModel(service_time_cv=0.5)
    
    # Test with moderate load
    metrics = model.calculate_metrics(
        arrival_rate=10,  # 10 req/s
        service_rate=20   # Can serve 20 req/s
    )
    print(f"  Load: 10 req/s arrival, 20 req/s service")
    print(f"  Utilization: {metrics.utilization:.1%}")
    print(f"  Wait time P50: {metrics.wait_time_p50_ms:.1f} ms")
    print(f"  Wait time P95: {metrics.wait_time_p95_ms:.1f} ms")
    print(f"  Total latency P50: {metrics.total_latency_p50_ms:.1f} ms")
    
    # Test tail factors
    from config_iq.core.queuing_model import calculate_tail_latency_factors
    factors = calculate_tail_latency_factors(100, 0.5)
    print(f"  Tail factors at 50% util:")
    print(f"    P95: {factors['p95_multiplier']:.1f}x")
    print(f"    P99: {factors['p99_multiplier']:.1f}x")
    
    print("  ✓ Queuing model test passed\n")


def test_sla_calculator():
    """Test full SLA calculator."""
    print("Testing SLA Calculator...")
    
    catalog = HardwareCatalog()
    patterns = WorkloadPatterns()
    roofline = RooflineAnalyzer(catalog)
    calculator = SLACalculator(catalog, patterns, roofline)
    
    # Create test input
    sizing_input = SizingInput(
        model_id="meta-llama/Llama-2-7b-hf",
        workload_type=WorkloadType.CHAT,
        concurrent_users=50,
        headroom_percent=25,
        vm_catalog=catalog.get_all_instances(),
        pricing_model=PricingModel.ONDEMAND,
        quantization="fp16"
    )
    
    # Get a compatible VM
    model = get_model_profile("meta-llama/Llama-2-7b-hf")
    compatible = catalog.filter_compatible_instances(
        model.model_size_gb, quantization="fp16"
    )
    print(f"  Found {len(compatible)} compatible instances")
    
    if compatible:
        vm = compatible[0]
        print(f"  Testing with: {vm.name}")
        
        # Generate recommendation
        rec = calculator.generate_recommendation(sizing_input, vm)
        
        print(f"  Performance prediction:")
        print(f"    TTFT P50: {rec.performance.ttft_p50_ms:.0f} ms")
        print(f"    TTFT P95: {rec.performance.ttft_p95_ms:.0f} ms")
        print(f"    TPOT: {rec.performance.tpot_p50_ms:.0f} ms")
        print(f"    Throughput: {rec.performance.throughput_tok_s:.0f} tok/s")
        print(f"    GPU util: {rec.performance.gpu_utilization:.0%}")
        
        print(f"  Cost analysis:")
        print(f"    Hourly: ${rec.hourly_cost_usd:.2f}")
        print(f"    Per request: ${rec.cost_per_request_usd:.4f}")
        print(f"    Per 1K tokens: ${rec.cost_per_1k_tokens_usd:.4f}")
        
        print(f"  SLA compliance: {'PASS' if rec.meets_sla else 'FAIL'}")
        if not rec.meets_sla:
            print(f"    Gaps: {rec.sla_gaps}")
    
    print("  ✓ SLA calculator test passed\n")


def test_end_to_end():
    """Test end-to-end recommendation generation."""
    print("Testing End-to-End Recommendation...")
    
    catalog = HardwareCatalog()
    patterns = WorkloadPatterns()
    roofline = RooflineAnalyzer(catalog)
    calculator = SLACalculator(catalog, patterns, roofline)
    
    sizing_input = SizingInput(
        model_id="mistralai/Mistral-7B-v0.1",
        workload_type=WorkloadType.CHAT,
        concurrent_users=100,
        headroom_percent=50,
        vm_catalog=catalog.get_all_instances(),
        pricing_model=PricingModel.ONDEMAND,
        quantization="fp16"
    )
    
    print(f"  Sizing for: {sizing_input.model_id}")
    print(f"  Workload: {sizing_input.workload_type.value}")
    print(f"  Users: {sizing_input.concurrent_users}")
    print(f"  Headroom: {sizing_input.headroom_percent}%")
    
    recommendations = calculator.generate_all_recommendations(sizing_input)
    print(f"  Generated {len(recommendations)} recommendations")
    
    if recommendations:
        print("\n  Top 3 recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            status = "✓" if rec.meets_sla else "✗"
            print(f"    {i}. {rec.vm_instance.name} - "
                  f"${rec.cost_per_request_usd:.4f}/req "
                  f"{rec.performance.throughput_tok_s:.0f} tok/s "
                  f"{status}")
    
    print("  ✓ End-to-end test passed\n")


if __name__ == "__main__":
    print("="*60)
    print("HARDWARE SIZING TOOL - TEST SUITE")
    print("="*60)
    print()
    
    test_hardware_catalog()
    test_model_profiler()
    test_workload_patterns()
    test_roofline_analyzer()
    test_queuing_model()
    test_sla_calculator()
    test_end_to_end()
    
    print("="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
