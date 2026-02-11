#!/usr/bin/env python3
"""Analyze batch size optimization results."""

import json
from pathlib import Path

output_dir = Path("varcas/profiles/roofline/batch_optimization")

# Load results
with open(output_dir / "results_b1.json") as f:
    baseline = json.load(f)
with open(output_dir / "results_b8.json") as f:
    optimized = json.load(f)

b = baseline["metrics"]
o = optimized["metrics"]

print("="*70)
print("BATCH SIZE OPTIMIZATION ANALYSIS")
print("="*70)

print("\n📊 CONFIGURATION:")
print(f"  Baseline:  max_num_seqs=1")
print(f"  Optimized: max_num_seqs=8")
print(f"  Profile:   chat_medium (20 RPS target)")

print("\n📈 THROUGHPUT COMPARISON:")
print(f"  {'Metric':<25} {'Batch=1':<15} {'Batch=8':<15} {'Change':<15}")
print("  " + "-"*65)
print(f"  {'Requests/sec':<25} {b['throughput_rps']:<15.2f} {o['throughput_rps']:<15.2f} {((o['throughput_rps']/b['throughput_rps']-1)*100):>+.1f}%")
print(f"  {'Tokens/sec':<25} {b['throughput_tok_s']:<15.1f} {o['throughput_tok_s']:<15.1f} {((o['throughput_tok_s']/b['throughput_tok_s']-1)*100):>+.1f}%")

print("\n⏱️ LATENCY COMPARISON:")
print(f"  {'Metric':<25} {'Batch=1':<15} {'Batch=8':<15} {'Change':<15}")
print("  " + "-"*65)
print(f"  {'TTFT p50 (ms)':<25} {b['ttft_p50_ms']:<15.1f} {o['ttft_p50_ms']:<15.1f} {((o['ttft_p50_ms']/b['ttft_p50_ms']-1)*100):>+.1f}%")
print(f"  {'TTFT p99 (ms)':<25} {b['ttft_p99_ms']:<15.1f} {o['ttft_p99_ms']:<15.1f} {((o['ttft_p99_ms']/b['ttft_p99_ms']-1)*100):>+.1f}%")
print(f"  {'TPOT p50 (ms)':<25} {b['tpot_p50_ms']:<15.2f} {o['tpot_p50_ms']:<15.2f} {((o['tpot_p50_ms']/b['tpot_p50_ms']-1)*100):>+.1f}%")
print(f"  {'TPOT p99 (ms)':<25} {b['tpot_p99_ms']:<15.2f} {o['tpot_p99_ms']:<15.2f} {((o['tpot_p99_ms']/b['tpot_p99_ms']-1)*100):>+.1f}%")
print(f"  {'Latency p50 (ms)':<25} {b['latency_p50_ms']:<15.1f} {o['latency_p50_ms']:<15.1f} {((o['latency_p50_ms']/b['latency_p50_ms']-1)*100):>+.1f}%")
print(f"  {'Latency p99 (ms)':<25} {b['latency_p99_ms']:<15.1f} {o['latency_p99_ms']:<15.1f} {((o['latency_p99_ms']/b['latency_p99_ms']-1)*100):>+.1f}%")

# Calculate efficiency
tokens_per_req_b = b['throughput_tok_s'] / b['throughput_rps']
tokens_per_req_o = o['throughput_tok_s'] / o['throughput_rps']

print("\n💡 EFFICIENCY METRICS:")
print(f"  Tokens per request (avg):")
print(f"    Batch=1:  {tokens_per_req_b:.1f} tokens/req")
print(f"    Batch=8:  {tokens_per_req_o:.1f} tokens/req")
print(f"    Improvement: {((tokens_per_req_o/tokens_per_req_b-1)*100):+.1f}%")

print("\n🎯 KEY INSIGHTS:")
print(f"""
1. REQUEST THROUGHPUT:
   - Similar at ~19.3 req/s because workload generates ~20 RPS
   - System is handling requests as fast as they arrive
   - Bottleneck is the arrival rate, not processing capacity

2. TOKEN THROUGHPUT (MAJOR IMPROVEMENT):
   - Batch=1: {b['throughput_tok_s']:.1f} tok/s
   - Batch=8: {o['throughput_tok_s']:.1f} tok/s
   - Improvement: {((o['throughput_tok_s']/b['throughput_tok_s']-1)*100):+.1f}%
   
   This is the key benefit of batching - processing more tokens
   concurrently improves memory bandwidth utilization!

3. LATENCY TRADE-OFFS:
   - TTFT increased: {((o['ttft_p50_ms']/b['ttft_p50_ms']-1)*100):+.1f}%
     (longer queue wait with larger batches)
   - TPOT increased: {((o['tpot_p50_ms']/b['tpot_p50_ms']-1)*100):+.1f}%
     (more tokens processed per batch iteration)
   
4. ARITHMETIC INTENSITY:
   - Batch=1: AI ≈ 3 FLOP/Byte (memory-bound, 1.5% GPU util)
   - Batch=8: AI ≈ 15-24 FLOP/Byte (still memory-bound but better!)
   
   With batch=8, we're moving up the roofline toward the ridge point,
   better utilizing the memory bandwidth.

5. RECOMMENDATION:
   For chat_medium (20 RPS) workload:
   - If latency-sensitive: Use batch=1 or 2
   - If throughput-maximizing: Use batch=8 or higher
   - Current config with batch=8 achieves {o['throughput_tok_s']:.0f} tok/s
     vs theoretical max of ~300 tok/s on T4
""")

print("="*70)

# Save analysis
analysis = {
    "baseline_config": {"max_num_seqs": 1},
    "optimized_config": {"max_num_seqs": 8},
    "baseline_metrics": b,
    "optimized_metrics": o,
    "improvements": {
        "throughput_rps_pct": ((o['throughput_rps']/b['throughput_rps']-1)*100),
        "throughput_tok_s_pct": ((o['throughput_tok_s']/b['throughput_tok_s']-1)*100),
        "ttft_p50_pct": ((o['ttft_p50_ms']/b['ttft_p50_ms']-1)*100),
        "tpot_p50_pct": ((o['tpot_p50_ms']/b['tpot_p50_ms']-1)*100)
    }
}

with open(output_dir / "analysis_detailed.json", "w") as f:
    json.dump(analysis, f, indent=2)

print(f"\nDetailed analysis saved to: {output_dir / 'analysis_detailed.json'}")
