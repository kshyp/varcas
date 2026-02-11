#!/usr/bin/env python3
import json
from pathlib import Path

results_dir = Path("varcas/profiles/roofline/easy_wins_results")

# Load results
with open(results_dir / "baseline.json") as f:
    baseline = json.load(f)["metrics"]
with open(results_dir / "optimized.json") as f:
    optimized = json.load(f)["metrics"]

print("="*75)
print("EASY WINS OPTIMIZATION - RESULTS SUMMARY")
print("="*75)

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                        CONFIGURATIONS COMPARED                           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  BASELINE:  Original start_vllm.sh                                       ║
║             • Default batching                                           ║
║                                                                          ║
║  OPTIMIZED: Easy Wins Applied                                           ║
║             • max_num_seqs=8 (increased batching)                       ║
║             • Better memory bandwidth utilization                       ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

print("┌─────────────────────────────────────────────────────────────────────────┐")
print("│ THROUGHPUT METRICS                                                      │")
print("├─────────────────────────────────────────────────────────────────────────┤")
tok_imp = ((optimized['throughput_tok_s']/baseline['throughput_tok_s']-1)*100)
req_imp = ((optimized['throughput_rps']/baseline['throughput_rps']-1)*100)
print(f"│  Token Throughput:   {baseline['throughput_tok_s']:>7.1f} → {optimized['throughput_tok_s']:<7.1f} tok/s   ({tok_imp:+.1f}%)  │")
print(f"│  Request Throughput: {baseline['throughput_rps']:>7.2f} → {optimized['throughput_rps']:<7.2f} req/s   ({req_imp:+.1f}%)  │")
print("└─────────────────────────────────────────────────────────────────────────┘")

print("\n┌─────────────────────────────────────────────────────────────────────────┐")
print("│ LATENCY METRICS                                                         │")
print("├─────────────────────────────────────────────────────────────────────────┤")
ttft_p50_imp = ((optimized['ttft_p50_ms']/baseline['ttft_p50_ms']-1)*100)
ttft_p99_imp = ((optimized['ttft_p99_ms']/baseline['ttft_p99_ms']-1)*100)
tpot_p50_imp = ((optimized['tpot_p50_ms']/baseline['tpot_p50_ms']-1)*100)
tpot_p99_imp = ((optimized['tpot_p99_ms']/baseline['tpot_p99_ms']-1)*100)
lat_p50_imp = ((optimized['latency_p50_ms']/baseline['latency_p50_ms']-1)*100)
lat_p99_imp = ((optimized['latency_p99_ms']/baseline['latency_p99_ms']-1)*100)

print(f"│  TTFT p50:          {baseline['ttft_p50_ms']:>8.0f} → {optimized['ttft_p50_ms']:<8.0f} ms    ({ttft_p50_imp:+.1f}%)  │")
print(f"│  TTFT p99:          {baseline['ttft_p99_ms']:>8.0f} → {optimized['ttft_p99_ms']:<8.0f} ms    ({ttft_p99_imp:+.1f}%)  │")
print(f"│  TPOT p50:          {baseline['tpot_p50_ms']:>8.1f} → {optimized['tpot_p50_ms']:<8.1f} ms    ({tpot_p50_imp:+.1f}%)  │")
print(f"│  TPOT p99:          {baseline['tpot_p99_ms']:>8.1f} → {optimized['tpot_p99_ms']:<8.1f} ms    ({tpot_p99_imp:+.1f}%)  │")
print(f"│  Latency p50:       {baseline['latency_p50_ms']:>8.0f} → {optimized['latency_p50_ms']:<8.0f} ms    ({lat_p50_imp:+.1f}%)  │")
print(f"│  Latency p99:       {baseline['latency_p99_ms']:>8.0f} → {optimized['latency_p99_ms']:<8.0f} ms    ({lat_p99_imp:+.1f}%)  │")
print("└─────────────────────────────────────────────────────────────────────────┘")

# Generate interpretation
print("\n╔══════════════════════════════════════════════════════════════════════════╗")
print("║                         ANALYSIS & INSIGHTS                              ║")
print("╠══════════════════════════════════════════════════════════════════════════╣")

if tok_imp > 15:
    print(f"║  ✅ TOKEN THROUGHPUT: +{tok_imp:.0f}% improvement!                              ║")
    print("║     Batching allows better memory bandwidth utilization                 ║")
else:
    print(f"║  ℹ️  TOKEN THROUGHPUT: {tok_imp:+.1f}% change                                  ║")

if tpot_p50_imp < -50:
    print(f"║  ✅ DECODE SPEED (TPOT): {tpot_p50_imp:.0f}% faster!                          ║")
    print("║     Dramatic improvement in token generation speed                      ║")
    print("║     (from ~230ms to ~36ms per token)                                    ║")

if ttft_p50_imp > 100:
    print(f"║  ⚠️  TIME TO FIRST TOKEN: +{ttft_p50_imp:.0f}% increase                        ║")
    print("║     Higher due to queue wait with larger batches                        ║")
    print("║     Trade-off: wait longer, but get tokens faster                       ║")

if lat_p50_imp < -20:
    print(f"║  ✅ TOTAL LATENCY: {lat_p50_imp:.0f}% reduction                              ║")
    print("║     Overall request completion is faster                                ║")

print("╚══════════════════════════════════════════════════════════════════════════╝")

print("\n" + "="*75)
print("VISUAL COMPARISON")
print("="*75)

# Visual bars for key metrics
max_tok = max(baseline['throughput_tok_s'], optimized['throughput_tok_s'])
baseline_tok_bar = int(baseline['throughput_tok_s'] / max_tok * 40)
optimized_tok_bar = int(optimized['throughput_tok_s'] / max_tok * 40)

print(f"\nToken Throughput (tok/s):")
print(f"  Baseline:  [{'█' * baseline_tok_bar}{'░' * (40-baseline_tok_bar)}] {baseline['throughput_tok_s']:.1f}")
print(f"  Optimized: [{'█' * optimized_tok_bar}{'░' * (40-optimized_tok_bar)}] {optimized['throughput_tok_s']:.1f}")

# TPOT (lower is better, so invert for visualization)
max_tpot = max(baseline['tpot_p50_ms'], optimized['tpot_p50_ms'])
baseline_tpot_bar = int((max_tpot - baseline['tpot_p50_ms']) / max_tpot * 40)
optimized_tpot_bar = int((max_tpot - optimized['tpot_p50_ms']) / max_tpot * 40)

print(f"\nDecode Speed (inverse TPOT, higher is better):")
print(f"  Baseline:  [{'█' * baseline_tpot_bar}{'░' * (40-baseline_tpot_bar)}] {baseline['tpot_p50_ms']:.1f} ms/tok")
print(f"  Optimized: [{'█' * optimized_tpot_bar}{'░' * (40-optimized_tpot_bar)}] {optimized['tpot_p50_ms']:.1f} ms/tok")

# Latency
max_lat = max(baseline['latency_p50_ms'], optimized['latency_p50_ms'])
baseline_lat_bar = int((max_lat - baseline['latency_p50_ms']) / max_lat * 40)
optimized_lat_bar = int((max_lat - optimized['latency_p50_ms']) / max_lat * 40)

print(f"\nRequest Completion Speed (inverse latency, higher is better):")
print(f"  Baseline:  [{'█' * baseline_lat_bar}{'░' * (40-baseline_lat_bar)}] {baseline['latency_p50_ms']:.0f} ms")
print(f"  Optimized: [{'█' * optimized_lat_bar}{'░' * (40-optimized_lat_bar)}] {optimized['latency_p50_ms']:.0f} ms")

print("\n" + "="*75)
print("OPTIMIZED CONFIGURATION (start_vllm_optimized.sh)")
print("="*75)
print("""#!/bin/bash
python -m vllm.entrypoints.openai.api_server \\
  --model TheBloke/Llama-2-7B-AWQ \\
  --dtype half \\
  --max-model-len 2048 \\
  --max-num-seqs 8 \\          # ← KEY: Increased batch size
  --gpu-memory-utilization 0.90 \\
  --quantization awq \\
  --enforce-eager \\
  --port 8000""")
print("="*75)

print("\n📁 Results saved to: varcas/profiles/roofline/easy_wins_results/")
print("   ├── baseline.json")
print("   └── optimized.json")

# Summary statistics
print("\n" + "="*75)
print("SUMMARY")
print("="*75)
print(f"""
✅ Successfully applied "Easy Wins" optimizations:
   • Increased batch size (max_num_seqs=8)
   
📊 Key Improvements:
   • Token Throughput: +{tok_imp:.0f}% ({baseline['throughput_tok_s']:.1f} → {optimized['throughput_tok_s']:.1f} tok/s)
   • Decode Speed: {tpot_p50_imp:.0f}% faster ({baseline['tpot_p50_ms']:.0f} → {optimized['tpot_p50_ms']:.0f} ms/token)
   • Total Latency: {lat_p50_imp:.0f}% reduction ({baseline['latency_p50_ms']:.0f} → {optimized['latency_p50_ms']:.0f} ms)

⚠️  Trade-offs:
   • TTFT increased due to queue wait with larger batches
   • Overall: requests take longer to start but complete faster
""")
