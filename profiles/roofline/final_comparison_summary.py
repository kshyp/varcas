#!/usr/bin/env python3
"""Generate final optimization summary."""

import json
from pathlib import Path

print("="*70)
print("BATCH SIZE OPTIMIZATION - FINAL SUMMARY")
print("="*70)

# Load all results
base = Path("varcas/profiles/roofline/batch_optimization")

# Normal load
with open(base / "results_b1.json") as f:
    normal_b1 = json.load(f)["metrics"]
with open(base / "results_b8.json") as f:
    normal_b8 = json.load(f)["metrics"]

# High load
with open(base / "highload_results_b1.json") as f:
    high_b1 = json.load(f)["metrics"]
with open(base / "highload_results_b8.json") as f:
    high_b8 = json.load(f)["metrics"]

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    OPTIMIZATION RESULTS SUMMARY                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Model:  TheBloke/Llama-2-7B-AWQ (4-bit)                             ║
║  GPU:    Tesla T4                                                    ║
║  Optimization:  max_num_seqs 1 → 8                                   ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("┌─────────────────────────────────────────────────────────────────────┐")
print("│ NORMAL LOAD (chat_medium - 20 RPS)                                  │")
print("├─────────────────────────────────────────────────────────────────────┤")
print(f"│  Token Throughput:  {normal_b1['throughput_tok_s']:>6.1f} → {normal_b8['throughput_tok_s']:<6.1f} tok/s  (+{((normal_b8['throughput_tok_s']/normal_b1['throughput_tok_s']-1)*100):.0f}%) │")
print(f"│  Request Throughput: {normal_b1['throughput_rps']:>6.2f} → {normal_b8['throughput_rps']:<6.2f} req/s  (rate limited)  │")
print(f"│  TTFT p50:          {normal_b1['ttft_p50_ms']:>6.0f} → {normal_b8['ttft_p50_ms']:<6.0f} ms      (+{((normal_b8['ttft_p50_ms']/normal_b1['ttft_p50_ms']-1)*100):.0f}%)   │")
print(f"│  TPOT p50:          {normal_b1['tpot_p50_ms']:>6.1f} → {normal_b8['tpot_p50_ms']:<6.1f} ms      (+{((normal_b8['tpot_p50_ms']/normal_b1['tpot_p50_ms']-1)*100):.0f}%)   │")
print("└─────────────────────────────────────────────────────────────────────┘")

print("\n┌─────────────────────────────────────────────────────────────────────┐")
print("│ HIGH LOAD (chat_high - 50 RPS)                                      │")
print("├─────────────────────────────────────────────────────────────────────┤")
print(f"│  Token Throughput:  {high_b1['throughput_tok_s']:>6.1f} → {high_b8['throughput_tok_s']:<6.1f} tok/s  (+{((high_b8['throughput_tok_s']/high_b1['throughput_tok_s']-1)*100):.0f}%) │")
print(f"│  Request Throughput: {high_b1['throughput_rps']:>6.2f} → {high_b8['throughput_rps']:<6.2f} req/s  (+{((high_b8['throughput_rps']/high_b1['throughput_rps']-1)*100):.1f}%)   │")
print(f"│  Latency p50:       {high_b1['latency_p50_ms']:>6.0f} → {high_b8['latency_p50_ms']:<6.0f} ms      (+{((high_b8['latency_p50_ms']/high_b1['latency_p50_ms']-1)*100):.0f}%)   │")
print("└─────────────────────────────────────────────────────────────────────┘")

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                         KEY INSIGHTS                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ✅ TOKEN THROUGHPUT: +350-390% improvement                          ║
║     • 22 → 99 tok/s (normal load)                                    ║
║     • 37 → 179 tok/s (high load)                                     ║
║                                                                      ║
║  ✅ ROOFLINE POSITION: Moving toward compute bound                   ║
║     • Batch=1: AI ≈ 3 FLOP/Byte (memory-bound, 1.5% util)            ║
║     • Batch=8: AI ≈ 15-24 FLOP/Byte (better memory BW util)          ║
║                                                                      ║
║  ⚠️  LATENCY TRADE-OFF: +30-70% increase                             ║
║     • TTFT increases due to queue wait                               ║
║     • TPOT increases due to batch processing                         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("RECOMMENDED CONFIGURATION:")
print("-" * 70)
print("""python -m vllm.entrypoints.openai.api_server \\
  --model TheBloke/Llama-2-7B-AWQ \\
  --dtype half \\
  --max-model-len 2048 \\
  --max-num-seqs 8 \\        # ← ADD THIS LINE
  --gpu-memory-utilization 0.90 \\
  --quantization awq \\
  --enforce-eager \\
  --port 8000""")
print("-" * 70)

print("\nFILES GENERATED:")
print(f"  📁 {base}")
print("     ├── results_b1.json, results_b8.json")
print("     ├── highload_results_b1.json, highload_results_b8.json")
print("     ├── analysis_detailed.json")
print("     └── BATCH_OPTIMIZATION_RESULTS.md")

print("\n" + "="*70)
