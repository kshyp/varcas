#!/usr/bin/env python3
import json
from pathlib import Path

results_dir = Path("varcas/profiles/roofline/medium_term_results")

with open(results_dir / "baseline.json") as f:
    baseline = json.load(f)["metrics"]
with open(results_dir / "advanced.json") as f:
    advanced = json.load(f)["metrics"]

print("="*75)
print("MEDIUM-TERM OPTIMIZATIONS - RESULTS SUMMARY")
print("="*75)

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                     CONFIGURATIONS COMPARED                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  BASELINE (Easy Wins):                                                   ║
║    • max_num_seqs=8                                                      ║
║                                                                          ║
║  ADVANCED (Medium-Term):                                                 ║
║    • max_num_seqs=8                                                      ║
║    • --enable-chunked-prefill                                            ║
║    • Better prefill/decode interleaving                                  ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

print("┌─────────────────────────────────────────────────────────────────────────┐")
print("│ THROUGHPUT METRICS                                                      │")
print("├─────────────────────────────────────────────────────────────────────────┤")
tok_imp = ((advanced['throughput_tok_s']/baseline['throughput_tok_s']-1)*100)
req_imp = ((advanced['throughput_rps']/baseline['throughput_rps']-1)*100)
print(f"│  Token Throughput:   {baseline['throughput_tok_s']:>7.1f} → {advanced['throughput_tok_s']:<7.1f} tok/s   ({tok_imp:+.1f}%)  │")
print(f"│  Request Throughput: {baseline['throughput_rps']:>7.2f} → {advanced['throughput_rps']:<7.2f} req/s   ({req_imp:+.1f}%)  │")
print("└─────────────────────────────────────────────────────────────────────────┘")

print("\n┌─────────────────────────────────────────────────────────────────────────┐")
print("│ LATENCY METRICS                                                         │")
print("├─────────────────────────────────────────────────────────────────────────┤")
ttft_p50_imp = ((advanced['ttft_p50_ms']/baseline['ttft_p50_ms']-1)*100)
tpot_p50_imp = ((advanced['tpot_p50_ms']/baseline['tpot_p50_ms']-1)*100)
lat_p50_imp = ((advanced['latency_p50_ms']/baseline['latency_p50_ms']-1)*100)

print(f"│  TTFT p50:          {baseline['ttft_p50_ms']:>8.0f} → {advanced['ttft_p50_ms']:<8.0f} ms    ({ttft_p50_imp:+.1f}%)  │")
print(f"│  TTFT p99:          {baseline['ttft_p99_ms']:>8.0f} → {advanced['ttft_p99_ms']:<8.0f} ms    ({((advanced['ttft_p99_ms']/baseline['ttft_p99_ms']-1)*100):+.1f}%)  │")
print(f"│  TPOT p50:          {baseline['tpot_p50_ms']:>8.1f} → {advanced['tpot_p50_ms']:<8.1f} ms    ({tpot_p50_imp:+.1f}%)  │")
print(f"│  TPOT p99:          {baseline['tpot_p99_ms']:>8.1f} → {advanced['tpot_p99_ms']:<8.1f} ms    ({((advanced['tpot_p99_ms']/baseline['tpot_p99_ms']-1)*100):+.1f}%)  │")
print(f"│  Latency p50:       {baseline['latency_p50_ms']:>8.0f} → {advanced['latency_p50_ms']:<8.0f} ms    ({lat_p50_imp:+.1f}%)  │")
print(f"│  Latency p99:       {baseline['latency_p99_ms']:>8.0f} → {advanced['latency_p99_ms']:<8.0f} ms    ({((advanced['latency_p99_ms']/baseline['latency_p99_ms']-1)*100):+.1f}%)  │")
print("└─────────────────────────────────────────────────────────────────────────┘")

print("\n╔══════════════════════════════════════════════════════════════════════════╗")
print("║                         ANALYSIS                                         ║")
print("╠══════════════════════════════════════════════════════════════════════════╣")

if abs(tok_imp) < 5:
    print(f"│  ℹ️  TOKEN THROUGHPUT: Minimal change ({tok_imp:+.1f}%)                          │")
    print("│     Chunked prefill doesn't significantly affect this workload          │")
    
if tpot_p50_imp > 5:
    print(f"│  ⚠️  DECODE SPEED: {tpot_p50_imp:+.1f}% slower                                    │")
    print("│     Chunked prefill adds overhead to token generation                   │")
    
if lat_p50_imp > 5:
    print(f"│  ⚠️  TOTAL LATENCY: {lat_p50_imp:+.1f}% increase                                 │")
    print("│     Overall request completion is slower with chunked prefill           │")

print("╚══════════════════════════════════════════════════════════════════════════╝")

print("\n" + "="*75)
print("INTERPRETATION")
print("="*75)
print("""
Chunked prefill is designed to improve interactivity by breaking up
large prefill operations into smaller chunks that can be interleaved
with decode operations.

For this workload (chat_medium with 50 token average input):
• Input sequences are relatively short
• Prefill is already fast (compute-bound)
• Chunking adds overhead without significant benefit
• Results show slight degradation in performance

RECOMMENDATION:
For short-input chat workloads, chunked prefill may not be beneficial.
Consider enabling it only for:
• Long-context workloads (RAG, documents)
• Mixed workloads with varying input lengths
• Scenarios where prefill latency affects TTFT significantly
""")

print("="*75)
print("COMPLETE OPTIMIZATION PROGRESSION")
print("="*75)

# Three-way comparison
print("""
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│     Metric      │   Original   │ Easy Wins    │ Medium-Term  │
│                 │  (batch=1)   │ (batch=8)    │(+chunked)    │
├─────────────────┼──────────────┼──────────────┼──────────────┤""")

# Original from easy_wins_results
with open("varcas/profiles/roofline/easy_wins_results/baseline.json") as f:
    orig = json.load(f)["metrics"]

print(f"│ Token Throughput│  {orig['throughput_tok_s']:>6.1f} tok/s │  {baseline['throughput_tok_s']:>6.1f} tok/s │  {advanced['throughput_tok_s']:>6.1f} tok/s │")
print(f"│ TPOT p50        │   {orig['tpot_p50_ms']:>6.1f} ms  │   {baseline['tpot_p50_ms']:>6.1f} ms  │   {advanced['tpot_p50_ms']:>6.1f} ms  │")
print(f"│ Latency p50     │  {orig['latency_p50_ms']:>6.0f} ms   │  {baseline['latency_p50_ms']:>6.0f} ms   │  {advanced['latency_p50_ms']:>6.0f} ms   │")
print(f"│ TTFT p50        │    {orig['ttft_p50_ms']:>6.0f} ms   │  {baseline['ttft_p50_ms']:>6.0f} ms   │  {advanced['ttft_p50_ms']:>6.0f} ms   │")
print("└─────────────────┴──────────────┴──────────────┴──────────────┘")

print(f"""
PROGRESSION SUMMARY:
• Easy Wins (batch=8): +{((baseline['throughput_tok_s']/orig['throughput_tok_s']-1)*100):.0f}% throughput, -{((1-baseline['tpot_p50_ms']/orig['tpot_p50_ms'])*100):.0f}% TPOT
• Medium-Term (+chunked): {((advanced['throughput_tok_s']/baseline['throughput_tok_s']-1)*100):+.1f}% throughput, {((advanced['tpot_p50_ms']/baseline['tpot_p50_ms']-1)*100):+.1f}% TPOT
""")

print("="*75)
