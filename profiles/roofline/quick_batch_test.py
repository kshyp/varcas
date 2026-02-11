#!/usr/bin/env python3
"""Quick batch size comparison - baseline vs optimized."""

import subprocess
import sys
import time
import json
from pathlib import Path

def test_batch_size(batch_size, duration=45):
    """Test a single batch size configuration."""
    print(f"\n{'='*60}")
    print(f"Testing batch_size={batch_size}")
    print('='*60)
    
    # Start vLLM
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "TheBloke/Llama-2-7B-AWQ",
        "--dtype", "half",
        "--max-model-len", "2048",
        "--max-num-seqs", str(batch_size),
        "--gpu-memory-utilization", "0.90",
        "--port", "8000",
        "--quantization", "awq",
        "--enforce-eager"
    ]
    
    print(f"Starting vLLM with max_num_seqs={batch_size}...")
    import os
    log = open(f"varcas/profiles/roofline/batch_optimization/vllm_b{batch_size}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    
    # Wait for ready
    import urllib.request
    ready = False
    for i in range(120):
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=1)
            ready = True
            print(f"vLLM ready in {i}s")
            break
        except:
            time.sleep(1)
    
    if not ready:
        print("vLLM failed to start")
        proc.terminate()
        return None
    
    time.sleep(3)
    
    # Run load test
    print(f"Running load test for {duration}s...")
    harness = Path("varcas/benchmark_harness/varcas_load_harness.py")
    result = subprocess.run(
        [sys.executable, str(harness), "--profile", "chat_medium", 
         "--duration", str(duration), "--meaningful-prompts",
         "--output", f"varcas/profiles/roofline/batch_optimization/results_b{batch_size}.json"],
        capture_output=True, text=True, timeout=duration*2 + 30
    )
    
    print(result.stdout)
    
    # Stop vLLM
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except:
        pass
    proc.wait()
    
    # Parse results
    try:
        with open(f"varcas/profiles/roofline/batch_optimization/results_b{batch_size}.json") as f:
            data = json.load(f)
        m = data["metrics"]
        return {
            "batch_size": batch_size,
            "throughput_rps": m["throughput_rps"],
            "throughput_tok_s": m["throughput_tok_s"],
            "ttft_p50_ms": m["ttft_p50_ms"],
            "tpot_p50_ms": m["tpot_p50_ms"],
            "latency_p50_ms": m["latency_p50_ms"],
            "ttft_p99_ms": m["ttft_p99_ms"],
            "latency_p99_ms": m["latency_p99_ms"]
        }
    except Exception as e:
        print(f"Error parsing results: {e}")
        return None

# Run tests
results = []
for bs in [1, 8]:
    result = test_batch_size(bs, duration=45)
    if result:
        results.append(result)
    time.sleep(3)

# Summary
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"\n{'Batch':<8} {'Req/s':<10} {'Tok/s':<12} {'TTFT p50':<12} {'TPOT p50':<12}")
print("-"*60)

for r in results:
    print(f"{r['batch_size']:<8} {r['throughput_rps']:<10.2f} {r['throughput_tok_s']:<12.1f} {r['ttft_p50_ms']:<12.1f} {r['tpot_p50_ms']:<12.2f}")

if len(results) == 2:
    baseline, optimized = results[0], results[1]
    improvement = ((optimized['throughput_rps'] / baseline['throughput_rps']) - 1) * 100
    print(f"\n🚀 Throughput improvement: {improvement:+.1f}%")
    
    with open("varcas/profiles/roofline/batch_optimization/comparison_summary.json", "w") as f:
        json.dump({"baseline": baseline, "optimized": optimized, "improvement_pct": improvement}, f, indent=2)

print("\nResults saved to varcas/profiles/roofline/batch_optimization/")
