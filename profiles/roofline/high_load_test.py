#!/usr/bin/env python3
"""Test with high load to see throughput limits."""

import subprocess
import sys
import time
import json
from pathlib import Path

def test_high_load(batch_size, duration=40):
    """Test with high RPS load."""
    print(f"\n{'='*60}")
    print(f"HIGH LOAD TEST: batch_size={batch_size}")
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
    
    import os
    log = open(f"varcas/profiles/roofline/batch_optimization/highload_vllm_b{batch_size}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    
    # Wait for ready
    import urllib.request
    for i in range(120):
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=1)
            print(f"vLLM ready in {i}s")
            break
        except:
            time.sleep(1)
    else:
        print("vLLM failed")
        proc.terminate()
        return None
    
    time.sleep(3)
    
    # Run HIGH load test using chat_high profile (50 RPS)
    print(f"Running HIGH load test (chat_high ~50 RPS) for {duration}s...")
    harness = Path("varcas/benchmark_harness/varcas_load_harness.py")
    result = subprocess.run(
        [sys.executable, str(harness), "--profile", "chat_high", 
         "--duration", str(duration), "--meaningful-prompts",
         "--output", f"varcas/profiles/roofline/batch_optimization/highload_results_b{batch_size}.json"],
        capture_output=True, text=True, timeout=duration*2 + 30
    )
    
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except:
        pass
    proc.wait()
    
    try:
        with open(f"varcas/profiles/roofline/batch_optimization/highload_results_b{batch_size}.json") as f:
            data = json.load(f)
        m = data["metrics"]
        return {
            "batch_size": batch_size,
            "requests": m["total_requests"],
            "success_rate": m["successful_requests"] / m["total_requests"] * 100,
            "throughput_rps": m["throughput_rps"],
            "throughput_tok_s": m["throughput_tok_s"],
            "ttft_p50_ms": m["ttft_p50_ms"],
            "tpot_p50_ms": m["tpot_p50_ms"],
            "latency_p50_ms": m["latency_p50_ms"]
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

# Run high load tests
print("="*60)
print("HIGH LOAD TESTING (50 RPS TARGET)")
print("="*60)
print("This will show the true throughput limits with different batch sizes")

results = []
for bs in [1, 8]:
    r = test_high_load(bs, duration=35)
    if r:
        results.append(r)
    time.sleep(3)

print("\n" + "="*60)
print("HIGH LOAD COMPARISON (50 RPS Target)")
print("="*60)
print(f"\n{'Batch':<8} {'Req':<8} {'Succ%':<8} {'Req/s':<10} {'Tok/s':<12} {'Latency p50':<12}")
print("-"*60)

for r in results:
    print(f"{r['batch_size']:<8} {r['requests']:<8} {r['success_rate']:<8.1f} {r['throughput_rps']:<10.2f} {r['throughput_tok_s']:<12.1f} {r['latency_p50_ms']:<12.1f}")

if len(results) == 2:
    b, o = results[0], results[1]
    print("\n📊 HIGH LOAD ANALYSIS:")
    print(f"  Batch=1 max throughput:  {b['throughput_rps']:.2f} req/s")
    print(f"  Batch=8 max throughput:  {o['throughput_rps']:.2f} req/s")
    print(f"  Improvement: {((o['throughput_rps']/b['throughput_rps']-1)*100):+.1f}%")
    print(f"\n  Batch=1 token throughput: {b['throughput_tok_s']:.1f} tok/s")
    print(f"  Batch=8 token throughput: {o['throughput_tok_s']:.1f} tok/s")
    print(f"  Improvement: {((o['throughput_tok_s']/b['throughput_tok_s']-1)*100):+.1f}%")
    
    with open("varcas/profiles/roofline/batch_optimization/highload_comparison.json", "w") as f:
        json.dump({"results": results}, f, indent=2)

print("\nResults saved!")
