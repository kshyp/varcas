#!/usr/bin/env python3
"""
Easy Wins Optimization Comparison
Compares baseline vs optimized vLLM configuration.
"""

import subprocess
import sys
import time
import json
import os
from pathlib import Path

def wait_for_vllm(port=8000, timeout=120):
    """Wait for vLLM to be ready."""
    import urllib.request
    for i in range(timeout):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=1)
            return True
        except:
            time.sleep(1)
    return False

def kill_vllm():
    """Kill any running vLLM processes."""
    subprocess.run("pkill -f 'vllm.entrypoints' 2>/dev/null", shell=True)
    time.sleep(2)

def run_test(name, config_script, duration=45):
    """Run a test with given configuration."""
    print(f"\n{'='*70}")
    print(f"TESTING: {name}")
    print(f"Config: {config_script}")
    print('='*70)
    
    # Kill any existing vLLM
    kill_vllm()
    
    # Start vLLM with the configuration
    print(f"\nStarting vLLM with {name}...")
    log_file = open(f"varcas/profiles/roofline/easy_wins_{name.lower().replace(' ', '_')}.log", "w")
    
    cmd = ["bash", config_script]
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, 
                           preexec_fn=os.setsid)
    
    # Wait for ready
    if not wait_for_vllm():
        print(f"❌ vLLM failed to start for {name}")
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except:
            pass
        return None
    
    print(f"✅ vLLM ready")
    time.sleep(3)
    
    # Run load test
    print(f"Running load test ({duration}s)...")
    output_file = f"varcas/profiles/roofline/easy_wins_{name.lower().replace(' ', '_')}.json"
    harness = Path("varcas/benchmark_harness/varcas_load_harness.py")
    
    result = subprocess.run(
        [sys.executable, str(harness), "--profile", "chat_medium", 
         "--duration", str(duration), "--meaningful-prompts",
         "--output", output_file],
        capture_output=True, text=True, timeout=duration*2 + 30
    )
    
    print(result.stdout[-2500:] if len(result.stdout) > 2500 else result.stdout)
    
    # Stop vLLM
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except:
        pass
    proc.wait()
    
    # Parse results
    try:
        with open(output_file) as f:
            data = json.load(f)
        m = data["metrics"]
        return {
            "config": name,
            "requests": m["total_requests"],
            "success_rate": m["successful_requests"] / m["total_requests"] * 100,
            "throughput_rps": m["throughput_rps"],
            "throughput_tok_s": m["throughput_tok_s"],
            "ttft_p50_ms": m["ttft_p50_ms"],
            "ttft_p99_ms": m["ttft_p99_ms"],
            "tpot_p50_ms": m["tpot_p50_ms"],
            "tpot_p99_ms": m["tpot_p99_ms"],
            "latency_p50_ms": m["latency_p50_ms"],
            "latency_p99_ms": m["latency_p99_ms"]
        }
    except Exception as e:
        print(f"Error parsing results: {e}")
        return None

# Run comparisons
print("="*70)
print("EASY WINS OPTIMIZATION COMPARISON")
print("="*70)
print("""
Comparing:
1. BASELINE: Original start_vllm.sh
   - No explicit batch size (default)
   - No CUDA graphs
   
2. OPTIMIZED: Easy wins applied
   - max_num_seqs=8 (increased batching)
   - enable-cuda-graph (reduced CPU overhead)
""")

results = []

# Test baseline
baseline = run_test("BASELINE", "start_vllm.sh", duration=45)
if baseline:
    results.append(baseline)
time.sleep(3)

# Test optimized
optimized = run_test("OPTIMIZED", "start_vllm_optimized.sh", duration=45)
if optimized:
    results.append(optimized)

# Summary
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)

if len(results) == 2:
    b, o = results[0], results[1]
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    EASY WINS OPTIMIZATION RESULTS                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  Baseline:   Original start_vllm.sh                                  ║
║  Optimized:  max_num_seqs=8 + CUDA graphs                            ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    print("┌────────────────────────────────────────────────────────────────────┐")
    print("│ THROUGHPUT METRICS                                                 │")
    print("├────────────────────────────────────────────────────────────────────┤")
    print(f"│  Token Throughput:  {b['throughput_tok_s']:>6.1f} → {o['throughput_tok_s']:<6.1f} tok/s  ({((o['throughput_tok_s']/b['throughput_tok_s']-1)*100):+.1f}%) │")
    print(f"│  Request Throughput: {b['throughput_rps']:>6.2f} → {o['throughput_rps']:<6.2f} req/s  ({((o['throughput_rps']/b['throughput_rps']-1)*100):+.1f}%) │")
    print("└────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌────────────────────────────────────────────────────────────────────┐")
    print("│ LATENCY METRICS                                                    │")
    print("├────────────────────────────────────────────────────────────────────┤")
    ttft_change = ((o['ttft_p50_ms']/b['ttft_p50_ms']-1)*100)
    tpot_change = ((o['tpot_p50_ms']/b['tpot_p50_ms']-1)*100)
    latency_change = ((o['latency_p50_ms']/b['latency_p50_ms']-1)*100)
    
    print(f"│  TTFT p50:         {b['ttft_p50_ms']:>7.0f} → {o['ttft_p50_ms']:<7.0f} ms     ({ttft_change:+.1f}%) │")
    print(f"│  TTFT p99:         {b['ttft_p99_ms']:>7.0f} → {o['ttft_p99_ms']:<7.0f} ms     ({((o['ttft_p99_ms']/b['ttft_p99_ms']-1)*100):+.1f}%) │")
    print(f"│  TPOT p50:         {b['tpot_p50_ms']:>7.1f} → {o['tpot_p50_ms']:<7.1f} ms     ({tpot_change:+.1f}%) │")
    print(f"│  TPOT p99:         {b['tpot_p99_ms']:>7.1f} → {o['tpot_p99_ms']:<7.1f} ms     ({((o['tpot_p99_ms']/b['tpot_p99_ms']-1)*100):+.1f}%) │")
    print(f"│  Latency p50:      {b['latency_p50_ms']:>7.0f} → {o['latency_p50_ms']:<7.0f} ms     ({latency_change:+.1f}%) │")
    print(f"│  Latency p99:      {b['latency_p99_ms']:>7.0f} → {o['latency_p99_ms']:<7.0f} ms     ({((o['latency_p99_ms']/b['latency_p99_ms']-1)*100):+.1f}%) │")
    print("└────────────────────────────────────────────────────────────────────┘")
    
    # Calculate key improvements
    tok_improvement = ((o['throughput_tok_s']/b['throughput_tok_s']-1)*100)
    
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║                         KEY FINDINGS                                 ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    
    if tok_improvement > 200:
        print(f"║  ✅ TOKEN THROUGHPUT: +{tok_improvement:.0f}% improvement!                        ║")
        print("║     Major gain from increased batching (max_num_seqs=8)              ║")
    elif tok_improvement > 50:
        print(f"║  ✅ TOKEN THROUGHPUT: +{tok_improvement:.0f}% improvement                       ║")
    else:
        print(f"║  ⚠️  TOKEN THROUGHPUT: {tok_improvement:+.1f}% change                           ║")
    
    if ttft_change < 0:
        print(f"║  ✅ TTFT IMPROVED: {ttft_change:.1f}% (CUDA graphs helping!)                    ║")
    else:
        print(f"║  ⚠️  TTFT INCREASED: +{ttft_change:.1f}% (batching trade-off)                    ║")
    
    if tpot_change < 0:
        print(f"║  ✅ TPOT IMPROVED: {tpot_change:.1f}% (CUDA graphs helping!)                    ║")
    else:
        print(f"║  ℹ️  TPOT INCREASED: +{tpot_change:.1f}% (expected with batching)                ║")
    
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    # Save detailed comparison
    comparison = {
        "baseline": b,
        "optimized": o,
        "improvements": {
            "token_throughput_pct": tok_improvement,
            "request_throughput_pct": ((o['throughput_rps']/b['throughput_rps']-1)*100),
            "ttft_p50_pct": ttft_change,
            "tpot_p50_pct": tpot_change,
            "latency_p50_pct": latency_change
        },
        "optimizations_applied": [
            "max_num_seqs=8 (increased batching)",
            "enable-cuda-graph (reduced CPU overhead)",
            "Flash Attention (already enabled in vLLM)"
        ]
    }
    
    with open("varcas/profiles/roofline/easy_wins_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n📁 Detailed comparison saved to: easy_wins_comparison.json")

print("\n" + "="*70)
print("COMPARISON COMPLETE")
print("="*70)
