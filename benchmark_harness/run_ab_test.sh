#!/bin/bash
# Run A/B test multiple times and average results
# IMPORTANT: Due to system non-determinism, only improvements >15-20% are reliably detectable

TRACE_FILE="prompts.json"
N_RUNS=${N_RUNS:-5}  # Default 5 runs, increase to 10 for marginal gains

echo "=========================================="
echo "A/B Testing with Statistical Averaging"
echo "=========================================="
echo "Runs per variant: $N_RUNS"
echo "Note: Improvements <15% may be indistinguishable from noise"
echo ""

echo "Running BASELINE ($N_RUNS times)..."
for i in $(seq 1 $N_RUNS); do
    echo "  Run $i/$N_RUNS..."
    python varcas_load_harness.py \
        --trace $TRACE_FILE \
        --output baseline_run$i.json 2>/dev/null
done

echo ""
echo "Apply your optimization, then press Enter to continue..."
read

echo "Running OPTIMIZED ($N_RUNS times)..."
for i in $(seq 1 $N_RUNS); do
    echo "  Run $i/$N_RUNS..."
    python varcas_load_harness.py \
        --trace $TRACE_FILE \
        --output optimized_run$i.json 2>/dev/null
done

echo ""
echo "=========================================="
echo "Statistical Analysis"
echo "=========================================="
python3 << 'PYEOF'
import json
import glob
import statistics
import math

def load_metrics(pattern):
    metrics = []
    for f in glob.glob(pattern):
        data = json.load(open(f))['metrics']
        metrics.append({
            'rps': data['throughput_rps'],
            'tok_s': data['throughput_tok_s'],
            'ttft_p50': data['ttft_p50_ms'],
            'latency_p50': data['latency_p50_ms']
        })
    return metrics

def avg_std(data, key):
    vals = [d[key] for d in data]
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0

def cohen_d(group1, group2, key):
    """Calculate effect size. d > 0.8 is considered 'large'."""
    m1, s1 = avg_std(group1, key)
    m2, s2 = avg_std(group2, key)
    pooled_std = math.sqrt(((len(group1)-1)*s1**2 + (len(group2)-1)*s2**2) / (len(group1)+len(group2)-2))
    return (m2 - m1) / pooled_std if pooled_std > 0 else 0

baseline = load_metrics('baseline_run*.json')
optimized = load_metrics('optimized_run*.json')

if not baseline or not optimized:
    print("ERROR: Missing result files")
    exit(1)

print(f"\nSamples: {len(baseline)} baseline, {len(optimized)} optimized\n")
print(f"{'Metric':<15} {'Baseline':<22} {'Optimized':<22} {'Change':<12} {'Effect'}")
print("-" * 85)

for key, name in [('rps', 'Req/s'), ('tok_s', 'Tok/s'), ('ttft_p50', 'TTFT p50'), ('latency_p50', 'Latency p50')]:
    b_mean, b_std = avg_std(baseline, key)
    o_mean, o_std = avg_std(optimized, key)
    change = (o_mean / b_mean - 1) * 100 if b_mean > 0 else 0
    d = cohen_d(baseline, optimized, key)
    
    # Effect size interpretation
    if abs(d) < 0.2:
        effect = "negligible"
    elif abs(d) < 0.5:
        effect = "small"
    elif abs(d) < 0.8:
        effect = "medium ⚠️"
    else:
        effect = "LARGE ✅"
    
    print(f"{name:<15} {b_mean:>7.2f} ±{b_std:<6.2f}      {o_mean:>7.2f} ±{o_std:<6.2f}      {change:>+7.1f}%     {d:>+.2f} {effect}")

print("\n" + "="*85)
print("Interpretation:")
print("  - Effect size (Cohen's d) > 0.8 = statistically significant improvement")
print("  - Improvements < 15% typically require 10+ runs to confirm")
print("  - When in doubt, trust Req/s over Tok/s (less variance)")
print("="*85)

PYEOF
