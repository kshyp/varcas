#!/bin/bash
# A/B Testing Workflow Example for vLLM
# This demonstrates how to run the exact same prompts against baseline and optimized vLLM

set -e

echo "=========================================="
echo "A/B Testing with Identical Prompts"
echo "=========================================="

# Step 1: Run baseline test and save prompts to trace file
echo ""
echo "Step 1: Running BASELINE test and saving prompts..."
python varcas_load_harness.py \
    --url http://localhost:8000 \
    --profile chat_medium \
    --duration 30 \
    --output baseline_result.json \
    --save-trace ab_test_trace.json \
    --meaningful-prompts

echo "Baseline complete. Prompts saved to ab_test_trace.json"

# Step 2: Apply your optimizations to vLLM here
# (e.g., restart vLLM with optimized settings, apply patches, etc.)
echo ""
echo "Step 2: Apply your vLLM optimizations now..."
echo "Press Enter when ready to run optimized test..."
read

# Step 3: Run optimized test with EXACT same prompts from trace
echo ""
echo "Step 3: Running OPTIMIZED test with same prompts..."
python varcas_load_harness.py \
    --url http://localhost:8000 \
    --trace ab_test_trace.json \
    --output optimized_result.json

echo "Optimized test complete."

# Step 4: Compare results
echo ""
echo "=========================================="
echo "Results Comparison"
echo "=========================================="

echo ""
echo "BASELINE:"
python3 -c "
import json
data = json.load(open('baseline_result.json'))['metrics']
print(f\"  Throughput: {data['throughput_rps']:.2f} req/s, {data['throughput_tok_s']:.1f} tok/s\")
print(f\"  TTFT: p50={data['ttft_p50_ms']:.1f}ms, p99={data['ttft_p99_ms']:.1f}ms\")
print(f\"  TPOT: p50={data['tpot_p50_ms']:.2f}ms, p99={data['tpot_p99_ms']:.2f}ms\")
print(f\"  Latency: p50={data['latency_p50_ms']:.1f}ms, p99={data['latency_p99_ms']:.1f}ms\")
"

echo ""
echo "OPTIMIZED:"
python3 -c "
import json
data = json.load(open('optimized_result.json'))['metrics']
print(f\"  Throughput: {data['throughput_rps']:.2f} req/s, {data['throughput_tok_s']:.1f} tok/s\")
print(f\"  TTFT: p50={data['ttft_p50_ms']:.1f}ms, p99={data['ttft_p99_ms']:.1f}ms\")
print(f\"  TPOT: p50={data['tpot_p50_ms']:.2f}ms, p99={data['tpot_p99_ms']:.2f}ms\")
print(f\"  Latency: p50={data['latency_p50_ms']:.1f}ms, p99={data['latency_p99_ms']:.1f}ms\")
"

echo ""
echo "A/B testing complete! Results saved to:"
echo "  - baseline_result.json"
echo "  - optimized_result.json"
echo "  - ab_test_trace.json (prompts used)"
