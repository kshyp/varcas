#!/bin/bash
# Run vLLM server with NCU profiling and varcas benchmark harness
# Based on the working run_ncu_profile.sh but adapted for varcas benchmark
# Usage: ./run_ncu_vllm_benchmark.sh [profile_name]

set -e

PROFILE_NAME=${1:-chat_medium}
OUTPUT_DIR="$HOME/varcas/profilers/nvidia"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/vllm_ncu_${PROFILE_NAME}_${TIMESTAMP}"
VLLM_PORT=8000
BENCHMARK_DURATION=30

# Configuration
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DTYPE="half"
MAX_MODEL_LEN=1024
GPU_MEM_UTIL=0.9

echo "═══════════════════════════════════════════════════════════════"
echo "  NCU Profiling: vLLM + $PROFILE_NAME Benchmark"
echo "═══════════════════════════════════════════════════════════════"
echo "Output: $OUTPUT_FILE.ncu-rep"
echo ""

# Cleanup
cleanup() {
    echo ""
    echo "Cleaning up..."
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 2
}
trap cleanup EXIT

# Initial cleanup
echo "Cleaning up previous processes..."
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "ncu " 2>/dev/null || true
sleep 2

rm -f "$OUTPUT_FILE.ncu-rep"

echo ""
echo "Step 1: Starting vLLM server with NCU profiling..."
echo ""

# Start vLLM with NCU profiling (based on working run_ncu_profile.sh)
ncu -f -o "$OUTPUT_FILE" \
    --target-processes all \
    python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --dtype $DTYPE \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --port $VLLM_PORT &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "NCU profiling started"
echo ""

# Wait for server to be ready
echo "Step 2: Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "✓ Server ready after ${i}s"
        break
    fi
    if ! ps -p $SERVER_PID > /dev/null 2>&1; then
        echo "✗ Server process died!"
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "✗ Server failed to start"
    exit 1
fi

echo ""
echo "Step 3: Running warmup..."
curl -s http://localhost:$VLLM_PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "'"$MODEL"'", "prompt": "Hello", "max_tokens": 10}' > /dev/null 2>&1 || true
echo "✓ Warmup complete"

echo ""
echo "Step 4: Running $PROFILE_NAME benchmark for ${BENCHMARK_DURATION}s..."
echo ""

cd "$HOME/varcas/benchmark_harness"

# Run benchmark
python varcas_load_harness.py \
    --profile "$PROFILE_NAME" \
    --duration $BENCHMARK_DURATION \
    --output "$OUTPUT_DIR/result_${PROFILE_NAME}_${TIMESTAMP}.json" || true

echo ""
echo "Step 5: Stopping server and finalizing profile..."
echo ""

# Kill server - NCU will finalize the profile
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# Give NCU time to write the profile
sleep 5

# Report results
if [ -f "${OUTPUT_FILE}.ncu-rep" ]; then
    SIZE=$(ls -lh "${OUTPUT_FILE}.ncu-rep" | awk '{print $5}')
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ✓ NCU Profile Created Successfully!"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "  Profile:   ${OUTPUT_FILE}.ncu-rep ($SIZE)"
    echo "  Benchmark: $OUTPUT_DIR/result_${PROFILE_NAME}_${TIMESTAMP}.json"
    echo ""
    echo "  View with: ncu-ui ${OUTPUT_FILE}.ncu-rep"
    echo "  Export:    ncu --import ${OUTPUT_FILE}.ncu-rep --page raw --csv > ${OUTPUT_FILE}.csv"
    echo "  Summary:   ncu --import ${OUTPUT_FILE}.ncu-rep --print-summary per-kernel"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
else
    echo "✗ Profile file not found"
    exit 1
fi
