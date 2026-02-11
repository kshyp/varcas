#!/bin/bash
# Profile vLLM server with NCU during benchmark execution
# Usage: ./profile_vllm_benchmark.sh [profile_name]

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
# Reduced GPU memory utilization to accommodate NCU overhead
GPU_MEM_UTIL=0.5

echo "═══════════════════════════════════════════════════════════════"
echo "  NCU Profiling: vLLM Server + $PROFILE_NAME Workload"
echo "═══════════════════════════════════════════════════════════════"
echo "Output: $OUTPUT_FILE.ncu-rep"
echo "Duration: ${BENCHMARK_DURATION}s benchmark"
echo "GPU Memory: ${GPU_MEM_UTIL} (reduced for NCU overhead)"
echo ""

# Cleanup function
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

# Start vLLM with NCU profiling
ncu -f -o "$OUTPUT_FILE" \
    --target-processes all \
    --set basic \
    --nvtx \
    python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --dtype $DTYPE \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --port $VLLM_PORT > /tmp/vllm_ncu_server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "NCU profiling started"
echo ""

# Wait for server to be ready
echo "Step 2: Waiting for server to be ready (this may take a while with NCU)..."
for i in $(seq 1 300); do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "✓ Server ready after ${i}s"
        break
    fi
    if ! ps -p $SERVER_PID > /dev/null 2>&1; then
        echo "✗ Server process died!"
        cat /tmp/vllm_ncu_server.log
        exit 1
    fi
    sleep 1
done

# Verify server is still running
if ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "✗ Server failed to start"
    cat /tmp/vllm_ncu_server.log
    exit 1
fi

echo ""
echo "Step 3: Running warmup request..."
curl -s http://localhost:$VLLM_PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "'"$MODEL"'", "prompt": "Hello", "max_tokens": 10}' > /dev/null 2>&1 || true
echo "✓ Warmup complete"

echo ""
echo "Step 4: Running $PROFILE_NAME benchmark for ${BENCHMARK_DURATION}s..."
echo ""

cd "$HOME/varcas/benchmark_harness"

# Run benchmark with specified duration
python varcas_load_harness.py \
    --profile "$PROFILE_NAME" \
    --duration $BENCHMARK_DURATION \
    --output "$OUTPUT_DIR/result_${PROFILE_NAME}_${TIMESTAMP}.json" || true

echo ""
echo "Step 5: Stopping server and finalizing profile..."
echo ""

# Kill server (NCU will finalize the profile)
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# Give NCU time to write the profile
sleep 5

# Check and report results
if [ -f "${OUTPUT_FILE}.ncu-rep" ]; then
    SIZE=$(ls -lh "${OUTPUT_FILE}.ncu-rep" | awk '{print $5}')
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ✓ NCU Profile Created Successfully!"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "Files:"
    echo "  Profile:      ${OUTPUT_FILE}.ncu-rep ($SIZE)"
    echo "  Benchmark:    $OUTPUT_DIR/result_${PROFILE_NAME}_${TIMESTAMP}.json"
    echo "  Server Log:   /tmp/vllm_ncu_server.log"
    echo ""
    echo "View Profile:"
    echo "  ncu-ui ${OUTPUT_FILE}.ncu-rep"
    echo ""
    echo "Export to CSV:"
    echo "  ncu --import ${OUTPUT_FILE}.ncu-rep --page raw --csv > ${OUTPUT_FILE}.csv"
    echo ""
    echo "Kernel Summary:"
    echo "  ncu --import ${OUTPUT_FILE}.ncu-rep --print-summary per-kernel"
    echo ""
    echo "Top Kernels by Time:"
    echo "  ncu --import ${OUTPUT_FILE}.ncu-rep --page details | head -50"
    echo "═══════════════════════════════════════════════════════════════"
else
    echo "✗ Profile file not found!"
    if [ -f "/tmp/vllm_ncu_server.log" ]; then
        echo ""
        echo "Server log tail:"
        tail -30 /tmp/vllm_ncu_server.log
    fi
    exit 1
fi
