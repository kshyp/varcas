#!/bin/bash
# Profile vLLM with nsys (Nsight Systems) during varcas benchmark execution
# This is lighter weight than NCU and captures system-wide GPU activity
# Usage: ./run_nsys_vllm_varcas.sh [profile_name]

set -e

PROFILE_NAME=${1:-chat_medium}
OUTPUT_DIR="$HOME/varcas/profilers/nvidia"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/vllm_nsys_${PROFILE_NAME}_${TIMESTAMP}"
VLLM_PORT=8000
BENCHMARK_DURATION=30

# Configuration
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DTYPE="half"
MAX_MODEL_LEN=1024
GPU_MEM_UTIL=0.7

echo "═══════════════════════════════════════════════════════════════"
echo "  Nsight Systems Profiling: vLLM + $PROFILE_NAME"
echo "═══════════════════════════════════════════════════════════════"
echo "Output: $OUTPUT_FILE.nsys-rep"
echo ""

# Cleanup
cleanup() {
    echo ""
    echo "Cleaning up..."
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "nsys " 2>/dev/null || true
    sleep 2
}
trap cleanup EXIT

# Initial cleanup
echo "Cleaning up previous processes..."
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "nsys " 2>/dev/null || true
sleep 2

rm -f "$OUTPUT_FILE.nsys-rep" "$OUTPUT_FILE.sqlite"

echo ""
echo "Step 1: Starting vLLM server with nsys profiling..."
echo ""

# Start vLLM with nsys profiling
# Using --delay to skip model loading and start profiling during benchmark
nsys profile -o "$OUTPUT_FILE" \
    --trace cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --sample=none \
    --delay=60 \
    --force-overwrite=true \
    python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --dtype $DTYPE \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --port $VLLM_PORT > /tmp/vllm_nsys.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "nsys profiling started (will begin after 60s delay)"
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
        cat /tmp/vllm_nsys.log
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "✗ Server failed to start"
    cat /tmp/vllm_nsys.log
    exit 1
fi

echo ""
echo "Step 3: Running warmup..."
curl -s http://localhost:$VLLM_PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "'"$MODEL"'", "prompt": "Hello", "max_tokens": 10}' > /dev/null 2>&1 || true
echo "✓ Warmup complete"

# Calculate remaining delay
REMAINING=$((60 - 45))  # Approximate time to start server
if [ $REMAINING -gt 0 ]; then
    echo ""
    echo "Waiting ${REMAINING}s for nsys profiling to start..."
    sleep $REMAINING
fi
echo ""
echo "✓ Profiling is now ACTIVE!"

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

# Kill server - nsys will finalize the profile
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# Give nsys time to write the profile
sleep 5

# Report results
if [ -f "${OUTPUT_FILE}.nsys-rep" ]; then
    SIZE=$(ls -lh "${OUTPUT_FILE}.nsys-rep" | awk '{print $5}')
    SQLITE_SIZE=$(ls -lh "${OUTPUT_FILE}.sqlite" 2>/dev/null | awk '{print $5}' || echo "N/A")
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ✓ Nsight Systems Profile Created Successfully!"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "Files:"
    echo "  Profile (.nsys-rep): ${OUTPUT_FILE}.nsys-rep ($SIZE)"
    echo "  Profile (.sqlite):   ${OUTPUT_FILE}.sqlite ($SQLITE_SIZE)"
    echo "  Benchmark Result:    $OUTPUT_DIR/result_${PROFILE_NAME}_${TIMESTAMP}.json"
    echo "  Server Log:          /tmp/vllm_nsys.log"
    echo ""
    echo "View Profile:"
    echo "  nsys-ui ${OUTPUT_FILE}.nsys-rep"
    echo ""
    echo "Export to CSV:"
    echo "  nsys export -t cuda_kernel,cuda_api,nvtx -o ${OUTPUT_FILE}.csv ${OUTPUT_FILE}.nsys-rep"
    echo ""
    echo "Generate Stats:"
    echo "  nsys stats --report cuda_kernel ${OUTPUT_FILE}.nsys-rep"
    echo "═══════════════════════════════════════════════════════════════"
else
    echo "✗ Profile file not found"
    exit 1
fi
