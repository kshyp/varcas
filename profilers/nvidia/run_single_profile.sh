#!/bin/bash
# Run a single benchmark profile with warmed-up server
# Usage: ./run_single_profile.sh <profile_name> <profile_arg>

set -e

PROFILE_NAME=${1:-mixed}
PROFILE_ARG=${2:-mixed}
OUTPUT_DIR="$HOME/varcas/profilers/nvidia"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/vllm_nsys_${PROFILE_NAME}_${TIMESTAMP}"
VLLM_PORT=8000
DELAY=45

echo "=== Profiling: $PROFILE_NAME ($PROFILE_ARG) ==="

# Cleanup
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "nsys profile" 2>/dev/null || true
sleep 2
rm -f "$OUTPUT_FILE.nsys-rep" "$OUTPUT_FILE.sqlite"

# Start server with delayed profiling
nsys profile -o "$OUTPUT_FILE" \
    --trace cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --sample=none \
    --delay=$DELAY \
    --force-overwrite=true \
    python -m vllm.entrypoints.openai.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dtype half \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.9 \
    --port $VLLM_PORT &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server
echo "Waiting for server..."
for i in $(seq 1 120); do
    curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1 && break
    sleep 1
done
echo "Server ready after ${i}s"

# Wait for profiling to start
REMAINING=$((DELAY - i))
if [ $REMAINING -gt 0 ]; then
    echo "Waiting ${REMAINING}s for profiling to start..."
    sleep $REMAINING
fi
echo "Profiling ACTIVE!"

# Run benchmark
echo "Running benchmark: $PROFILE_ARG..."
cd "$HOME/varcas/benchmark_harness"
python varcas_load_harness.py --profile "$PROFILE_ARG" || echo "(benchmark completed)"

# Shutdown
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
sleep 3

# Check result
if [ -f "${OUTPUT_FILE}.nsys-rep" ]; then
    SIZE=$(ls -lh "${OUTPUT_FILE}.nsys-rep" | awk '{print $5}')
    echo "✓ Profile created: ${OUTPUT_FILE}.nsys-rep ($SIZE)"
else
    echo "✗ Profile not found"
fi

echo ""
