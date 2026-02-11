#!/bin/bash
# Nsight Systems Profile Script for vLLM - Simple & Robust
# Usage: ./run_nsys_vllm_simple.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_NAME="vllm_nsys_${TIMESTAMP}"
OUTPUT_FILE="$SCRIPT_DIR/$PROFILE_NAME"
VLLM_PORT=8000
DELAY=60

echo "=================================================="
echo "Nsight Systems Profile - vLLM"
echo "=================================================="

# Cleanup
echo "Cleaning up existing processes..."
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
pkill -9 -f "nsys profile" 2>/dev/null
sleep 2
rm -f "$OUTPUT_FILE.nsys-rep" "$OUTPUT_FILE.sqlite"

echo ""
echo "Starting vLLM server with ${DELAY}s delayed profiling..."
echo "(Server will warm up during this time)"
echo ""

# Start server with delayed profiling
nsys profile -o "$OUTPUT_FILE" \
  --trace cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --gpu-metrics-device=all \
  --delay=$DELAY \
  python -m vllm.entrypoints.openai.api_server \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype half \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.9 \
  --port $VLLM_PORT &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

# Wait for profiling delay to complete
REMAINING=$((DELAY - i))
if [ $REMAINING -gt 0 ]; then
    echo "Waiting ${REMAINING}s for profiling to start..."
    sleep $REMAINING
fi

echo ""
echo "Running benchmark..."
cd "$HOME/varcas/benchmark_harness"
python varcas_load_harness.py --profile mixed

echo ""
echo "Shutting down..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
sleep 3

echo ""
echo "=================================================="
echo "Profile Complete!"
echo "=================================================="
ls -lh "$OUTPUT_FILE"* 2>/dev/null
echo ""
echo "To view: nsys-ui $OUTPUT_FILE.nsys-rep"
echo "To analyze: nsys stats $OUTPUT_FILE.nsys-rep"
