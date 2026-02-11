#!/bin/bash
# Nsight Systems Profile Script for vLLM

OUTPUT_DIR="$HOME/varcas/profilers/nvidia"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/vllm_profile_${TIMESTAMP}"
PORT=8000
DELAY=60

echo "=== Nsight Systems Profile - vLLM ==="
echo "Output: $OUTPUT_FILE.nsys-rep"
echo ""

# Kill any existing processes
echo "Stopping any existing vLLM/nsys processes..."
ps aux | grep "vllm.entrypoints" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
ps aux | grep "nsys profile" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
sleep 2

echo "Starting vLLM with delayed profiling (${DELAY}s)..."
nsys profile -o "$OUTPUT_FILE" \
  --trace cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --gpu-metrics-device=all \
  --delay=$DELAY \
  --force-overwrite=true \
  python -m vllm.entrypoints.openai.api_server \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype half \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.9 \
  --port $PORT &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server
echo "Waiting for server..."
for i in $(seq 1 120); do
    curl -s http://localhost:$PORT/health > /dev/null 2>&1 && break
    sleep 1
done
echo "Server ready"

# Wait for profiling to start
if [ $i -lt $DELAY ]; then
    sleep $((DELAY - i))
fi
echo "Profiling active!"

# Run benchmark
echo "Running benchmark..."
cd "$HOME/varcas/benchmark_harness"
python varcas_load_harness.py --profile mixed

# Cleanup
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
sleep 3

echo ""
echo "=== Profile Complete ==="
ls -lh "$OUTPUT_FILE"*
echo ""
echo "View: nsys-ui $OUTPUT_FILE.nsys-rep"
