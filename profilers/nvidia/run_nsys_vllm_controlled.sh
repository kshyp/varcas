#!/bin/bash
set -e

# Nsight Systems Profile Script for vLLM - Controlled Start/Stop
# This script:
#   1. Launches vLLM server with profiling suspended (--suspend=true)
#   2. Warms up the server with sample requests
#   3. Explicitly starts profiling with 'nsys start'
#   4. Runs the benchmark
#   5. Stops profiling with 'nsys stop'
#
# This gives precise control over what gets profiled.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_NAME="vllm_nsys_controlled_${TIMESTAMP}"
OUTPUT_FILE="$OUTPUT_DIR/$PROFILE_NAME"

# Configuration
VLLM_PORT=8000
WARMUP_REQUESTS=10

echo "=================================================="
echo "Nsight Systems Profile - vLLM (Controlled Capture)"
echo "=================================================="
echo "Output: $OUTPUT_FILE.nsys-rep"
echo ""

# Clean up any existing vLLM processes
echo "[1/6] Cleaning up existing processes..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "nsys profile" 2>/dev/null || true
sleep 3 || true

# Remove any stale profile files
rm -f "$OUTPUT_FILE.nsys-rep" "$OUTPUT_FILE.sqlite" || true

echo ""
echo "[2/6] Launching vLLM server with profiling SUSPENDED..."
echo "       Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0"
echo ""

# Launch vLLM with nsys but profiling is initially suspended
nsys launch \
  --trace cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --gpu-metrics-device=all \
  --sample=none \
  --force-overwrite=true \
  --output "$OUTPUT_FILE" \
  python -m vllm.entrypoints.openai.api_server \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype half \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.9 \
  --port $VLLM_PORT &

SERVER_PID=$!
echo "       Server PID: $SERVER_PID"

# Wait a moment for nsys to initialize
sleep 2

# Check if nsys session was created
echo ""
echo "[3/6] Waiting for vLLM server to be ready..."
MAX_WAIT=300
for i in $(seq 1 $MAX_WAIT); do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "       Server is ready! (took ${i}s)"
        break
    fi
    if ! ps -p $SERVER_PID > /dev/null; then
        echo "       ERROR: Server process died!"
        exit 1
    fi
    if [ $i -eq $MAX_WAIT ]; then
        echo "       ERROR: Server failed to start within ${MAX_WAIT}s"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

echo ""
echo "[4/6] Warming up server with $WARMUP_REQUESTS requests..."
for i in $(seq 1 $WARMUP_REQUESTS); do
    curl -s -X POST http://localhost:$VLLM_PORT/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "prompt": "Hello, how are you?",
            "max_tokens": 50,
            "temperature": 0.7
        }' > /dev/null 2>&1
    if [ $((i % 5)) -eq 0 ]; then
        echo "       Completed $i/$WARMUP_REQUESTS warmup requests"
    fi
done
echo "       Warmup complete!"

echo ""
echo "[5/6] Starting profiling and running benchmark..."
echo "       Profile: 60% chat, 30% RAG, 10% code"
echo ""

# Start profiling
nsys start || echo "       (nsys start may have auto-started with launch)"

cd "$HOME/varcas/benchmark_harness"
python varcas_load_harness.py --profile mixed || true

echo ""
echo "[6/6] Stopping profiling and shutting down..."

# Stop profiling (this generates the .nsys-rep file)
nsys stop --output "$OUTPUT_FILE" 2>/dev/null || true

# Kill the server
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# Wait for file generation
sleep 3 || true

echo ""
echo "=================================================="
echo "Profile Complete!"
echo "=================================================="
echo "Output Files:"
ls -lh "$OUTPUT_FILE"* 2>/dev/null || ls -lh "$OUTPUT_DIR"/*.nsys-rep 2>/dev/null | tail -5 || echo "  No files found"
echo ""
echo "To view the profile:"
echo "  nsys-ui $OUTPUT_FILE.nsys-rep 2>/dev/null || nsys-ui \$(ls -t $OUTPUT_DIR/*.nsys-rep | head -1)"
echo ""
echo "To analyze in CLI:"
echo "  nsys stats $OUTPUT_FILE.nsys-rep 2>/dev/null || nsys stats \$(ls -t $OUTPUT_DIR/*.nsys-rep | head -1)"
echo "=================================================="
