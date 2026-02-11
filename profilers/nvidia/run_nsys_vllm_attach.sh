#!/bin/bash
set -e

# Nsight Systems Profile Script for vLLM - Attach Mode
# This script:
#   1. Starts vLLM server normally (no profiling during startup)
#   2. Warms up the server with sample requests
#   3. Attaches nsys to the RUNNING server process
#   4. Runs the benchmark while profiling
#   5. Detaches and saves profile

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_NAME="vllm_nsys_attach_${TIMESTAMP}"
OUTPUT_FILE="$OUTPUT_DIR/$PROFILE_NAME"

# Configuration
VLLM_PORT=8000
WARMUP_REQUESTS=10

echo "=================================================="
echo "Nsight Systems Profile - vLLM (Attach Mode)"
echo "=================================================="
echo "Output: $OUTPUT_FILE.nsys-rep"
echo ""

# Clean up any existing vLLM processes
echo "[1/7] Cleaning up existing vLLM processes..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -f "nsys profile" 2>/dev/null || true
sleep 3

# Remove any stale profile files
rm -f "$OUTPUT_FILE.nsys-rep" "$OUTPUT_FILE.sqlite"

echo ""
echo "[2/7] Starting vLLM server (NO profiling during startup)..."
echo "       Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0"
echo ""

# Start vLLM server WITHOUT nsys profiling
python -m vllm.entrypoints.openai.api_server \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype half \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.9 \
  --port $VLLM_PORT &

SERVER_PID=$!
echo "       Server PID: $SERVER_PID"

echo ""
echo "[3/7] Waiting for vLLM server to be ready..."
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
echo "[4/7] Warming up server with $WARMUP_REQUESTS requests..."
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
echo "[5/7] Attaching Nsight Systems to server PID $SERVER_PID..."
echo "       Output: $OUTPUT_FILE.nsys-rep"
echo ""

# Start nsys attach in background
nsys attach -o "$OUTPUT_FILE" \
  --trace cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --gpu-metrics-device=all \
  --sample=none \
  --force-overwrite=true \
  $SERVER_PID &

NSYS_PID=$!
echo "       Nsys attach PID: $NSYS_PID"

# Wait for nsys to attach
echo "       Waiting for nsys to attach..."
sleep 5

# Check if nsys is still running
if ! ps -p $NSYS_PID > /dev/null; then
    echo "       ERROR: nsys attach failed to start!"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "[6/7] Running mixed workload benchmark while profiling..."
echo "       Profile: 60% chat, 30% RAG, 10% code"
echo ""

cd "$HOME/varcas/benchmark_harness"
python varcas_load_harness.py --profile mixed || true

echo ""
echo "[7/7] Benchmark complete, detaching nsys and shutting down..."

# Stop nsys profiling (detach)
if ps -p $NSYS_PID > /dev/null; then
    echo "       Stopping nsys profiling..."
    kill -INT $NSYS_PID 2>/dev/null || true
    wait $NSYS_PID 2>/dev/null || true
fi

# Wait for nsys to finalize
echo "       Waiting for Nsight Systems to finalize profile..."
sleep 5

# Kill the server
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "=================================================="
echo "Profile Complete!"
echo "=================================================="
echo "Output Files:"
ls -lh "$OUTPUT_FILE"* 2>/dev/null || echo "  No files generated"
echo ""
if [ -f "$OUTPUT_FILE.nsys-rep" ]; then
    echo "  - $OUTPUT_FILE.nsys-rep"
    echo ""
    echo "To view the profile:"
    echo "  nsys-ui $OUTPUT_FILE.nsys-rep"
    echo ""
    echo "To analyze in CLI:"
    echo "  nsys stats $OUTPUT_FILE.nsys-rep"
    echo "  nsys stats -r cuda_gpu_kern_sum $OUTPUT_FILE.nsys-rep"
else
    echo "WARNING: Profile file not found. Check for errors above."
fi
echo "=================================================="
