#!/bin/bash
set -e

# Nsight Systems Profile Script for vLLM with Delayed Profiling
# This script:
#   1. Starts vLLM server with nsys profiling DELAYED
#   2. Server warms up naturally during the delay period
#   3. Profiling starts automatically after delay
#   4. Runs the mixed workload benchmark
#   5. Saves the profile to ~/varcas/profilers/nvidia/
#
# This approach captures only the actual inference workload,
# excluding model loading and CUDA graph compilation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_NAME="vllm_nsys_delayed_${TIMESTAMP}"
OUTPUT_FILE="$OUTPUT_DIR/$PROFILE_NAME"

# Configuration
VLLM_PORT=8000
# Delay in seconds - should be long enough for server startup + warmup
# Model load (~20s) + CUDA graphs (~10s) + buffer (~15s) = ~45s minimum
PROFILING_DELAY=60

echo "=================================================="
echo "Nsight Systems Profile - vLLM (Delayed Capture)"
echo "=================================================="
echo "Output: $OUTPUT_FILE.nsys-rep"
echo "Profiling Delay: ${PROFILING_DELAY}s (for warmup)"
echo ""

# Clean up any existing vLLM processes
echo "[1/4] Cleaning up existing vLLM processes..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -f "nsys profile" 2>/dev/null || true
sleep 3 || true

# Remove any stale profile files
rm -f "$OUTPUT_FILE.nsys-rep" "$OUTPUT_FILE.sqlite" || true

echo ""
echo "[2/4] Starting vLLM server with DELAYED profiling..."
echo "       Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0"
echo "       Profiling will start after ${PROFILING_DELAY}s"
echo "       (Server will be ready and warmed up by then)"
echo ""

# Start vLLM server with nsys profiling delayed
nsys profile -o "$OUTPUT_FILE" \
  --trace cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --gpu-metrics-device=all \
  --sample=none \
  --delay=$PROFILING_DELAY \
  --force-overwrite=true \
  python -m vllm.entrypoints.openai.api_server \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype half \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.9 \
  --port $VLLM_PORT &

SERVER_PID=$!
echo "       Server PID: $SERVER_PID"
echo "       Nsys is waiting ${PROFILING_DELAY}s before starting capture..."

echo ""
echo "[3/4] Waiting for server to be ready (during delay period)..."
MAX_WAIT=300
for i in $(seq 1 $MAX_WAIT); do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "       Server is ready! (took ${i}s)"
        echo "       Profiling will start in $((PROFILING_DELAY - i))s..."
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

# Wait for profiling to actually start
SECONDS_ELAPSED=$i
REMAINING_DELAY=$((PROFILING_DELAY - SECONDS_ELAPSED))
if [ $REMAINING_DELAY -gt 0 ]; then
    echo "       Waiting additional ${REMAINING_DELAY}s for profiling to start..."
    sleep $REMAINING_DELAY
fi

echo ""
echo "       Profiling should now be ACTIVE!"
echo ""
echo "[4/4] Running mixed workload benchmark..."
echo "       Profile: 60% chat, 30% RAG, 10% code"
echo ""

cd "$HOME/varcas/benchmark_harness"
python varcas_load_harness.py --profile mixed || true

echo ""
echo "Benchmark complete, shutting down server..."

# Kill the server - nsys will finalize the profile on exit
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# Wait for nsys to finalize
echo "       Waiting for Nsight Systems to finalize profile..."
sleep 5 || true

echo ""
echo "=================================================="
echo "Profile Complete!"
echo "=================================================="
echo "Output Files:"
ls -lh "$OUTPUT_FILE"* 2>/dev/null || echo "  No files found"
echo ""
if [ -f "$OUTPUT_FILE.nsys-rep" ]; then
    echo "  Profile: $OUTPUT_FILE.nsys-rep"
    echo ""
    echo "To view the profile:"
    echo "  nsys-ui $OUTPUT_FILE.nsys-rep"
    echo ""
    echo "To analyze in CLI:"
    echo "  nsys stats $OUTPUT_FILE.nsys-rep"
    echo "  nsys stats -r cuda_gpu_kern_sum $OUTPUT_FILE.nsys-rep"
fi
echo "=================================================="
