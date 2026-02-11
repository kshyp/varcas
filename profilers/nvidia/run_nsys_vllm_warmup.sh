#!/bin/bash
set -e

# Nsight Systems Profile Script for vLLM - Profiles the SERVER
# This script:
#   1. Starts vLLM server with nsys profiling (but delayed)
#   2. Server warms up during the delay period
#   3. Profiling starts automatically after delay
#   4. Runs the mixed workload benchmark
#   5. Saves the profile to ~/varcas/profilers/nvidia/
#
# This captures CUDA kernels, memory ops, etc. from the vLLM server process.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_NAME="vllm_nsys_server_warmup_${TIMESTAMP}"
OUTPUT_FILE="$OUTPUT_DIR/$PROFILE_NAME"

# Configuration
VLLM_PORT=8000
# Delay in seconds - server warms up during this time
# Model load (~20s) + CUDA graphs (~10s) + buffer (~15s) = ~45s minimum
PROFILING_DELAY=60

echo "=================================================="
echo "Nsight Systems Profile - vLLM SERVER (with warmup)"
echo "=================================================="
echo "Output: $OUTPUT_FILE.nsys-rep"
echo "Delay: ${PROFILING_DELAY}s (for server warmup)"
echo ""

# Clean up any existing vLLM processes
echo "[1/5] Cleaning up existing vLLM processes..."
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "nsys profile" 2>/dev/null || true
sleep 2

# Remove any stale profile files
rm -f "$OUTPUT_FILE.nsys-rep" "$OUTPUT_FILE.sqlite"

echo ""
echo "[2/5] Starting vLLM SERVER with DELAYED profiling..."
echo "       Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0"
echo "       Profiling starts after ${PROFILING_DELAY}s"
echo "       (Server warms up during this time)"
echo ""

# Start vLLM SERVER with nsys profiling delayed
# This profiles the SERVER process where GPU work happens
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

echo ""
echo "[3/5] Waiting for server to be ready (during warmup period)..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "       Server ready after ${i}s"
        break
    fi
    sleep 1
done

# Wait for profiling to start
REMAINING=$((PROFILING_DELAY - i))
if [ $REMAINING -gt 0 ]; then
    echo "       Waiting ${REMAINING}s for profiling to start..."
    sleep $REMAINING
fi

echo "       Profiling is now ACTIVE on the SERVER!"

echo ""
echo "[4/5] Running mixed workload benchmark..."
echo "       (This will trigger GPU kernels in the server)"
echo ""

cd "$HOME/varcas/benchmark_harness"
python varcas_load_harness.py --profile mixed

echo ""
echo "[5/5] Shutting down server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
sleep 3

echo ""
echo "=================================================="
echo "Profile Complete!"
echo "=================================================="
ls -lh "$OUTPUT_FILE"* 2>/dev/null
echo ""
echo "This profile contains SERVER data:"
echo "  - CUDA kernels (flashinfer, gemm, etc.)"
echo "  - GPU memory operations"
echo "  - NVTX ranges from vLLM"
echo ""
echo "To view:   nsys-ui $OUTPUT_FILE.nsys-rep"
echo "To analyze: nsys stats -r cuda_gpu_kern_sum $OUTPUT_FILE.nsys-rep"
echo "=================================================="
