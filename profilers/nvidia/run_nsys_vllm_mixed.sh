#!/bin/bash
set -e

# Nsight Systems Profile Script for vLLM with Mixed Workload
# This script:
#   1. Starts vLLM server with nsys profiling
#   2. Runs the mixed workload benchmark (60% chat, 30% RAG, 10% code)
#   3. Saves the profile to ~/varcas/profilers/nvidia/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_NAME="vllm_nsys_mixed_${TIMESTAMP}"
OUTPUT_FILE="$OUTPUT_DIR/$PROFILE_NAME"

# Change to project root for benchmark execution
cd "$HOME"

echo "=================================================="
echo "Nsight Systems Profile - vLLM Mixed Workload"
echo "=================================================="
echo "Output: $OUTPUT_FILE.nsys-rep"
echo ""

# Clean up any existing vLLM processes
echo "[1/5] Cleaning up existing vLLM processes..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 3

# Remove any stale profile files
rm -f "$OUTPUT_FILE.nsys-rep" "$OUTPUT_FILE.sqlite"

echo ""
echo "[2/5] Starting vLLM server with Nsight Systems profiling..."
echo "       Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0"
echo "       Config: --dtype half --max-model-len 1024 --gpu-memory-utilization 0.9"
echo ""

# Start vLLM server with nsys profiling
nsys profile -o "$OUTPUT_FILE" \
  --trace cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --gpu-metrics-device=all \
  --sample=none \
  python -m vllm.entrypoints.openai.api_server \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype half \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.9 \
  --port 8000 &

SERVER_PID=$!
echo "       Server PID: $SERVER_PID"

echo ""
echo "[3/5] Waiting for vLLM server to be ready..."
# Wait for server to be ready with timeout
MAX_WAIT=300
for i in $(seq 1 $MAX_WAIT); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
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
echo "[4/5] Running mixed workload benchmark..."
echo "       Profile: 60% chat, 30% RAG, 10% code"
echo ""

# Run the mixed benchmark
cd "$HOME/varcas/benchmark_harness"
python varcas_load_harness.py --profile mixed

echo ""
echo "[5/5] Benchmark complete, shutting down server and finalizing profile..."

# Kill the server gracefully - nsys will finalize the profile on exit
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# Wait for nsys to finalize the profile
echo "       Waiting for Nsight Systems to finalize profile..."
sleep 5

echo ""
echo "=================================================="
echo "Profile Complete!"
echo "=================================================="
echo "Output Files:"
echo "  - $OUTPUT_FILE.nsys-rep"
if [ -f "$OUTPUT_FILE.sqlite" ]; then
    echo "  - $OUTPUT_FILE.sqlite"
fi
echo ""
echo "To view the profile:"
echo "  nsys-ui $OUTPUT_FILE.nsys-rep"
echo ""
echo "To export to JSON:"
echo "  nsys export $OUTPUT_FILE.nsys-rep --type json --output $OUTPUT_FILE.json"
echo ""
echo "Key Metrics to Analyze:"
echo "  - GPU utilization timeline"
echo "  - CUDA kernel execution patterns"
echo "  - Memory allocation/deallocation"
echo "  - API call latency distribution"
echo "=================================================="
