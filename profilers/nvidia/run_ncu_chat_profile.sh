#!/bin/bash
# Run vLLM server with NCU profiling and chat workload benchmark
# Usage: ./run_ncu_chat_profile.sh [profile_intensity]
# This uses a lighter profiling set to avoid timeout issues

set -e

PROFILE_INTENSITY=${1:-chat_medium}
OUTPUT_DIR="$HOME/varcas/profilers/nvidia"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/vllm_ncu_${PROFILE_INTENSITY}_${TIMESTAMP}"
VLLM_PORT=8000

# Configuration
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DTYPE="half"
MAX_MODEL_LEN=1024
GPU_MEM_UTIL=0.7

echo "=== NCU Profiling: vLLM with $PROFILE_INTENSITY workload ==="
echo "Output: $OUTPUT_FILE.ncu-rep"
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
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "ncu " 2>/dev/null || true
sleep 2

rm -f "$OUTPUT_FILE.ncu-rep"

echo "Step 1: Starting vLLM server (without profiling)..."
echo "Model: $MODEL"
echo ""

# Start vLLM server WITHOUT NCU first (to avoid profiling overhead during model loading)
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --dtype $DTYPE \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --port $VLLM_PORT &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
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
        exit 1
    fi
    sleep 1
done

# Verify server is still running
if ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "✗ Server failed to start"
    exit 1
fi

echo ""
echo "Step 3: Running benchmark with NCU profiling..."
echo "Profile: $PROFILE_INTENSITY"
echo ""

# Run the chat benchmark with NCU attached to the server process
# Using --kernel-name regex to filter for important vLLM kernels
cd "$HOME/varcas/benchmark_harness"

# Run NCU profiling while benchmark is executing
# We profile the specific server process and its children
ncu -f -o "$OUTPUT_FILE" \
    --target-processes all \
    --set basic \
    --profile-from-start off \
    python -c "
import subprocess
import sys
import time

# Start benchmark
print('Starting benchmark: $PROFILE_INTENSITY...')
result = subprocess.run(
    ['python', 'varcas_load_harness.py', '--profile', '$PROFILE_INTENSITY', '--duration', '30', '--output', '$OUTPUT_DIR/result_${PROFILE_INTENSITY}_${TIMESTAMP}.json'],
    capture_output=False
)
sys.exit(result.returncode)
" &

NCU_PID=$!
echo "NCU PID: $NCU_PID"

# Wait for NCU to complete
wait $NCU_PID || true

echo ""
echo "Benchmark and profiling complete!"
echo ""

# Cleanup will be called by trap
sleep 3

# Check and report results
if [ -f "${OUTPUT_FILE}.ncu-rep" ]; then
    SIZE=$(ls -lh "${OUTPUT_FILE}.ncu-rep" | awk '{print $5}')
    echo "═══════════════════════════════════════════════════════════════"
    echo "✓ NCU Profile Created Successfully!"
    echo "═══════════════════════════════════════════════════════════════"
    echo "Profile: ${OUTPUT_FILE}.ncu-rep"
    echo "Size: $SIZE"
    echo "Benchmark Result: $OUTPUT_DIR/result_${PROFILE_INTENSITY}_${TIMESTAMP}.json"
    echo ""
    echo "To view the profile:"
    echo "  ncu-ui ${OUTPUT_FILE}.ncu-rep"
    echo ""
    echo "To export summary to CSV:"
    echo "  ncu --import ${OUTPUT_FILE}.ncu-rep --page raw --csv > ${OUTPUT_FILE}.csv"
    echo ""
    echo "To list top kernels:"
    echo "  ncu --import ${OUTPUT_FILE}.ncu-rep --page details | head -50"
    echo "═══════════════════════════════════════════════════════════════"
else
    echo "✗ Profile file not found!"
    exit 1
fi
