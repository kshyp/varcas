#!/bin/bash
# Simple NCU profiling for vLLM with chat benchmark
# Usage: ./run_ncu_profile_simple.sh [profile_name]

set -e

PROFILE_NAME=${1:-chat_medium}
OUTPUT_DIR="$HOME/varcas/profilers/nvidia"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/vllm_ncu_${PROFILE_NAME}_${TIMESTAMP}"
VLLM_PORT=8000
BENCHMARK_DURATION=20

# Configuration
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DTYPE="half"
MAX_MODEL_LEN=1024
GPU_MEM_UTIL=0.7

echo "═══════════════════════════════════════════════════════════════"
echo "  NCU Profiling: vLLM + $PROFILE_NAME (Limited)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Cleanup
cleanup() {
    echo ""
    echo "Cleaning up..."
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 2
}
trap cleanup EXIT

pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

echo "Step 1: Starting vLLM server..."

# Start vLLM normally first
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --dtype $DTYPE \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --port $VLLM_PORT > /tmp/vllm_ncu.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server
echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1 && break
    sleep 1
done
echo "✓ Server ready"

# Warmup
curl -s http://localhost:$VLLM_PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "'"$MODEL"'", "prompt": "Hello", "max_tokens": 10}' > /dev/null 2>&1
echo "✓ Warmup complete"

echo ""
echo "Step 2: Running NCU profiling with benchmark..."
echo "This will profile kernels during ${BENCHMARK_DURATION}s of benchmark execution"
echo ""

# Create a benchmark runner script
cat > /tmp/run_benchmark_simple.py << 'EOF'
import sys
import os
sys.path.insert(0, os.path.expanduser('~/varcas/benchmark_harness'))

import asyncio
from varcas_load_harness import LoadHarness, get_chat_profile, get_rag_profile, get_code_profile, get_mixed_profile, get_burst_profile

async def main():
    profile_name = sys.argv[1] if len(sys.argv) > 1 else "chat_medium"
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    # Get profile
    if profile_name.startswith('chat_'):
        intensity = profile_name.split('_')[1]
        profile = get_chat_profile(intensity)
    elif profile_name == 'mixed':
        profile = get_mixed_profile()
    elif profile_name == 'burst':
        profile = get_burst_profile()
    else:
        profile = get_chat_profile('medium')
    
    profile.duration_seconds = duration
    
    print(f"Running {profile.name} benchmark for {duration}s...")
    
    harness = LoadHarness("http://localhost:8000")
    async with harness:
        result = await harness.run(profile)
    
    print(f"\nCompleted: {result.total_requests} requests")
    print(f"Success rate: {result.successful_requests / result.total_requests * 100:.1f}%")
    print(f"Throughput: {result.throughput_rps:.2f} req/s")
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
EOF

cd "$HOME/varcas/benchmark_harness"

# Run benchmark with NCU - limit to first 100 kernel launches to keep file size manageable
# This captures enough kernels for analysis without making the profile too large
timeout 120 ncu -f -o "$OUTPUT_FILE" \
    --target-processes all \
    --set basic \
    --launch-count 100 \
    python /tmp/run_benchmark_simple.py "$PROFILE_NAME" $BENCHMARK_DURATION || true

echo ""
echo "Step 3: Finalizing..."

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
sleep 3

# Report results
if [ -f "${OUTPUT_FILE}.ncu-rep" ]; then
    SIZE=$(ls -lh "${OUTPUT_FILE}.ncu-rep" | awk '{print $5}')
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ✓ NCU Profile Created!"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Profile: ${OUTPUT_FILE}.ncu-rep ($SIZE)"
    echo ""
    echo "  View:   ncu-ui ${OUTPUT_FILE}.ncu-rep"
    echo "  CSV:    ncu --import ${OUTPUT_FILE}.ncu-rep --page raw --csv > ${OUTPUT_FILE}.csv"
    echo "  Summary: ncu --import ${OUTPUT_FILE}.ncu-rep --print-summary per-kernel"
    echo "═══════════════════════════════════════════════════════════════"
else
    echo "⚠ Profile file not created (this can happen if no kernels were captured)"
    echo ""
    echo "Try running with nsys instead:"
    echo "  ./run_nsys_vllm_simple.sh"
fi
