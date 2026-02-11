#!/bin/bash
# Run vLLM server and profile benchmark execution with NCU
# This starts server normally, then profiles only the benchmark execution
# Usage: ./run_ncu_attach_profile.sh [profile_intensity]

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

echo "=== NCU Attach Profiling: vLLM with $PROFILE_INTENSITY workload ==="
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
sleep 2

rm -f "$OUTPUT_FILE.ncu-rep"

echo "Step 1: Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --dtype $DTYPE \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --port $VLLM_PORT > /tmp/vllm_server.log 2>&1 &

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
        cat /tmp/vllm_server.log
        exit 1
    fi
    sleep 1
done

# Verify server is still running
if ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "✗ Server failed to start"
    cat /tmp/vllm_server.log
    exit 1
fi

echo ""
echo "Step 3: Running warmup request..."
# Warmup to ensure CUDA context is initialized
curl -s http://localhost:$VLLM_PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "'"$MODEL"'", "prompt": "Hello", "max_tokens": 10}' > /dev/null 2>&1 || true
echo "✓ Warmup complete"

echo ""
echo "Step 4: Running benchmark with NCU profiling..."
echo "Profile: $PROFILE_INTENSITY"
echo ""

cd "$HOME/varcas/benchmark_harness"

# Create a Python script that runs the benchmark and can be profiled
cat > /tmp/run_benchmark.py << 'EOF'
import asyncio
import sys
import os
sys.path.insert(0, os.getcwd())

from varcas_load_harness import LoadHarness, get_chat_profile, get_rag_profile, get_code_profile, get_mixed_profile, get_burst_profile, get_closed_loop_profile

async def main():
    profile_name = sys.argv[1] if len(sys.argv) > 1 else "chat_medium"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "result.json"
    
    # Get profile
    if profile_name.startswith('chat_'):
        intensity = profile_name.split('_')[1]
        profile = get_chat_profile(intensity)
    elif profile_name.startswith('rag_'):
        parts = profile_name.split('_')
        size = parts[1]
        intensity = parts[2] if len(parts) > 2 else 'medium'
        profile = get_rag_profile(size, intensity)
    elif profile_name.startswith('code_'):
        intensity = profile_name.split('_')[1]
        profile = get_code_profile(intensity)
    elif profile_name == 'mixed':
        profile = get_mixed_profile()
    elif profile_name == 'burst':
        profile = get_burst_profile()
    elif profile_name == 'closed_loop':
        profile = get_closed_loop_profile(concurrency=10)
    else:
        profile = get_chat_profile('medium')
    
    # Use shorter duration for profiling
    profile.duration_seconds = 30
    
    print(f"Running benchmark: {profile.name}")
    print(f"Duration: {profile.duration_seconds}s")
    
    harness = LoadHarness("http://localhost:8000")
    async with harness:
        result = await harness.run(profile)
    
    print(f"\nCompleted: {result.total_requests} requests")
    print(f"Success rate: {result.successful_requests / result.total_requests * 100:.1f}%")
    print(f"Throughput: {result.throughput_rps:.2f} req/s")
    
    result.save(output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
EOF

# Run the benchmark script with NCU profiling
ncu -f -o "$OUTPUT_FILE" \
    --target-processes all \
    --set basic \
    python /tmp/run_benchmark.py "$PROFILE_INTENSITY" "$OUTPUT_DIR/result_${PROFILE_INTENSITY}_${TIMESTAMP}.json"

echo ""
echo "Benchmark and profiling complete!"
echo ""

# Cleanup will be called by trap
sleep 2

# Check and report results
if [ -f "${OUTPUT_FILE}.ncu-rep" ]; then
    SIZE=$(ls -lh "${OUTPUT_FILE}.ncu-rep" | awk '{print $5}')
    echo "═══════════════════════════════════════════════════════════════"
    echo "✓ NCU Profile Created Successfully!"
    echo "═══════════════════════════════════════════════════════════════"
    echo "Profile:       ${OUTPUT_FILE}.ncu-rep"
    echo "Size:          $SIZE"
    echo "Benchmark:     $OUTPUT_DIR/result_${PROFILE_INTENSITY}_${TIMESTAMP}.json"
    echo "Server Log:    /tmp/vllm_server.log"
    echo ""
    echo "To view the profile:"
    echo "  ncu-ui ${OUTPUT_FILE}.ncu-rep"
    echo ""
    echo "To export to CSV:"
    echo "  ncu --import ${OUTPUT_FILE}.ncu-rep --page raw --csv > ${OUTPUT_FILE}.csv"
    echo ""
    echo "To view kernel summary:"
    echo "  ncu --import ${OUTPUT_FILE}.ncu-rep --print-summary per-kernel"
    echo "═══════════════════════════════════════════════════════════════"
else
    echo "✗ Profile file not found!"
    exit 1
fi
