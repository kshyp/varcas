#!/bin/bash
# Capture nsys profiles for all benchmark types with warmup
# Generates separate profiles for: chat, rag, mixed, bursty, code

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"
VLLM_PORT=8000
WARMUP_REQUESTS=15
PROFILING_DELAY=45

echo "=================================================="
echo "vLLM Nsight Systems - All Benchmark Profiles"
echo "=================================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to cleanup processes
cleanup() {
    echo "Cleaning up..."
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "nsys profile" 2>/dev/null || true
    sleep 2
}

# Function to wait for server
wait_for_server() {
    local max_wait=${1:-120}
    for i in $(seq 1 $max_wait); do
        if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

# Function to warm up server
warmup_server() {
    echo "  Warming up with $WARMUP_REQUESTS requests..."
    for i in $(seq 1 $WARMUP_REQUESTS); do
        curl -s -X POST http://localhost:$VLLM_PORT/v1/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "prompt": "Hello, how are you?",
                "max_tokens": 50,
                "temperature": 0.7
            }' > /dev/null 2>&1 &
        if [ $((i % 5)) -eq 0 ]; then
            wait
        fi
    done
    wait
    echo "  Warmup complete!"
}

# Function to run a single benchmark profile
run_benchmark_profile() {
    local profile_name=$1
    local profile_arg=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_file="$OUTPUT_DIR/vllm_nsys_${profile_name}_${timestamp}"
    
    echo ""
    echo "=================================================="
    echo "Profiling: $profile_name"
    echo "Profile: $profile_arg"
    echo "Output: ${output_file}.nsys-rep"
    echo "=================================================="
    
    # Cleanup before starting
    cleanup
    sleep 2
    
    # Start vLLM server with delayed profiling
    echo "[1/4] Starting vLLM server with ${PROFILING_DELAY}s profiling delay..."
    nsys profile -o "$output_file" \
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
    echo "  Server PID: $SERVER_PID"
    
    # Wait for server to be ready
    echo "[2/4] Waiting for server to be ready..."
    if ! wait_for_server 120; then
        echo "ERROR: Server failed to start!"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi
    echo "  Server ready!"
    
    # Wait for profiling delay to complete
    echo "[3/4] Waiting for profiling to activate..."
    sleep $PROFILING_DELAY
    echo "  Profiling is ACTIVE!"
    
    # Warmup is done implicitly during the delay, but do a few more requests
    warmup_server
    
    # Run the benchmark
    echo "[4/4] Running benchmark: $profile_arg..."
    cd "$HOME/varcas/benchmark_harness"
    python varcas_load_harness.py --profile "$profile_arg" || echo "  (benchmark completed with warnings)"
    
    # Shutdown
    echo "Shutting down server..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 3
    
    # Check if profile was created
    if [ -f "${output_file}.nsys-rep" ]; then
        local size=$(ls -lh "${output_file}.nsys-rep" | awk '{print $5}')
        echo "✓ Profile created: ${output_file}.nsys-rep ($size)"
    else
        echo "✗ WARNING: Profile file not found!"
    fi
    
    return 0
}

# Main execution
main() {
    cleanup
    
    # Define benchmarks to run
    declare -a BENCHMARKS=(
        "chat:chat_medium"
        "rag:rag_medium"  
        "mixed:mixed"
        "bursty:burst"
        "code:code_medium"
    )
    
    echo "Will run ${#BENCHMARKS[@]} benchmark profiles:"
    for bench in "${BENCHMARKS[@]}"; do
        IFS=':' read -r name arg <<< "$bench"
        echo "  - $name ($arg)"
    done
    echo ""
    read -p "Press Enter to start or Ctrl+C to cancel..."
    
    # Run each benchmark
    SUCCESS_COUNT=0
    for bench in "${BENCHMARKS[@]}"; do
        IFS=':' read -r name arg <<< "$bench"
        echo ""
        echo "Starting benchmark $((SUCCESS_COUNT + 1))/${#BENCHMARKS[@]}..."
        
        if run_benchmark_profile "$name" "$arg"; then
            ((SUCCESS_COUNT++))
        else
            echo "ERROR: Failed to run $name benchmark"
        fi
        
        # Small delay between runs
        sleep 3
    done
    
    # Summary
    echo ""
    echo "=================================================="
    echo "All Benchmarks Complete!"
    echo "=================================================="
    echo "Successfully captured: $SUCCESS_COUNT/${#BENCHMARKS[@]} profiles"
    echo ""
    echo "Generated profiles:"
    ls -lh $OUTPUT_DIR/vllm_nsys_*.nsys-rep 2>/dev/null | tail -10
    echo ""
    echo "To analyze a profile:"
    echo "  nsys stats -r cuda_gpu_kern_sum <profile.nsys-rep>"
    echo "  nsys-ui <profile.nsys-rep>"
    echo "=================================================="
}

# Run main
main "$@"
