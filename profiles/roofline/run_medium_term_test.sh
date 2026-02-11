#!/bin/bash
set -e

cd /home/sujatha
RESULTS_DIR="varcas/profiles/roofline/medium_term_results"
mkdir -p "$RESULTS_DIR"

# Function to run test
run_test() {
    local name=$1
    local script=$2
    
    echo ""
    echo "=========================================="
    echo "TESTING: $name"
    echo "=========================================="
    
    # Kill any existing vLLM
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 2
    
    # Start vLLM
    echo "Starting vLLM..."
    bash "$script" > "$RESULTS_DIR/${name}_server.log" 2>&1 &
    VLLM_PID=$!
    
    # Wait for ready
    for i in {1..120}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "vLLM ready in ${i}s"
            break
        fi
        sleep 1
    done
    
    # Check if started
    if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "ERROR: vLLM failed to start"
        kill $VLLM_PID 2>/dev/null || true
        return 1
    fi
    
    sleep 3
    
    # Run test
    echo "Running load test..."
    python varcas/benchmark_harness/varcas_load_harness.py \
        --profile chat_medium \
        --duration 40 \
        --meaningful-prompts \
        --output "$RESULTS_DIR/${name}.json" 2>&1 | tail -30
    
    # Stop vLLM
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    
    echo "$name test complete"
    return 0
}

# Test 1: Optimized Baseline (batch=8)
echo ""
echo "############################################"
echo "# TEST 1: OPTIMIZED BASELINE (batch=8)     #"
echo "############################################"
if ! run_test "baseline" "start_vllm_optimized.sh"; then
    echo "Baseline test failed, continuing..."
fi

sleep 3

# Test 2: Advanced (with medium-term optimizations)
echo ""
echo "############################################"
echo "# TEST 2: ADVANCED (medium-term wins)      #"
echo "############################################"
if ! run_test "advanced" "start_vllm_advanced.sh"; then
    echo "Advanced test failed"
fi

echo ""
echo "=========================================="
echo "All tests complete!"
echo "Results in: $RESULTS_DIR"
echo "=========================================="
