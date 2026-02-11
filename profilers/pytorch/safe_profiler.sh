#!/bin/bash
# Safe PyTorch Profiler for vLLM - Avoids threading issues
# This script uses continuous profiling (no start/stop API calls)
# to avoid the NCCL threading error

set -e

OUTPUT_DIR="${1:-/home/sujatha/varcas/profilers/pytorch/output/safe_profile_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Safe PyTorch Profiler for vLLM"
echo "=========================================="
echo "Output: $OUTPUT_DIR"
echo ""
echo "NOTE: This script avoids NCCL threading issues by:"
echo "  1. Using continuous profiling (no API calls)"
echo "  2. Disabling CUDA graphs (--enforce-eager)"
echo "  3. Disabling stack tracing (causes thread issues)"
echo ""

# Kill any existing vLLM
echo "Cleaning up existing vLLM processes..."
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
sleep 3

# Start vLLM with built-in profiler enabled
# Key settings to avoid crashes:
# - enforce-eager: Disables CUDA graphs (can interfere with profiler)
# - torch_profiler_with_stack=false: Avoids thread safety issues
# - NO start_profile/stop_profile API calls - profiling runs continuously
echo "Starting vLLM with PyTorch profiler enabled..."
echo "  - Profiling will run continuously during benchmark"
echo "  - Traces saved to: $OUTPUT_DIR"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dtype half \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.8 \
    --port 8000 \
    --enforce-eager \
    --max-num-seqs 64 \
    --profiler-config.torch_profiler_dir="$OUTPUT_DIR" \
    --profiler-config.torch_profiler_with_stack=false \
    --profiler-config.torch_profiler_use_gzip=true \
    2>&1 | tee "$OUTPUT_DIR/vllm_server.log" &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for vLLM to be ready
echo ""
echo "Waiting for vLLM to be ready..."
for i in {1..180}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ vLLM is ready! (took ${i}s)"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "✗ Timeout waiting for vLLM"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Verify profiler endpoints exist (optional, for info only)
echo ""
echo "Checking available endpoints..."
curl -s http://localhost:8000/v1/models > /dev/null 2>&1 && echo "  ✓ API server responding" || echo "  ✗ API not responding"

# Warmup with short requests
echo ""
echo "Sending warmup requests..."
for i in {1..5}; do
    curl -s -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "prompt": "Hi", "max_tokens": 10}' \
        > /dev/null 2>&1 && echo "  Warmup $i/5 done" || echo "  Warmup $i/5 failed"
    sleep 0.5
done

# Run benchmark
echo ""
echo "========================================"
echo "Running Mixed Benchmark (120 seconds)"
echo "========================================"
echo "Profiling is running continuously..."
echo ""

cd /home/sujatha/varcas/benchmark_harness
timeout 130 bash mixed.sh 2>&1 | tee "$OUTPUT_DIR/benchmark_output.log" || echo "Benchmark completed or timed out"

echo ""
echo "========================================"
echo "Benchmark complete!"
echo "========================================"

# Wait for traces to be written
echo ""
echo "Waiting for profiler to flush traces..."
sleep 5

# Stop vLLM - this will also trigger final trace save
echo ""
echo "Stopping vLLM (this may take a moment for trace finalization)..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

# Check results
echo ""
echo "=========================================="
echo "Profiling Complete!"
echo "=========================================="
echo "Output: $OUTPUT_DIR"
echo ""

# List all files
echo "Generated files:"
ls -lh "$OUTPUT_DIR/" 2>/dev/null || echo "  (no files)"

# Look for trace files
echo ""
echo "Trace files:"
find "$OUTPUT_DIR" -type f \( -name "*.json*" -o -name "*.trace*" -o -name "*.pt*" -o -name "*.gz" \) 2>/dev/null | while read f; do
    size=$(ls -lh "$f" 2>/dev/null | awk '{print $5}')
    echo "  - $(basename $f) ($size)"
done

# Check for the specific vLLM trace files
echo ""
echo "vLLM profiler traces (if any):"
find "$OUTPUT_DIR" -name "*.pt.trace.json*" -o -name "*chrome*" 2>/dev/null | head -10

echo ""
echo "To view traces:"
echo "  1. Open Chrome and navigate to chrome://tracing"
echo "  2. Load the .json or .json.gz files from $OUTPUT_DIR"
echo "  3. Or use https://ui.perfetto.dev/"
echo ""
echo "NOTE: If no trace files were generated, check vllm_server.log for errors."
echo "=========================================="
