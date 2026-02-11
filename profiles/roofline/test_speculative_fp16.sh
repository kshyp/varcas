#!/bin/bash
# Test speculative decoding with FP16 model

cd /home/sujatha
mkdir -p varcas/profiles/roofline/speculative_fp16_results

# Kill any existing
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3

echo "=========================================="
echo "TESTING: Speculative Decoding with FP16"
echo "Target: meta-llama/Llama-2-7b-hf"
echo "Draft: JackFram/llama-160m"
echo "=========================================="

# Check available GPU memory
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader

SPEC_CONFIG='{"method": "draft_model", "model": "JackFram/llama-160m", "num_speculative_tokens": 5}'

echo ""
echo "Starting vLLM with speculative decoding..."
echo "Speculative config: $SPEC_CONFIG"

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --dtype float16 \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --port 8000 \
  --speculative-config "$SPEC_CONFIG" > varcas/profiles/roofline/speculative_fp16_results/server.log 2>&1 &

PID=$!
echo "PID: $PID"

# Wait for ready (longer timeout for model download)
echo ""
echo "Waiting for vLLM to be ready (may download models)..."
for i in {1..180}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ vLLM ready in ${i}s"
        break
    fi
    
    if ! kill -0 $PID 2>/dev/null; then
        echo "❌ vLLM process died"
        echo ""
        echo "Last 50 lines of log:"
        tail -50 varcas/profiles/roofline/speculative_fp16_results/server.log
        exit 1
    fi
    
    sleep 1
    if [ $((i % 30)) -eq 0 ]; then
        echo "  ... still waiting (${i}s)"
    fi
done

if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ vLLM failed to start"
    tail -50 varcas/profiles/roofline/speculative_fp16_results/server.log
    exit 1
fi

# Run test
echo ""
echo "Running load test (30s)..."
python varcas/benchmark_harness/varcas_load_harness.py \
    --profile chat_medium \
    --duration 30 \
    --meaningful-prompts \
    --output varcas/profiles/roofline/speculative_fp16_results/speculative_fp16.json 2>&1 | tail -20

# Stop vLLM
kill $PID 2>/dev/null
wait $PID 2>/dev/null

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
