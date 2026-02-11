#!/bin/bash
cd /home/sujatha
mkdir -p varcas/profiles/roofline/speculative_results

# Kill any existing
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3

echo "=========================================="
echo "TESTING: Speculative Decoding"
echo "Draft Model: JackFram/llama-160m"
echo "Target Model: TheBloke/Llama-2-7B-AWQ"
echo "=========================================="

# Start vLLM with speculative decoding
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-2-7B-AWQ \
  --dtype half \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --quantization awq \
  --enforce-eager \
  --port 8000 \
  --speculative-model JackFram/llama-160m \
  --num-speculative-tokens 5 > varcas/profiles/roofline/speculative_results/server.log 2>&1 &

PID=$!
echo "Started vLLM (PID: $PID)"

# Wait for ready with longer timeout (downloading draft model)
echo "Waiting for vLLM to be ready (may download draft model)..."
for i in {1..180}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ vLLM ready in ${i}s"
        break
    fi
    
    # Check if process died
    if ! kill -0 $PID 2>/dev/null; then
        echo "❌ vLLM process died"
        cat varcas/profiles/roofline/speculative_results/server.log | tail -50
        exit 1
    fi
    
    sleep 1
    if [ $((i % 30)) -eq 0 ]; then
        echo "  ... still waiting (${i}s)"
    fi
done

# Check if started
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ vLLM failed to start"
    cat varcas/profiles/roofline/speculative_results/server.log | tail -50
    exit 1
fi

# Run test
echo ""
echo "Running load test..."
python varcas/benchmark_harness/varcas_load_harness.py \
    --profile chat_medium \
    --duration 35 \
    --meaningful-prompts \
    --output varcas/profiles/roofline/speculative_results/speculative.json 2>&1 | tail -20

# Stop vLLM
kill $PID 2>/dev/null
wait $PID 2>/dev/null

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
