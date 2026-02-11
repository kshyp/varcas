#!/bin/bash
cd /home/sujatha

# Start optimized vLLM
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-2-7B-AWQ \
  --dtype half \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --quantization awq \
  --enforce-eager \
  --port 8000 > varcas/profiles/roofline/easy_wins_results/optimized_server.log 2>&1 &

VLLM_PID=$!
echo "Started vLLM optimized (PID: $VLLM_PID)"

# Wait for ready
for i in {1..120}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM ready in ${i}s"
        break
    fi
    sleep 1
done

# Run test
sleep 3
python varcas/benchmark_harness/varcas_load_harness.py \
  --profile chat_medium \
  --duration 40 \
  --meaningful-prompts \
  --output varcas/profiles/roofline/easy_wins_results/optimized.json

# Stop vLLM
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "Optimized test complete"
