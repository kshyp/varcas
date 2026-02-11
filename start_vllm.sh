
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-2-7B-AWQ \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --quantization awq \
  --enforce-eager \
  --port 8000
