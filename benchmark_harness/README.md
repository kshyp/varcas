# Varcas Load Harness v1.0

A production-grade load generator for vLLM performance testing and optimization.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start vLLM (example)
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-model-len 2048

# Run basic test
python varcas_load_harness.py --profile chat_medium --duration 60

# Test RAG with long context
python varcas_load_harness.py --profile rag_large --duration 120
