# Install
pip install aiohttp numpy

# Basic chat test
python varcas_load_harness.py --profile chat_medium --duration 60 --max-context 761

# Your 2048+ context RAG test
python varcas_load_harness.py --profile rag_large --duration 120

# Bursty traffic
python varcas_load_harness.py --profile burst --duration 120

# Customer trace replay
python varcas_load_harness.py --trace customer.json --duration 300

## examples.sh


# 1. Basic chat workload (medium intensity)
python varcas_load_harness.py --profile chat_medium --duration 60

# 2. RAG workload with large context (4K-8K tokens)
python varcas_load_harness.py --profile rag_large --duration 120

# 3. Bursty traffic (Hawkes process)
python varcas_load_harness.py --profile burst --duration 120

# 4. Mixed workload (chat + RAG + code)
python varcas_load_harness.py --profile mixed --duration 120

# 5. Closed-loop with fixed concurrency
python varcas_load_harness.py --profile closed_loop --duration 60

# 6. Custom vLLM endpoint
python varcas_load_harness.py --url http://192.168.1.100:8000 --profile chat_high

# 7. Replay from customer trace
python varcas_load_harness.py --trace customer_trace.json --duration 300

# 8. Save to specific file
python varcas_load_harness.py --profile rag_xlarge --output rag_stress_test.json
