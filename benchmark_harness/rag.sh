#Rag Workloads (long context)
#80% context, 20% question
#Tests chunked prefill, KV-cache management

python varcas_load_harness.py --profile rag_small     # 500 token context
python varcas_load_harness.py --profile rag_medium    # 2K token context
python varcas_load_harness.py --profile rag_large     # 6K token context
python varcas_load_harness.py --profile rag_xlarge    # 12K token context


