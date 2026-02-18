#Rag Workloads (long context)
#80% context, 20% question
#Tests chunked prefill, KV-cache management

python varcas_load_harness.py --profile rag_small --meaningful-prompts     # 500 token context
python varcas_load_harness.py --profile rag_medium --meaningful-prompts   # 2K token context
python varcas_load_harness.py --profile rag_large --meaningful-prompts    # 6K token context
python varcas_load_harness.py --profile rag_xlarge --meaningful-prompts   # 12K token context


