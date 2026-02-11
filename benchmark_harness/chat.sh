#Chat workloads
#Short inputs (10-300 tokens)
#Medium outputs (20-800 tokens)
#Poisson arrival

python varcas_load_harness.py --profile chat_low  --meaningful-prompts    # 5 RPS
python varcas_load_harness.py --profile chat_medium --meaningful-prompts  # 20 RPS
python varcas_load_harness.py --profile chat_high --meaningful-prompts    # 50 RPS
