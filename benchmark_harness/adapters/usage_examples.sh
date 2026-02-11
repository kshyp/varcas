# Convert MLPerf dataset to harness profile
python varcas_adapters.py --format mlperf --input mlperf_dataset.jsonl --scenario server --output mlperf_profile.json

#Scenarios:
#single_stream: Sequential, measure latency
#multi_stream: Fixed rate concurrent
#server: Poisson arrival, latency constraint
#offline: Batch throughput


# Use in harness
python varcas_load_harness.py --profile mlperf_profile.json --duration 300

# Replicate vLLM's synthetic benchmark
python varcas_adapters.py --format vllm --target-qps 20 --num-prompts 1000 --output vllm_bench.json

# Or with ShareGPT dataset
python varcas_adapters.py --format vllm --input sharegpt.json --target-qps 10 --output vllm_sharegpt.json

# Convert ShareGPT to profile
python varcas_adapters.py --format sharegpt --input sharegpt.json --output sharegpt_profile.json

# Extract ground truth for validation
python -c "
from varcas_adapters import ShareGPTAdapter
import json
gt = ShareGPTAdapter.to_ground_truth('sharegpt.json', 100)
json.dump(gt, open('sharegpt_ground_truth.json', 'w'), indent=2)
"

# Azure trace (timestamps in milliseconds)
python varcas_adapters.py --format azure --input azure_trace.csv --time-unit ms --output azure_profile.json

# Azure trace (timestamps in seconds)
python varcas_adapters.py --format azure --input azure_trace.csv --time-unit s --output azure_profile.json

