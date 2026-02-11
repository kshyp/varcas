from varcas_adapters import (
    MLPerfAdapter, ShareGPTAdapter, AzureTracesAdapter,
    vLLMBenchmarkAdapter, load_external_format
)

# MLPerf server scenario
mlperf_profile = MLPerfAdapter.from_dataset(
    "mlperf_data.jsonl",
    scenario="server",
    target_latency_ms=500
)

# ShareGPT with statistics
sharegpt_profile = ShareGPTAdapter.from_file(
    "sharegpt.json",
    max_conversations=500
)

# Azure production trace
azure_profile = AzureTracesAdapter.from_file(
    "azure_trace.csv",
    time_unit="ms"
)

# vLLM benchmark replication
vllm_profile = vLLMBenchmarkAdapter.replicate(
    dataset_type="synthetic",
    target_qps=25.0,
    num_prompts=2000
)

# Universal loader
profile = load_external_format(
    "trace.csv",
    "azure",
    time_unit="s"
)

# Run with harness
async with LoadHarness() as harness:
    result = await harness.run(azure_profile)
