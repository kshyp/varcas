# ConfigIQ: Hardware Sizing Tool for LLM Inference

A comprehensive tool to calculate optimal hardware configurations for LLM inference workloads on vLLM, using static roofline analysis and M/G/1 queuing models.

## Features

- **Model Support**: Works with any HuggingFace Transformers model (Llama, Mistral, Qwen, DeepSeek, etc.)
- **Workload Types**: Chat, RAG, and Code generation with industry-standard SLA targets
- **Multi-GPU Support**: Tensor parallelism (TP) with realistic NVLink/PCIe interconnect modeling
- **Cost Optimization**: Sorts recommendations by cost per request
- **SLA Prediction**: P50/P95/P99 latency predictions using roofline + queuing models
- **Cloud Providers**: GCP VM catalog with extensible architecture for AWS/Azure

## Architecture

```
config_iq/
├── core/
│   ├── types.py              # Data classes and enums
│   ├── hardware_catalog.py   # Cloud VM/GPU specifications
│   ├── workload_patterns.py  # Traffic patterns & SLA targets
│   ├── model_profiler.py     # HF model analysis
│   ├── roofline_analyzer.py  # Static roofline analysis
│   ├── queuing_model.py      # M/G/1 queuing model
│   └── sla_calculator.py     # Recommendation engine
├── data/
│   ├── gcp_catalog.json      # GCP VM instances
│   └── workload_defaults.json # SLA targets & patterns
└── cli/
    └── main.py               # CLI interface
```

## Installation

```bash
cd varcas/config_iq
pip install -e .
```

## Usage

### Basic Usage

```bash
python -m config_iq.cli.main \
  --model meta-llama/Llama-2-7b-hf \
  --workload chat \
  --users 100
```

### Advanced Options

```bash
python -m config_iq.cli.main \
  --model meta-llama/Llama-2-70b-hf \
  --workload rag \
  --users 50 \
  --headroom 50 \
  --quantization fp16 \
  --pricing ondemand \
  --top 5 \
  --output results.json \
  --zone us-central1-a
```

### Zone/Region Filtering

Different zones have different GPU availability. Use `--zone` or `--region` to filter:

```bash
# List all available zones
python -m config_iq.cli.main --list-zones

# Filter by specific zone
python -m config_iq.cli.main --model meta-llama/Llama-2-7b-hf --users 100 --zone us-central1-a

# Filter by region (includes all zones in that region)
python -m config_iq.cli.main --model meta-llama/Llama-2-7b-hf --users 100 --region us-west1
```

**Note**: High-end GPUs (A100, H100) have limited zone availability:
- A100: us-central1, us-west1, europe-west4
- H100: us-central1, us-west1 (very limited)
- L4: All regions (widely available)

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | HuggingFace model ID | Required |
| `--workload` | Workload type: chat, rag, code | chat |
| `--users` | Concurrent users to support | Required |
| `--headroom` | Headroom % (0, 25, 50, 100) | 25 |
| `--quantization` | Quantization: fp16, bf16, int8, fp8, int4 | fp16 |
| `--pricing` | Pricing model: ondemand, spot | ondemand |
| `--zone` | GCP zone (e.g., us-central1-a) | All zones |
| `--region` | GCP region (e.g., us-central1) | All regions |
| `--top` | Show top N recommendations | 3 |
| `--output` | Save results to JSON | - |
| `--list-zones` | List available zones and exit | - |

## Output Format

### SLA Compliance Report

```
================================================================================
SLA COMPLIANCE REPORT
================================================================================

Configuration: a2-highgpu-1g
GPU: nvidia-a100-40gb x 1
Tensor Parallel: 1
Instances needed: 2
Headroom: 50%

Metric               Target          Predicted       Status     Gap       
----------------------------------------------------------------------
TTFT P50             200             12              ✓          -94.2%    
TTFT P95             —               51              ✓          -89.9%    
TTFT P99             —               78              ✓          -92.2%    
TPOT P50             50              3               ✓          -93.8%    
TPOT P95             —               13              ✓          -83.2%    
E2E Latency          2000            478             ✓          -76.1%    

Tail Latency Analysis:
  TTFT P95: 51 (4.3x P50) - ~ Acceptable
  TTFT P99: 78 (6.6x P50) - ⚠ High Variance
```

### Throughput Analysis

```
================================================================================
THROUGHPUT ANALYSIS
================================================================================

Throughput at SLA point:
  • 314 tok/s
  • 2.1 req/s
  • 100% GPU utilization

Burst capacity (2.0x traffic):
  • 628 tok/s
  • 99% GPU utilization

Requests per hour: 7539
Tokens per hour: 1130862
```

### Cost Analysis

```
================================================================================
COST ANALYSIS
================================================================================

Hourly cost per instance: $1.0500
Total hourly cost (2 instances): $2.1000

Cost per 1K tokens: $0.00186
Cost per request: $0.000279

Monthly estimate (730 hours): $1533.0000
```

## Methodology

### Static Roofline Analysis

The tool uses roofline analysis to predict performance ceilings:

1. **Prefill Phase**: Compute-bound for large batches
   - FLOPs = 2 × params × batch × seq_length
   - Memory traffic = model weights + activations

2. **Decode Phase**: Memory-bound (KV cache intensive)
   - FLOPs = 2 × params × batch
   - Memory traffic = model weights + KV cache read/write

3. **Efficiency Factors**:
   - Attention kernel: 65% of theoretical
   - FFN/Linear: 75% of theoretical
   - Memory bandwidth: 75% of peak
   - Tensor parallel: 90-95% depending on TP size

### M/G/1 Queuing Model

Tail latencies (P95/P99) are predicted using M/G/1 queuing theory:

- **Arrival**: Poisson process (exponential inter-arrival)
- **Service**: General distribution (estimated from roofline)
- **Servers**: Single (one GPU or TP group)

Tail latency multipliers:
- P95 ≈ 4.3x P50 at 50% utilization
- P99 ≈ 6.6x P50 at 50% utilization

### SLA Targets by Workload

| Workload | TTFT P50 | TPOT P50 | E2E (512 tok) |
|----------|----------|----------|---------------|
| Chat     | 200ms    | 50ms     | 2000ms        |
| RAG      | 800ms    | 40ms     | 1500ms        |
| Code     | 500ms    | 35ms     | 1200ms        |

### Headroom Configuration

| Headroom | Utilization Target | Burst Capacity |
|----------|-------------------|----------------|
| 0%       | 95%               | 1.0x           |
| 25%      | 75%               | 1.33x          |
| 50%      | 50%               | 2.0x           |
| 100%     | 50%               | 2.0x           |

## Supported Hardware

### GCP VM Instances

| Instance | GPU | VRAM | NVLink | Price/hr |
|----------|-----|------|--------|----------|
| g2-standard-4 | 1x L4 | 24GB | PCIe | $1.05 |
| a2-highgpu-1g | 1x A100 40GB | 40GB | N/A | $3.67 |
| a2-highgpu-8g | 8x A100 40GB | 320GB | NVLink | $29.36 |
| a2-ultragpu-8g | 8x A100 80GB | 640GB | NVLink | $40.24 |
| a3-highgpu-8g | 8x H100 80GB | 640GB | NVLink | $80.00 |

### Interconnect Modeling

- **NVLink (A100/H100)**: 600-900 GB/s, 5% TP overhead
- **PCIe (L4/T4)**: 32-64 GB/s, 15-20% TP overhead

## Extending the Tool

### Adding New Cloud Providers

1. Create a new catalog JSON file:
```json
{
  "metadata": {"provider": "AWS"},
  "vm_instances": [
    {
      "name": "p4d.24xlarge",
      "gpus": [{"type": "nvidia-a100-40gb", "count": 8, ...}],
      ...
    }
  ]
}
```

2. Load the new catalog:
```python
catalog = HardwareCatalog("path/to/aws_catalog.json")
```

### Adding Custom Workloads

Edit `data/workload_defaults.json`:
```json
{
  "workloads": {
    "my_workload": {
      "sla_targets": {
        "ttft_p50_ms": 100,
        "tpot_p50_ms": 20
      },
      ...
    }
  }
}
```

## Testing

```bash
cd varcas/config_iq
python tests/test_sizing.py
```

## Limitations

1. **Static Analysis**: Does not account for dynamic batching effects
2. **Simplified Model**: Assumes uniform token distribution
3. **Network Overhead**: Does not model inter-node communication for pipeline parallelism
4. **Model Support**: Requires known model architecture or manual specification

## Prefix Caching for RAG

ConfigIQ now models **prefix caching** for RAG workloads, which significantly improves TTFT (Time To First Token) when documents or system prompts are reused across requests.

### How It Works

In RAG workloads, the input typically consists of:
- **Cacheable**: Retrieved documents + system prompt (~75% of tokens)
- **Non-cacheable**: User query + dynamic instructions (~25% of tokens)

With vLLM's prefix caching, cacheable tokens that are already in the KV cache don't need to be re-computed, reducing prefill latency dramatically.

### Default RAG Cache Configuration

```json
{
  "prefix_cache": {
    "enabled": true,
    "cacheable_token_ratio": 0.75,    // 75% of input is cacheable
    "cache_hit_rate": 0.70,            // 70% cache hit rate
    "avg_prefix_length": 3500          // ~3500 tokens per cached prefix
  }
}
```

### Impact on Estimates

For a typical RAG workload with 6000 input tokens:

| Metric | Without Cache | With Cache (70% hit) | Improvement |
|--------|---------------|---------------------|-------------|
| Effective Prefill | 6000 tokens | ~2300 tokens | **62% reduction** |
| TTFT | ~800ms | ~300ms | **2.6x faster** |

### Customizing Cache Settings

You can modify the default cache configuration in `data/workload_defaults.json` or override per-workload in code:

```python
from config_iq.core.types import PrefixCacheConfig, WorkloadType
from config_iq.core.workload_patterns import WorkloadPatterns

patterns = WorkloadPatterns()
workload = patterns.get_workload(WorkloadType.RAG)

# Customize cache settings
workload.prefix_cache.cache_hit_rate = 0.85  // Higher hit rate
workload.prefix_cache.cacheable_token_ratio = 0.80
```

## Future Enhancements

- [x] Prefix caching for RAG workloads
- [ ] Dynamic batching model
- [ ] Pipeline parallelism support
- [ ] Multi-node scaling
- [ ] Profiling-based calibration
- [ ] AWS/Azure catalog
- [ ] Web UI
- [ ] Historical recommendation tracking

## License

MIT License
