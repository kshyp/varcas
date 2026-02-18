# Varcas: vLLM Performance Optimization & Validation Framework

Varcas is a comprehensive framework for profiling, optimizing, and validating vLLM inference performance. It provides tools for hardware sizing, load testing, performance profiling, and accuracy validation.

## Overview

Varcas ("Validation and Resource Configuration Analysis System") helps you:

- **Size hardware** for LLM inference workloads using roofline analysis and queuing models
- **Profile performance** with PyTorch Profiler, Nsight Systems, and Nsight Compute
- **Generate load** with realistic workload patterns (chat, RAG, code generation)
- **Validate accuracy** with ground truth comparison and degradation metrics
- **Optimize configurations** based on empirical analysis

## Project Structure

```
varcas/
├── benchmark_harness/       # Production-grade load generator for vLLM
│   ├── varcas_load_harness.py   # Main load testing tool
│   ├── adapters/                # Customer trace adapters
│   └── *.sh                     # Quick test scripts (chat, rag, code, etc.)
│
├── config_iq/              # Hardware sizing tool (ConfigIQ)
│   ├── core/                    # Analysis engines
│   │   ├── roofline_analyzer.py # Static roofline analysis
│   │   ├── queuing_model.py     # M/G/1 queuing model
│   │   └── sla_calculator.py    # Recommendation engine
│   └── cli/main.py              # CLI interface
│
├── profilers/              # Performance profiling tools
│   ├── pytorch/                 # PyTorch Profiler integration
│   └── nvidia/                  # NVIDIA Nsight (nsys, ncu)
│
├── profiles/               # Analysis results and reports
│   └── roofline/                # Roofline analysis outputs
│       ├── roofline_static.py   # Static analysis tool
│       ├── roofline_dynamic.py  # Dynamic profiling tool
│       └── *_SUMMARY.md         # Optimization reports
│
├── optimizer/              # System optimization tools
│   └── linux/                   # Linux tuning scripts
│
├── ground_truth_generator/ # Ground truth generation for validation
│   └── ground_truth_generator.py # External API-based generator
│
├── validator/              # Accuracy validation tools
│   └── varcas_validator.py      # Validation framework
│
└── start_vllm.sh           # Example vLLM startup script
```

## Quick Start

### 1. Hardware Sizing (ConfigIQ)

Determine optimal hardware for your workload:

```bash
cd config_iq
pip install -e .

# Size for 100 concurrent chat users
python -m config_iq.cli.main \
  --model meta-llama/Llama-2-7b-hf \
  --workload chat \
  --users 100
```

### 2. Load Testing

Run realistic load tests against your vLLM deployment:

```bash
cd benchmark_harness

# Start vLLM first
vllm serve meta-llama/Llama-2-7b-hf --max-model-len 2048

# Run chat workload at medium intensity (20 RPS)
python varcas_load_harness.py --profile chat_medium --duration 60

# Run mixed workload (60% chat, 30% RAG, 10% code)
python varcas_load_harness.py --profile mixed --duration 120
```

### 3. Performance Profiling

Profile vLLM with NVIDIA tools:

```bash
cd profilers/nvidia

# Nsight Systems (recommended)
./run_nsys_vllm_varcas.sh chat_medium

# View results
nsys-ui vllm_nsys_chat_medium_*.nsys-rep
```

### 4. Ground Truth Generation

Generate reference answers for validation:

```bash
cd ground_truth_generator
python ground_truth_generator.py --input prompts.json --output ground_truth.json
```

### 5. Accuracy Validation

Validate model outputs against ground truth:

```bash
cd validator
python varcas_validator.py --ground-truth ground_truth.json --results vllm_output.json
```

## Workload Profiles

The benchmark harness supports these predefined profiles:

| Profile | Description | Target RPS | Use Case |
|---------|-------------|------------|----------|
| `chat_low` | Chat - low intensity | 5 | Development/testing |
| `chat_medium` | Chat - medium intensity | 20 | Production chatbots |
| `chat_high` | Chat - high intensity | 50 | High-traffic apps |
| `rag_small_low` | RAG - small context | 2 | Document Q&A |
| `rag_medium_medium` | RAG - medium context | 10 | Knowledge bases |
| `code_low` | Code generation - low | 3 | IDE suggestions |
| `code_medium` | Code generation - medium | 15 | Code completion |
| `mixed` | Mixed workloads | 25 | General purpose |
| `burst` | Bursty traffic | Variable | Traffic spikes |

## Key Components

### ConfigIQ
Hardware sizing tool using roofline analysis and M/G/1 queuing models. Predicts latency percentiles (P50/P95/P99) and recommends cost-optimized configurations.

**Features:**
- Multi-GPU tensor parallelism support
- Prefix caching for RAG workloads
- GCP/AWS/Azure VM catalog
- Cost per request analysis

See [config_iq/README.md](config_iq/README.md)

### Benchmark Harness
Production-grade load generator with:
- Poisson arrival patterns (realistic traffic)
- NVTX annotation support
- Customer trace replay
- A/B testing framework

See [benchmark_harness/README.md](benchmark_harness/README.md)

### Profilers

**PyTorch Profiler** (`profilers/pytorch/`):
- Safe profiling for vLLM's multi-process architecture
- Chrome/Perfetto trace visualization
- Minimal overhead configuration

**NVIDIA Profilers** (`profilers/nvidia/`):
- Nsight Systems: Timeline analysis
- Nsight Compute: Kernel-level metrics
- Multiple workload profiles

### Roofline Analysis
Static and dynamic roofline modeling for performance bounds:
- Compute vs memory-bound phase identification
- Batch size optimization
- Hardware utilization prediction

See [profiles/roofline/README.md](profiles/roofline/README.md)

## Requirements

- Python 3.8+
- CUDA-capable GPU (for profiling)
- vLLM 0.15.1+ (tested version)
- NVIDIA drivers (for GPU profiling)

### Python Dependencies

```bash
pip install -r config_iq/requirements.txt
```

## Optimization Workflow

1. **Size Hardware**: Use ConfigIQ to determine initial hardware
2. **Establish Baseline**: Run benchmark harness with default settings
3. **Profile**: Use NVIDIA profilers to identify bottlenecks
4. **Analyze**: Run roofline analysis to find optimization opportunities
5. **Optimize**: Apply configuration changes (batch size, quantization, etc.)
6. **Validate**: Compare against baseline with accuracy validation

## Example: End-to-End Optimization

```bash
# 1. Size hardware for 50 concurrent users
cd config_iq
python -m config_iq.cli.main --model meta-llama/Llama-2-7b-hf --workload chat --users 50

# 2. Start vLLM with recommended config
cd ..
./start_vllm.sh

# 3. Run baseline benchmark
cd benchmark_harness
python varcas_load_harness.py --profile chat_medium --duration 60 > baseline.json

# 4. Profile with nsys
cd ../profilers/nvidia
./run_nsys_vllm_varcas.sh chat_medium

# 5. Analyze and optimize
cd ../../profiles/roofline
python roofline_static.py --model meta-llama/Llama-2-7b-hf --gpu T4
# Apply recommended --max-num-seqs from output

# 6. Re-run benchmark and compare
# ...
```

## Documentation

| Component | Documentation |
|-----------|---------------|
| ConfigIQ | [config_iq/README.md](config_iq/README.md) |
| Benchmark Harness | [benchmark_harness/README.md](benchmark_harness/README.md) |
| PyTorch Profiler | [profilers/pytorch/README.md](profilers/pytorch/README.md) |
| NVIDIA Profilers | [profilers/nvidia/README.md](profilers/nvidia/README.md) |
| Roofline Analysis | [profiles/roofline/README.md](profiles/roofline/README.md) |
| Ground Truth | [ground_truth_generator/README.md](ground_truth_generator/README.md) |
| Validator | [validator/README.md](validator/README.md) |
| Optimizer | [optimizer/README.md](optimizer/README.md) |

## License

MIT License

## Contributing

This is a research and optimization framework. Contributions welcome!

## Troubleshooting

### vLLM fails to start
- Check port 8000: `lsof -i :8000`
- Kill existing processes: `pkill -f vllm.entrypoints`
- Verify GPU: `nvidia-smi`

### Profiling crashes
- Use `--enforce-eager` to disable CUDA graphs
- Check PyTorch profiler compatibility with vLLM version
- See `profilers/pytorch/CRASH_ANALYSIS.md` for known issues

### Out of memory
- Reduce `--max-num-seqs`
- Lower `--gpu-memory-utilization`
- Use quantization (AWQ, GPTQ, FP8)
