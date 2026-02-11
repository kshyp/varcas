# Roofline Analysis for vLLM

This directory contains roofline analysis results for the Llama-2-7B-AWQ model running on vLLM with a Tesla T4 GPU.

## What is Roofline Analysis?

The roofline model characterizes the performance of computing workloads by plotting **attainable performance** (TFLOPS) vs **arithmetic intensity** (FLOPs per byte). It helps identify whether a workload is:
- **Compute-bound**: Limited by GPU compute capability (above the ridge point)
- **Memory-bound**: Limited by memory bandwidth (below the ridge point)

## Files

| File | Description |
|------|-------------|
| `static_analysis.json` | Theoretical roofline based on model architecture |
| `dynamic_analysis.json` | Measured performance from load testing |
| `load_test_results.json` | Detailed load test metrics (1214 requests) |
| `roofline_report.html` | Interactive visualization report |
| `roofline_static.py` | Static analysis tool |
| `roofline_dynamic.py` | Dynamic profiling tool (NCU-based) |
| `visualize_roofline.py` | Visualization generator |
| `run_roofline_analysis.py` | Master orchestration script |

## Key Results

### Hardware Configuration
- **GPU**: Tesla T4
- **Peak FP16 Performance**: 65 TFLOPS
- **Memory Bandwidth**: 320 GB/s
- **Ridge Point**: 203 FLOP/Byte

### Model Configuration
- **Model**: Llama-2-7B-AWQ (4-bit quantization)
- **Effective Parameters**: 1.68B (after 4-bit quantization)
- **Hidden Size**: 4096
- **Layers**: 32

### Phase Analysis

#### 1. Prefill Phase (Input Processing)
| Batch | Seq Length | Arithmetic Intensity | Attainable TFLOPS | Bound Type |
|-------|------------|---------------------|-------------------|------------|
| 1 | 128 | 366.55 | 65.0 (100%) | Compute |
| 1 | 512 | 1210.94 | 65.0 (100%) | Compute |

**Conclusion**: Prefill is **compute-bound** with high arithmetic intensity.

#### 2. Decode Phase (Token Generation)
| Batch | Context Length | Arithmetic Intensity | Attainable TFLOPS | Bound Type |
|-------|---------------|---------------------|-------------------|------------|
| 1 | 512 | 3.06 | 0.98 (1.5%) | Memory |
| 1 | 2048 | 3.00 | 0.96 (1.5%) | Memory |

**Conclusion**: Decode is **memory-bound** with low arithmetic intensity.

### Workload Analysis (Chat Medium - 20 RPS)

| Metric | Value |
|--------|-------|
| Total Requests | 1,214 |
| Success Rate | 100% |
| Throughput | 20.26 req/s |
| Token Throughput | 173.6 tok/s |
| TTFT (p50/p99) | 803ms / 50,674ms |
| TPOT (p50/p99) | 279ms / 343ms |
| Dominant Bottleneck | **MEMORY** |
| Memory Required | 3.38 GB |

## Roofline Plot Interpretation

```
Performance (TFLOPS)
    │
 65 ├────────────────────┐ ← Peak Compute (FP16 Tensor Cores)
    │                     \
    │                      \
    │                       \
    │                        \
 10 ├─────────────────────────\────
    │                          \
  1 ├──────────────●────────────\─── ← Decode (AI ≈ 3)
    │            /                \
 0.1├───────────/──────────────────\────
    │          /                    Ridge Point (AI = 203)
    └─────────┼──────────────────────────
             1   10   100  1000
           Arithmetic Intensity (FLOP/Byte)
           
           ● = Workload positions
           Prefill: AI > 200 (Compute-bound)
           Decode:  AI < 5   (Memory-bound)
```

## Optimization Recommendations

Based on the roofline analysis, here are the key optimization opportunities:

### 1. Increase Batch Size (High Impact)
Since decode is memory-bound, increasing batch size improves arithmetic intensity:
- **Current**: Batch=1, AI ≈ 3
- **Target**: Batch=8-16, AI ≈ 15-25 (closer to ridge point)
- **Expected Gain**: 3-5x throughput improvement for decode phase

### 2. Optimize KV Cache (Medium Impact)
The decode phase's memory traffic is dominated by KV cache reads:
- Use **Grouped Query Attention (GQA)** or **Multi-Query Attention (MQA)**
- Enable **KV cache quantization** (8-bit instead of 16-bit)
- Implement **KV cache paging** for better memory utilization

### 3. Continuous Batching (High Impact)
Overlap prefill and decode operations:
- Prefill is compute-bound and underutilizes memory bandwidth
- Decode is memory-bound and underutilizes compute
- Continuous batching allows both to proceed concurrently

### 4. Quantization Already Applied
The model is already using 4-bit AWQ quantization:
- Model size reduced from 6.7B to ~1.68B effective parameters
- Memory footprint: ~3.38 GB total (model + KV cache)
- Further quantization (e.g., 2-bit) may hurt accuracy significantly

### 5. Kernel Optimization (Medium Impact)
- Use **Flash Attention** or **Flash Infer** for memory-efficient attention
- Enable **CUDA graphs** for reduced CPU overhead (disabled during profiling)
- Consider **TensorRT-LLM** for optimized kernels

## Running the Analysis

### Static Analysis Only (Fast)
```bash
python varcas/profiles/roofline/roofline_static.py \
    --model llama2-7b \
    --quantization awq \
    --bits 4
```

### Full Analysis with Load Testing
```bash
python varcas/profiles/roofline/run_roofline_analysis.py \
    --model TheBloke/Llama-2-7B-AWQ \
    --quantization awq \
    --profile chat_medium \
    --duration 60
```

### Analysis with Existing vLLM Server
```bash
# Start vLLM separately
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-7B-AWQ \
    --quantization awq \
    --port 8000

# Run analysis
python varcas/profiles/roofline/run_roofline_analysis.py \
    --profile chat_medium \
    --no-manage-vllm
```

## Visualization

Open `roofline_report.html` in a web browser to view:
- Interactive roofline plots
- Kernel-level performance breakdown
- Workload characterization
- Optimization recommendations

## Technical Details

### Arithmetic Intensity Calculations

For transformer inference:

**Prefill AI** ≈ (2 × batch × seq_len × hidden_size × layers) / (model_params × bytes_per_param)

**Decode AI** ≈ (2 × batch × hidden_size × layers) / (model_params × bytes_per_param + KV_cache)

The low AI for decode (< 5) explains why it's memory-bound on T4 (ridge point = 203).

### Memory Bandwidth vs Compute

On Tesla T4:
- Memory bandwidth: 320 GB/s
- Peak FP16 compute: 65 TFLOPS
- Ridge point: 65,000 / 320 = 203 FLOP/Byte

Any workload with AI < 203 is memory-bound; AI > 203 is compute-bound.

## References

1. Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: an insightful visual performance model for multicore architectures. Communications of the ACM, 52(4), 65-76.
2. Kaggle Blog: Transformer Math
3. NVIDIA Performance Optimization Guides

---

Generated: 2026-02-11
Analysis Tool: varcas/profiles/roofline/
