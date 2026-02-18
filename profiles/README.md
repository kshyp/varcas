# Performance Profiles

Analysis results and profiling data from vLLM optimization experiments.

## Overview

This directory contains performance analysis outputs, optimization reports, and profiling data collected during vLLM tuning experiments. Use these as reference for understanding optimization trade-offs and expected performance gains.

## Structure

```
profiles/
└── roofline/              # Roofline analysis results
    ├── roofline_static.py     # Static analysis tool
    ├── roofline_dynamic.py    # Dynamic profiling tool
    ├── visualize_roofline.py  # Report generator
    ├── *_SUMMARY.md           # Optimization reports
    └── */                     # Result directories
```

## Roofline Analysis

The roofline model analyzes the relationship between arithmetic intensity (FLOPs/byte) and achievable performance (FLOPs/sec).

### Static Analysis

Predict performance bounds without running benchmarks:

```bash
cd roofline
python roofline_static.py --model meta-llama/Llama-2-7b-hf --gpu T4
```

**Output:**
- Theoretical max throughput
- Compute vs memory-bound classification
- Recommended batch sizes

### Dynamic Analysis

Profile actual vLLM execution:

```bash
python roofline_dynamic.py \
  --results load_test_results.json \
  --model meta-llama/Llama-2-7b-hf \
  --gpu T4
```

**Output:**
- Measured vs predicted performance
- Bottleneck identification
- Utilization heatmaps

## Optimization Reports

### FINAL_OPTIMIZATION_SUMMARY.md
Complete optimization journey from baseline to production:
- Phase 1: Roofline analysis
- Phase 2: Easy wins (+22% throughput)
- Phase 3: Medium-term optimizations
- Phase 4: Speculative decoding attempts
- **Final recommendation**: `--max-num-seqs 8`

### EASY_WINS_SUMMARY.md
Quick optimizations that delivered results:
- Batch size tuning
- Memory utilization adjustment
- Immediate 22% throughput improvement

### MEDIUM_TERM_RESULTS.md
Advanced optimizations attempted:
- Chunked prefill (-4% throughput, not recommended)
- FP8 KV cache (hardware limitation on T4)
- Prefix caching (compatibility issues)

### SPECULATIVE_DECODING_ATTEMPT.md
Documentation of speculative decoding attempts:
- Draft model compatibility issues
- N-gram pattern matching
- Hardware/quantization constraints

### BATCH_OPTIMIZATION_RESULTS.md
Detailed batch size analysis:
- Batch 1 vs 8 comparison
- Latency vs throughput trade-offs
- Memory pressure analysis

## Result Directories

| Directory | Contents |
|-----------|----------|
| `easy_wins_results/` | Baseline vs optimized comparison |
| `medium_term_results/` | Advanced optimization tests |
| `speculative_results/` | Failed speculative decoding attempts |
| `speculative_fp16_results/` | FP16 speculative tests |
| `batch_optimization/` | Batch size sweep data |
| `analysis_*/` | Generated HTML reports |

## Key Findings

### What Worked

| Optimization | Improvement | Status |
|--------------|-------------|--------|
| Batch size 1 → 8 | +22% throughput | ✅ Deployed |
| GPU memory 0.85 → 0.90 | +5% capacity | ✅ Deployed |

### What Didn't Work

| Optimization | Issue | Status |
|--------------|-------|--------|
| Chunked prefill | Overhead for short inputs | ❌ Skipped |
| FP8 KV cache | T4 doesn't support FP8 | ❌ Hardware limitation |
| Speculative decoding | Model compatibility | ❌ Not working |
| Prefix caching | AWQ conflicts | ❌ Compatibility |

### Production Configuration

Recommended settings from analysis:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-2-7B-AWQ \
  --dtype half \
  --max-model-len 2048 \
  --max-num-seqs 8 \          # ← KEY OPTIMIZATION
  --gpu-memory-utilization 0.90 \
  --quantization awq \
  --enforce-eager \
  --port 8000
```

## Tools

### batch_size_optimization.py

Automated batch size tuning:

```bash
python batch_size_optimization.py \
  --model TheBloke/Llama-2-7B-AWQ \
  --start 1 --end 16 \
  --metric throughput
```

### visualize_roofline.py

Generate HTML reports:

```bash
python visualize_roofline.py \
  --static static_analysis.json \
  --dynamic dynamic_analysis.json \
  --output report.html
```

### run_roofline_analysis.py

Master orchestrator for complete analysis:

```bash
python run_roofline_analysis.py \
  --model meta-llama/Llama-2-7b-hf \
  --gpu T4 \
  --duration 300
```

## Interpreting Results

### Performance Metrics

| Metric | Good | Bad |
|--------|------|-----|
| Token throughput | >200 tok/s | <150 tok/s |
| TTFT (P50) | <100ms | >500ms |
| TPOT (P50) | <50ms | >100ms |
| GPU utilization | 80-95% | <50% or 100% |

### Roofline Chart

```
Performance (TFLOP/s)
  │
  │    ╭────── Roofline ceiling
  │   ╱
  │  ╱
  │ ╱  ● Phase A (compute-bound)
  │╱
  │●   Phase B (memory-bound)
  └──────────────────
    Low    High
    Arithmetic Intensity
```

**Compute-bound**: Increase batch size
**Memory-bound**: Quantization, cache optimization

## Using These Results

### For New Deployments

1. Check `FINAL_OPTIMIZATION_SUMMARY.md` for recommended config
2. Apply settings from `start_vllm_optimized.sh`
3. Validate with your workload profile

### For Optimization Projects

1. Run `roofline_static.py` for your model/GPU combination
2. Compare with results in this directory
3. Focus on optimizations that worked for similar hardware

### For Research

The raw data files (`.json`, `.log`) contain:
- Per-request latency breakdowns
- Token-level timing
- GPU utilization traces
- Memory usage profiles

## Data Retention

Large profiling files are excluded from git (see `.gitignore`):
- `*.sqlite` - Nsight Systems databases
- `*.nsys-rep` - Nsight reports
- `*.ncu-rep` - Nsight Compute reports
- `*.pt.trace.json.gz` - PyTorch traces

Keep these locally for detailed analysis or archive to cloud storage.

## Contributing Results

When adding new optimization experiments:

1. Create descriptive directory: `experiment_name_results/`
2. Include configuration files used
3. Add summary document: `EXPERIMENT_NAME_SUMMARY.md`
4. Update this README with key findings
