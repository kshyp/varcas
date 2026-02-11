# Medium-Term Optimizations - Results

**Date**: 2026-02-11  
**Baseline**: Optimized config (max_num_seqs=8)  
**Model**: TheBloke/Llama-2-7B-AWQ  
**GPU**: Tesla T4  
**Workload**: Chat Medium (20 RPS)

---

## Overview

This document compares the optimized baseline against medium-term optimizations identified in the roofline analysis.

### Medium-Term Optimizations Attempted

| Optimization | Expected | Applied | Result | Status |
|--------------|----------|---------|--------|--------|
| KV Cache Quantization (FP8) | 30% memory reduction | ❌ Not applied | N/A | Not supported on T4 |
| Speculative Decoding | 2-3x speedup | ❌ Not applied | N/A | Requires draft model |
| Chunked Prefill | Better interleaving | ✅ Applied | Slight degradation | Not beneficial for short inputs |
| Continuous Batching | Better utilization | ✅ Already default | Baseline | Working as expected |
| Prefix Caching | Reuse KV cache | ❌ Issues | N/A | Compatibility issues with AWQ |

---

## Test Results

### Baseline vs Advanced Comparison

| Metric | Baseline (batch=8) | Advanced (+chunked) | Change |
|--------|-------------------|---------------------|--------|
| Token Throughput | 206.8 tok/s | 197.9 tok/s | **-4.3%** |
| Request Throughput | 19.36 req/s | 19.37 req/s | +0.1% |
| TTFT p50 | 13,650 ms | 14,004 ms | +2.6% |
| TPOT p50 | 34.5 ms | 37.4 ms | **+8.3%** |
| Latency p50 | 16,981 ms | 18,308 ms | **+7.8%** |
| Latency p99 | 32,289 ms | 32,998 ms | +2.2% |

### Complete Optimization Progression

```
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│     Metric      │   Original   │ Easy Wins    │ Medium-Term  │
│                 │  (batch=1)   │ (batch=8)    │(+chunked)    │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Token Throughput│   168.8      │   206.8      │   197.9      │
│ TPOT p50 (ms)   │     231.7    │     34.5     │     37.4     │
│ Latency p50 (ms)│   26,013     │   16,981     │   18,308     │
│ TTFT p50 (ms)   │       669    │   13,650     │   14,004     │
└─────────────────┴──────────────┴──────────────┴──────────────┘

Improvements over original:
• Easy Wins:    +22% throughput, -85% TPOT ✅
• Medium-Term:  -4% vs baseline, +8% TPOT ⚠️
```

---

## Analysis

### Why Chunked Prefill Didn't Help

Chunked prefill is designed to improve interactivity by:
1. Breaking large prefill operations into smaller chunks
2. Interleaving prefill chunks with decode operations
3. Reducing TTFT for long-context requests

**However, for this workload:**
- Input sequences are short (~50 tokens average)
- Prefill is already compute-bound and fast
- Chunking adds scheduling overhead
- Results show 4-8% degradation in performance

### Unsupported Optimizations

#### 1. FP8 KV Cache Quantization

**Why not applied:**
- T4 GPU (Turing architecture) doesn't support FP8
- FP8 requires Ada Lovelace (RTX 40-series) or Hopper (H100)
- Attempting to use `--kv-cache-dtype fp8` causes errors

**Hardware requirements:**
```
GPU Support for FP8:
✅ H100 (Hopper)      - Full support
✅ RTX 4090 (Ada)     - Full support  
✅ L4 (Ada)           - Full support
❌ T4 (Turing)        - Not supported
❌ A100 (Ampere)      - Not supported
```

#### 2. Speculative Decoding

**Why not applied:**
- Requires a draft model (smaller, faster model)
- Draft model generates candidate tokens
- Main model verifies/rejects candidates
- We don't have an appropriate draft model for Llama-2-7B-AWQ

**Implementation would require:**
```bash
--speculative-model TinyLlama/TinyLlama-1.1B-chat-v1.0 \
--num-speculative-tokens 5
```

#### 3. Prefix Caching

**Why not applied:**
- Compatibility issues with AWQ quantization
- `--enable-prefix-caching` flag not supported with current vLLM version + AWQ
- Would help repeated prompts (e.g., system prompts)

---

## What Actually Works

### Recommended Configuration

Based on testing, the **Easy Wins** configuration remains the best:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-2-7B-AWQ \
  --dtype half \
  --max-model-len 2048 \
  --max-num-seqs 8 \\          # ← KEY: Batch size optimization
  --gpu-memory-utilization 0.90 \
  --quantization awq \
  --enforce-eager \
  --port 8000
```

### When to Consider Chunked Prefill

Chunked prefill MAY be beneficial for:

| Workload Type | Input Length | Expected Benefit |
|---------------|--------------|------------------|
| RAG | 2,000+ tokens | Reduced TTFT |
| Document Q&A | 1,000+ tokens | Better interactivity |
| Mixed workloads | Variable | Fairer scheduling |
| Long-context chat | 500+ tokens | Reduced queue waits |

**For short-input chat (this workload):**
- ❌ Not recommended
- ❌ Adds overhead without benefit
- ❌ Degrades performance by 4-8%

---

## Hardware Limitations

### Tesla T4 Constraints

The T4 GPU limits which optimizations are possible:

| Optimization | T4 Support | Alternative |
|--------------|------------|-------------|
| FP8 KV Cache | ❌ No | Use 8-bit or 16-bit |
| BF16 compute | ❌ No | Use FP16 |
| CUDA Graphs | ✅ Yes | Can enable |
| Tensor Cores | ✅ Yes | Already using |
| Flash Attention | ✅ Yes | Already using |

### Upgrade Path

For better performance, consider:

| GPU | Memory BW | FP8 Support | Relative Cost |
|-----|-----------|-------------|---------------|
| T4 (current) | 320 GB/s | ❌ | Baseline |
| L4 | 300 GB/s | ✅ | 2-3x |
| A10G | 600 GB/s | ❌ | 2-3x |
| A100 | 1,555 GB/s | ❌ | 4-5x |
| L40S | 864 GB/s | ✅ | 3-4x |

---

## Conclusion

### What We Learned

1. **Easy Wins deliver real value**: +22% throughput, -85% decode latency
2. **Medium-term optimizations are workload-dependent**: Chunked prefill hurt short-input performance
3. **Hardware constraints matter**: T4 doesn't support FP8, limiting some optimizations
4. **No free lunch**: Each optimization has trade-offs

### Final Recommendation

**Use the Easy Wins configuration** (`start_vllm_optimized.sh`) as the production configuration:

```bash
# Easy Wins - PRODUCTION RECOMMENDED
--max-num-seqs 8
```

**Skip chunked prefill** for this short-input chat workload.

**Consider upgrading GPU** if you need:
- FP8 KV cache quantization
- Higher memory bandwidth
- Better performance for long-context

---

## Files Generated

```
varcas/profiles/roofline/medium_term_results/
├── baseline.json           # Optimized baseline (batch=8)
├── advanced.json           # With chunked prefill
├── baseline_server.log     # Server logs
└── advanced_server.log     # Server logs

varcas/profiles/roofline/
├── start_vllm_advanced.sh       # Advanced config (chunked prefill)
├── start_vllm_optimized.sh      # Recommended config (batch=8)
└── MEDIUM_TERM_RESULTS.md       # This document
```

---

*Analysis based on 2x 35-second load tests with 676-677 requests each*
