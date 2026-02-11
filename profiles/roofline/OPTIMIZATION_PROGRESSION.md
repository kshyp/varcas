# Complete Optimization Progression

**Date**: 2026-02-11  
**Model**: TheBloke/Llama-2-7B-AWQ (4-bit AWQ)  
**GPU**: Tesla T4  
**Workload**: Chat Medium (20 RPS, ~50 input tokens, ~150 output tokens)

---

## Executive Summary

This document presents the complete optimization journey from original configuration through easy wins to medium-term optimizations.

### Results Overview

| Phase | Configuration | Token Throughput | TPOT | Latency p50 | Status |
|-------|--------------|------------------|------|-------------|--------|
| **Original** | Baseline | 168.8 tok/s | 231.7 ms | 26,013 ms | Starting point |
| **Easy Wins** | +batch=8 | **206.8 tok/s** | **34.5 ms** | **16,981 ms** | ✅ **RECOMMENDED** |
| **Medium-Term** | +chunked | 197.9 tok/s | 37.4 ms | 18,308 ms | ❌ Not beneficial |

**Key Achievement**: +22% throughput, -85% decode latency with Easy Wins

---

## Phase 1: Original Baseline

### Configuration
```bash
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-2-7B-AWQ \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --quantization awq \
  --enforce-eager \
  --port 8000
```

### Performance
- Token Throughput: 168.8 tok/s
- TPOT (p50): 231.7 ms/token
- Latency (p50): 26,013 ms
- TTFT (p50): 669 ms

### Roofline Analysis
- Decode AI: ~3 FLOP/Byte (memory-bound)
- GPU Utilization: ~1.5% (severely underutilized)
- Bottleneck: Memory bandwidth

---

## Phase 2: Easy Wins ✅

### Configuration
```bash
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-2-7B-AWQ \
  --dtype half \
  --max-model-len 2048 \
  --max-num-seqs 8 \          # ← ADDED
  --gpu-memory-utilization 0.90 \
  --quantization awq \
  --enforce-eager \
  --port 8000
```

### Optimizations Applied
| Optimization | Expected | Actual | Status |
|--------------|----------|--------|--------|
| Batch size (8) | 3-5x throughput | +22% throughput | ✅ |
| CUDA Graphs | 10-20% latency | N/A (AWQ incompatibility) | ⚠️ |
| Flash Attention | Baseline | Already enabled | ✅ |

### Performance
- Token Throughput: **206.8 tok/s** (+22%)
- TPOT (p50): **34.5 ms/token** (-85%)
- Latency (p50): **16,981 ms** (-35%)
- TTFT (p50): 13,650 ms (+1940%)

### Key Improvements
```
Decode Speed (TPOT):
Before: [████████████████████████████████████████] 231.7 ms/token
After:  [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  34.5 ms/token
                              6.7x faster!

Token Throughput:
Before: [████████████████████████████░░░░░░░░░░░░] 168.8 tok/s
After:  [██████████████████████████████████░░░░░░] 206.8 tok/s
                              +22% improvement
```

### Trade-offs
- ✅ Much faster decode (6.7x)
- ✅ Higher throughput (+22%)
- ✅ Lower total latency (-35%)
- ⚠️ Higher TTFT (+1940%) - wait longer, but process faster

---

## Phase 3: Medium-Term Optimizations ❌

### Attempted Optimizations

| Optimization | Expected | Applied | Result | Reason |
|--------------|----------|---------|--------|--------|
| Chunked Prefill | Better interleaving | ✅ | -4% throughput | Not beneficial for short inputs |
| FP8 KV Cache | 30% memory reduction | ❌ | N/A | T4 doesn't support FP8 |
| Speculative Decoding | 2-3x speedup | ❌ | N/A | No draft model available |
| Prefix Caching | KV reuse | ❌ | N/A | AWQ compatibility issues |

### Configuration
```bash
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-2-7B-AWQ \
  --dtype half \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --quantization awq \
  --enforce-eager \
  --port 8000 \
  --enable-chunked-prefill      # ← ADDED
```

### Performance vs Baseline
- Token Throughput: 197.9 tok/s (-4.3% vs Easy Wins)
- TPOT (p50): 37.4 ms/token (+8.3%)
- Latency (p50): 18,308 ms (+7.8%)

### Why It Didn't Help

Chunked prefill is designed for:
- Long input sequences (1000+ tokens)
- Mixed workloads with varying lengths
- Reducing TTFT for large prefills

**This workload characteristics:**
- Short inputs (~50 tokens)
- Prefill already fast (compute-bound)
- Chunking adds scheduling overhead
- Result: 4-8% performance degradation

---

## Visual Comparison

### Throughput Progression
```
Token Throughput (tok/s)
═══════════════════════════════════════════════════════════

Original:  [████████████████████████████░░░░░░░░░░░░░░] 168.8
Easy Wins: [██████████████████████████████████░░░░░░░░] 206.8  ✓
Medium:    [████████████████████████████████░░░░░░░░░░] 197.9  ✗
            0        50       100      150      200      250
```

### Decode Speed Progression
```
Time Per Token (ms) - Lower is Better
═══════════════════════════════════════════════════════════

Original:  [████████████████████████████████████████] 231.7
Easy Wins: [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  34.5  ✓
Medium:    [█████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  37.4  ✗
            0        50       100      150      200      250
```

### Roofline Position
```
Performance (TFLOPS)
    │
 65 ├────────────────────────────┐ ← Peak Compute (Prefill)
    │                             \
 10 ├                              \
    │                               ● Easy Wins (AI≈20)
  1 ├──────────●───────────────────/────
    │      Original              /
 0.1├────(AI≈3)────────────────/──────
    └──────────┼───────────────┼────────
              10             203
           AI (FLOP/Byte)
           
Easy Wins moved us higher on the memory-bound slope!
```

---

## Recommendations

### Production Configuration

**Use: `start_vllm_optimized.sh`**
```bash
#!/bin/bash
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-2-7B-AWQ \
  --dtype half \
  --max-model-len 2048 \
  --max-num-seqs 8 \\          # ← KEY OPTIMIZATION
  --gpu-memory-utilization 0.90 \
  --quantization awq \
  --enforce-eager \
  --port 8000
```

### When to Use Each Configuration

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Production (default) | Easy Wins | Best balance of throughput/latency |
| Latency-sensitive | Original | Fastest TTFT |
| Throughput-maximized | Easy Wins | +22% tokens/s |
| Long-context (RAG) | Easy Wins + chunked | Chunked helps with 1000+ tokens |
| Short chat | Easy Wins | Chunked doesn't help |

---

## Hardware Upgrade Path

### Current Limitations (T4)
- ❌ No FP8 support
- ❌ Limited memory bandwidth (320 GB/s)
- ❌ Older architecture (Turing)

### Recommended Upgrades

| GPU | Memory BW | FP8 | Est. Improvement | Cost |
|-----|-----------|-----|------------------|------|
| T4 (current) | 320 GB/s | ❌ | Baseline | - |
| L4 | 300 GB/s | ✅ | +20% efficiency | 2-3x |
| A10G | 600 GB/s | ❌ | +50-80% throughput | 2-3x |
| A100 | 1,555 GB/s | ❌ | +100-150% throughput | 4-5x |
| L40S | 864 GB/s | ✅ | +100% throughput | 3-4x |

---

## Key Learnings

### What Worked
1. ✅ **Batch size optimization** - Significant improvement (+22% throughput, -85% TPOT)
2. ✅ **Roofline analysis** - Accurately identified memory-bound bottleneck
3. ✅ **Measured trade-offs** - Higher TTFT but faster overall completion

### What Didn't Work
1. ❌ **Chunked prefill** - Added overhead without benefit for short inputs
2. ❌ **FP8 KV cache** - Hardware limitation (T4)
3. ❌ **Speculative decoding** - Requires draft model

### General Principles
1. 📊 **Measure everything** - Don't assume optimizations help
2. 🎯 **Workload matters** - Same optimization, different results per workload
3. 🔧 **Hardware constraints** - Know your GPU's capabilities
4. ⚖️ **Trade-offs exist** - Throughput vs latency is a real trade-off

---

## Files Reference

```
varcas/profiles/roofline/
│
├── start_vllm.sh                    # Original configuration
├── start_vllm_optimized.sh          # ✅ RECOMMENDED (Easy Wins)
├── start_vllm_advanced.sh           # Medium-term (chunked prefill)
│
├── OPTIMIZATION_PROGRESSION.md      # This document
├── EASY_WINS_SUMMARY.md             # Easy wins results
├── MEDIUM_TERM_RESULTS.md           # Medium-term results
├── RESULTS_SUMMARY.md               # Roofline analysis
│
├── easy_wins_results/
│   ├── baseline.json                # Original test results
│   └── optimized.json               # Easy wins test results
│
└── medium_term_results/
    ├── baseline.json                # Easy wins baseline
    └── advanced.json                # Chunked prefill results
```

---

## Summary

**The Easy Wins optimization delivered significant improvements:**
- 🚀 **+22% token throughput**
- ⚡ **-85% decode latency** (6.7x faster)
- ⏱️ **-35% total latency**

**Medium-term optimizations didn't help** for this specific workload due to hardware constraints and workload characteristics.

**Final Recommendation**: Use `start_vllm_optimized.sh` with `--max-num-seqs 8` for production deployment.

---

*Generated by roofline analysis and optimization tools*
