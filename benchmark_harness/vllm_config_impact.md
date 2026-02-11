# vLLM Configuration Impact Guide

Based on production benchmarking experience and the ~5-10% measurement noise floor.

---

## 🔥 High Impact (>20% - Easily Measurable)

| Flag | Expected Gain | Notes |
|------|---------------|-------|
| `--quantization awq/gptq/int4` | 20-40% throughput | INT4 vs FP16. Memory bandwidth bound → quantization wins big |
| `--speculative-decoding-model` | 20-50% | Best for latency, moderate for throughput. Draft model overhead matters |
| `--tensor-parallel-size 2+` | 30-100%+ | When single GPU memory insufficient. Overhead if not needed |
| `--pipeline-parallel-size 2+` | 50-200%+ | Multi-node scaling. Linear scaling until network bottleneck |
| `--max-model-len` (reduce) | 20-50% | Shorter context = less KV cache pressure. Big win for chat workloads |
| `--enable-prefix-caching` | 20-60% | Huge for RAG/multi-turn chat (repeated prefixes). Zero gain for unique prompts |
| `--num-scheduler-steps 8+` | 15-30% | Batched decoding steps. Diminishing returns after 8 |
| `--chunked-prefill-size` | 20-40% | Better interleaving of prefill/decode. Critical for mixed workloads |

**Verdict:** These are architectural changes. Run 3-5x and you'll clearly see the difference.

---

## ⚡ Medium Impact (10-20% - Measurable with Care)

| Flag | Expected Gain | Notes |
|------|---------------|-------|
| `--max-num-batched-tokens` | 10-20% | Higher = better batching, but memory pressure. Tune per workload |
| `--max-num-seqs` | 10-15% | More concurrent sequences. Diminishing returns, watch memory |
| `--gpu-memory-utilization 0.95` | 10-20% | Higher = more KV cache. 0.9→0.95 often measurable |
| `--swap-space 4+` | 10-20% | Prevents OOM death spirals. Zero gain if you don't hit memory limits |
| `--dtype float16/bfloat16` | 10-15% | bfloat16 slightly faster on Ampere+. Not always consistent |
| `--enable-lora` (vs full fine-tune) | 15-25% | Memory savings allow higher batching. Per-request overhead exists |
| `--disable-custom-all-reduce` | 10-20% | Sometimes faster for small tensor parallel. Test both ways |
| `--num-lookahead-slots` | 10-20% | Speculative decoding tuning. Draft quality dependent |

**Verdict:** Run 5-10x to confirm. Gains are real but can be swamped by noise in single runs.

---

## 🔇 Noise (<10% - Likely Indistinguishable)

| Flag | Expected Gain | Notes |
|------|---------------|-------|
| `--block-size 8/16/32` | 3-8% | Memory alignment effects. Lost in noise unless extreme workloads |
| `--enforce-eager` | ±5% | Sometimes faster (no CUDA graph overhead), sometimes slower. YMMV |
| `--worker-use-ray` | ±3% | Process vs thread workers. Highly system dependent |
| `--disable-log-probs` | 1-3% | Logprob computation is cheap unless you request top-100 |
| `--skip-tokenizer-init` | <1% | Startup time only, not inference |
| `--use-v2-block-manager` | ±5% | Usually slightly better, but within noise for most workloads |
| `--seed 42` | 0-5% | Only affects stochastic sampling. No impact on greedy |
| `--temperature/top_p/top_k` | N/A | Changes output quality, not throughput (with greedy) |
| `--repetition-penalty` | <1% | Negligible overhead |
| `--presence-penalty/frequency-penalty` | <1% | Same as above |

**Verdict:** Don't A/B test these. If you think it matters, run 20+ times or use profiling tools (nsys, py-spy).

---

## 🎲 Wildcard (Context Dependent)

| Flag | Impact | Notes |
|------|--------|-------|
| `--load-format safetensors` | 0-50% | Faster loading, zero inference impact. Matters for autoscaling |
| `--kv-cache-dtype fp8` | 10-30% | Newer feature. Memory savings → higher batching. Hardware dependent |
| `--quantization-kv-cache` | 15-25% | Experimental. Can hurt accuracy, measure both perf and quality |
| `--enable-chunked-prefill` | 0-40% | Already enabled by default in newer versions. Massive for RAG |
| `--max-cpu-loras` / `--max-gpu-loras` | 0-30% | If using LoRA. Zero if not. Switching overhead matters |

---

## Quick Decision Tree

```
Want bigger gains? → Quantization, Speculative Decoding, Prefix Caching
Running out of memory? → TP/PP, Quantization, Reduce max-model-len  
Mixed workload (prefill+decode)? → Chunked prefill, Num scheduler steps
Same prompts repeatedly? → Prefix caching (massive)
Unique prompts only? → Focus on batching (max-num-batched-tokens)
```

---

## Recommended A/B Testing Priority

1. **Definitely test** (3-5 runs): Quantization level, Speculative decode, TP size, Prefix caching
2. **Maybe test** (5-10 runs): Max batched tokens, GPU util, Num scheduler steps
3. **Don't bother**: Block size, Seed, Eager mode, Minor dtype changes
