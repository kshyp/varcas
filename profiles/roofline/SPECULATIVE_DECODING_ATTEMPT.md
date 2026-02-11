# Speculative Decoding Attempt with llama-160m

**Date**: 2026-02-11  
**Target Model**: TheBloke/Llama-2-7B-AWQ  
**Draft Model**: JackFram/llama-160m  
**GPU**: Tesla T4  
**vLLM Version**: 0.15.1

---

## Objective

Attempt to implement speculative decoding using JackFram/llama-160m as the draft model for Llama-2-7B-AWQ, expecting 2-3x speedup.

---

## What is Speculative Decoding?

Speculative decoding accelerates inference by:
1. Using a small draft model (llama-160m) to quickly generate candidate tokens
2. Running the large target model (llama-2-7b) to verify/reject candidates
3. When draft is correct: Save compute (major win!)
4. When draft is wrong: Fall back to target model (minor overhead)

**Expected benefits:**
- 2-3x speedup for token generation
- Maintains same output quality as target model
- Draft model runs in parallel with target model

---

## Attempt 1: Draft Model (llama-160m)

### Configuration
```bash
SPEC_CONFIG='{"method": "draft_model", "model": "JackFram/llama-160m", "num_speculative_tokens": 5}'

python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-2-7B-AWQ \
  --dtype half \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --quantization awq \
  --enforce-eager \
  --port 8000 \
  --speculative-config "$SPEC_CONFIG"
```

### Result: ❌ Failed

**Error:**
```
ValueError: could not determine the shape of object type 'torch.storage.UntypedStorage'
```

**Root Cause:**
The JackFram/llama-160m model appears to have incompatible weight format with vLLM 0.15.1. The error occurs during weight loading in the model executor.

**Full error trace:**
```
File "/opt/conda/lib/python3.10/site-packages/vllm/model_executor/model_loader/weight_utils.py", line 727, in safetensors_weights_iterator
    param = f.get_tensor(name)
ValueError: could not determine the shape of object type 'torch.storage.UntypedStorage'
```

This suggests the safetensors format of llama-160m may be incompatible or corrupted.

---

## Attempt 2: N-Gram Speculative Decoding

As an alternative, we attempted n-gram speculative decoding which doesn't require a separate draft model.

### Configuration
```bash
SPEC_CONFIG='{"method": "ngram", "prompt_lookup_max": 8, "prompt_lookup_min": 1, "num_speculative_tokens": 5}'
```

### How N-Gram Works
- Matches sequences in the prompt to predict continuation
- Uses repeating patterns within the context
- No additional model needed

### Result: ❌ Failed to Start

The process didn't complete initialization within the timeout period. This may be due to:
- Compatibility issues with AWQ quantization
- Additional overhead for n-gram lookup
- Memory constraints on T4

---

## Why Speculative Decoding Failed

### Technical Limitations

| Issue | Description |
|-------|-------------|
| Model Compatibility | llama-160m weights incompatible with vLLM 0.15.1 |
| AWQ Quantization | May conflict with speculative decoding mechanisms |
| Memory Constraints | Running two models requires more GPU memory |
| T4 Architecture | Older GPU may have compatibility issues |

### Alternative Draft Models to Try

If you want to attempt speculative decoding again, consider these compatible draft models:

| Draft Model | Size | Compatibility | Notes |
|-------------|------|---------------|-------|
| JackFram/llama-160m | 160M | ❌ Incompatible | Weight format issue |
| TinyLlama/TinyLlama-1.1B | 1.1B | ✅ Likely compatible | Same architecture |
| meta-llama/Llama-2-7b-hf | 7B | ✅ Compatible | Too large for draft |
| princeton-nlp/Sheared-LLaMA-1.3B | 1.3B | ✅ Likely compatible | Pruned model |

### Configuration for TinyLlama (if trying again)
```bash
SPEC_CONFIG='{"method": "draft_model", "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "num_speculative_tokens": 5}'
```

---

## Lessons Learned

### What We Discovered

1. **vLLM speculative decoding is complex**
   - Requires specific model compatibility
   - Different methods (draft_model, ngram, eagle)
   - Version-specific configurations

2. **Draft model selection matters**
   - Not all small models work as drafts
   - Weight format compatibility is critical
   - Same architecture family preferred

3. **Hardware limitations**
   - T4 has memory constraints for running two models
   - llama-160m + llama-2-7b-awq would use ~4-5GB combined
   - May work but requires careful memory management

### Recommended Next Steps

1. **Try TinyLlama-1.1B as draft model**
   ```bash
   --speculative-config '{"method": "draft_model", "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "num_speculative_tokens": 5}'
   ```

2. **Use non-quantized target model**
   - AWQ quantization may conflict with speculative decoding
   - Try with standard FP16 model

3. **Upgrade vLLM version**
   - vLLM 0.15.1 may have bugs with speculative decoding
   - Try v0.15.2 or later

4. **Use EAGLE speculative decoding**
   - More advanced method with better compatibility
   - Requires training but higher quality

---

## Current Status

**Speculative decoding is NOT working** with the current setup due to:
- ❌ Draft model weight incompatibility
- ❌ Potential AWQ quantization conflicts
- ❌ N-gram method also failing

**Recommendation**: Stick with the Easy Wins configuration (`--max-num-seqs 8`) which delivers proven +22% throughput improvement.

---

## Files Generated

```
varcas/profiles/roofline/
├── SPECULATIVE_DECODING_ATTEMPT.md  # This document
├── start_vllm_speculative.sh        # Draft model config (not working)
├── start_vllm_speculative_ngram.sh  # N-gram config (not working)
└── speculative_results/             # Test outputs
```

---

## References

- vLLM Speculative Decoding Docs: https://docs.vllm.ai/en/latest/features/spec_decode.html
- JackFram/llama-160m: https://huggingface.co/JackFram/llama-160m
- Speculative Decoding Paper: https://arxiv.org/abs/2211.17192

---

*Attempted speculative decoding with llama-160m as draft model. Unable to complete due to model compatibility issues.*
