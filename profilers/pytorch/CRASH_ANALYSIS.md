# vLLM PyTorch Profiler Crash Analysis

## Summary

The system crashes/hangs when trying to profile vLLM using PyTorch profiler with the mixed benchmark. There are **multiple distinct issues** causing the failures.

---

## Issue #1: Threading/NCCL Fatal Error (CRITICAL)

**Error Message:**
```
ERROR: External init callback must run in same thread as registerClient (-1785755904 != 599564928)
```

**Location:** `pytorch_mixed_20260208_034857/vllm_server.log`

**Root Cause:**
- vLLM 0.15.1 uses a multi-process architecture with Ray
- The EngineCore runs in a separate process (pid=7083) from the API server (pid=6945)
- When PyTorch profiler is enabled via `--profiler-config.torch_profiler_dir`, it tries to initialize NCCL callbacks
- NCCL requires all initialization to happen in the same thread
- The profiler's `registerClient` and `init callback` are happening in different threads due to vLLM's async architecture

**Why it crashes:**
This is a **fatal NCCL error** that causes:
1. Deadlock in the distributed communication
2. CUDA operations hang
3. System becomes unresponsive

**Affected Scripts:**
- `run_vllm_pytorch_profiler.sh` (uses `--profiler-config.torch_profiler_dir`)
- `profile_vllm_mixed.sh` (same issue)

---

## Issue #2: Missing Profiler Configuration (404 Errors)

**Error Message:**
```
POST /start_profile HTTP/1.1" 404 Not Found
POST /stop_profile HTTP/1.1" 404 Not Found
```

**Location:** `simple_profile_20260208_034542/vllm_server.log`

**Root Cause:**
- `simple_profiler.sh` starts vLLM WITHOUT any `--profiler-config` flag
- But then tries to call `/start_profile` and `/stop_profile` API endpoints
- These endpoints only exist when vLLM is started with profiler config

**Code Problem:**
```bash
# simple_profiler.sh - MISSING profiler-config
python -m vllm.entrypoints.openai.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    ... \
    # NO --profiler-config flag!

# Later tries to call:
curl -X POST http://localhost:8000/start_profile  # → 404
```

---

## Issue #3: Token Length Exceeds Model Capacity

**Error Message:**
```
vllm.exceptions.VLLMValidationError: This model's maximum context length is 257 tokens. 
However, your request has 295 input tokens. 
```

**Root Cause:**
- The benchmark generates prompts that are too long for the model's remaining context
- Model has `max_model_len=1024` configured
- When input has 295 tokens, only 1024-295=729 tokens remain for output
- But benchmark requests may exceed this

**Note:** This doesn't crash the system, but causes request failures (400 Bad Request)

---

## Issue #4: External Profiler Architecture Mismatch

**Script:** `profile_vllm_server.py`

**Problem:**
- Tries to profile vLLM externally using PyTorch profiler
- Creates a profiler in the main script's process
- But vLLM runs in a SEPARATE process (subprocess.Popen)
- The profiler profiles the WRONG process (the script, not vLLM)

**Code Problem:**
```python
# This profiles the CURRENT process, NOT the vLLM process!
self.profiler = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    ...
)
```

---

## Solutions

### Solution A: Use vLLM Built-in Profiler (Without API Calls)

Instead of using `/start_profile` API, let vLLM profile continuously:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dtype half \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.9 \
    --port 8000 \
    --profiler-config.torch_profiler_dir="$OUTPUT_DIR" \
    --profiler-config.torch_profiler_with_stack=false  # Disable stack to avoid threading issues
```

Then run benchmark WITHOUT calling start/stop APIs - profiling runs continuously.

### Solution B: Use Environment Variable Profiling

Set PyTorch profiler via environment variables instead of CLI args:

```bash
export TORCH_PROFILER_ENABLED=1
export TORCH_PROFILER_DIR="$OUTPUT_DIR"
# Then start vLLM normally
```

### Solution C: Use Nsight Systems Instead

The nsys profiler (already in ~/run_nsys_profile.sh) works at the system level and avoids these threading issues.

### Solution D: Disable CUDA Graphs + Reduce Concurrency

CUDA graphs can interfere with PyTorch profiler:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --enforce-eager \\\    # Disable CUDA graphs
    --max-num-seqs 64 \\\    # Reduce concurrency
    ...
```

---

## Recommended Fix

The safest approach is a modified script that:

1. **Starts vLLM with profiler enabled** (but without stack tracing to avoid threading issues)
2. **Does NOT use the `/start_profile` API** (avoiding the NCCL threading error)
3. **Lets profiling run continuously** during the benchmark
4. **Stops vLLM normally** - traces are saved on shutdown

See `safe_profiler.sh` (to be created) for implementation.

---

## Files Analysis Summary

| Script | Issue | Status |
|--------|-------|--------|
| `simple_profiler.sh` | 404 errors - missing `--profiler-config` | BROKEN |
| `run_vllm_pytorch_profiler.sh` | NCCL threading error | BROKEN |
| `profile_vllm_mixed.sh` | NCCL threading error | BROKEN |
| `profile_vllm_server.py` | Profiles wrong process | BROKEN |
| `manual_profiler_control.py` | Uses wrong API endpoints | BROKEN |

---

## Technical Details

### NCCL Threading Error Deep Dive

```
ERROR: External init callback must run in same thread as registerClient
```

This comes from NVIDIA's NCCL library (libnccl.so). NCCL maintains thread-local state for:
- CUDA context management
- Communication handle registration
- Callback registration

When PyTorch profiler starts, it:
1. Registers a client with NCCL (in thread A - API server thread)
2. Later tries to init callback from the EngineCore process (thread B)
3. NCCL detects different thread IDs and aborts

This is a known issue with PyTorch profiler + multiprocessing + NCCL.

### vLLM Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM Process                             │
│  ┌──────────────┐         ┌──────────────────────────────┐  │
│  │ API Server   │────────▶│  AsyncLLM (async_llm.py)     │  │
│  │ (pid=6945)   │  IPC    │  - Routes requests           │  │
│  └──────────────┘         └──────────────────────────────┘  │
│                                      │                      │
│                                      ▼                      │
│                          ┌──────────────────────────────┐  │
│                          │  EngineCore (core.py)        │  │
│                          │  (pid=7083, separate process)│  │
│                          │  - Runs model inference      │  │
│                          │  - CUDA/NCCL operations      │  │
│                          └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

The PyTorch profiler tries to profile both processes but NCCL initialization happens in the EngineCore while the profiler is registered from AsyncLLM - different threads/processes.

---

## Affected Log Files

- `pytorch_mixed_20260208_034857/vllm_server.log` - NCCL threading error
- `api_profile_20260208_030913/vllm_server.log` - Same NCCL error
- `simple_profile_20260208_034542/vllm_server.log` - 404 errors for profiler APIs
