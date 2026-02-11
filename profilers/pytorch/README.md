# PyTorch Profiler for vLLM

This directory contains a working solution for profiling vLLM with PyTorch profiler.

## ⚠️ Important Notice

**Many profiling approaches were tried and failed** due to vLLM 0.15.1's multi-process architecture and NCCL threading constraints. See `CRASH_ANALYSIS.md` for details on what was tried and why it failed.

## The Working Solution

The only reliable method is **`safe_profiler.sh`**, which:
- Uses continuous profiling (no start/stop API calls that trigger NCCL threading errors)
- Disables CUDA graphs (`--enforce-eager`) to avoid interference
- Disables stack tracing (`torch_profiler_with_stack=false`) to avoid thread safety issues

## Quick Start

```bash
cd ~/varcas/profilers/pytorch
./safe_profiler.sh
```

This will:
1. Start vLLM with PyTorch profiler enabled
2. Run warmup requests
3. Run the mixed benchmark (60% chat, 30% RAG, 10% code) for 120 seconds
4. Save traces to `output/safe_profile_YYYYMMDD_HHMMSS/`

## Usage

### Default run
```bash
./safe_profiler.sh
```

### Custom output directory
```bash
./safe_profiler.sh /path/to/custom/output
```

## Output Files

After profiling completes, check the output directory for:

| File | Description |
|------|-------------|
| `vllm_server.log` | vLLM server logs |
| `benchmark_output.log` | Benchmark output |
| `*.pt.trace.json.gz` | PyTorch profiler traces (if generated) |

## Viewing Results

### Chrome Tracing
1. Open Chrome/Edge and navigate to `chrome://tracing`
2. Load the `.json.gz` or `.json` files from the output directory

### Perfetto UI
1. Go to https://ui.perfetto.dev/
2. Open the trace files

## Why Other Approaches Failed

### ❌ Using `/start_profile` and `/stop_profile` APIs
- These APIs trigger NCCL initialization from different threads
- Causes fatal error: `External init callback must run in same thread as registerClient`
- Results in deadlocks/hangs

### ❌ External PyTorch Profiler
- Creating a `torch.profiler.profile()` in a separate script profiles the WRONG process
- vLLM runs in a subprocess, not the main script

### ❌ Stack Tracing Enabled
- `torch_profiler_with_stack=true` causes thread safety issues with vLLM's multi-process architecture

### ❌ CUDA Graphs + Profiling
- CUDA graphs (enabled by default) interfere with PyTorch profiler
- Must use `--enforce-eager` to disable them

See `CRASH_ANALYSIS.md` for full technical details.

## Script History (Deleted)

The following scripts were deleted because they were broken or unsafe:

| Script | Reason Deleted |
|--------|----------------|
| `simple_profiler.sh` | 404 errors - called profiling APIs without enabling profiler |
| `run_vllm_pytorch_profiler.sh` | NCCL threading error from using start/stop APIs |
| `profile_vllm_mixed.sh` | Missing `--enforce-eager`, could cause CUDA graph conflicts |
| `profile_vllm_api_controlled.sh` | Used broken profiling APIs |
| `profile_with_api_endpoints.sh` | Used broken profiling APIs |
| `run_profiling_with_api.sh` | Used broken profiling APIs |
| `run_profile_mixed.sh` | Called broken `profile_vllm_server.py` |
| `run_mixed_benchmark_with_profiler.sh` | No `--enforce-eager`, enabled stack tracing |
| `run_full_profiling.sh` | No `--enforce-eager`, enabled stack tracing |
| `run_pytorch_profiler.sh` | Incomplete/duplicate functionality |
| `start_vllm_and_profile.sh` | Complex, used unreliable methods |
| `start_vllm_profiling.sh` | Enabled stack tracing, no `--enforce-eager` |
| `start_vllm_with_torch_profiler.sh` | Enabled stack tracing, no `--enforce-eager` |
| `profile_vllm_server.py` | External profiler - profiles wrong process |
| `torch_profiler_runner.py` | External profiler - profiles wrong process |
| `manual_profiler_control.py` | Used wrong API endpoints |

## Troubleshooting

### No trace files generated
- Check `vllm_server.log` for errors
- Ensure vLLM started successfully
- Verify benchmark completed successfully

### vLLM fails to start
- Check if port 8000 is in use: `lsof -i :8000`
- Kill existing vLLM: `pkill -f vllm.entrypoints`
- Check GPU availability: `nvidia-smi`

### System still hangs
- The `safe_profiler.sh` should avoid this, but if it happens:
  - Kill vLLM: `pkill -9 -f vllm.entrypoints`
  - Check for zombie processes: `ps aux | grep vllm`
  - Consider using Nsight Systems instead (see `~/run_nsys_profile.sh`)
