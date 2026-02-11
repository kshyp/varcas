# NVIDIA Profilers for vLLM + Varcas Benchmark

This directory contains scripts for profiling vLLM inference using NVIDIA's profiling tools (Nsight Systems and Nsight Compute) with the Varcas load harness.

## Available Scripts

### 1. `run_nsys_vllm_varcas.sh` (Recommended - Most Reliable)
Profiles vLLM using **Nsight Systems** (nsys) during benchmark execution.

**Features:**
- Lower overhead than NCU
- Captures CUDA kernels, NVTX, and OS runtime
- Uses delay to skip model loading and profile only inference
- Recommended for general profiling

**Usage:**
```bash
./run_nsys_vllm_varcas.sh [profile_name]
```

**Examples:**
```bash
./run_nsys_vllm_varcas.sh chat_medium    # Chat workload at 20 RPS
./run_nsys_vllm_varcas.sh chat_low       # Chat workload at 5 RPS
./run_nsys_vllm_varcas.sh chat_high      # Chat workload at 50 RPS
./run_nsys_vllm_varcas.sh mixed          # Mixed workload (chat + rag + code)
```

**View Results:**
```bash
# Open in Nsight Systems UI
nsys-ui vllm_nsys_chat_medium_YYYYMMDD_HHMMSS.nsys-rep

# Export to CSV
nsys export -t cuda_kernel,cuda_api,nvtx -o output.csv vllm_nsys_chat_medium_YYYYMMDD_HHMMSS.nsys-rep

# Generate statistics
nsys stats --report cuda_kernel vllm_nsys_chat_medium_YYYYMMDD_HHMMSS.nsys-rep
```

---

### 2. `run_ncu_vllm_final.sh` (Detailed Kernel Analysis)
Profiles vLLM using **Nsight Compute** (ncu) during benchmark execution.

**Features:**
- Detailed kernel-level metrics
- Memory bandwidth and compute utilization
- Higher overhead than nsys (slower)
- Requires lower GPU memory setting

**Usage:**
```bash
./run_ncu_vllm_final.sh [profile_name]
```

**Examples:**
```bash
./run_ncu_vllm_final.sh chat_medium
./run_ncu_vllm_final.sh chat_low
```

**View Results:**
```bash
# Open in Nsight Compute UI
ncu-ui vllm_ncu_chat_medium_YYYYMMDD_HHMMSS.ncu-rep

# Export to CSV
ncu --import vllm_ncu_chat_medium_YYYYMMDD_HHMMSS.ncu-rep --page raw --csv > output.csv

# Print kernel summary
ncu --import vllm_ncu_chat_medium_YYYYMMDD_HHMMSS.ncu-rep --print-summary per-kernel
```

---

### 3. `run_single_profile.sh` (Existing - Nsys with Warmup)
Original script for profiling with nsys using delayed profiling.

**Usage:**
```bash
./run_single_profile.sh [profile_name] [profile_arg]
```

---

### 4. Other Existing Scripts

| Script | Purpose |
|--------|---------|
| `run_nsys_profile.sh` | Basic nsys profiling |
| `run_nsys_vllm_controlled.sh` | Controlled nsys profiling with triggers |
| `run_nsys_vllm_delay.sh` | Nsys with delay |
| `run_nsys_vllm_mixed.sh` | Mixed workload profiling |
| `run_nsys_vllm_simple.sh` | Simple nsys profiling |
| `run_nsys_vllm_warmup.sh` | Profiling with warmup |
| `run_all_benchmark_profiles.sh` | Run all benchmark profiles |

---

## Available Workload Profiles

The varcas load harness supports these profiles:

| Profile | Description | Target RPS |
|---------|-------------|------------|
| `chat_low` | Chat workload - low intensity | 5 RPS |
| `chat_medium` | Chat workload - medium intensity | 20 RPS |
| `chat_high` | Chat workload - high intensity | 50 RPS |
| `rag_small_low` | RAG with small context | 2 RPS |
| `rag_medium_medium` | RAG with medium context | 10 RPS |
| `code_low` | Code generation - low | 3 RPS |
| `code_medium` | Code generation - medium | 15 RPS |
| `mixed` | Mixed workloads | 25 RPS |
| `burst` | Bursty traffic pattern | Variable |

---

## Quick Start

1. **Make scripts executable:**
```bash
cd ~/varcas/profilers/nvidia
chmod +x *.sh
```

2. **Run a profile with nsys (recommended):**
```bash
./run_nsys_vllm_varcas.sh chat_medium
```

3. **View the results:**
```bash
nsys-ui vllm_nsys_chat_medium_*.nsys-rep
```

---

## Output Files

Each profiling run generates:

| File | Description |
|------|-------------|
| `vllm_nsys_*_YYYYMMDD_HHMMSS.nsys-rep` | Nsight Systems report |
| `vllm_nsys_*_YYYYMMDD_HHMMSS.sqlite` | SQLite database with trace data |
| `vllm_ncu_*_YYYYMMDD_HHMMSS.ncu-rep` | Nsight Compute report |
| `result_*_YYYYMMDD_HHMMSS.json` | Benchmark results (metrics, latencies) |

---

## Troubleshooting

### "No kernels were profiled" (NCU)
- NCU may have higher overhead; try nsys instead
- Check GPU memory availability with `nvidia-smi`

### "Server failed to start" (Out of Memory)
- NCU profiling consumes GPU memory
- The scripts use `--gpu-memory-utilization 0.6` for NCU
- Kill any zombie processes: `pkill -9 -f vllm`

### Slow Server Startup
- NCU profiling adds significant overhead during model loading
- This is expected - the server will take longer to start
- nsys with `--delay` option avoids this issue

---

## Tool Comparison

| Feature | Nsight Systems (nsys) | Nsight Compute (ncu) |
|---------|----------------------|---------------------|
| Overhead | Lower | Higher |
| Granularity | System-wide | Kernel-level |
| Best For | Timeline analysis, bottlenecks | Detailed kernel metrics |
| File Size | Larger | Smaller |
| Startup Impact | Minimal (with delay) | Significant |

---

## Additional Resources

- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Varcas Benchmark Harness](../../benchmark_harness/)
