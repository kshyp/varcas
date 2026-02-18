# System Optimizer

OS-level optimization scripts for maximizing vLLM inference performance on Linux systems.

## Overview

These optimizations tune the Linux kernel and system settings to reduce latency, prevent stalls, and improve GPU memory allocation for LLM inference workloads.

## Linux Optimizations

### tune.sh

Core system tuning script for VM-based deployments:

```bash
sudo ./linux/tune.sh
```

**Optimizations applied:**

1. **Disable swap** (`vm.swappiness=0`)
   - Prevents disk I/O stalls during inference
   - Ensures consistent latency by avoiding swap thrashing

2. **Increase locked memory** (`ulimit -l unlimited`)
   - Allows CUDA to lock memory pages
   - Prevents GPU memory allocation failures
   - Reduces page fault overhead

## Usage

### One-time Setup

```bash
cd optimizer/linux
sudo ./tune.sh
```

### Persistent Configuration

To make settings persistent across reboots, add to `/etc/sysctl.conf`:

```bash
# Disable swap
vm.swappiness=0

# Optional: Increase shared memory
kernel.shmmax=68719476736
kernel.shmall=4294967296
```

And to `/etc/security/limits.conf`:

```bash
# Allow unlimited locked memory for CUDA
* soft memlock unlimited
* hard memlock unlimited
```

### Docker/Container Environments

For containerized deployments, set at runtime:

```bash
docker run --ulimit memlock=-1:-1 ...
```

Or in docker-compose:

```yaml
services:
  vllm:
    ulimits:
      memlock: -1
    sysctls:
      - vm.swappiness=0
```

## Why These Matter for vLLM

| Setting | Default | Optimized | Impact |
|---------|---------|-----------|--------|
| vm.swappiness | 60 | 0 | Prevents 100ms+ latency spikes from swap I/O |
| memlock limit | 64KB | unlimited | Enables large GPU memory allocations |

### Swap Impact on LLM Inference

When `vm.swappiness > 0`:
1. Kernel may swap GPU-mapped pages under memory pressure
2. Causes unpredictable latency spikes (50-500ms)
3. Violates real-time SLA requirements

### Locked Memory for CUDA

CUDA uses locked (pinned) memory for:
- GPU→CPU transfers
- Unified memory
- NCCL communication buffers

Default limits (64KB) are insufficient for multi-GPU inference.

## Cloud Provider Specific Notes

### AWS EC2
- Settings apply to all instance types
- Consider using dedicated instances for consistent performance
- Additional optimizations available via Elastic Fabric Adapter (EFA)

### Google Cloud Platform
- Works with all GPU instance types (A2, A3, G2)
- No additional configuration needed for NVLink

### Azure
- Apply to NC-series and ND-series VMs
- May need to disable hypervisor scheduling for bare-metal performance

## Verification

Check settings after tuning:

```bash
# Check swap settings
cat /proc/sys/vm/swappiness
# Expected: 0

# Check memory limits
ulimit -l
# Expected: unlimited (or very large number)

# Check current swap usage
free -h
swapon -s
```

## Advanced Optimizations (Optional)

### CPU Governor

For CPU-bound preprocessing:

```bash
# Set performance governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### NUMA Optimization

For multi-socket systems:

```bash
# Run vLLM on specific NUMA node
numactl --cpunodebind=0 --membind=0 python -m vllm.entrypoints.openai.api_server ...
```

### IRQ Affinity

Distribute network interrupts:

```bash
# Distribute NIC interrupts across CPUs
./set_irq_affinity.sh eth0
```

## Troubleshooting

### "Cannot allocate memory" errors
- Check `ulimit -l` - should be unlimited
- Verify sufficient system RAM (not just GPU VRAM)
- Check dmesg for OOM killer activity

### Still seeing swap activity
```bash
# Disable swap completely
sudo swapoff -a

# Or check for zram/zswap
systemctl status zramswap
```

### Permission denied
```bash
# Run with sudo
sudo ./tune.sh

# Or apply settings manually with sudo
sudo sysctl -w vm.swappiness=0
sudo ulimit -l unlimited  # Note: ulimit may not work with sudo
```

## Safety

These optimizations are generally safe for inference workloads:
- No data loss risk
- Reversible (reboot restores defaults)
- Well-established practices in HPC/GPU computing

However:
- **Don't disable swap on systems with <32GB RAM** (unless vLLM is the only service)
- **Test before production deployment**
- **Monitor OOM events** after disabling swap
