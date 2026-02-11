#!/bin/bash
# gpu_vmstat.sh - GPU equivalent of vmstat 1

echo "timestamp, gpu_idx, gpu_name, temp_c, gpu_util%, mem_util%, mem_used_mb, mem_total_mb, power_w, pwr_limit_w"

while true; do
    nvidia-smi --query-gpu=timestamp,name,index,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit --format=csv,noheader,nounits
    sleep 1
done
