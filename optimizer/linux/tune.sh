#For VMs - many other affinity settings are abstracted away

# 1. Disable swap (prevents disk I/O stalls)
sudo sysctl -w vm.swappiness=0

# 2. Increase locked memory for CUDA
ulimit -l unlimited
