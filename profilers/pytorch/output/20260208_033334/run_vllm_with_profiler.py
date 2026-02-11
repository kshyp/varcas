import os
import sys
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, record_function
import subprocess
import time
import signal

# Configuration
output_dir = os.environ.get('VLLM_TORCH_PROFILER_DIR', '/tmp/pytorch_profile')
port = os.environ.get('VLLM_PORT', '8000')
model = os.environ.get('VLLM_MODEL', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
profile_duration = int(os.environ.get('PROFILE_DURATION', '300'))

print(f"[Profiler] Output directory: {output_dir}")
print(f"[Profiler] Profile duration: {profile_duration}s")

# Start vLLM as a subprocess
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '0'

cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", model,
    "--dtype", "half",
    "--max-model-len", "1024",
    "--gpu-memory-utilization", "0.9",
    "--port", port,
]

log_file = os.path.join(output_dir, "vllm_server.log")
with open(log_file, 'w') as lf:
    process = subprocess.Popen(
        cmd,
        stdout=lf,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid
    )

print(f"[Profiler] vLLM PID: {process.pid}")

# Write PID file for later cleanup
with open(os.path.join(output_dir, "vllm.pid"), 'w') as f:
    f.write(str(process.pid))

# Wait for vLLM to be ready
import urllib.request
import urllib.error

url = f"http://localhost:{port}/health"
start_time = time.time()
ready = False

while time.time() - start_time < 300:
    try:
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                ready = True
                break
    except:
        if process.poll() is not None:
            print("[Profiler] ERROR: vLLM process terminated unexpectedly")
            sys.exit(1)
        time.sleep(1)

if not ready:
    print("[Profiler] ERROR: vLLM failed to start within timeout")
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except:
        pass
    sys.exit(1)

print("[Profiler] vLLM is ready!")

# Keep the process running - the actual profiling will be done by external script
print(f"[Profiler] vLLM running. Waiting for termination signal...")

# Wait for signal or keep running
try:
    while process.poll() is None:
        time.sleep(1)
except KeyboardInterrupt:
    print("[Profiler] Interrupted, stopping vLLM...")
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=30)
    except:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except:
            pass

print("[Profiler] vLLM stopped.")
