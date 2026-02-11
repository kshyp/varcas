#!/usr/bin/env python3
"""
Profile vLLM using the LLM class directly (single process)
This avoids the NCCL threading issues with the API server.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Configuration
OUTPUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/sujatha/varcas/profilers/pytorch/output/llm_profile_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {OUTPUT_DIR}")

# Set profiling directory
os.environ["VLLM_TORCH_PROFILER_DIR"] = str(OUTPUT_DIR)

# Import vLLM
from vllm import LLM, SamplingParams

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# Load the harness requests
sys.path.insert(0, "/home/sujatha/varcas/benchmark_harness")

def load_mixed_requests():
    """Generate mixed workload requests similar to the harness."""
    chat_prompts = [
        "Explain machine learning in simple terms",
        "What are the benefits of exercise?",
        "How do computers work?",
        "Tell me about quantum physics",
        "What is artificial intelligence?",
    ]
    
    rag_prompts = [
        "Based on the following context about neural networks, explain backpropagation",
        "Given this text about climate change, what are the main causes?",
        "From the provided information about space exploration, describe Mars missions",
    ]
    
    code_prompts = [
        "Write a Python function to sort a list",
        "Create a JavaScript function to reverse a string",
        "Implement a binary search algorithm in Python",
    ]
    
    # Mix: 60% chat, 30% RAG, 10% code
    import random
    random.seed(42)
    
    requests = []
    for _ in range(100):  # 100 requests
        r = random.random()
        if r < 0.6:
            prompt = random.choice(chat_prompts)
        elif r < 0.9:
            prompt = random.choice(rag_prompts)
        else:
            prompt = random.choice(code_prompts)
        
        requests.append({
            "prompt": prompt,
            "max_tokens": random.randint(20, 100)
        })
    
    return requests

def main():
    log("Initializing vLLM with profiler...")
    
    # Initialize LLM with profiling enabled via env var
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="half",
        max_model_len=1024,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    
    log("vLLM loaded. Starting profiler...")
    
    # Start profiling
    llm.start_profile()
    log("Profiler started!")
    
    # Load requests
    requests = load_mixed_requests()
    log(f"Loaded {len(requests)} requests")
    
    # Warmup
    log("Warming up...")
    warmup_params = SamplingParams(max_tokens=10, temperature=0.7)
    for i in range(3):
        llm.generate("Hello", warmup_params)
    log("Warmup complete")
    
    # Run benchmark
    log("Running benchmark...")
    start_time = time.time()
    
    for i, req in enumerate(requests):
        sampling_params = SamplingParams(
            max_tokens=req["max_tokens"],
            temperature=0.7
        )
        output = llm.generate(req["prompt"], sampling_params)
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            log(f"  Completed {i+1}/{len(requests)} requests ({elapsed:.1f}s)")
    
    elapsed = time.time() - start_time
    log(f"Benchmark complete: {len(requests)} requests in {elapsed:.1f}s ({len(requests)/elapsed:.2f} req/s)")
    
    # Stop profiling
    log("Stopping profiler...")
    llm.stop_profile()
    log("Profiler stopped!")
    
    # Give time for traces to be written
    time.sleep(3)
    
    # Check output
    log("Checking output files...")
    trace_files = list(OUTPUT_DIR.glob("*.json*")) + list(OUTPUT_DIR.glob("*.trace*")) + list(OUTPUT_DIR.glob("*.pt*"))
    
    if trace_files:
        log(f"Found {len(trace_files)} trace files:")
        for f in trace_files:
            size = f.stat().st_size
            log(f"  - {f.name} ({size/1024/1024:.2f} MB)")
    else:
        log("No trace files found in output directory")
        log(f"Contents of {OUTPUT_DIR}:")
        for f in OUTPUT_DIR.iterdir():
            log(f"  - {f.name}")
    
    log(f"Done! Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
