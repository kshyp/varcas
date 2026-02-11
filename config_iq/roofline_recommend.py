#!/usr/bin/env python3
"""
Roofline‑Driven Hardware Recommendation Engine – Static Analysis Module
-----------------------------------------------------------------------
Modular CLI tools for model profiling, workload definition, hardware catalog,
roofline estimation, and configuration search.

Dependencies:
    pip install click huggingface_hub requests
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import click
import requests
from huggingface_hub import model_info as hf_model_info

# ----------------------------------------------------------------------
# 1. Hardware Catalog (Static – extendable)
# ----------------------------------------------------------------------
HARDWARE_CATALOG = [
    # AWS instances (example subset)
    {
        "id": "aws:g5.xlarge",
        "cloud": "aws",
        "instance_type": "g5.xlarge",
        "gpu_model": "A10G",
        "gpu_count": 1,
        "gpu_memory_gb": 24,
        "peak_tflops_fp16": 125,        # per GPU
        "peak_tflops_tc": 250,          # per GPU (tensor cores)
        "peak_bandwidth_gbps": 600,     # per GPU
        "interconnect_type": "pcie4",
        "interconnect_bw_gbps": 32,
        "price_hour": 1.21,
        "region": "us-east-1",
    },
    {
        "id": "aws:g5.12xlarge",
        "cloud": "aws",
        "instance_type": "g5.12xlarge",
        "gpu_model": "A10G",
        "gpu_count": 4,
        "gpu_memory_gb": 24,
        "peak_tflops_fp16": 125,
        "peak_tflops_tc": 250,
        "peak_bandwidth_gbps": 600,
        "interconnect_type": "pcie4",   # A10G does not have NVLink
        "interconnect_bw_gbps": 32,
        "price_hour": 4.56,
        "region": "us-east-1",
    },
    {
        "id": "aws:g5.48xlarge",
        "cloud": "aws",
        "instance_type": "g5.48xlarge",
        "gpu_model": "A10G",
        "gpu_count": 8,
        "gpu_memory_gb": 24,
        "peak_tflops_fp16": 125,
        "peak_tflops_tc": 250,
        "peak_bandwidth_gbps": 600,
        "interconnect_type": "pcie4",
        "interconnect_bw_gbps": 32,
        "price_hour": 8.15,
        "region": "us-east-1",
    },
    {
        "id": "aws:p4d.24xlarge",
        "cloud": "aws",
        "instance_type": "p4d.24xlarge",
        "gpu_model": "A100",
        "gpu_count": 8,
        "gpu_memory_gb": 40,
        "peak_tflops_fp16": 312,        # without sparsity
        "peak_tflops_tc": 624,          # with tensor cores
        "peak_bandwidth_gbps": 1550,
        "interconnect_type": "nvlink",
        "interconnect_bw_gbps": 600,
        "price_hour": 32.77,
        "region": "us-east-1",
    },
    {
        "id": "aws:p5.48xlarge",
        "cloud": "aws",
        "instance_type": "p5.48xlarge",
        "gpu_model": "H100",
        "gpu_count": 8,
        "gpu_memory_gb": 80,
        "peak_tflops_fp16": 989,        # with FP16
        "peak_tflops_tc": 1979,         # with FP8
        "peak_bandwidth_gbps": 2000,
        "interconnect_type": "nvlink",
        "interconnect_bw_gbps": 900,
        "price_hour": 98.32,
        "region": "us-east-1",
    },
]

# ----------------------------------------------------------------------
# 2. Workload Profiles (from comprehensive table)
# ----------------------------------------------------------------------
WORKLOAD_PROFILES = {
    "chatbot": {
        "description": "Conversational AI, real‑time",
        "input_tokens": (256, 512),
        "output_tokens": (128, 256),
        "batch_size": 1,
        "concurrency_qps": 5,
        "sla_ttft_ms": 200,
        "sla_tpot_ms": 40,
        "sla_p95_ratio": 2.0,
    },
    "code-completion": {
        "description": "Inline code suggestions",
        "input_tokens": (50, 200),
        "output_tokens": (10, 30),
        "batch_size": 1,
        "concurrency_qps": 2,
        "sla_ttft_ms": 100,
        "sla_tpot_ms": 20,
        "sla_p95_ratio": 2.0,
    },
    "translation": {
        "description": "Real‑time translation",
        "input_tokens": (50, 100),
        "output_tokens": (50, 100),
        "batch_size": 1,
        "concurrency_qps": 5,
        "sla_ttft_ms": 300,
        "sla_tpot_ms": 50,
        "sla_p95_ratio": 2.0,
    },
    "summarization": {
        "description": "Interactive summarization",
        "input_tokens": (1024, 2048),
        "output_tokens": (128, 512),
        "batch_size": 1,
        "concurrency_qps": 1,
        "sla_ttft_ms": 500,
        "sla_tpot_ms": 50,
        "sla_p95_ratio": 2.0,
    },
    "batch": {
        "description": "Offline batch inference",
        "input_tokens": (512, 2048),
        "output_tokens": (64, 512),
        "batch_size": 64,
        "concurrency_qps": None,
        "sla_ttft_ms": None,
        "sla_tpot_ms": None,
        "sla_p95_ratio": None,
    },
    "embedding": {
        "description": "Embedding generation",
        "input_tokens": (128, 512),
        "output_tokens": 0,
        "batch_size": 128,
        "concurrency_qps": None,
        "sla_ttft_ms": None,
        "sla_tpot_ms": None,
        "sla_p95_ratio": None,
    },
}

# ----------------------------------------------------------------------
# 3. Calibration Database (Derating Factors)
# ----------------------------------------------------------------------
# Source: MLPerf Inference, internal benchmarks, public blogs
DERATING_FACTORS = {
    "A100": {
        "prefill_compute": 0.75,
        "decode_memory": 0.55,
    },
    "H100": {
        "prefill_compute": 0.80,
        "decode_memory": 0.60,
    },
    "A10G": {
        "prefill_compute": 0.65,
        "decode_memory": 0.45,
    },
}

# Communication overhead (added to latency, not overlapped)
COMM_OVERHEAD = {
    1: 1.0,   # no overhead
    2: 1.10,
    4: 1.20,
    8: 1.35,
}

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def get_dtype_bytes(dtype: str) -> int:
    """Return bytes per parameter for given dtype."""
    return {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "fp8": 1,
    }.get(dtype, 2)  # default to FP16

def compute_kv_cache_bytes_per_token(
    n_layers: int, hidden_size: int, dtype_bytes: int
) -> int:
    """KV cache size (bytes) per token (both K and V)."""
    return 2 * n_layers * hidden_size * dtype_bytes

# ----------------------------------------------------------------------
# CLI Commands
# ----------------------------------------------------------------------
@click.group()
def cli():
    """Roofline‑based hardware recommendation toolkit."""
    pass

# ---------- model-info -------------------------------------------------
@cli.command("model-info")
@click.option("--model-id", required=True, help="Hugging Face model ID")
@click.option("--revision", default="main", help="Model revision")
@click.option("--trust-remote-code", is_flag=True, help="Trust remote code")
def model_info_cmd(model_id: str, revision: str, trust_remote_code: bool):
    """Fetch model architecture and parameter count from Hugging Face Hub."""
    try:
        info = hf_model_info(model_id, revision=revision)
        config = getattr(info, "config", {})
        if not config:
            # fallback: download config.json directly
            url = f"https://huggingface.co/{model_id}/raw/{revision}/config.json"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            config = resp.json()

        # Extract parameter count – may be stored directly or computed
        n_params = config.get("num_parameters")
        if n_params is None:
            # estimate from architecture
            if "num_hidden_layers" in config and "hidden_size" in config:
                n_layers = config["num_hidden_layers"]
                hidden_size = config["hidden_size"]
                intermediate_size = config.get("intermediate_size", 4 * hidden_size)
                vocab_size = config.get("vocab_size", 32000)
                # decoder-only transformer (no encoder)
                params_per_layer = (
                    hidden_size * hidden_size * 4 +  # QKV + O
                    hidden_size * intermediate_size * 2 +  # FFN up & down
                    hidden_size + intermediate_size  # biases (approx)
                )
                n_params = (
                    vocab_size * hidden_size +  # embedding
                    n_layers * params_per_layer +
                    2 * hidden_size +  # final LN
                    hidden_size * vocab_size  # lm_head
                )
                # rough approximation, but good enough for roofline
                n_params = int(n_params * 1.05)  # safety factor

        output = {
            "model_id": model_id,
            "parameter_count": n_params,
            "dtype": config.get("torch_dtype", "float16"),
            "architecture": config.get("architectures", ["Unknown"])[0],
            "num_hidden_layers": config.get("num_hidden_layers"),
            "hidden_size": config.get("hidden_size"),
            "vocab_size": config.get("vocab_size"),
            "intermediate_size": config.get("intermediate_size"),
            "model_type": config.get("model_type"),
        }
        click.echo(json.dumps(output, indent=2))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

# ---------- workload-defaults -----------------------------------------
@cli.command("workload-defaults")
@click.option("--deployment-type", required=True, type=click.Choice(list(WORKLOAD_PROFILES.keys())))
@click.option("--input-len", type=int, help="Override input length")
@click.option("--output-len", type=int, help="Override output length")
@click.option("--batch-size", type=int, help="Override batch size")
def workload_defaults_cmd(deployment_type: str, input_len: int, output_len: int, batch_size: int):
    """Return workload parameters for a given deployment type."""
    profile = WORKLOAD_PROFILES[deployment_type].copy()
    if input_len:
        profile["input_tokens"] = (input_len, input_len)
    if output_len:
        profile["output_tokens"] = (output_len, output_len)
    if batch_size:
        profile["batch_size"] = batch_size

    # Use typical (midpoint) values for static analysis
    if isinstance(profile["input_tokens"], tuple):
        profile["input_tokens_typical"] = (profile["input_tokens"][0] + profile["input_tokens"][1]) // 2
    else:
        profile["input_tokens_typical"] = profile["input_tokens"]
    if isinstance(profile["output_tokens"], tuple):
        profile["output_tokens_typical"] = (profile["output_tokens"][0] + profile["output_tokens"][1]) // 2
    else:
        profile["output_tokens_typical"] = profile["output_tokens"]

    click.echo(json.dumps(profile, indent=2, default=str))

# ---------- hardware-ls ------------------------------------------------
@cli.command("hardware-ls")
@click.option("--cloud", default="aws", help="Cloud provider (aws/gcp/azure)")
@click.option("--gpu-type", help="Filter by GPU model (e.g., A100, H100, A10G)")
@click.option("--min-memory-gb", type=int, help="Minimum GPU memory per GPU")
@click.option("--max-price", type=float, help="Maximum price per hour")
@click.option("--region", default="us-east-1", help="Cloud region")
def hardware_ls_cmd(cloud: str, gpu_type: str, min_memory_gb: int, max_price: float, region: str):
    """List available hardware from catalog."""
    results = []
    for hw in HARDWARE_CATALOG:
        if hw["cloud"] != cloud:
            continue
        if gpu_type and gpu_type.lower() not in hw["gpu_model"].lower():
            continue
        if min_memory_gb and hw["gpu_memory_gb"] < min_memory_gb:
            continue
        if max_price and hw["price_hour"] > max_price:
            continue
        results.append(hw)

    if not results:
        click.echo("[]")
    else:
        click.echo(json.dumps(results, indent=2))

# ---------- roofline-estimate -----------------------------------------
@cli.command("roofline-estimate")
@click.option("--model-id", required=True)
@click.option("--deployment-type", required=True, type=click.Choice(list(WORKLOAD_PROFILES.keys())))
@click.option("--hardware-id", required=True, help="Hardware ID from catalog")
@click.option("--tp-size", type=int, default=1, help="Tensor parallelism size")
@click.option("--batch-size", type=int, help="Override batch size")
@click.option("--input-len", type=int, help="Override input length")
@click.option("--output-len", type=int, help="Override output length")
@click.option("--util-compute", type=float, help="Override compute utilization")
@click.option("--util-memory", type=float, help="Override memory utilization")
def roofline_estimate_cmd(
    model_id: str,
    deployment_type: str,
    hardware_id: str,
    tp_size: int,
    batch_size: int,
    input_len: int,
    output_len: int,
    util_compute: float,
    util_memory: float,
):
    """Estimate TTFT, TPOT, and throughput for a single configuration."""
    # 1. Fetch model info
    try:
        info = hf_model_info(model_id)
        config = getattr(info, "config", {})
        if not config:
            url = f"https://huggingface.co/{model_id}/raw/main/config.json"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            config = resp.json()
    except Exception as e:
        click.echo(f"Error fetching model info: {e}", err=True)
        sys.exit(1)

    n_params = config.get("num_parameters")
    if n_params is None:
        # fallback approximation
        click.echo("Parameter count not found; using rough estimate.", err=True)
        n_params = 7_000_000_000  # default to 7B

    dtype = config.get("torch_dtype", "float16")
    dtype_bytes = get_dtype_bytes(dtype)
    n_layers = config.get("num_hidden_layers", 32)
    hidden_size = config.get("hidden_size", 4096)

    # 2. Get workload defaults
    profile = WORKLOAD_PROFILES[deployment_type]
    if not batch_size:
        batch_size = profile["batch_size"]
    if not input_len:
        input_len = (profile["input_tokens"][0] + profile["input_tokens"][1]) // 2
    if not output_len:
        output_len = (profile["output_tokens"][0] + profile["output_tokens"][1]) // 2

    # 3. Find hardware spec
    hw = next((h for h in HARDWARE_CATALOG if h["id"] == hardware_id), None)
    if not hw:
        click.echo(f"Hardware ID {hardware_id} not found", err=True)
        sys.exit(1)

    gpu_model = hw["gpu_model"]
    gpu_count = hw["gpu_count"]
    peak_tflops_per_gpu = hw["peak_tflops_fp16"]
    peak_bw_per_gpu = hw["peak_bandwidth_gbps"]
    interconnect_bw = hw["interconnect_bw_gbps"]

    # Derating factors
    util_comp = util_compute or DERATING_FACTORS.get(gpu_model, {}).get("prefill_compute", 0.7)
    util_mem = util_memory or DERATING_FACTORS.get(gpu_model, {}).get("decode_memory", 0.5)

    # Effective per‑GPU rates
    eff_tflops = peak_tflops_per_gpu * util_comp
    eff_bw = peak_bw_per_gpu * util_mem

    # 4. Communication overhead factor
    comm_factor = COMM_OVERHEAD.get(tp_size, 1.0)

    # 5. Compute FLOPs and memory traffic
    weights_bytes = n_params * dtype_bytes
    kv_cache_bytes_per_token = compute_kv_cache_bytes_per_token(n_layers, hidden_size, dtype_bytes)

    # Prefill
    prefill_flops = 2 * n_params * input_len
    # Each GPU handles 1/tp of the weights and KV cache
    weights_per_gpu = weights_bytes / tp_size
    kv_cache_per_gpu = kv_cache_bytes_per_token / tp_size

    # Memory bytes per GPU for prefill: each GPU reads its shard of weights (once per batch)
    prefill_bytes_per_gpu = weights_per_gpu + kv_cache_per_gpu * input_len
    # Compute time per GPU
    prefill_compute_s = prefill_flops / tp_size / (eff_tflops * 1e12)
    prefill_memory_s = prefill_bytes_per_gpu / (eff_bw * 1e9)
    prefill_latency_s = max(prefill_compute_s, prefill_memory_s) * comm_factor
    ttft_ms = prefill_latency_s * 1000

    # Decode (per token)
    decode_flops_per_token = 2 * n_params
    decode_flops_per_gpu = decode_flops_per_token / tp_size
    decode_bytes_per_gpu = weights_per_gpu / batch_size + kv_cache_per_gpu  # per token

    decode_compute_s = decode_flops_per_gpu / (eff_tflops * 1e12)
    decode_memory_s = decode_bytes_per_gpu / (eff_bw * 1e9)
    decode_latency_s = max(decode_compute_s, decode_memory_s) * comm_factor
    tpot_ms = decode_latency_s * 1000

    # Throughput (tokens/sec) – simplified, assumes continuous batching not modelled
    total_time_per_req_s = ttft_ms / 1000 + output_len * tpot_ms / 1000
    throughput_req_per_sec = 1 / total_time_per_req_s if total_time_per_req_s > 0 else 0
    throughput_tokens_per_sec = throughput_req_per_sec * (input_len + output_len)

    # Bottleneck
    bottleneck = "compute" if prefill_compute_s > prefill_memory_s else "memory"

    output = {
        "model_id": model_id,
        "hardware_id": hardware_id,
        "tp_size": tp_size,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "ttft_ms": round(ttft_ms, 2),
        "tpot_ms": round(tpot_ms, 2),
        "throughput_req_per_sec": round(throughput_req_per_sec, 2),
        "throughput_tokens_per_sec": round(throughput_tokens_per_sec, 2),
        "prefill_compute_ms": round(prefill_compute_s * 1000, 2),
        "prefill_memory_ms": round(prefill_memory_s * 1000, 2),
        "decode_compute_ms": round(decode_compute_s * 1000, 2),
        "decode_memory_ms": round(decode_memory_s * 1000, 2),
        "bottleneck": bottleneck,
        "comm_overhead_factor": comm_factor,
        "utilization_assumed": {"compute": util_comp, "memory": util_mem},
    }
    click.echo(json.dumps(output, indent=2))

# ---------- search-configs --------------------------------------------
@cli.command("search-configs")
@click.option("--model-id", required=True)
@click.option("--deployment-type", required=True, type=click.Choice(list(WORKLOAD_PROFILES.keys())))
@click.option("--sla-ttft", type=int, help="Override SLA TTFT (ms)")
@click.option("--sla-tpot", type=int, help="Override SLA TPOT (ms)")
@click.option("--safety-margin", type=float, default=0.3, help="Fraction to tighten SLA (0.0-1.0)")
@click.option("--max-headroom", type=float, default=0.6, help="Reject configs with headroom > this (0.0-1.0+)")
@click.option("--max-price", type=float, help="Maximum $/hr")
@click.option("--output-format", type=click.Choice(["json", "table"]), default="table")
def search_configs_cmd(
    model_id: str,
    deployment_type: str,
    sla_ttft: int,
    sla_tpot: int,
    safety_margin: float,
    max_headroom: float,
    max_price: float,
    output_format: str,
):
    """Search and rank hardware configurations that meet SLA with healthy headroom."""
    # 1. Get workload defaults and SLA
    profile = WORKLOAD_PROFILES[deployment_type]
    target_ttft = sla_ttft or profile.get("sla_ttft_ms")
    target_tpot = sla_tpot or profile.get("sla_tpot_ms")

    if target_ttft is None or target_tpot is None:
        click.echo("Error: SLA not defined for this deployment type. Please provide --sla-ttft/--sla-tpot.", err=True)
        sys.exit(1)

    # Apply safety margin (tighten SLA)
    adjusted_ttft = target_ttft * (1 - safety_margin)
    adjusted_tpot = target_tpot * (1 - safety_margin)

    # 2. Get model info (parameter count, etc.)
    try:
        info = hf_model_info(model_id)
        config = getattr(info, "config", {})
        if not config:
            url = f"https://huggingface.co/{model_id}/raw/main/config.json"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            config = resp.json()
    except Exception as e:
        click.echo(f"Error fetching model info: {e}", err=True)
        sys.exit(1)

    n_params = config.get("num_parameters")
    if n_params is None:
        n_params = 7_000_000_000
    dtype = config.get("torch_dtype", "float16")
    dtype_bytes = get_dtype_bytes(dtype)
    n_layers = config.get("num_hidden_layers", 32)
    hidden_size = config.get("hidden_size", 4096)

    weights_bytes = n_params * dtype_bytes
    kv_cache_bytes_per_token = compute_kv_cache_bytes_per_token(n_layers, hidden_size, dtype_bytes)

    # 3. Filter hardware catalog
    hw_list = HARDWARE_CATALOG
    if max_price:
        hw_list = [h for h in hw_list if h["price_hour"] <= max_price]

    feasible_configs = []

    # 4. Enumerate TP sizes (1,2,4,8 if GPU count allows)
    tp_sizes = [1, 2, 4, 8]

    for hw in hw_list:
        gpu_count = hw["gpu_count"]
        gpu_model = hw["gpu_model"]
        peak_tflops = hw["peak_tflops_fp16"]
        peak_bw = hw["peak_bandwidth_gbps"]
        price = hw["price_hour"]

        # Derating factors
        util_comp = DERATING_FACTORS.get(gpu_model, {}).get("prefill_compute", 0.7)
        util_mem = DERATING_FACTORS.get(gpu_model, {}).get("decode_memory", 0.5)
        eff_tflops = peak_tflops * util_comp
        eff_bw = peak_bw * util_mem

        for tp in tp_sizes:
            if tp > gpu_count:
                continue
            # Check memory footprint: weights + KV cache (max seq len)
            # Estimate max seq len from workload (output+input) – conservative
            max_seq_len = profile.get("input_tokens", (512,512))[1] + profile.get("output_tokens", (128,128))[1]
            kv_cache_total = kv_cache_bytes_per_token * max_seq_len / tp  # per GPU
            weights_per_gpu = weights_bytes / tp
            mem_footprint_gb = (weights_per_gpu + kv_cache_total) / (1024**3)
            if mem_footprint_gb > hw["gpu_memory_gb"] * 0.9:  # 90% utilization max
                continue

            # Estimate performance using roofline estimator (reuse logic)
            batch_size = profile["batch_size"]
            input_len = (profile["input_tokens"][0] + profile["input_tokens"][1]) // 2
            output_len = (profile["output_tokens"][0] + profile["output_tokens"][1]) // 2

            # Prefill
            prefill_flops = 2 * n_params * input_len
            weights_per_gpu = weights_bytes / tp
            kv_cache_per_gpu = kv_cache_bytes_per_token / tp
            prefill_bytes_per_gpu = weights_per_gpu + kv_cache_per_gpu * input_len
            prefill_compute_s = prefill_flops / tp / (eff_tflops * 1e12)
            prefill_memory_s = prefill_bytes_per_gpu / (eff_bw * 1e9)
            comm_factor = COMM_OVERHEAD.get(tp, 1.0)
            ttft_s = max(prefill_compute_s, prefill_memory_s) * comm_factor
            ttft_ms = ttft_s * 1000

            # Decode
            decode_flops_per_token = 2 * n_params
            decode_flops_per_gpu = decode_flops_per_token / tp
            decode_bytes_per_gpu = weights_per_gpu / batch_size + kv_cache_per_gpu
            decode_compute_s = decode_flops_per_gpu / (eff_tflops * 1e12)
            decode_memory_s = decode_bytes_per_gpu / (eff_bw * 1e9)
            tpot_s = max(decode_compute_s, decode_memory_s) * comm_factor
            tpot_ms = tpot_s * 1000

            # Check adjusted SLA
            if ttft_ms <= adjusted_ttft and tpot_ms <= adjusted_tpot:
                # Compute headroom relative to original SLA
                headroom_ttft = (target_ttft - ttft_ms) / ttft_ms if ttft_ms > 0 else 0
                headroom_tpot = (target_tpot - tpot_ms) / tpot_ms if tpot_ms > 0 else 0
                headroom = min(headroom_ttft, headroom_tpot)

                # Reject if headroom exceeds max_headroom (overprovisioned)
                if headroom > max_headroom:
                    continue

                # Throughput
                total_time_per_req_s = ttft_ms/1000 + output_len * tpot_ms/1000
                throughput_req = 1 / total_time_per_req_s if total_time_per_req_s > 0 else 0

                feasible_configs.append({
                    "hardware_id": hw["id"],
                    "instance_type": hw["instance_type"],
                    "gpu_model": hw["gpu_model"],
                    "gpu_count": hw["gpu_count"],
                    "tp_size": tp,
                    "batch_size": batch_size,
                    "ttft_ms": round(ttft_ms, 2),
                    "tpot_ms": round(tpot_ms, 2),
                    "headroom_pct": round(headroom * 100, 1),
                    "throughput_req_per_sec": round(throughput_req, 2),
                    "price_hour": hw["price_hour"],
                    "cost_per_1M_tokens": round((hw["price_hour"] / (throughput_req * 3600)) * 1e6, 2) if throughput_req > 0 else None,
                })

    # Rank by price, then headroom (lower headroom preferred within acceptable range)
    feasible_configs.sort(key=lambda x: (x["price_hour"], x["headroom_pct"]))

    if output_format == "json":
        click.echo(json.dumps(feasible_configs, indent=2))
    else:
        # Simple ASCII table
        if not feasible_configs:
            click.echo("No feasible configurations found.")
            return
        header = f"{'Instance':<20} {'GPU':<8} {'TP':<4} {'TTFT(ms)':<10} {'TPOT(ms)':<10} {'Headroom':<10} {'Req/s':<8} {'$/hr':<8} {'$/1M':<10}"
        click.echo(header)
        click.echo("-" * len(header))
        for c in feasible_configs[:10]:  # top 10
            click.echo(
                f"{c['instance_type']:<20} {c['gpu_count']}x{c['gpu_model']:<6} "
                f"{c['tp_size']:<4} {c['ttft_ms']:<10} {c['tpot_ms']:<10} "
                f"{c['headroom_pct']:<10} {c['throughput_req_per_sec']:<8} "
                f"{c['price_hour']:<8} {c['cost_per_1M_tokens']:<10}"
            )

# ----------------------------------------------------------------------
if __name__ == "__main__":
    cli()
