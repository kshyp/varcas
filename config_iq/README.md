Subcommand	Function
model-info	Fetches model config from Hugging Face, computes parameter count (if missing).
workload-defaults	Returns typical input/output lengths, batch size, and SLAs for a given deployment type.
hardware-ls	Lists hardware from built‑in catalog; filterable by GPU, price, etc.
roofline-estimate	Runs derated roofline for a single (model, hardware, TP) combo; outputs TTFT, TPOT, throughput, bottleneck.
search-configs	Main recommendation engine. Enumerates all (hardware, TP) combos, checks memory fit, computes predicted latency, applies safety margin, filters by headroom, ranks by cost.

✅ Key Features Implemented
Derating factors per GPU (A100, H100, A10G) for prefill (compute) and decode (memory).

Communication overhead as a multiplicative factor based on TP size.

Memory footprint check – weights + KV cache must fit in GPU memory (90% limit).

Safety margin – tightens SLA to avoid fragile configs.

Headroom capping – rejects over‑provisioned configs (>60% headroom).

Cost per 1M tokens – rough estimate for throughput‑oriented comparison.

Simple table output – human‑readable ranking.

📦 Extending the Tool
Hardware catalog – add more instance types (GCP, Azure) and update pricing via cloud APIs.

Derating factors – populate from a JSON file; allow user overrides.

Parallelism search – add pipeline parallelism (PP) and data parallelism (DP) for throughput workloads.

Continuous batching – improve throughput model using vLLM’s dynamic batching heuristics.

Model‑specific calibration – store per‑model utilization factors after benchmarking.
