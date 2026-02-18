"""Type definitions for the hardware sizing tool."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class WorkloadType(str, Enum):
    CHAT = "chat"
    RAG = "rag"
    CODE = "code"


class PricingModel(str, Enum):
    ONDEMAND = "ondemand"
    SPOT = "spot"


@dataclass
class GPUSpec:
    """GPU hardware specifications."""
    type: str
    count: int
    vram_gb: int
    compute_capability: str
    tensor_cores: bool
    fp16_tflops: float
    bf16_tflops: float
    fp8_tflops: float
    int8_tflops: float
    memory_bw_gbps: float


@dataclass
class HardwareConfig:
    """User-specified hardware configuration (DIY or VM).
    
    This is a lightweight spec that can be used for both:
    - Custom DIY configs (e.g., n1-standard-4 + 1x T4)
    - Catalog VM instances (converted to this format)
    """
    vcpus: int
    memory_gb: int
    gpu_type: str
    gpu_count: int
    gpu_vram_gb: int  # VRAM per GPU
    name: str = "custom"  # Optional name for the config
    
    @property
    def total_vram_gb(self) -> int:
        return self.gpu_vram_gb * self.gpu_count
    
    @property
    def max_vram_per_gpu_gb(self) -> int:
        return self.gpu_vram_gb


@dataclass
class VMInstance:
    """VM instance configuration."""
    name: str
    family: str
    vcpus: int
    memory_gb: int
    gpus: List[GPUSpec]
    ondemand_price_usd: float
    spot_price_usd: float
    network_bw_gbps: float
    available_zones: List[str] = None

    def __post_init__(self):
        if self.available_zones is None:
            self.available_zones = []
    
    def to_hardware_config(self) -> HardwareConfig:
        """Convert VMInstance to HardwareConfig for evaluation."""
        if not self.gpus:
            raise ValueError(f"VM {self.name} has no GPUs")
        return HardwareConfig(
            vcpus=self.vcpus,
            memory_gb=self.memory_gb,
            gpu_type=self.gpu_type,
            gpu_count=self.gpu_count,
            gpu_vram_gb=self.max_vram_per_gpu_gb,
            name=self.name
        )

    @property
    def total_vram_gb(self) -> int:
        return sum(g.vram_gb * g.count for g in self.gpus)

    @property
    def gpu_count(self) -> int:
        return sum(g.count for g in self.gpus)

    @property
    def gpu_type(self) -> str:
        return self.gpus[0].type if self.gpus else ""
    
    @property
    def max_vram_per_gpu_gb(self) -> int:
        """Get the maximum VRAM available on any single GPU in this instance.
        
        This is important because tensor parallelism requires the model to fit
        on each individual GPU, not just the total VRAM across all GPUs.
        """
        if not self.gpus:
            return 0
        return max(g.vram_gb for g in self.gpus)
    
    def is_available_in_zone(self, zone: str) -> bool:
        """Check if this instance is available in the given zone."""
        if not self.available_zones:
            return True  # If no zones specified, assume available everywhere
        return zone in self.available_zones
    
    def is_available_in_region(self, region: str) -> bool:
        """Check if this instance is available in the given region."""
        if not self.available_zones:
            return True
        return any(z.startswith(f"{region}-") for z in self.available_zones)


@dataclass
class TokenDistribution:
    """Token count distribution for workload."""
    mean: float
    std: float
    min: int
    max: int
    distribution: str = "lognormal"


@dataclass
class PrefixCacheConfig:
    """Configuration for prefix caching behavior.
    
    Prefix caching allows reusing KV cache for common prefixes across requests.
    This is especially important for RAG workloads where retrieved documents
    and system prompts are often shared.
    """
    enabled: bool = False
    # Percentage of input tokens that are cacheable (e.g., retrieved docs, system prompt)
    cacheable_token_ratio: float = 0.0
    # Expected cache hit rate (0.0 to 1.0)
    cache_hit_rate: float = 0.0
    # Average length of cacheable prefix in tokens
    avg_prefix_length: int = 0


@dataclass
class WorkloadCharacteristics:
    """Traffic pattern and token distribution for a workload type."""
    description: str
    arrival_process: str
    peak_factor: float
    burstiness: float
    input_tokens: TokenDistribution
    output_tokens: TokenDistribution
    context_tokens: TokenDistribution
    prefix_cache: PrefixCacheConfig = field(default_factory=lambda: PrefixCacheConfig())


@dataclass
class SLATargets:
    """SLA targets for a workload type."""
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float
    e2e_latency_per_512_tokens_ms: float
    concurrent_users_per_1k_tok_s: float


@dataclass
class ModelCharacteristics:
    """Characteristics of an LLM model."""
    model_id: str
    parameters_b: float
    hidden_dim: int
    num_layers: int
    num_heads: int
    vocab_size: int
    kv_cache_per_token_kb: float
    flops_per_token: float
    memory_per_token_activations_kb: float


@dataclass
class RooflineMetrics:
    """Performance metrics from roofline analysis."""
    compute_bound: bool
    operational_intensity: float
    achievable_tflops: float
    memory_bound_bw_gbps: float
    
    # Token generation metrics
    tokens_per_sec_single: float
    tokens_per_sec_batch: float
    
    # Latency components
    prefill_latency_ms: float
    decode_latency_per_token_ms: float
    
    # Utilization
    compute_utilization: float
    memory_bw_utilization: float


@dataclass
class QueuingMetrics:
    """Queuing model predictions."""
    arrival_rate: float  # requests per second
    service_rate: float  # requests per second
    utilization: float
    
    # Wait times (queuing delay)
    wait_time_p50_ms: float
    wait_time_p95_ms: float
    wait_time_p99_ms: float
    
    # Total latency = queuing + service
    total_latency_p50_ms: float
    total_latency_p95_ms: float
    total_latency_p99_ms: float


@dataclass
class PerformancePrediction:
    """Complete performance prediction for a configuration."""
    # TTFT (Time To First Token)
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    
    # TPOT (Time Per Output Token)
    tpot_p50_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float
    
    # E2E Latency
    e2e_latency_p50_ms: float
    e2e_latency_p95_ms: float
    e2e_latency_p99_ms: float
    
    # Throughput
    throughput_tok_s: float
    throughput_req_s: float
    
    # Utilization
    gpu_utilization: float
    memory_utilization: float
    
    # Burst capacity
    burst_throughput_tok_s: float
    burst_gpu_utilization: float


@dataclass
class SizingRecommendation:
    """A hardware configuration recommendation."""
    vm_instance: VMInstance
    tensor_parallel_size: int
    pipeline_parallel_size: int
    
    # Cost analysis
    hourly_cost_usd: float
    cost_per_1k_tokens_usd: float
    cost_per_request_usd: float
    
    # Performance prediction
    performance: PerformancePrediction
    
    # SLA compliance
    meets_sla: bool
    sla_gaps: Dict[str, float]  # metric -> gap percentage
    
    # Scaling info
    num_instances_needed: int
    total_hourly_cost_usd: float
    
    # Headroom info
    headroom_percent: int
    peak_capacity_factor: float


@dataclass
class SizingInput:
    """Input parameters for the sizing tool."""
    model_id: str
    workload_type: WorkloadType
    concurrent_users: int
    headroom_percent: int  # 0, 25, 50, 100
    vm_catalog: List[VMInstance]
    pricing_model: PricingModel = PricingModel.ONDEMAND
    region: str = "us-central1"
    quantization: str = "fp16"  # fp16, bf16, int8, fp8
    context_length: int = 4096
    # Override prefix cache config (uses workload defaults if None)
    prefix_cache: Optional[PrefixCacheConfig] = None
    
    @property
    def effective_concurrent_users(self) -> int:
        """Apply headroom to concurrent users for capacity planning."""
        headroom_factors = {0: 1.0, 25: 1.33, 50: 2.0, 100: 2.0}
        factor = headroom_factors.get(self.headroom_percent, 1.0)
        return int(self.concurrent_users * factor)
