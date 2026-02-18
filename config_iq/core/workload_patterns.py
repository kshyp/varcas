"""Workload pattern definitions and traffic modeling."""

import json
import os
import math
import random
from typing import Dict, Optional
from .types import (
    WorkloadCharacteristics, 
    TokenDistribution, 
    SLATargets,
    WorkloadType,
    PrefixCacheConfig
)


class WorkloadPatterns:
    """Manages workload patterns and SLA targets."""
    
    def __init__(self, defaults_path: Optional[str] = None):
        """Initialize workload patterns.
        
        Args:
            defaults_path: Path to defaults JSON file. If None, uses default.
        """
        if defaults_path is None:
            defaults_path = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'workload_defaults.json'
            )
        
        self.defaults_path = defaults_path
        self.workloads: Dict[WorkloadType, WorkloadCharacteristics] = {}
        self.sla_targets: Dict[WorkloadType, SLATargets] = {}
        self.headroom_factors: Dict[int, Dict] = {}
        self.vllm_overhead: Dict = {}
        self._load_defaults()
    
    def _load_defaults(self):
        """Load workload defaults from file."""
        with open(self.defaults_path, 'r') as f:
            data = json.load(f)
        
        for workload_name, workload_data in data.get('workloads', {}).items():
            workload_type = WorkloadType(workload_name)
            
            # Parse token distributions
            input_dist = TokenDistribution(**workload_data['token_distribution']['input_tokens'])
            output_dist = TokenDistribution(**workload_data['token_distribution']['output_tokens'])
            context_dist = TokenDistribution(**workload_data['token_distribution']['context_tokens'])
            
            # Parse prefix cache config if present
            cache_config = PrefixCacheConfig()
            if 'prefix_cache' in workload_data:
                cache_data = workload_data['prefix_cache']
                cache_config = PrefixCacheConfig(
                    enabled=cache_data.get('enabled', False),
                    cacheable_token_ratio=cache_data.get('cacheable_token_ratio', 0.0),
                    cache_hit_rate=cache_data.get('cache_hit_rate', 0.0),
                    avg_prefix_length=cache_data.get('avg_prefix_length', 0)
                )
            
            self.workloads[workload_type] = WorkloadCharacteristics(
                description=workload_data['description'],
                arrival_process=workload_data['traffic_pattern']['arrival_process'],
                peak_factor=workload_data['traffic_pattern']['peak_factor'],
                burstiness=workload_data['traffic_pattern']['burstiness'],
                input_tokens=input_dist,
                output_tokens=output_dist,
                context_tokens=context_dist,
                prefix_cache=cache_config
            )
            
            # Parse SLA targets
            sla_data = workload_data['sla_targets']
            self.sla_targets[workload_type] = SLATargets(
                ttft_p50_ms=sla_data['ttft_p50_ms'],
                ttft_p95_ms=sla_data['ttft_p95_ms'],
                ttft_p99_ms=sla_data['ttft_p99_ms'],
                tpot_p50_ms=sla_data['tpot_p50_ms'],
                tpot_p95_ms=sla_data['tpot_p95_ms'],
                tpot_p99_ms=sla_data['tpot_p99_ms'],
                e2e_latency_per_512_tokens_ms=sla_data['e2e_latency_per_512_tokens_ms'],
                concurrent_users_per_1k_tok_s=sla_data['concurrent_users_per_1k_tok_s']
            )
        
        self.headroom_factors = {
            int(k): v for k, v in data.get('headroom_factors', {}).items()
        }
        self.vllm_overhead = data.get('vllm_overhead_factors', {})
    
    def get_workload(self, workload_type: WorkloadType) -> WorkloadCharacteristics:
        """Get workload characteristics."""
        return self.workloads[workload_type]
    
    def get_sla_targets(self, workload_type: WorkloadType) -> SLATargets:
        """Get SLA targets for a workload type."""
        return self.sla_targets[workload_type]
    
    def get_headroom_config(self, headroom_percent: int) -> Dict:
        """Get headroom configuration."""
        return self.headroom_factors.get(headroom_percent, self.headroom_factors[25])
    
    def sample_tokens(
        self, 
        distribution: TokenDistribution,
        n_samples: int = 1,
        seed: int = 42
    ) -> list:
        """Sample token counts from distribution."""
        rng = random.Random(seed)
        samples = []
        
        for _ in range(n_samples):
            if distribution.distribution == "lognormal":
                # Convert mean/std to lognormal params
                if distribution.std < 0.1:
                    val = distribution.mean
                else:
                    sigma_sq = math.log(1 + (distribution.std / distribution.mean) ** 2)
                    mu = math.log(distribution.mean) - 0.5 * sigma_sq
                    sigma = math.sqrt(sigma_sq)
                    val = rng.lognormvariate(mu, sigma)
            elif distribution.distribution == "normal":
                val = rng.gauss(distribution.mean, distribution.std)
            else:  # uniform
                val = rng.uniform(distribution.min, distribution.max)
            
            # Clamp to min/max
            val = max(distribution.min, min(distribution.max, int(val)))
            samples.append(val)
        
        return samples if n_samples > 1 else samples[0]
    
    def get_expected_tokens(self, workload_type: WorkloadType) -> tuple:
        """Get expected input and output token counts.
        
        Returns:
            (input_tokens, output_tokens) tuple
        """
        workload = self.workloads[workload_type]
        return (
            workload.input_tokens.mean,
            workload.output_tokens.mean
        )
    
    def get_effective_prefill_tokens(self, workload_type: WorkloadType) -> float:
        """Calculate effective prefill tokens accounting for prefix cache.
        
        When prefix caching is enabled, cacheable tokens (like retrieved documents
        and system prompts) don't need to be re-computed if they're in cache.
        
        Returns:
            Effective number of tokens that need actual prefill computation
        """
        workload = self.workloads[workload_type]
        cache = workload.prefix_cache
        
        if not cache.enabled or cache.cache_hit_rate <= 0:
            # No caching - all tokens need prefill
            return workload.input_tokens.mean + workload.context_tokens.mean
        
        # Total tokens that would need prefill without caching
        total_tokens = workload.input_tokens.mean + workload.context_tokens.mean
        
        # Cacheable portion (e.g., retrieved docs, system prompt)
        cacheable_tokens = total_tokens * cache.cacheable_token_ratio
        
        # Non-cacheable portion (user query, dynamic instructions)
        non_cacheable_tokens = total_tokens - cacheable_tokens
        
        # With cache hits, we only compute non-cacheable + cache misses
        effective_tokens = non_cacheable_tokens + cacheable_tokens * (1 - cache.cache_hit_rate)
        
        return effective_tokens
    
    def get_cache_memory_overhead(self, workload_type: WorkloadType) -> float:
        """Calculate additional memory needed for prefix cache.
        
        Cached prefixes consume KV cache memory. This estimates the
        additional memory needed per sequence for cached content.
        
        Returns:
            Additional KV cache memory per sequence in GB
        """
        workload = self.workloads[workload_type]
        cache = workload.prefix_cache
        
        if not cache.enabled or cache.cache_hit_rate <= 0:
            return 0.0
        
        # Approximate KV cache size for cached prefix
        # KV cache per token is roughly 2 * num_layers * hidden_dim * bytes_per_param
        # This is a simplified estimate - actual implementation would use model profile
        # Returns as a fraction of per-token KV cache that needs to be stored
        return 0.0  # Placeholder - actual calculation in roofline_analyzer
    
    def calculate_request_rate(
        self,
        workload_type: WorkloadType,
        concurrent_users: int,
        include_burst: bool = False
    ) -> float:
        """Calculate expected request rate (requests per second).
        
        Args:
            workload_type: Type of workload
            concurrent_users: Number of concurrent users
            include_burst: Whether to include burst factor
        
        Returns:
            Expected requests per second
        """
        workload = self.workloads[workload_type]
        sla = self.sla_targets[workload_type]
        
        # Calculate based on think time model
        # Users send requests, wait for response, then think before next request
        expected_input, expected_output = self.get_expected_tokens(workload_type)
        
        # Time per request = TTFT + (output_tokens * TPOT)
        time_per_request_ms = (
            sla.ttft_p50_ms + 
            expected_output * sla.tpot_p50_ms
        )
        
        # Request rate = users / time_per_request
        request_rate = (concurrent_users * 1000) / time_per_request_ms
        
        if include_burst:
            request_rate *= workload.peak_factor
        
        return request_rate
    
    def calculate_token_rate(
        self,
        workload_type: WorkloadType,
        concurrent_users: int,
        include_burst: bool = False
    ) -> tuple:
        """Calculate expected token rates.
        
        Returns:
            (input_tokens_per_sec, output_tokens_per_sec) tuple
        """
        request_rate = self.calculate_request_rate(
            workload_type, concurrent_users, include_burst
        )
        input_tokens, output_tokens = self.get_expected_tokens(workload_type)
        
        return (
            request_rate * input_tokens,
            request_rate * output_tokens
        )
    
    def get_vllm_overhead(self, key: str, default=None):
        """Get vLLM overhead factor."""
        return self.vllm_overhead.get(key, default)
    
    def calculate_effective_capacity(
        self,
        workload_type: WorkloadType,
        headroom_percent: int
    ) -> Dict:
        """Calculate effective capacity parameters with headroom.
        
        Returns:
            Dict with utilization_target, burst_multiplier, etc.
        """
        headroom = self.get_headroom_config(headroom_percent)
        workload = self.workloads[workload_type]
        
        return {
            **headroom,
            'peak_factor': workload.peak_factor,
            'burstiness': workload.burstiness
        }
