"""SLA calculation and compliance checking."""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .types import (
    VMInstance, HardwareConfig, WorkloadType, SLATargets,
    PerformancePrediction, SizingRecommendation, SizingInput,
    PricingModel
)
from .model_profiler import ModelProfile
from .hardware_catalog import HardwareCatalog
from .workload_patterns import WorkloadPatterns
from .model_profiler import get_model_profile
from .roofline_analyzer import RooflineAnalyzer
from .queuing_model import MG1QueuingModel, calculate_tail_latency_factors


@dataclass
class SLAMetrics:
    """SLA metrics with target vs predicted comparison."""
    metric_name: str
    target_value: float
    predicted_value: float
    unit: str
    
    @property
    def gap_percent(self) -> float:
        """Percentage gap from target (negative means better than target)."""
        if self.target_value == 0:
            return 0
        return ((self.predicted_value - self.target_value) / self.target_value) * 100
    
    @property
    def meets_sla(self) -> bool:
        """Check if predicted value meets target (latency <= target)."""
        return self.predicted_value <= self.target_value
    
    @property
    def status(self) -> str:
        """Get status indicator."""
        if self.meets_sla:
            return "✓"
        elif self.gap_percent <= 20:
            return "~"  # Within 20%
        else:
            return "✗"


class SLACalculator:
    """Calculates SLA predictions and compliance."""
    
    def __init__(
        self,
        hardware_catalog: HardwareCatalog,
        workload_patterns: WorkloadPatterns,
        roofline_analyzer: RooflineAnalyzer
    ):
        self.hardware = hardware_catalog
        self.workloads = workload_patterns
        self.roofline = roofline_analyzer
        self.queuing = MG1QueuingModel(service_time_cv=0.5)
    
    def calculate_performance(
        self,
        vm: VMInstance,
        model: ModelProfile,
        workload_type: WorkloadType,
        concurrent_users: int,
        headroom_percent: int,
        quantization: str = "fp16",
        prefix_cache_override: Optional[any] = None
    ) -> PerformancePrediction:
        """Calculate full performance prediction for a configuration.
        
        This combines roofline analysis for P50 with queuing model for tail latencies.
        
        Args:
            vm: VM instance
            model: Model profile
            workload_type: Type of workload
            concurrent_users: Number of concurrent users
            headroom_percent: Headroom percentage
            quantization: Quantization type
            prefix_cache_override: Optional prefix cache config override
        """
        # Get workload characteristics
        workload = self.workloads.get_workload(workload_type)
        
        # Calculate effective prefill tokens with prefix cache
        effective_prefill = self.workloads.get_effective_prefill_tokens(workload_type)
        total_input = int(workload.input_tokens.mean + workload.context_tokens.mean)
        cached_tokens = total_input - int(effective_prefill)
        
        input_tokens = total_input  # Total for full context
        output_tokens = int(workload.output_tokens.mean)
        
        # Calculate request rate
        request_rate = self.workloads.calculate_request_rate(
            workload_type, concurrent_users, include_burst=False
        )
        
        # Determine tensor parallel size based on GPU count and VRAM
        gpu_count = vm.gpu_count
        gpu_vram = vm.max_vram_per_gpu_gb
        tp_size = model.calculate_tensor_parallel_degree(gpu_count, gpu_vram, quantization)
        
        # Find optimal batch size
        max_batch = self.roofline.find_max_batch_size(
            vm, model, 4096, quantization
        )
        
        # For single-user latency, use batch_size=1
        # For throughput, calculate effective batch size from concurrency
        effective_batch = min(max_batch, max(1, concurrent_users // 10))
        
        # Roofline analysis for P50 (single request)
        single_metrics = self.roofline.analyze_full_request(
            vm, model, batch_size=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            quantization=quantization,
            tensor_parallel_size=tp_size,
            cached_tokens=cached_tokens
        )
        
        # Roofline analysis for throughput (batched)
        batch_metrics = self.roofline.analyze_full_request(
            vm, model, batch_size=effective_batch,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            quantization=quantization,
            tensor_parallel_size=tp_size,
            cached_tokens=cached_tokens
        )
        
        # Base latencies from roofline (P50)
        ttft_p50 = single_metrics["prefill_latency_ms"]
        tpot_p50 = single_metrics["decode_latency_per_token_ms"]
        e2e_p50 = single_metrics["total_latency_ms"]
        
        # Calculate service rate for queuing model
        # Service rate = how many requests can we serve per second
        service_time_per_request_ms = (
            batch_metrics["prefill_latency_ms"] + 
            output_tokens * batch_metrics["decode_latency_per_token_ms"]
        )
        service_rate = 1000 / service_time_per_request_ms * effective_batch
        
        # Apply queuing model for tail latencies
        queuing_metrics = self.queuing.calculate_metrics(request_rate, service_rate)
        
        # Calculate tail latencies by adding queuing delay
        utilization = queuing_metrics.utilization
        
        # Tail factors from queuing model
        tail_factors = calculate_tail_latency_factors(ttft_p50, utilization)
        
        ttft_p95 = tail_factors["p95_ms"]
        ttft_p99 = tail_factors["p99_ms"]
        
        # TPOT tail latencies (less affected by queuing)
        tpot_tail_factors = calculate_tail_latency_factors(tpot_p50, utilization * 0.5)
        tpot_p95 = tpot_tail_factors["p95_ms"]
        tpot_p99 = tpot_tail_factors["p99_ms"]
        
        # E2E tail latencies
        e2e_p95 = ttft_p95 + output_tokens * tpot_p95
        e2e_p99 = ttft_p99 + output_tokens * tpot_p99
        
        # Throughput calculations
        throughput_tok_s = batch_metrics["tokens_per_second"]
        throughput_req_s = throughput_tok_s / output_tokens
        
        # Utilization
        gpu_util = batch_metrics["compute_utilization"]
        
        # Burst capacity
        headroom_config = self.workloads.get_headroom_config(headroom_percent)
        burst_multiplier = headroom_config["burst_multiplier"]
        burst_throughput = throughput_tok_s * burst_multiplier
        burst_utilization = min(0.99, gpu_util * burst_multiplier)
        
        return PerformancePrediction(
            ttft_p50_ms=ttft_p50,
            ttft_p95_ms=ttft_p95,
            ttft_p99_ms=ttft_p99,
            tpot_p50_ms=tpot_p50,
            tpot_p95_ms=tpot_p95,
            tpot_p99_ms=tpot_p99,
            e2e_latency_p50_ms=e2e_p50,
            e2e_latency_p95_ms=e2e_p95,
            e2e_latency_p99_ms=e2e_p99,
            throughput_tok_s=throughput_tok_s,
            throughput_req_s=throughput_req_s,
            gpu_utilization=gpu_util,
            memory_utilization=batch_metrics["memory_utilization"],
            burst_throughput_tok_s=burst_throughput,
            burst_gpu_utilization=burst_utilization
        )
    
    def check_sla_compliance(
        self,
        prediction: PerformancePrediction,
        workload_type: WorkloadType
    ) -> Tuple[bool, Dict[str, float]]:
        """Check if prediction meets SLA targets.
        
        Returns:
            (meets_sla, gaps_dict)
        """
        targets = self.workloads.get_sla_targets(workload_type)
        gaps = {}
        
        metrics_to_check = [
            ("ttft_p50", prediction.ttft_p50_ms, targets.ttft_p50_ms),
            ("ttft_p95", prediction.ttft_p95_ms, targets.ttft_p95_ms),
            ("ttft_p99", prediction.ttft_p99_ms, targets.ttft_p99_ms),
            ("tpot_p50", prediction.tpot_p50_ms, targets.tpot_p50_ms),
            ("tpot_p95", prediction.tpot_p95_ms, targets.tpot_p95_ms),
            ("tpot_p99", prediction.tpot_p99_ms, targets.tpot_p99_ms),
        ]
        
        all_meet = True
        for name, predicted, target in metrics_to_check:
            gap = ((predicted - target) / target * 100) if target > 0 else 0
            gaps[name] = gap
            if predicted > target:
                all_meet = False
        
        return all_meet, gaps
    
    def calculate_cost_metrics(
        self,
        vm: VMInstance,
        prediction: PerformancePrediction,
        pricing_model: PricingModel
    ) -> Dict[str, float]:
        """Calculate cost metrics for a configuration.
        
        Returns:
            Dict with hourly_cost, cost_per_1k_tokens, cost_per_request
        """
        # Hourly cost
        hourly_cost = (
            vm.ondemand_price_usd 
            if pricing_model == PricingModel.ONDEMAND 
            else vm.spot_price_usd
        )
        
        # Cost per 1K tokens
        tokens_per_hour = prediction.throughput_tok_s * 3600
        if tokens_per_hour > 0:
            cost_per_1k_tokens = (hourly_cost / tokens_per_hour) * 1000
        else:
            cost_per_1k_tokens = float('inf')
        
        # Cost per request
        requests_per_hour = prediction.throughput_req_s * 3600
        if requests_per_hour > 0:
            cost_per_request = hourly_cost / requests_per_hour
        else:
            cost_per_request = float('inf')
        
        return {
            "hourly_cost_usd": hourly_cost,
            "cost_per_1k_tokens_usd": cost_per_1k_tokens,
            "cost_per_request_usd": cost_per_request,
            "tokens_per_hour": tokens_per_hour,
            "requests_per_hour": requests_per_hour
        }
    
    def evaluate_hardware_config(
        self,
        sizing_input: SizingInput,
        hw_config: HardwareConfig
    ) -> SizingRecommendation:
        """Evaluate a single hardware configuration.
        
        This is the core evaluation function that works with any hardware config,
        whether from a catalog VM or a custom DIY configuration.
        
        Args:
            sizing_input: Input parameters (model, workload, users, etc.)
            hw_config: Hardware configuration to evaluate
            
        Returns:
            SizingRecommendation with performance predictions and SLA analysis
            
        Raises:
            ValueError: If the model doesn't fit on the specified hardware
        """
        # Get model profile
        model = get_model_profile(sizing_input.model_id)
        
        # Check if model fits with tensor parallelism
        tp_size = model.calculate_tensor_parallel_degree(
            hw_config.gpu_count, 
            hw_config.gpu_vram_gb, 
            sizing_input.quantization
        )
        
        mem_required = model.calculate_memory_required_per_gpu(tp_size, sizing_input.quantization)
        if hw_config.gpu_vram_gb < mem_required:
            raise ValueError(
                f"Model doesn't fit: requires {mem_required:.1f} GB per GPU, "
                f"but {hw_config.gpu_type} has only {hw_config.gpu_vram_gb} GB. "
                f"Try using {tp_size * 2}x GPUs with tensor parallelism, "
                f"or use a GPU with more VRAM (e.g., A100 40GB)."
            )
        
        # Get GPU specs from catalog (for compute characteristics)
        gpu_specs = self._get_gpu_specs(hw_config.gpu_type)
        if gpu_specs is None:
            raise ValueError(f"Unknown GPU type: {hw_config.gpu_type}")
        
        # Calculate performance
        performance = self._calculate_performance_for_config(
            hw_config=hw_config,
            gpu_specs=gpu_specs,
            model=model,
            workload_type=sizing_input.workload_type,
            concurrent_users=sizing_input.concurrent_users,
            headroom_percent=sizing_input.headroom_percent,
            quantization=sizing_input.quantization
        )
        
        # Check SLA compliance
        meets_sla, sla_gaps = self.check_sla_compliance(
            performance, sizing_input.workload_type
        )
        
        # Calculate costs (if available, otherwise estimate)
        costs = self._calculate_cost_metrics_for_config(
            hw_config, performance, sizing_input.pricing_model
        )
        
        # Calculate instances needed
        headroom_config = self.workloads.get_headroom_config(
            sizing_input.headroom_percent
        )
        target_util = headroom_config["utilization_target"]
        
        if performance.gpu_utilization > 0:
            instances_needed = max(1, int(
                performance.gpu_utilization / target_util
            ))
        else:
            instances_needed = 1
        
        total_hourly_cost = costs["hourly_cost_usd"] * instances_needed
        
        # Adjust costs for multiple instances
        if instances_needed > 1:
            costs["cost_per_request_usd"] *= instances_needed
            costs["cost_per_1k_tokens_usd"] *= instances_needed
        
        return SizingRecommendation(
            vm_instance=hw_config,  # Now HardwareConfig is compatible
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
            hourly_cost_usd=costs["hourly_cost_usd"],
            cost_per_1k_tokens_usd=costs["cost_per_1k_tokens_usd"],
            cost_per_request_usd=costs["cost_per_request_usd"],
            performance=performance,
            meets_sla=meets_sla,
            sla_gaps=sla_gaps,
            num_instances_needed=instances_needed,
            total_hourly_cost_usd=total_hourly_cost,
            headroom_percent=sizing_input.headroom_percent,
            peak_capacity_factor=headroom_config["burst_multiplier"]
        )
    
    def _get_gpu_specs(self, gpu_type: str):
        """Get GPU specifications from the hardware catalog."""
        # This would ideally be in HardwareCatalog, but we'll look it up here
        for vm in self.hardware.get_all_instances():
            for gpu in vm.gpus:
                if gpu.type == gpu_type:
                    return gpu
        return None
    
    def _calculate_performance_for_config(
        self,
        hw_config: HardwareConfig,
        gpu_specs,
        model: ModelProfile,
        workload_type: WorkloadType,
        concurrent_users: int,
        headroom_percent: int,
        quantization: str = "fp16"
    ) -> PerformancePrediction:
        """Calculate performance for a hardware configuration."""
        # Get workload characteristics
        workload = self.workloads.get_workload(workload_type)
        
        # Calculate effective prefill tokens with prefix cache
        effective_prefill = self.workloads.get_effective_prefill_tokens(workload_type)
        total_input = int(workload.input_tokens.mean + workload.context_tokens.mean)
        cached_tokens = total_input - int(effective_prefill)
        
        input_tokens = total_input  # Total for full context
        output_tokens = int(workload.output_tokens.mean)
        
        # Calculate request rate
        request_rate = self.workloads.calculate_request_rate(
            workload_type, concurrent_users, include_burst=False
        )
        
        # Determine tensor parallel size
        gpu_count = hw_config.gpu_count
        tp_size = model.calculate_tensor_parallel_degree(
            gpu_count, hw_config.gpu_vram_gb, quantization
        )
        
        # Create a mock VMInstance for roofline analysis (we'll refactor this later)
        from .types import VMInstance, GPUSpec
        mock_gpus = [gpu_specs] * gpu_count
        mock_vm = VMInstance(
            name=hw_config.name,
            family="custom",
            vcpus=hw_config.vcpus,
            memory_gb=hw_config.memory_gb,
            gpus=mock_gpus,
            ondemand_price_usd=0.0,
            spot_price_usd=0.0,
            network_bw_gbps=10.0,
            available_zones=[]
        )
        
        # Find optimal batch size
        max_batch = self.roofline.find_max_batch_size(
            mock_vm, model, 4096, quantization
        )
        
        # For single-user latency, use batch_size=1
        effective_batch = min(max_batch, max(1, concurrent_users // 10))
        
        # Roofline analysis for P50 (single request)
        single_metrics = self.roofline.analyze_full_request(
            mock_vm, model, batch_size=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            quantization=quantization,
            tensor_parallel_size=tp_size,
            cached_tokens=cached_tokens
        )
        
        # Roofline analysis for throughput (batched)
        batch_metrics = self.roofline.analyze_full_request(
            mock_vm, model, batch_size=effective_batch,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            quantization=quantization,
            tensor_parallel_size=tp_size,
            cached_tokens=cached_tokens
        )
        
        # Base latencies from roofline (P50)
        ttft_p50 = single_metrics["prefill_latency_ms"]
        tpot_p50 = single_metrics["decode_latency_per_token_ms"]
        e2e_p50 = single_metrics["total_latency_ms"]
        
        # Calculate service rate for queuing model
        service_time_per_request_ms = (
            batch_metrics["prefill_latency_ms"] + 
            output_tokens * batch_metrics["decode_latency_per_token_ms"]
        )
        service_rate = 1000 / service_time_per_request_ms * effective_batch
        
        # Apply queuing model for tail latencies
        queuing_metrics = self.queuing.calculate_metrics(request_rate, service_rate)
        utilization = queuing_metrics.utilization
        
        # Tail factors from queuing model
        tail_factors = calculate_tail_latency_factors(ttft_p50, utilization)
        ttft_p95 = tail_factors["p95_ms"]
        ttft_p99 = tail_factors["p99_ms"]
        
        # TPOT tail latencies
        tpot_tail_factors = calculate_tail_latency_factors(tpot_p50, utilization * 0.5)
        tpot_p95 = tpot_tail_factors["p95_ms"]
        tpot_p99 = tpot_tail_factors["p99_ms"]
        
        # E2E tail latencies
        e2e_p95 = ttft_p95 + output_tokens * tpot_p95
        e2e_p99 = ttft_p99 + output_tokens * tpot_p99
        
        # Throughput calculations
        throughput_tok_s = batch_metrics["tokens_per_second"]
        throughput_req_s = throughput_tok_s / output_tokens
        
        # Utilization
        gpu_util = batch_metrics["compute_utilization"]
        
        # Burst capacity
        headroom_config = self.workloads.get_headroom_config(headroom_percent)
        burst_multiplier = headroom_config["burst_multiplier"]
        burst_throughput = throughput_tok_s * burst_multiplier
        burst_utilization = min(0.99, gpu_util * burst_multiplier)
        
        return PerformancePrediction(
            ttft_p50_ms=ttft_p50,
            ttft_p95_ms=ttft_p95,
            ttft_p99_ms=ttft_p99,
            tpot_p50_ms=tpot_p50,
            tpot_p95_ms=tpot_p95,
            tpot_p99_ms=tpot_p99,
            e2e_latency_p50_ms=e2e_p50,
            e2e_latency_p95_ms=e2e_p95,
            e2e_latency_p99_ms=e2e_p99,
            throughput_tok_s=throughput_tok_s,
            throughput_req_s=throughput_req_s,
            gpu_utilization=gpu_util,
            memory_utilization=batch_metrics["memory_utilization"],
            burst_throughput_tok_s=burst_throughput,
            burst_gpu_utilization=burst_utilization
        )
    
    def _calculate_cost_metrics_for_config(
        self,
        hw_config: HardwareConfig,
        prediction: PerformancePrediction,
        pricing_model: PricingModel
    ) -> Dict[str, float]:
        """Calculate cost metrics for a hardware configuration.
        
        For custom configs, we estimate cost based on GPU pricing.
        """
        # Try to find a similar VM in catalog for pricing reference
        hourly_cost = 0.0
        
        # Look for a VM with the same GPU type to get pricing
        for vm in self.hardware.get_all_instances():
            if vm.gpu_type == hw_config.gpu_type and vm.gpu_count == hw_config.gpu_count:
                hourly_cost = (
                    vm.ondemand_price_usd 
                    if pricing_model == PricingModel.ONDEMAND 
                    else vm.spot_price_usd
                )
                break
        
        # If no match found, estimate based on GPU count (rough heuristic)
        if hourly_cost == 0.0:
            # Find any VM with this GPU type to get per-GPU price
            for vm in self.hardware.get_all_instances():
                if vm.gpu_type == hw_config.gpu_type and vm.gpu_count > 0:
                    gpu_price = vm.ondemand_price_usd / vm.gpu_count
                    hourly_cost = gpu_price * hw_config.gpu_count
                    break
        
        # Cost per 1K tokens
        tokens_per_hour = prediction.throughput_tok_s * 3600
        if tokens_per_hour > 0:
            cost_per_1k_tokens = (hourly_cost / tokens_per_hour) * 1000
        else:
            cost_per_1k_tokens = float('inf')
        
        # Cost per request
        requests_per_hour = prediction.throughput_req_s * 3600
        if requests_per_hour > 0:
            cost_per_request = hourly_cost / requests_per_hour
        else:
            cost_per_request = float('inf')
        
        return {
            "hourly_cost_usd": hourly_cost,
            "cost_per_1k_tokens_usd": cost_per_1k_tokens,
            "cost_per_request_usd": cost_per_request,
            "tokens_per_hour": tokens_per_hour,
            "requests_per_hour": requests_per_hour
        }

    def generate_recommendation(
        self,
        sizing_input: SizingInput,
        vm: VMInstance
    ) -> SizingRecommendation:
        """Generate a complete sizing recommendation (legacy method)."""
        # Convert VM to HardwareConfig and use the new method
        hw_config = vm.to_hardware_config()
        rec = self.evaluate_hardware_config(sizing_input, hw_config)
        # Preserve the original VMInstance reference
        rec.vm_instance = vm
        return rec
        
        # Calculate performance
        performance = self.calculate_performance(
            vm=vm,
            model=model,
            workload_type=sizing_input.workload_type,
            concurrent_users=sizing_input.concurrent_users,
            headroom_percent=sizing_input.headroom_percent,
            quantization=sizing_input.quantization
        )
        
        # Check SLA compliance
        meets_sla, sla_gaps = self.check_sla_compliance(
            performance, sizing_input.workload_type
        )
        
        # Calculate costs
        costs = self.calculate_cost_metrics(
            vm, performance, sizing_input.pricing_model
        )
        
        # Calculate instances needed
        headroom_config = self.workloads.get_headroom_config(
            sizing_input.headroom_percent
        )
        target_util = headroom_config["utilization_target"]
        
        if performance.gpu_utilization > 0:
            instances_needed = max(1, int(
                performance.gpu_utilization / target_util
            ))
        else:
            instances_needed = 1
        
        total_hourly_cost = costs["hourly_cost_usd"] * instances_needed
        
        # Adjust costs for multiple instances
        if instances_needed > 1:
            costs["cost_per_request_usd"] *= instances_needed
            costs["cost_per_1k_tokens_usd"] *= instances_needed
        
        return SizingRecommendation(
            vm_instance=vm,
            tensor_parallel_size=model.calculate_tensor_parallel_degree(vm.gpu_count, vm.max_vram_per_gpu_gb, sizing_input.quantization),
            pipeline_parallel_size=1,  # Simplified for now
            hourly_cost_usd=costs["hourly_cost_usd"],
            cost_per_1k_tokens_usd=costs["cost_per_1k_tokens_usd"],
            cost_per_request_usd=costs["cost_per_request_usd"],
            performance=performance,
            meets_sla=meets_sla,
            sla_gaps=sla_gaps,
            num_instances_needed=instances_needed,
            total_hourly_cost_usd=total_hourly_cost,
            headroom_percent=sizing_input.headroom_percent,
            peak_capacity_factor=headroom_config["burst_multiplier"]
        )
    
    def generate_all_recommendations(
        self,
        sizing_input: SizingInput
    ) -> List[SizingRecommendation]:
        """Generate recommendations for all compatible hardware configurations."""
        model = get_model_profile(sizing_input.model_id)
        
        # Use pre-filtered catalog from sizing_input if available, otherwise filter from hardware
        if sizing_input.vm_catalog:
            # Filter the provided catalog for VRAM compatibility
            # Need to check if model fits with tensor parallelism
            compatible = []
            for vm in sizing_input.vm_catalog:
                gpu_count = vm.gpu_count
                gpu_vram = vm.max_vram_per_gpu_gb
                
                # Try different TP sizes to find one that fits
                # TP size must be <= gpu_count and a power of 2 (typically)
                valid_tp_sizes = [tp for tp in [1, 2, 4, 8] if tp <= gpu_count]
                
                for tp_size in valid_tp_sizes:
                    # Calculate memory per GPU with this TP size
                    # Model is sharded across TP group, so each GPU holds model_size/tp_size
                    model_size_gb = model.get_model_size_for_quantization(sizing_input.quantization)
                    model_per_gpu = model_size_gb / tp_size
                    
                    # Overhead: CUDA context + activations + safety margin
                    cuda_overhead_gb = 1.5
                    activation_buffer_gb = model_per_gpu * 0.10
                    safety_margin_gb = 0.5
                    total_per_gpu = model_per_gpu + cuda_overhead_gb + activation_buffer_gb + safety_margin_gb
                    
                    if gpu_vram >= total_per_gpu:
                        compatible.append(vm)
                        break  # Found a valid TP size for this VM
        else:
            # Filter compatible instances from hardware catalog
            compatible = self.hardware.filter_compatible_instances(
                model_vram_requirement_gb=model.model_size_gb,
                min_gpu_count=1,
                max_gpu_count=8,
                quantization=sizing_input.quantization
            )
        
        recommendations = []
        for vm in compatible:
            try:
                rec = self.generate_recommendation(sizing_input, vm)
                recommendations.append(rec)
            except Exception as e:
                # Skip configurations that can't be analyzed
                continue
        
        # Sort by cost per request (most economical first)
        recommendations.sort(key=lambda r: r.cost_per_request_usd)
        
        return recommendations
    
    def get_sla_comparison_table(
        self,
        recommendation: SizingRecommendation,
        workload_type: WorkloadType
    ) -> List[SLAMetrics]:
        """Generate SLA comparison table rows."""
        targets = self.workloads.get_sla_targets(workload_type)
        perf = recommendation.performance
        
        metrics = [
            ("TTFT P50", targets.ttft_p50_ms, perf.ttft_p50_ms, "ms"),
            ("TTFT P95", targets.ttft_p95_ms, perf.ttft_p95_ms, "ms"),
            ("TTFT P99", targets.ttft_p99_ms, perf.ttft_p99_ms, "ms"),
            ("TPOT P50", targets.tpot_p50_ms, perf.tpot_p50_ms, "ms"),
            ("TPOT P95", targets.tpot_p95_ms, perf.tpot_p95_ms, "ms"),
            ("E2E Latency", targets.e2e_latency_per_512_tokens_ms, 
             perf.e2e_latency_p50_ms, "ms"),
        ]
        
        return [
            SLAMetrics(name, target, predicted, unit)
            for name, target, predicted, unit in metrics
        ]
