"""M/G/1 queuing model for latency prediction."""

import math
from typing import Dict, Tuple
from .types import QueuingMetrics


class MG1QueuingModel:
    """M/G/1 queuing model for inference latency prediction.
    
    The M/G/1 model assumes:
    - Poisson arrivals (M)
    - General service time distribution (G)
    - Single server (1)
    
    For LLM inference, the "server" can be thought of as a single GPU
    or a tensor-parallel group acting as one unit.
    """
    
    def __init__(self, service_time_cv: float = 0.5):
        """Initialize M/G/1 model.
        
        Args:
            service_time_cv: Coefficient of variation for service times
                            (std/mean). Default 0.5 based on typical
                            inference variability.
        """
        self.service_time_cv = service_time_cv
    
    def calculate_metrics(
        self,
        arrival_rate: float,  # requests per second
        service_rate: float,  # requests per second
    ) -> QueuingMetrics:
        """Calculate queuing metrics.
        
        Args:
            arrival_rate: Lambda (λ) - average arrival rate
            service_rate: Mu (μ) - average service rate
        
        Returns:
            QueuingMetrics with wait times and latencies
        """
        if service_rate <= 0:
            raise ValueError("Service rate must be positive")
        
        if arrival_rate >= service_rate:
            # System is unstable (utilization >= 1)
            return QueuingMetrics(
                arrival_rate=arrival_rate,
                service_rate=service_rate,
                utilization=1.0,
                wait_time_p50_ms=float('inf'),
                wait_time_p95_ms=float('inf'),
                wait_time_p99_ms=float('inf'),
                total_latency_p50_ms=float('inf'),
                total_latency_p95_ms=float('inf'),
                total_latency_p99_ms=float('inf')
            )
        
        # Utilization (rho)
        rho = arrival_rate / service_rate
        
        # Mean service time
        mean_service_time_ms = 1000 / service_rate
        
        # Pollaczek-Khinchine formula for M/G/1 queue
        # E[W] = (ρ * E[S] * (1 + C_s^2)) / (2 * (1 - ρ))
        # where E[S] is mean service time, C_s is CV of service time
        
        cv_squared = self.service_time_cv ** 2
        mean_wait_time_ms = (
            rho * mean_service_time_ms * (1 + cv_squared) / (2 * (1 - rho))
        )
        
        # For M/G/1, waiting time distribution can be approximated
        # Using exponential approximation for percentiles
        
        # P50: median of waiting time
        wait_p50 = mean_wait_time_ms * math.log(2)
        
        # P95 and P99 using approximation
        # For exponential: P95 = -ln(0.05) * mean ≈ 3 * mean
        # For heavy-tailed (G/G/1-like), use higher multipliers
        tail_factor = 1 + cv_squared  # Adjust for variability
        
        wait_p95 = mean_wait_time_ms * (3.0 * tail_factor)
        wait_p99 = mean_wait_time_ms * (4.6 * tail_factor)
        
        # Cap extreme values to avoid overflow
        max_latency_ms = 60000  # 60 seconds
        wait_p50 = min(wait_p50, max_latency_ms)
        wait_p95 = min(wait_p95, max_latency_ms)
        wait_p99 = min(wait_p99, max_latency_ms)
        
        # Total latency = wait time + service time
        # Service time is roughly constant (p50 from roofline)
        total_p50 = min(wait_p50 + mean_service_time_ms, max_latency_ms)
        total_p95 = min(wait_p95 + mean_service_time_ms, max_latency_ms)
        total_p99 = min(wait_p99 + mean_service_time_ms, max_latency_ms)
        
        return QueuingMetrics(
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            utilization=rho,
            wait_time_p50_ms=wait_p50,
            wait_time_p95_ms=wait_p95,
            wait_time_p99_ms=wait_p99,
            total_latency_p50_ms=total_p50,
            total_latency_p95_ms=total_p95,
            total_latency_p99_ms=total_p99
        )
    
    def calculate_batch_queuing(
        self,
        arrival_rate: float,
        batch_size: int,
        batch_latency_ms: float
    ) -> QueuingMetrics:
        """Calculate metrics for batched inference.
        
        In batched inference, requests wait to form a batch.
        
        Args:
            arrival_rate: Requests per second
            batch_size: Target batch size
            batch_latency_ms: Time to process one batch
        
        Returns:
            QueuingMetrics
        """
        # Effective service rate for batches
        batch_service_rate = 1000 / batch_latency_ms
        
        # Request service rate (batches * batch_size)
        request_service_rate = batch_service_rate * batch_size
        
        return self.calculate_metrics(arrival_rate, request_service_rate)
    
    def find_capacity_for_sla(
        self,
        arrival_rate: float,
        target_latency_p95_ms: float,
        base_service_time_ms: float
    ) -> Tuple[float, float]:
        """Find required service rate to meet latency SLA.
        
        Args:
            arrival_rate: Expected request rate
            target_latency_p95_ms: Target P95 latency
            base_service_time_ms: Base processing time without queuing
        
        Returns:
            (required_service_rate, resulting_utilization)
        """
        # Target wait time = target_latency - service_time
        target_wait_ms = max(0, target_latency_p95_ms - base_service_time_ms)
        
        if target_wait_ms <= 0:
            # Service time alone exceeds target
            return float('inf'), 0
        
        # Solve Pollaczek-Khinchine for service rate
        # target_wait = (ρ * S * (1 + C_s^2)) / (2 * (1 - ρ))
        # where ρ = λ/μ, S = 1/μ
        
        cv_squared = self.service_time_cv ** 2
        
        # Rearranging:
        # target_wait = (λ/μ * 1/μ * (1+C_s^2)) / (2 * (1 - λ/μ))
        # target_wait * 2 * (1 - λ/μ) = λ/μ^2 * (1+C_s^2)
        # Let x = 1/μ (service time)
        # target_wait * 2 * (1 - λ*x) = λ*x^2 * (1+C_s^2)
        # 2*target_wait - 2*target_wait*λ*x = λ*(1+C_s^2)*x^2
        # λ*(1+C_s^2)*x^2 + 2*target_wait*λ*x - 2*target_wait = 0
        
        a = arrival_rate * (1 + cv_squared)
        b = 2 * target_wait_ms * arrival_rate / 1000  # convert to seconds
        c = -2 * target_wait_ms / 1000
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return float('inf'), 0
        
        x = (-b + math.sqrt(discriminant)) / (2*a)
        required_service_rate = 1 / x if x > 0 else float('inf')
        
        utilization = arrival_rate / required_service_rate if required_service_rate > 0 else 0
        
        return required_service_rate, utilization
    
    def calculate_percentile_multiplier(
        self,
        percentile: float,
        utilization: float
    ) -> float:
        """Calculate latency multiplier for a given percentile.
        
        For example, if p50 latency is X, what is p99 latency as a multiple of X?
        
        Args:
            percentile: Target percentile (e.g., 99 for P99)
            utilization: System utilization (0-1)
        
        Returns:
            Multiplier relative to mean latency
        """
        # Cap utilization to avoid infinity
        capped_util = min(utilization, 0.99)
        
        if capped_util >= 1.0:
            return float('inf')
        
        # For M/G/1, the multiplier depends on utilization and service CV
        # Higher utilization and higher CV = higher multipliers
        
        cv_squared = self.service_time_cv ** 2
        
        # Base multiplier for exponential service (M/M/1)
        if percentile == 50:
            base_mult = math.log(2)
        elif percentile == 95:
            base_mult = 3.0
        elif percentile == 99:
            base_mult = 4.6
        else:
            # General formula for exponential
            base_mult = -math.log(1 - percentile/100)
        
        # Adjust for utilization and service time variability
        # As utilization approaches 1, tail gets heavier
        util_factor = 1 / (1 - capped_util)
        
        # Adjust for service time CV
        cv_factor = 1 + (cv_squared - 1) * 0.5
        
        return base_mult * util_factor * cv_factor


def calculate_tail_latency_factors(
    base_latency_ms: float,
    utilization: float,
    service_cv: float = 0.5
) -> Dict[str, float]:
    """Calculate tail latency factors for display.
    
    Returns multipliers showing how much higher P95/P99 are than P50.
    
    Example: {"p95": 2.1, "p99": 2.9} means P95 is 2.1x P50, P99 is 2.9x P50
    """
    model = MG1QueuingModel(service_cv)
    
    p95_mult = model.calculate_percentile_multiplier(95, utilization)
    p99_mult = model.calculate_percentile_multiplier(99, utilization)
    
    # Adjust to be relative to P50
    p50_mult = model.calculate_percentile_multiplier(50, utilization)
    
    return {
        "p95_multiplier": p95_mult / p50_mult if p50_mult > 0 else float('inf'),
        "p99_multiplier": p99_mult / p50_mult if p50_mult > 0 else float('inf'),
        "p95_ms": base_latency_ms * (p95_mult / p50_mult),
        "p99_ms": base_latency_ms * (p99_mult / p50_mult)
    }
