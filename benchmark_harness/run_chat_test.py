#!/usr/bin/env python3
"""
Chat Test Script for SLA Verification
Matches the "concurrent users" parameter used in config_iq evaluation

This script runs a closed-loop chat workload test that directly corresponds
to the config_iq tool's "concurrent users" parameter for validation.

SLA Targets (Chat Workload):
    TTFT P50: ≤200ms, TTFT P95: ≤500ms, TTFT P99: ≤1000ms
    TPOT P50: ≤50ms,  TPOT P95: ≤80ms,  TPOT P99: ≤120ms

Token Distribution (matches config_iq):
    Input:  mean=50, std=30, min=10, max=500
    Output: mean=150, std=80, min=20, max=1000
"""

import argparse
import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Dict, Optional

# Import the harness
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from varcas_load_harness import (
    LoadHarness, LoadProfile, WorkloadProfile, WorkloadType,
    TokenDistribution, ArrivalProcess, get_closed_loop_profile
)


# SLA targets from config_iq/data/workload_defaults.json
CHAT_SLA_TARGETS = {
    "ttft_p50_ms": 200,
    "ttft_p95_ms": 500,
    "ttft_p99_ms": 1000,
    "tpot_p50_ms": 50,
    "tpot_p95_ms": 80,
    "tpot_p99_ms": 120,
}


def check_sla(metric_name: str, value: float, target: float) -> str:
    """Check if a metric meets its SLA target."""
    status = "✓" if value <= target else "✗"
    if value <= target:
        return f"{status} {metric_name:12s}: {value:6.1f}ms (target: ≤{target}ms)"
    else:
        excess = value - target
        return f"{status} {metric_name:12s}: {value:6.1f}ms (target: ≤{target}ms) - EXCEEDED by {excess:.1f}ms"


def print_results(result, concurrent_users: int):
    """Print test results with SLA compliance check."""
    print("\n" + "=" * 50)
    print("Test Complete")
    print("=" * 50)
    
    print(f"\nSummary:")
    print("-" * 50)
    print(f"Total Requests:     {result.total_requests}")
    print(f"Success Rate:       {result.successful_requests / result.total_requests * 100:.1f}%")
    print(f"Throughput:         {result.throughput_rps:.2f} req/s, {result.throughput_tok_s:.1f} tok/s")
    print(f"TTFT:               p50={result.ttft_p50:.1f}ms, p99={result.ttft_p99:.1f}ms")
    print(f"TPOT:               p50={result.tpot_p50:.2f}ms, p99={result.tpot_p99:.2f}ms")
    print(f"Latency:            p50={result.latency_p50:.1f}ms, p99={result.latency_p99:.1f}ms")
    
    print(f"\nSLA Targets for Chat Workload (from config_iq):")
    print("-" * 50)
    print("TTFT P50: ≤200ms, TTFT P95: ≤500ms, TTFT P99: ≤1000ms")
    print("TPOT P50: ≤50ms,  TPOT P95: ≤80ms,  TPOT P99: ≤120ms")
    
    print(f"\nSLA Compliance Check ({concurrent_users} concurrent users):")
    print("-" * 50)
    print(check_sla("TTFT P50", result.ttft_p50, CHAT_SLA_TARGETS["ttft_p50_ms"]))
    print(check_sla("TTFT P99", result.ttft_p99, CHAT_SLA_TARGETS["ttft_p99_ms"]))
    print(check_sla("TPOT P50", result.tpot_p50, CHAT_SLA_TARGETS["tpot_p50_ms"]))
    print(check_sla("TPOT P99", result.tpot_p99, CHAT_SLA_TARGETS["tpot_p99_ms"]))
    
    # Overall compliance
    meets_sla = (
        result.ttft_p50 <= CHAT_SLA_TARGETS["ttft_p50_ms"] and
        result.ttft_p99 <= CHAT_SLA_TARGETS["ttft_p99_ms"] and
        result.tpot_p50 <= CHAT_SLA_TARGETS["tpot_p50_ms"] and
        result.tpot_p99 <= CHAT_SLA_TARGETS["tpot_p99_ms"]
    )
    
    print("-" * 50)
    if meets_sla:
        print("✓ OVERALL: Configuration MEETS chat SLA targets")
    else:
        print("✗ OVERALL: Configuration DOES NOT MEET chat SLA targets")
    print("-" * 50)


async def run_test(
    vllm_url: str,
    concurrent_users: int,
    duration: int,
    output_file: str,
    temperature: float = 0.0,
    meaningful_prompts: bool = False,
    max_context: Optional[int] = None
) -> None:
    """Run the chat test."""
    
    # Create the load profile matching config_iq chat workload
    profile = get_closed_loop_profile(concurrency=concurrent_users, duration=duration)
    
    print("=" * 50)
    print("Chat Test - SLA Verification")
    print("=" * 50)
    print(f"vLLM URL:          {vllm_url}")
    print(f"Concurrent Users:  {concurrent_users}")
    print(f"Duration:          {duration}s")
    print(f"Output File:       {output_file}")
    print(f"Temperature:       {temperature}")
    print(f"Prompts:           {'Meaningful' if meaningful_prompts else 'Synthetic'}")
    print("=" * 50)
    print()
    print("SLA Targets (Chat Workload):")
    print("  TTFT P50: ≤200ms, TTFT P95: ≤500ms, TTFT P99: ≤1000ms")
    print("  TPOT P50: ≤50ms,  TPOT P95: ≤80ms,  TPOT P99: ≤120ms")
    print()
    print("Token Distribution:")
    print("  Input:  mean=50, std=30, min=10, max=500")
    print("  Output: mean=150, std=80, min=20, max=1000")
    print()
    
    # Check if vLLM is accessible
    import aiohttp
    print("Checking vLLM server...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{vllm_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    print("✓ vLLM server is accessible")
                else:
                    print(f"⚠ vLLM server returned status {resp.status}")
    except Exception as e:
        print(f"⚠ Could not connect to vLLM: {e}")
        print("  Make sure vLLM is running before starting the test.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    print()
    print(f"Starting load test with {concurrent_users} concurrent users...")
    print()
    
    # Create harness
    harness = LoadHarness(
        vllm_url=vllm_url,
        use_meaningful_prompts=meaningful_prompts,
        temperature=temperature,
        seed=42
    )
    
    # Auto-detect model context length if not specified
    if max_context:
        profile.model_max_context = max_context
    else:
        async with harness:
            detected = await harness.detect_model_context_length()
            if detected:
                profile.model_max_context = detected
                print(f"Auto-detected model max context: {detected} tokens")
            else:
                print(f"Could not detect model context, using default: {profile.model_max_context} tokens")
    
    print(f"Starting load test: {profile.name}")
    print(f"Model max context: {profile.model_max_context} tokens")
    print(f"Duration: {profile.duration_seconds}s")
    print(f"Mode: Closed loop (concurrency: {profile.concurrency})")
    print(f"Temperature: {temperature} ({'deterministic' if temperature == 0 else 'stochastic'})")
    print(f"Target: {vllm_url}")
    print()
    
    # Run the test
    async with harness:
        result = await harness.run(profile)
    
    # Save results
    result.save(output_file)
    
    # Print results
    print_results(result, concurrent_users)
    
    print(f"\nFull results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Chat Test Script for SLA Verification - Matches config_iq 'concurrent users' parameter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with 100 concurrent users (default)
  python run_chat_test.py

  # Run with 50 concurrent users for 60 seconds
  python run_chat_test.py --users 50 --duration 60

  # Run against remote vLLM server
  python run_chat_test.py --url http://vllm-server:8000 --users 200 --duration 180

SLA Targets (Chat Workload from config_iq):
  TTFT P50: ≤200ms, TTFT P95: ≤500ms, TTFT P99: ≤1000ms
  TPOT P50: ≤50ms,  TPOT P95: ≤80ms,  TPOT P99: ≤120ms
        """
    )
    
    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='vLLM endpoint URL (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--users', '-u',
        type=int,
        default=100,
        help='Number of concurrent users (default: 100)'
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=120,
        help='Test duration in seconds (default: 120)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output file for results (default: auto-generated)'
    )
    parser.add_argument(
        '--output-dir',
        default='./results',
        help='Output directory for results (default: ./results)'
    )
    parser.add_argument(
        '--meaningful-prompts',
        action='store_true',
        help='Use meaningful prompts instead of synthetic'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature (default: 0.0 for deterministic)'
    )
    parser.add_argument(
        '--max-context',
        type=int,
        default=None,
        help='Model maximum context length (auto-detect if not specified)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(
            args.output_dir,
            f"chat_test_users{args.users}_{timestamp}.json"
        )
    
    # Run the test
    asyncio.run(run_test(
        vllm_url=args.url,
        concurrent_users=args.users,
        duration=args.duration,
        output_file=args.output,
        temperature=args.temperature,
        meaningful_prompts=args.meaningful_prompts,
        max_context=args.max_context
    ))


if __name__ == "__main__":
    main()
