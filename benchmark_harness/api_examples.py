"""
Programmatic usage of Varcas Load Harness
"""
import asyncio
from varcas_load_harness import LoadHarness, LoadProfile, WorkloadProfile, TokenDistribution
from varcas_load_harness import ArrivalProcess, WorkloadType

async def custom_test():
    # Define custom workload
    profile = LoadProfile(
        name="my_custom_test",
        arrival_process=ArrivalProcess.POISSON,
        target_rps=15.0,
        duration_seconds=90,
        open_loop=True,
        seed=42,
        workloads=[
            WorkloadProfile(
                name="custom_rag",
                workload_type=WorkloadType.RAG,
                input_dist=TokenDistribution(
                    mean=3000,      # 3K token contexts
                    std=1000,
                    min=500,
                    max=8000
                ),
                output_dist=TokenDistribution(
                    mean=250,
                    std=100,
                    min=50,
                    max=1000
                )
            )
        ]
    )
    
    # Run test
    async with LoadHarness("http://localhost:8000") as harness:
        result = await harness.run(profile)
    
    # Access metrics
    print(f"Throughput: {result.throughput_rps:.2f} req/s")
    print(f"TTFT p99: {result.ttft_p99:.1f} ms")
    
    # Save detailed results
    result.save("custom_test.json")
    
    return result

# Run comparison test
async def compare_configs():
    """A/B test two vLLM configurations."""
    profile = LoadProfile(
        name="ab_test",
        arrival_process=ArrivalProcess.POISSON,
        target_rps=20.0,
        duration_seconds=60,
        open_loop=True,
        seed=42  # Same seed = same traffic
    )
    
    # Test A: Default vLLM
    result_a = await run_test(profile, vllm_flags=[])
    
    # Test B: Optimized - restart vLLM with new flags first
    input("Restart vLLM with optimized config, then press Enter...")
    
    result_b = await run_test(profile, vllm_flags=["--enable-chunked-prefill"])
    
    # Compare
    print(f"Config A: {result_a.throughput_rps:.2f} req/s, TTFT p99: {result_a.ttft_p99:.1f}ms")
    print(f"Config B: {result_b.throughput_rps:.2f} req/s, TTFT p99: {result_b.ttft_p99:.1f}ms")
    print(f"Improvement: {(result_b.throughput_rps/result_a.throughput_rps - 1)*100:.1f}%")

if __name__ == "__main__":
    # Run custom test
    asyncio.run(custom_test())
    
    # Or run comparison
    # asyncio.run(compare_configs())
