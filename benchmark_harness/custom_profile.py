from varcas_load_harness import LoadHarness, LoadProfile
from varcas_load_harness import ArrivalProcess, WorkloadType
from varcas_load_harness import WorkloadProfile, TokenDistribution

# Define custom profile
profile = LoadProfile(
    name="stress_test",
    arrival_process=ArrivalProcess.HAWKES,
    hawkes_base_rate=10.0,
    hawkes_excitation=1.5,  # Highly bursty
    duration_seconds=300,
    workloads=[
        WorkloadProfile(
            name="critical_rag",
            workload_type=WorkloadType.RAG,
            input_dist=TokenDistribution(
                mean=8000, std=2000, min=4000, max=16000
            ),
            output_dist=TokenDistribution(
                mean=300, std=100, min=100, max=1000
            )
        )
    ]
)

# Execute
async with LoadHarness("http://localhost:8000") as harness:
    result = await harness.run(profile)
    
print(f"Saturation at: {result.throughput_rps} req/s")
print(f"Tail latency: {result.latency_p99_ms} ms")

# Save for database
result.save(f"experiments/{profile.name}_{timestamp}.json")
