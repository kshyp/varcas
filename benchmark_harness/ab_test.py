async def ab_test_config():
    profile = get_rag_profile("large", "high")
    profile.seed = 42  # Identical traffic
    
    # Test A: Default vLLM
    result_a = await run_test(profile, vllm_flags=[])
    
    # Test B: Optimized
    result_b = await run_test(profile, vllm_flags=["--enable-chunked-prefill"])
    
    improvement = (result_b.throughput_rps / result_a.throughput_rps - 1) * 100
    print(f"Throughput improvement: {improvement:.1f}%")
    
    # Validate latency SLO
    if result_b.latency_p99_ms < 2000:  # 2 second SLO
        print("Config B accepted")
    else:
        print("Config B rejected: latency regression")
