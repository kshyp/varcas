# Example: Chunked prefill optimization sweep
for context_size in ["small", "medium", "large", "xlarge"]:
    for chunked in [False, True]:
        profile = get_rag_profile(context_size, "high")
        # Run with vLLM --enable-chunked-prefill if chunked=True
        result = await harness.run(profile)
        # Log: (rag_large, chunked_prefill=True, throughput=X, latency=Y)
