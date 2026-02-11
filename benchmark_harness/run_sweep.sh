for ctx in small medium large xlarge; do
    for flag in "" "--enable-chunked-prefill"; do
        python varcas_load_harness.py --profile rag_${ctx} --output rag_${ctx}_${flag}.json
    done
done
