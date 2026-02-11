# 1. Run harness to generate prompts and baseline outputs
python varcas_load_harness.py --profile rag_large --output baseline.json

# 2. Generate ground truth using Kimi (or GPT-4, Claude)
export MOONSHOT_API_KEY="sk-kimi-nPZL0EOkHnHEbUI1XcT1lqahNmREvDG5myxATxjJVByOH1soxVtuKcOmeuBfqVLx"
python varcas_ground_truth.py \
    --from-harness baseline.json \
    --provider kimi \
    --output ground_truth.json

# 3. Now validate baseline against ground truth
python varcas_validator.py validate \
    --experiment baseline.json \
    --ground-truth ground_truth.json \
    --output baseline_validation.json

# 4. Run optimized config
python varcas_load_harness.py --profile rag_large --output optimized.json

# 5. Validate optimized (same ground truth)
python varcas_validator.py validate \
    --experiment optimized.json \
    --ground-truth ground_truth.json \
    --output optimized_validation.json

# 6. Compare for degradation
python varcas_validator.py compare \
    --baseline baseline_validation.json \
    --candidate optimized_validation.json \
    --output degradation_report.json
