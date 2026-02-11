from varcas_validator import compare_experiments, ValidationMethod

# Run comparison
degradation = compare_experiments(
    baseline_file="baseline.json",
    candidate_file="optimized.json",
    method=ValidationMethod.SEMANTIC_SIMILARITY,
    epsilon_abs=0.02,      # 2% absolute tolerance
    epsilon_rel=5.0        # 5% relative tolerance
)

# Customer-facing summary
summary = degradation.to_customer_summary()
print(summary['accuracy_preserved'])  # True/False
print(summary['degradation_status'])   # "ACCEPTABLE"
print(summary['metrics']['relative_change_percent'])  # -1.20

# Decision
if summary['accuracy_preserved']:
    print("Optimization approved")
else:
    print(f"Optimization rejected: {degradation.degradation_status}")
