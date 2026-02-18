# Varcas Validator

Post-hoc accuracy and safety validation framework for LLM inference. Compare model outputs against ground truth and detect performance degradation.

## Overview

The Validator provides comprehensive accuracy validation with multiple validation methods, statistical analysis, and clean degradation metrics for customer reporting.

## Features

- **Multiple validation methods**: Exact match, semantic similarity, ROUGE-L, code execution, LLM-as-judge
- **Statistical analysis**: Mean, std, percentiles (P10, P50, P90)
- **Degradation detection**: Compare optimized vs baseline configurations
- **Pass rate tracking**: Configurable thresholds for acceptance criteria
- **Batch processing**: Validate thousands of outputs efficiently

## Quick Start

```bash
# Validate against ground truth
python varcas_validator.py \
  --ground-truth ground_truth.json \
  --results vllm_output.json \
  --method semantic_similarity \
  --threshold 0.85
```

## Validation Methods

| Method | Use Case | Description |
|--------|----------|-------------|
| `exact_match` | Code, structured data | Character-by-character comparison |
| `semantic_similarity` | QA, chat, summaries | Sentence embeddings similarity |
| `rouge_l` | Summarization | Longest common subsequence F1 |
| `code_execution` | Code generation | Execute and compare outputs |
| `llm_judge` | Open-ended | Another LLM judges quality |

## Usage

### Basic Validation

```python
from varcas_validator import VarcasValidator, ValidationMethod

validator = VarcasValidator(
    method=ValidationMethod.SEMANTIC_SIMILARITY,
    threshold=0.85
)

# Load ground truth and results
with open("ground_truth.json") as f:
    ground_truth = json.load(f)

with open("vllm_output.json") as f:
    results = json.load(f)

# Validate
report = validator.validate_batch(ground_truth, results)

print(f"Pass rate: {report.pass_rate:.1%}")
print(f"Mean score: {report.mean_score:.3f}")
```

### Command Line

```bash
# Basic validation
python varcas_validator.py \
  --ground-truth ground_truth.json \
  --results vllm_output.json

# With specific method and threshold
python varcas_validator.py \
  --ground-truth ground_truth.json \
  --results vllm_output.json \
  --method rouge_l \
  --threshold 0.75 \
  --output validation_report.json

# Degradation analysis (compare optimized vs baseline)
python varcas_validator.py \
  --ground-truth ground_truth.json \
  --baseline baseline_results.json \
  --optimized optimized_results.json \
  --method semantic_similarity
```

### Degradation Analysis

Compare two configurations (e.g., baseline vs optimized):

```python
from varcas_validator import DegradationAnalyzer

analyzer = DegradationAnalyzer()

report = analyzer.compare(
    baseline_results=baseline_data,
    optimized_results=optimized_data,
    ground_truth=ground_truth_data,
    method=ValidationMethod.SEMANTIC_SIMILARITY
)

print(f"Degradation: {report.degradation_percent:.1f}%")
print(f"Acceptable: {report.is_acceptable}")
```

## Input Format

### Ground Truth (from ground_truth_generator)

```json
[
  {
    "example_id": "qa_001",
    "prompt": "What is machine learning?",
    "reference": "Machine learning is a subset of AI...",
    "validation_type": "qa"
  }
]
```

### Model Results (from benchmark harness)

```json
[
  {
    "request_id": "req_001",
    "example_id": "qa_001",
    "generated_text": "Machine learning enables computers to learn...",
    "tokens_generated": 45,
    "latency_ms": 523.4
  }
]
```

## Output Format

### Validation Report

```json
{
  "validation_metrics": {
    "total_validated": 100,
    "mean_score": 0.892,
    "std_score": 0.123,
    "min_score": 0.456,
    "max_score": 0.998,
    "p10_score": 0.712,
    "p50_score": 0.912,
    "p90_score": 0.978,
    "threshold": 0.85,
    "pass_rate": 0.87,
    "passed_count": 87
  },
  "individual_results": [
    {
      "request_id": "req_001",
      "example_id": "qa_001",
      "score": 0.945,
      "passed": true,
      "method": "semantic_similarity"
    }
  ]
}
```

### Degradation Report

```json
{
  "baseline_mean": 0.912,
  "optimized_mean": 0.898,
  "absolute_degradation": -0.014,
  "degradation_percent": -1.5,
  "is_acceptable": true,
  "threshold": 5.0,
  "confidence_interval": [-2.3, 0.1],
  "sample_size": 100
}
```

## Validation Types

Different validation strategies for different workloads:

| Type | Description | Default Method |
|------|-------------|----------------|
| `qa` | Question answering | semantic_similarity |
| `code` | Code generation | code_execution |
| `summary` | Text summarization | rouge_l |
| `chat` | Conversational | semantic_similarity |

## Metrics Explanation

### Core Metrics

- **Mean Score**: Average similarity across all examples
- **Std Score**: Standard deviation (lower = more consistent)
- **P50 Score**: Median score (robust to outliers)
- **P90 Score**: 90th percentile (worst 10% performance)

### Pass Rate

Percentage of examples exceeding threshold:
- Default threshold: 0.85 (85% similarity)
- Configurable per validation method
- Primary acceptance criterion

### Degradation Metrics

- **Absolute**: Mean difference (optimized - baseline)
- **Percent**: Percentage change from baseline
- **Acceptable**: Within configurable threshold (default: <5% degradation)

## Example Workflows

### Validate Optimization

```bash
# 1. Generate ground truth (one-time)
cd ../ground_truth_generator
python ground_truth_generator.py --input prompts.json --output gt.json

# 2. Run baseline benchmark
cd ../benchmark_harness
python varcas_load_harness.py --profile chat_medium --save-results baseline.json

# 3. Apply optimizations (e.g., increase batch size)
# ... modify vLLM config ...

# 4. Run optimized benchmark
python varcas_load_harness.py --profile chat_medium --save-results optimized.json

# 5. Validate no accuracy degradation
cd ../validator
python varcas_validator.py \
  --ground-truth gt.json \
  --baseline baseline.json \
  --optimized optimized.json \
  --threshold 5.0
```

### Batch Validation

```bash
# Validate all results in a directory
for result in results/*.json; do
  python varcas_validator.py \
    --ground-truth ground_truth.json \
    --results "$result" \
    --output "validation/$(basename $result)"
done
```

## Files

| File | Description |
|------|-------------|
| `varcas_validator.py` | Main validation framework |
| `do_validation.py` | Simple validation script |
| `do_validation.sh` | Batch validation wrapper |
| `do_validate_single.sh` | Single example validation |
| `customer_metrics.txt` | Example metric definitions |

## Integration with Benchmark Harness

The validator works seamlessly with benchmark harness output:

```python
# In benchmark harness, save results with --save-results
python varcas_load_harness.py --profile chat_medium --save-results output.json

# Output format is automatically compatible with validator
python varcas_validator.py --ground-truth gt.json --results output.json
```

## Troubleshooting

### Low scores on valid outputs
- Try `semantic_similarity` instead of `exact_match`
- Check for whitespace/formatting differences
- Adjust threshold based on validation type

### Missing examples in results
- Verify `example_id` matching between ground truth and results
- Check for request failures in benchmark output

### Code execution failures
- Ensure execution environment has required dependencies
- Set appropriate timeouts for long-running code
- Sanitize code before execution (security)

## Configuration

Create `.validator_config.json` for default settings:

```json
{
  "default_method": "semantic_similarity",
  "default_threshold": 0.85,
  "semantic_similarity": {
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  },
  "code_execution": {
    "timeout_seconds": 5,
    "allowed_imports": ["math", "json", "re"]
  }
}
```

## Performance

Validation throughput:
- **Exact match**: ~10,000 examples/sec
- **Semantic similarity**: ~100 examples/sec (GPU), ~10/sec (CPU)
- **ROUGE-L**: ~500 examples/sec
- **Code execution**: ~10 examples/sec (with timeout)

For large datasets, use batch processing and consider parallel execution.
