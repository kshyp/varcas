# Ground Truth Generator

Generate high-quality reference answers using external APIs (Kimi, OpenAI, Claude) for validation purposes. Features intelligent caching to minimize API costs.

## Overview

The Ground Truth Generator creates reference outputs that serve as the "gold standard" for validating LLM inference results. It extracts prompts from actual load test traces and generates deterministic answers (temperature=0) using high-quality external models.

## Features

- **Multi-provider support**: Kimi, OpenAI GPT-4, Claude
- **Smart caching**: Cache responses to avoid redundant API calls
- **Async processing**: Concurrent requests for faster generation
- **TTL expiration**: Automatic cache cleanup for stale entries
- **Trace integration**: Extract prompts directly from benchmark harness output

## Quick Start

```bash
# Set your API key
export KIMI_API_KEY="your-api-key-here"

# Generate ground truth from prompts
python ground_truth_generator.py \
  --input prompts.json \
  --output ground_truth.json \
  --provider kimi \
  --model kimi-latest
```

## Usage

### Basic Usage

```python
from ground_truth_generator import GroundTruthGenerator, GroundTruthConfig

config = GroundTruthConfig(
    provider="kimi",
    api_key="your-api-key",
    model="kimi-latest",
    temperature=0.0,  # Deterministic
    max_tokens=2048,
    concurrent_requests=10,
    cache_ttl_days=30
)

generator = GroundTruthGenerator(config)

# Generate from prompts
prompts = [
    {"prompt": "What is machine learning?", "example_id": "qa_001"},
    {"prompt": "Explain quantum computing", "example_id": "qa_002"}
]

results = asyncio.run(generator.generate_batch(prompts))
```

### Extract from Trace File

```python
# Extract prompts from benchmark harness output
from ground_truth_generator import extract_from_trace

prompts = extract_from_trace("benchmark_output.json")
results = asyncio.run(generator.generate_batch(prompts))
```

### Command Line Interface

```bash
# Basic generation
python ground_truth_generator.py \
  --input prompts.json \
  --output ground_truth.json \
  --provider kimi \
  --api-key $KIMI_API_KEY

# With custom settings
python ground_truth_generator.py \
  --input prompts.json \
  --output ground_truth.json \
  --provider openai \
  --model gpt-4 \
  --temperature 0.0 \
  --concurrent 20 \
  --cache-dir .cache
```

## Input Format

`prompts.json`:
```json
[
  {
    "example_id": "chat_001",
    "prompt": "What are the benefits of renewable energy?",
    "validation_type": "qa",
    "metadata": {"category": "science", "difficulty": "easy"}
  },
  {
    "example_id": "code_001",
    "prompt": "Write a Python function to reverse a string",
    "validation_type": "code",
    "metadata": {"language": "python", "task": "string-manipulation"}
  }
]
```

## Output Format

`ground_truth.json`:
```json
[
  {
    "example_id": "chat_001",
    "prompt": "What are the benefits of renewable energy?",
    "reference": "Renewable energy offers numerous benefits including...",
    "provider": "kimi",
    "model": "kimi-latest",
    "tokens_generated": 156,
    "cached": false,
    "generated_at": "2026-02-11T10:30:00"
  }
]
```

## Supported Providers

| Provider | Endpoint | Default Model |
|----------|----------|---------------|
| Kimi | api.moonshot.cn | kimi-latest |
| OpenAI | api.openai.com | gpt-4 |
| Claude | api.anthropic.com | claude-3-opus |
| Custom | User-defined | User-defined |

## Caching

Responses are cached by default to minimize API costs:

- **Cache key**: MD5 hash of (provider + model + prompt)
- **Location**: `.varcas_cache/` (configurable)
- **TTL**: 30 days (configurable)
- **Format**: JSON with metadata

Cache hit example:
```
Loaded 45 cached entries (3 expired)
Processing: 100%|████████████████| 50/50 [00:02<00:00, 25.12it/s]
Results: 50 total, 45 from cache, 5 new API calls
```

## Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Ground truth source** | External API (Kimi/GPT-4/Claude) | Higher quality than local small models |
| **Generation strategy** | Generate once, validate multiple | Cost efficiency, consistency |
| **Determinism** | temperature=0 | Same prompt → same reference |
| **Prompt extraction** | From harness output | Ground truth matches real test distribution |

## Cost Optimization

1. **Use caching**: Cache hits avoid API calls entirely
2. **Generate once**: Reuse ground truth across multiple validation runs
3. **Batch requests**: Concurrent processing reduces wall time
4. **Extract from traces**: Use actual prompts from load tests (no synthetic data)

## Integration with Validator

Use generated ground truth with the validation module:

```bash
# 1. Generate ground truth
python ground_truth_generator.py --input prompts.json --output ground_truth.json

# 2. Run inference
python benchmark_harness/varcas_load_harness.py --profile chat_medium --save-results vllm_output.json

# 3. Validate
python validator/varcas_validator.py \
  --ground-truth ground_truth.json \
  --results vllm_output.json \
  --method semantic_similarity
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `KIMI_API_KEY` | Kimi API key | For Kimi provider |
| `OPENAI_API_KEY` | OpenAI API key | For OpenAI provider |
| `ANTHROPIC_API_KEY` | Claude API key | For Claude provider |

## Files

| File | Description |
|------|-------------|
| `ground_truth_generator.py` | Main generator implementation |
| `workflow.sh` | Example workflow script |
| `cached_run_examples.sh` | Examples with caching |
| `design_decisions.txt` | Design rationale documentation |

## Troubleshooting

### API rate limit errors
- Reduce `--concurrent` setting
- Add delays between batches
- Use caching to minimize calls

### Cache not working
- Verify cache directory exists and is writable
- Check TTL settings (old entries expire)
- Ensure consistent provider/model names

### Different results for same prompt
- Verify temperature=0 for determinism
- Check for model updates (version drift)
- Ensure consistent system prompts
