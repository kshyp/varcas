# First run - generates and caches
python varcas_ground_truth.py --from-harness baseline.json --provider kimi

# Second run - uses cache, no API calls
python varcas_ground_truth.py --from-harness optimized.json --provider kimi

# Force refresh
python varcas_ground_truth.py --from-harness baseline.json --provider kimi --no-cache

# Custom cache location
python varcas_ground_truth.py --from-harness baseline.json --cache-dir /tmp/my_cache
