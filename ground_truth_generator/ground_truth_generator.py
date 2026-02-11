"""
Varcas Ground Truth Generator v1.1
Generate reference answers using external API with caching.
"""

import asyncio
import json
import hashlib
import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import aiohttp


@dataclass
class GroundTruthConfig:
    """Configuration for ground truth generation."""
    provider: str  # "kimi", "openai", "claude", "custom"
    api_key: str
    api_base: Optional[str] = None
    model: str = "kimi-latest"
    temperature: float = 0.0
    max_tokens: int = 2048
    concurrent_requests: int = 10
    cache_dir: str = ".varcas_cache"  # Cache directory
    cache_ttl_days: int = 30  # Cache expiration


class GroundTruthGenerator:
    """
    Generate ground truth references from high-quality external API with caching.
    """
    
    PROVIDER_ENDPOINTS = {
        "kimi": "https://api.moonshot.cn/v1/chat/completions",
        "openai": "https://api.openai.com/v1/chat/completions",
        "claude": "https://api.anthropic.com/v1/messages",
    }
    
    def __init__(self, config: GroundTruthConfig):
        self.config = config
        self.endpoint = config.api_base or self.PROVIDER_ENDPOINTS.get(config.provider)
        if not self.endpoint:
            raise ValueError(f"Unknown provider: {config.provider}")
        
        # Initialize cache
        self._init_cache()
    
    def _init_cache(self):
        """Create cache directory if needed."""
        if not os.path.exists(self.config.cache_dir):
            os.makedirs(self.config.cache_dir)
        
        self.cache_file = os.path.join(
            self.config.cache_dir, 
            f"{self.config.provider}_{self.config.model.replace('/', '_')}_cache.json"
        )
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Dict]:
        """Load cache from disk."""
        if not os.path.exists(self.cache_file):
            return {}
        
        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
            
            # Filter expired entries
            valid_cache = {}
            now = datetime.now()
            ttl = timedelta(days=self.config.cache_ttl_days)
            
            for key, entry in cache.items():
                cached_time = datetime.fromisoformat(entry.get("cached_at", "2000-01-01"))
                if now - cached_time < ttl:
                    valid_cache[key] = entry
            
            print(f"Loaded {len(valid_cache)} cached entries ({len(cache) - len(valid_cache)} expired)")
            return valid_cache
            
        except (json.JSONDecodeError, IOError):
            return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key for prompt."""
        # Hash of provider + model + prompt + temperature
        key_string = f"{self.config.provider}:{self.config.model}:{self.config.temperature}:{prompt}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    def _get_cached(self, prompt: str) -> Optional[Dict]:
        """Check if prompt is in cache."""
        key = self._get_cache_key(prompt)
        entry = self.cache.get(key)
        
        if entry and not entry.get("error"):
            return {
                "prompt_id": "cached",
                "prompt": prompt,
                "reference": entry["reference"],
                "error": None,
                "tokens_input": entry.get("tokens_input", 0),
                "tokens_generated": entry.get("tokens_generated", 0),
                "cached": True
            }
        return None
    
    def _set_cached(self, prompt: str, result: Dict):
        """Store result in cache."""
        key = self._get_cache_key(prompt)
        self.cache[key] = {
            "reference": result.get("reference", ""),
            "tokens_input": result.get("tokens_input", 0),
            "tokens_generated": result.get("tokens_generated", 0),
            "cached_at": datetime.now().isoformat()
        }
        # Save periodically (every 10 new entries)
        if len(self.cache) % 10 == 0:
            self._save_cache()
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=self._get_headers(),
            timeout=aiohttp.ClientTimeout(total=120)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
        self._save_cache()  # Final cache save
    
    def _get_headers(self) -> Dict[str, str]:
        if self.config.provider == "kimi":
            return {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        elif self.config.provider == "openai":
            return {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        elif self.config.provider == "claude":
            return {
                "x-api-key": self.config.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
        return {}
    
    def _build_payload(self, prompt: str) -> Dict:
        if self.config.provider == "claude":
            return {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
        else:
            return {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
    
    async def _generate_single(self, prompt: str, prompt_id: str) -> Dict:
        """Generate ground truth for single prompt with caching."""
        
        # Check cache first
        cached = self._get_cached(prompt)
        if cached:
            cached["prompt_id"] = prompt_id
            print(f"  [CACHE HIT] {prompt_id}")
            return cached
        
        # Generate from API
        print(f"  [API CALL] {prompt_id}")
        payload = self._build_payload(prompt)
        
        try:
            async with self.session.post(self.endpoint, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    result = {
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "reference": "",
                        "error": f"HTTP {resp.status}: {error_text}",
                        "tokens_generated": 0,
                        "cached": False
                    }
                    # Don't cache errors
                    return result
                
                data = await resp.json()
                
                # Extract response text
                if self.config.provider == "claude":
                    text = data.get("content", [{}])[0].get("text", "")
                else:
                    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                usage = data.get("usage", {})
                
                result = {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "reference": text,
                    "error": None,
                    "tokens_input": usage.get("prompt_tokens", 0),
                    "tokens_generated": usage.get("completion_tokens", len(text) // 4),
                    "cached": False
                }
                
                # Cache successful result
                self._set_cached(prompt, result)
                return result
                
        except Exception as e:
            return {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "reference": "",
                "error": str(e),
                "tokens_generated": 0,
                "cached": False
            }
    
    async def generate_from_prompts(self, prompts: List[str]) -> List[Dict]:
        """Generate ground truth for list of prompts with caching."""
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        
        async def bounded_generate(prompt: str, idx: str):
            async with semaphore:
                return await self._generate_single(prompt, idx)
        
        tasks = [
            bounded_generate(prompt, f"gt_{i:04d}")
            for i, prompt in enumerate(prompts)
        ]
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def generate_from_harness_output(self, harness_output_file: str) -> List[Dict]:
        """Generate ground truth from harness experiment output."""
        with open(harness_output_file, 'r') as f:
            data = json.load(f)
        
        # Extract unique prompts
        seen_prompts = {}
        for record in data.get("records", []):
            prompt = record.get("prompt_text", "")
            if prompt and prompt not in seen_prompts:
                seen_prompts[prompt] = record.get("request_id", "")
        
        prompts = list(seen_prompts.keys())
        
        # Check how many are cached
        cached_count = sum(1 for p in prompts if self._get_cached(p))
        print(f"Generating ground truth for {len(prompts)} unique prompts...")
        print(f"  Cached: {cached_count}, To generate: {len(prompts) - cached_count}")
        
        results = await self.generate_from_prompts(prompts)
        
        # Add request_id mapping
        for r in results:
            r["request_id"] = seen_prompts.get(r["prompt"], "")
        
        return results
    
    def save_ground_truth(self, results: List[Dict], filepath: str):
        """Save in validator-compatible format."""
        ground_truth = []
        for r in results:
            if r.get("error"):
                continue
            
            val_type = self._infer_validation_type(r["prompt"])
            
            ground_truth.append({
                "example_id": f"{val_type}:{r['prompt_id']}",
                "prompt": r["prompt"],
                "reference": r["reference"],
                "validation_type": val_type,
                "metadata": {
                    "source": self.config.provider,
                    "model": self.config.model,
                    "tokens_input": r.get("tokens_input", 0),
                    "tokens_generated": r.get("tokens_generated", 0),
                    "from_cache": r.get("cached", False)
                }
            })
        
        with open(filepath, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        # Count cache hits
        cache_hits = sum(1 for r in results if r.get("cached"))
        print(f"Saved {len(ground_truth)} ground truth examples to {filepath}")
        print(f"  From cache: {cache_hits}, From API: {len(ground_truth) - cache_hits}")
        
        return ground_truth
    
    def _infer_validation_type(self, prompt: str) -> str:
        """Infer validation type from prompt characteristics."""
        prompt_lower = prompt.lower()
        
        if "context:" in prompt_lower or len(prompt) > 1000:
            return "rag"
        elif "def " in prompt or "class " in prompt or "import " in prompt:
            return "code"
        elif "write" in prompt_lower and "code" in prompt_lower:
            return "code"
        else:
            return "chat"


# ============================================================================
# CLI
# ============================================================================

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Varcas Ground Truth Generator')
    parser.add_argument('--from-harness', required=True, help='Harness output JSON')
    parser.add_argument('--provider', default='kimi', choices=['kimi', 'openai', 'claude'])
    parser.add_argument('--api-key', help='API key (or set env var)')
    parser.add_argument('--model', help='Model name')
    parser.add_argument('--output', default='ground_truth.json', help='Output file')
    parser.add_argument('--concurrent', type=int, default=10, help='Concurrent requests')
    parser.add_argument('--cache-dir', default='.varcas_cache', help='Cache directory')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    
    args = parser.parse_args()
    
    # Get API key
    env_var_name = "MOONSHOT_API_KEY" if args.provider == "kimi" else f"{args.provider.upper()}_API_KEY"
    api_key = args.api_key or os.getenv(env_var_name)
    if not api_key:
        raise ValueError(f"Provide --api-key or set {env_var_name}")
    
    default_models = {
        "kimi": "kimi-latest",
        "openai": "gpt-4",
        "claude": "claude-3-opus-20240229"
    }
    
    config = GroundTruthConfig(
        provider=args.provider,
        api_key=api_key,
        model=args.model or default_models[args.provider],
        concurrent_requests=args.concurrent,
        cache_dir=args.cache_dir if not args.no_cache else None
    )
    
    async with GroundTruthGenerator(config) as generator:
        results = await generator.generate_from_harness_output(args.from_harness)
        generator.save_ground_truth(results, args.output)
        
        # Print stats
        errors = sum(1 for r in results if r.get("error"))
        success = len(results) - errors
        api_calls = sum(1 for r in results if not r.get("cached"))
        cache_hits = sum(1 for r in results if r.get("cached"))
        total_tokens = sum(r.get("tokens_generated", 0) for r in results if not r.get("cached"))
        
        print(f"\nGeneration complete:")
        print(f"  Success: {success}/{len(results)}")
        print(f"  Errors: {errors}")
        print(f"  API calls: {api_calls}")
        print(f"  Cache hits: {cache_hits}")
        print(f"  Total tokens (API only): {total_tokens}")
        
        if api_calls > 0:
            cost_per_1m = 0.015 if args.provider == "kimi" else 0.03  # Approximate
            print(f"  Est. cost: ${total_tokens * cost_per_1m / 1000000:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
