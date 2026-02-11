"""
Varcas Load Harness v1.0
A configurable load generator for vLLM performance testing.
"""

import asyncio
import json
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union, Any
from enum import Enum
import statistics
from datetime import datetime

# Optional: numpy for better distributions
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Fallback implementations


class ArrivalProcess(Enum):
    POISSON = "poisson"
    HAWKES = "hawkes"
    TRACE = "trace"
    CYCLICAL = "cyclical"


class WorkloadType(Enum):
    CHAT = "chat"
    RAG = "rag"
    CODE = "code"
    BATCH = "batch"


@dataclass
class TokenDistribution:
    """Statistical distribution for token counts."""
    mean: float
    std: float
    min: int = 1
    max: int = 8192
    distribution: str = "lognormal"  # lognormal, normal, uniform
    
    def sample(self, rng: random.Random) -> int:
        if self.distribution == "lognormal":
            if HAS_NUMPY:
                mu = np.log(self.mean) - 0.5 * np.log(1 + (self.std/self.mean)**2)
                sigma = np.sqrt(np.log(1 + (self.std/self.mean)**2))
                val = rng.lognormvariate(mu, sigma) if hasattr(rng, 'lognormvariate') else self._fallback_lognormal(rng)
            else:
                val = self._fallback_lognormal(rng)
        elif self.distribution == "normal":
            val = rng.gauss(self.mean, self.std)
        else:  # uniform
            val = rng.uniform(self.min, self.max)
        
        return int(max(self.min, min(self.max, val)))
    
    def _fallback_lognormal(self, rng: random.Random) -> float:
        # Approximate lognormal with gamma for standard library
        # mean = k*theta, var = k*theta^2
        if self.std < 0.1:
            return self.mean
        theta = (self.std ** 2) / self.mean
        k = self.mean / theta
        # Use normal approximation for large k
        if k > 10:
            return rng.gauss(self.mean, self.std)
        # Fallback: exponential-ish
        return rng.expovariate(1.0 / self.mean)


@dataclass
class WorkloadProfile:
    """Defines a single workload type's characteristics."""
    name: str
    workload_type: WorkloadType
    input_dist: TokenDistribution
    output_dist: TokenDistribution
    weight: float = 1.0
    
    system_prompt: str = ""
    context_template: str = "{context}\n\nQuestion: {question}"
    question_template: str = "Explain the following in detail:"


@dataclass
class LoadProfile:
    """Complete load configuration."""
    name: str
    arrival_process: ArrivalProcess
    
    # Arrival parameters
    target_rps: float = 10.0
    concurrency: int = 10
    duration_seconds: int = 60
    
    # Hawkes parameters
    hawkes_base_rate: float = 5.0
    hawkes_excitation: float = 0.5
    hawkes_decay: float = 1.0
    
    # Cyclical parameters
    cyclical_base: float = 5.0
    cyclical_amplitude: float = 10.0
    cyclical_period: float = 3600
    
    # Trace replay - either timestamps or full prompts
    trace_timestamps: Optional[List[float]] = None
    trace_prompts: Optional[List[Dict]] = None  # Full prompts for exact replay
    
    # Workload mix
    workloads: List[WorkloadProfile] = field(default_factory=list)
    
    # Control
    seed: int = 42
    open_loop: bool = True
    
    # Model context limit - total tokens (input + output) will be capped to this
    model_max_context: int = 8192
    # Safety margin to reserve for special tokens and estimation errors
    context_margin: int = 50
    
    def __post_init__(self):
        if not self.workloads:
            self.workloads = [WorkloadProfile(
                name="default_chat",
                workload_type=WorkloadType.CHAT,
                input_dist=TokenDistribution(mean=50, std=30, min=10, max=500),
                output_dist=TokenDistribution(mean=100, std=50, min=20, max=1000)
            )]


@dataclass
class RequestRecord:
    """Single request telemetry."""
    request_id: str
    workload_name: str
    timestamp_sent: float = 0.0
    timestamp_first_token: Optional[float] = None
    timestamp_last_token: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_text: str = ""
    output_text: str = ""
    error: Optional[str] = None
    
    @property
    def ttft_ms(self) -> Optional[float]:
        if self.timestamp_first_token and self.timestamp_sent:
            return (self.timestamp_first_token - self.timestamp_sent) * 1000
        return None
    
    @property
    def total_latency_ms(self) -> Optional[float]:
        if self.timestamp_last_token and self.timestamp_sent:
            return (self.timestamp_last_token - self.timestamp_sent) * 1000
        return None
    
    @property
    def tpot_ms(self) -> Optional[float]:
        if self.timestamp_last_token and self.timestamp_first_token and self.output_tokens > 1:
            return (self.timestamp_last_token - self.timestamp_first_token) * 1000 / (self.output_tokens - 1)
        return None


@dataclass
class ExperimentResult:
    """Complete experiment results."""
    experiment_id: str
    profile_name: str
    config: Dict
    start_time: str
    end_time: str = ""
    records: List[RequestRecord] = field(default_factory=list)
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    throughput_rps: float = 0.0
    throughput_tok_s: float = 0.0
    
    ttft_p50: float = 0.0
    ttft_p99: float = 0.0
    tpot_p50: float = 0.0
    tpot_p99: float = 0.0
    latency_p50: float = 0.0
    latency_p99: float = 0.0
    
    def compute_metrics(self):
        self.total_requests = len(self.records)
        self.successful_requests = sum(1 for r in self.records if r.error is None)
        self.failed_requests = self.total_requests - self.successful_requests
        
        if not self.records:
            return
        
        duration = max(1.0, self.records[-1].timestamp_sent - self.records[0].timestamp_sent)
        self.throughput_rps = self.successful_requests / duration
        
        total_tokens = sum(r.output_tokens for r in self.records if r.error is None)
        self.throughput_tok_s = total_tokens / duration
        
        ttfts = [r.ttft_ms for r in self.records if r.ttft_ms is not None]
        tpots = [r.tpot_ms for r in self.records if r.tpot_ms is not None]
        latencies = [r.total_latency_ms for r in self.records if r.total_latency_ms is not None]
        
        if ttfts:
            self.ttft_p50 = statistics.median(ttfts)
            self.ttft_p99 = self._percentile(ttfts, 99)
        
        if tpots:
            self.tpot_p50 = statistics.median(tpots)
            self.tpot_p99 = self._percentile(tpots, 99)
        
        if latencies:
            self.latency_p50 = statistics.median(latencies)
            self.latency_p99 = self._percentile(latencies, 99)
    
    def _percentile(self, data: List[float], p: float) -> float:
        if HAS_NUMPY:
            return float(np.percentile(data, p))
        # Fallback: nearest rank method
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100.0
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
    
    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "profile_name": self.profile_name,
            "config": self.config,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metrics": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "throughput_rps": self.throughput_rps,
                "throughput_tok_s": self.throughput_tok_s,
                "ttft_p50_ms": self.ttft_p50,
                "ttft_p99_ms": self.ttft_p99,
                "tpot_p50_ms": self.tpot_p50,
                "tpot_p99_ms": self.tpot_p99,
                "latency_p50_ms": self.latency_p50,
                "latency_p99_ms": self.latency_p99
            },
            "records": [asdict(r) for r in self.records]
        }
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


import os


class SyntheticContentGenerator:
    """Generates text content to target specific token counts.
    
    Can use LLM API (Kimi/Moonshot) to generate meaningful content,
    or fall back to synthetic gibberish if no API key is provided.
    """
    
    def __init__(self, char_per_token: float = 4.0, 
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 cache_file: str = "prompt_cache.json",
                 use_llm: bool = False):
        self.char_per_token = char_per_token
        self.use_llm = use_llm and (api_key or os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY"))
        self.api_key = api_key or os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
        self.api_base = api_base or os.environ.get("KIMI_API_BASE") or os.environ.get("MOONSHOT_API_BASE") or "https://api.moonshot.cn/v1"
        self.cache_file = cache_file
        self._cache = self._load_cache()
        self._cache_dirty = False
        
        # Topics for meaningful content generation
        self.topics = [
            "explain machine learning concepts",
            "describe a scientific phenomenon", 
            "tell a short story about everyday life",
            "explain historical events",
            "describe how technology works",
            "discuss philosophical questions",
            "explain economic concepts",
            "describe nature and the environment",
            "explain health and medical topics",
            "describe art and culture",
            "explain programming concepts",
            "discuss social issues"
        ]
        
        # Fallback synthetic words
        self.words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their",
            "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
            "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them",
            "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
            "think", "also", "back", "after", "use", "two", "how", "our", "work",
            "first", "well", "way", "even", "new", "want", "because", "any", "these",
            "give", "day", "most", "us", "is", "was", "are", "were", "been", "has"
        ]
        
        # Pre-generated meaningful templates for different sizes
        self._templates = self._init_templates()
    
    def _load_cache(self) -> Dict:
        """Load cached prompts from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}
    
    def save_cache(self):
        """Save cache to disk if dirty."""
        if self._cache_dirty and self.cache_file:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self._cache, f, indent=2)
                self._cache_dirty = False
            except IOError as e:
                print(f"Warning: Could not save prompt cache: {e}")
    
    def _init_templates(self) -> Dict[int, List[str]]:
        """Initialize pre-written meaningful templates of various lengths."""
        return {
            50: [
                "What are the main benefits of regular exercise for mental health?",
                "Explain how photosynthesis works in simple terms.",
                "What causes the seasons to change throughout the year?",
                "Describe the process of learning a new language.",
                "How does artificial intelligence impact daily life?",
            ],
            100: [
                "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions with minimal human intervention.",
                "Climate change refers to long-term shifts in temperatures and weather patterns. While these changes can be natural, human activities since the 1800s have been the main driver, primarily due to burning fossil fuels like coal, oil, and gas.",
                "The internet is a global network of interconnected computers that communicate using standardized protocols. It enables the sharing of information, communication, and access to services across geographical boundaries instantaneously.",
            ],
            200: [
                "Python is a high-level programming language known for its simplicity and readability. Created by Guido van Rossum and first released in 1991, Python emphasizes code readability with its use of significant indentation. It supports multiple programming paradigms, including structured, object-oriented, and functional programming. Python's extensive standard library and large ecosystem of third-party packages make it popular for web development, data analysis, artificial intelligence, scientific computing, and automation.",
                "The human brain is an incredibly complex organ that serves as the control center of the nervous system. It contains approximately 86 billion neurons that communicate through synapses. The brain is responsible for processing sensory information, controlling motor functions, regulating emotions, forming memories, and enabling consciousness. Different regions specialize in different functions: the frontal lobe handles decision-making, the temporal lobe processes auditory information, and the occipital lobe manages visual processing.",
            ],
            500: [
                "Blockchain technology is a decentralized digital ledger that records transactions across multiple computers in a way that ensures security, transparency, and immutability. Each block in the chain contains a cryptographic hash of the previous block, a timestamp, and transaction data. This structure makes it virtually impossible to alter historical records without consensus from the network. Bitcoin, created in 2009 by an unknown person using the pseudonym Satoshi Nakamoto, was the first application of blockchain technology. Since then, the technology has found applications beyond cryptocurrency, including supply chain management, voting systems, digital identity verification, and smart contracts. The key advantages of blockchain include reduced transaction costs, increased transparency, enhanced security through cryptography, and elimination of single points of failure. However, challenges remain regarding scalability, energy consumption (particularly for proof-of-work systems), and regulatory uncertainty. Ethereum extended blockchain capabilities by introducing programmable smart contracts, enabling decentralized applications (DApps) that run exactly as programmed without downtime, censorship, or third-party interference.",
                "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy stored in glucose molecules. This process occurs primarily in the chloroplasts of plant cells, specifically using chlorophyll, the green pigment that gives plants their color. The overall chemical equation for photosynthesis is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. This means six molecules of carbon dioxide and six molecules of water, using light energy, produce one glucose molecule and six oxygen molecules. The process consists of two main stages: the light-dependent reactions and the Calvin cycle (light-independent reactions). In the light-dependent reactions, which occur in the thylakoid membranes, chlorophyll absorbs light energy to produce ATP and NADPH, releasing oxygen as a byproduct. The Calvin cycle occurs in the stroma and uses the ATP and NADPH to convert carbon dioxide into glucose through a series of enzyme-mediated reactions. Photosynthesis is crucial for life on Earth as it produces the oxygen we breathe and forms the base of most food chains.",
            ],
            1000: [
                "The Renaissance was a fervent period of European cultural, artistic, political, and economic rebirth following the Middle Ages, generally considered to have taken place from the 14th to the 17th century. The term 'Renaissance' comes from the French word for 'rebirth,' reflecting the renewed interest in classical antiquity, particularly the art, literature, and philosophy of ancient Greece and Rome. This movement began in Florence, Italy, fueled by the wealth of powerful families like the Medici, who were patrons of the arts. The Renaissance saw remarkable achievements across multiple fields. In art, masters like Leonardo da Vinci, Michelangelo, and Raphael revolutionized painting and sculpture through techniques such as linear perspective, chiaroscuro (the use of strong contrasts between light and dark), and anatomical accuracy. Leonardo's Mona Lisa and The Last Supper, Michelangelo's David and the Sistine Chapel ceiling, and Raphael's School of Athens remain iconic works. In science, figures like Galileo Galilei and Nicolaus Copernicus challenged geocentric views of the universe, laying groundwork for modern astronomy and physics. The printing press, invented by Johannes Gutenberg around 1440, democratized knowledge by making books more accessible. In literature, writers like Dante Alighieri, Geoffrey Chaucer, and William Shakespeare elevated vernacular languages and explored complex human emotions and experiences. The Renaissance also saw the rise of humanism, an intellectual movement emphasizing human potential, individual achievement, and secular concerns alongside religious ones. This period fundamentally transformed European society and laid the foundations for the modern Western world.",
            ],
            2000: [
                "Climate change represents one of the most significant challenges facing humanity in the 21st century. The Earth's average temperature has increased by approximately 1.1 degrees Celsius since the pre-industrial era, primarily due to human activities that release greenhouse gases into the atmosphere. The burning of fossil fuels—coal, oil, and natural gas—for energy production, transportation, and industrial processes is the largest source of emissions, responsible for about 75% of global greenhouse gas emissions. Deforestation, agriculture, and industrial processes contribute significantly as well. The consequences of climate change are far-reaching and already observable: rising sea levels threaten coastal communities, more frequent and intense extreme weather events cause destruction and displacement, shifting precipitation patterns affect agriculture and water security, and warming temperatures disrupt ecosystems and biodiversity. Ocean acidification, caused by absorbed carbon dioxide, endangers marine life and fisheries that billions depend on for protein. The Paris Agreement, adopted in 2015, represents a global commitment to limit warming to well below 2 degrees Celsius above pre-industrial levels, with efforts to limit it to 1.5 degrees. Achieving this requires rapid decarbonization of energy systems, transitioning to renewable sources like solar, wind, and hydroelectric power, improving energy efficiency, electrifying transportation, and developing carbon removal technologies. Adaptation strategies are equally important, including building resilient infrastructure, developing drought-resistant crops, protecting natural ecosystems that serve as carbon sinks, and implementing early warning systems for extreme weather. The economic implications are substantial—estimated costs of inaction far exceed investments in mitigation and adaptation. Climate change also raises profound ethical questions about intergenerational justice, as today's emissions burden future generations, and climate justice, as vulnerable populations who contributed least to the problem often suffer most from its impacts. Addressing climate change requires unprecedented international cooperation, technological innovation, policy reforms including carbon pricing, and transformations in individual and collective behavior.",
            ],
        }
    
    async def _generate_via_api(self, target_tokens: int, topic: Optional[str] = None) -> Optional[str]:
        """Generate meaningful content using Kimi/Moonshot API."""
        if not self.api_key:
            return None
        
        import aiohttp
        
        # Scale target to approximate characters
        target_chars = int(target_tokens * self.char_per_token * 0.8)  # 0.8 factor for safety
        
        selected_topic = topic or random.choice(self.topics)
        
        prompt = f"Write a concise, informative paragraph about: {selected_topic}. "
        prompt += f"Aim for approximately {target_chars} characters. Be factual and educational."
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "moonshot-v1-8k",  # or kimi model
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": min(target_tokens, 2000),
                "temperature": 0.7
            }
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "choices" in data and len(data["choices"]) > 0:
                            content = data["choices"][0]["message"]["content"].strip()
                            # Ensure it ends with a period
                            if not content.endswith(('.', '!', '?')):
                                content += "."
                            return content
                    else:
                        print(f"API error: HTTP {resp.status}")
                        
        except Exception as e:
            print(f"API generation failed: {e}")
        
        return None
    
    def _get_cached_or_generate(self, target_tokens: int, rng: random.Random, 
                                 topic: Optional[str] = None) -> str:
        """Try cache first, then templates, then fallback to synthetic."""
        cache_key = f"{target_tokens}_{topic or 'any'}"
        
        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try to find closest template size
        available_sizes = sorted(self._templates.keys())
        closest_size = None
        for size in available_sizes:
            if size >= target_tokens * 0.5:
                closest_size = size
                break
        
        if closest_size:
            templates = self._templates[closest_size]
            content = rng.choice(templates)
            # Truncate or pad to target length
            target_chars = int(target_tokens * self.char_per_token)
            if len(content) > target_chars:
                content = content[:target_chars]
                # Ensure it ends cleanly
                last_period = content.rfind('.')
                if last_period > target_chars * 0.7:
                    content = content[:last_period + 1]
            
            # Store in cache
            self._cache[cache_key] = content
            self._cache_dirty = True
            return content
        
        # Fallback to synthetic
        return self._generate_synthetic(target_tokens, rng)
    
    def _generate_synthetic(self, target_tokens: int, rng: random.Random) -> str:
        """Original synthetic generation as fallback."""
        target_chars = int(target_tokens * self.char_per_token)
        
        words = []
        current_chars = 0
        
        while current_chars < target_chars:
            word = rng.choice(self.words)
            words.append(word)
            current_chars += len(word) + 1
        
        text = " ".join(words)
        
        if len(text) > target_chars:
            text = text[:target_chars]
        
        return text.capitalize() + "."
    
    def generate_to_length(self, target_tokens: int, rng: random.Random, 
                          topic: Optional[str] = None,
                          use_meaningful: Optional[bool] = None) -> str:
        """Generate text of approximately target_tokens length.
        
        Args:
            target_tokens: Approximate number of tokens desired
            rng: Random number generator
            topic: Optional topic for meaningful generation
            use_meaningful: Override to force meaningful/synthetic generation
        
        Returns:
            Generated text string
        """
        use_llm = use_meaningful if use_meaningful is not None else self.use_llm
        
        if use_llm:
            # For async context, we can't await here, so use templates/caching
            return self._get_cached_or_generate(target_tokens, rng, topic)
        
        return self._generate_synthetic(target_tokens, rng)
        target_chars = int(target_tokens * self.char_per_token)
        
        words = []
        current_chars = 0
        
        while current_chars < target_chars:
            word = rng.choice(self.words)
            words.append(word)
            current_chars += len(word) + 1
        
        text = " ".join(words)
        
        if len(text) > target_chars:
            text = text[:target_chars]
        
        return text.capitalize() + "."
    
    def build_rag_prompt(self, context_tokens: int, question_tokens: int, 
                         rng: random.Random, use_meaningful: bool = False) -> tuple[str, str]:
        context = self.generate_to_length(context_tokens, rng, topic="provide context information", use_meaningful=use_meaningful)
        question = self.generate_to_length(question_tokens, rng, topic="ask a relevant question", use_meaningful=use_meaningful)
        
        prompt = f"Context: {context}\n\nQuestion: {question}"
        return prompt, question
    
    def build_chat_prompt(self, history_tokens: int, message_tokens: int,
                          rng: random.Random, use_meaningful: bool = False) -> str:
        if history_tokens > 0:
            history = self.generate_to_length(history_tokens, rng, topic="continue conversation", use_meaningful=use_meaningful)
            message = self.generate_to_length(message_tokens, rng, topic="user query", use_meaningful=use_meaningful)
            return f"Previous conversation: {history}\n\nUser: {message}\nAssistant:"
        else:
            message = self.generate_to_length(message_tokens, rng, topic="user query", use_meaningful=use_meaningful)
            return f"User: {message}\nAssistant:"
    
    def build_code_prompt(self, context_tokens: int, instruction_tokens: int,
                         rng: random.Random, use_meaningful: bool = False) -> str:
        context = self.generate_to_length(context_tokens, rng, topic="code context", use_meaningful=use_meaningful)
        instruction = self.generate_to_length(instruction_tokens, rng, topic="programming task", use_meaningful=use_meaningful)
        
        return f"# File context:\n# {context}\n\n# Task: {instruction}\n"


class ArrivalTimeGenerator:
    """Generates inter-arrival times for different processes."""
    
    def __init__(self, profile: LoadProfile, rng: random.Random):
        self.profile = profile
        self.rng = rng
        self.start_time = time.time()
        
        self.hawkes_last_arrival = 0.0
        self.hawkes_intensity = profile.hawkes_base_rate
        
        # For trace replay with prompts
        self._trace_idx = 0
        self._trace_offset = time.time()
    
    def has_more(self) -> bool:
        """Check if there are more arrivals in the trace."""
        if self.profile.arrival_process == ArrivalProcess.TRACE:
            if self.profile.trace_prompts:
                return self._trace_idx < len(self.profile.trace_prompts)
            elif self.profile.trace_timestamps:
                return self._trace_idx < len(self.profile.trace_timestamps)
            return False
        return True  # Non-trace modes run forever until stop_event
    
    def next_interarrival(self) -> float:
        if self.profile.arrival_process == ArrivalProcess.POISSON:
            return self.rng.expovariate(self.profile.target_rps)
        
        elif self.profile.arrival_process == ArrivalProcess.HAWKES:
            now = time.time() - self.start_time
            time_since_last = now - self.hawkes_last_arrival
            
            self.hawkes_intensity = (self.profile.hawkes_base_rate + 
                                    (self.hawkes_intensity - self.profile.hawkes_base_rate) * 
                                    (0.5 ** (time_since_last / self.profile.hawkes_decay)))
            
            wait = self.rng.expovariate(max(0.1, self.hawkes_intensity))
            self.hawkes_intensity += self.profile.hawkes_excitation
            self.hawkes_last_arrival = now + wait
            
            return wait
        
        elif self.profile.arrival_process == ArrivalProcess.CYCLICAL:
            now = time.time() - self.start_time
            import math
            rate = (self.profile.cyclical_base + 
                   self.profile.cyclical_amplitude * 
                   math.sin(2 * 3.14159 * now / self.profile.cyclical_period))
            rate = max(0.1, rate)
            return self.rng.expovariate(rate)
        
        elif self.profile.arrival_process == ArrivalProcess.TRACE:
            # Check if we have full prompts or just timestamps
            if self.profile.trace_prompts:
                if self._trace_idx >= len(self.profile.trace_prompts):
                    return float('inf')  # Signal end of trace
                next_time = self.profile.trace_prompts[self._trace_idx]["timestamp"]
            elif self.profile.trace_timestamps:
                if self._trace_idx >= len(self.profile.trace_timestamps):
                    return float('inf')  # Signal end of trace
                next_time = self.profile.trace_timestamps[self._trace_idx]
            else:
                return float('inf')
            
            self._trace_idx += 1
            
            now = time.time() - self._trace_offset
            wait = max(0, next_time - now)
            return wait
        
        return 1.0 / self.profile.target_rps
    
    def get_next_prompt(self) -> Optional[Dict]:
        """Get the next pre-stored prompt for trace replay.
        
        Must be called AFTER next_interarrival() since that increments the index.
        """
        if self.profile.arrival_process == ArrivalProcess.TRACE and self.profile.trace_prompts:
            idx = self._trace_idx - 1  # Already incremented in next_interarrival
            if 0 <= idx < len(self.profile.trace_prompts):
                return self.profile.trace_prompts[idx]
        return None


class LoadHarness:
    """Main load generation harness."""
    
    def __init__(self, vllm_url: str = "http://localhost:8000",
                 use_meaningful_prompts: bool = False,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 prompt_cache_file: str = "prompt_cache.json",
                 temperature: float = 0.0,
                 seed: int = 42):
        self.vllm_url = vllm_url
        self.content_gen = SyntheticContentGenerator(
            use_llm=use_meaningful_prompts,
            api_key=api_key,
            api_base=api_base,
            cache_file=prompt_cache_file
        )
        self.use_meaningful = use_meaningful_prompts
        self.temperature = temperature
        self.seed = seed
        self.session = None
    
    async def detect_model_context_length(self) -> Optional[int]:
        """Query vLLM for the model's max context length."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                # Try /v1/models endpoint first
                async with session.get(f"{self.vllm_url}/v1/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "data" in data and len(data["data"]) > 0:
                            model_info = data["data"][0]
                            # Check for max_model_len in various possible locations
                            max_len = None
                            if "max_model_len" in model_info:
                                max_len = model_info["max_model_len"]
                            elif "root" in model_info and "max_model_len" in model_info["root"]:
                                max_len = model_info["root"]["max_model_len"]
                            elif "sampling_params" in model_info:
                                max_len = model_info["sampling_params"].get("max_model_len")
                            
                            if max_len and isinstance(max_len, (int, float)):
                                return int(max_len)
                
                # Fallback: try /health or /metrics endpoints if available
                async with session.get(f"{self.vllm_url}/health") as resp:
                    if resp.status == 200:
                        # Some versions expose model info in headers or body
                        pass
                        
        except Exception:
            pass
        
        return None
    
    async def __aenter__(self):
        import aiohttp
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            headers={"Content-Type": "application/json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _select_workload(self, profile: LoadProfile, rng: random.Random) -> WorkloadProfile:
        weights = [w.weight for w in profile.workloads]
        total = sum(weights)
        r = rng.uniform(0, total)
        
        cumulative = 0
        for workload in profile.workloads:
            cumulative += workload.weight
            if r <= cumulative:
                return workload
        
        return profile.workloads[-1]
    
    def _build_request(self, workload: WorkloadProfile, profile: LoadProfile, 
                       rng: random.Random, stored_prompt: Optional[Dict] = None) -> Dict:
        # If we have a stored prompt from trace replay, use it directly
        if stored_prompt and stored_prompt.get("prompt"):
            return {
                "prompt": stored_prompt["prompt"],
                "max_tokens": stored_prompt.get("max_tokens", 150),
                "temperature": 0.7,
                "stream": True
            }
        
        # Otherwise, generate a new prompt
        input_tokens = workload.input_dist.sample(rng)
        output_tokens = workload.output_dist.sample(rng)
        
        # Ensure total tokens fit within model's context limit (with safety margin)
        effective_max = profile.model_max_context - profile.context_margin
        total_tokens = input_tokens + output_tokens
        if total_tokens > effective_max:
            # Scale down proportionally to fit
            scale = effective_max / total_tokens
            input_tokens = int(input_tokens * scale)
            output_tokens = int(output_tokens * scale)
            # Ensure minimums
            input_tokens = max(workload.input_dist.min, input_tokens)
            output_tokens = max(workload.output_dist.min, output_tokens)
        
        if workload.workload_type == WorkloadType.RAG:
            context_tokens = int(input_tokens * 0.8)
            question_tokens = input_tokens - context_tokens
            prompt, _ = self.content_gen.build_rag_prompt(context_tokens, question_tokens, rng, use_meaningful=self.use_meaningful)
        
        elif workload.workload_type == WorkloadType.CODE:
            context_tokens = int(input_tokens * 0.7)
            instruction_tokens = input_tokens - context_tokens
            prompt = self.content_gen.build_code_prompt(context_tokens, instruction_tokens, rng, use_meaningful=self.use_meaningful)
        
        else:
            prompt = self.content_gen.build_chat_prompt(0, input_tokens, rng, use_meaningful=self.use_meaningful)
        
            return {
            "prompt": prompt,
            "max_tokens": output_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "stream": True
        }
    
    async def _send_request(self, request_id: str, payload: Dict, 
                           record: RequestRecord) -> None:
        url = f"{self.vllm_url}/v1/completions"
        
        record.timestamp_sent = time.time()
        record.prompt_text = payload["prompt"]
        record.input_tokens = len(payload["prompt"]) // 4
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    record.error = f"HTTP {response.status}"
                    return
                
                first_token = True
                output_text = []
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line or line == "data: [DONE]":
                        continue
                    
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if "choices" in data and len(data["choices"]) > 0:
                                text = data["choices"][0].get("text", "")
                                output_text.append(text)
                                
                                if first_token:
                                    record.timestamp_first_token = time.time()
                                    first_token = False
                        except json.JSONDecodeError:
                            continue
                
                record.timestamp_last_token = time.time()
                record.output_text = "".join(output_text)
                record.output_tokens = len(record.output_text) // 4
                
        except Exception as e:
            record.error = str(e)
    
    async def _worker_closed_loop(self, profile: LoadProfile, rng: random.Random,
                                   results_queue: asyncio.Queue, 
                                   stop_event: asyncio.Event):
        while not stop_event.is_set():
            workload = self._select_workload(profile, rng)
            payload = self._build_request(workload, profile, rng)
            
            record = RequestRecord(
                request_id=str(uuid.uuid4()),
                workload_name=workload.name
            )
            
            await self._send_request(record.request_id, payload, record)
            await results_queue.put(record)
    
    async def _worker_open_loop(self, profile: LoadProfile, rng: random.Random,
                                 arrival_gen: ArrivalTimeGenerator,
                                 results_queue: asyncio.Queue,
                                 stop_event: asyncio.Event):
        while not stop_event.is_set():
            # Check if trace has more entries before waiting
            if profile.arrival_process == ArrivalProcess.TRACE:
                if not arrival_gen.has_more():
                    # End of trace, exit worker
                    break
            
            interarrival = arrival_gen.next_interarrival()
            
            # Check for end of trace (infinite wait)
            if interarrival == float('inf'):
                break
            
            await asyncio.sleep(interarrival)
            
            if stop_event.is_set():
                break
            
            # Check if we have a stored prompt from trace replay
            stored_prompt = arrival_gen.get_next_prompt()
            
            if stored_prompt:
                # Use stored prompt, select workload based on stored type if available
                workload_type = stored_prompt.get("workload_type", "chat")
                workload = self._select_workload(profile, rng)
                payload = self._build_request(workload, profile, rng, stored_prompt=stored_prompt)
            else:
                workload = self._select_workload(profile, rng)
                payload = self._build_request(workload, profile, rng)
            
            record = RequestRecord(
                request_id=str(uuid.uuid4()),
                workload_name=stored_prompt.get("workload_type", workload.name) if stored_prompt else workload.name
            )
            
            asyncio.create_task(self._send_request(record.request_id, payload, record))
            await results_queue.put(record)
    
    async def run(self, profile: LoadProfile) -> ExperimentResult:
        experiment_id = str(uuid.uuid4())[:8]
        rng = random.Random(profile.seed)
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            profile_name=profile.name,
            config={
                "arrival_process": profile.arrival_process.value,
                "target_rps": profile.target_rps,
                "concurrency": profile.concurrency,
                "duration_seconds": profile.duration_seconds,
                "open_loop": profile.open_loop,
                "workloads": [
                    {
                        "name": w.name,
                        "type": w.workload_type.value,
                        "weight": w.weight,
                        "input_mean": w.input_dist.mean,
                        "output_mean": w.output_dist.mean
                    }
                    for w in profile.workloads
                ]
            },
            start_time=datetime.now().isoformat(),
            end_time=""
        )
        
        results_queue = asyncio.Queue()
        stop_event = asyncio.Event()
        
        if profile.open_loop:
            arrival_gen = ArrivalTimeGenerator(profile, rng)
            workers = [
                asyncio.create_task(
                    self._worker_open_loop(profile, rng, arrival_gen, results_queue, stop_event)
                )
            ]
        else:
            workers = [
                asyncio.create_task(
                    self._worker_closed_loop(profile, rng, results_queue, stop_event)
                )
                for _ in range(profile.concurrency)
            ]
        
        async def collect_results():
            while not stop_event.is_set() or not results_queue.empty():
                try:
                    record = await asyncio.wait_for(results_queue.get(), timeout=1.0)
                    result.records.append(record)
                except asyncio.TimeoutError:
                    continue
        
        collector = asyncio.create_task(collect_results())
        
        # For trace replay, wait for workers to finish instead of fixed duration
        if profile.arrival_process == ArrivalProcess.TRACE:
            # Wait for all workers to complete (they exit when trace is exhausted)
            await asyncio.gather(*workers, return_exceptions=True)
            # Give some time for in-flight requests to complete
            await asyncio.sleep(2.0)
            stop_event.set()
        else:
            # Normal mode: run for fixed duration
            await asyncio.sleep(profile.duration_seconds)
            stop_event.set()
            await asyncio.gather(*workers, return_exceptions=True)
        
        await collector
        
        result.end_time = datetime.now().isoformat()
        result.compute_metrics()
        
        return result


# ============================================================================
# PRE-BUILT PROFILES
# ============================================================================

def get_chat_profile(intensity: str = "medium") -> LoadProfile:
    intensities = {"low": 5.0, "medium": 20.0, "high": 50.0}
    
    return LoadProfile(
        name=f"chat_{intensity}",
        arrival_process=ArrivalProcess.POISSON,
        target_rps=intensities.get(intensity, 20.0),
        duration_seconds=60,
        open_loop=True,
        workloads=[
            WorkloadProfile(
                name="chat",
                workload_type=WorkloadType.CHAT,
                input_dist=TokenDistribution(mean=50, std=30, min=10, max=300),
                output_dist=TokenDistribution(mean=150, std=80, min=20, max=800)
            )
        ]
    )


def get_rag_profile(context_size: str = "medium", intensity: str = "medium") -> LoadProfile:
    context_sizes = {
        "small": (500, 200),
        "medium": (2000, 800),
        "large": (6000, 2000),
        "xlarge": (12000, 3000)
    }
    intensities = {"low": 2.0, "medium": 10.0, "high": 30.0}
    
    ctx_mean, ctx_std = context_sizes.get(context_size, (2000, 800))
    
    return LoadProfile(
        name=f"rag_{context_size}_{intensity}",
        arrival_process=ArrivalProcess.POISSON,
        target_rps=intensities.get(intensity, 10.0),
        duration_seconds=120,
        open_loop=True,
        workloads=[
            WorkloadProfile(
                name="rag",
                workload_type=WorkloadType.RAG,
                input_dist=TokenDistribution(mean=ctx_mean + 50, std=ctx_std, 
                                            min=100, max=16000),
                output_dist=TokenDistribution(mean=200, std=100, min=50, max=1000)
            )
        ]
    )


def get_code_profile(intensity: str = "medium") -> LoadProfile:
    intensities = {"low": 3.0, "medium": 15.0, "high": 40.0}
    
    return LoadProfile(
        name=f"code_{intensity}",
        arrival_process=ArrivalProcess.HAWKES,
        hawkes_base_rate=10.0,
        hawkes_excitation=0.8,
        hawkes_decay=2.0,
        duration_seconds=90,
        open_loop=True,
        workloads=[
            WorkloadProfile(
                name="code",
                workload_type=WorkloadType.CODE,
                input_dist=TokenDistribution(mean=1500, std=800, min=200, max=8000),
                output_dist=TokenDistribution(mean=400, std=300, min=50, max=2000)
            )
        ]
    )


def get_mixed_profile() -> LoadProfile:
    return LoadProfile(
        name="mixed",
        arrival_process=ArrivalProcess.POISSON,
        target_rps=25.0,
        duration_seconds=120,
        open_loop=True,
        workloads=[
            WorkloadProfile(
                name="chat",
                workload_type=WorkloadType.CHAT,
                input_dist=TokenDistribution(mean=50, std=30, min=10, max=300),
                output_dist=TokenDistribution(mean=150, std=80, min=20, max=800),
                weight=0.6
            ),
            WorkloadProfile(
                name="rag",
                workload_type=WorkloadType.RAG,
                input_dist=TokenDistribution(mean=2500, std=1000, min=500, max=8000),
                output_dist=TokenDistribution(mean=200, std=100, min=50, max=1000),
                weight=0.3
            ),
            WorkloadProfile(
                name="code",
                workload_type=WorkloadType.CODE,
                input_dist=TokenDistribution(mean=2000, std=1000, min=300, max=8000),
                output_dist=TokenDistribution(mean=500, std=400, min=100, max=3000),
                weight=0.1
            )
        ]
    )


def get_burst_profile() -> LoadProfile:
    return LoadProfile(
        name="burst",
        arrival_process=ArrivalProcess.HAWKES,
        hawkes_base_rate=5.0,
        hawkes_excitation=2.0,
        hawkes_decay=0.5,
        duration_seconds=120,
        open_loop=True,
        workloads=[
            WorkloadProfile(
                name="chat",
                workload_type=WorkloadType.CHAT,
                input_dist=TokenDistribution(mean=80, std=40, min=10, max=500),
                output_dist=TokenDistribution(mean=200, std=150, min=20, max=1500)
            )
        ]
    )


def get_closed_loop_profile(concurrency: int = 10) -> LoadProfile:
    return LoadProfile(
        name=f"closed_loop_c{concurrency}",
        arrival_process=ArrivalProcess.POISSON,
        concurrency=concurrency,
        duration_seconds=60,
        open_loop=False,
        workloads=[
            WorkloadProfile(
                name="chat",
                workload_type=WorkloadType.CHAT,
                input_dist=TokenDistribution(mean=50, std=30, min=10, max=300),
                output_dist=TokenDistribution(mean=150, std=80, min=20, max=800)
            )
        ]
    )


# ============================================================================
# MAIN
# ============================================================================

def save_trace_from_results(records: List[RequestRecord], output_file: str, 
                            profile: LoadProfile) -> None:
    """Save prompts from a run to a trace file for replay."""
    trace_data = []
    
    # Calculate timestamps relative to first request
    if not records:
        print(f"Warning: No records to save to trace")
        return
    
    first_timestamp = min(r.timestamp_sent for r in records if r.timestamp_sent > 0)
    
    for record in records:
        # Use actual timestamp relative to start
        timestamp = record.timestamp_sent - first_timestamp if record.timestamp_sent > 0 else 0
        trace_entry = {
            "timestamp": timestamp,
            "prompt": record.prompt_text,
            "max_tokens": record.output_tokens if record.output_tokens > 0 else 150,
            "workload_type": record.workload_name,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens
        }
        trace_data.append(trace_entry)
    
    # Sort by timestamp to ensure proper ordering
    trace_data.sort(key=lambda x: x["timestamp"])
    
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "source": "varcas_load_harness",
                "total_requests": len(records),
                "profile": profile.name,
                "original_duration_seconds": int(trace_data[-1]["timestamp"]) if trace_data else 60,
                "exported_at": datetime.now().isoformat()
            },
            "prompts": trace_data
        }, f, indent=2)
    
    print(f"Trace saved to {output_file} with {len(records)} prompts")


def profile_from_results(results_file: str) -> LoadProfile:
    """Build profile from a previous results.json file."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    if "records" not in data:
        raise ValueError(f"Invalid results file: {results_file} (no 'records' key)")
    
    # Extract prompts and metadata from records
    prompts = []
    for record in data["records"]:
        prompts.append({
            "timestamp": record.get("timestamp_sent", 0),
            "prompt": record.get("prompt_text", ""),
            "max_tokens": record.get("output_tokens", 150),
            "input_tokens": record.get("input_tokens", 0),
            "output_tokens": record.get("output_tokens", 0),
            "workload_type": record.get("workload_name", "chat")
        })
    
    # Sort by timestamp to preserve order
    prompts.sort(key=lambda x: x["timestamp"])
    
    # Normalize timestamps to start from 0
    if prompts:
        start_time = prompts[0]["timestamp"]
        for p in prompts:
            p["timestamp"] = max(0, p["timestamp"] - start_time)
    
    # Compute duration
    duration = int(prompts[-1]["timestamp"]) + 10 if prompts else 60
    
    # Extract original config if available
    config = data.get("config", {})
    
    return LoadProfile(
        name=f"replay_{data.get('profile_name', 'unknown')}",
        arrival_process=ArrivalProcess.TRACE,
        trace_prompts=prompts,  # Store prompts directly
        duration_seconds=duration,
        open_loop=True,
        workloads=[
            WorkloadProfile(
                name="replay",
                workload_type=WorkloadType.CHAT,
                input_dist=TokenDistribution(mean=100, std=50, min=10, max=8000),
                output_dist=TokenDistribution(mean=200, std=100, min=20, max=4000)
            )
        ]
    )


def profile_from_trace(trace_file: str, name: str = "customer_trace",
                       workload_type: WorkloadType = WorkloadType.CHAT) -> LoadProfile:
    """Build profile from customer trace file (JSON or CSV)."""
    timestamps = []
    input_tokens = []
    output_tokens = []
    prompts = []  # Store full prompts if available
    metadata = {}  # Metadata from saved trace
    
    if trace_file.endswith('.json'):
        with open(trace_file, 'r') as f:
            data = json.load(f)
            
            # Handle our exported trace format with "prompts" array
            if "prompts" in data:
                metadata = data.get("metadata", {})
                for entry in data["prompts"]:
                    timestamps.append(entry.get('timestamp', 0))
                    input_tokens.append(entry.get('input_tokens', 50))
                    output_tokens.append(entry.get('output_tokens', 100))
                    prompts.append({
                        "timestamp": entry.get('timestamp', 0),
                        "prompt": entry.get('prompt', ''),
                        "max_tokens": entry.get('max_tokens', 100),
                        "input_tokens": entry.get('input_tokens', 50),
                        "output_tokens": entry.get('output_tokens', 100),
                        "workload_type": entry.get('workload_type', 'chat')
                    })
            else:
                # Handle simple trace format
                for row in data:
                    timestamps.append(row['timestamp'])
                    input_tokens.append(row.get('input_tokens', 50))
                    output_tokens.append(row.get('output_tokens', 100))
    
    elif trace_file.endswith('.csv'):
        import csv
        with open(trace_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps.append(float(row['timestamp']))
                input_tokens.append(int(row.get('input_tokens', 50)))
                output_tokens.append(int(row.get('output_tokens', 100)))
    
    if timestamps and not prompts:
        start = min(timestamps)
        timestamps = [t - start for t in timestamps]
    
    # If we have full prompts, use them directly
    if prompts:
        # Use metadata for duration if available, otherwise compute from timestamps
        if metadata.get("original_duration_seconds"):
            duration = metadata["original_duration_seconds"]
        else:
            duration = int(max(p["timestamp"] for p in prompts)) + 10 if prompts else 60
        
        # Use profile name from metadata if available
        profile_name = metadata.get("profile", name)
        
        return LoadProfile(
            name=f"replay_{profile_name}",
            arrival_process=ArrivalProcess.TRACE,
            trace_prompts=prompts,
            duration_seconds=duration,
            open_loop=True,
            workloads=[
                WorkloadProfile(
                    name="trace_workload",
                    workload_type=workload_type,
                    input_dist=TokenDistribution(
                        mean=statistics.mean(input_tokens) if input_tokens else 50,
                        std=statistics.stdev(input_tokens) if len(input_tokens) > 1 else 20,
                        min=min(input_tokens) if input_tokens else 10,
                        max=max(input_tokens) if input_tokens else 1000
                    ),
                    output_dist=TokenDistribution(
                        mean=statistics.mean(output_tokens) if output_tokens else 100,
                        std=statistics.stdev(output_tokens) if len(output_tokens) > 1 else 50,
                        min=min(output_tokens) if output_tokens else 20,
                        max=max(output_tokens) if output_tokens else 2000
                    )
                )
            ]
        )
    
    # Original logic for simple traces without full prompts
    input_dist = TokenDistribution(
        mean=statistics.mean(input_tokens) if input_tokens else 50,
        std=statistics.stdev(input_tokens) if len(input_tokens) > 1 else 20,
        min=min(input_tokens) if input_tokens else 10,
        max=max(input_tokens) if input_tokens else 1000
    )
    
    output_dist = TokenDistribution(
        mean=statistics.mean(output_tokens) if output_tokens else 100,
        std=statistics.stdev(output_tokens) if len(output_tokens) > 1 else 50,
        min=min(output_tokens) if output_tokens else 20,
        max=max(output_tokens) if output_tokens else 2000
    )
    
    return LoadProfile(
        name=name,
        arrival_process=ArrivalProcess.TRACE,
        trace_timestamps=timestamps,
        duration_seconds=int(max(timestamps)) + 10 if timestamps else 60,
        open_loop=True,
        workloads=[
            WorkloadProfile(
                name="trace_workload",
                workload_type=workload_type,
                input_dist=input_dist,
                output_dist=output_dist
            )
        ]
    )


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Varcas Load Harness')
    parser.add_argument('--url', default='http://localhost:8000', help='vLLM endpoint')
    parser.add_argument('--profile', default='chat_medium', 
                       choices=['chat_low', 'chat_medium', 'chat_high',
                               'rag_small', 'rag_medium', 'rag_large', 'rag_xlarge',
                               'code_low', 'code_medium', 'code_high',
                               'mixed', 'burst', 'closed_loop'],
                       help='Preset profile')
    parser.add_argument('--duration', type=int, default=None, help='Override duration')
    parser.add_argument('--output', default='result.json', help='Output file')
    parser.add_argument('--trace', default=None, help='Load from trace file or previous results.json')
    parser.add_argument('--max-context', type=int, default=None, 
                       help='Model maximum context length (input + output tokens)')
    parser.add_argument('--context-margin', type=int, default=None,
                       help='Safety margin to reserve for special tokens (default: 50)')
    
    # A/B Testing: Save prompts for replay
    parser.add_argument('--save-trace', default=None, 
                       help='Save prompts to trace file for A/B testing replay')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature (default: 0.0 for deterministic A/B testing)')
    
    # Meaningful prompt generation options
    parser.add_argument('--meaningful-prompts', action='store_true',
                       help='Use pre-written meaningful prompts instead of synthetic gibberish')
    parser.add_argument('--api-key', default=None,
                       help='Kimi/Moonshot API key for LLM-generated prompts (or set KIMI_API_KEY env var)')
    parser.add_argument('--api-base', default=None,
                       help='API base URL (default: https://api.moonshot.cn/v1)')
    parser.add_argument('--prompt-cache', default='prompt_cache.json',
                       help='Cache file for generated prompts (default: prompt_cache.json)')
    
    args = parser.parse_args()
    
    if args.trace:
        # Check if it's a results.json file
        try:
            with open(args.trace, 'r') as f:
                data = json.load(f)
            if "records" in data and "experiment_id" in data:
                print(f"Loading prompts from previous results: {args.trace}")
                profile = profile_from_results(args.trace)
            else:
                profile = profile_from_trace(args.trace)
        except (json.JSONDecodeError, FileNotFoundError):
            profile = profile_from_trace(args.trace)
    elif args.profile.startswith('chat_'):
        intensity = args.profile.split('_')[1]
        profile = get_chat_profile(intensity)
    elif args.profile.startswith('rag_'):
        parts = args.profile.split('_')
        size = parts[1]
        intensity = parts[2] if len(parts) > 2 else 'medium'
        profile = get_rag_profile(size, intensity)
    elif args.profile.startswith('code_'):
        intensity = args.profile.split('_')[1]
        profile = get_code_profile(intensity)
    elif args.profile == 'mixed':
        profile = get_mixed_profile()
    elif args.profile == 'burst':
        profile = get_burst_profile()
    elif args.profile == 'closed_loop':
        profile = get_closed_loop_profile(concurrency=10)
    else:
        profile = get_chat_profile('medium')
    
    if args.duration:
        profile.duration_seconds = args.duration
    
    # Auto-detect model context length from vLLM if not specified
    if args.max_context:
        profile.model_max_context = args.max_context
    else:
        async with LoadHarness(args.url) as harness:
            detected = await harness.detect_model_context_length()
            if detected:
                profile.model_max_context = detected
                print(f"Auto-detected model max context: {detected} tokens")
            else:
                print(f"Could not detect model context, using default: {profile.model_max_context} tokens")
    
    if args.context_margin is not None:
        profile.context_margin = args.context_margin
    
    print(f"Starting load test: {profile.name}")
    print(f"Model max context: {profile.model_max_context} tokens (margin: {profile.context_margin})")
    print(f"Duration: {profile.duration_seconds}s")
    print(f"Mode: {'Open loop' if profile.open_loop else 'Closed loop'}")
    print(f"Temperature: {args.temperature} ({'deterministic' if args.temperature == 0 else 'stochastic'})")
    print(f"Target: {args.url}")
    
    harness = LoadHarness(
        args.url,
        use_meaningful_prompts=args.meaningful_prompts,
        api_key=args.api_key,
        api_base=args.api_base,
        prompt_cache_file=args.prompt_cache,
        temperature=args.temperature,
        seed=42
    )
    async with harness:
        result = await harness.run(profile)
    
    # Save prompt cache if it was used
    if args.meaningful_prompts or args.api_key:
        harness.content_gen.save_cache()
    
    # Save trace for A/B testing replay
    if args.save_trace:
        save_trace_from_results(result.records, args.save_trace, profile)
        print(f"Trace saved to {args.save_trace} for A/B testing replay")
    
    print(f"\nCompleted: {result.total_requests} requests")
    print(f"Success rate: {result.successful_requests / result.total_requests * 100:.1f}%")
    print(f"Throughput: {result.throughput_rps:.2f} req/s, {result.throughput_tok_s:.1f} tok/s")
    print(f"TTFT: p50={result.ttft_p50:.1f}ms, p99={result.ttft_p99:.1f}ms")
    print(f"TPOT: p50={result.tpot_p50:.2f}ms, p99={result.tpot_p99:.2f}ms")
    print(f"Latency: p50={result.latency_p50:.1f}ms, p99={result.latency_p99:.1f}ms")
    
    result.save(args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
