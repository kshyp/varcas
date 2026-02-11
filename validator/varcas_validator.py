"""
Varcas Validation Module v2.0
Post-hoc accuracy and safety validation with degradation metrics.
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable, Union, Tuple
from enum import Enum
import statistics


class ValidationMethod(Enum):
    EXACT_MATCH = "exact_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    ROUGE_L = "rouge_l"
    CODE_EXECUTION = "code_execution"
    LLM_JUDGE = "llm_judge"


@dataclass
class GroundTruthExample:
    """A single validation example."""
    example_id: str
    prompt: str
    reference: str
    validation_type: str = "qa"  # "qa", "code", "summary", "chat"
    metadata: Dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validating one LLM output."""
    request_id: str
    example_id: str
    generated_text: str
    reference_text: str
    
    method: ValidationMethod
    score: float  # 0.0 to 1.0 (or higher for similarity)
    passed: bool
    
    tokens_generated: int = 0
    tokens_reference: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class ValidationMetrics:
    """Core metrics for a validation run."""
    total_validated: int = 0
    mean_score: float = 0.0
    std_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    p10_score: float = 0.0
    p50_score: float = 0.0
    p90_score: float = 0.0
    
    # Pass rate at threshold
    threshold: float = 0.85
    pass_rate: float = 0.0
    passed_count: int = 0
    
    def compute(self, scores: List[float], threshold: float = 0.85):
        if not scores:
            return
        
        self.total_validated = len(scores)
        self.mean_score = statistics.mean(scores)
        self.std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
        self.min_score = min(scores)
        self.max_score = max(scores)
        self.p10_score = self._percentile(scores, 10)
        self.p50_score = statistics.median(scores)
        self.p90_score = self._percentile(scores, 90)
        
        self.threshold = threshold
        self.passed_count = sum(1 for s in scores if s >= threshold)
        self.pass_rate = self.passed_count / len(scores)
    
    def _percentile(self, data: List[float], p: float) -> float:
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100.0
        f = int(k)
        c = min(f + 1, len(sorted_data) - 1)
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


@dataclass
class DegradationReport:
    """
    Clean degradation metrics for customer reporting.
    
    This is the primary output for safety validation decisions.
    """
    # Absolute scores
    baseline_mean: float = 0.0
    candidate_mean: float = 0.0
    
    # Degradation metrics (the key numbers)
    absolute_degradation: float = 0.0  # candidate - baseline (negative = worse)
    relative_degradation_percent: float = 0.0  # (candidate - baseline) / baseline * 100
    
    # Customer-friendly interpretation
    degradation_status: str = "UNKNOWN"  # "NO_DEGRADATION", "ACCEPTABLE", "WARNING", "CRITICAL"
    
    # Thresholds used
    epsilon_absolute: float = 0.02  # 2% absolute similarity drop
    epsilon_relative: float = 5.0   # 5% relative degradation
    
    # Statistical significance
    confidence_interval_95: Tuple[float, float] = (0.0, 0.0)
    statistically_significant: bool = False
    
    # Per-type breakdown
    by_type: Dict[str, Dict] = field(default_factory=dict)
    
    def compute(self, baseline_metrics: ValidationMetrics, 
                candidate_metrics: ValidationMetrics,
                epsilon_abs: float = 0.02,
                epsilon_rel: float = 5.0):
        """Compute degradation from two validation runs."""
        self.baseline_mean = baseline_metrics.mean_score
        self.candidate_mean = candidate_metrics.mean_score
        self.epsilon_absolute = epsilon_abs
        self.epsilon_relative = epsilon_rel
        
        # Core degradation calculation
        self.absolute_degradation = self.candidate_mean - self.baseline_mean
        if self.baseline_mean > 0:
            self.relative_degradation_percent = (
                self.absolute_degradation / self.baseline_mean * 100
            )
        
        # Status classification
        if self.absolute_degradation >= 0:
            self.degradation_status = "NO_DEGRADATION"
        elif abs(self.absolute_degradation) < epsilon_abs:
            self.degradation_status = "ACCEPTABLE"
        elif abs(self.relative_degradation_percent) < epsilon_rel:
            self.degradation_status = "ACCEPTABLE"
        elif abs(self.relative_degradation_percent) < 10.0:
            self.degradation_status = "WARNING"
        else:
            self.degradation_status = "CRITICAL"
        
        # Simple confidence interval (bootstrap would be better)
        if candidate_metrics.std_score > 0:
            n = candidate_metrics.total_validated
            se = candidate_metrics.std_score / (n ** 0.5)
            self.confidence_interval_95 = (
                self.candidate_mean - 1.96 * se,
                self.candidate_mean + 1.96 * se
            )
            # Significant if baseline outside CI
            self.statistically_significant = not (
                self.confidence_interval_95[0] <= self.baseline_mean <= self.confidence_interval_95[1]
            )
    
    def to_customer_summary(self) -> Dict:
        """
        Clean summary for customer reporting.
        """
        return {
            "accuracy_preserved": self.degradation_status in ["NO_DEGRADATION", "ACCEPTABLE"],
            "degradation_status": self.degradation_status,
            "metrics": {
                "baseline_accuracy": round(self.baseline_mean, 4),
                "optimized_accuracy": round(self.candidate_mean, 4),
                "absolute_change": round(self.absolute_degradation, 4),
                "relative_change_percent": round(self.relative_degradation_percent, 2)
            },
            "thresholds": {
                "epsilon_absolute": self.epsilon_absolute,
                "epsilon_relative_percent": self.epsilon_relative
            },
            "interpretation": self._get_interpretation()
        }
    
    def _get_interpretation(self) -> str:
        if self.degradation_status == "NO_DEGRADATION":
            return "Optimization maintained or improved accuracy."
        elif self.degradation_status == "ACCEPTABLE":
            return f"Accuracy change within tolerance ({self.relative_degradation_percent:+.1f}%)."
        elif self.degradation_status == "WARNING":
            return f"Moderate accuracy degradation ({self.relative_degradation_percent:.1f}%). Review recommended."
        else:
            return f"Significant accuracy degradation ({self.relative_degradation_percent:.1f}%). Optimization rejected."
    
    def to_dict(self) -> Dict:
        return {
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "absolute_degradation": self.absolute_degradation,
            "relative_degradation_percent": self.relative_degradation_percent,
            "degradation_status": self.degradation_status,
            "epsilon_absolute": self.epsilon_absolute,
            "epsilon_relative": self.epsilon_relative,
            "statistically_significant": self.statistically_significant,
            "confidence_interval_95": list(self.confidence_interval_95),
            "by_type": self.by_type,
            "customer_summary": self.to_customer_summary()
        }


@dataclass
class ValidationReport:
    """Complete validation report for a single experiment."""
    experiment_id: str
    validation_method: ValidationMethod
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    results: List[ValidationResult] = field(default_factory=list)
    
    def compute_metrics(self, threshold: float = 0.85):
        scores = [r.score for r in self.results if r.error is None]
        self.metrics.compute(scores, threshold)
    
    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "validation_method": self.validation_method.value,
            "metrics": {
                "total_validated": self.metrics.total_validated,
                "mean_score": self.metrics.mean_score,
                "std_score": self.metrics.std_score,
                "min_score": self.metrics.min_score,
                "max_score": self.metrics.max_score,
                "p10_score": self.metrics.p10_score,
                "p50_score": self.metrics.p50_score,
                "p90_score": self.metrics.p90_score,
                "threshold": self.metrics.threshold,
                "pass_rate": self.metrics.pass_rate,
                "passed_count": self.metrics.passed_count
            },
            "results": [asdict(r) for r in self.results]
        }
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def compare_to(self, baseline_report: 'ValidationReport', 
                   epsilon_abs: float = 0.02,
                   epsilon_rel: float = 5.0) -> DegradationReport:
        """Compare this report to a baseline and generate degradation metrics."""
        deg = DegradationReport()
        deg.compute(baseline_report.metrics, self.metrics, epsilon_abs, epsilon_rel)
        
        # Per-type comparison
        self._compute_by_type_breakdown(deg, baseline_report)
        
        return deg
    
    def _compute_by_type_breakdown(self, deg: DegradationReport, 
                                   baseline_report: 'ValidationReport'):
        """Compute degradation per validation type."""
        # Group by type
        def group_by_type(report: ValidationReport) -> Dict[str, List[float]]:
            groups = {}
            for r in report.results:
                vtype = r.example_id.split(":")[0] if ":" in r.example_id else "unknown"
                if vtype not in groups:
                    groups[vtype] = []
                if r.error is None:
                    groups[vtype].append(r.score)
            return groups
        
        baseline_by_type = group_by_type(baseline_report)
        candidate_by_type = group_by_type(self)
        
        for vtype in set(baseline_by_type.keys()) | set(candidate_by_type.keys()):
            base_scores = baseline_by_type.get(vtype, [])
            cand_scores = candidate_by_type.get(vtype, [])
            
            base_mean = statistics.mean(base_scores) if base_scores else 0.0
            cand_mean = statistics.mean(cand_scores) if cand_scores else 0.0
            
            deg.by_type[vtype] = {
                "baseline_mean": base_mean,
                "candidate_mean": cand_mean,
                "absolute_degradation": cand_mean - base_mean,
                "relative_degradation_percent": (
                    (cand_mean - base_mean) / base_mean * 100 if base_mean > 0 else 0.0
                ),
                "sample_count": len(cand_scores)
            }


# ============================================================================
# VALIDATORS
# ============================================================================

class SemanticSimilarityValidator:
    """Validate using embedding-based semantic similarity."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
    
    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("pip install sentence-transformers")
        return self._model
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        model = self._load_model()
        embeddings = model.encode([text1, text2])
        import numpy as np
        return float(np.dot(embeddings[0], embeddings[1]) / 
                    (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
    
    def validate_batch(self, 
                       experiment_results: List[Dict],
                       ground_truth: List[GroundTruthExample],
                       threshold: float = 0.85) -> ValidationReport:
        report = ValidationReport(
            experiment_id=experiment_results[0].get("experiment_id", "unknown") if experiment_results else "unknown",
            validation_method=ValidationMethod.SEMANTIC_SIMILARITY
        )
        
        gt_by_prompt = {}
        for gt in ground_truth:
            prompt_hash = hashlib.md5(gt.prompt.encode()).hexdigest()[:16]
            gt_by_prompt[prompt_hash] = gt
        
        for result in experiment_results:
            prompt = result.get("prompt_text", "")
            generated = result.get("output_text", "")
            request_id = result.get("request_id", "unknown")
            
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:16]
            gt = gt_by_prompt.get(prompt_hash)
            
            if gt is None:
                continue
            
            try:
                similarity = self.compute_similarity(generated, gt.reference)
                
                val_result = ValidationResult(
                    request_id=request_id,
                    example_id=gt.example_id,
                    generated_text=generated[:500],
                    reference_text=gt.reference[:500],
                    method=ValidationMethod.SEMANTIC_SIMILARITY,
                    score=similarity,
                    passed=similarity >= threshold,
                    tokens_generated=len(generated) // 4,
                    tokens_reference=len(gt.reference) // 4
                )
                report.results.append(val_result)
                
            except Exception as e:
                report.results.append(ValidationResult(
                    request_id=request_id,
                    example_id=gt.example_id,
                    generated_text="",
                    reference_text="",
                    method=ValidationMethod.SEMANTIC_SIMILARITY,
                    score=0.0,
                    passed=False,
                    error=str(e)
                ))
        
        report.compute_metrics(threshold)
        return report


class ExactMatchValidator:
    """For structured outputs, code, or exact answers."""
    
    def validate_batch(self,
                       experiment_results: List[Dict],
                       ground_truth: List[GroundTruthExample],
                       threshold: float = 1.0) -> ValidationReport:
        report = ValidationReport(
            experiment_id=experiment_results[0].get("experiment_id", "unknown") if experiment_results else "unknown",
            validation_method=ValidationMethod.EXACT_MATCH
        )
        
        gt_by_prompt = {gt.prompt: gt for gt in ground_truth}
        
        for result in experiment_results:
            prompt = result.get("prompt_text", "")
            generated = result.get("output_text", "").strip().lower()
            request_id = result.get("request_id", "unknown")
            
            gt = gt_by_prompt.get(prompt)
            if gt is None:
                continue
            
            reference = gt.reference.strip().lower()
            match = generated == reference
            
            val_result = ValidationResult(
                request_id=request_id,
                example_id=gt.example_id,
                generated_text=result.get("output_text", "")[:200],
                reference_text=gt.reference[:200],
                method=ValidationMethod.EXACT_MATCH,
                score=1.0 if match else 0.0,
                passed=match
            )
            report.results.append(val_result)
        
        report.compute_metrics(threshold)
        return report


# ============================================================================
# GROUND TRUTH DATASETS
# ============================================================================

def create_portable_qa_dataset() -> List[GroundTruthExample]:
    return [
        GroundTruthExample("qa:capital_france", "What is the capital of France?", "Paris"),
        GroundTruthExample("qa:water_formula", "What is the chemical formula for water?", "H2O"),
        GroundTruthExample("qa:python_list", "How do you create an empty list in Python?", "[] or list()"),
        GroundTruthExample("qa:earth_shape", "What shape is the Earth?", "Sphere or oblate spheroid"),
        GroundTruthExample("qa:light_speed", "What is the speed of light?", "299,792,458 meters per second"),
    ]


# ============================================================================
# WORKFLOW FUNCTIONS
# ============================================================================

def validate_experiment(experiment_file: str, 
                        ground_truth_file: Optional[str] = None,
                        method: ValidationMethod = ValidationMethod.SEMANTIC_SIMILARITY,
                        threshold: float = 0.85) -> ValidationReport:
    """Validate a single experiment."""
    with open(experiment_file, 'r') as f:
        exp_data = json.load(f)
    
    records = exp_data.get("records", [])
    
    if ground_truth_file:
        with open(ground_truth_file, 'r') as f:
            gt_data = json.load(f)
        ground_truth = [GroundTruthExample(**g) for g in gt_data]
    else:
        ground_truth = create_portable_qa_dataset()
    
    if method == ValidationMethod.SEMANTIC_SIMILARITY:
        validator = SemanticSimilarityValidator()
    elif method == ValidationMethod.EXACT_MATCH:
        validator = ExactMatchValidator()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return validator.validate_batch(records, ground_truth, threshold)


def compare_experiments(baseline_file: str,
                        candidate_file: str,
                        ground_truth_file: Optional[str] = None,
                        method: ValidationMethod = ValidationMethod.SEMANTIC_SIMILARITY,
                        epsilon_abs: float = 0.02,
                        epsilon_rel: float = 5.0) -> DegradationReport:
    """
    Compare two experiments and generate degradation report.
    
    This is the primary API for safety validation.
    """
    # Validate both
    baseline_report = validate_experiment(baseline_file, ground_truth_file, method)
    candidate_report = validate_experiment(candidate_file, ground_truth_file, method)
    
    # Compare
    degradation = candidate_report.compare_to(baseline_report, epsilon_abs, epsilon_rel)
    
    return degradation


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Varcas Validator v2.0')
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Validate single experiment
    validate_parser = subparsers.add_parser('validate', help='Validate single experiment')
    validate_parser.add_argument('--experiment', required=True, help='Experiment JSON')
    validate_parser.add_argument('--ground-truth', help='Ground truth JSON')
    validate_parser.add_argument('--method', default='semantic_similarity')
    validate_parser.add_argument('--output', help='Output file')
    
    # Compare two experiments
    compare_parser = subparsers.add_parser('compare', help='Compare baseline vs candidate')
    compare_parser.add_argument('--baseline', required=True, help='Baseline experiment')
    compare_parser.add_argument('--candidate', required=True, help='Candidate experiment')
    compare_parser.add_argument('--ground-truth', help='Ground truth JSON')
    compare_parser.add_argument('--method', default='semantic_similarity')
    compare_parser.add_argument('--epsilon-abs', type=float, default=0.02, help='Absolute tolerance')
    compare_parser.add_argument('--epsilon-rel', type=float, default=5.0, help='Relative tolerance %')
    compare_parser.add_argument('--output', help='Output file')
    
    args = parser.parse_args()
    
    if args.command == 'validate':
        method = ValidationMethod(args.method)
        report = validate_experiment(args.experiment, args.ground_truth, method)
        
        print(f"Validation: {report.experiment_id}")
        print(f"Mean score: {report.metrics.mean_score:.4f}")
        print(f"Pass rate: {report.metrics.pass_rate*100:.1f}%")
        
        if args.output:
            report.save(args.output)
    
    elif args.command == 'compare':
        method = ValidationMethod(args.method)
        deg = compare_experiments(
            args.baseline, args.candidate, args.ground_truth,
            method, args.epsilon_abs, args.epsilon_rel
        )
        
        # Print clean customer report
        summary = deg.to_customer_summary()
        
        print("\n" + "="*60)
        print("ACCURACY VALIDATION REPORT")
        print("="*60)
        print(f"Status: {summary['degradation_status']}")
        print(f"Accuracy Preserved: {'YES' if summary['accuracy_preserved'] else 'NO'}")
        print()
        print("Metrics:")
        print(f"  Baseline:  {summary['metrics']['baseline_accuracy']:.4f}")
        print(f"  Optimized: {summary['metrics']['optimized_accuracy']:.4f}")
        print(f"  Change:    {summary['metrics']['absolute_change']:+.4f} ({summary['metrics']['relative_change_percent']:+.2f}%)")
        print()
        print("Interpretation:")
        print(f"  {summary['interpretation']}")
        print("="*60)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(deg.to_dict(), f, indent=2)
            print(f"\nFull report saved to: {args.output}")
