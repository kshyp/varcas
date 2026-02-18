"""
ConfigIQ: Hardware Sizing Tool for LLM Inference

A tool to calculate optimal hardware configurations for LLM inference workloads
based on static roofline analysis and queuing models.
"""

__version__ = "1.0.0"

from .core import (
    WorkloadType,
    PricingModel,
    SizingInput,
    SizingRecommendation,
    HardwareCatalog,
    WorkloadPatterns,
    get_model_profile,
    ModelProfile,
    RooflineAnalyzer,
    SLACalculator,
)

__all__ = [
    "WorkloadType",
    "PricingModel",
    "SizingInput",
    "SizingRecommendation",
    "HardwareCatalog",
    "WorkloadPatterns",
    "get_model_profile",
    "ModelProfile",
    "RooflineAnalyzer",
    "SLACalculator",
]
