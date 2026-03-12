"""Evaluation package."""

from prism.evaluation.age_lasso import benchmark_age_models
from prism.evaluation.disease_risk import benchmark_disease_models

__all__ = ["benchmark_age_models", "benchmark_disease_models"]
