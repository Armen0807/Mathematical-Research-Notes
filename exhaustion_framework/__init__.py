"""
Exhaustion Framework Library
============================

A Python framework for measuring asymptotic convergence using weighted norms
and exhaustion functions.

This library implements the metrics defined in the paper:
"Measuring Asymptotic Convergence: A Unified Framework from Isotropic Infinity to Anisotropic Ends"
(A. Petrosyan, 2025).

It is particularly useful for:
- Quantifying tail risks in heavy-tailed distributions (Pareto, Student-t).
- Restoring convergence rates in the Central Limit Theorem context.
"""

# Metadata
__version__ = "1.0.0"
__author__ = "Armen Petrosyan"
__license__ = "MIT"


try:
    from .metric import weighted_kolmogorov_error
except ImportError:
    pass

__all__ = [
    "weighted_kolmogorov_error",
    "__version__",
    "__author__"
]