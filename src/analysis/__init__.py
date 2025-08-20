"""
Theoretical analysis module for Hessian-aware sampling.

This module provides theoretical tools for analyzing MCMC sampler performance.
"""

from .theoretical_analysis import *

__all__ = [
    'compute_asymptotic_variance',
    'theoretical_optimal_step_size',
    'estimate_mixing_time',
    'spectral_gap_estimate',
    'analyze_convergence_rate',
    'compute_theoretical_ess'
]