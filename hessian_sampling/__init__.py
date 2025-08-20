"""
Hessian Aware Sampling in High Dimensions

A package for efficient MCMC sampling using Hessian information
to improve convergence in high-dimensional spaces.
"""

__version__ = "0.1.0"
__author__ = "Hessian Sampling Team"

from .src.core.hessian_utils import (
    compute_hessian_autodiff,
    compute_hessian_finite_diff,
    hessian_eigendecomposition,
    condition_hessian
)

from .src.core.sampling_base import BaseSampler
from .src.samplers.hessian_sampler import HessianAwareSampler

__all__ = [
    "compute_hessian_autodiff",
    "compute_hessian_finite_diff", 
    "hessian_eigendecomposition",
    "condition_hessian",
    "BaseSampler",
    "HessianAwareSampler"
]