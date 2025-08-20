"""
Theoretical performance analysis for MCMC samplers.

This module provides theoretical tools for:
- Asymptotic variance analysis
- Optimal step size computation
- Mixing time estimation
- Spectral gap analysis
- Convergence rate theory
"""

import numpy as np
import warnings
from typing import Callable, Optional, Dict, Any, Tuple, Union, List
from scipy import linalg, optimize
from scipy.special import gamma, factorial
from scipy.stats import multivariate_normal
import pandas as pd

try:
    from ..core.sampling_base import BaseSampler
    from ..benchmarks.performance_metrics import effective_sample_size, integrated_autocorr_time
except ImportError:
    from core.sampling_base import BaseSampler
    from benchmarks.performance_metrics import effective_sample_size, integrated_autocorr_time


def compute_asymptotic_variance(sampler: BaseSampler, 
                              target_dist: Any,
                              test_function: Optional[Callable] = None,
                              n_samples: int = 10000) -> float:
    """
    Compute asymptotic variance of MCMC estimator.
    
    The asymptotic variance determines the efficiency of the MCMC estimator
    for a given test function.
    
    Args:
        sampler: MCMC sampler
        target_dist: Target distribution
        test_function: Function to compute variance for (default: identity)
        n_samples: Number of samples for estimation
        
    Returns:
        Estimated asymptotic variance
    """
    if test_function is None:
        # Use identity function (variance of the chain itself)
        def test_function(x):
            return x[0] if len(x) > 1 else x
    
    try:
        # Generate initial state
        initial_state = np.zeros(sampler.dim)
        if hasattr(target_dist, 'true_mean'):
            initial_state = target_dist.true_mean()
        
        # Generate samples
        results = sampler.sample(
            n_samples=n_samples,
            initial_state=initial_state,
            burnin=1000,
            return_diagnostics=False
        )
        
        # Compute test function values
        test_values = np.array([test_function(sample) for sample in results.samples])
        
        # Estimate asymptotic variance using batch means
        return _estimate_asymptotic_variance_batch_means(test_values)
        
    except Exception as e:
        warnings.warn(f"Failed to compute asymptotic variance: {e}")
        return np.inf


def _estimate_asymptotic_variance_batch_means(values: np.ndarray, 
                                            batch_size: int = None) -> float:
    """Estimate asymptotic variance using batch means method."""
    n = len(values)
    
    if batch_size is None:
        batch_size = max(10, int(np.sqrt(n)))
    
    n_batches = n // batch_size
    if n_batches < 2:
        return np.var(values)
    
    # Compute batch means
    batch_means = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_mean = np.mean(values[start_idx:end_idx])
        batch_means.append(batch_mean)
    
    batch_means = np.array(batch_means)
    
    # Asymptotic variance estimate
    batch_var = np.var(batch_means, ddof=1)
    asymptotic_var = batch_size * batch_var
    
    return asymptotic_var


def theoretical_optimal_step_size(hessian_eigenvalues: np.ndarray,
                                method: str = 'optimal_acceptance') -> float:
    """
    Compute theoretically optimal step size.
    
    Args:
        hessian_eigenvalues: Eigenvalues of Hessian matrix
        method: Method to use ('optimal_acceptance', 'min_variance', 'geometric_mean')
        
    Returns:
        Optimal step size
    """
    eigenvalues = hessian_eigenvalues[hessian_eigenvalues > 1e-12]  # Remove near-zero
    
    if len(eigenvalues) == 0:
        return 1.0
    
    if method == 'optimal_acceptance':
        # Step size for optimal acceptance rate (~57.4% in high dimensions)
        # Based on Roberts & Rosenthal optimal scaling theory
        dimension = len(eigenvalues)
        
        if dimension == 1:
            # 1D case: optimal acceptance ~44%
            step_size = 2.38 / np.sqrt(np.mean(eigenvalues))
        else:
            # High-dimensional case: optimal acceptance ~23.4%
            step_size = 2.38 / np.sqrt(dimension * np.mean(eigenvalues))
        
        return step_size
        
    elif method == 'min_variance':
        # Minimize asymptotic variance (approximate)
        # Use harmonic mean of eigenvalues
        harmonic_mean = len(eigenvalues) / np.sum(1.0 / eigenvalues)
        return 1.0 / np.sqrt(harmonic_mean)
        
    elif method == 'geometric_mean':
        # Use geometric mean of eigenvalues
        geometric_mean = np.exp(np.mean(np.log(eigenvalues)))
        return 1.0 / np.sqrt(geometric_mean)
        
    else:
        raise ValueError(f"Unknown method: {method}")


def estimate_mixing_time(chain: np.ndarray, 
                        epsilon: float = 0.01,
                        method: str = 'autocorrelation') -> int:
    """
    Estimate mixing time of MCMC chain.
    
    Mixing time is the number of steps needed to get within ε of 
    the stationary distribution.
    
    Args:
        chain: MCMC chain
        epsilon: Tolerance for convergence
        method: Method to use ('autocorrelation', 'variation_distance')
        
    Returns:
        Estimated mixing time
    """
    n_samples = len(chain)
    
    if method == 'autocorrelation':
        # Estimate via autocorrelation decay
        autocorr_time = integrated_autocorr_time(chain)
        # Rule of thumb: mixing time ≈ 6-10 × autocorrelation time
        return int(8 * autocorr_time)
        
    elif method == 'variation_distance':
        # Estimate using running variation distance
        return _estimate_mixing_time_variation(chain, epsilon)
        
    else:
        raise ValueError(f"Unknown method: {method}")


def _estimate_mixing_time_variation(chain: np.ndarray, epsilon: float) -> int:
    """Estimate mixing time using total variation distance."""
    n = len(chain)
    window_size = min(500, n // 10)
    
    if window_size < 10:
        return n // 2
    
    # Compute empirical distributions for different segments
    reference_dist = _empirical_distribution(chain[-window_size:])
    
    for t in range(window_size, n - window_size, window_size):
        current_dist = _empirical_distribution(chain[t:t+window_size])
        
        # Total variation distance
        tv_distance = _total_variation_distance(reference_dist, current_dist)
        
        if tv_distance < epsilon:
            return t
    
    return n // 2  # Fallback


def _empirical_distribution(samples: np.ndarray, n_bins: int = 50) -> np.ndarray:
    """Compute empirical distribution from samples."""
    counts, _ = np.histogram(samples, bins=n_bins, density=True)
    return counts / np.sum(counts)


def _total_variation_distance(dist1: np.ndarray, dist2: np.ndarray) -> float:
    """Compute total variation distance between two distributions."""
    return 0.5 * np.sum(np.abs(dist1 - dist2))


def spectral_gap_estimate(transition_matrix: np.ndarray) -> float:
    """
    Estimate spectral gap of transition matrix.
    
    The spectral gap determines the convergence rate of the Markov chain.
    
    Args:
        transition_matrix: Transition matrix of the Markov chain
        
    Returns:
        Estimated spectral gap (1 - second largest eigenvalue)
    """
    try:
        # Compute eigenvalues
        eigenvalues = linalg.eigvals(transition_matrix)
        eigenvalues = np.real(eigenvalues)  # Take real part
        
        # Sort eigenvalues in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Spectral gap is 1 - |λ₂| where λ₂ is second largest eigenvalue
        if len(eigenvalues) > 1:
            spectral_gap = 1.0 - abs(eigenvalues[1])
            return max(0.0, spectral_gap)
        else:
            return 1.0
            
    except Exception:
        warnings.warn("Failed to compute spectral gap")
        return 0.0


def analyze_convergence_rate(chains: List[np.ndarray],
                           target_distribution: Optional[Any] = None) -> Dict[str, Any]:
    """
    Analyze theoretical and empirical convergence rates.
    
    Args:
        chains: List of MCMC chains
        target_distribution: Target distribution (optional)
        
    Returns:
        Dictionary with convergence analysis
    """
    analysis = {}
    
    # Combine chains for analysis
    if len(chains) == 1:
        combined_chain = chains[0]
    else:
        combined_chain = np.concatenate(chains, axis=0)
    
    # Basic statistics
    analysis['chain_length'] = len(combined_chain)
    analysis['n_chains'] = len(chains)
    
    # Autocorrelation analysis
    autocorr_time = integrated_autocorr_time(combined_chain)
    analysis['autocorr_time'] = autocorr_time
    
    # Effective sample size
    ess = effective_sample_size(combined_chain)
    analysis['effective_sample_size'] = ess
    analysis['efficiency'] = ess / len(combined_chain)
    
    # Mixing time estimate
    mixing_time = estimate_mixing_time(combined_chain[:, 0] if combined_chain.ndim > 1 
                                     else combined_chain)
    analysis['mixing_time'] = mixing_time
    
    # Convergence rate
    # Rate = -1/τ where τ is autocorrelation time
    convergence_rate = -1.0 / max(autocorr_time, 1.0)
    analysis['convergence_rate'] = convergence_rate
    
    # Theoretical predictions (if target distribution available)
    if target_distribution is not None and hasattr(target_distribution, 'true_cov'):
        try:
            true_cov = target_distribution.true_cov()
            eigenvals = linalg.eigvals(true_cov)
            condition_number = np.max(eigenvals) / np.min(eigenvals[eigenvals > 1e-12])
            
            analysis['condition_number'] = condition_number
            analysis['theoretical_mixing_time'] = _theoretical_mixing_time(condition_number)
            
        except Exception as e:
            warnings.warn(f"Failed to compute theoretical predictions: {e}")
    
    return analysis


def _theoretical_mixing_time(condition_number: float, dimension: int = None) -> float:
    """Estimate theoretical mixing time from condition number."""
    # For Gaussian targets, mixing time scales with condition number
    # This is a rough approximation
    
    if dimension is not None:
        # Include dimensional scaling
        return condition_number * np.log(dimension)
    else:
        return condition_number


def compute_theoretical_ess(n_samples: int,
                          autocorr_time: float,
                          method: str = 'standard') -> float:
    """
    Compute theoretical effective sample size.
    
    Args:
        n_samples: Number of MCMC samples
        autocorr_time: Integrated autocorrelation time
        method: Method to use ('standard', 'sokal_windowing')
        
    Returns:
        Theoretical ESS
    """
    if method == 'standard':
        # Standard formula: ESS = N / (1 + 2τ)
        return n_samples / (1.0 + 2.0 * autocorr_time)
        
    elif method == 'sokal_windowing':
        # Sokal windowing with bias correction
        # More conservative estimate
        bias_correction = 1.0 + autocorr_time / n_samples
        return (n_samples / (1.0 + 2.0 * autocorr_time)) / bias_correction
        
    else:
        raise ValueError(f"Unknown method: {method}")


def optimal_preconditioning_matrix(hessian: np.ndarray,
                                 method: str = 'cholesky') -> np.ndarray:
    """
    Compute optimal preconditioning matrix from Hessian.
    
    Args:
        hessian: Hessian matrix
        method: Method to use ('cholesky', 'eigendecomp', 'sqrt')
        
    Returns:
        Preconditioning matrix
    """
    # Regularize Hessian for numerical stability
    regularized_hessian = hessian + 1e-6 * np.eye(hessian.shape[0])
    
    try:
        if method == 'cholesky':
            # Cholesky decomposition: P = L where H = L L^T
            L = linalg.cholesky(regularized_hessian, lower=True)
            return L
            
        elif method == 'eigendecomp':
            # Eigendecomposition: P = Q Λ^{1/2} where H = Q Λ Q^T
            eigenvals, eigenvecs = linalg.eigh(regularized_hessian)
            eigenvals = np.maximum(eigenvals, 1e-12)  # Ensure positive
            sqrt_eigenvals = np.sqrt(eigenvals)
            return eigenvecs @ np.diag(sqrt_eigenvals)
            
        elif method == 'sqrt':
            # Matrix square root
            return linalg.sqrtm(regularized_hessian).real
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
    except Exception as e:
        warnings.warn(f"Failed to compute preconditioning matrix: {e}")
        return np.eye(hessian.shape[0])


def analyze_hessian_conditioning(hessian: np.ndarray) -> Dict[str, Any]:
    """
    Analyze conditioning properties of Hessian matrix.
    
    Args:
        hessian: Hessian matrix
        
    Returns:
        Dictionary with conditioning analysis
    """
    analysis = {}
    
    try:
        # Basic properties
        analysis['shape'] = hessian.shape
        analysis['is_symmetric'] = np.allclose(hessian, hessian.T)
        analysis['frobenius_norm'] = linalg.norm(hessian, 'fro')
        
        # Eigenvalue analysis
        eigenvals = linalg.eigvals(hessian)
        eigenvals = eigenvals.real  # Take real part
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove near-zero
        
        if len(eigenvals) > 0:
            analysis['min_eigenvalue'] = np.min(eigenvals)
            analysis['max_eigenvalue'] = np.max(eigenvals)
            analysis['condition_number'] = np.max(eigenvals) / np.min(eigenvals)
            analysis['determinant'] = np.prod(eigenvals)
            analysis['trace'] = np.sum(eigenvals)
            analysis['rank'] = len(eigenvals)
            
            # Eigenvalue distribution
            analysis['eigenvalue_range'] = np.max(eigenvals) - np.min(eigenvals)
            analysis['eigenvalue_std'] = np.std(eigenvals)
            analysis['eigenvalue_skewness'] = _compute_skewness(eigenvals)
            
        # Positive definiteness
        analysis['is_positive_definite'] = np.all(eigenvals > 0)
        analysis['is_positive_semidefinite'] = np.all(eigenvals >= 0)
        
        # Numerical conditioning
        try:
            rcond = 1.0 / linalg.cond(hessian)
            analysis['reciprocal_condition'] = rcond
            analysis['is_well_conditioned'] = rcond > 1e-12
        except:
            analysis['reciprocal_condition'] = 0.0
            analysis['is_well_conditioned'] = False
        
    except Exception as e:
        warnings.warn(f"Failed to analyze Hessian conditioning: {e}")
        analysis['error'] = str(e)
    
    return analysis


def _compute_skewness(data: np.ndarray) -> float:
    """Compute skewness of data."""
    if len(data) < 3:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return 0.0
    
    skewness = np.mean(((data - mean) / std) ** 3)
    return skewness


def dimensional_scaling_theory(dimension: int,
                             condition_number: float = 1.0,
                             sampler_type: str = 'metropolis') -> Dict[str, float]:
    """
    Theoretical predictions for dimensional scaling.
    
    Args:
        dimension: Problem dimension
        condition_number: Condition number of target covariance
        sampler_type: Type of sampler ('metropolis', 'langevin', 'hmc')
        
    Returns:
        Dictionary with scaling predictions
    """
    predictions = {}
    
    if sampler_type == 'metropolis':
        # Random walk Metropolis scaling
        predictions['optimal_step_size'] = 2.38 / np.sqrt(dimension)
        predictions['optimal_acceptance_rate'] = 0.234
        predictions['mixing_time_scaling'] = dimension * condition_number
        predictions['ess_scaling'] = 1.0 / dimension
        
    elif sampler_type == 'langevin':
        # Langevin dynamics scaling
        predictions['optimal_step_size'] = 1.0 / dimension**(1/3)
        predictions['optimal_acceptance_rate'] = 0.574
        predictions['mixing_time_scaling'] = dimension**(2/3) * condition_number
        predictions['ess_scaling'] = dimension**(-2/3)
        
    elif sampler_type == 'hmc':
        # Hamiltonian Monte Carlo scaling
        predictions['optimal_step_size'] = 1.0 / np.sqrt(dimension)
        predictions['optimal_acceptance_rate'] = 0.651
        predictions['mixing_time_scaling'] = np.sqrt(dimension * condition_number)
        predictions['ess_scaling'] = dimension**(-1/2)
        
    elif sampler_type == 'hessian_metropolis':
        # Hessian-aware Metropolis (theoretical)
        predictions['optimal_step_size'] = 2.38  # Dimension-independent
        predictions['optimal_acceptance_rate'] = 0.234
        predictions['mixing_time_scaling'] = condition_number  # No dimensional scaling
        predictions['ess_scaling'] = 1.0  # Dimension-independent
        
    else:
        warnings.warn(f"Unknown sampler type: {sampler_type}")
        return {}
    
    # Add general predictions
    predictions['dimension'] = dimension
    predictions['condition_number'] = condition_number
    predictions['sampler_type'] = sampler_type
    
    return predictions


def compare_theoretical_empirical(theoretical_results: Dict[str, Any],
                                empirical_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Compare theoretical predictions with empirical results.
    
    Args:
        theoretical_results: Dictionary of theoretical predictions
        empirical_results: Dictionary of empirical measurements
        
    Returns:
        DataFrame comparing theoretical vs empirical
    """
    comparison_data = []
    
    common_keys = set(theoretical_results.keys()) & set(empirical_results.keys())
    
    for key in common_keys:
        theo_val = theoretical_results[key]
        emp_val = empirical_results[key]
        
        if isinstance(theo_val, (int, float)) and isinstance(emp_val, (int, float)):
            relative_error = abs(theo_val - emp_val) / max(abs(theo_val), abs(emp_val), 1e-12)
            
            comparison_data.append({
                'Metric': key,
                'Theoretical': theo_val,
                'Empirical': emp_val,
                'Absolute_Error': abs(theo_val - emp_val),
                'Relative_Error': relative_error,
                'Agreement': 'Good' if relative_error < 0.5 else 'Poor'
            })
    
    return pd.DataFrame(comparison_data)


def theoretical_performance_bounds(dimension: int,
                                 condition_number: float,
                                 sampler_type: str = 'metropolis') -> Dict[str, Tuple[float, float]]:
    """
    Compute theoretical performance bounds.
    
    Args:
        dimension: Problem dimension
        condition_number: Condition number
        sampler_type: Type of sampler
        
    Returns:
        Dictionary of (lower_bound, upper_bound) pairs
    """
    bounds = {}
    
    # Mixing time bounds
    if sampler_type == 'metropolis':
        # Lower bound from spectral gap
        mixing_lower = 0.5 * np.log(dimension) * condition_number
        mixing_upper = 10.0 * dimension * condition_number
        bounds['mixing_time'] = (mixing_lower, mixing_upper)
        
    elif sampler_type == 'langevin':
        mixing_lower = 0.5 * dimension**(2/3) * condition_number
        mixing_upper = 5.0 * dimension**(2/3) * condition_number
        bounds['mixing_time'] = (mixing_lower, mixing_upper)
        
    # ESS bounds
    n_samples = 10000  # Reference sample size
    ess_upper = n_samples  # Perfect efficiency
    
    if sampler_type == 'metropolis':
        ess_lower = n_samples / (10.0 * dimension * condition_number)
    else:
        ess_lower = n_samples / (5.0 * dimension * condition_number)
    
    bounds['effective_sample_size'] = (ess_lower, ess_upper)
    
    # Acceptance rate bounds
    if sampler_type == 'metropolis':
        bounds['acceptance_rate'] = (0.1, 0.5)  # Reasonable range
    elif sampler_type == 'langevin':
        bounds['acceptance_rate'] = (0.3, 0.8)
    elif sampler_type == 'hmc':
        bounds['acceptance_rate'] = (0.5, 0.9)
    
    return bounds