"""
Comprehensive performance metrics for MCMC samplers.

This module provides advanced statistical diagnostics including:
- Effective sample size calculations
- Autocorrelation analysis
- Multivariate diagnostics
- Acceptance rate analysis
- Convergence assessment
"""

import numpy as np
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union
from scipy import stats
from scipy.special import logsumexp
from scipy.fft import fft, ifft
import pandas as pd


def effective_sample_size(samples: np.ndarray, 
                         method: str = 'fft',
                         c: float = 5.0) -> float:
    """
    Compute effective sample size for MCMC samples.
    
    Args:
        samples: MCMC samples, shape (n_samples, n_dims) or (n_samples,)
        method: Method to use ('fft', 'direct', 'batch')
        c: Cutoff parameter for autocorrelation
        
    Returns:
        Effective sample size
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    
    n_samples, n_dims = samples.shape
    
    if n_samples < 10:
        return float(n_samples)
    
    if method == 'fft':
        return _ess_fft(samples, c)
    elif method == 'direct':
        return _ess_direct(samples, c)
    elif method == 'batch':
        return _ess_batch(samples)
    else:
        raise ValueError(f"Unknown method: {method}")


def _ess_fft(samples: np.ndarray, c: float = 5.0) -> float:
    """FFT-based ESS calculation."""
    n_samples, n_dims = samples.shape
    ess_values = []
    
    for d in range(min(n_dims, 5)):  # Use up to 5 dimensions
        x = samples[:, d] - np.mean(samples[:, d])
        
        if np.std(x) == 0:
            continue
            
        # FFT-based autocorrelation
        n = len(x)
        f = fft(x, n=2*n)
        acorr = ifft(f * np.conj(f))[:n].real
        acorr = acorr / acorr[0] if acorr[0] > 0 else np.zeros_like(acorr)
        
        # Automatic windowing
        tau_int = _integrated_autocorr_time(acorr, c)
        W = min(len(acorr), int(c * tau_int))
        
        if W >= 1:
            sum_acorr = 1 + 2 * np.sum(acorr[1:W])
            ess = n / max(1.0, sum_acorr)
            ess_values.append(ess)
    
    return np.mean(ess_values) if ess_values else float(n_samples)


def _ess_direct(samples: np.ndarray, c: float = 5.0) -> float:
    """Direct autocorrelation ESS calculation."""
    n_samples, n_dims = samples.shape
    
    # Use first dimension
    x = samples[:, 0] - np.mean(samples[:, 0])
    
    if np.std(x) == 0:
        return float(n_samples)
    
    # Direct autocorrelation calculation
    autocorr = []
    n = len(x)
    
    max_lag = min(n // 4, 200)  # Limit computation
    
    for lag in range(max_lag):
        if lag == 0:
            autocorr.append(1.0)
        else:
            corr = np.corrcoef(x[:-lag], x[lag:])[0, 1]
            if np.isnan(corr):
                break
            autocorr.append(corr)
    
    autocorr = np.array(autocorr)
    
    # Find cutoff
    tau_int = _integrated_autocorr_time(autocorr, c)
    W = min(len(autocorr), int(c * tau_int))
    
    sum_acorr = 1 + 2 * np.sum(autocorr[1:W])
    ess = n / max(1.0, sum_acorr)
    
    return ess


def _ess_batch(samples: np.ndarray, batch_size: int = None) -> float:
    """Batch-based ESS estimation."""
    n_samples = len(samples)
    
    if batch_size is None:
        batch_size = max(10, int(np.sqrt(n_samples)))
    
    # Use first dimension
    x = samples[:, 0] if samples.ndim > 1 else samples
    
    n_batches = n_samples // batch_size
    if n_batches < 2:
        return float(n_samples)
    
    # Compute batch means
    batch_means = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_means.append(np.mean(x[start_idx:end_idx]))
    
    batch_means = np.array(batch_means)
    
    # Compute batch variance and within-batch variance
    overall_mean = np.mean(x[:n_batches * batch_size])
    batch_var = np.var(batch_means) * batch_size
    within_var = np.mean([np.var(x[i*batch_size:(i+1)*batch_size]) 
                          for i in range(n_batches)])
    
    if within_var > 0:
        ess = (n_batches * batch_size * within_var) / max(batch_var, within_var)
        return min(float(n_samples), max(1.0, ess))
    else:
        return float(n_samples)


def _integrated_autocorr_time(autocorr: np.ndarray, c: float = 5.0) -> float:
    """Compute integrated autocorrelation time with automatic windowing."""
    tau_int = 0.5  # Start with 0.5
    
    for W in range(1, len(autocorr)):
        tau_int_new = 1 + 2 * np.sum(autocorr[1:W+1])
        
        # Automatic windowing condition
        if W >= c * tau_int_new:
            return max(0.5, tau_int_new)
        
        tau_int = tau_int_new
    
    return max(0.5, tau_int)


def integrated_autocorr_time(samples: np.ndarray, c: float = 5.0) -> float:
    """
    Compute integrated autocorrelation time.
    
    Args:
        samples: MCMC samples
        c: Cutoff parameter
        
    Returns:
        Integrated autocorrelation time
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    
    x = samples[:, 0] - np.mean(samples[:, 0])
    
    if np.std(x) == 0:
        return 1.0
    
    # FFT autocorrelation
    n = len(x)
    f = fft(x, n=2*n)
    acorr = ifft(f * np.conj(f))[:n].real
    acorr = acorr / acorr[0] if acorr[0] > 0 else np.zeros_like(acorr)
    
    return _integrated_autocorr_time(acorr, c)


def potential_scale_reduction_factor(chains: List[np.ndarray]) -> float:
    """
    Compute R-hat statistic for multiple chains.
    
    Args:
        chains: List of MCMC chains, each shape (n_samples, n_dims)
        
    Returns:
        R-hat statistic
    """
    if len(chains) < 2:
        raise ValueError("Need at least 2 chains for R-hat")
    
    # Convert to consistent format
    chains = [chain.reshape(-1, chain.shape[-1]) if chain.ndim > 1 
              else chain.reshape(-1, 1) for chain in chains]
    
    n_dims = chains[0].shape[1]
    r_hat_values = []
    
    for d in range(min(n_dims, 5)):  # Compute for up to 5 dimensions
        chain_data = [chain[:, d] for chain in chains]
        r_hat = _compute_r_hat_single_dim(chain_data)
        if r_hat is not None:
            r_hat_values.append(r_hat)
    
    return np.max(r_hat_values) if r_hat_values else 1.0


def _compute_r_hat_single_dim(chain_data: List[np.ndarray]) -> Optional[float]:
    """Compute R-hat for single dimension."""
    try:
        m = len(chain_data)  # number of chains
        n = len(chain_data[0])  # length of each chain
        
        # Check all chains same length
        if not all(len(chain) == n for chain in chain_data):
            # Use minimum length
            min_n = min(len(chain) for chain in chain_data)
            chain_data = [chain[:min_n] for chain in chain_data]
            n = min_n
        
        if n < 4:
            return None
        
        # Chain means and variances
        chain_means = np.array([np.mean(chain) for chain in chain_data])
        chain_vars = np.array([np.var(chain, ddof=1) for chain in chain_data])
        
        # Overall mean
        overall_mean = np.mean(chain_means)
        
        # Within-chain variance
        W = np.mean(chain_vars)
        
        # Between-chain variance
        B = n * np.var(chain_means, ddof=1)
        
        # Marginal posterior variance
        if W > 0:
            var_plus = ((n - 1) * W + B) / n
            r_hat = np.sqrt(var_plus / W) if W > 0 else 1.0
            return float(r_hat)
        else:
            return 1.0
            
    except Exception:
        return None


def multivariate_effective_sample_size(samples: np.ndarray, 
                                     method: str = 'det') -> float:
    """
    Compute multivariate effective sample size.
    
    Args:
        samples: MCMC samples, shape (n_samples, n_dims)
        method: Method to use ('det', 'trace', 'min')
        
    Returns:
        Multivariate ESS
    """
    if samples.ndim == 1:
        return effective_sample_size(samples)
    
    n_samples, n_dims = samples.shape
    
    if n_dims == 1:
        return effective_sample_size(samples[:, 0])
    
    if method == 'det':
        return _multivariate_ess_det(samples)
    elif method == 'trace':
        return _multivariate_ess_trace(samples)
    elif method == 'min':
        return _multivariate_ess_min(samples)
    else:
        raise ValueError(f"Unknown method: {method}")


def _multivariate_ess_det(samples: np.ndarray) -> float:
    """Determinant-based multivariate ESS."""
    n_samples, n_dims = samples.shape
    
    # Compute sample covariance
    sample_cov = np.cov(samples.T)
    
    # Compute autocorrelation matrices at different lags
    max_lag = min(n_samples // 4, 50)
    autocov_matrices = []
    
    for lag in range(max_lag):
        if lag == 0:
            autocov_matrices.append(sample_cov)
        else:
            if n_samples - lag < 10:
                break
            cov_lag = np.cov(samples[:-lag].T, samples[lag:].T)[:n_dims, n_dims:]
            autocov_matrices.append(cov_lag)
    
    # Sum autocorrelations
    try:
        sum_autocov = autocov_matrices[0]
        for i in range(1, len(autocov_matrices)):
            sum_autocov += 2 * autocov_matrices[i]
        
        # Compute determinants
        det_sample = np.linalg.det(sample_cov)
        det_sum = np.linalg.det(sum_autocov)
        
        if det_sum > 0 and det_sample > 0:
            ess = n_samples * (det_sample / det_sum) ** (1.0 / n_dims)
            return max(1.0, min(float(n_samples), ess))
        else:
            return _multivariate_ess_min(samples)
            
    except (np.linalg.LinAlgError, OverflowError):
        return _multivariate_ess_min(samples)


def _multivariate_ess_trace(samples: np.ndarray) -> float:
    """Trace-based multivariate ESS."""
    n_samples, n_dims = samples.shape
    
    # Compute ESS for each dimension
    ess_values = []
    for d in range(n_dims):
        ess_d = effective_sample_size(samples[:, d])
        ess_values.append(ess_d)
    
    # Weighted average by variance
    sample_vars = np.var(samples, axis=0)
    total_var = np.sum(sample_vars)
    
    if total_var > 0:
        weights = sample_vars / total_var
        return np.sum(weights * ess_values)
    else:
        return np.mean(ess_values)


def _multivariate_ess_min(samples: np.ndarray) -> float:
    """Minimum ESS across dimensions."""
    n_samples, n_dims = samples.shape
    
    ess_values = []
    for d in range(n_dims):
        ess_d = effective_sample_size(samples[:, d])
        ess_values.append(ess_d)
    
    return np.min(ess_values)


def acceptance_rate_analysis(sampler_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze acceptance rates from sampler statistics.
    
    Args:
        sampler_stats: List of step information dictionaries
        
    Returns:
        Dictionary with acceptance rate analysis
    """
    if not sampler_stats:
        return {}
    
    # Extract acceptance information
    acceptances = []
    acceptance_probs = []
    
    for stat in sampler_stats:
        if 'accepted' in stat:
            acceptances.append(stat['accepted'])
        if 'acceptance_prob' in stat:
            acceptance_probs.append(stat['acceptance_prob'])
    
    if not acceptances:
        return {}
    
    acceptances = np.array(acceptances)
    acceptance_probs = np.array(acceptance_probs) if acceptance_probs else None
    
    analysis = {
        'overall_rate': np.mean(acceptances),
        'n_total': len(acceptances),
        'n_accepted': np.sum(acceptances),
    }
    
    # Running acceptance rate
    window_size = min(100, len(acceptances) // 10)
    if window_size > 0:
        running_rates = []
        for i in range(window_size, len(acceptances), window_size):
            rate = np.mean(acceptances[i-window_size:i])
            running_rates.append(rate)
        analysis['running_rates'] = running_rates
        analysis['rate_stability'] = np.std(running_rates) if len(running_rates) > 1 else 0.0
    
    # Acceptance probability statistics
    if acceptance_probs is not None:
        analysis['mean_accept_prob'] = np.mean(acceptance_probs)
        analysis['std_accept_prob'] = np.std(acceptance_probs)
        analysis['min_accept_prob'] = np.min(acceptance_probs)
        analysis['max_accept_prob'] = np.max(acceptance_probs)
    
    return analysis


def geweke_diagnostic(chain: np.ndarray, 
                     first_fraction: float = 0.1,
                     last_fraction: float = 0.5) -> float:
    """
    Compute Geweke convergence diagnostic.
    
    Tests equality of means from first and last parts of the chain.
    
    Args:
        chain: MCMC chain
        first_fraction: Fraction of chain to use for first segment
        last_fraction: Fraction of chain to use for last segment
        
    Returns:
        Z-score for Geweke test
    """
    n = len(chain)
    
    # Define segments
    first_end = int(first_fraction * n)
    last_start = int((1 - last_fraction) * n)
    
    if first_end >= last_start or first_end < 10 or n - last_start < 10:
        return 0.0
    
    # Extract segments
    first_segment = chain[:first_end]
    last_segment = chain[last_start:]
    
    # Compute means
    mean_first = np.mean(first_segment)
    mean_last = np.mean(last_segment)
    
    # Compute spectral densities at zero frequency (approximate)
    var_first = _spectral_density_zero(first_segment)
    var_last = _spectral_density_zero(last_segment)
    
    # Compute test statistic
    pooled_var = var_first / len(first_segment) + var_last / len(last_segment)
    
    if pooled_var > 0:
        z_score = (mean_first - mean_last) / np.sqrt(pooled_var)
        return float(z_score)
    else:
        return 0.0


def _spectral_density_zero(x: np.ndarray, max_lag: int = None) -> float:
    """Estimate spectral density at zero frequency."""
    x = x - np.mean(x)
    n = len(x)
    
    if max_lag is None:
        max_lag = min(n // 4, 50)
    
    # Compute autocorrelations
    autocorrs = []
    for lag in range(min(max_lag, n)):
        if lag == 0:
            autocorr = np.var(x)
        else:
            autocorr = np.mean(x[:-lag] * x[lag:])
        autocorrs.append(autocorr)
    
    # Spectral density at zero
    spectral_density = autocorrs[0] + 2 * np.sum(autocorrs[1:])
    return max(0.0, spectral_density)


def heidelberger_welch_test(chain: np.ndarray, 
                           alpha: float = 0.05,
                           eps: float = 0.1) -> Dict[str, Any]:
    """
    Heidelberger-Welch convergence test.
    
    Args:
        chain: MCMC chain
        alpha: Significance level
        eps: Tolerance for convergence
        
    Returns:
        Dictionary with test results
    """
    n = len(chain)
    
    if n < 50:
        return {'converged': False, 'reason': 'chain too short'}
    
    # Remove initial portion and test for stationarity
    for start_frac in [0.1, 0.2, 0.3, 0.4, 0.5]:
        start_idx = int(start_frac * n)
        test_chain = chain[start_idx:]
        
        if len(test_chain) < 20:
            continue
        
        # Stationarity test using runs test
        median_val = np.median(test_chain)
        runs = _count_runs(test_chain > median_val)
        
        # Expected number of runs under null hypothesis
        n_test = len(test_chain)
        n_above = np.sum(test_chain > median_val)
        n_below = n_test - n_above
        
        if n_above == 0 or n_below == 0:
            continue
        
        expected_runs = (2 * n_above * n_below) / n_test + 1
        var_runs = (2 * n_above * n_below * (2 * n_above * n_below - n_test)) / (n_test ** 2 * (n_test - 1))
        
        if var_runs > 0:
            z_stat = (runs - expected_runs) / np.sqrt(var_runs)
            p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
            
            if p_value > alpha:  # Chain appears stationary
                # Test if remaining chain length is sufficient
                remaining_chain = chain[start_idx:]
                ess = effective_sample_size(remaining_chain)
                
                if ess >= 200:  # Sufficient effective samples
                    return {
                        'converged': True,
                        'burnin': start_idx,
                        'effective_samples': ess,
                        'stationarity_p': p_value
                    }
    
    return {'converged': False, 'reason': 'failed stationarity test'}


def _count_runs(binary_seq: np.ndarray) -> int:
    """Count number of runs in binary sequence."""
    if len(binary_seq) == 0:
        return 0
    
    runs = 1
    for i in range(1, len(binary_seq)):
        if binary_seq[i] != binary_seq[i-1]:
            runs += 1
    
    return runs


def mcmc_summary_statistics(samples: np.ndarray,
                          chain_names: List[str] = None) -> pd.DataFrame:
    """
    Compute comprehensive summary statistics for MCMC samples.
    
    Args:
        samples: MCMC samples, shape (n_samples, n_dims) or list of chains
        chain_names: Names for each chain/dimension
        
    Returns:
        DataFrame with summary statistics
    """
    if isinstance(samples, list):
        # Multiple chains case
        summaries = []
        for i, chain in enumerate(samples):
            name = chain_names[i] if chain_names else f'Chain_{i}'
            summary = _compute_single_chain_summary(chain, name)
            summaries.append(summary)
        return pd.concat(summaries, ignore_index=True)
    else:
        # Single chain case
        if samples.ndim == 1:
            name = chain_names[0] if chain_names else 'Parameter_0'
            return _compute_single_chain_summary(samples, name)
        else:
            summaries = []
            for d in range(samples.shape[1]):
                name = chain_names[d] if chain_names else f'Parameter_{d}'
                summary = _compute_single_chain_summary(samples[:, d], name)
                summaries.append(summary)
            return pd.concat(summaries, ignore_index=True)


def _compute_single_chain_summary(chain: np.ndarray, name: str) -> pd.DataFrame:
    """Compute summary for single chain."""
    summary = {
        'Parameter': name,
        'Mean': np.mean(chain),
        'Std': np.std(chain),
        'Min': np.min(chain),
        'Max': np.max(chain),
        'Q2.5': np.percentile(chain, 2.5),
        'Q25': np.percentile(chain, 25),
        'Q50': np.percentile(chain, 50),
        'Q75': np.percentile(chain, 75),
        'Q97.5': np.percentile(chain, 97.5),
        'ESS': effective_sample_size(chain),
        'Autocorr_time': integrated_autocorr_time(chain),
        'Geweke_z': geweke_diagnostic(chain)
    }
    
    return pd.DataFrame([summary])