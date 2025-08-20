"""
Abstract base class for MCMC samplers with common functionality.

This module provides the foundation for all sampling algorithms,
including adaptive step sizing, convergence diagnostics, and
standard sampling workflows.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any, Tuple, List
import numpy as np
import warnings
from dataclasses import dataclass
import time


@dataclass
class SamplingResults:
    """Container for sampling results and diagnostics."""
    samples: np.ndarray
    log_probs: np.ndarray
    acceptance_rate: float
    n_samples: int
    effective_sample_size: Optional[float] = None
    r_hat: Optional[float] = None
    autocorr_time: Optional[float] = None
    sampling_time: float = 0.0
    step_sizes: Optional[np.ndarray] = None
    diagnostics: Optional[Dict[str, Any]] = None


class BaseSampler(ABC):
    """
    Abstract base class for MCMC samplers.
    
    Provides common functionality including:
    - Adaptive step size adjustment
    - Acceptance rate tracking
    - Convergence diagnostics
    - Sample collection and storage
    """
    
    def __init__(self, 
                 target_log_prob: Callable[[np.ndarray], float],
                 dim: int,
                 step_size: float = 0.1,
                 target_acceptance: float = 0.574,
                 adapt_step_size: bool = True,
                 max_step_size: float = 1.0,
                 min_step_size: float = 1e-6):
        """
        Initialize base sampler.
        
        Args:
            target_log_prob: Function computing log probability density
            dim: Dimensionality of the problem
            step_size: Initial step size
            target_acceptance: Target acceptance rate for adaptation
            adapt_step_size: Whether to adapt step size during sampling
            max_step_size: Maximum allowed step size
            min_step_size: Minimum allowed step size
        """
        self.target_log_prob = target_log_prob
        self.dim = dim
        self.step_size = step_size
        self.target_acceptance = target_acceptance
        self.adapt_step_size = adapt_step_size
        self.max_step_size = max_step_size
        self.min_step_size = min_step_size
        
        # Tracking variables
        self.n_accepted = 0
        self.n_proposals = 0
        self.step_size_history = []
        
        # Validation
        self._validate_initialization()
    
    def _validate_initialization(self) -> None:
        """Validate initialization parameters."""
        if not callable(self.target_log_prob):
            raise TypeError("target_log_prob must be callable")
        
        if not isinstance(self.dim, int) or self.dim <= 0:
            raise ValueError("dim must be a positive integer")
        
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        
        if not 0 < self.target_acceptance < 1:
            raise ValueError("target_acceptance must be between 0 and 1")
        
        if self.max_step_size <= self.min_step_size:
            raise ValueError("max_step_size must be greater than min_step_size")
    
    @abstractmethod
    def step(self, current_state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform single sampling step.
        
        Args:
            current_state: Current position in state space
            
        Returns:
            new_state: New position after sampling step
            info: Dictionary containing step information (acceptance, etc.)
        """
        pass
    
    def sample(self, 
               n_samples: int,
               initial_state: np.ndarray,
               burnin: int = 1000,
               thin: int = 1,
               return_diagnostics: bool = True,
               progress_callback: Optional[Callable[[int, int], None]] = None) -> SamplingResults:
        """
        Generate samples from target distribution.
        
        Args:
            n_samples: Number of samples to generate (after burnin/thinning)
            initial_state: Starting position for sampler
            burnin: Number of burnin samples to discard
            thin: Thinning interval (keep every thin-th sample)
            return_diagnostics: Whether to compute convergence diagnostics
            progress_callback: Optional callback for progress updates
            
        Returns:
            SamplingResults object containing samples and diagnostics
        """
        if not isinstance(initial_state, np.ndarray):
            raise TypeError("initial_state must be numpy array")
        
        if initial_state.shape != (self.dim,):
            raise ValueError(f"initial_state must have shape ({self.dim},)")
        
        if n_samples <= 0 or burnin < 0 or thin <= 0:
            raise ValueError("n_samples must be positive, burnin non-negative, thin positive")
        
        # Total iterations needed
        total_iterations = burnin + n_samples * thin
        
        # Storage
        samples = np.zeros((n_samples, self.dim))
        log_probs = np.zeros(n_samples)
        step_sizes = np.zeros(total_iterations) if self.adapt_step_size else None
        
        # Initialize
        current_state = initial_state.copy()
        current_log_prob = self.target_log_prob(current_state)
        
        self.n_accepted = 0
        self.n_proposals = 0
        
        start_time = time.time()
        
        try:
            # Sampling loop
            sample_idx = 0
            for iteration in range(total_iterations):
                # Perform sampling step
                new_state, step_info = self.step(current_state)
                
                # Update acceptance tracking
                if step_info.get('accepted', False):
                    self.n_accepted += 1
                    current_state = new_state
                    current_log_prob = step_info.get('log_prob', current_log_prob)
                
                self.n_proposals += 1
                
                # Adapt step size during burnin
                if self.adapt_step_size and iteration < burnin:
                    if iteration > 0 and iteration % 50 == 0:  # Adapt every 50 steps
                        self._adapt_step_size()
                
                # Store step size
                if step_sizes is not None:
                    step_sizes[iteration] = self.step_size
                
                # Store sample (after burnin, with thinning)
                if iteration >= burnin and (iteration - burnin) % thin == 0:
                    samples[sample_idx] = current_state
                    log_probs[sample_idx] = current_log_prob
                    sample_idx += 1
                
                # Progress callback
                if progress_callback and iteration % 100 == 0:
                    progress_callback(iteration, total_iterations)
            
            sampling_time = time.time() - start_time
            
            # Compute final acceptance rate
            acceptance_rate = self.n_accepted / self.n_proposals if self.n_proposals > 0 else 0.0
            
            # Create results object
            results = SamplingResults(
                samples=samples,
                log_probs=log_probs,
                acceptance_rate=acceptance_rate,
                n_samples=n_samples,
                sampling_time=sampling_time,
                step_sizes=step_sizes
            )
            
            # Compute diagnostics if requested
            if return_diagnostics:
                results = self._compute_diagnostics(results)
            
            return results
            
        except KeyboardInterrupt:
            warnings.warn("Sampling interrupted by user")
            # Return partial results
            partial_samples = samples[:sample_idx] if sample_idx > 0 else samples[:1]
            partial_log_probs = log_probs[:sample_idx] if sample_idx > 0 else log_probs[:1]
            
            return SamplingResults(
                samples=partial_samples,
                log_probs=partial_log_probs,
                acceptance_rate=self.n_accepted / max(1, self.n_proposals),
                n_samples=len(partial_samples),
                sampling_time=time.time() - start_time
            )
    
    def _adapt_step_size(self, adaptation_rate: float = 0.1) -> None:
        """
        Adapt step size based on current acceptance rate.
        
        Args:
            adaptation_rate: Rate of adaptation (smaller = more conservative)
        """
        if self.n_proposals == 0:
            return
        
        current_acceptance = self.n_accepted / self.n_proposals
        
        # Dual averaging adaptation
        error = current_acceptance - self.target_acceptance
        
        # Update step size
        log_step_size = np.log(self.step_size) + adaptation_rate * error
        new_step_size = np.exp(log_step_size)
        
        # Clip to bounds
        self.step_size = np.clip(new_step_size, self.min_step_size, self.max_step_size)
        
        self.step_size_history.append(self.step_size)
    
    def _compute_diagnostics(self, results: SamplingResults) -> SamplingResults:
        """
        Compute convergence diagnostics for sampling results.
        
        Args:
            results: SamplingResults object to add diagnostics to
            
        Returns:
            Updated SamplingResults with diagnostics
        """
        samples = results.samples
        
        try:
            # Effective sample size (simple autocorrelation-based estimate)
            results.effective_sample_size = self._compute_ess(samples)
            
            # Autocorrelation time
            results.autocorr_time = self._compute_autocorr_time(samples)
            
            # R-hat (requires multiple chains - skip for single chain)
            results.r_hat = None  # TODO: implement for multiple chains
            
            # Additional diagnostics
            diagnostics = {
                'mean': np.mean(samples, axis=0),
                'std': np.std(samples, axis=0),
                'quantiles': {
                    '5%': np.percentile(samples, 5, axis=0),
                    '25%': np.percentile(samples, 25, axis=0),
                    '50%': np.percentile(samples, 50, axis=0),
                    '75%': np.percentile(samples, 75, axis=0),
                    '95%': np.percentile(samples, 95, axis=0)
                }
            }
            
            results.diagnostics = diagnostics
            
        except Exception as e:
            warnings.warn(f"Failed to compute some diagnostics: {e}")
        
        return results
    
    def _compute_ess(self, samples: np.ndarray) -> float:
        """
        Compute effective sample size using autocorrelation.
        
        Args:
            samples: MCMC samples (n_samples × dim)
            
        Returns:
            Estimated effective sample size
        """
        n_samples = len(samples)
        
        if n_samples < 10:
            return float(n_samples)
        
        try:
            # Compute autocorrelation for each dimension
            autocorrs = []
            
            for d in range(self.dim):
                x = samples[:, d]
                x_centered = x - np.mean(x)
                
                # FFT-based autocorrelation
                n = len(x_centered)
                f = np.fft.fft(x_centered, n=2*n)
                acorr = np.fft.ifft(f * np.conj(f))[:n].real
                acorr = acorr / acorr[0]
                
                # Find where autocorrelation drops below threshold
                threshold = 0.05
                cutoff = np.where(acorr < threshold)[0]
                
                if len(cutoff) > 0:
                    tau = cutoff[0]
                else:
                    tau = n // 4  # Conservative fallback
                
                autocorrs.append(tau)
            
            # Average across dimensions
            avg_autocorr = np.mean(autocorrs)
            ess = n_samples / (1 + 2 * avg_autocorr)
            
            return max(1.0, ess)
            
        except Exception:
            return float(n_samples)  # Fallback
    
    def _compute_autocorr_time(self, samples: np.ndarray) -> float:
        """
        Compute integrated autocorrelation time.
        
        Args:
            samples: MCMC samples
            
        Returns:
            Autocorrelation time
        """
        try:
            n_samples = len(samples)
            
            # Use first dimension as representative
            x = samples[:, 0] - np.mean(samples[:, 0])
            
            # Compute autocorrelation function
            n = len(x)
            f = np.fft.fft(x, n=2*n)
            acorr = np.fft.ifft(f * np.conj(f))[:n].real
            acorr = acorr / acorr[0]
            
            # Integrate until autocorrelation becomes small
            cumsum = np.cumsum(acorr)
            # Find where cumulative sum stabilizes
            for i in range(1, len(cumsum)):
                if i >= 6 * cumsum[i]:  # Standard criterion
                    return cumsum[i]
            
            return cumsum[-1]
            
        except Exception:
            return float('nan')
    
    def get_acceptance_rate(self) -> float:
        """Get current acceptance rate."""
        return self.n_accepted / max(1, self.n_proposals)
    
    def reset_stats(self) -> None:
        """Reset acceptance statistics."""
        self.n_accepted = 0
        self.n_proposals = 0
        self.step_size_history.clear()
    
    def set_step_size(self, step_size: float) -> None:
        """Set step size with validation."""
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        
        self.step_size = np.clip(step_size, self.min_step_size, self.max_step_size)
    
    def warmup(self, 
               initial_state: np.ndarray,
               n_warmup: int = 1000,
               progress_callback: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        """
        Warmup phase to adapt step size and find good starting point.
        
        Args:
            initial_state: Starting position
            n_warmup: Number of warmup iterations
            progress_callback: Optional progress callback
            
        Returns:
            Final state after warmup
        """
        current_state = initial_state.copy()
        
        for i in range(n_warmup):
            new_state, step_info = self.step(current_state)
            
            if step_info.get('accepted', False):
                current_state = new_state
                self.n_accepted += 1
            
            self.n_proposals += 1
            
            # Adapt step size
            if self.adapt_step_size and i % 50 == 0:
                self._adapt_step_size()
            
            if progress_callback and i % 100 == 0:
                progress_callback(i, n_warmup)
        
        return current_state