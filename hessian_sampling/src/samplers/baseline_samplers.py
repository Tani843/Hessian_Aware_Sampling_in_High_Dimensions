"""
Baseline MCMC samplers for comparison with Hessian-aware methods.

This module implements standard MCMC algorithms including:
- Standard Metropolis-Hastings with random walk proposals
- Hamiltonian Monte Carlo (HMC) 
- Langevin dynamics
These serve as baseline comparisons for Hessian-aware sampling methods.
"""

import numpy as np
from typing import Callable, Dict, Any, Tuple, Optional
try:
    from ..core.sampling_base import BaseSampler
except ImportError:
    from core.sampling_base import BaseSampler


class StandardMetropolis(BaseSampler):
    """
    Classical random-walk Metropolis-Hastings sampler.
    
    Uses Gaussian proposal distributions centered at current state
    with isotropic covariance scaled by step_size.
    """
    
    def __init__(self, 
                 target_log_prob: Callable[[np.ndarray], float],
                 dim: int,
                 step_size: float = 0.1,
                 **kwargs):
        """
        Initialize Standard Metropolis sampler.
        
        Args:
            target_log_prob: Log probability density function
            dim: Dimension of the problem
            step_size: Standard deviation of proposal distribution
            **kwargs: Additional arguments passed to BaseSampler
        """
        super().__init__(target_log_prob, dim, step_size, **kwargs)
        
    def step(self, current_state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform single Metropolis-Hastings step.
        
        Args:
            current_state: Current position
            
        Returns:
            new_state: Proposed or current state
            info: Step information including acceptance
        """
        # Generate proposal
        proposal = current_state + self.step_size * np.random.randn(self.dim)
        
        # Compute log probabilities
        current_log_prob = self.target_log_prob(current_state)
        proposal_log_prob = self.target_log_prob(proposal)
        
        # Metropolis acceptance probability
        log_alpha = proposal_log_prob - current_log_prob
        alpha = min(1.0, np.exp(log_alpha))
        
        # Accept or reject
        if np.random.rand() < alpha:
            return proposal, {
                'accepted': True,
                'log_prob': proposal_log_prob,
                'acceptance_prob': alpha,
                'proposal': proposal
            }
        else:
            return current_state, {
                'accepted': False,
                'log_prob': current_log_prob,
                'acceptance_prob': alpha,
                'proposal': proposal
            }


class LangevinDynamics(BaseSampler):
    """
    Standard Langevin dynamics sampler (without preconditioning).
    
    Uses gradient information for directed proposals but without
    Hessian-based preconditioning.
    """
    
    def __init__(self,
                 target_log_prob: Callable[[np.ndarray], float],
                 target_log_prob_grad: Callable[[np.ndarray], np.ndarray],
                 dim: int,
                 step_size: float = 0.01,
                 **kwargs):
        """
        Initialize Langevin dynamics sampler.
        
        Args:
            target_log_prob: Log probability density function
            target_log_prob_grad: Gradient of log probability
            dim: Dimension of the problem
            step_size: Step size for Langevin dynamics
            **kwargs: Additional arguments passed to BaseSampler
        """
        super().__init__(target_log_prob, dim, step_size, **kwargs)
        self.target_log_prob_grad = target_log_prob_grad
        
    def step(self, current_state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform single Langevin dynamics step.
        
        Args:
            current_state: Current position
            
        Returns:
            new_state: New position after Langevin step
            info: Step information
        """
        # Compute gradient at current state
        grad = self.target_log_prob_grad(current_state)
        
        # Langevin proposal
        noise = np.random.randn(self.dim)
        proposal = current_state + 0.5 * self.step_size * grad + np.sqrt(self.step_size) * noise
        
        # Compute log probabilities for Metropolis correction
        current_log_prob = self.target_log_prob(current_state)
        proposal_log_prob = self.target_log_prob(proposal)
        
        # Compute proposal densities for Metropolis correction
        proposal_grad = self.target_log_prob_grad(proposal)
        
        # Forward proposal log density: q(y|x)
        forward_mean = current_state + 0.5 * self.step_size * grad
        forward_diff = proposal - forward_mean
        forward_log_q = -0.5 * np.sum(forward_diff**2) / self.step_size
        
        # Backward proposal log density: q(x|y)
        backward_mean = proposal + 0.5 * self.step_size * proposal_grad
        backward_diff = current_state - backward_mean
        backward_log_q = -0.5 * np.sum(backward_diff**2) / self.step_size
        
        # Metropolis correction
        log_alpha = (proposal_log_prob + backward_log_q) - (current_log_prob + forward_log_q)
        alpha = min(1.0, np.exp(log_alpha))
        
        # Accept or reject
        if np.random.rand() < alpha:
            return proposal, {
                'accepted': True,
                'log_prob': proposal_log_prob,
                'acceptance_prob': alpha,
                'proposal': proposal,
                'gradient_norm': np.linalg.norm(grad)
            }
        else:
            return current_state, {
                'accepted': False,
                'log_prob': current_log_prob,
                'acceptance_prob': alpha,
                'proposal': proposal,
                'gradient_norm': np.linalg.norm(grad)
            }


class HamiltonianMonteCarlo(BaseSampler):
    """
    Standard Hamiltonian Monte Carlo without Hessian information.
    
    Uses gradient information for Hamiltonian dynamics but with
    identity mass matrix (no preconditioning).
    """
    
    def __init__(self,
                 target_log_prob: Callable[[np.ndarray], float],
                 target_log_prob_grad: Callable[[np.ndarray], np.ndarray],
                 dim: int,
                 step_size: float = 0.1,
                 n_leapfrog: int = 10,
                 **kwargs):
        """
        Initialize HMC sampler.
        
        Args:
            target_log_prob: Log probability density function
            target_log_prob_grad: Gradient of log probability
            dim: Dimension of the problem
            step_size: Leapfrog step size
            n_leapfrog: Number of leapfrog steps
            **kwargs: Additional arguments passed to BaseSampler
        """
        super().__init__(target_log_prob, dim, step_size, **kwargs)
        self.target_log_prob_grad = target_log_prob_grad
        self.n_leapfrog = n_leapfrog
        
    def step(self, current_state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform single HMC step.
        
        Args:
            current_state: Current position
            
        Returns:
            new_state: New position after HMC step
            info: Step information
        """
        # Sample initial momentum
        initial_momentum = np.random.randn(self.dim)
        
        # Current energy
        current_log_prob = self.target_log_prob(current_state)
        current_kinetic = 0.5 * np.sum(initial_momentum**2)
        current_energy = -current_log_prob + current_kinetic
        
        # Leapfrog integration
        q = current_state.copy()
        p = initial_momentum.copy()
        
        # Half step for momentum
        grad = self.target_log_prob_grad(q)
        p += 0.5 * self.step_size * grad
        
        # Full leapfrog steps
        for _ in range(self.n_leapfrog - 1):
            # Full step for position
            q += self.step_size * p
            
            # Full step for momentum
            grad = self.target_log_prob_grad(q)
            p += self.step_size * grad
        
        # Final full step for position
        q += self.step_size * p
        
        # Final half step for momentum
        grad = self.target_log_prob_grad(q)
        p += 0.5 * self.step_size * grad
        
        # Compute proposed energy
        proposal_log_prob = self.target_log_prob(q)
        proposal_kinetic = 0.5 * np.sum(p**2)
        proposal_energy = -proposal_log_prob + proposal_kinetic
        
        # Metropolis acceptance
        log_alpha = current_energy - proposal_energy
        alpha = min(1.0, np.exp(log_alpha))
        
        # Accept or reject
        if np.random.rand() < alpha:
            return q, {
                'accepted': True,
                'log_prob': proposal_log_prob,
                'acceptance_prob': alpha,
                'proposal': q,
                'energy_change': proposal_energy - current_energy,
                'n_leapfrog': self.n_leapfrog
            }
        else:
            return current_state, {
                'accepted': False,
                'log_prob': current_log_prob,
                'acceptance_prob': alpha,
                'proposal': q,
                'energy_change': proposal_energy - current_energy,
                'n_leapfrog': self.n_leapfrog
            }
    
    def set_n_leapfrog(self, n_leapfrog: int) -> None:
        """Set number of leapfrog steps."""
        if n_leapfrog <= 0:
            raise ValueError("n_leapfrog must be positive")
        self.n_leapfrog = n_leapfrog


class AdaptiveMetropolis(BaseSampler):
    """
    Adaptive Metropolis sampler that learns covariance structure.
    
    Adapts the proposal covariance based on sample history,
    but without using Hessian information.
    """
    
    def __init__(self,
                 target_log_prob: Callable[[np.ndarray], float],
                 dim: int,
                 step_size: float = 0.1,
                 adaptation_start: int = 100,
                 adaptation_rate: float = 0.01,
                 **kwargs):
        """
        Initialize Adaptive Metropolis sampler.
        
        Args:
            target_log_prob: Log probability density function
            dim: Dimension of the problem
            step_size: Scaling factor for proposal covariance
            adaptation_start: Number of samples before starting adaptation
            adaptation_rate: Rate of covariance adaptation
            **kwargs: Additional arguments passed to BaseSampler
        """
        super().__init__(target_log_prob, dim, step_size, **kwargs)
        self.adaptation_start = adaptation_start
        self.adaptation_rate = adaptation_rate
        
        # Adaptation variables
        self.sample_count = 0
        self.running_mean = np.zeros(dim)
        self.running_cov = np.eye(dim)
        self.samples_history = []
        
    def step(self, current_state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform single Adaptive Metropolis step.
        
        Args:
            current_state: Current position
            
        Returns:
            new_state: New position after step
            info: Step information
        """
        # Update sample history for adaptation
        self.samples_history.append(current_state.copy())
        self.sample_count += 1
        
        # Adapt covariance if we have enough samples
        if self.sample_count >= self.adaptation_start:
            self._update_covariance()
        
        # Generate proposal using current covariance
        if self.sample_count < self.adaptation_start:
            # Use identity covariance initially
            proposal = current_state + self.step_size * np.random.randn(self.dim)
        else:
            # Use adapted covariance
            noise = np.random.randn(self.dim)
            try:
                L = np.linalg.cholesky(self.running_cov)
                proposal = current_state + self.step_size * L @ noise
            except np.linalg.LinAlgError:
                # Fallback to identity if covariance is not positive definite
                proposal = current_state + self.step_size * noise
        
        # Metropolis acceptance
        current_log_prob = self.target_log_prob(current_state)
        proposal_log_prob = self.target_log_prob(proposal)
        
        log_alpha = proposal_log_prob - current_log_prob
        alpha = min(1.0, np.exp(log_alpha))
        
        # Accept or reject
        if np.random.rand() < alpha:
            return proposal, {
                'accepted': True,
                'log_prob': proposal_log_prob,
                'acceptance_prob': alpha,
                'proposal': proposal,
                'covariance_adapted': self.sample_count >= self.adaptation_start
            }
        else:
            return current_state, {
                'accepted': False,
                'log_prob': current_log_prob,
                'acceptance_prob': alpha,
                'proposal': proposal,
                'covariance_adapted': self.sample_count >= self.adaptation_start
            }
    
    def _update_covariance(self) -> None:
        """Update running covariance estimate."""
        if len(self.samples_history) < 2:
            return
        
        # Convert to array for easier computation
        samples = np.array(self.samples_history[-min(1000, len(self.samples_history)):])
        
        # Compute sample covariance
        sample_mean = np.mean(samples, axis=0)
        sample_cov = np.cov(samples.T)
        
        # Add small regularization
        regularization = 1e-6 * np.eye(self.dim)
        sample_cov += regularization
        
        # Update running estimates
        if self.sample_count == self.adaptation_start:
            self.running_mean = sample_mean
            self.running_cov = sample_cov
        else:
            # Exponential moving average
            alpha = self.adaptation_rate
            self.running_mean = (1 - alpha) * self.running_mean + alpha * sample_mean
            self.running_cov = (1 - alpha) * self.running_cov + alpha * sample_cov
    
    def reset_adaptation(self) -> None:
        """Reset adaptation state."""
        self.sample_count = 0
        self.running_mean = np.zeros(self.dim)
        self.running_cov = np.eye(self.dim)
        self.samples_history.clear()