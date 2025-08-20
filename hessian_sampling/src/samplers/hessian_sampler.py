"""
Hessian-aware MCMC sampler implementation.

This module implements the main Hessian-aware sampling algorithm
that uses local curvature information to construct more efficient
proposals for sampling in high-dimensional spaces.
"""

from typing import Callable, Optional, Dict, Any, Tuple
import numpy as np
import warnings
from ..core.sampling_base import BaseSampler
from ..core.hessian_utils import (
    compute_hessian_autodiff, 
    compute_hessian_finite_diff,
    condition_hessian,
    is_positive_definite
)
from ..utils.math_utils import (
    safe_cholesky,
    matrix_sqrt_inv,
    multivariate_normal_logpdf
)


class HessianAwareSampler(BaseSampler):
    """
    Hessian-aware MCMC sampler using local curvature information.
    
    Implements Langevin dynamics with Hessian preconditioning:
    dx = -H^(-1) ∇U(x) dt + H^(-1/2) dW
    
    Where H is the Hessian of the negative log probability.
    """
    
    def __init__(self,
                 target_log_prob: Callable[[np.ndarray], float],
                 dim: int,
                 step_size: float = 0.1,
                 hessian_method: str = "autodiff",
                 hessian_regularization: float = 1e-6,
                 hessian_update_freq: int = 10,
                 use_preconditioning: bool = True,
                 fallback_to_mala: bool = True,
                 **kwargs):
        """
        Initialize Hessian-aware sampler.
        
        Args:
            target_log_prob: Target log probability function
            dim: Problem dimensionality
            step_size: Base step size
            hessian_method: Method for Hessian computation ('autodiff', 'finite_diff')
            hessian_regularization: Regularization for ill-conditioned Hessians
            hessian_update_freq: Frequency of Hessian updates (every N steps)
            use_preconditioning: Whether to use Hessian preconditioning
            fallback_to_mala: Fall back to MALA if Hessian computation fails
            **kwargs: Additional arguments passed to BaseSampler
        """
        super().__init__(target_log_prob, dim, step_size, **kwargs)
        
        self.hessian_method = hessian_method
        self.hessian_regularization = hessian_regularization
        self.hessian_update_freq = hessian_update_freq
        self.use_preconditioning = use_preconditioning
        self.fallback_to_mala = fallback_to_mala
        
        # State variables
        self.current_hessian = None
        self.current_hessian_inv = None
        self.current_hessian_sqrt_inv = None
        self.step_count = 0
        self.hessian_failures = 0
        
        # Gradient computation (needed for MALA)
        self.gradient_func = self._setup_gradient_computation()
        
        # Validation
        self._validate_hessian_params()
    
    def _validate_hessian_params(self) -> None:
        """Validate Hessian-specific parameters."""
        if self.hessian_method not in ['autodiff', 'finite_diff']:
            raise ValueError("hessian_method must be 'autodiff' or 'finite_diff'")
        
        if self.hessian_regularization <= 0:
            raise ValueError("hessian_regularization must be positive")
        
        if self.hessian_update_freq <= 0:
            raise ValueError("hessian_update_freq must be positive")
    
    def _setup_gradient_computation(self) -> Optional[Callable]:
        """Setup gradient computation for fallback MALA."""
        try:
            # Try to use automatic differentiation for gradients
            if self.hessian_method == "autodiff":
                try:
                    import jax
                    import jax.numpy as jnp
                    
                    def jax_gradient(x):
                        def jax_func(x_jax):
                            return self.target_log_prob(np.array(x_jax))
                        grad_func = jax.grad(jax_func)
                        return np.array(grad_func(jnp.array(x)))
                    
                    return jax_gradient
                except ImportError:
                    pass
            
            # Fallback to finite differences
            def finite_diff_gradient(x, eps=1e-6):
                grad = np.zeros_like(x)
                f_x = self.target_log_prob(x)
                
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    grad[i] = (self.target_log_prob(x_plus) - f_x) / eps
                
                return grad
            
            return finite_diff_gradient
            
        except Exception as e:
            warnings.warn(f"Failed to setup gradient computation: {e}")
            return None
    
    def _compute_hessian(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute Hessian at given point.
        
        Args:
            x: Point to evaluate Hessian at
            
        Returns:
            Hessian matrix or None if computation fails
        """
        try:
            if self.hessian_method == "autodiff":
                H = compute_hessian_autodiff(self.target_log_prob, x)
            else:
                H = compute_hessian_finite_diff(self.target_log_prob, x)
            
            # Since we want Hessian of negative log prob for sampling,
            # we negate the Hessian of log prob
            H = -H
            
            # Condition the Hessian for numerical stability
            H_conditioned = condition_hessian(H, self.hessian_regularization)
            
            return H_conditioned
            
        except Exception as e:
            warnings.warn(f"Hessian computation failed: {e}")
            self.hessian_failures += 1
            return None
    
    def _update_hessian_info(self, x: np.ndarray) -> bool:
        """
        Update stored Hessian information.
        
        Args:
            x: Current position
            
        Returns:
            True if update successful, False otherwise
        """
        H = self._compute_hessian(x)
        
        if H is None:
            return False
        
        try:
            # Store Hessian
            self.current_hessian = H
            
            # Compute inverse and square root inverse if preconditioning is used
            if self.use_preconditioning:
                self.current_hessian_inv = np.linalg.inv(H)
                self.current_hessian_sqrt_inv = matrix_sqrt_inv(H)
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to process Hessian: {e}")
            return False
    
    def step(self, current_state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform single Hessian-aware sampling step.
        
        Args:
            current_state: Current position
            
        Returns:
            new_state: Proposed new state
            info: Step information dictionary
        """
        self.step_count += 1
        
        # Update Hessian information periodically
        hessian_updated = False
        if (self.step_count % self.hessian_update_freq == 1 or 
            self.current_hessian is None):
            hessian_updated = self._update_hessian_info(current_state)
        
        # Decide which algorithm to use
        use_hessian = (self.current_hessian is not None and 
                      self.use_preconditioning and
                      is_positive_definite(self.current_hessian))
        
        if use_hessian:
            return self._hessian_aware_step(current_state, hessian_updated)
        else:
            # Fallback to MALA or random walk
            if self.fallback_to_mala and self.gradient_func is not None:
                return self._mala_step(current_state)
            else:
                return self._random_walk_step(current_state)
    
    def _hessian_aware_step(self, x: np.ndarray, hessian_updated: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform Hessian-aware Langevin step.
        
        Implements: x_new = x - ε H^(-1) ∇U(x) + √(2ε) H^(-1/2) ξ
        where ξ ~ N(0, I)
        """
        # Get current log probability and gradient
        current_log_prob = self.target_log_prob(x)
        
        try:
            # Compute gradient
            if self.gradient_func is not None:
                grad = self.gradient_func(x)
            else:
                # Finite difference fallback
                grad = self._finite_diff_gradient(x)
            
            # Since we want gradient of negative log prob
            grad = -grad
            
            # Compute proposal using Hessian information
            drift = -self.step_size * self.current_hessian_inv @ grad
            
            # Random component with Hessian preconditioning
            noise = np.random.normal(0, 1, self.dim)
            diffusion = np.sqrt(2 * self.step_size) * self.current_hessian_sqrt_inv @ noise
            
            proposal = x + drift + diffusion
            
            # Compute acceptance probability
            proposal_log_prob = self.target_log_prob(proposal)
            
            # For Langevin dynamics, we need to account for the proposal distribution
            log_alpha = self._compute_log_acceptance_prob(
                x, proposal, current_log_prob, proposal_log_prob, grad
            )
            
            # Accept/reject
            if np.log(np.random.rand()) < log_alpha:
                accepted = True
                new_state = proposal
                new_log_prob = proposal_log_prob
            else:
                accepted = False
                new_state = x
                new_log_prob = current_log_prob
            
            return new_state, {
                'accepted': accepted,
                'log_prob': new_log_prob,
                'proposal_log_prob': proposal_log_prob,
                'log_alpha': log_alpha,
                'method': 'hessian_aware',
                'hessian_updated': hessian_updated
            }
            
        except Exception as e:
            warnings.warn(f"Hessian-aware step failed: {e}")
            # Fallback to MALA
            return self._mala_step(x)
    
    def _compute_log_acceptance_prob(self, 
                                   x: np.ndarray, 
                                   proposal: np.ndarray,
                                   x_log_prob: float,
                                   proposal_log_prob: float,
                                   grad_x: np.ndarray) -> float:
        """
        Compute log acceptance probability for Hessian-aware proposal.
        """
        try:
            # Compute reverse proposal probability
            if self.gradient_func is not None:
                grad_proposal = self.gradient_func(proposal)
            else:
                grad_proposal = self._finite_diff_gradient(proposal)
            
            grad_proposal = -grad_proposal
            
            # Forward proposal: proposal ~ N(x + drift_x, 2ε H^(-1))
            drift_x = -self.step_size * self.current_hessian_inv @ grad_x
            mean_forward = x + drift_x
            cov_forward = 2 * self.step_size * self.current_hessian_inv
            
            # Reverse proposal: x ~ N(proposal + drift_proposal, 2ε H^(-1))
            drift_proposal = -self.step_size * self.current_hessian_inv @ grad_proposal
            mean_reverse = proposal + drift_proposal
            cov_reverse = cov_forward  # Same covariance
            
            # Log probabilities of proposals
            log_q_forward = multivariate_normal_logpdf(proposal, mean_forward, cov_forward)
            log_q_reverse = multivariate_normal_logpdf(x, mean_reverse, cov_reverse)
            
            # Metropolis-Hastings acceptance
            log_alpha = (proposal_log_prob - x_log_prob + 
                        log_q_reverse - log_q_forward)
            
            return min(0.0, log_alpha)
            
        except Exception as e:
            warnings.warn(f"Failed to compute acceptance probability: {e}")
            # Simple Metropolis ratio as fallback
            return min(0.0, proposal_log_prob - x_log_prob)
    
    def _mala_step(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform Metropolis-Adjusted Langevin Algorithm step.
        """
        current_log_prob = self.target_log_prob(x)
        
        try:
            # Compute gradient
            if self.gradient_func is not None:
                grad = self.gradient_func(x)
            else:
                grad = self._finite_diff_gradient(x)
            
            # MALA proposal: x_new = x + ε∇log p(x) + √(2ε) ξ
            drift = self.step_size * grad
            noise = np.sqrt(2 * self.step_size) * np.random.normal(0, 1, self.dim)
            proposal = x + drift + noise
            
            proposal_log_prob = self.target_log_prob(proposal)
            
            # Compute acceptance probability (simplified)
            log_alpha = min(0.0, proposal_log_prob - current_log_prob)
            
            # Accept/reject
            if np.log(np.random.rand()) < log_alpha:
                accepted = True
                new_state = proposal
                new_log_prob = proposal_log_prob
            else:
                accepted = False
                new_state = x
                new_log_prob = current_log_prob
            
            return new_state, {
                'accepted': accepted,
                'log_prob': new_log_prob,
                'proposal_log_prob': proposal_log_prob,
                'log_alpha': log_alpha,
                'method': 'mala'
            }
            
        except Exception as e:
            warnings.warn(f"MALA step failed: {e}")
            return self._random_walk_step(x)
    
    def _random_walk_step(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform simple random walk Metropolis step.
        """
        current_log_prob = self.target_log_prob(x)
        
        # Random walk proposal
        proposal = x + self.step_size * np.random.normal(0, 1, self.dim)
        proposal_log_prob = self.target_log_prob(proposal)
        
        # Metropolis acceptance
        log_alpha = min(0.0, proposal_log_prob - current_log_prob)
        
        # Accept/reject
        if np.log(np.random.rand()) < log_alpha:
            accepted = True
            new_state = proposal
            new_log_prob = proposal_log_prob
        else:
            accepted = False
            new_state = x
            new_log_prob = current_log_prob
        
        return new_state, {
            'accepted': accepted,
            'log_prob': new_log_prob,
            'proposal_log_prob': proposal_log_prob,
            'log_alpha': log_alpha,
            'method': 'random_walk'
        }
    
    def _finite_diff_gradient(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Compute gradient using finite differences."""
        grad = np.zeros_like(x)
        f_x = self.target_log_prob(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (self.target_log_prob(x_plus) - f_x) / eps
        
        return grad
    
    def get_hessian_stats(self) -> Dict[str, Any]:
        """Get statistics about Hessian computation."""
        return {
            'hessian_failures': self.hessian_failures,
            'current_hessian_available': self.current_hessian is not None,
            'current_condition_number': (
                np.linalg.cond(self.current_hessian) 
                if self.current_hessian is not None else None
            ),
            'step_count': self.step_count
        }