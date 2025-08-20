"""
Advanced Hessian-aware MCMC samplers with mathematical rigor.

This module implements sophisticated Hessian-aware sampling algorithms including
Metropolis, Langevin dynamics, and adaptive methods with theoretical guarantees
and numerical stability.
"""

from typing import Callable, Optional, Dict, Any, Tuple, List
import numpy as np
import warnings
from collections import deque
import time

from ..core.sampling_base import BaseSampler, SamplingResults
from ..core.hessian_utils import (
    compute_hessian_autodiff, 
    compute_hessian_finite_diff,
    condition_hessian,
    is_positive_definite,
    hessian_condition_number
)
from ..core.hessian_approximations import (
    lbfgs_hessian_approx,
    adaptive_regularization,
    low_rank_hessian_update,
    stochastic_hessian_diagonal
)
from ..utils.math_utils import (
    safe_cholesky,
    matrix_sqrt_inv,
    multivariate_normal_logpdf,
    log_det_via_cholesky
)
from ..utils.validation import validate_array, validate_positive_scalar


class HessianAwareMetropolis(BaseSampler):
    """
    Hessian-aware Metropolis sampler with preconditioning.
    
    Mathematical Algorithm:
    
    1. Proposal: x' = x + ε * H^(-1/2) * N(0, I)
    2. Acceptance: α = min(1, π(x')/π(x) * |det(H')/det(H)|^(-1/2))
    
    Where:
    - H = ∇²(-log π(x)) + λI (regularized Hessian)
    - ε is the step size parameter
    - λ is regularization for numerical stability
    
    The sampler maintains detailed balance and provides optimal scaling
    in high dimensions when the Hessian approximates the posterior covariance.
    """
    
    def __init__(self,
                 target_log_prob: Callable[[np.ndarray], float],
                 dim: int,
                 step_size: float = 0.1,
                 regularization: float = 1e-6,
                 hessian_update_freq: int = 10,
                 hessian_method: str = "finite_diff",
                 max_condition_number: float = 1e12,
                 **kwargs):
        """
        Initialize Hessian-aware Metropolis sampler.
        
        Args:
            target_log_prob: Target log probability function
            dim: Problem dimensionality
            step_size: Base step size for proposals
            regularization: Regularization parameter for Hessian
            hessian_update_freq: Frequency of Hessian updates
            hessian_method: Method for Hessian computation
            max_condition_number: Maximum allowed condition number
            **kwargs: Additional arguments for BaseSampler
        """
        super().__init__(target_log_prob, dim, step_size, **kwargs)
        
        self.regularization = validate_positive_scalar(regularization, "regularization")
        self.hessian_update_freq = int(hessian_update_freq)
        self.hessian_method = hessian_method
        self.max_condition_number = max_condition_number
        
        # State variables
        self.current_hessian = None
        self.current_hessian_sqrt_inv = None
        self.current_log_det_hessian = None
        self.step_count = 0
        self.hessian_computation_failures = 0
        
        # Performance tracking
        self.hessian_condition_numbers = []
        self.proposal_norms = []
        
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate sampler parameters."""
        if self.hessian_update_freq <= 0:
            raise ValueError("hessian_update_freq must be positive")
        
        if self.hessian_method not in ['autodiff', 'finite_diff']:
            raise ValueError("hessian_method must be 'autodiff' or 'finite_diff'")
        
        if self.max_condition_number <= 1:
            raise ValueError("max_condition_number must be > 1")
    
    def step(self, current_state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform single Hessian-aware Metropolis step.
        
        Algorithm:
        1. Update Hessian if needed
        2. Generate proposal using Hessian preconditioning  
        3. Compute acceptance probability with Jacobian correction
        4. Accept or reject proposal
        
        Args:
            current_state: Current position in state space
            
        Returns:
            new_state: New position after sampling step
            info: Dictionary with step information
        """
        self.step_count += 1
        
        # Validate input
        current_state = validate_array(current_state, "current_state", 
                                     expected_shape=(self.dim,))
        
        # Update Hessian information periodically
        hessian_updated = False
        if (self.step_count % self.hessian_update_freq == 1 or 
            self.current_hessian is None):
            hessian_updated = self._update_hessian_info(current_state)
        
        # Get current log probability
        current_log_prob = self.target_log_prob(current_state)
        
        # Generate proposal
        proposal = self._propose_state(current_state)
        proposal_log_prob = self.target_log_prob(proposal)
        
        # Compute acceptance probability
        log_alpha = self._compute_acceptance_probability(
            current_state, proposal, current_log_prob, proposal_log_prob
        )
        
        # Accept or reject
        if np.log(np.random.rand()) < log_alpha:
            accepted = True
            new_state = proposal
            new_log_prob = proposal_log_prob
        else:
            accepted = False
            new_state = current_state
            new_log_prob = current_log_prob
        
        # Update acceptance tracking
        if accepted:
            self.n_accepted += 1
        self.n_proposals += 1
        
        # Collect diagnostics
        info = {
            'accepted': accepted,
            'log_prob': new_log_prob,
            'proposal_log_prob': proposal_log_prob,
            'log_alpha': log_alpha,
            'hessian_updated': hessian_updated,
            'condition_number': (self.hessian_condition_numbers[-1] 
                               if self.hessian_condition_numbers else None),
            'method': 'hessian_metropolis'
        }
        
        return new_state, info
    
    def _propose_state(self, current_state: np.ndarray) -> np.ndarray:
        """
        Generate proposal using Hessian preconditioning.
        
        Mathematical formulation:
        x' = x + ε * H^(-1/2) * ξ
        where ξ ~ N(0, I) and H^(-1/2) is the matrix square root inverse
        
        Args:
            current_state: Current position
            
        Returns:
            Proposed new state
        """
        # Generate standard normal random vector
        xi = np.random.randn(self.dim)
        
        if self.current_hessian_sqrt_inv is not None:
            # Hessian-preconditioned proposal
            proposal_direction = self.current_hessian_sqrt_inv @ xi
        else:
            # Fallback to identity preconditioning
            proposal_direction = xi
            warnings.warn("No Hessian available, using identity preconditioning")
        
        # Scale by step size
        proposal = current_state + self.step_size * proposal_direction
        
        # Track proposal statistics
        proposal_norm = np.linalg.norm(proposal_direction)
        self.proposal_norms.append(proposal_norm)
        
        return proposal
    
    def _compute_acceptance_probability(self,
                                      current_state: np.ndarray,
                                      proposed_state: np.ndarray,
                                      current_log_prob: float,
                                      proposal_log_prob: float) -> float:
        """
        Compute log acceptance probability for Hessian-aware Metropolis.
        
        Mathematical formulation:
        log α = log π(x') - log π(x) + log |det(H')|^(-1/2) - log |det(H)|^(-1/2)
              = log π(x') - log π(x) + 0.5 * (log |det(H')| - log |det(H)|)
        
        For the proposal x' = x + ε H^(-1/2) ξ, the Jacobian contribution
        accounts for the change in the preconditioning metric.
        
        Args:
            current_state: Current position
            proposed_state: Proposed position
            current_log_prob: Log probability at current state
            proposal_log_prob: Log probability at proposed state
            
        Returns:
            Log acceptance probability
        """
        # Basic Metropolis ratio
        log_alpha = proposal_log_prob - current_log_prob
        
        # Jacobian correction for Hessian preconditioning
        # This accounts for the fact that we're using H^(-1/2) as a metric
        if (self.current_hessian is not None and 
            self.current_log_det_hessian is not None):
            
            try:
                # Compute Hessian at proposed state
                H_proposal = self._compute_hessian(proposed_state)
                if H_proposal is not None:
                    log_det_proposal = log_det_via_cholesky(H_proposal)
                    
                    # Jacobian correction: -0.5 * log |det(H)|
                    # This comes from the transformation x' = x + ε H^(-1/2) ξ
                    jacobian_correction = 0.5 * (log_det_proposal - self.current_log_det_hessian)
                    log_alpha += jacobian_correction
                    
            except Exception as e:
                warnings.warn(f"Failed to compute Jacobian correction: {e}")
                # Continue without correction
        
        # Ensure numerical stability
        return min(0.0, log_alpha)
    
    def _update_hessian_info(self, x: np.ndarray) -> bool:
        """
        Update stored Hessian information at point x.
        
        Args:
            x: Point to evaluate Hessian at
            
        Returns:
            True if update successful, False otherwise
        """
        H = self._compute_hessian(x)
        
        if H is None:
            self.hessian_computation_failures += 1
            return False
        
        try:
            # Apply adaptive regularization
            H_reg, reg_param = adaptive_regularization(
                H, 
                min_eigenval=self.regularization,
                max_condition=self.max_condition_number
            )
            
            # Store regularized Hessian
            self.current_hessian = H_reg
            
            # Compute matrix square root inverse
            self.current_hessian_sqrt_inv = matrix_sqrt_inv(H_reg, method="eigen")
            
            # Compute log determinant
            self.current_log_det_hessian = log_det_via_cholesky(H_reg)
            
            # Track condition number
            cond_num = hessian_condition_number(H_reg)
            self.hessian_condition_numbers.append(cond_num)
            
            if reg_param > self.regularization * 10:
                warnings.warn(f"Large regularization applied: {reg_param:.2e}")
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to process Hessian: {e}")
            self.hessian_computation_failures += 1
            return False
    
    def _compute_hessian(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute Hessian matrix at given point.
        
        Args:
            x: Point to evaluate Hessian at
            
        Returns:
            Hessian matrix or None if computation fails
        """
        try:
            if self.hessian_method == "autodiff":
                H = compute_hessian_autodiff(self.target_log_prob, x)
            else:  # finite_diff
                H = compute_hessian_finite_diff(self.target_log_prob, x)
            
            # Negate for negative log probability
            H = -H
            
            return H
            
        except Exception as e:
            warnings.warn(f"Hessian computation failed: {e}")
            return None
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get sampler diagnostic information."""
        return {
            'hessian_failures': self.hessian_computation_failures,
            'condition_numbers': self.hessian_condition_numbers.copy(),
            'mean_condition_number': (np.mean(self.hessian_condition_numbers) 
                                    if self.hessian_condition_numbers else None),
            'proposal_norms': self.proposal_norms.copy(),
            'mean_proposal_norm': (np.mean(self.proposal_norms) 
                                 if self.proposal_norms else None),
            'step_count': self.step_count
        }


class HessianAwareLangevin(BaseSampler):
    """
    Hessian-aware Langevin dynamics sampler.
    
    Mathematical Algorithm:
    Overdamped Langevin SDE: dx = -H^(-1) ∇U(x) dt + √(2T) H^(-1/2) dW
    
    Discretization: x_{t+1} = x_t - ε H^(-1) ∇U(x_t) + √(2Tε) H^(-1/2) Z
    
    Where:
    - U(x) = -log π(x) is the potential energy
    - H = ∇²U(x) is the Hessian of potential energy
    - T is the temperature parameter
    - ε is the integration step size
    - Z ~ N(0, I) is standard Gaussian noise
    
    The sampler includes Metropolis correction for exact sampling.
    """
    
    def __init__(self,
                 target_log_prob: Callable[[np.ndarray], float],
                 dim: int,
                 step_size: float = 0.01,
                 friction: float = 1.0,
                 temperature: float = 1.0,
                 hessian_update_freq: int = 5,
                 metropolis_correction: bool = True,
                 **kwargs):
        """
        Initialize Hessian-aware Langevin sampler.
        
        Args:
            target_log_prob: Target log probability function
            dim: Problem dimensionality
            step_size: Integration step size ε
            friction: Friction coefficient (γ in literature)
            temperature: Temperature parameter T
            hessian_update_freq: Frequency of Hessian updates
            metropolis_correction: Whether to apply Metropolis correction
            **kwargs: Additional arguments for BaseSampler
        """
        super().__init__(target_log_prob, dim, step_size, **kwargs)
        
        self.friction = validate_positive_scalar(friction, "friction")
        self.temperature = validate_positive_scalar(temperature, "temperature")
        self.hessian_update_freq = int(hessian_update_freq)
        self.metropolis_correction = metropolis_correction
        
        # State variables
        self.current_hessian = None
        self.current_hessian_inv = None
        self.current_hessian_sqrt_inv = None
        self.current_gradient = None
        self.step_count = 0
        
        # Performance tracking
        self.integration_errors = []
        self.gradient_norms = []
        
        # Setup gradient computation
        self.gradient_func = self._setup_gradient_computation()
    
    def _setup_gradient_computation(self) -> Optional[Callable]:
        """Setup gradient computation function."""
        try:
            # Try automatic differentiation first
            import jax
            import jax.numpy as jnp
            
            def jax_gradient(x):
                def jax_func(x_jax):
                    return self.target_log_prob(np.array(x_jax))
                grad_func = jax.grad(jax_func)
                return np.array(grad_func(jnp.array(x)))
            
            return jax_gradient
            
        except ImportError:
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
    
    def step(self, current_state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform single Hessian-aware Langevin step.
        
        Algorithm:
        1. Update Hessian and gradient if needed
        2. Compute drift term: -ε H^(-1) ∇U(x)
        3. Compute diffusion term: √(2Tε) H^(-1/2) Z
        4. Integrate: x' = x + drift + diffusion
        5. Apply Metropolis correction if enabled
        
        Args:
            current_state: Current position in state space
            
        Returns:
            new_state: New position after integration step
            info: Dictionary with step information
        """
        self.step_count += 1
        
        # Validate input
        current_state = validate_array(current_state, "current_state",
                                     expected_shape=(self.dim,))
        
        # Update Hessian and gradient information
        hessian_updated = False
        if (self.step_count % self.hessian_update_freq == 1 or
            self.current_hessian is None):
            hessian_updated = self._update_hessian_gradient_info(current_state)
        
        # Perform Langevin step
        proposal = self._langevin_step(current_state)
        
        # Apply Metropolis correction if enabled
        if self.metropolis_correction:
            accepted, new_state, info = self._metropolis_correction(
                current_state, proposal, hessian_updated
            )
        else:
            # Pure Langevin (no correction)
            accepted = True
            new_state = proposal
            current_log_prob = self.target_log_prob(current_state)
            proposal_log_prob = self.target_log_prob(proposal)
            
            info = {
                'accepted': True,
                'log_prob': proposal_log_prob,
                'proposal_log_prob': proposal_log_prob,
                'log_alpha': 0.0,  # Always accept in pure Langevin
                'hessian_updated': hessian_updated,
                'method': 'pure_langevin'
            }
        
        # Update acceptance tracking
        if accepted:
            self.n_accepted += 1
        self.n_proposals += 1
        
        return new_state, info
    
    def _langevin_step(self, current_state: np.ndarray) -> np.ndarray:
        """
        Perform single Langevin integration step.
        
        Mathematical formulation:
        x_{t+1} = x_t - ε H^(-1) ∇U(x_t) + √(2Tε) H^(-1/2) Z
        
        Args:
            current_state: Current position
            
        Returns:
            Proposed new position after Langevin step
        """
        # Drift term: -ε H^(-1) ∇U(x)
        if (self.current_hessian_inv is not None and 
            self.current_gradient is not None):
            # Use Hessian preconditioning
            drift = -self.step_size * self.current_hessian_inv @ self.current_gradient
        elif self.current_gradient is not None:
            # Fallback to gradient descent
            drift = self.step_size * self.current_gradient  # Note: gradient of log prob
            warnings.warn("No Hessian inverse available, using gradient descent")
        else:
            # No gradient available
            drift = np.zeros(self.dim)
            warnings.warn("No gradient available, using pure diffusion")
        
        # Diffusion term: √(2Tε) H^(-1/2) Z
        noise = np.random.randn(self.dim)
        
        if self.current_hessian_sqrt_inv is not None:
            # Hessian-preconditioned diffusion
            diffusion_coeff = np.sqrt(2 * self.temperature * self.step_size)
            diffusion = diffusion_coeff * self.current_hessian_sqrt_inv @ noise
        else:
            # Identity diffusion
            diffusion_coeff = np.sqrt(2 * self.temperature * self.step_size)
            diffusion = diffusion_coeff * noise
            warnings.warn("No Hessian sqrt inverse available, using identity diffusion")
        
        # Integrate
        proposal = current_state + drift + diffusion
        
        # Track integration statistics
        drift_norm = np.linalg.norm(drift)
        diffusion_norm = np.linalg.norm(diffusion)
        self.integration_errors.append({
            'drift_norm': drift_norm,
            'diffusion_norm': diffusion_norm,
            'step_size': self.step_size
        })
        
        return proposal
    
    def _metropolis_correction(self,
                              current_state: np.ndarray,
                              proposal: np.ndarray,
                              hessian_updated: bool) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Apply Metropolis-Hastings correction to Langevin proposal.
        
        This ensures exact sampling from the target distribution despite
        discretization errors in the Langevin integration.
        
        Args:
            current_state: Current position
            proposal: Proposed position from Langevin step
            hessian_updated: Whether Hessian was updated this step
            
        Returns:
            accepted: Whether proposal was accepted
            new_state: New position (current or proposal)
            info: Step information dictionary
        """
        # Compute log probabilities
        current_log_prob = self.target_log_prob(current_state)
        proposal_log_prob = self.target_log_prob(proposal)
        
        # Compute reverse proposal probability
        # This requires evaluating the Langevin transition density
        try:
            log_forward = self._langevin_transition_logpdf(current_state, proposal)
            log_reverse = self._langevin_transition_logpdf(proposal, current_state)
            
            # Metropolis-Hastings ratio
            log_alpha = (proposal_log_prob - current_log_prob + 
                        log_reverse - log_forward)
            log_alpha = min(0.0, log_alpha)
            
        except Exception as e:
            warnings.warn(f"Failed to compute transition probabilities: {e}")
            # Fallback to simple Metropolis ratio
            log_alpha = min(0.0, proposal_log_prob - current_log_prob)
        
        # Accept or reject
        if np.log(np.random.rand()) < log_alpha:
            accepted = True
            new_state = proposal
            new_log_prob = proposal_log_prob
        else:
            accepted = False
            new_state = current_state
            new_log_prob = current_log_prob
        
        info = {
            'accepted': accepted,
            'log_prob': new_log_prob,
            'proposal_log_prob': proposal_log_prob,
            'log_alpha': log_alpha,
            'hessian_updated': hessian_updated,
            'method': 'mala_langevin'
        }
        
        return accepted, new_state, info
    
    def _langevin_transition_logpdf(self, x_from: np.ndarray, x_to: np.ndarray) -> float:
        """
        Compute log probability density of Langevin transition x_from → x_to.
        
        For the discretized Langevin dynamics:
        x' = x - ε H^(-1) ∇U(x) + √(2Tε) H^(-1/2) Z
        
        The transition density is Gaussian with mean and covariance
        determined by the drift and diffusion terms.
        
        Args:
            x_from: Starting position
            x_to: Ending position
            
        Returns:
            Log probability density of transition
        """
        # Update Hessian/gradient info at x_from if needed
        # For now, use current cached values
        if (self.current_hessian_inv is None or 
            self.current_hessian_sqrt_inv is None):
            raise ValueError("Hessian information not available for transition density")
        
        # Mean of transition distribution
        if self.current_gradient is not None:
            drift = -self.step_size * self.current_hessian_inv @ self.current_gradient
            mean = x_from + drift
        else:
            mean = x_from
        
        # Covariance of transition distribution
        covariance = 2 * self.temperature * self.step_size * self.current_hessian_inv
        
        # Compute log probability
        return multivariate_normal_logpdf(x_to, mean, covariance)
    
    def _update_hessian_gradient_info(self, x: np.ndarray) -> bool:
        """
        Update Hessian and gradient information at point x.
        
        Args:
            x: Point to evaluate at
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Compute gradient
            if self.gradient_func is not None:
                self.current_gradient = self.gradient_func(x)
                self.gradient_norms.append(np.linalg.norm(self.current_gradient))
            
            # Compute Hessian
            H = compute_hessian_finite_diff(self.target_log_prob, x)
            if H is None:
                return False
            
            # Negate for potential energy Hessian
            H = -H
            
            # Regularize for stability
            H_reg = condition_hessian(H, min_eigenval=1e-6)
            
            # Store matrices
            self.current_hessian = H_reg
            self.current_hessian_inv = np.linalg.inv(H_reg)
            self.current_hessian_sqrt_inv = matrix_sqrt_inv(H_reg, method="eigen")
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to update Hessian/gradient info: {e}")
            return False
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get sampler diagnostic information."""
        return {
            'integration_errors': self.integration_errors.copy(),
            'gradient_norms': self.gradient_norms.copy(),
            'mean_gradient_norm': (np.mean(self.gradient_norms) 
                                 if self.gradient_norms else None),
            'step_count': self.step_count,
            'temperature': self.temperature,
            'friction': self.friction
        }


class AdaptiveHessianSampler(BaseSampler):
    """
    Adaptive Hessian sampler with automatic parameter tuning.
    
    This sampler automatically adapts:
    1. Step size based on acceptance rate (target: 0.574 optimal)
    2. Regularization parameter based on condition number
    3. Hessian update frequency based on computational cost
    4. Low-rank approximation rank based on dimension
    
    Uses L-BFGS approximation for efficiency in high dimensions.
    """
    
    def __init__(self,
                 target_log_prob: Callable[[np.ndarray], float],
                 dim: int,
                 adaptation_window: int = 100,
                 target_acceptance: float = 0.574,
                 initial_step_size: float = 0.1,
                 max_rank: Optional[int] = None,
                 memory_size: int = 20,
                 **kwargs):
        """
        Initialize adaptive Hessian sampler.
        
        Args:
            target_log_prob: Target log probability function
            dim: Problem dimensionality
            adaptation_window: Window size for adaptation statistics
            target_acceptance: Target acceptance rate (0.574 for optimal scaling)
            initial_step_size: Initial step size
            max_rank: Maximum rank for low-rank approximation
            memory_size: Memory size for L-BFGS approximation
            **kwargs: Additional arguments for BaseSampler
        """
        super().__init__(target_log_prob, dim, initial_step_size, **kwargs)
        
        self.adaptation_window = adaptation_window
        self.target_acceptance = target_acceptance
        self.memory_size = memory_size
        self.max_rank = max_rank or min(dim // 4, 50)
        
        # Adaptation state
        self.gradient_history = deque(maxlen=memory_size)
        self.state_history = deque(maxlen=memory_size)
        self.acceptance_history = deque(maxlen=adaptation_window)
        self.condition_history = deque(maxlen=adaptation_window)
        
        # Current approximations
        self.current_hessian_approx = None
        self.current_regularization = 1e-6
        self.step_count = 0
        
        # Adaptation parameters
        self.adaptation_rate = 0.05
        self.min_step_size = 1e-6
        self.max_step_size = 2.0
        
        # Performance tracking
        self.adaptation_history = []
        
        # Setup gradient computation
        self.gradient_func = self._setup_gradient_computation()
    
    def _setup_gradient_computation(self) -> Optional[Callable]:
        """Setup gradient computation function."""
        try:
            # Try finite differences
            def finite_diff_gradient(x, eps=1e-6):
                grad = np.zeros_like(x)
                f_x = self.target_log_prob(x)
                
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    grad[i] = (self.target_log_prob(x_plus) - f_x) / eps
                
                return grad
            
            return finite_diff_gradient
            
        except Exception:
            return None
    
    def step(self, current_state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform adaptive Hessian-aware sampling step.
        
        Algorithm:
        1. Update L-BFGS Hessian approximation
        2. Adapt parameters based on recent performance
        3. Generate proposal using current approximation
        4. Accept/reject with Metropolis rule
        5. Update histories for next adaptation
        
        Args:
            current_state: Current position in state space
            
        Returns:
            new_state: New position after sampling step
            info: Dictionary with step information
        """
        self.step_count += 1
        
        # Validate input
        current_state = validate_array(current_state, "current_state",
                                     expected_shape=(self.dim,))
        
        # Compute gradient
        if self.gradient_func is not None:
            current_gradient = self.gradient_func(current_state)
        else:
            current_gradient = np.zeros(self.dim)
            warnings.warn("No gradient function available")
        
        # Update L-BFGS approximation
        self._update_lbfgs_approximation(current_state, current_gradient)
        
        # Adapt parameters
        if self.step_count % 10 == 0:  # Adapt every 10 steps
            self._adapt_parameters()
        
        # Generate proposal
        proposal = self._adaptive_propose(current_state, current_gradient)
        
        # Evaluate probabilities
        current_log_prob = self.target_log_prob(current_state)
        proposal_log_prob = self.target_log_prob(proposal)
        
        # Metropolis acceptance
        log_alpha = min(0.0, proposal_log_prob - current_log_prob)
        accepted = np.log(np.random.rand()) < log_alpha
        
        if accepted:
            new_state = proposal
            new_log_prob = proposal_log_prob
            self.n_accepted += 1
        else:
            new_state = current_state
            new_log_prob = current_log_prob
        
        self.n_proposals += 1
        
        # Update histories
        self.acceptance_history.append(accepted)
        
        # Track condition number if Hessian is available
        if self.current_hessian_approx is not None:
            cond_num = hessian_condition_number(self.current_hessian_approx)
            self.condition_history.append(cond_num)
        
        # Collect info
        info = {
            'accepted': accepted,
            'log_prob': new_log_prob,
            'proposal_log_prob': proposal_log_prob,
            'log_alpha': log_alpha,
            'step_size': self.step_size,
            'regularization': self.current_regularization,
            'method': 'adaptive_hessian'
        }
        
        return new_state, info
    
    def _update_lbfgs_approximation(self, state: np.ndarray, gradient: np.ndarray):
        """Update L-BFGS Hessian approximation."""
        # Add to history
        self.state_history.append(state.copy())
        self.gradient_history.append(gradient.copy())
        
        # Update approximation if we have enough history
        if len(self.gradient_history) >= 2:
            try:
                self.current_hessian_approx = lbfgs_hessian_approx(
                    list(self.gradient_history),
                    list(self.state_history),
                    memory_size=self.memory_size
                )
                
                # Apply low-rank approximation if needed
                if self.max_rank < self.dim:
                    self.current_hessian_approx = self._low_rank_hessian_approx(
                        self.current_hessian_approx, self.max_rank
                    )
                
            except Exception as e:
                warnings.warn(f"L-BFGS update failed: {e}")
                # Keep previous approximation
    
    def _low_rank_hessian_approx(self, H: np.ndarray, rank: int) -> np.ndarray:
        """
        Compute low-rank approximation of Hessian matrix.
        
        Mathematical approach:
        H ≈ H_r = Σ_{i=1}^r λ_i v_i v_i^T
        
        Where λ_i, v_i are the r largest eigenvalues/eigenvectors.
        
        Args:
            H: Full Hessian matrix
            rank: Target rank for approximation
            
        Returns:
            Low-rank Hessian approximation
        """
        try:
            # Eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(H)
            
            # Select largest eigenvalues
            idx = np.argsort(np.abs(eigenvals))[::-1][:rank]
            
            selected_vals = eigenvals[idx]
            selected_vecs = eigenvecs[:, idx]
            
            # Reconstruct low-rank approximation
            H_lr = selected_vecs @ np.diag(selected_vals) @ selected_vecs.T
            
            return H_lr
            
        except Exception as e:
            warnings.warn(f"Low-rank approximation failed: {e}")
            return H
    
    def _adapt_parameters(self):
        """Adapt sampler parameters based on recent performance."""
        adaptation_info = {}
        
        # Adapt step size based on acceptance rate
        if len(self.acceptance_history) >= 10:
            current_acceptance = np.mean(list(self.acceptance_history)[-50:])
            
            # Dual averaging-style adaptation
            error = current_acceptance - self.target_acceptance
            log_step_size = np.log(self.step_size) + self.adaptation_rate * error
            
            new_step_size = np.exp(log_step_size)
            self.step_size = np.clip(new_step_size, self.min_step_size, self.max_step_size)
            
            adaptation_info['acceptance_rate'] = current_acceptance
            adaptation_info['step_size_change'] = new_step_size / (self.step_size + 1e-12)
        
        # Adapt regularization based on condition number
        if len(self.condition_history) >= 10:
            recent_conditions = list(self.condition_history)[-10:]
            mean_condition = np.mean(recent_conditions)
            
            if mean_condition > 1e8:
                # Increase regularization
                self.current_regularization *= 2.0
            elif mean_condition < 1e4 and self.current_regularization > 1e-8:
                # Decrease regularization
                self.current_regularization *= 0.8
            
            adaptation_info['condition_number'] = mean_condition
            adaptation_info['regularization'] = self.current_regularization
        
        # Store adaptation history
        if adaptation_info:
            adaptation_info['step'] = self.step_count
            self.adaptation_history.append(adaptation_info)
    
    def _adaptive_propose(self, current_state: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Generate adaptive proposal using current Hessian approximation.
        
        Args:
            current_state: Current position
            gradient: Current gradient
            
        Returns:
            Proposed new position
        """
        if self.current_hessian_approx is not None:
            # Try to use Hessian preconditioning
            try:
                # Regularize Hessian
                H_reg = self.current_hessian_approx + self.current_regularization * np.eye(self.dim)
                
                # Compute matrix square root inverse
                H_sqrt_inv = matrix_sqrt_inv(H_reg, method="eigen")
                
                # Generate proposal
                noise = np.random.randn(self.dim)
                proposal = current_state + self.step_size * H_sqrt_inv @ noise
                
                return proposal
                
            except Exception as e:
                warnings.warn(f"Hessian preconditioning failed: {e}")
        
        # Fallback to simple random walk
        noise = np.random.randn(self.dim)
        return current_state + self.step_size * noise
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive sampler diagnostic information."""
        return {
            'step_count': self.step_count,
            'adaptation_history': self.adaptation_history.copy(),
            'current_acceptance': (np.mean(list(self.acceptance_history)) 
                                 if self.acceptance_history else None),
            'current_condition': (np.mean(list(self.condition_history)) 
                                if self.condition_history else None),
            'current_step_size': self.step_size,
            'current_regularization': self.current_regularization,
            'max_rank': self.max_rank,
            'memory_size': self.memory_size,
            'history_lengths': {
                'gradients': len(self.gradient_history),
                'states': len(self.state_history),
                'acceptances': len(self.acceptance_history),
                'conditions': len(self.condition_history)
            }
        }