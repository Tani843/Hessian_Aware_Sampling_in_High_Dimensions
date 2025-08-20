"""
Efficient Hessian approximation methods for high-dimensional sampling.

This module provides computationally efficient methods for approximating
Hessian matrices when exact computation is prohibitively expensive, including
limited-memory BFGS, stochastic estimation, and block-wise computation.
"""

from typing import List, Callable, Optional, Tuple
import numpy as np
import warnings
from collections import deque
from ..utils.math_utils import safe_cholesky, is_symmetric


def lbfgs_hessian_approx(gradient_history: List[np.ndarray], 
                        state_history: List[np.ndarray],
                        memory_size: int = 10,
                        damping: float = 1e-6) -> np.ndarray:
    """
    Limited-memory BFGS approximation of inverse Hessian.
    
    Constructs B_k ≈ H^(-1) using the L-BFGS two-loop recursion algorithm.
    The approximation uses only the last m gradient and state differences.
    
    Mathematical formulation:
    s_k = x_{k+1} - x_k  (state differences)
    y_k = g_{k+1} - g_k  (gradient differences)
    ρ_k = 1 / (y_k^T s_k)  (curvature condition)
    
    Two-loop recursion:
    H_k^(-1) ≈ (I - ρ_{k-1} s_{k-1} y_{k-1}^T) ... (I - ρ_0 s_0 y_0^T) H_0^(-1) 
              (I - ρ_0 y_0 s_0^T) ... (I - ρ_{k-1} y_{k-1} s_{k-1}^T)
    
    Args:
        gradient_history: List of gradient vectors [g_0, g_1, ..., g_k]
        state_history: List of state vectors [x_0, x_1, ..., x_k]  
        memory_size: Maximum number of vector pairs to store
        damping: Damping factor for initial Hessian approximation
        
    Returns:
        Approximate inverse Hessian matrix
        
    Raises:
        ValueError: If histories have different lengths or insufficient data
    """
    if len(gradient_history) != len(state_history):
        raise ValueError("Gradient and state histories must have same length")
    
    if len(gradient_history) < 2:
        # Return identity scaled by damping for insufficient history
        dim = len(gradient_history[-1]) if gradient_history else 1
        return (1.0 / damping) * np.eye(dim)
    
    # Limit memory usage
    n_history = min(len(gradient_history) - 1, memory_size)
    
    # Extract recent history
    recent_grads = gradient_history[-n_history-1:]
    recent_states = state_history[-n_history-1:]
    
    dim = len(recent_grads[0])
    
    # Compute differences
    s_vectors = []  # State differences
    y_vectors = []  # Gradient differences  
    rho_values = []  # Curvature values
    
    for i in range(n_history):
        s_k = recent_states[i+1] - recent_states[i]
        y_k = recent_grads[i+1] - recent_grads[i]
        
        # Check curvature condition: y_k^T s_k > 0
        sy_dot = np.dot(y_k, s_k)
        
        if sy_dot > 1e-12:  # Positive curvature
            s_vectors.append(s_k)
            y_vectors.append(y_k)
            rho_values.append(1.0 / sy_dot)
        else:
            # Skip this update due to negative curvature
            warnings.warn("Negative curvature detected in L-BFGS, skipping update")
    
    if not s_vectors:
        # No valid updates, return scaled identity
        return (1.0 / damping) * np.eye(dim)
    
    # Initial Hessian approximation H_0^(-1)
    # Use Barzilai-Borwein scaling: γ_k = s_k^T y_k / y_k^T y_k
    if len(y_vectors) > 0:
        s_last = s_vectors[-1]
        y_last = y_vectors[-1]
        gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)
        gamma = max(gamma, 1e-12)  # Ensure positive
        H0_inv = gamma * np.eye(dim)
    else:
        H0_inv = (1.0 / damping) * np.eye(dim)
    
    # L-BFGS two-loop recursion
    def lbfgs_multiply(v: np.ndarray) -> np.ndarray:
        """Apply L-BFGS inverse Hessian approximation to vector v."""
        q = v.copy()
        alphas = []
        
        # First loop (backward)
        for i in reversed(range(len(s_vectors))):
            alpha = rho_values[i] * np.dot(s_vectors[i], q)
            q -= alpha * y_vectors[i]
            alphas.append(alpha)
        
        alphas.reverse()
        
        # Apply initial Hessian
        r = H0_inv @ q
        
        # Second loop (forward)
        for i in range(len(s_vectors)):
            beta = rho_values[i] * np.dot(y_vectors[i], r)
            r += (alphas[i] - beta) * s_vectors[i]
        
        return r
    
    # Construct full matrix (for moderate dimensions)
    if dim <= 1000:  # Memory threshold
        H_inv = np.zeros((dim, dim))
        for i in range(dim):
            e_i = np.zeros(dim)
            e_i[i] = 1.0
            H_inv[:, i] = lbfgs_multiply(e_i)
        
        # Ensure symmetry
        H_inv = 0.5 * (H_inv + H_inv.T)
        
        return H_inv
    else:
        # For high dimensions, return the multiplication operator
        # This should be handled differently in practice
        warnings.warn("High-dimensional L-BFGS approximation, returning scaled identity")
        return (1.0 / damping) * np.eye(dim)


def hutchinson_trace_estimator(hessian_vec_prod: Callable[[np.ndarray], np.ndarray],
                              dim: int,
                              n_samples: int = 10,
                              distribution: str = "rademacher") -> float:
    """
    Hutchinson's stochastic trace estimator for tr(H).
    
    Mathematical formulation:
    tr(H) = E[z^T H z] where z has mean 0 and cov I
    
    For Rademacher distribution: z_i ∈ {-1, +1} with P(z_i = ±1) = 1/2
    For Gaussian distribution: z ~ N(0, I)
    
    The estimator is:
    tr(H) ≈ (1/n) Σ_{i=1}^n z_i^T H z_i
    
    Variance analysis:
    Var[estimator] = (2/n) Σ_{i,j} H_{ij}^2 (for Rademacher)
    
    Args:
        hessian_vec_prod: Function computing H @ v for vector v
        dim: Dimension of the space
        n_samples: Number of random samples for estimation
        distribution: Random distribution ("rademacher" or "gaussian")
        
    Returns:
        Estimated trace of Hessian matrix
        
    Raises:
        ValueError: If distribution type is invalid
    """
    if distribution not in ["rademacher", "gaussian"]:
        raise ValueError("distribution must be 'rademacher' or 'gaussian'")
    
    trace_estimates = []
    
    for _ in range(n_samples):
        # Generate random vector
        if distribution == "rademacher":
            # Rademacher: {-1, +1}^dim with equal probability
            z = 2 * np.random.randint(0, 2, size=dim) - 1
            z = z.astype(float)
        else:  # gaussian
            # Standard normal
            z = np.random.randn(dim)
        
        # Compute H @ z
        Hz = hessian_vec_prod(z)
        
        # Estimate: z^T H z
        trace_contribution = np.dot(z, Hz)
        trace_estimates.append(trace_contribution)
    
    return np.mean(trace_estimates)


def stochastic_hessian_diagonal(func: Callable[[np.ndarray], float],
                               x: np.ndarray,
                               n_samples: int = 20,
                               eps: float = 1e-5) -> np.ndarray:
    """
    Stochastic estimation of Hessian diagonal using finite differences.
    
    Mathematical formulation:
    H_{ii} = ∂²f/∂x_i² ≈ (f(x + ε e_i) - 2f(x) + f(x - ε e_i)) / ε²
    
    Stochastic version uses random directions:
    H_{ii} ≈ E[(f(x + ε z e_i) - 2f(x) + f(x - ε z e_i)) / ε²]
    where z ~ uniform[-1, 1] or other distributions
    
    This reduces function evaluations from O(d) to O(n_samples) per diagonal element.
    
    Args:
        func: Scalar function to compute Hessian diagonal for
        x: Point at which to evaluate Hessian diagonal
        n_samples: Number of stochastic samples per diagonal element
        eps: Finite difference step size
        
    Returns:
        Estimated diagonal of Hessian matrix
    """
    dim = len(x)
    f_center = func(x)
    diagonal = np.zeros(dim)
    
    for i in range(dim):
        estimates = []
        
        for _ in range(n_samples):
            # Random perturbation magnitude
            z = np.random.uniform(-1, 1)
            
            # Perturbed points
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps * z
            x_minus[i] -= eps * z
            
            # Finite difference approximation
            f_plus = func(x_plus)
            f_minus = func(x_minus)
            
            second_derivative = (f_plus - 2*f_center + f_minus) / (eps * z)**2
            estimates.append(second_derivative)
        
        diagonal[i] = np.mean(estimates)
    
    return diagonal


def block_hessian_computation(func: Callable[[np.ndarray], float],
                             x: np.ndarray,
                             block_size: int = 100,
                             method: str = "finite_diff",
                             eps: float = 1e-6) -> np.ndarray:
    """
    Block-wise Hessian computation for high-dimensional problems.
    
    Mathematical approach:
    Partition the Hessian H into blocks:
    H = [H₁₁  H₁₂  ...  H₁ₖ]
        [H₂₁  H₂₂  ...  H₂ₖ]
        [... ... ... ...]
        [Hₖ₁  Hₖ₂  ...  Hₖₖ]
    
    Compute each block H_ij separately to manage memory usage.
    For diagonal blocks, use more accurate methods.
    For off-diagonal blocks, may use approximations or set to zero.
    
    Memory complexity: O(block_size²) instead of O(d²)
    
    Args:
        func: Scalar function to compute Hessian for
        x: Point at which to evaluate Hessian
        block_size: Size of each block
        method: Computation method ("finite_diff" or "diagonal_only")
        eps: Step size for finite differences
        
    Returns:
        Block-wise computed Hessian matrix
        
    Raises:
        ValueError: If method is not recognized
    """
    if method not in ["finite_diff", "diagonal_only"]:
        raise ValueError("method must be 'finite_diff' or 'diagonal_only'")
    
    dim = len(x)
    n_blocks = (dim + block_size - 1) // block_size  # Ceiling division
    
    H = np.zeros((dim, dim))
    
    if method == "diagonal_only":
        # Only compute diagonal blocks, assume others are zero
        for block_i in range(n_blocks):
            start_i = block_i * block_size
            end_i = min((block_i + 1) * block_size, dim)
            block_dim_i = end_i - start_i
            
            # Extract subvector
            x_block = x[start_i:end_i]
            
            # Define block function
            def block_func(x_sub):
                x_full = x.copy()
                x_full[start_i:end_i] = x_sub
                return func(x_full)
            
            # Compute block Hessian
            H_block = _finite_diff_hessian_block(block_func, x_block, eps)
            
            # Insert into full matrix
            H[start_i:end_i, start_i:end_i] = H_block
    
    else:  # finite_diff
        # Compute all blocks (memory intensive for large problems)
        for block_i in range(n_blocks):
            for block_j in range(block_i, n_blocks):  # Upper triangular
                start_i = block_i * block_size
                end_i = min((block_i + 1) * block_size, dim)
                start_j = block_j * block_size  
                end_j = min((block_j + 1) * block_size, dim)
                
                # Compute mixed partial derivatives
                H_block = _finite_diff_mixed_block(func, x, 
                                                  start_i, end_i, start_j, end_j, eps)
                
                # Insert into full matrix (symmetric)
                H[start_i:end_i, start_j:end_j] = H_block
                if block_i != block_j:
                    H[start_j:end_j, start_i:end_i] = H_block.T
    
    return H


def _finite_diff_hessian_block(func: Callable[[np.ndarray], float],
                              x: np.ndarray,
                              eps: float) -> np.ndarray:
    """Compute Hessian of a block using finite differences."""
    dim = len(x)
    H = np.zeros((dim, dim))
    f_center = func(x)
    
    # Diagonal elements
    for i in range(dim):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        
        H[i, i] = (func(x_plus) - 2*f_center + func(x_minus)) / (eps**2)
    
    # Off-diagonal elements
    for i in range(dim):
        for j in range(i+1, dim):
            x_pp = x.copy()
            x_pm = x.copy()  
            x_mp = x.copy()
            x_mm = x.copy()
            
            x_pp[i] += eps; x_pp[j] += eps
            x_pm[i] += eps; x_pm[j] -= eps
            x_mp[i] -= eps; x_mp[j] += eps
            x_mm[i] -= eps; x_mm[j] -= eps
            
            mixed_deriv = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4*eps**2)
            H[i, j] = mixed_deriv
            H[j, i] = mixed_deriv
    
    return H


def _finite_diff_mixed_block(func: Callable[[np.ndarray], float],
                            x: np.ndarray,
                            start_i: int, end_i: int,
                            start_j: int, end_j: int,
                            eps: float) -> np.ndarray:
    """Compute mixed partial derivatives between two blocks."""
    dim_i = end_i - start_i
    dim_j = end_j - start_j
    H_block = np.zeros((dim_i, dim_j))
    
    for i in range(dim_i):
        for j in range(dim_j):
            # Global indices
            gi = start_i + i
            gj = start_j + j
            
            if gi == gj:
                # Diagonal element - use second derivative formula
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[gi] += eps
                x_minus[gi] -= eps
                
                f_center = func(x)
                H_block[i, j] = (func(x_plus) - 2*f_center + func(x_minus)) / (eps**2)
            else:
                # Mixed partial derivative
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                
                x_pp[gi] += eps; x_pp[gj] += eps
                x_pm[gi] += eps; x_pm[gj] -= eps
                x_mp[gi] -= eps; x_mp[gj] += eps
                x_mm[gi] -= eps; x_mm[gj] -= eps
                
                mixed_deriv = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4*eps**2)
                H_block[i, j] = mixed_deriv
    
    return H_block


def adaptive_regularization(H: np.ndarray, 
                          min_eigenval: float = 1e-6,
                          max_condition: float = 1e12) -> Tuple[np.ndarray, float]:
    """
    Adaptive regularization of Hessian matrix based on spectral properties.
    
    Mathematical approach:
    1. Compute eigendecomposition: H = Q Λ Q^T
    2. Modify eigenvalues: Λ_reg = max(Λ, λ_min)
    3. Limit condition number: Λ_reg = min(Λ_reg, λ_max/κ_max Λ_max)
    4. Reconstruct: H_reg = Q Λ_reg Q^T
    
    Args:
        H: Input Hessian matrix
        min_eigenval: Minimum allowed eigenvalue
        max_condition: Maximum allowed condition number
        
    Returns:
        Regularized Hessian matrix and regularization parameter used
    """
    eigenvals, eigenvecs = np.linalg.eigh(H)
    
    # Apply minimum eigenvalue threshold
    eigenvals_reg = np.maximum(eigenvals, min_eigenval)
    
    # Limit condition number
    max_eigenval = np.max(eigenvals_reg)
    min_allowed = max_eigenval / max_condition
    eigenvals_reg = np.maximum(eigenvals_reg, min_allowed)
    
    # Compute regularization parameter
    regularization = np.max(eigenvals_reg - eigenvals)
    
    # Reconstruct matrix
    H_reg = eigenvecs @ np.diag(eigenvals_reg) @ eigenvecs.T
    
    return H_reg, regularization


def low_rank_hessian_update(H_prev: np.ndarray,
                           s: np.ndarray,
                           y: np.ndarray,
                           rank_limit: int = None) -> np.ndarray:
    """
    Low-rank update to Hessian approximation using BFGS formula.
    
    Mathematical formulation:
    BFGS update: H_{k+1} = H_k - (H_k s s^T H_k)/(s^T H_k s) + (y y^T)/(s^T y)
    
    For low-rank approximation, keep only the r largest eigenvalues/vectors.
    
    Args:
        H_prev: Previous Hessian approximation
        s: State difference vector
        y: Gradient difference vector  
        rank_limit: Maximum rank to maintain
        
    Returns:
        Updated low-rank Hessian approximation
    """
    # Check BFGS condition
    sy = np.dot(s, y)
    if sy <= 1e-12:
        warnings.warn("BFGS curvature condition violated, skipping update")
        return H_prev
    
    # BFGS update
    Hs = H_prev @ s
    sHs = np.dot(s, Hs)
    
    if sHs <= 1e-12:
        warnings.warn("Denominator too small in BFGS update")
        return H_prev
    
    # Standard BFGS update
    H_new = H_prev - np.outer(Hs, Hs) / sHs + np.outer(y, y) / sy
    
    # Low-rank approximation if specified
    if rank_limit is not None and rank_limit < H_new.shape[0]:
        eigenvals, eigenvecs = np.linalg.eigh(H_new)
        
        # Keep largest eigenvalues
        idx = np.argsort(eigenvals)[::-1][:rank_limit]
        eigenvals_trunc = eigenvals[idx]
        eigenvecs_trunc = eigenvecs[:, idx]
        
        # Reconstruct low-rank approximation
        H_new = eigenvecs_trunc @ np.diag(eigenvals_trunc) @ eigenvecs_trunc.T
    
    return H_new