"""
Hessian computation utilities for high-dimensional sampling.

This module provides functions for computing Hessian matrices using automatic
differentiation and finite differences, along with utilities for handling
numerical stability issues in high-dimensional spaces.
"""

from typing import Callable, Tuple, Optional
import numpy as np
import warnings
from scipy.linalg import eigvals, eigvalsh
from functools import wraps

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available. Automatic differentiation will be limited.")

try:
    import jax
    import jax.numpy as jnp
    from jax import hessian, grad
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def _validate_function_input(func: Callable, x: np.ndarray) -> None:
    """Validate function and input for Hessian computation."""
    if not callable(func):
        raise TypeError("func must be callable")
    
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a numpy array")
    
    if x.ndim != 1:
        raise ValueError("x must be a 1-dimensional array")
    
    # Test function evaluation
    try:
        result = func(x)
        if not np.isscalar(result):
            raise ValueError("Function must return a scalar value")
    except Exception as e:
        raise ValueError(f"Function evaluation failed: {e}")


def compute_hessian_autodiff(func: Callable[[np.ndarray], float], 
                           x: np.ndarray,
                           backend: str = "jax") -> np.ndarray:
    """
    Compute Hessian matrix using automatic differentiation.
    
    Args:
        func: Target function returning scalar value
        x: Point at which to evaluate Hessian (1D array)
        backend: AD backend to use ('jax' or 'torch')
        
    Returns:
        Hessian matrix H[i,j] = ∂²f/∂x_i∂x_j
        
    Raises:
        ValueError: If function is invalid or backend unavailable
        RuntimeError: If computation fails
    """
    _validate_function_input(func, x)
    
    if backend == "jax" and HAS_JAX:
        return _compute_hessian_jax(func, x)
    elif backend == "torch" and HAS_TORCH:
        return _compute_hessian_torch(func, x)
    else:
        available_backends = []
        if HAS_JAX:
            available_backends.append("jax")
        if HAS_TORCH:
            available_backends.append("torch")
        
        if not available_backends:
            raise RuntimeError("No automatic differentiation backend available")
        
        warnings.warn(f"Backend {backend} not available, using {available_backends[0]}")
        return compute_hessian_autodiff(func, x, backend=available_backends[0])


def _compute_hessian_jax(func: Callable, x: np.ndarray) -> np.ndarray:
    """Compute Hessian using JAX."""
    def jax_func(x_jax):
        return func(np.array(x_jax))
    
    try:
        hess_func = hessian(jax_func)
        x_jax = jnp.array(x)
        H = hess_func(x_jax)
        return np.array(H)
    except Exception as e:
        raise RuntimeError(f"JAX Hessian computation failed: {e}")


def _compute_hessian_torch(func: Callable, x: np.ndarray) -> np.ndarray:
    """Compute Hessian using PyTorch."""
    x_torch = torch.tensor(x, requires_grad=True, dtype=torch.float64)
    
    try:
        # Compute function value
        f_val = func(x_torch.detach().numpy())
        f_tensor = torch.tensor(f_val, dtype=torch.float64)
        
        # Compute gradient
        grad_outputs = torch.ones_like(f_tensor)
        grads = torch.autograd.grad(f_tensor, x_torch, create_graph=True)[0]
        
        # Compute Hessian
        n = len(x)
        H = torch.zeros(n, n, dtype=torch.float64)
        
        for i in range(n):
            grad2 = torch.autograd.grad(grads[i], x_torch, retain_graph=True)[0]
            H[i] = grad2
            
        return H.detach().numpy()
    except Exception as e:
        raise RuntimeError(f"PyTorch Hessian computation failed: {e}")


def compute_hessian_finite_diff(func: Callable[[np.ndarray], float],
                               x: np.ndarray,
                               eps: float = 1e-6,
                               method: str = "central") -> np.ndarray:
    """
    Compute Hessian matrix using finite differences.
    
    Args:
        func: Target function returning scalar value
        x: Point at which to evaluate Hessian
        eps: Step size for finite differences
        method: Finite difference method ('central', 'forward')
        
    Returns:
        Hessian matrix approximation
        
    Raises:
        ValueError: If inputs are invalid
    """
    _validate_function_input(func, x)
    
    if eps <= 0:
        raise ValueError("eps must be positive")
    
    n = len(x)
    H = np.zeros((n, n))
    
    if method == "central":
        # Central differences: f(x+h) - 2f(x) + f(x-h) / h²
        f_center = func(x)
        
        # Diagonal elements
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            H[i, i] = (func(x_plus) - 2*f_center + func(x_minus)) / (eps**2)
        
        # Off-diagonal elements
        for i in range(n):
            for j in range(i+1, n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                
                x_pp[i] += eps; x_pp[j] += eps
                x_pm[i] += eps; x_pm[j] -= eps
                x_mp[i] -= eps; x_mp[j] += eps
                x_mm[i] -= eps; x_mm[j] -= eps
                
                H[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4*eps**2)
                H[j, i] = H[i, j]  # Symmetry
                
    elif method == "forward":
        # Forward differences
        f_center = func(x)
        
        for i in range(n):
            x_i = x.copy()
            x_i[i] += eps
            f_i = func(x_i)
            
            for j in range(n):
                x_j = x.copy()
                x_j[j] += eps
                f_j = func(x_j)
                
                x_ij = x.copy()
                x_ij[i] += eps
                x_ij[j] += eps
                f_ij = func(x_ij)
                
                H[i, j] = (f_ij - f_i - f_j + f_center) / (eps**2)
    else:
        raise ValueError("method must be 'central' or 'forward'")
    
    return H


def hessian_eigendecomposition(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigendecomposition of Hessian matrix.
    
    Args:
        H: Hessian matrix (must be symmetric)
        
    Returns:
        eigenvalues: Eigenvalues in descending order
        eigenvectors: Corresponding eigenvectors as columns
        
    Raises:
        ValueError: If H is not square or symmetric
    """
    if not isinstance(H, np.ndarray):
        raise TypeError("H must be a numpy array")
    
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix")
    
    # Check symmetry
    if not np.allclose(H, H.T, rtol=1e-10):
        warnings.warn("Hessian is not symmetric, symmetrizing...")
        H = (H + H.T) / 2
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Eigendecomposition failed: {e}")


def condition_hessian(H: np.ndarray, 
                     min_eigenval: float = 1e-6,
                     regularization: str = "identity") -> np.ndarray:
    """
    Condition Hessian matrix for numerical stability.
    
    Handles ill-conditioned Hessians by adding regularization to ensure
    all eigenvalues are above min_eigenval.
    
    Args:
        H: Hessian matrix
        min_eigenval: Minimum allowed eigenvalue
        regularization: Type of regularization ('identity', 'diagonal')
        
    Returns:
        Conditioned Hessian matrix
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(H, np.ndarray):
        raise TypeError("H must be a numpy array")
    
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix")
    
    if min_eigenval <= 0:
        raise ValueError("min_eigenval must be positive")
    
    # Ensure symmetry
    H_sym = (H + H.T) / 2
    
    try:
        eigenvals = eigvalsh(H_sym)
        min_eig = np.min(eigenvals)
        
        if min_eig >= min_eigenval:
            return H_sym
        
        # Apply regularization
        if regularization == "identity":
            reg_param = max(0, min_eigenval - min_eig)
            H_reg = H_sym + reg_param * np.eye(H.shape[0])
        elif regularization == "diagonal":
            # Add to diagonal elements proportional to their magnitude
            diag_reg = np.maximum(min_eigenval - np.diag(H_sym), 0)
            H_reg = H_sym + np.diag(diag_reg)
        else:
            raise ValueError("regularization must be 'identity' or 'diagonal'")
        
        return H_reg
        
    except Exception as e:
        raise RuntimeError(f"Hessian conditioning failed: {e}")


def hessian_condition_number(H: np.ndarray) -> float:
    """
    Compute condition number of Hessian matrix.
    
    Args:
        H: Hessian matrix
        
    Returns:
        Condition number (ratio of largest to smallest eigenvalue)
    """
    if not isinstance(H, np.ndarray):
        raise TypeError("H must be a numpy array")
    
    try:
        eigenvals = eigvalsh(H)
        eigenvals = eigenvals[eigenvals > 0]  # Only positive eigenvalues
        
        if len(eigenvals) == 0:
            return np.inf
        
        return np.max(eigenvals) / np.min(eigenvals)
    except Exception:
        return np.inf


def is_positive_definite(H: np.ndarray, tol: float = 1e-12) -> bool:
    """
    Check if Hessian matrix is positive definite.
    
    Args:
        H: Hessian matrix
        tol: Tolerance for numerical precision
        
    Returns:
        True if positive definite, False otherwise
    """
    try:
        eigenvals = eigvalsh(H)
        return np.all(eigenvals > tol)
    except Exception:
        return False