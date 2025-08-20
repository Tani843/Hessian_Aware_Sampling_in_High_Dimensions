"""
Mathematical utilities for high-dimensional operations.

This module provides numerically stable implementations of common
mathematical operations needed for Hessian-aware sampling, including
Cholesky decomposition, matrix operations, and probability computations.
"""

from typing import Optional, Tuple
import numpy as np
import warnings
from scipy.linalg import cholesky, solve_triangular, LinAlgError
from scipy.special import logsumexp


def safe_cholesky(A: np.ndarray, 
                  regularization: float = 1e-6,
                  max_tries: int = 10) -> np.ndarray:
    """
    Compute Cholesky decomposition with automatic regularization for stability.
    
    For a positive definite matrix A, returns lower triangular L such that A = L @ L.T.
    If A is not positive definite, adds regularization until decomposition succeeds.
    
    Args:
        A: Square symmetric matrix to decompose
        regularization: Initial regularization parameter
        max_tries: Maximum number of regularization attempts
        
    Returns:
        Lower triangular Cholesky factor L
        
    Raises:
        ValueError: If A is not square or symmetric
        LinAlgError: If decomposition fails after max_tries
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("A must be a numpy array")
    
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    
    n = A.shape[0]
    
    # Check and enforce symmetry
    if not np.allclose(A, A.T, rtol=1e-12):
        warnings.warn("Matrix is not symmetric, symmetrizing...")
        A = (A + A.T) / 2
    
    reg = regularization
    
    for attempt in range(max_tries):
        try:
            if attempt == 0:
                # First try without regularization
                L = cholesky(A, lower=True)
            else:
                # Add regularization
                A_reg = A + reg * np.eye(n)
                L = cholesky(A_reg, lower=True)
            
            return L
            
        except LinAlgError:
            if attempt == max_tries - 1:
                raise LinAlgError(f"Cholesky decomposition failed after {max_tries} attempts")
            
            # Increase regularization for next attempt
            reg *= 10
            if attempt == 0:
                warnings.warn(f"Matrix not positive definite, adding regularization {reg}")
    
    # Should not reach here
    raise LinAlgError("Unexpected error in Cholesky decomposition")


def woodbury_matrix_identity(A_inv: np.ndarray, 
                           U: np.ndarray, 
                           V: np.ndarray) -> np.ndarray:
    """
    Compute (A + U @ V.T)^(-1) using Woodbury matrix identity.
    
    Uses the identity: (A + U @ V.T)^(-1) = A^(-1) - A^(-1) @ U @ (I + V.T @ A^(-1) @ U)^(-1) @ V.T @ A^(-1)
    
    Efficient for low-rank updates to invertible matrices.
    
    Args:
        A_inv: Inverse of matrix A
        U: Left factor matrix (n × k)
        V: Right factor matrix (n × k)
        
    Returns:
        Inverse of (A + U @ V.T)
        
    Raises:
        ValueError: If matrix dimensions are incompatible
    """
    if not all(isinstance(x, np.ndarray) for x in [A_inv, U, V]):
        raise TypeError("All inputs must be numpy arrays")
    
    n = A_inv.shape[0]
    if A_inv.shape != (n, n):
        raise ValueError("A_inv must be square")
    
    if U.shape[0] != n or V.shape[0] != n:
        raise ValueError("U and V must have same first dimension as A_inv")
    
    if U.shape[1] != V.shape[1]:
        raise ValueError("U and V must have same second dimension")
    
    k = U.shape[1]
    
    try:
        # Compute V.T @ A_inv @ U
        temp1 = A_inv @ U  # n × k
        temp2 = V.T @ temp1  # k × k
        
        # Compute (I + V.T @ A_inv @ U)^(-1)
        I_k = np.eye(k)
        middle_inv = np.linalg.inv(I_k + temp2)  # k × k
        
        # Apply Woodbury identity
        result = A_inv - temp1 @ middle_inv @ V.T @ A_inv
        
        return result
        
    except LinAlgError as e:
        raise LinAlgError(f"Woodbury matrix identity computation failed: {e}")


def log_det_via_cholesky(A: np.ndarray) -> float:
    """
    Compute log determinant using Cholesky decomposition.
    
    For positive definite matrix A with Cholesky decomposition A = L @ L.T,
    log(det(A)) = 2 * sum(log(diag(L))).
    
    Args:
        A: Positive definite matrix
        
    Returns:
        log(det(A))
        
    Raises:
        ValueError: If A is not positive definite
    """
    try:
        L = safe_cholesky(A)
        return 2 * np.sum(np.log(np.diag(L)))
    except LinAlgError:
        raise ValueError("Matrix must be positive definite for log determinant computation")


def multivariate_normal_logpdf(x: np.ndarray, 
                              mean: np.ndarray, 
                              cov: np.ndarray,
                              allow_singular: bool = False) -> float:
    """
    Compute log probability density of multivariate normal distribution.
    
    Uses numerically stable computation via Cholesky decomposition.
    
    Args:
        x: Point to evaluate (1D array)
        mean: Mean vector
        cov: Covariance matrix
        allow_singular: If True, use pseudo-inverse for singular matrices
        
    Returns:
        Log probability density
        
    Raises:
        ValueError: If inputs have incompatible dimensions
    """
    if not all(isinstance(arr, np.ndarray) for arr in [x, mean, cov]):
        raise TypeError("All inputs must be numpy arrays")
    
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    
    if x.ndim != 1 or mean.ndim != 1:
        raise ValueError("x and mean must be 1-dimensional")
    
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square matrix")
    
    dim = len(x)
    if len(mean) != dim or cov.shape[0] != dim:
        raise ValueError("Inconsistent dimensions")
    
    # Center the data
    diff = x - mean
    
    try:
        # Use Cholesky for numerical stability
        L = safe_cholesky(cov)
        
        # Solve L @ y = diff for y
        y = solve_triangular(L, diff, lower=True)
        
        # Compute log probability
        log_det = 2 * np.sum(np.log(np.diag(L)))
        quad_form = np.dot(y, y)
        
        log_prob = -0.5 * (dim * np.log(2 * np.pi) + log_det + quad_form)
        
        return log_prob
        
    except LinAlgError:
        if allow_singular:
            warnings.warn("Covariance matrix is singular, using pseudo-inverse")
            
            # Use SVD for singular matrices
            U, s, Vt = np.linalg.svd(cov)
            
            # Tolerance for singular values
            tol = np.finfo(s.dtype).eps * max(cov.shape) * s.max()
            non_zero = s > tol
            
            if not np.any(non_zero):
                raise ValueError("Covariance matrix is completely singular")
            
            # Pseudo-inverse and log determinant
            s_inv = np.zeros_like(s)
            s_inv[non_zero] = 1.0 / s[non_zero]
            
            cov_pinv = U @ np.diag(s_inv) @ Vt
            log_det = np.sum(np.log(s[non_zero]))
            
            quad_form = diff @ cov_pinv @ diff
            effective_dim = np.sum(non_zero)
            
            log_prob = -0.5 * (effective_dim * np.log(2 * np.pi) + log_det + quad_form)
            
            return log_prob
        else:
            raise ValueError("Covariance matrix is singular. Set allow_singular=True to handle.")


def stable_logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Numerically stable log-sum-exp computation.
    
    Computes log(sum(exp(a))) in a numerically stable way.
    
    Args:
        a: Input array
        axis: Axis along which to compute (None for all elements)
        
    Returns:
        log(sum(exp(a)))
    """
    return logsumexp(a, axis=axis)


def matrix_sqrt_inv(A: np.ndarray, method: str = "cholesky") -> np.ndarray:
    """
    Compute A^(-1/2) for positive definite matrix A.
    
    Args:
        A: Positive definite matrix
        method: Method to use ('cholesky', 'eigen', 'svd')
        
    Returns:
        A^(-1/2)
        
    Raises:
        ValueError: If method is unknown or computation fails
    """
    if method == "cholesky":
        try:
            L = safe_cholesky(A)
            # A = L @ L.T, so A^(-1/2) = L^(-T)
            L_inv = solve_triangular(L, np.eye(A.shape[0]), lower=True)
            return L_inv.T
        except LinAlgError:
            warnings.warn("Cholesky method failed, falling back to eigendecomposition")
            method = "eigen"
    
    if method == "eigen":
        eigenvals, eigenvecs = np.linalg.eigh(A)
        
        if np.any(eigenvals <= 0):
            warnings.warn("Non-positive eigenvalues found, using regularization")
            eigenvals = np.maximum(eigenvals, 1e-12)
        
        sqrt_inv_eigenvals = 1.0 / np.sqrt(eigenvals)
        return eigenvecs @ np.diag(sqrt_inv_eigenvals) @ eigenvecs.T
    
    elif method == "svd":
        U, s, Vt = np.linalg.svd(A)
        
        if np.any(s <= 0):
            warnings.warn("Non-positive singular values found, using regularization")
            s = np.maximum(s, 1e-12)
        
        sqrt_inv_s = 1.0 / np.sqrt(s)
        return U @ np.diag(sqrt_inv_s) @ Vt
    
    else:
        raise ValueError("method must be 'cholesky', 'eigen', or 'svd'")


def condition_number(A: np.ndarray) -> float:
    """
    Compute condition number of matrix A.
    
    Args:
        A: Input matrix
        
    Returns:
        Condition number (ratio of largest to smallest singular value)
    """
    s = np.linalg.svd(A, compute_uv=False)
    return s.max() / s.min() if s.min() > 0 else np.inf


def is_symmetric(A: np.ndarray, tol: float = 1e-12) -> bool:
    """
    Check if matrix is symmetric within tolerance.
    
    Args:
        A: Input matrix
        tol: Tolerance for symmetry check
        
    Returns:
        True if symmetric, False otherwise
    """
    if A.shape[0] != A.shape[1]:
        return False
    
    return np.allclose(A, A.T, rtol=tol, atol=tol)


def nearest_positive_definite(A: np.ndarray) -> np.ndarray:
    """
    Find nearest positive definite matrix to A.
    
    Uses eigendecomposition and sets negative eigenvalues to small positive values.
    
    Args:
        A: Input symmetric matrix
        
    Returns:
        Nearest positive definite matrix
    """
    # Ensure symmetry
    A_sym = (A + A.T) / 2
    
    # Eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(A_sym)
    
    # Replace negative/zero eigenvalues with small positive values
    eigenvals = np.maximum(eigenvals, 1e-12)
    
    # Reconstruct matrix
    return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T


def gradcheck(func: callable, x: np.ndarray, eps: float = 1e-6) -> Tuple[bool, float]:
    """
    Check gradient implementation using finite differences.
    
    Args:
        func: Function returning (value, gradient) tuple
        x: Point to check gradient at
        eps: Finite difference step size
        
    Returns:
        (is_correct, max_error): Boolean indicating correctness and maximum error
    """
    f_val, grad_analytic = func(x)
    
    grad_numerical = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        
        f_plus, _ = func(x_plus)
        f_minus, _ = func(x_minus)
        
        grad_numerical[i] = (f_plus - f_minus) / (2 * eps)
    
    error = np.abs(grad_analytic - grad_numerical)
    max_error = np.max(error)
    
    # Consider correct if relative error is small
    rel_error = error / (np.abs(grad_analytic) + eps)
    is_correct = np.all(rel_error < 1e-4)
    
    return is_correct, max_error