"""
Input validation utilities for Hessian-aware sampling.

This module provides comprehensive validation functions for 
user inputs, numerical stability checks, and data integrity
verification throughout the sampling process.
"""

from typing import Callable, Any, Optional, Union, Tuple, Dict
import numpy as np
import warnings
from functools import wraps


def validate_array(arr: Any, 
                  name: str,
                  expected_shape: Optional[Tuple[int, ...]] = None,
                  expected_dtype: Optional[type] = None,
                  allow_none: bool = False,
                  min_val: Optional[float] = None,
                  max_val: Optional[float] = None) -> np.ndarray:
    """
    Validate numpy array with comprehensive checks.
    
    Args:
        arr: Array to validate
        name: Name for error messages
        expected_shape: Expected shape (None to skip check)
        expected_dtype: Expected dtype (None to skip check)
        allow_none: Whether None is acceptable
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Validated numpy array
        
    Raises:
        TypeError: If type is invalid
        ValueError: If values are invalid
    """
    if arr is None:
        if allow_none:
            return None
        else:
            raise TypeError(f"{name} cannot be None")
    
    # Convert to numpy array
    try:
        arr = np.asarray(arr)
    except Exception as e:
        raise TypeError(f"{name} must be convertible to numpy array: {e}")
    
    # Check dtype
    if expected_dtype is not None:
        if not np.issubdtype(arr.dtype, expected_dtype):
            try:
                arr = arr.astype(expected_dtype)
            except Exception as e:
                raise TypeError(f"{name} must be convertible to {expected_dtype}: {e}")
    
    # Check shape
    if expected_shape is not None:
        if arr.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values")
    
    if np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains infinite values")
    
    # Check value bounds
    if min_val is not None:
        if np.any(arr < min_val):
            raise ValueError(f"{name} contains values below minimum {min_val}")
    
    if max_val is not None:
        if np.any(arr > max_val):
            raise ValueError(f"{name} contains values above maximum {max_val}")
    
    return arr


def validate_function(func: Any, 
                     name: str,
                     test_input: Optional[np.ndarray] = None,
                     expected_output_type: type = float) -> Callable:
    """
    Validate function with optional test evaluation.
    
    Args:
        func: Function to validate
        name: Name for error messages
        test_input: Test input for function evaluation
        expected_output_type: Expected output type
        
    Returns:
        Validated function
        
    Raises:
        TypeError: If function is invalid
        ValueError: If function evaluation fails
    """
    if not callable(func):
        raise TypeError(f"{name} must be callable")
    
    if test_input is not None:
        try:
            result = func(test_input)
            
            if not isinstance(result, expected_output_type):
                if expected_output_type == float and np.isscalar(result):
                    # Allow scalar types that can be converted to float
                    try:
                        float(result)
                    except (TypeError, ValueError):
                        raise TypeError(f"{name} must return {expected_output_type}, got {type(result)}")
                else:
                    raise TypeError(f"{name} must return {expected_output_type}, got {type(result)}")
            
            # Check for NaN or infinite output
            if np.isnan(result) or np.isinf(result):
                warnings.warn(f"{name} returned NaN/infinite value for test input")
                
        except Exception as e:
            raise ValueError(f"{name} evaluation failed: {e}")
    
    return func


def validate_positive_scalar(value: Any, 
                           name: str,
                           allow_zero: bool = False) -> float:
    """
    Validate positive scalar value.
    
    Args:
        value: Value to validate
        name: Name for error messages
        allow_zero: Whether zero is acceptable
        
    Returns:
        Validated float value
        
    Raises:
        TypeError: If not numeric
        ValueError: If not positive
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be numeric")
    
    if np.isnan(value) or np.isinf(value):
        raise ValueError(f"{name} cannot be NaN or infinite")
    
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    
    return value


def validate_integer(value: Any,
                    name: str,
                    min_val: Optional[int] = None,
                    max_val: Optional[int] = None) -> int:
    """
    Validate integer value with optional bounds.
    
    Args:
        value: Value to validate
        name: Name for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Validated integer
        
    Raises:
        TypeError: If not integer
        ValueError: If outside bounds
    """
    if not isinstance(value, (int, np.integer)):
        # Try to convert
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise TypeError(f"{name} must be an integer")
    
    value = int(value)
    
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}")
    
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}")
    
    return value


def validate_probability(value: Any, name: str) -> float:
    """
    Validate probability value (between 0 and 1).
    
    Args:
        value: Value to validate
        name: Name for error messages
        
    Returns:
        Validated probability
        
    Raises:
        TypeError: If not numeric
        ValueError: If outside [0, 1]
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be numeric")
    
    if np.isnan(value) or np.isinf(value):
        raise ValueError(f"{name} cannot be NaN or infinite")
    
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    
    return value


def validate_matrix(matrix: Any,
                   name: str,
                   square: bool = False,
                   symmetric: bool = False,
                   positive_definite: bool = False,
                   min_eigenval: Optional[float] = None) -> np.ndarray:
    """
    Validate matrix with various structural properties.
    
    Args:
        matrix: Matrix to validate
        name: Name for error messages
        square: Whether matrix must be square
        symmetric: Whether matrix must be symmetric
        positive_definite: Whether matrix must be positive definite
        min_eigenval: Minimum eigenvalue requirement
        
    Returns:
        Validated matrix
        
    Raises:
        ValueError: If matrix properties are invalid
    """
    matrix = validate_array(matrix, name, expected_dtype=float)
    
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional")
    
    rows, cols = matrix.shape
    
    if square and rows != cols:
        raise ValueError(f"{name} must be square")
    
    if symmetric:
        if rows != cols:
            raise ValueError(f"{name} must be square to check symmetry")
        
        if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
            raise ValueError(f"{name} must be symmetric")
    
    if positive_definite or min_eigenval is not None:
        if rows != cols:
            raise ValueError(f"{name} must be square to check eigenvalues")
        
        try:
            eigenvals = np.linalg.eigvals(matrix)
        except np.linalg.LinAlgError:
            raise ValueError(f"{name} eigenvalue computation failed")
        
        if positive_definite:
            if not np.all(eigenvals > 0):
                raise ValueError(f"{name} must be positive definite")
        
        if min_eigenval is not None:
            if np.min(eigenvals) < min_eigenval:
                raise ValueError(f"{name} minimum eigenvalue ({np.min(eigenvals)}) "
                               f"is below required threshold ({min_eigenval})")
    
    return matrix


def validate_sampling_params(n_samples: int,
                           burnin: int = 0,
                           thin: int = 1,
                           initial_state: Optional[np.ndarray] = None,
                           dim: Optional[int] = None) -> Tuple[int, int, int, Optional[np.ndarray]]:
    """
    Validate sampling parameters for consistency.
    
    Args:
        n_samples: Number of samples
        burnin: Burnin period
        thin: Thinning interval
        initial_state: Initial state
        dim: Expected dimensionality
        
    Returns:
        Validated parameters
        
    Raises:
        ValueError: If parameters are inconsistent
    """
    n_samples = validate_integer(n_samples, "n_samples", min_val=1)
    burnin = validate_integer(burnin, "burnin", min_val=0)
    thin = validate_integer(thin, "thin", min_val=1)
    
    if initial_state is not None:
        initial_state = validate_array(initial_state, "initial_state", expected_dtype=float)
        
        if initial_state.ndim != 1:
            raise ValueError("initial_state must be 1-dimensional")
        
        if dim is not None and len(initial_state) != dim:
            raise ValueError(f"initial_state dimension ({len(initial_state)}) "
                           f"does not match expected dimension ({dim})")
    
    return n_samples, burnin, thin, initial_state


def check_numerical_stability(values: np.ndarray,
                            name: str,
                            warn_threshold: float = 1e-10,
                            error_threshold: float = 1e-16) -> bool:
    """
    Check numerical stability of computed values.
    
    Args:
        values: Values to check
        name: Name for messages
        warn_threshold: Threshold for warnings
        error_threshold: Threshold for errors
        
    Returns:
        True if stable, False otherwise
        
    Raises:
        ValueError: If severely unstable
    """
    if np.any(np.isnan(values)):
        raise ValueError(f"{name} contains NaN values")
    
    if np.any(np.isinf(values)):
        raise ValueError(f"{name} contains infinite values")
    
    # Check for very small values that might indicate numerical issues
    min_abs_val = np.min(np.abs(values[values != 0]))
    
    if min_abs_val < error_threshold:
        raise ValueError(f"{name} contains extremely small values ({min_abs_val}) "
                        f"indicating numerical instability")
    
    if min_abs_val < warn_threshold:
        warnings.warn(f"{name} contains small values ({min_abs_val}) "
                     f"that may indicate numerical issues")
        return False
    
    return True


def validate_step_size_adaptation(step_size: float,
                                 acceptance_rate: float,
                                 target_acceptance: float,
                                 adaptation_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate step size adaptation parameters.
    
    Args:
        step_size: Current step size
        acceptance_rate: Current acceptance rate
        target_acceptance: Target acceptance rate
        adaptation_params: Additional adaptation parameters
        
    Returns:
        Validated adaptation parameters
        
    Raises:
        ValueError: If parameters are invalid
    """
    step_size = validate_positive_scalar(step_size, "step_size")
    acceptance_rate = validate_probability(acceptance_rate, "acceptance_rate")
    target_acceptance = validate_probability(target_acceptance, "target_acceptance")
    
    if adaptation_params is None:
        adaptation_params = {}
    
    # Default adaptation parameters
    defaults = {
        'adaptation_rate': 0.1,
        'min_step_size': 1e-6,
        'max_step_size': 1.0,
        'stability_threshold': 0.05
    }
    
    for key, default_val in defaults.items():
        if key not in adaptation_params:
            adaptation_params[key] = default_val
        else:
            if key.endswith('_step_size'):
                adaptation_params[key] = validate_positive_scalar(
                    adaptation_params[key], key
                )
            elif key == 'adaptation_rate':
                adaptation_params[key] = validate_probability(
                    adaptation_params[key], key
                )
            elif key == 'stability_threshold':
                adaptation_params[key] = validate_positive_scalar(
                    adaptation_params[key], key
                )
    
    # Check consistency
    if adaptation_params['min_step_size'] >= adaptation_params['max_step_size']:
        raise ValueError("min_step_size must be less than max_step_size")
    
    return adaptation_params


def robust_function_wrapper(func: Callable,
                          name: str,
                          fallback_value: Optional[Any] = None,
                          max_retries: int = 3) -> Callable:
    """
    Wrap function with error handling and retries.
    
    Args:
        func: Function to wrap
        name: Function name for error messages
        fallback_value: Value to return if all attempts fail
        max_retries: Maximum number of retry attempts
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                
                # Basic validation of result
                if result is not None:
                    if isinstance(result, np.ndarray):
                        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                            raise ValueError("Function returned NaN/infinite values")
                    elif np.isscalar(result):
                        if np.isnan(result) or np.isinf(result):
                            raise ValueError("Function returned NaN/infinite value")
                
                return result
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    warnings.warn(f"{name} attempt {attempt + 1} failed: {e}")
                    continue
        
        # All attempts failed
        if fallback_value is not None:
            warnings.warn(f"{name} failed after {max_retries} attempts, "
                         f"using fallback value: {last_exception}")
            return fallback_value
        else:
            raise RuntimeError(f"{name} failed after {max_retries} attempts: {last_exception}")
    
    return wrapper