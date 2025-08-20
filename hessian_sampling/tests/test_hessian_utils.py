"""
Unit tests for Hessian utility functions.
"""

import unittest
import numpy as np
import warnings
from unittest.mock import patch, MagicMock

# Add the src directory to the path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.hessian_utils import (
    compute_hessian_autodiff,
    compute_hessian_finite_diff,
    hessian_eigendecomposition,
    condition_hessian,
    hessian_condition_number,
    is_positive_definite
)


class TestHessianUtils(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.dim = 3
        self.test_point = np.array([1.0, 2.0, 3.0])
        
        # Simple quadratic function: f(x) = x^T A x + b^T x + c
        self.A = np.array([[2.0, 0.5, 0.0],
                          [0.5, 3.0, 0.2],
                          [0.0, 0.2, 1.5]])
        self.b = np.array([1.0, -0.5, 2.0])
        self.c = 5.0
        
        def quadratic_function(x):
            return x @ self.A @ x + self.b @ x + self.c
        
        self.quadratic_func = quadratic_function
        self.true_hessian = 2 * self.A  # Hessian of quadratic is 2*A
    
    def test_finite_diff_hessian_quadratic(self):
        """Test finite difference Hessian on quadratic function."""
        H = compute_hessian_finite_diff(self.quadratic_func, self.test_point)
        
        # Should be close to true Hessian
        np.testing.assert_allclose(H, self.true_hessian, rtol=1e-4, atol=1e-6)
        
        # Should be symmetric
        np.testing.assert_allclose(H, H.T, rtol=1e-10)
    
    def test_finite_diff_methods(self):
        """Test different finite difference methods."""
        H_central = compute_hessian_finite_diff(
            self.quadratic_func, self.test_point, method="central"
        )
        H_forward = compute_hessian_finite_diff(
            self.quadratic_func, self.test_point, method="forward"
        )
        
        # Central differences should be more accurate
        error_central = np.linalg.norm(H_central - self.true_hessian)
        error_forward = np.linalg.norm(H_forward - self.true_hessian)
        
        self.assertLess(error_central, error_forward)
    
    def test_finite_diff_step_size(self):
        """Test effect of step size on finite differences."""
        eps_values = [1e-4, 1e-6, 1e-8]
        errors = []
        
        for eps in eps_values:
            H = compute_hessian_finite_diff(self.quadratic_func, self.test_point, eps=eps)
            error = np.linalg.norm(H - self.true_hessian)
            errors.append(error)
        
        # Should have optimal step size around 1e-6
        optimal_idx = np.argmin(errors)
        self.assertEqual(optimal_idx, 1)  # 1e-6 should be best
    
    @patch('core.hessian_utils.HAS_JAX', True)
    def test_autodiff_hessian_mock(self):
        """Test autodiff Hessian with mocked JAX."""
        # Mock JAX functions
        mock_hessian_func = MagicMock()
        mock_hessian_func.return_value = self.true_hessian
        
        with patch('core.hessian_utils.hessian', mock_hessian_func):
            with patch('core.hessian_utils.jnp') as mock_jnp:
                mock_jnp.array.return_value = self.test_point
                
                H = compute_hessian_autodiff(self.quadratic_func, self.test_point)
                np.testing.assert_allclose(H, self.true_hessian)
    
    def test_hessian_eigendecomposition(self):
        """Test eigendecomposition of Hessian matrices."""
        # Positive definite matrix
        H_pd = np.array([[2.0, 0.5], [0.5, 3.0]])
        eigenvals, eigenvecs = hessian_eigendecomposition(H_pd)
        
        # Check reconstruction
        H_reconstructed = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        np.testing.assert_allclose(H_pd, H_reconstructed, rtol=1e-10)
        
        # Eigenvalues should be in descending order
        self.assertTrue(np.all(eigenvals[:-1] >= eigenvals[1:]))
    
    def test_eigendecomposition_edge_cases(self):
        """Test eigendecomposition edge cases."""
        # Non-symmetric matrix (should be symmetrized)
        H_nonsym = np.array([[1.0, 2.0], [1.5, 3.0]])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eigenvals, eigenvecs = hessian_eigendecomposition(H_nonsym)
            self.assertTrue(any("not symmetric" in str(warning.message) for warning in w))
        
        # Singular matrix
        H_singular = np.array([[1.0, 1.0], [1.0, 1.0]])
        eigenvals, eigenvecs = hessian_eigendecomposition(H_singular)
        
        # One eigenvalue should be zero (or very small)
        self.assertTrue(np.any(np.abs(eigenvals) < 1e-10))
    
    def test_condition_hessian_positive_definite(self):
        """Test conditioning of already positive definite matrix."""
        H_pd = np.array([[2.0, 0.5], [0.5, 3.0]])
        H_conditioned = condition_hessian(H_pd, min_eigenval=0.1)
        
        # Should remain unchanged (already positive definite)
        np.testing.assert_allclose(H_pd, H_conditioned, rtol=1e-10)
    
    def test_condition_hessian_needs_regularization(self):
        """Test conditioning of matrix needing regularization."""
        # Matrix with small eigenvalue
        H = np.array([[1.0, 0.0], [0.0, 1e-8]])
        min_eigenval = 1e-6
        
        H_conditioned = condition_hessian(H, min_eigenval=min_eigenval)
        
        # All eigenvalues should be above threshold
        eigenvals = np.linalg.eigvals(H_conditioned)
        self.assertTrue(np.all(eigenvals >= min_eigenval - 1e-12))
    
    def test_condition_hessian_regularization_methods(self):
        """Test different regularization methods."""
        H = np.array([[1.0, 0.0], [0.0, 1e-8]])
        min_eigenval = 1e-6
        
        # Identity regularization
        H_identity = condition_hessian(H, min_eigenval, regularization="identity")
        eigenvals_identity = np.linalg.eigvals(H_identity)
        
        # Diagonal regularization  
        H_diagonal = condition_hessian(H, min_eigenval, regularization="diagonal")
        eigenvals_diagonal = np.linalg.eigvals(H_diagonal)
        
        # Both should satisfy minimum eigenvalue constraint
        self.assertTrue(np.all(eigenvals_identity >= min_eigenval - 1e-12))
        self.assertTrue(np.all(eigenvals_diagonal >= min_eigenval - 1e-12))
    
    def test_hessian_condition_number(self):
        """Test condition number computation."""
        # Well-conditioned matrix
        H_good = np.eye(3)
        cond_good = hessian_condition_number(H_good)
        self.assertAlmostEqual(cond_good, 1.0, places=10)
        
        # Ill-conditioned matrix
        H_bad = np.diag([100.0, 1.0, 0.01])
        cond_bad = hessian_condition_number(H_bad)
        self.assertAlmostEqual(cond_bad, 10000.0, places=8)
        
        # Matrix with negative eigenvalues
        H_neg = np.diag([1.0, -0.5, 0.1])
        cond_neg = hessian_condition_number(H_neg)
        # Should only consider positive eigenvalues
        self.assertAlmostEqual(cond_neg, 10.0, places=8)
    
    def test_is_positive_definite(self):
        """Test positive definiteness check."""
        # Positive definite matrix
        H_pd = np.array([[2.0, 0.5], [0.5, 3.0]])
        self.assertTrue(is_positive_definite(H_pd))
        
        # Positive semi-definite matrix
        H_psd = np.array([[1.0, 1.0], [1.0, 1.0]])
        self.assertFalse(is_positive_definite(H_psd))
        
        # Indefinite matrix
        H_indef = np.array([[1.0, 0.0], [0.0, -1.0]])
        self.assertFalse(is_positive_definite(H_indef))
        
        # Test with tolerance
        H_near_psd = np.array([[1.0, 0.0], [0.0, 1e-15]])
        self.assertFalse(is_positive_definite(H_near_psd, tol=1e-12))
        self.assertTrue(is_positive_definite(H_near_psd, tol=1e-16))
    
    def test_input_validation(self):
        """Test input validation for all functions."""
        # Non-callable function
        with self.assertRaises(TypeError):
            compute_hessian_finite_diff("not_callable", self.test_point)
        
        # Wrong input dimensions
        with self.assertRaises(ValueError):
            compute_hessian_finite_diff(self.quadratic_func, np.array([[1, 2], [3, 4]]))
        
        # Non-square matrix for eigendecomposition
        with self.assertRaises(ValueError):
            hessian_eigendecomposition(np.array([[1, 2, 3], [4, 5, 6]]))
        
        # Invalid regularization method
        H = np.eye(2)
        with self.assertRaises(ValueError):
            condition_hessian(H, regularization="invalid")
        
        # Negative min_eigenval
        with self.assertRaises(ValueError):
            condition_hessian(H, min_eigenval=-1.0)
    
    def test_numerical_edge_cases(self):
        """Test numerical edge cases."""
        # Very large function values
        def large_function(x):
            return 1e20 * np.sum(x**2)
        
        H = compute_hessian_finite_diff(large_function, np.array([1.0, 2.0]))
        self.assertFalse(np.any(np.isnan(H)))
        self.assertFalse(np.any(np.isinf(H)))
        
        # Function returning NaN
        def nan_function(x):
            return np.nan
        
        with self.assertRaises(ValueError):
            compute_hessian_finite_diff(nan_function, np.array([1.0]))
    
    def test_rosenbrock_function(self):
        """Test on Rosenbrock function - a challenging case."""
        def rosenbrock(x):
            return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
        
        x = np.array([0.5, 0.5])
        H = compute_hessian_finite_diff(rosenbrock, x)
        
        # Check that Hessian is computed (basic sanity check)
        self.assertEqual(H.shape, (2, 2))
        self.assertFalse(np.any(np.isnan(H)))
        
        # Rosenbrock Hessian should be symmetric
        np.testing.assert_allclose(H, H.T, rtol=1e-8)
        
        # At x=[0.5, 0.5], the function is highly curved
        # Condition number should be large
        cond_num = hessian_condition_number(H)
        self.assertGreater(cond_num, 10.0)


class TestSpecialCases(unittest.TestCase):
    """Test special mathematical cases."""
    
    def test_zero_function(self):
        """Test Hessian of zero function."""
        def zero_func(x):
            return 0.0
        
        x = np.array([1.0, 2.0])
        H = compute_hessian_finite_diff(zero_func, x)
        
        # Hessian should be zero matrix
        np.testing.assert_allclose(H, np.zeros((2, 2)), atol=1e-10)
    
    def test_linear_function(self):
        """Test Hessian of linear function."""
        def linear_func(x):
            return 2*x[0] + 3*x[1] + 5
        
        x = np.array([1.0, 2.0])
        H = compute_hessian_finite_diff(linear_func, x)
        
        # Hessian of linear function should be zero
        np.testing.assert_allclose(H, np.zeros((2, 2)), atol=1e-8)
    
    def test_separable_function(self):
        """Test Hessian of separable function."""
        def separable_func(x):
            return x[0]**2 + 2*x[1]**2 + 3*x[2]**2
        
        x = np.array([1.0, 2.0, 3.0])
        H = compute_hessian_finite_diff(separable_func, x)
        
        # Hessian should be diagonal
        expected_H = np.diag([2.0, 4.0, 6.0])
        np.testing.assert_allclose(H, expected_H, rtol=1e-6)
    
    def test_high_dimensional(self):
        """Test on higher dimensional problems."""
        dim = 10
        
        def high_dim_quadratic(x):
            A = np.eye(dim)
            return x @ A @ x
        
        x = np.random.randn(dim)
        H = compute_hessian_finite_diff(high_dim_quadratic, x)
        
        # Should be close to 2*I
        expected_H = 2 * np.eye(dim)
        np.testing.assert_allclose(H, expected_H, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()