"""
Unit tests for mathematical utility functions.
"""

import unittest
import numpy as np
import warnings
from scipy.linalg import LinAlgError

# Add the src directory to the path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.math_utils import (
    safe_cholesky,
    woodbury_matrix_identity,
    log_det_via_cholesky,
    multivariate_normal_logpdf,
    stable_logsumexp,
    matrix_sqrt_inv,
    condition_number,
    is_symmetric,
    nearest_positive_definite,
    gradcheck
)


class TestMathUtils(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Positive definite test matrix
        self.pd_matrix = np.array([[4.0, 2.0], [2.0, 3.0]])
        
        # Ill-conditioned but positive definite matrix
        self.ill_cond_matrix = np.array([[1.0, 0.0], [0.0, 1e-8]])
        
        # Non-positive definite matrix
        self.non_pd_matrix = np.array([[1.0, 2.0], [2.0, 1.0]])  # Singular
    
    def test_safe_cholesky_positive_definite(self):
        """Test Cholesky decomposition of positive definite matrix."""
        L = safe_cholesky(self.pd_matrix)
        
        # Check reconstruction
        reconstructed = L @ L.T
        np.testing.assert_allclose(reconstructed, self.pd_matrix, rtol=1e-10)
        
        # Check lower triangular
        self.assertTrue(np.allclose(np.triu(L, k=1), 0))
    
    def test_safe_cholesky_ill_conditioned(self):
        """Test Cholesky with ill-conditioned matrix."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            L = safe_cholesky(self.ill_cond_matrix)
            
            # Should issue regularization warning
            self.assertTrue(any("regularization" in str(warning.message) for warning in w))
        
        # Should still produce valid decomposition
        reconstructed = L @ L.T
        # Note: won't exactly match original due to regularization
        self.assertFalse(np.any(np.isnan(L)))
        self.assertFalse(np.any(np.isinf(L)))
    
    def test_safe_cholesky_non_positive_definite(self):
        """Test Cholesky with non-positive definite matrix."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            L = safe_cholesky(self.non_pd_matrix)
            
            # Should issue regularization warning
            self.assertTrue(any("regularization" in str(warning.message) for warning in w))
        
        # Should produce valid decomposition after regularization
        reconstructed = L @ L.T
        eigenvals = np.linalg.eigvals(reconstructed)
        self.assertTrue(np.all(eigenvals > 0))
    
    def test_safe_cholesky_non_symmetric(self):
        """Test Cholesky with non-symmetric matrix."""
        non_sym = np.array([[1.0, 2.0], [1.5, 3.0]])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            L = safe_cholesky(non_sym)
            
            # Should issue symmetry warning
            self.assertTrue(any("symmetric" in str(warning.message) for warning in w))
    
    def test_safe_cholesky_max_tries(self):
        """Test Cholesky max tries limit."""
        # Extremely bad matrix
        bad_matrix = np.array([[-1.0, 0.0], [0.0, -1.0]])
        
        with self.assertRaises(LinAlgError):
            safe_cholesky(bad_matrix, max_tries=2)
    
    def test_woodbury_matrix_identity_basic(self):
        """Test basic Woodbury matrix identity."""
        n, k = 3, 2
        A_inv = np.eye(n)
        U = np.random.randn(n, k)
        V = np.random.randn(n, k)
        
        # Compute using Woodbury identity
        result_woodbury = woodbury_matrix_identity(A_inv, U, V)
        
        # Compute directly
        A = np.eye(n)  # A_inv^(-1)
        update = U @ V.T
        result_direct = np.linalg.inv(A + update)
        
        np.testing.assert_allclose(result_woodbury, result_direct, rtol=1e-10)
    
    def test_woodbury_matrix_identity_rank_one(self):
        """Test Woodbury with rank-one update."""
        n = 4
        A_inv = 2.0 * np.eye(n)  # A = 0.5 * I
        u = np.random.randn(n)
        v = np.random.randn(n)
        
        U = u.reshape(-1, 1)
        V = v.reshape(-1, 1)
        
        result = woodbury_matrix_identity(A_inv, U, V)
        
        # Verify dimensions
        self.assertEqual(result.shape, (n, n))
        
        # Should be symmetric for this case
        if np.allclose(u, v):
            np.testing.assert_allclose(result, result.T, rtol=1e-10)
    
    def test_woodbury_input_validation(self):
        """Test input validation for Woodbury identity."""
        A_inv = np.eye(3)
        U = np.random.randn(3, 2)
        
        # Wrong V dimensions
        V_wrong = np.random.randn(2, 2)  # Should be (3, 2)
        with self.assertRaises(ValueError):
            woodbury_matrix_identity(A_inv, U, V_wrong)
        
        # Non-square A_inv
        A_inv_wrong = np.random.randn(3, 2)
        V = np.random.randn(3, 2)
        with self.assertRaises(ValueError):
            woodbury_matrix_identity(A_inv_wrong, U, V)
    
    def test_log_det_via_cholesky(self):
        """Test log determinant computation."""
        # Test with known matrix
        A = self.pd_matrix
        log_det_chol = log_det_via_cholesky(A)
        log_det_direct = np.log(np.linalg.det(A))
        
        np.testing.assert_allclose(log_det_chol, log_det_direct, rtol=1e-10)
    
    def test_log_det_large_determinant(self):
        """Test log determinant with large determinant."""
        # Create matrix with large determinant
        A = 100.0 * np.eye(3)
        log_det_chol = log_det_via_cholesky(A)
        
        # Should be 3 * log(100) = 3 * log(10^2) = 6 * log(10)
        expected = 3 * np.log(100.0)
        np.testing.assert_allclose(log_det_chol, expected, rtol=1e-10)
    
    def test_multivariate_normal_logpdf_standard(self):
        """Test multivariate normal log PDF with standard normal."""
        x = np.array([0.0, 0.0])
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        
        log_pdf = multivariate_normal_logpdf(x, mean, cov)
        
        # For standard 2D normal at origin: -0.5 * (2 * log(2π)) = -log(2π)
        expected = -np.log(2 * np.pi)
        np.testing.assert_allclose(log_pdf, expected, rtol=1e-10)
    
    def test_multivariate_normal_logpdf_general(self):
        """Test multivariate normal log PDF with general parameters."""
        x = np.array([1.0, 2.0])
        mean = np.array([0.5, 1.5])
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        
        log_pdf = multivariate_normal_logpdf(x, mean, cov)
        
        # Compare with scipy
        from scipy.stats import multivariate_normal
        expected = multivariate_normal.logpdf(x, mean, cov)
        
        np.testing.assert_allclose(log_pdf, expected, rtol=1e-10)
    
    def test_multivariate_normal_singular_cov(self):
        """Test multivariate normal with singular covariance."""
        x = np.array([1.0, 1.0])
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # Singular
        
        # Without allow_singular, should raise error
        with self.assertRaises(ValueError):
            multivariate_normal_logpdf(x, mean, cov, allow_singular=False)
        
        # With allow_singular, should work
        log_pdf = multivariate_normal_logpdf(x, mean, cov, allow_singular=True)
        self.assertFalse(np.isnan(log_pdf))
        self.assertFalse(np.isinf(log_pdf))
    
    def test_stable_logsumexp(self):
        """Test stable log-sum-exp computation."""
        # Simple case
        a = np.array([1.0, 2.0, 3.0])
        result = stable_logsumexp(a)
        expected = np.log(np.exp(1.0) + np.exp(2.0) + np.exp(3.0))
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        
        # Case with very large numbers (would overflow naive computation)
        a_large = np.array([1000.0, 1001.0, 1002.0])
        result_large = stable_logsumexp(a_large)
        # Should not be inf or nan
        self.assertFalse(np.isinf(result_large))
        self.assertFalse(np.isnan(result_large))
    
    def test_matrix_sqrt_inv_cholesky(self):
        """Test matrix square root inverse with Cholesky method."""
        A = self.pd_matrix
        A_sqrt_inv = matrix_sqrt_inv(A, method="cholesky")
        
        # Check: A_sqrt_inv @ A_sqrt_inv should equal A^(-1)
        A_inv = np.linalg.inv(A)
        reconstructed_inv = A_sqrt_inv.T @ A_sqrt_inv
        
        np.testing.assert_allclose(reconstructed_inv, A_inv, rtol=1e-10)
    
    def test_matrix_sqrt_inv_eigen(self):
        """Test matrix square root inverse with eigendecomposition method."""
        A = self.pd_matrix
        A_sqrt_inv = matrix_sqrt_inv(A, method="eigen")
        
        # Check: A^(1/2) @ A^(-1/2) should equal I
        A_sqrt = np.linalg.inv(A_sqrt_inv)
        should_be_A = A_sqrt @ A_sqrt
        
        np.testing.assert_allclose(should_be_A, A, rtol=1e-10)
    
    def test_matrix_sqrt_inv_svd(self):
        """Test matrix square root inverse with SVD method."""
        A = self.pd_matrix
        A_sqrt_inv = matrix_sqrt_inv(A, method="svd")
        
        # Basic check: should be square and same size
        self.assertEqual(A_sqrt_inv.shape, A.shape)
        
        # Should not contain NaN or inf
        self.assertFalse(np.any(np.isnan(A_sqrt_inv)))
        self.assertFalse(np.any(np.isinf(A_sqrt_inv)))
    
    def test_matrix_sqrt_inv_fallback(self):
        """Test matrix square root inverse method fallback."""
        # Use ill-conditioned matrix that might fail Cholesky
        A = self.ill_cond_matrix
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            A_sqrt_inv = matrix_sqrt_inv(A, method="cholesky")
            
            # Should warn about fallback
            self.assertTrue(any("fallback" in str(warning.message) for warning in w))
    
    def test_condition_number_well_conditioned(self):
        """Test condition number of well-conditioned matrix."""
        A = np.eye(3)
        cond = condition_number(A)
        self.assertAlmostEqual(cond, 1.0, places=10)
    
    def test_condition_number_ill_conditioned(self):
        """Test condition number of ill-conditioned matrix."""
        A = np.diag([1.0, 1e-6])
        cond = condition_number(A)
        expected = 1.0 / 1e-6
        self.assertAlmostEqual(cond, expected, places=5)
    
    def test_condition_number_singular(self):
        """Test condition number of singular matrix."""
        A = np.array([[1.0, 1.0], [1.0, 1.0]])
        cond = condition_number(A)
        self.assertEqual(cond, np.inf)
    
    def test_is_symmetric_true(self):
        """Test symmetry check on symmetric matrix."""
        A = np.array([[1.0, 2.0], [2.0, 3.0]])
        self.assertTrue(is_symmetric(A))
    
    def test_is_symmetric_false(self):
        """Test symmetry check on non-symmetric matrix."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertFalse(is_symmetric(A))
    
    def test_is_symmetric_nearly(self):
        """Test symmetry check with numerical tolerance."""
        A = np.array([[1.0, 2.0], [2.0 + 1e-15, 3.0]])
        
        # Should be symmetric with default tolerance
        self.assertTrue(is_symmetric(A))
        
        # Should not be symmetric with tighter tolerance
        self.assertFalse(is_symmetric(A, tol=1e-16))
    
    def test_is_symmetric_non_square(self):
        """Test symmetry check on non-square matrix."""
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertFalse(is_symmetric(A))
    
    def test_nearest_positive_definite_already_pd(self):
        """Test nearest PD on already positive definite matrix."""
        A = self.pd_matrix
        A_pd = nearest_positive_definite(A)
        
        # Should be very close to original
        np.testing.assert_allclose(A_pd, A, rtol=1e-10)
    
    def test_nearest_positive_definite_indefinite(self):
        """Test nearest PD on indefinite matrix."""
        A = np.array([[1.0, 0.0], [0.0, -1.0]])
        A_pd = nearest_positive_definite(A)
        
        # Result should be positive definite
        eigenvals = np.linalg.eigvals(A_pd)
        self.assertTrue(np.all(eigenvals > 0))
    
    def test_nearest_positive_definite_singular(self):
        """Test nearest PD on singular matrix."""
        A = self.non_pd_matrix  # Singular matrix
        A_pd = nearest_positive_definite(A)
        
        # Result should be positive definite
        eigenvals = np.linalg.eigvals(A_pd)
        self.assertTrue(np.all(eigenvals > 1e-15))
    
    def test_gradcheck_correct_gradient(self):
        """Test gradient check with correct gradient."""
        def func_with_grad(x):
            # f(x) = x^2, grad = 2x
            value = np.sum(x**2)
            grad = 2 * x
            return value, grad
        
        x = np.array([1.0, 2.0, 3.0])
        is_correct, max_error = gradcheck(func_with_grad, x)
        
        self.assertTrue(is_correct)
        self.assertLess(max_error, 1e-6)
    
    def test_gradcheck_incorrect_gradient(self):
        """Test gradient check with incorrect gradient."""
        def func_with_wrong_grad(x):
            # f(x) = x^2, but return wrong gradient
            value = np.sum(x**2)
            grad = x  # Wrong! Should be 2x
            return value, grad
        
        x = np.array([1.0, 2.0, 3.0])
        is_correct, max_error = gradcheck(func_with_wrong_grad, x)
        
        self.assertFalse(is_correct)
        self.assertGreater(max_error, 0.5)  # Should be significant error
    
    def test_input_type_validation(self):
        """Test input type validation across functions."""
        # Non-array input
        with self.assertRaises(TypeError):
            safe_cholesky("not an array")
        
        with self.assertRaises(TypeError):
            multivariate_normal_logpdf("not an array", np.zeros(2), np.eye(2))
        
        # Wrong dimensions
        with self.assertRaises(ValueError):
            log_det_via_cholesky(np.array([1, 2, 3]))  # Not 2D
        
        with self.assertRaises(ValueError):
            multivariate_normal_logpdf(np.zeros(2), np.zeros(3), np.eye(2))  # Inconsistent dims


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of mathematical operations."""
    
    def test_cholesky_near_singular(self):
        """Test Cholesky on nearly singular matrices."""
        eigenvals = [1.0, 1e-12]
        Q = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
        A = Q @ np.diag(eigenvals) @ Q.T
        
        # Should handle gracefully with regularization
        L = safe_cholesky(A)
        self.assertFalse(np.any(np.isnan(L)))
        self.assertFalse(np.any(np.isinf(L)))
    
    def test_log_det_large_matrix(self):
        """Test log determinant with large matrix."""
        n = 100
        A = 2.0 * np.eye(n)  # det(A) = 2^100, log(det(A)) = 100 * log(2)
        
        log_det = log_det_via_cholesky(A)
        expected = n * np.log(2.0)
        
        np.testing.assert_allclose(log_det, expected, rtol=1e-10)
    
    def test_matrix_sqrt_inv_extreme_condition(self):
        """Test matrix square root inverse with extreme condition numbers."""
        eigenvals = [1e6, 1e-6]
        Q = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
        A = Q @ np.diag(eigenvals) @ Q.T
        
        A_sqrt_inv = matrix_sqrt_inv(A, method="eigen")
        
        # Should not blow up
        self.assertFalse(np.any(np.isnan(A_sqrt_inv)))
        self.assertFalse(np.any(np.isinf(A_sqrt_inv)))
        
        # Check basic property: A^(-1/2) @ A^(-1/2) = A^(-1)
        A_inv_approx = A_sqrt_inv @ A_sqrt_inv.T
        A_inv_true = np.linalg.inv(A)
        
        # Should be reasonably close despite conditioning
        np.testing.assert_allclose(A_inv_approx, A_inv_true, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()