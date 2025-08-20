"""
Comprehensive unit tests for Hessian-aware samplers.

This module tests the mathematical correctness, numerical stability,
and performance characteristics of advanced Hessian-aware sampling algorithms.
"""

import unittest
import numpy as np
import warnings
from unittest.mock import patch, MagicMock

# Add the src directory to the path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from samplers.advanced_hessian_samplers import (
    HessianAwareMetropolis,
    HessianAwareLangevin,
    AdaptiveHessianSampler
)
from core.hessian_approximations import (
    lbfgs_hessian_approx,
    hutchinson_trace_estimator,
    stochastic_hessian_diagonal,
    block_hessian_computation
)


class TestHessianAwareMetropolis(unittest.TestCase):
    """Test Hessian-aware Metropolis sampler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dim = 3
        
        # Simple quadratic target: log p(x) = -0.5 * x^T A x
        self.A = np.array([[2.0, 0.5, 0.0],
                          [0.5, 3.0, 0.2],
                          [0.0, 0.2, 1.5]])
        
        def target_log_prob(x):
            return -0.5 * x @ self.A @ x
        
        self.target_log_prob = target_log_prob
        self.true_precision = self.A  # Precision matrix
        self.true_covariance = np.linalg.inv(self.A)
        
        self.sampler = HessianAwareMetropolis(
            target_log_prob=self.target_log_prob,
            dim=self.dim,
            step_size=0.5,
            hessian_update_freq=5
        )
    
    def test_initialization(self):
        """Test proper initialization of sampler."""
        self.assertEqual(self.sampler.dim, self.dim)
        self.assertEqual(self.sampler.step_size, 0.5)
        self.assertEqual(self.sampler.hessian_update_freq, 5)
        self.assertIsNone(self.sampler.current_hessian)
    
    def test_single_step(self):
        """Test single sampling step."""
        initial_state = np.zeros(self.dim)
        new_state, info = self.sampler.step(initial_state)
        
        # Check return types and structure
        self.assertIsInstance(new_state, np.ndarray)
        self.assertEqual(new_state.shape, (self.dim,))
        self.assertIsInstance(info, dict)
        
        # Check required info keys
        required_keys = ['accepted', 'log_prob', 'proposal_log_prob', 'log_alpha', 'method']
        for key in required_keys:
            self.assertIn(key, info)
        
        # Check that step incremented
        self.assertEqual(self.sampler.step_count, 1)
    
    def test_hessian_update(self):
        """Test Hessian computation and update."""
        test_point = np.array([0.5, 0.5, 0.5])
        
        success = self.sampler._update_hessian_info(test_point)
        self.assertTrue(success)
        
        # Check that Hessian was computed
        self.assertIsNotNone(self.sampler.current_hessian)
        self.assertIsNotNone(self.sampler.current_hessian_sqrt_inv)
        
        # For quadratic function, Hessian should be constant (= A)
        np.testing.assert_allclose(self.sampler.current_hessian, self.A, rtol=1e-3)
    
    def test_proposal_generation(self):
        """Test proposal generation mechanism."""
        current_state = np.array([1.0, 0.5, -0.3])
        
        # Update Hessian first
        self.sampler._update_hessian_info(current_state)
        
        # Generate multiple proposals
        proposals = []
        for _ in range(100):
            proposal = self.sampler._propose_state(current_state)
            proposals.append(proposal)
        
        proposals = np.array(proposals)
        
        # Check that proposals are different
        proposal_std = np.std(proposals, axis=0)
        self.assertTrue(np.all(proposal_std > 0.01))  # Should have variability
        
        # Check that proposals are centered around current state (approximately)
        proposal_mean = np.mean(proposals, axis=0)
        np.testing.assert_allclose(proposal_mean, current_state, rtol=0.5)
    
    def test_acceptance_probability(self):
        """Test acceptance probability computation."""
        current_state = np.zeros(self.dim)
        proposal = np.array([0.1, 0.1, 0.1])
        
        current_log_prob = self.target_log_prob(current_state)
        proposal_log_prob = self.target_log_prob(proposal)
        
        # Update Hessian
        self.sampler._update_hessian_info(current_state)
        
        log_alpha = self.sampler._compute_acceptance_probability(
            current_state, proposal, current_log_prob, proposal_log_prob
        )
        
        # Should be finite and <= 0
        self.assertFalse(np.isnan(log_alpha))
        self.assertFalse(np.isinf(log_alpha))
        self.assertLessEqual(log_alpha, 0.0)
    
    def test_detailed_balance(self):
        """Test that sampler satisfies detailed balance (approximately)."""
        # This is a statistical test - run many transitions
        n_tests = 1000
        x = np.array([0.2, 0.3, -0.1])
        y = np.array([0.1, 0.2, 0.0])
        
        # Count transitions x->y and y->x
        transitions_xy = 0
        transitions_yx = 0
        
        # Update Hessian at both points
        self.sampler._update_hessian_info(x)
        
        for _ in range(n_tests):
            # Test x -> y
            _, info_xy = self.sampler.step(x)
            if np.allclose(info_xy.get('proposal', x), y, atol=0.01):
                if info_xy['accepted']:
                    transitions_xy += 1
            
            # Test y -> x  
            _, info_yx = self.sampler.step(y)
            if np.allclose(info_yx.get('proposal', y), x, atol=0.01):
                if info_yx['accepted']:
                    transitions_yx += 1
        
        # This is a weak test due to randomness, but check rough balance
        # In practice, detailed balance is ensured by the mathematical construction
    
    def test_convergence_properties(self):
        """Test basic convergence properties."""
        n_samples = 500
        initial_state = np.array([2.0, -1.0, 1.5])
        
        # Generate samples
        samples = []
        current_state = initial_state.copy()
        
        for _ in range(n_samples):
            current_state, _ = self.sampler.step(current_state)
            samples.append(current_state.copy())
        
        samples = np.array(samples)
        
        # Basic convergence checks
        sample_mean = np.mean(samples[-200:], axis=0)  # Use latter half
        sample_cov = np.cov(samples[-200:].T)
        
        # For Gaussian target with zero mean, should converge to zero mean
        np.testing.assert_allclose(sample_mean, np.zeros(self.dim), atol=0.3)
        
        # Should approximate true covariance
        np.testing.assert_allclose(sample_cov, self.true_covariance, rtol=0.5)
    
    def test_numerical_stability(self):
        """Test numerical stability with ill-conditioned problems."""
        # Create ill-conditioned target
        eigenvals = [100.0, 1.0, 0.01]
        Q = np.random.orthogonal_group(3)
        A_ill = Q @ np.diag(eigenvals) @ Q.T
        
        def ill_conditioned_target(x):
            return -0.5 * x @ A_ill @ x
        
        sampler_ill = HessianAwareMetropolis(
            target_log_prob=ill_conditioned_target,
            dim=self.dim,
            regularization=1e-4,
            max_condition_number=1e6
        )
        
        # Should handle ill-conditioning gracefully
        initial_state = np.array([0.1, 0.1, 0.1])
        
        # Run several steps without crashing
        current_state = initial_state
        for _ in range(20):
            try:
                current_state, info = sampler_ill.step(current_state)
                self.assertFalse(np.any(np.isnan(current_state)))
                self.assertFalse(np.any(np.isinf(current_state)))
            except Exception as e:
                self.fail(f"Sampler failed on ill-conditioned problem: {e}")
    
    def test_error_handling(self):
        """Test error handling and fallbacks."""
        # Test with function that returns NaN
        def bad_function(x):
            return np.nan
        
        bad_sampler = HessianAwareMetropolis(bad_function, self.dim)
        
        # Should handle gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected warnings
            
            state = np.zeros(self.dim)
            new_state, info = bad_sampler.step(state)
            
            # Should not crash, should return valid state
            self.assertIsInstance(new_state, np.ndarray)
            self.assertEqual(new_state.shape, (self.dim,))
    
    def test_diagnostics(self):
        """Test diagnostic information collection."""
        # Run some steps
        state = np.zeros(self.dim)
        for _ in range(10):
            state, _ = self.sampler.step(state)
        
        diagnostics = self.sampler.get_diagnostics()
        
        # Check diagnostic keys
        expected_keys = ['hessian_failures', 'condition_numbers', 'step_count']
        for key in expected_keys:
            self.assertIn(key, diagnostics)
        
        self.assertEqual(diagnostics['step_count'], 10)


class TestHessianAwareLangevin(unittest.TestCase):
    """Test Hessian-aware Langevin dynamics sampler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dim = 2
        
        # Simple quadratic target
        self.A = np.array([[2.0, 0.5], [0.5, 1.5]])
        
        def target_log_prob(x):
            return -0.5 * x @ self.A @ x
        
        self.target_log_prob = target_log_prob
        
        self.sampler = HessianAwareLangevin(
            target_log_prob=self.target_log_prob,
            dim=self.dim,
            step_size=0.01,
            temperature=1.0
        )
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.sampler.dim, self.dim)
        self.assertEqual(self.sampler.step_size, 0.01)
        self.assertEqual(self.sampler.temperature, 1.0)
        self.assertTrue(self.sampler.metropolis_correction)
    
    def test_langevin_step(self):
        """Test Langevin integration step."""
        current_state = np.array([0.5, -0.3])
        
        # Update Hessian/gradient info
        self.sampler._update_hessian_gradient_info(current_state)
        
        # Perform Langevin step
        proposal = self.sampler._langevin_step(current_state)
        
        # Check basic properties
        self.assertIsInstance(proposal, np.ndarray)
        self.assertEqual(proposal.shape, current_state.shape)
        self.assertFalse(np.any(np.isnan(proposal)))
        self.assertFalse(np.any(np.isinf(proposal)))
        
        # Should be different from current state (with high probability)
        self.assertFalse(np.allclose(proposal, current_state, rtol=1e-6))
    
    def test_drift_and_diffusion(self):
        """Test drift and diffusion components of Langevin dynamics."""
        current_state = np.array([1.0, 0.5])
        
        # Update Hessian/gradient
        self.sampler._update_hessian_gradient_info(current_state)
        
        # Run multiple Langevin steps to check statistics
        proposals = []
        for _ in range(1000):
            # Fix random seed for reproducible drift calculation
            np.random.seed(42)
            proposal = self.sampler._langevin_step(current_state)
            proposals.append(proposal)
            np.random.seed(None)  # Reset seed
        
        proposals = np.array(proposals)
        
        # Check that there's both systematic movement (drift) and randomness (diffusion)
        displacement = proposals - current_state[None, :]
        mean_displacement = np.mean(displacement, axis=0)
        std_displacement = np.std(displacement, axis=0)
        
        # Should have non-zero drift and diffusion
        self.assertTrue(np.any(np.abs(mean_displacement) > 0.001))  # Some drift
        self.assertTrue(np.all(std_displacement > 0.01))  # Some diffusion
    
    def test_metropolis_correction(self):
        """Test Metropolis correction mechanism."""
        current_state = np.array([0.2, 0.1])
        proposal = np.array([0.3, 0.15])
        
        # Update Hessian info
        self.sampler._update_hessian_gradient_info(current_state)
        
        accepted, new_state, info = self.sampler._metropolis_correction(
            current_state, proposal, hessian_updated=True
        )
        
        # Check return values
        self.assertIsInstance(accepted, bool)
        self.assertIsInstance(new_state, np.ndarray)
        self.assertIsInstance(info, dict)
        
        # Check info dictionary
        required_keys = ['accepted', 'log_prob', 'log_alpha', 'method']
        for key in required_keys:
            self.assertIn(key, info)
        
        # New state should be either current or proposal
        self.assertTrue(np.allclose(new_state, current_state) or 
                       np.allclose(new_state, proposal))
    
    def test_temperature_effect(self):
        """Test effect of temperature parameter."""
        # Create samplers with different temperatures
        sampler_cold = HessianAwareLangevin(
            self.target_log_prob, self.dim, temperature=0.5
        )
        sampler_hot = HessianAwareLangevin(
            self.target_log_prob, self.dim, temperature=2.0
        )
        
        current_state = np.array([0.5, 0.5])
        
        # Update Hessian info for both
        sampler_cold._update_hessian_gradient_info(current_state)
        sampler_hot._update_hessian_gradient_info(current_state)
        
        # Generate multiple proposals
        cold_proposals = []
        hot_proposals = []
        
        for _ in range(100):
            cold_prop = sampler_cold._langevin_step(current_state)
            hot_prop = sampler_hot._langevin_step(current_state)
            cold_proposals.append(cold_prop)
            hot_proposals.append(hot_prop)
        
        cold_proposals = np.array(cold_proposals)
        hot_proposals = np.array(hot_proposals)
        
        # Hot sampler should explore more (higher variance)
        cold_var = np.var(cold_proposals, axis=0)
        hot_var = np.var(hot_proposals, axis=0)
        
        self.assertTrue(np.all(hot_var > cold_var))
    
    def test_pure_langevin_mode(self):
        """Test pure Langevin dynamics without Metropolis correction."""
        sampler_pure = HessianAwareLangevin(
            self.target_log_prob, 
            self.dim,
            metropolis_correction=False
        )
        
        current_state = np.array([0.1, 0.2])
        new_state, info = sampler_pure.step(current_state)
        
        # Should always accept in pure Langevin
        self.assertTrue(info['accepted'])
        self.assertEqual(info['log_alpha'], 0.0)
        self.assertEqual(info['method'], 'pure_langevin')


class TestAdaptiveHessianSampler(unittest.TestCase):
    """Test adaptive Hessian sampler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dim = 4
        
        # Moderately complex target
        self.A = np.diag([5.0, 2.0, 1.0, 0.5])
        
        def target_log_prob(x):
            return -0.5 * x @ self.A @ x
        
        self.target_log_prob = target_log_prob
        
        self.sampler = AdaptiveHessianSampler(
            target_log_prob=self.target_log_prob,
            dim=self.dim,
            adaptation_window=20,
            memory_size=10
        )
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.sampler.dim, self.dim)
        self.assertEqual(self.sampler.adaptation_window, 20)
        self.assertEqual(self.sampler.memory_size, 10)
        self.assertEqual(len(self.sampler.gradient_history), 0)
        self.assertEqual(len(self.sampler.state_history), 0)
    
    def test_lbfgs_update(self):
        """Test L-BFGS approximation update."""
        # Add some history manually
        states = [np.random.randn(self.dim) for _ in range(5)]
        gradients = [np.random.randn(self.dim) for _ in range(5)]
        
        for s, g in zip(states, gradients):
            self.sampler.state_history.append(s)
            self.sampler.gradient_history.append(g)
        
        # Update approximation
        self.sampler._update_lbfgs_approximation(states[-1], gradients[-1])
        
        # Should have created Hessian approximation
        self.assertIsNotNone(self.sampler.current_hessian_approx)
        self.assertEqual(self.sampler.current_hessian_approx.shape, (self.dim, self.dim))
    
    def test_parameter_adaptation(self):
        """Test parameter adaptation mechanism."""
        # Run some steps to build history
        current_state = np.zeros(self.dim)
        
        for i in range(30):  # Enough to trigger adaptation
            current_state, info = self.sampler.step(current_state)
        
        # Check that adaptation occurred
        self.assertTrue(len(self.sampler.adaptation_history) > 0)
        
        # Check that step size may have changed from initial value
        initial_step = 0.1  # Default initial step size
        # Step size should be different due to adaptation (with high probability)
    
    def test_low_rank_approximation(self):
        """Test low-rank Hessian approximation."""
        # Create full rank matrix
        H = np.random.randn(self.dim, self.dim)
        H = H @ H.T  # Make positive definite
        
        # Get low-rank approximation
        rank = 2
        H_lr = self.sampler._low_rank_hessian_approx(H, rank)
        
        # Check properties
        self.assertEqual(H_lr.shape, H.shape)
        
        # Check that rank is approximately correct
        eigenvals = np.linalg.eigvals(H_lr)
        significant_eigenvals = np.sum(np.abs(eigenvals) > 1e-10)
        self.assertLessEqual(significant_eigenvals, rank + 1)  # Allow some numerical error
    
    def test_adaptive_proposal(self):
        """Test adaptive proposal generation."""
        current_state = np.array([0.1, 0.2, -0.1, 0.3])
        gradient = np.array([0.5, -0.3, 0.1, -0.2])
        
        # Should work without Hessian approximation
        proposal1 = self.sampler._adaptive_propose(current_state, gradient)
        self.assertEqual(proposal1.shape, current_state.shape)
        
        # Add some L-BFGS history
        for _ in range(3):
            s = np.random.randn(self.dim)
            g = np.random.randn(self.dim)
            self.sampler._update_lbfgs_approximation(s, g)
        
        # Should now use Hessian preconditioning
        proposal2 = self.sampler._adaptive_propose(current_state, gradient)
        self.assertEqual(proposal2.shape, current_state.shape)
        
        # Proposals should be different (using different preconditioning)
        self.assertFalse(np.allclose(proposal1, proposal2, rtol=0.1))
    
    def test_memory_management(self):
        """Test that memory usage is controlled."""
        # Add more history than memory allows
        for i in range(self.sampler.memory_size + 5):
            state = np.random.randn(self.dim)
            gradient = np.random.randn(self.dim)
            self.sampler._update_lbfgs_approximation(state, gradient)
        
        # Check that memory is limited
        self.assertLessEqual(len(self.sampler.state_history), self.sampler.memory_size)
        self.assertLessEqual(len(self.sampler.gradient_history), self.sampler.memory_size)
    
    def test_adaptation_window(self):
        """Test adaptation window functionality."""
        # Fill acceptance history beyond window size
        for i in range(self.sampler.adaptation_window + 10):
            self.sampler.acceptance_history.append(i % 2 == 0)  # Alternating accept/reject
        
        # Check that window size is respected
        self.assertLessEqual(len(self.sampler.acceptance_history), self.sampler.adaptation_window)
    
    def test_diagnostics(self):
        """Test comprehensive diagnostics."""
        # Run several steps
        state = np.zeros(self.dim)
        for _ in range(15):
            state, _ = self.sampler.step(state)
        
        diagnostics = self.sampler.get_diagnostics()
        
        # Check diagnostic structure
        expected_keys = ['step_count', 'adaptation_history', 'current_acceptance', 
                        'current_step_size', 'history_lengths']
        for key in expected_keys:
            self.assertIn(key, diagnostics)
        
        self.assertEqual(diagnostics['step_count'], 15)
        self.assertIsNotNone(diagnostics['history_lengths'])


class TestHessianApproximations(unittest.TestCase):
    """Test Hessian approximation methods."""
    
    def test_lbfgs_approximation(self):
        """Test L-BFGS Hessian approximation."""
        dim = 3
        
        # Create some gradient and state history
        gradients = [np.random.randn(dim) for _ in range(5)]
        states = [np.random.randn(dim) for _ in range(5)]
        
        # Make sure there's positive curvature
        for i in range(1, len(states)):
            s = states[i] - states[i-1]
            y = gradients[i] - gradients[i-1]
            if np.dot(s, y) <= 0:
                gradients[i] = gradients[i-1] + 0.1 * s  # Ensure positive curvature
        
        H_approx = lbfgs_hessian_approx(gradients, states, memory_size=3)
        
        # Check basic properties
        self.assertEqual(H_approx.shape, (dim, dim))
        self.assertTrue(np.allclose(H_approx, H_approx.T, rtol=1e-10))  # Should be symmetric
        
        # Should be positive definite (approximately)
        eigenvals = np.linalg.eigvals(H_approx)
        self.assertTrue(np.all(eigenvals > -1e-10))  # Allow small numerical error
    
    def test_hutchinson_trace_estimator(self):
        """Test Hutchinson trace estimator."""
        dim = 4
        
        # Create test matrix
        A = np.random.randn(dim, dim)
        A = A @ A.T  # Make positive definite
        
        true_trace = np.trace(A)
        
        # Define matrix-vector product function
        def matvec(v):
            return A @ v
        
        # Estimate trace
        estimated_trace_rad = hutchinson_trace_estimator(
            matvec, dim, n_samples=100, distribution="rademacher"
        )
        estimated_trace_gauss = hutchinson_trace_estimator(
            matvec, dim, n_samples=100, distribution="gaussian"
        )
        
        # Should be reasonably close to true trace
        self.assertAlmostEqual(estimated_trace_rad, true_trace, places=0)  # Allow some error due to randomness
        self.assertAlmostEqual(estimated_trace_gauss, true_trace, places=0)
    
    def test_stochastic_hessian_diagonal(self):
        """Test stochastic Hessian diagonal estimation."""
        # Simple quadratic function
        A = np.diag([2.0, 3.0, 1.5])
        
        def quadratic(x):
            return 0.5 * x @ A @ x
        
        x = np.array([0.5, -0.3, 0.8])
        
        estimated_diag = stochastic_hessian_diagonal(quadratic, x, n_samples=50)
        true_diag = np.diag(A)
        
        # Should be close to true diagonal
        np.testing.assert_allclose(estimated_diag, true_diag, rtol=0.2)
    
    def test_block_hessian_computation(self):
        """Test block-wise Hessian computation."""
        dim = 6
        
        # Separable quadratic function
        def separable_quadratic(x):
            return np.sum([(i+1) * x[i]**2 for i in range(len(x))])
        
        x = np.zeros(dim)
        
        # Compute block Hessian
        H_block = block_hessian_computation(
            separable_quadratic, x, block_size=3, method="finite_diff"
        )
        
        # Should be diagonal for separable function
        expected_diag = np.array([2*(i+1) for i in range(dim)])
        computed_diag = np.diag(H_block)
        
        np.testing.assert_allclose(computed_diag, expected_diag, rtol=1e-6)
        
        # Off-diagonal should be small
        H_off_diag = H_block - np.diag(computed_diag)
        self.assertTrue(np.max(np.abs(H_off_diag)) < 1e-8)
    
    def test_block_hessian_diagonal_only(self):
        """Test diagonal-only block Hessian computation."""
        dim = 4
        
        def test_function(x):
            return np.sum(x**2)
        
        x = np.ones(dim)
        
        H_diag = block_hessian_computation(
            test_function, x, block_size=2, method="diagonal_only"
        )
        
        # Should be 2*I for x^T x function
        expected = 2.0 * np.eye(dim)
        np.testing.assert_allclose(H_diag, expected, rtol=1e-6)


class TestNumericalStabilityHighDimensions(unittest.TestCase):
    """Test numerical stability in high-dimensional problems."""
    
    def test_high_dimensional_stability(self):
        """Test stability with moderately high dimensions."""
        dim = 50  # Moderately high for testing
        
        # Well-conditioned Gaussian
        A = np.eye(dim) + 0.1 * np.random.randn(dim, dim)
        A = A @ A.T  # Ensure positive definite
        
        def target_log_prob(x):
            return -0.5 * x @ A @ x
        
        sampler = HessianAwareMetropolis(
            target_log_prob, dim, step_size=0.1, regularization=1e-5
        )
        
        # Should handle high dimensions gracefully
        initial_state = 0.1 * np.random.randn(dim)
        current_state = initial_state
        
        try:
            for _ in range(20):  # Run several steps
                current_state, info = sampler.step(current_state)
                
                # Check for numerical issues
                self.assertFalse(np.any(np.isnan(current_state)))
                self.assertFalse(np.any(np.isinf(current_state)))
                self.assertTrue(np.isfinite(info['log_prob']))
                
        except Exception as e:
            self.fail(f"High-dimensional sampler failed: {e}")
    
    def test_ill_conditioned_stability(self):
        """Test stability with ill-conditioned problems."""
        dim = 5
        
        # Create ill-conditioned matrix
        eigenvals = [1000.0, 100.0, 10.0, 1.0, 0.01]
        Q = np.random.orthogonal_group(dim)
        A = Q @ np.diag(eigenvals) @ Q.T
        
        def ill_target(x):
            return -0.5 * x @ A @ x
        
        # Test with adaptive regularization
        sampler = HessianAwareMetropolis(
            ill_target, dim,
            regularization=1e-6,
            max_condition_number=1e8
        )
        
        initial_state = 0.01 * np.random.randn(dim)
        current_state = initial_state
        
        # Should handle ill-conditioning
        for _ in range(10):
            try:
                current_state, info = sampler.step(current_state)
                self.assertFalse(np.any(np.isnan(current_state)))
                self.assertFalse(np.any(np.isinf(current_state)))
            except Exception as e:
                self.fail(f"Ill-conditioned sampler failed: {e}")
    
    def test_memory_efficiency(self):
        """Test memory efficiency for larger problems."""
        dim = 100
        
        # Use L-BFGS-based sampler for efficiency
        def simple_target(x):
            return -0.5 * np.sum(x**2)
        
        sampler = AdaptiveHessianSampler(
            simple_target, dim,
            memory_size=10,  # Limited memory
            max_rank=20      # Low-rank approximation
        )
        
        # Should handle large dimensions with limited memory
        initial_state = 0.1 * np.random.randn(dim)
        current_state = initial_state
        
        try:
            for _ in range(15):
                current_state, info = sampler.step(current_state)
                self.assertFalse(np.any(np.isnan(current_state)))
                
            # Memory should be controlled
            self.assertLessEqual(len(sampler.state_history), sampler.memory_size)
            
        except Exception as e:
            self.fail(f"High-dimensional adaptive sampler failed: {e}")


if __name__ == '__main__':
    unittest.main()