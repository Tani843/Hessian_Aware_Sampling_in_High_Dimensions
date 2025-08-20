"""
Unit tests for the base sampler class.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Add the src directory to the path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.sampling_base import BaseSampler, SamplingResults


class MockSampler(BaseSampler):
    """Mock implementation of BaseSampler for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_count = 0
        self.force_acceptance = None  # For controlling acceptance in tests
    
    def step(self, current_state):
        """Mock step function."""
        self.step_count += 1
        
        # Simple random walk proposal
        proposal = current_state + 0.1 * np.random.randn(self.dim)
        
        # Compute log probabilities
        current_log_prob = self.target_log_prob(current_state)
        proposal_log_prob = self.target_log_prob(proposal)
        
        # Accept/reject
        if self.force_acceptance is not None:
            accepted = self.force_acceptance
        else:
            log_alpha = min(0.0, proposal_log_prob - current_log_prob)
            accepted = np.log(np.random.rand()) < log_alpha
        
        if accepted:
            new_state = proposal
            new_log_prob = proposal_log_prob
        else:
            new_state = current_state
            new_log_prob = current_log_prob
        
        return new_state, {
            'accepted': accepted,
            'log_prob': new_log_prob,
            'proposal_log_prob': proposal_log_prob
        }


class TestBaseSampler(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.dim = 3
        
        # Simple quadratic target distribution
        def target_log_prob(x):
            return -0.5 * np.sum(x**2)
        
        self.target_log_prob = target_log_prob
        self.initial_state = np.zeros(self.dim)
        
        # Create mock sampler
        self.sampler = MockSampler(
            target_log_prob=self.target_log_prob,
            dim=self.dim,
            step_size=0.1
        )
    
    def test_initialization_valid(self):
        """Test valid initialization."""
        sampler = MockSampler(
            target_log_prob=self.target_log_prob,
            dim=5,
            step_size=0.5,
            target_acceptance=0.6,
            adapt_step_size=True
        )
        
        self.assertEqual(sampler.dim, 5)
        self.assertEqual(sampler.step_size, 0.5)
        self.assertEqual(sampler.target_acceptance, 0.6)
        self.assertTrue(sampler.adapt_step_size)
    
    def test_initialization_invalid_params(self):
        """Test initialization with invalid parameters."""
        # Non-callable target function
        with self.assertRaises(TypeError):
            MockSampler("not_callable", self.dim)
        
        # Invalid dimension
        with self.assertRaises(ValueError):
            MockSampler(self.target_log_prob, 0)
        
        with self.assertRaises(ValueError):
            MockSampler(self.target_log_prob, -1)
        
        # Invalid step size
        with self.assertRaises(ValueError):
            MockSampler(self.target_log_prob, self.dim, step_size=0.0)
        
        with self.assertRaises(ValueError):
            MockSampler(self.target_log_prob, self.dim, step_size=-0.1)
        
        # Invalid target acceptance
        with self.assertRaises(ValueError):
            MockSampler(self.target_log_prob, self.dim, target_acceptance=0.0)
        
        with self.assertRaises(ValueError):
            MockSampler(self.target_log_prob, self.dim, target_acceptance=1.0)
        
        with self.assertRaises(ValueError):
            MockSampler(self.target_log_prob, self.dim, target_acceptance=1.5)
    
    def test_single_step(self):
        """Test single sampling step."""
        current_state = np.array([1.0, 2.0, 3.0])
        new_state, info = self.sampler.step(current_state)
        
        # Check return types
        self.assertIsInstance(new_state, np.ndarray)
        self.assertIsInstance(info, dict)
        
        # Check required info keys
        self.assertIn('accepted', info)
        self.assertIn('log_prob', info)
        
        # Check dimensions
        self.assertEqual(new_state.shape, current_state.shape)
        
        # Check that step count increased
        self.assertEqual(self.sampler.step_count, 1)
    
    def test_sample_basic(self):
        """Test basic sampling functionality."""
        n_samples = 100
        results = self.sampler.sample(
            n_samples=n_samples,
            initial_state=self.initial_state,
            burnin=0,
            thin=1,
            return_diagnostics=False
        )
        
        # Check results type and attributes
        self.assertIsInstance(results, SamplingResults)
        self.assertEqual(results.samples.shape, (n_samples, self.dim))
        self.assertEqual(results.log_probs.shape, (n_samples,))
        self.assertGreaterEqual(results.acceptance_rate, 0.0)
        self.assertLessEqual(results.acceptance_rate, 1.0)
        self.assertEqual(results.n_samples, n_samples)
        self.assertGreater(results.sampling_time, 0.0)
    
    def test_sample_with_burnin(self):
        """Test sampling with burnin period."""
        n_samples = 50
        burnin = 100
        
        results = self.sampler.sample(
            n_samples=n_samples,
            initial_state=self.initial_state,
            burnin=burnin,
            return_diagnostics=False
        )
        
        # Should still get n_samples after burnin
        self.assertEqual(results.samples.shape[0], n_samples)
        
        # Total steps should be burnin + n_samples
        expected_total_steps = burnin + n_samples
        self.assertEqual(self.sampler.n_proposals, expected_total_steps)
    
    def test_sample_with_thinning(self):
        """Test sampling with thinning."""
        n_samples = 50
        thin = 3
        
        results = self.sampler.sample(
            n_samples=n_samples,
            initial_state=self.initial_state,
            burnin=0,
            thin=thin,
            return_diagnostics=False
        )
        
        # Should get n_samples despite thinning
        self.assertEqual(results.samples.shape[0], n_samples)
        
        # Total steps should be n_samples * thin
        expected_total_steps = n_samples * thin
        self.assertEqual(self.sampler.n_proposals, expected_total_steps)
    
    def test_sample_with_diagnostics(self):
        """Test sampling with diagnostics enabled."""
        n_samples = 100
        
        results = self.sampler.sample(
            n_samples=n_samples,
            initial_state=self.initial_state,
            return_diagnostics=True
        )
        
        # Check that diagnostics are computed
        self.assertIsNotNone(results.effective_sample_size)
        self.assertIsNotNone(results.autocorr_time)
        self.assertIsNotNone(results.diagnostics)
        
        # Check diagnostics content
        self.assertIn('mean', results.diagnostics)
        self.assertIn('std', results.diagnostics)
        self.assertIn('quantiles', results.diagnostics)
    
    def test_sample_invalid_params(self):
        """Test sampling with invalid parameters."""
        # Invalid n_samples
        with self.assertRaises(ValueError):
            self.sampler.sample(0, self.initial_state)
        
        with self.assertRaises(ValueError):
            self.sampler.sample(-10, self.initial_state)
        
        # Invalid initial state type
        with self.assertRaises(TypeError):
            self.sampler.sample(100, "not_array")
        
        # Invalid initial state shape
        wrong_shape = np.zeros(self.dim + 1)
        with self.assertRaises(ValueError):
            self.sampler.sample(100, wrong_shape)
        
        # Invalid burnin
        with self.assertRaises(ValueError):
            self.sampler.sample(100, self.initial_state, burnin=-10)
        
        # Invalid thin
        with self.assertRaises(ValueError):
            self.sampler.sample(100, self.initial_state, thin=0)
    
    def test_step_size_adaptation(self):
        """Test step size adaptation functionality."""
        sampler = MockSampler(
            target_log_prob=self.target_log_prob,
            dim=self.dim,
            adapt_step_size=True,
            target_acceptance=0.5
        )
        
        initial_step_size = sampler.step_size
        
        # Force high acceptance to test adaptation
        sampler.force_acceptance = True
        sampler.n_accepted = 90
        sampler.n_proposals = 100
        
        sampler._adapt_step_size()
        
        # Step size should increase (high acceptance rate)
        self.assertGreater(sampler.step_size, initial_step_size)
        
        # Reset and test low acceptance
        sampler.step_size = initial_step_size
        sampler.n_accepted = 10
        sampler.n_proposals = 100
        
        sampler._adapt_step_size()
        
        # Step size should decrease (low acceptance rate)
        self.assertLess(sampler.step_size, initial_step_size)
    
    def test_step_size_bounds(self):
        """Test step size adaptation respects bounds."""
        sampler = MockSampler(
            target_log_prob=self.target_log_prob,
            dim=self.dim,
            step_size=0.1,
            min_step_size=0.01,
            max_step_size=1.0
        )
        
        # Force very low acceptance to drive step size down
        sampler.n_accepted = 0
        sampler.n_proposals = 100
        
        for _ in range(20):  # Multiple adaptations
            sampler._adapt_step_size()
        
        # Should respect minimum bound
        self.assertGreaterEqual(sampler.step_size, sampler.min_step_size)
        
        # Reset and force very high acceptance
        sampler.step_size = 0.1
        sampler.n_accepted = 100
        sampler.n_proposals = 100
        
        for _ in range(20):
            sampler._adapt_step_size()
        
        # Should respect maximum bound
        self.assertLessEqual(sampler.step_size, sampler.max_step_size)
    
    def test_acceptance_rate_tracking(self):
        """Test acceptance rate tracking."""
        self.sampler.n_accepted = 75
        self.sampler.n_proposals = 100
        
        acceptance_rate = self.sampler.get_acceptance_rate()
        self.assertAlmostEqual(acceptance_rate, 0.75)
        
        # Test edge case with no proposals
        sampler_empty = MockSampler(self.target_log_prob, self.dim)
        self.assertEqual(sampler_empty.get_acceptance_rate(), 0.0)
    
    def test_reset_stats(self):
        """Test statistics reset functionality."""
        self.sampler.n_accepted = 50
        self.sampler.n_proposals = 100
        self.sampler.step_size_history = [0.1, 0.2, 0.3]
        
        self.sampler.reset_stats()
        
        self.assertEqual(self.sampler.n_accepted, 0)
        self.assertEqual(self.sampler.n_proposals, 0)
        self.assertEqual(len(self.sampler.step_size_history), 0)
    
    def test_set_step_size(self):
        """Test step size setting with validation."""
        # Valid step size
        self.sampler.set_step_size(0.5)
        self.assertEqual(self.sampler.step_size, 0.5)
        
        # Invalid step size
        with self.assertRaises(ValueError):
            self.sampler.set_step_size(0.0)
        
        with self.assertRaises(ValueError):
            self.sampler.set_step_size(-0.1)
        
        # Step size exceeding bounds should be clipped
        sampler = MockSampler(
            target_log_prob=self.target_log_prob,
            dim=self.dim,
            min_step_size=0.01,
            max_step_size=1.0
        )
        
        sampler.set_step_size(2.0)  # Above max
        self.assertEqual(sampler.step_size, 1.0)
        
        sampler.set_step_size(0.001)  # Below min
        self.assertEqual(sampler.step_size, 0.01)
    
    def test_warmup(self):
        """Test warmup functionality."""
        n_warmup = 50
        initial_state = np.array([1.0, 2.0, 3.0])
        
        final_state = self.sampler.warmup(initial_state, n_warmup)
        
        # Should return valid state
        self.assertIsInstance(final_state, np.ndarray)
        self.assertEqual(final_state.shape, initial_state.shape)
        
        # Should have run n_warmup steps
        self.assertEqual(self.sampler.n_proposals, n_warmup)
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        callback_calls = []
        
        def progress_callback(iteration, total):
            callback_calls.append((iteration, total))
        
        # Sample with callback
        results = self.sampler.sample(
            n_samples=50,
            initial_state=self.initial_state,
            burnin=50,
            progress_callback=progress_callback,
            return_diagnostics=False
        )
        
        # Should have made some callback calls
        self.assertGreater(len(callback_calls), 0)
        
        # Check callback arguments
        for iteration, total in callback_calls:
            self.assertIsInstance(iteration, int)
            self.assertIsInstance(total, int)
            self.assertLessEqual(iteration, total)
    
    def test_keyboard_interrupt_handling(self):
        """Test handling of keyboard interrupt during sampling."""
        # Mock the step function to raise KeyboardInterrupt
        def interrupt_step(current_state):
            raise KeyboardInterrupt()
        
        with patch.object(self.sampler, 'step', side_effect=interrupt_step):
            with patch('builtins.input', return_value=''):  # Mock input for warnings
                results = self.sampler.sample(
                    n_samples=100,
                    initial_state=self.initial_state,
                    return_diagnostics=False
                )
        
        # Should return partial results
        self.assertIsInstance(results, SamplingResults)
        self.assertGreaterEqual(results.samples.shape[0], 0)


class TestSamplingResults(unittest.TestCase):
    """Test SamplingResults dataclass functionality."""
    
    def test_sampling_results_creation(self):
        """Test creation of SamplingResults object."""
        samples = np.random.randn(100, 5)
        log_probs = np.random.randn(100)
        
        results = SamplingResults(
            samples=samples,
            log_probs=log_probs,
            acceptance_rate=0.65,
            n_samples=100,
            sampling_time=10.5
        )
        
        self.assertEqual(results.samples.shape, (100, 5))
        self.assertEqual(results.log_probs.shape, (100,))
        self.assertEqual(results.acceptance_rate, 0.65)
        self.assertEqual(results.n_samples, 100)
        self.assertEqual(results.sampling_time, 10.5)
        
        # Optional fields should be None by default
        self.assertIsNone(results.effective_sample_size)
        self.assertIsNone(results.r_hat)
        self.assertIsNone(results.autocorr_time)


class TestDiagnostics(unittest.TestCase):
    """Test convergence diagnostics functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dim = 2
        
        def target_log_prob(x):
            return -0.5 * np.sum(x**2)
        
        self.sampler = MockSampler(target_log_prob, self.dim)
    
    def test_effective_sample_size_computation(self):
        """Test effective sample size computation."""
        # Create samples with known autocorrelation
        n_samples = 1000
        samples = np.random.randn(n_samples, self.dim)
        
        # Add some autocorrelation
        for i in range(1, n_samples):
            samples[i] += 0.5 * samples[i-1]  # AR(1) process
        
        ess = self.sampler._compute_ess(samples)
        
        # ESS should be less than n_samples due to autocorrelation
        self.assertLess(ess, n_samples)
        self.assertGreater(ess, 0)
    
    def test_autocorrelation_time_computation(self):
        """Test autocorrelation time computation."""
        # Independent samples should have small autocorr time
        samples_indep = np.random.randn(1000, self.dim)
        autocorr_time_indep = self.sampler._compute_autocorr_time(samples_indep)
        
        self.assertGreater(autocorr_time_indep, 0)
        self.assertLess(autocorr_time_indep, 10)  # Should be small for independent samples
        
        # Highly correlated samples should have large autocorr time
        samples_corr = np.random.randn(1000, self.dim)
        for i in range(1, 1000):
            samples_corr[i] += 0.9 * samples_corr[i-1]
        
        autocorr_time_corr = self.sampler._compute_autocorr_time(samples_corr)
        
        # Should be larger than independent case
        self.assertGreater(autocorr_time_corr, autocorr_time_indep)
    
    def test_diagnostics_edge_cases(self):
        """Test diagnostics with edge cases."""
        # Very short chain
        short_samples = np.random.randn(5, self.dim)
        ess_short = self.sampler._compute_ess(short_samples)
        
        # Should handle gracefully
        self.assertGreater(ess_short, 0)
        self.assertLessEqual(ess_short, 5)
    
    def test_quantile_computation(self):
        """Test quantile computation in diagnostics."""
        n_samples = 1000
        samples = np.random.randn(n_samples, self.dim)
        
        results = SamplingResults(
            samples=samples,
            log_probs=np.random.randn(n_samples),
            acceptance_rate=0.5,
            n_samples=n_samples
        )
        
        results = self.sampler._compute_diagnostics(results)
        
        # Check quantiles
        quantiles = results.diagnostics['quantiles']
        self.assertIn('5%', quantiles)
        self.assertIn('50%', quantiles)
        self.assertIn('95%', quantiles)
        
        # 50% should be close to mean for normal distribution
        medians = quantiles['50%']
        means = results.diagnostics['mean']
        
        np.testing.assert_allclose(medians, means, atol=0.1)


if __name__ == '__main__':
    unittest.main()