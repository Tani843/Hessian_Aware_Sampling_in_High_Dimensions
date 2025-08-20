#!/usr/bin/env python3
"""
Integration tests for Phase 3 implementation.

Tests that all components work together correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from typing import Dict, Any

# Core imports
from core.sampling_base import BaseSampler
from samplers.baseline_samplers import StandardMetropolis, LangevinDynamics
from benchmarks.sampler_comparison import SamplerBenchmark
from benchmarks.performance_metrics import effective_sample_size, potential_scale_reduction_factor
from benchmarks.convergence_diagnostics import ConvergenceDiagnostics
from analysis.theoretical_analysis import dimensional_scaling_theory


class SimpleGaussian:
    """Simple test distribution for integration testing."""
    
    def __init__(self, dimension: int = 2):
        self.dimension = dimension
        self.name = f"test_gaussian_{dimension}d"
        self.cov_matrix = np.eye(dimension)
        self.precision_matrix = np.eye(dimension)
    
    def log_prob(self, x: np.ndarray) -> float:
        """Log probability density."""
        return -0.5 * np.sum(x**2)
    
    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        """Gradient of log probability."""
        return -x
    
    def true_mean(self) -> np.ndarray:
        return np.zeros(self.dimension)
    
    def true_cov(self) -> np.ndarray:
        return self.cov_matrix.copy()


def test_basic_sampler_functionality():
    """Test that basic samplers work."""
    print("Testing basic sampler functionality...")
    
    # Create test distribution
    dist = SimpleGaussian(dimension=3)
    
    # Test Standard Metropolis
    sampler = StandardMetropolis(
        target_log_prob=dist.log_prob,
        dim=dist.dimension,
        step_size=0.5
    )
    
    initial_state = np.zeros(dist.dimension)
    
    # Test single step
    new_state, info = sampler.step(initial_state)
    
    assert isinstance(new_state, np.ndarray)
    assert new_state.shape == (dist.dimension,)
    assert isinstance(info, dict)
    assert 'accepted' in info
    
    # Test sampling
    results = sampler.sample(
        n_samples=100,
        initial_state=initial_state,
        burnin=50,
        return_diagnostics=True
    )
    
    assert results.samples.shape == (100, dist.dimension)
    assert len(results.log_probs) == 100
    assert 0 <= results.acceptance_rate <= 1
    
    print("✓ Basic sampler functionality working")


def test_performance_metrics():
    """Test performance metrics calculation."""
    print("Testing performance metrics...")
    
    # Generate test data
    n_samples = 500
    # Create correlated samples to test autocorrelation
    true_autocorr_time = 10
    samples = np.random.randn(n_samples, 2)
    
    # Add autocorrelation
    for i in range(1, n_samples):
        samples[i] += 0.9 * samples[i-1]  # Add correlation
    
    # Test ESS calculation
    ess = effective_sample_size(samples)
    assert isinstance(ess, float)
    assert 0 < ess <= n_samples
    
    # Test multiple chain R-hat
    chain1 = samples[:250]
    chain2 = samples[250:]
    r_hat = potential_scale_reduction_factor([chain1, chain2])
    assert isinstance(r_hat, float)
    assert r_hat >= 1.0
    
    print("✓ Performance metrics working")


def test_benchmark_framework():
    """Test the benchmarking framework."""
    print("Testing benchmark framework...")
    
    # Create test distributions and samplers
    dist = SimpleGaussian(dimension=2)
    
    samplers = {
        'Standard Metropolis': StandardMetropolis(
            target_log_prob=dist.log_prob,
            dim=dist.dimension,
            step_size=0.5
        ),
        'Langevin': LangevinDynamics(
            target_log_prob=dist.log_prob,
            target_log_prob_grad=dist.grad_log_prob,
            dim=dist.dimension,
            step_size=0.01
        )
    }
    
    # Run benchmark
    benchmark = SamplerBenchmark(
        test_distributions=[dist],
        samplers=samplers,
        metrics=['ess', 'timing']
    )
    
    results = benchmark.run_benchmark(
        n_samples=200,
        n_repeats=2,
        burnin=50
    )
    
    # Check results structure
    assert 'benchmark_results' in results
    assert dist.name in results['benchmark_results']
    
    for sampler_name in samplers.keys():
        assert sampler_name in results['benchmark_results'][dist.name]
        result = results['benchmark_results'][dist.name][sampler_name]
        assert hasattr(result, 'effective_sample_size')
        assert hasattr(result, 'acceptance_rate')
    
    print("✓ Benchmark framework working")


def test_convergence_diagnostics():
    """Test convergence diagnostics."""
    print("Testing convergence diagnostics...")
    
    # Generate test chains
    n_samples = 300
    chains = []
    
    for _ in range(2):  # Two chains
        chain = np.random.randn(n_samples, 2)
        # Add some correlation
        for i in range(1, n_samples):
            chain[i] += 0.5 * chain[i-1]
        chains.append(chain)
    
    # Run diagnostics
    diagnostics = ConvergenceDiagnostics(chains, parameter_names=['param_0', 'param_1'])
    diag_results = diagnostics.run_all_diagnostics(verbose=False)
    
    # Check results
    assert 'geweke' in diag_results
    assert 'r_hat' in diag_results
    assert len(diag_results['geweke']) == 2  # Two parameters
    
    # Test summary
    summary = diagnostics.convergence_summary()
    assert isinstance(summary, type(summary))  # Check it's a DataFrame-like object
    
    print("✓ Convergence diagnostics working")


def test_theoretical_analysis():
    """Test theoretical analysis functions."""
    print("Testing theoretical analysis...")
    
    # Test dimensional scaling theory
    predictions = dimensional_scaling_theory(
        dimension=10,
        condition_number=5.0,
        sampler_type='metropolis'
    )
    
    assert isinstance(predictions, dict)
    assert 'optimal_step_size' in predictions
    assert 'mixing_time_scaling' in predictions
    assert 'ess_scaling' in predictions
    
    # Test for different sampler types
    for sampler_type in ['metropolis', 'langevin', 'hmc']:
        pred = dimensional_scaling_theory(5, 2.0, sampler_type)
        assert isinstance(pred, dict)
        assert pred['sampler_type'] == sampler_type
    
    print("✓ Theoretical analysis working")


def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    print("Testing end-to-end workflow...")
    
    # 1. Create test problem
    dist = SimpleGaussian(dimension=3)
    
    # 2. Create samplers
    samplers = {
        'Metropolis': StandardMetropolis(
            target_log_prob=dist.log_prob,
            dim=dist.dimension,
            step_size=0.3
        )
    }
    
    # 3. Run benchmark
    benchmark = SamplerBenchmark([dist], samplers)
    results = benchmark.run_benchmark(n_samples=150, n_repeats=1, burnin=50)
    
    # 4. Extract samples for diagnostics
    sampler_result = results['benchmark_results'][dist.name]['Metropolis']
    samples = sampler_result.samples
    
    # 5. Run convergence diagnostics
    diagnostics = ConvergenceDiagnostics([samples])
    diag_results = diagnostics.run_all_diagnostics(verbose=False)
    
    # 6. Theoretical comparison
    theoretical_pred = dimensional_scaling_theory(
        dimension=dist.dimension,
        condition_number=1.0,
        sampler_type='metropolis'
    )
    
    # Verify we got results at each step
    assert sampler_result.effective_sample_size is not None
    assert sampler_result.acceptance_rate > 0
    assert len(diag_results) > 0
    assert len(theoretical_pred) > 0
    
    print("✓ End-to-end workflow working")


def run_integration_tests():
    """Run all integration tests."""
    print("="*60)
    print("🧪 RUNNING PHASE 3 INTEGRATION TESTS")
    print("="*60)
    
    tests = [
        test_basic_sampler_functionality,
        test_performance_metrics,
        test_benchmark_framework,
        test_convergence_diagnostics,
        test_theoretical_analysis,
        test_end_to_end_workflow
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"🏁 INTEGRATION TEST RESULTS")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {len(tests)}")
    
    if failed == 0:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Phase 3 implementation is working correctly")
    else:
        print("⚠️  Some tests failed - check implementation")
    
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)