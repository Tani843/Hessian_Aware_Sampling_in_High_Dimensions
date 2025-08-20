"""
Basic example demonstrating Hessian-aware sampling.

This example shows how to use the Hessian-aware sampler on various
test distributions and compares it with standard MCMC methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.hessian_utils import compute_hessian_finite_diff
from samplers.hessian_sampler import HessianAwareSampler
from test_distributions import get_test_distribution, create_test_suite
from visualization.plotting import (
    plot_sampling_results, 
    plot_comparison,
    plot_hessian_properties
)


def run_basic_example():
    """Run basic example with multivariate Gaussian."""
    print("=" * 60)
    print("BASIC HESSIAN-AWARE SAMPLING EXAMPLE")
    print("=" * 60)
    
    # Setup
    np.random.seed(42)
    dim = 5
    
    # Create a moderately challenging Gaussian distribution
    print(f"\n1. Creating {dim}D Gaussian distribution with condition number 50...")
    dist = get_test_distribution('gaussian', dim, condition_number=50.0, seed=42)
    print(f"   Distribution: {dist.name}")
    print(f"   True mean: {dist.true_mean()}")
    print(f"   Condition number: {np.linalg.cond(dist.true_cov()):.1f}")
    
    # Create sampler
    print("\n2. Setting up Hessian-aware sampler...")
    sampler = HessianAwareSampler(
        target_log_prob=dist.log_prob,
        dim=dim,
        step_size=0.1,
        hessian_method='finite_diff',
        hessian_update_freq=10,
        use_preconditioning=True,
        adapt_step_size=True
    )
    
    # Initial state
    initial_state = np.random.randn(dim)
    print(f"   Initial state: {initial_state}")
    
    # Warmup
    print("\n3. Running warmup phase...")
    warmed_up_state = sampler.warmup(initial_state, n_warmup=500)
    print(f"   Final acceptance rate after warmup: {sampler.get_acceptance_rate():.3f}")
    print(f"   Final step size: {sampler.step_size:.4f}")
    
    # Sampling
    print("\n4. Generating samples...")
    results = sampler.sample(
        n_samples=2000,
        initial_state=warmed_up_state,
        burnin=500,
        thin=2,
        return_diagnostics=True
    )
    
    print(f"   Samples generated: {results.n_samples}")
    print(f"   Final acceptance rate: {results.acceptance_rate:.3f}")
    print(f"   Effective sample size: {results.effective_sample_size:.1f}")
    print(f"   Autocorr time: {results.autocorr_time:.1f}")
    print(f"   Sampling time: {results.sampling_time:.2f} seconds")
    
    # Compare with true statistics
    print("\n5. Comparing with true statistics...")
    sample_mean = np.mean(results.samples, axis=0)
    sample_cov = np.cov(results.samples.T)
    true_mean = dist.true_mean()
    true_cov = dist.true_cov()
    
    mean_error = np.linalg.norm(sample_mean - true_mean)
    cov_error = np.linalg.norm(sample_cov - true_cov, 'fro')
    
    print(f"   Mean error: {mean_error:.4f}")
    print(f"   Covariance error (Frobenius): {cov_error:.4f}")
    print(f"   Sample mean: {sample_mean}")
    print(f"   True mean: {true_mean}")
    
    # Visualize results
    print("\n6. Creating visualizations...")
    try:
        fig = plot_sampling_results(results, max_dims_to_plot=3)
        plt.savefig('basic_example_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: basic_example_results.png")
    except ImportError:
        print("   Matplotlib not available - skipping plots")
    
    # Hessian analysis
    print("\n7. Analyzing Hessian properties...")
    test_point = np.zeros(dim)
    hessian = compute_hessian_finite_diff(dist.log_prob, test_point)
    
    eigenvals = np.linalg.eigvals(hessian)
    condition_num = np.max(eigenvals) / np.min(eigenvals) if np.min(eigenvals) > 0 else np.inf
    
    print(f"   Hessian at origin:")
    print(f"   Max eigenvalue: {np.max(eigenvals):.3f}")
    print(f"   Min eigenvalue: {np.min(eigenvals):.3f}")
    print(f"   Condition number: {condition_num:.1f}")
    
    try:
        fig_hess = plot_hessian_properties(hessian, title="Hessian at Origin")
        plt.savefig('basic_example_hessian.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: basic_example_hessian.png")
    except ImportError:
        pass
    
    print("\n8. Getting sampler statistics...")
    hess_stats = sampler.get_hessian_stats()
    print(f"   Hessian failures: {hess_stats['hessian_failures']}")
    print(f"   Total steps: {hess_stats['step_count']}")
    
    return results, sampler


def compare_samplers_example():
    """Compare Hessian-aware sampler with fallback methods."""
    print("\n" + "=" * 60)
    print("SAMPLER COMPARISON EXAMPLE")
    print("=" * 60)
    
    np.random.seed(123)
    dim = 3
    
    # Create challenging Rosenbrock distribution
    print(f"\n1. Creating challenging Rosenbrock distribution...")
    dist = get_test_distribution('rosenbrock', dim)
    print(f"   Distribution: {dist.name}")
    
    initial_state = np.ones(dim) + 0.5 * np.random.randn(dim)
    n_samples = 1000
    
    # Hessian-aware sampler
    print("\n2. Running Hessian-aware sampler...")
    ha_sampler = HessianAwareSampler(
        target_log_prob=dist.log_prob,
        dim=dim,
        step_size=0.01,
        hessian_method='finite_diff',
        use_preconditioning=True
    )
    
    ha_results = ha_sampler.sample(
        n_samples=n_samples,
        initial_state=initial_state.copy(),
        burnin=500,
        return_diagnostics=True
    )
    
    print(f"   Acceptance rate: {ha_results.acceptance_rate:.3f}")
    print(f"   Effective sample size: {ha_results.effective_sample_size:.1f}")
    
    # Fallback sampler (no Hessian preconditioning)
    print("\n3. Running fallback sampler...")
    fallback_sampler = HessianAwareSampler(
        target_log_prob=dist.log_prob,
        dim=dim,
        step_size=0.01,
        use_preconditioning=False,
        fallback_to_mala=True
    )
    
    fallback_results = fallback_sampler.sample(
        n_samples=n_samples,
        initial_state=initial_state.copy(),
        burnin=500,
        return_diagnostics=True
    )
    
    print(f"   Acceptance rate: {fallback_results.acceptance_rate:.3f}")
    print(f"   Effective sample size: {fallback_results.effective_sample_size:.1f}")
    
    # Compare results
    print("\n4. Comparing sampler performance...")
    
    results_dict = {
        'Hessian-aware': ha_results,
        'MALA fallback': fallback_results
    }
    
    try:
        fig = plot_comparison(results_dict, dimension=0)
        plt.savefig('sampler_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: sampler_comparison.png")
    except ImportError:
        pass
    
    # Performance metrics
    print("\n   Performance Summary:")
    print(f"   {'Method':<15} {'Accept Rate':<12} {'ESS':<8} {'Time (s)':<10}")
    print(f"   {'-'*15} {'-'*12} {'-'*8} {'-'*10}")
    print(f"   {'Hessian-aware':<15} {ha_results.acceptance_rate:<12.3f} "
          f"{ha_results.effective_sample_size:<8.1f} {ha_results.sampling_time:<10.2f}")
    print(f"   {'MALA fallback':<15} {fallback_results.acceptance_rate:<12.3f} "
          f"{fallback_results.effective_sample_size:<8.1f} {fallback_results.sampling_time:<10.2f}")
    
    return results_dict


def distribution_suite_example():
    """Test sampler on suite of distributions."""
    print("\n" + "=" * 60)
    print("DISTRIBUTION SUITE EXAMPLE")
    print("=" * 60)
    
    np.random.seed(456)
    dim = 4
    
    # Get test suite
    distributions = create_test_suite(dim)
    
    print(f"\n1. Testing on {len(distributions)} distributions...")
    
    results_summary = {}
    
    for name, dist in distributions.items():
        print(f"\n   Testing: {name}")
        
        try:
            # Create sampler
            sampler = HessianAwareSampler(
                target_log_prob=dist.log_prob,
                dim=dim,
                step_size=0.05,
                hessian_method='finite_diff',
                hessian_update_freq=20
            )
            
            # Sample
            initial_state = np.random.randn(dim)
            results = sampler.sample(
                n_samples=500,
                initial_state=initial_state,
                burnin=200,
                return_diagnostics=True
            )
            
            # Store results
            results_summary[name] = {
                'acceptance_rate': results.acceptance_rate,
                'ess': results.effective_sample_size,
                'time': results.sampling_time,
                'hessian_failures': sampler.get_hessian_stats()['hessian_failures']
            }
            
            print(f"      ✓ Accept: {results.acceptance_rate:.3f}, "
                  f"ESS: {results.effective_sample_size:.1f}, "
                  f"Time: {results.sampling_time:.2f}s")
            
        except Exception as e:
            print(f"      ✗ Failed: {str(e)}")
            results_summary[name] = {'error': str(e)}
    
    # Summary table
    print("\n2. Results Summary:")
    print(f"   {'Distribution':<20} {'Accept Rate':<12} {'ESS':<8} {'Time (s)':<10} {'H-Fails':<8}")
    print(f"   {'-'*20} {'-'*12} {'-'*8} {'-'*10} {'-'*8}")
    
    for name, stats in results_summary.items():
        if 'error' not in stats:
            print(f"   {name:<20} {stats['acceptance_rate']:<12.3f} "
                  f"{stats['ess']:<8.1f} {stats['time']:<10.2f} {stats['hessian_failures']:<8}")
        else:
            print(f"   {name:<20} {'ERROR':<12}")
    
    return results_summary


def advanced_features_example():
    """Demonstrate advanced features of the sampler."""
    print("\n" + "=" * 60)
    print("ADVANCED FEATURES EXAMPLE")  
    print("=" * 60)
    
    np.random.seed(789)
    dim = 6
    
    # Create mixture distribution (challenging multimodal case)
    print(f"\n1. Creating mixture distribution...")
    dist = get_test_distribution('mixture', dim, n_components=2, separation=4.0)
    print(f"   Distribution: {dist.name}")
    
    # Test different Hessian update frequencies
    print("\n2. Testing different Hessian update frequencies...")
    
    frequencies = [5, 20, 50]
    freq_results = {}
    
    for freq in frequencies:
        print(f"   Testing frequency: {freq}")
        
        sampler = HessianAwareSampler(
            target_log_prob=dist.log_prob,
            dim=dim,
            step_size=0.1,
            hessian_update_freq=freq,
            adapt_step_size=True
        )
        
        results = sampler.sample(
            n_samples=800,
            initial_state=np.zeros(dim),
            burnin=200,
            return_diagnostics=True
        )
        
        freq_results[f'freq_{freq}'] = results
        print(f"      Accept: {results.acceptance_rate:.3f}, "
              f"ESS: {results.effective_sample_size:.1f}")
    
    # Test regularization parameters
    print("\n3. Testing Hessian regularization...")
    
    regularizations = [1e-8, 1e-6, 1e-4]
    reg_results = {}
    
    for reg in regularizations:
        print(f"   Testing regularization: {reg:.0e}")
        
        sampler = HessianAwareSampler(
            target_log_prob=dist.log_prob,
            dim=dim,
            hessian_regularization=reg,
            step_size=0.1
        )
        
        results = sampler.sample(
            n_samples=500,
            initial_state=np.zeros(dim),
            burnin=200,
            return_diagnostics=False
        )
        
        reg_results[f'reg_{reg:.0e}'] = results
        print(f"      Accept: {results.acceptance_rate:.3f}")
    
    print("\n4. Performance comparison:")
    print("   Update frequency effects:")
    for name, results in freq_results.items():
        freq = name.split('_')[1]
        print(f"     Freq {freq:<3}: Accept={results.acceptance_rate:.3f}, "
              f"ESS={results.effective_sample_size:.1f}")
    
    print("   Regularization effects:")
    for name, results in reg_results.items():
        reg = name.split('_')[1]
        print(f"     Reg {reg:<5}: Accept={results.acceptance_rate:.3f}")
    
    return freq_results, reg_results


def main():
    """Run all examples."""
    print("HESSIAN-AWARE SAMPLING EXAMPLES")
    print("This script demonstrates the capabilities of the Hessian-aware sampler")
    print("on various test distributions and compares different configurations.\n")
    
    try:
        # Basic example
        basic_results, basic_sampler = run_basic_example()
        
        # Sampler comparison
        comparison_results = compare_samplers_example()
        
        # Distribution suite
        suite_results = distribution_suite_example()
        
        # Advanced features
        advanced_results = advanced_features_example()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - basic_example_results.png (if matplotlib available)")
        print("  - basic_example_hessian.png (if matplotlib available)")
        print("  - sampler_comparison.png (if matplotlib available)")
        
        print(f"\nKey findings:")
        print(f"  - Hessian-aware sampling can improve efficiency on ill-conditioned problems")
        print(f"  - Automatic fallback to MALA/random walk ensures robustness")
        print(f"  - Adaptive step sizing helps achieve target acceptance rates")
        print(f"  - Regularization handles numerical instabilities in Hessian computation")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\nThis might be due to missing dependencies.")
        print(f"Try installing: pip install numpy scipy matplotlib")
        
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)