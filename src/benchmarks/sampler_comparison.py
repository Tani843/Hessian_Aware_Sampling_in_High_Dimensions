"""
Comprehensive benchmarking framework for Hessian-aware samplers.

This module provides extensive benchmarking capabilities including:
- Performance comparison across multiple samplers
- Effective sample size analysis
- Convergence rate measurement
- Computational cost analysis
- Statistical significance testing
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass
from collections import defaultdict
import json
from scipy import stats
from scipy.special import logsumexp

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("Matplotlib/Seaborn not available. Plotting will be disabled.")

try:
    from ..core.sampling_base import BaseSampler, SamplingResults
    from ..utils.validation import validate_array, validate_positive_scalar
except ImportError:
    from core.sampling_base import BaseSampler, SamplingResults
    try:
        from utils.validation import validate_array, validate_positive_scalar
    except ImportError:
        def validate_array(x):
            return x
        def validate_positive_scalar(x):
            return x

# Try to import Hessian samplers if available
try:
    try:
        from ..samplers.advanced_hessian_samplers import (
            HessianAwareMetropolis,
            HessianAwareLangevin,
            AdaptiveHessianSampler
        )
    except ImportError:
        from samplers.advanced_hessian_samplers import (
            HessianAwareMetropolis,
            HessianAwareLangevin,
            AdaptiveHessianSampler
        )
except ImportError:
    # Hessian samplers not available
    pass


@dataclass
class BenchmarkResult:
    """Container for individual benchmark results."""
    sampler_name: str
    distribution_name: str
    dimension: int
    n_samples: int
    samples: np.ndarray
    log_probs: np.ndarray
    sampling_time: float
    acceptance_rate: float
    effective_sample_size: Optional[float] = None
    r_hat: Optional[float] = None
    autocorr_time: Optional[float] = None
    ess_per_second: Optional[float] = None
    mean_squared_error: Optional[float] = None
    diagnostics: Optional[Dict[str, Any]] = None


@dataclass
class ComparisonMetrics:
    """Container for sampler comparison metrics."""
    ess_comparison: pd.DataFrame
    timing_comparison: pd.DataFrame
    convergence_comparison: pd.DataFrame
    statistical_tests: Dict[str, Any]
    dimensional_scaling: Optional[pd.DataFrame] = None


class SamplerBenchmark:
    """
    Comprehensive benchmarking framework for MCMC samplers.
    
    Features:
    - Multiple sampler comparison
    - Various performance metrics
    - Statistical significance testing
    - Dimensional scaling analysis
    - Automated reporting
    """
    
    def __init__(self,
                 test_distributions: List[Any],
                 samplers: Dict[str, BaseSampler],
                 metrics: List[str] = None):
        """
        Initialize benchmarking framework.
        
        Args:
            test_distributions: List of test distributions to benchmark on
            samplers: Dictionary of {name: sampler} pairs
            metrics: List of metrics to compute ('ess', 'timing', 'convergence', etc.)
        """
        self.test_distributions = test_distributions
        self.samplers = samplers
        self.metrics = metrics or ['ess', 'timing', 'convergence', 'accuracy']
        
        # Storage for results
        self.benchmark_results: Dict[str, Dict[str, BenchmarkResult]] = defaultdict(dict)
        self.comparison_metrics: Optional[ComparisonMetrics] = None
        
        # Configuration
        self.random_seed = 42
        self.verbose = True
        
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate benchmark inputs."""
        if not self.test_distributions:
            raise ValueError("At least one test distribution required")
        
        if not self.samplers:
            raise ValueError("At least one sampler required")
        
        for name, sampler in self.samplers.items():
            if not isinstance(sampler, BaseSampler):
                raise TypeError(f"Sampler {name} must inherit from BaseSampler")
        
        valid_metrics = {'ess', 'timing', 'convergence', 'accuracy', 'scaling'}
        invalid_metrics = set(self.metrics) - valid_metrics
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}")
    
    def run_benchmark(self,
                      n_samples: int,
                      n_repeats: int = 5,
                      burnin: int = 500,
                      thin: int = 1,
                      progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across all samplers and distributions.
        
        Args:
            n_samples: Number of samples to generate per run
            n_repeats: Number of repeated runs for statistical reliability
            burnin: Number of burnin samples
            thin: Thinning interval
            progress_callback: Optional progress reporting function
            
        Returns:
            Dictionary containing all benchmark results and comparisons
        """
        if self.verbose:
            print("=" * 60)
            print("COMPREHENSIVE SAMPLER BENCHMARK")
            print("=" * 60)
            print(f"Distributions: {len(self.test_distributions)}")
            print(f"Samplers: {list(self.samplers.keys())}")
            print(f"Samples per run: {n_samples}")
            print(f"Repeats: {n_repeats}")
            print(f"Metrics: {self.metrics}")
        
        np.random.seed(self.random_seed)
        
        total_runs = len(self.test_distributions) * len(self.samplers) * n_repeats
        current_run = 0
        
        # Run benchmarks
        for dist in self.test_distributions:
            dist_name = getattr(dist, 'name', str(dist))
            
            if self.verbose:
                print(f"\n📊 Testing distribution: {dist_name}")
            
            for sampler_name, sampler in self.samplers.items():
                if self.verbose:
                    print(f"  🔄 Running {sampler_name}...")
                
                results_list = []
                
                for repeat in range(n_repeats):
                    current_run += 1
                    
                    try:
                        # Run single benchmark
                        result = self._run_single_benchmark(
                            dist, sampler, sampler_name,
                            n_samples, burnin, thin
                        )
                        results_list.append(result)
                        
                        if progress_callback:
                            progress_callback(current_run, total_runs)
                        
                    except Exception as e:
                        warnings.warn(f"Benchmark failed for {sampler_name} on {dist_name}: {e}")
                        continue
                
                # Aggregate results across repeats
                if results_list:
                    aggregated_result = self._aggregate_results(results_list)
                    self.benchmark_results[dist_name][sampler_name] = aggregated_result
                    
                    if self.verbose:
                        self._print_single_result(aggregated_result)
        
        # Compute comparison metrics
        if self.verbose:
            print(f"\n📈 Computing comparison metrics...")
        
        self.comparison_metrics = self._compute_comparison_metrics()
        
        # Generate summary
        summary = {
            'benchmark_results': self.benchmark_results,
            'comparison_metrics': self.comparison_metrics,
            'configuration': {
                'n_samples': n_samples,
                'n_repeats': n_repeats,
                'burnin': burnin,
                'thin': thin,
                'metrics': self.metrics,
                'random_seed': self.random_seed
            }
        }
        
        if self.verbose:
            print(f"\n✅ Benchmark complete! Results available in summary.")
        
        return summary
    
    def _run_single_benchmark(self,
                             distribution: Any,
                             sampler: BaseSampler,
                             sampler_name: str,
                             n_samples: int,
                             burnin: int,
                             thin: int) -> BenchmarkResult:
        """Run benchmark on single sampler-distribution pair."""
        dim = sampler.dim
        dist_name = getattr(distribution, 'name', str(distribution))
        
        # Generate initial state
        initial_state = self._generate_initial_state(distribution, dim)
        
        # Time the sampling
        start_time = time.perf_counter()
        
        try:
            # Generate samples using the base sampling interface
            if hasattr(sampler, 'sample'):
                results = sampler.sample(
                    n_samples=n_samples,
                    initial_state=initial_state,
                    burnin=burnin,
                    thin=thin,
                    return_diagnostics=True
                )
                samples = results.samples
                log_probs = results.log_probs
                acceptance_rate = results.acceptance_rate
                sampling_time = results.sampling_time
                
            else:
                # Fallback to manual sampling
                samples = []
                log_probs = []
                current_state = initial_state.copy()
                n_accepted = 0
                
                # Burnin
                for _ in range(burnin):
                    current_state, info = sampler.step(current_state)
                    if info.get('accepted', False):
                        n_accepted += 1
                
                # Sampling
                for i in range(n_samples * thin):
                    current_state, info = sampler.step(current_state)
                    if info.get('accepted', False):
                        n_accepted += 1
                    
                    if i % thin == 0:
                        samples.append(current_state.copy())
                        log_probs.append(info.get('log_prob', 0.0))
                
                samples = np.array(samples)
                log_probs = np.array(log_probs)
                acceptance_rate = n_accepted / (burnin + n_samples * thin)
                sampling_time = time.perf_counter() - start_time
        
        except Exception as e:
            raise RuntimeError(f"Sampling failed: {e}")
        
        # Compute performance metrics
        ess = self._compute_effective_sample_size(samples)
        autocorr_time = self._compute_autocorr_time(samples)
        r_hat = self._compute_r_hat(samples) if samples.shape[0] > 100 else None
        
        # Compute accuracy if true statistics available
        mse = None
        if hasattr(distribution, 'true_mean') and hasattr(distribution, 'true_cov'):
            try:
                mse = self._compute_accuracy_metrics(
                    samples, distribution.true_mean(), distribution.true_cov()
                )
            except:
                pass
        
        # Create result
        result = BenchmarkResult(
            sampler_name=sampler_name,
            distribution_name=dist_name,
            dimension=dim,
            n_samples=n_samples,
            samples=samples,
            log_probs=log_probs,
            sampling_time=sampling_time,
            acceptance_rate=acceptance_rate,
            effective_sample_size=ess,
            r_hat=r_hat,
            autocorr_time=autocorr_time,
            ess_per_second=ess / sampling_time if ess and sampling_time > 0 else None,
            mean_squared_error=mse,
            diagnostics=getattr(sampler, 'get_diagnostics', lambda: {})()
        )
        
        return result
    
    def _generate_initial_state(self, distribution: Any, dim: int) -> np.ndarray:
        """Generate reasonable initial state for sampling."""
        if hasattr(distribution, 'true_mean'):
            # Start near true mean if available
            mean = distribution.true_mean()
            return mean + 0.1 * np.random.randn(dim)
        else:
            # Start at origin with small perturbation
            return 0.1 * np.random.randn(dim)
    
    def _compute_effective_sample_size(self, samples: np.ndarray) -> float:
        """Compute effective sample size using autocorrelation."""
        if len(samples) < 10:
            return float(len(samples))
        
        try:
            # Use first dimension for ESS calculation
            x = samples[:, 0] - np.mean(samples[:, 0])
            
            # FFT-based autocorrelation
            n = len(x)
            f = np.fft.fft(x, n=2*n)
            acorr = np.fft.ifft(f * np.conj(f))[:n].real
            acorr = acorr / acorr[0] if acorr[0] > 0 else acorr
            
            # Find where autocorr drops below threshold
            cutoff = np.where(acorr < 0.05)[0]
            if len(cutoff) > 0:
                tau = cutoff[0]
            else:
                tau = n // 4
            
            ess = n / (1 + 2 * tau)
            return max(1.0, ess)
            
        except Exception:
            return float(len(samples))
    
    def _compute_autocorr_time(self, samples: np.ndarray) -> float:
        """Compute integrated autocorrelation time."""
        try:
            x = samples[:, 0] - np.mean(samples[:, 0])
            
            # Compute autocorrelation
            n = len(x)
            f = np.fft.fft(x, n=2*n)
            acorr = np.fft.ifft(f * np.conj(f))[:n].real
            acorr = acorr / acorr[0] if acorr[0] > 0 else acorr
            
            # Integrated autocorr time
            cumsum = np.cumsum(acorr)
            for i in range(1, len(cumsum)):
                if i >= 6 * cumsum[i]:
                    return cumsum[i]
            
            return cumsum[-1] if len(cumsum) > 0 else 1.0
            
        except Exception:
            return 1.0
    
    def _compute_r_hat(self, samples: np.ndarray) -> Optional[float]:
        """Compute R-hat convergence diagnostic."""
        try:
            n = len(samples)
            if n < 50:
                return None
            
            # Split into two chains
            mid = n // 2
            chain1 = samples[:mid, 0]
            chain2 = samples[mid:, 0]
            
            # Compute R-hat
            m = len(chain1)
            chain_means = np.array([np.mean(chain1), np.mean(chain2)])
            chain_vars = np.array([np.var(chain1), np.var(chain2)])
            
            W = np.mean(chain_vars)
            B = m * np.var(chain_means)
            
            if W > 0:
                V_hat = ((m - 1) * W + B) / m
                r_hat = np.sqrt(V_hat / W)
                return float(r_hat)
            else:
                return 1.0
                
        except Exception:
            return None
    
    def _compute_accuracy_metrics(self,
                                samples: np.ndarray,
                                true_mean: np.ndarray,
                                true_cov: np.ndarray) -> float:
        """Compute accuracy metrics compared to true distribution."""
        sample_mean = np.mean(samples, axis=0)
        sample_cov = np.cov(samples.T)
        
        # Mean squared error for mean
        mean_mse = np.mean((sample_mean - true_mean)**2)
        
        # Frobenius norm error for covariance  
        cov_error = np.linalg.norm(sample_cov - true_cov, 'fro')
        
        # Combined metric
        return mean_mse + 0.1 * cov_error
    
    def _aggregate_results(self, results_list: List[BenchmarkResult]) -> BenchmarkResult:
        """Aggregate results from multiple runs."""
        if len(results_list) == 1:
            return results_list[0]
        
        # Use first result as template
        base_result = results_list[0]
        
        # Aggregate metrics
        acceptance_rates = [r.acceptance_rate for r in results_list]
        sampling_times = [r.sampling_time for r in results_list]
        ess_values = [r.effective_sample_size for r in results_list if r.effective_sample_size]
        ess_per_second = [r.ess_per_second for r in results_list if r.ess_per_second]
        mse_values = [r.mean_squared_error for r in results_list if r.mean_squared_error]
        
        # Create aggregated result
        aggregated = BenchmarkResult(
            sampler_name=base_result.sampler_name,
            distribution_name=base_result.distribution_name,
            dimension=base_result.dimension,
            n_samples=base_result.n_samples,
            samples=base_result.samples,  # Use first run's samples
            log_probs=base_result.log_probs,
            sampling_time=np.mean(sampling_times),
            acceptance_rate=np.mean(acceptance_rates),
            effective_sample_size=np.mean(ess_values) if ess_values else None,
            ess_per_second=np.mean(ess_per_second) if ess_per_second else None,
            mean_squared_error=np.mean(mse_values) if mse_values else None,
            diagnostics={
                'std_acceptance_rate': np.std(acceptance_rates),
                'std_sampling_time': np.std(sampling_times),
                'std_ess': np.std(ess_values) if ess_values else None,
                'n_repeats': len(results_list)
            }
        )
        
        return aggregated
    
    def _print_single_result(self, result: BenchmarkResult):
        """Print single benchmark result."""
        if not self.verbose:
            return
        
        ess_str = f"{result.effective_sample_size:.1f}" if result.effective_sample_size else "N/A"
        ess_per_sec_str = f"{result.ess_per_second:.1f}" if result.ess_per_second else "N/A"
        
        print(f"    Accept: {result.acceptance_rate:.3f}, "
              f"Time: {result.sampling_time:.2f}s, "
              f"ESS: {ess_str}, "
              f"ESS/s: {ess_per_sec_str}")
    
    def _compute_comparison_metrics(self) -> ComparisonMetrics:
        """Compute comprehensive comparison metrics."""
        # Extract data for comparison
        comparison_data = []
        
        for dist_name, sampler_results in self.benchmark_results.items():
            for sampler_name, result in sampler_results.items():
                comparison_data.append({
                    'Distribution': dist_name,
                    'Sampler': sampler_name,
                    'Dimension': result.dimension,
                    'ESS': result.effective_sample_size,
                    'Time': result.sampling_time,
                    'ESS_per_second': result.ess_per_second,
                    'Acceptance_rate': result.acceptance_rate,
                    'MSE': result.mean_squared_error,
                    'R_hat': result.r_hat,
                    'Autocorr_time': result.autocorr_time
                })
        
        df = pd.DataFrame(comparison_data)
        
        # ESS comparison
        ess_comparison = df.pivot(
            index=['Distribution', 'Dimension'],
            columns='Sampler',
            values='ESS'
        )
        
        # Timing comparison
        timing_comparison = df.pivot(
            index=['Distribution', 'Dimension'],
            columns='Sampler',
            values='ESS_per_second'
        )
        
        # Convergence comparison
        convergence_comparison = df.pivot(
            index=['Distribution', 'Dimension'],
            columns='Sampler',
            values='R_hat'
        )
        
        # Statistical tests
        statistical_tests = self._perform_statistical_tests(df)
        
        return ComparisonMetrics(
            ess_comparison=ess_comparison,
            timing_comparison=timing_comparison,
            convergence_comparison=convergence_comparison,
            statistical_tests=statistical_tests
        )
    
    def _perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        tests = {}
        
        samplers = df['Sampler'].unique()
        if len(samplers) < 2:
            return tests
        
        # Pairwise comparisons for ESS
        for metric in ['ESS', 'ESS_per_second']:
            metric_tests = {}
            
            for i, sampler1 in enumerate(samplers):
                for sampler2 in samplers[i+1:]:
                    data1 = df[df['Sampler'] == sampler1][metric].dropna()
                    data2 = df[df['Sampler'] == sampler2][metric].dropna()
                    
                    if len(data1) > 1 and len(data2) > 1:
                        try:
                            statistic, p_value = stats.mannwhitneyu(
                                data1, data2, alternative='two-sided'
                            )
                            metric_tests[f"{sampler1}_vs_{sampler2}"] = {
                                'statistic': float(statistic),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05
                            }
                        except Exception:
                            continue
            
            tests[metric] = metric_tests
        
        return tests
    
    def compare_effective_sample_size(self) -> pd.DataFrame:
        """Compare effective sample sizes across samplers."""
        if self.comparison_metrics is None:
            raise ValueError("Must run benchmark first")
        
        return self.comparison_metrics.ess_comparison
    
    def compare_convergence_rates(self) -> pd.DataFrame:
        """Compare convergence rates across samplers."""
        if self.comparison_metrics is None:
            raise ValueError("Must run benchmark first")
        
        return self.comparison_metrics.convergence_comparison
    
    def analyze_computational_cost(self) -> Dict[str, Any]:
        """Analyze computational cost across samplers."""
        if not self.benchmark_results:
            raise ValueError("Must run benchmark first")
        
        cost_analysis = {}
        
        for dist_name, sampler_results in self.benchmark_results.items():
            dist_analysis = {}
            
            for sampler_name, result in sampler_results.items():
                dist_analysis[sampler_name] = {
                    'total_time': result.sampling_time,
                    'time_per_sample': result.sampling_time / result.n_samples,
                    'time_per_ess': (result.sampling_time / result.effective_sample_size
                                   if result.effective_sample_size else np.inf),
                    'ess_per_second': result.ess_per_second
                }
            
            cost_analysis[dist_name] = dist_analysis
        
        return cost_analysis
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive benchmark report."""
        if not self.benchmark_results or not self.comparison_metrics:
            raise ValueError("Must run benchmark first")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE SAMPLER BENCHMARK REPORT")
        report_lines.append("=" * 80)
        
        # Summary statistics
        report_lines.append("\n📊 SUMMARY STATISTICS")
        report_lines.append("-" * 40)
        
        total_samplers = len(self.samplers)
        total_distributions = len(self.test_distributions)
        
        report_lines.append(f"Samplers tested: {total_samplers}")
        report_lines.append(f"Distributions: {total_distributions}")
        report_lines.append(f"Metrics computed: {', '.join(self.metrics)}")
        
        # Performance rankings
        report_lines.append("\n🏆 PERFORMANCE RANKINGS")
        report_lines.append("-" * 40)
        
        # ESS ranking
        ess_means = self.comparison_metrics.ess_comparison.mean(axis=0).sort_values(ascending=False)
        report_lines.append("\nEffective Sample Size (average):")
        for sampler, ess in ess_means.items():
            if not pd.isna(ess):
                report_lines.append(f"  {sampler}: {ess:.1f}")
        
        # ESS/second ranking
        timing_means = self.comparison_metrics.timing_comparison.mean(axis=0).sort_values(ascending=False)
        report_lines.append("\nESS per Second (average):")
        for sampler, ess_per_sec in timing_means.items():
            if not pd.isna(ess_per_sec):
                report_lines.append(f"  {sampler}: {ess_per_sec:.1f}")
        
        # Distribution-specific results
        report_lines.append("\n📈 DISTRIBUTION-SPECIFIC RESULTS")
        report_lines.append("-" * 40)
        
        for dist_name, sampler_results in self.benchmark_results.items():
            report_lines.append(f"\n{dist_name}:")
            
            # Sort by ESS/second
            sorted_results = sorted(
                sampler_results.items(),
                key=lambda x: x[1].ess_per_second or 0,
                reverse=True
            )
            
            for sampler_name, result in sorted_results:
                ess_val = f"{result.effective_sample_size:.1f}" if result.effective_sample_size else 'N/A'
                ess_per_sec_val = f"{result.ess_per_second:.1f}" if result.ess_per_second else 'N/A'
                
                report_lines.append(
                    f"  {sampler_name:<20}: "
                    f"ESS={ess_val:<8}, "
                    f"ESS/s={ess_per_sec_val:<8}, "
                    f"Accept={result.acceptance_rate:.3f}"
                )
        
        # Statistical significance
        if self.comparison_metrics.statistical_tests:
            report_lines.append("\n🔬 STATISTICAL SIGNIFICANCE TESTS")
            report_lines.append("-" * 40)
            
            for metric, tests in self.comparison_metrics.statistical_tests.items():
                report_lines.append(f"\n{metric}:")
                for comparison, test_result in tests.items():
                    significance = "***" if test_result['significant'] else ""
                    report_lines.append(
                        f"  {comparison}: p={test_result['p_value']:.4f} {significance}"
                    )
        
        # Recommendations
        report_lines.append("\n💡 RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        # Best overall performer
        if len(timing_means) > 0:
            best_overall = timing_means.index[0]
            report_lines.append(f"• Best overall performance: {best_overall}")
        
        # High-dimensional recommendations
        high_dim_results = {
            name: result for dist_name, sampler_results in self.benchmark_results.items()
            for name, result in sampler_results.items()
            if result.dimension >= 10
        }
        
        if high_dim_results:
            high_dim_best = max(
                high_dim_results.items(),
                key=lambda x: x[1].ess_per_second or 0
            )
            report_lines.append(f"• Best for high dimensions: {high_dim_best[0]}")
        
        report_lines.append("\n" + "=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text