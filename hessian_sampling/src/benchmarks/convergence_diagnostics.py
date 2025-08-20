"""
Advanced convergence diagnostics for MCMC sampling.

This module provides comprehensive convergence assessment tools including:
- Geweke diagnostic for within-chain convergence
- Heidelberger-Welch test for stationarity
- Multiple chain diagnostics
- Visual convergence tools
- Automated convergence detection
"""

import numpy as np
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.fft import fft, ifft
import pandas as pd

try:
    from .performance_metrics import (
        effective_sample_size,
        potential_scale_reduction_factor,
        geweke_diagnostic,
        heidelberger_welch_test
    )
except ImportError:
    from performance_metrics import (
        effective_sample_size,
        potential_scale_reduction_factor,
        geweke_diagnostic,
        heidelberger_welch_test
    )


@dataclass
class ConvergenceResult:
    """Results from convergence diagnostic."""
    converged: bool
    diagnostic_name: str
    statistic: float
    p_value: Optional[float] = None
    critical_value: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = None


class ConvergenceDiagnostics:
    """
    Comprehensive convergence diagnostics for MCMC chains.
    
    Provides multiple diagnostic tests and automated convergence detection.
    """
    
    def __init__(self, 
                 chains: Union[np.ndarray, List[np.ndarray]],
                 parameter_names: Optional[List[str]] = None):
        """
        Initialize convergence diagnostics.
        
        Args:
            chains: MCMC chains - either single chain (n_samples, n_dims) 
                   or list of chains for multiple chain analysis
            parameter_names: Names for parameters/dimensions
        """
        self.chains = self._standardize_chains(chains)
        self.n_chains = len(self.chains)
        self.n_samples = len(self.chains[0])
        self.n_dims = self.chains[0].shape[1] if self.chains[0].ndim > 1 else 1
        
        self.parameter_names = (parameter_names or 
                              [f'param_{i}' for i in range(self.n_dims)])
        
        self._validate_inputs()
    
    def _standardize_chains(self, chains):
        """Convert chains to standardized format."""
        if isinstance(chains, np.ndarray):
            if chains.ndim == 1:
                return [chains.reshape(-1, 1)]
            elif chains.ndim == 2:
                return [chains]
            else:
                raise ValueError("Chain array must be 1D or 2D")
        elif isinstance(chains, list):
            standardized = []
            for chain in chains:
                if chain.ndim == 1:
                    standardized.append(chain.reshape(-1, 1))
                else:
                    standardized.append(chain)
            return standardized
        else:
            raise TypeError("Chains must be numpy array or list of arrays")
    
    def _validate_inputs(self):
        """Validate input chains."""
        if self.n_chains == 0:
            raise ValueError("No chains provided")
        
        # Check consistent dimensions
        first_shape = self.chains[0].shape
        for i, chain in enumerate(self.chains):
            if chain.shape != first_shape:
                raise ValueError(f"Chain {i} has inconsistent shape")
        
        if self.n_samples < 10:
            warnings.warn("Very short chains may give unreliable diagnostics")
    
    def run_all_diagnostics(self, 
                          alpha: float = 0.05,
                          verbose: bool = True) -> Dict[str, List[ConvergenceResult]]:
        """
        Run all available convergence diagnostics.
        
        Args:
            alpha: Significance level for tests
            verbose: Print diagnostic results
            
        Returns:
            Dictionary with results for each diagnostic
        """
        results = {}
        
        if verbose:
            print("🔍 Running convergence diagnostics...")
            print(f"   Chains: {self.n_chains}")
            print(f"   Samples per chain: {self.n_samples}")
            print(f"   Parameters: {self.n_dims}")
        
        # Geweke diagnostic (within-chain)
        if verbose:
            print("\n   📊 Geweke diagnostic...")
        results['geweke'] = self.geweke_test(alpha=alpha)
        
        # Heidelberger-Welch test (stationarity)
        if verbose:
            print("   📊 Heidelberger-Welch test...")
        results['heidelberger_welch'] = self.heidelberger_welch_test(alpha=alpha)
        
        # Multiple chain diagnostics
        if self.n_chains > 1:
            if verbose:
                print("   📊 R-hat diagnostic...")
            results['r_hat'] = self.r_hat_test()
            
            if verbose:
                print("   📊 Between-chain comparison...")
            results['between_chain'] = self.between_chain_test(alpha=alpha)
        
        # Effective sample size check
        if verbose:
            print("   📊 Effective sample size check...")
        results['ess_check'] = self.effective_sample_size_check()
        
        # Autocorrelation analysis
        if verbose:
            print("   📊 Autocorrelation analysis...")
        results['autocorrelation'] = self.autocorrelation_test()
        
        # Summary
        if verbose:
            self._print_summary(results)
        
        return results
    
    def geweke_test(self, 
                   alpha: float = 0.05,
                   first_frac: float = 0.1,
                   last_frac: float = 0.5) -> List[ConvergenceResult]:
        """
        Run Geweke convergence test for each parameter.
        
        Tests if first and last portions of chain have same mean.
        
        Args:
            alpha: Significance level
            first_frac: Fraction of chain for first segment
            last_frac: Fraction of chain for last segment
            
        Returns:
            List of convergence results
        """
        results = []
        critical_value = stats.norm.ppf(1 - alpha/2)
        
        for dim in range(self.n_dims):
            param_name = self.parameter_names[dim]
            
            # Test each chain separately
            z_scores = []
            for chain_idx, chain in enumerate(self.chains):
                param_chain = chain[:, dim] if chain.ndim > 1 else chain
                z_score = geweke_diagnostic(param_chain, first_frac, last_frac)
                z_scores.append(z_score)
            
            # Use mean Z-score across chains
            mean_z = np.mean(z_scores)
            p_value = 2 * (1 - stats.norm.cdf(np.abs(mean_z)))
            converged = np.abs(mean_z) < critical_value
            
            result = ConvergenceResult(
                converged=converged,
                diagnostic_name='geweke',
                statistic=mean_z,
                p_value=p_value,
                critical_value=critical_value,
                message=f"{'✓' if converged else '✗'} {param_name}: Z={mean_z:.3f}, p={p_value:.4f}",
                details={'z_scores_per_chain': z_scores}
            )
            results.append(result)
        
        return results
    
    def heidelberger_welch_test(self, alpha: float = 0.05) -> List[ConvergenceResult]:
        """
        Run Heidelberger-Welch stationarity test.
        
        Args:
            alpha: Significance level
            
        Returns:
            List of convergence results
        """
        results = []
        
        for dim in range(self.n_dims):
            param_name = self.parameter_names[dim]
            
            # Test each chain
            chain_results = []
            for chain_idx, chain in enumerate(self.chains):
                param_chain = chain[:, dim] if chain.ndim > 1 else chain
                test_result = heidelberger_welch_test(param_chain, alpha)
                chain_results.append(test_result)
            
            # Aggregate results
            all_converged = all(res.get('converged', False) for res in chain_results)
            
            if all_converged:
                mean_ess = np.mean([res.get('effective_samples', 0) for res in chain_results])
                message = f"✓ {param_name}: Stationary (ESS={mean_ess:.1f})"
            else:
                reasons = [res.get('reason', 'unknown') for res in chain_results]
                message = f"✗ {param_name}: Non-stationary ({', '.join(set(reasons))})"
            
            result = ConvergenceResult(
                converged=all_converged,
                diagnostic_name='heidelberger_welch',
                statistic=0.0,  # No single statistic
                message=message,
                details={'chain_results': chain_results}
            )
            results.append(result)
        
        return results
    
    def r_hat_test(self, threshold: float = 1.1) -> List[ConvergenceResult]:
        """
        Compute R-hat convergence diagnostic.
        
        Args:
            threshold: Convergence threshold (typically 1.1)
            
        Returns:
            List of convergence results
        """
        if self.n_chains < 2:
            return [ConvergenceResult(
                converged=False,
                diagnostic_name='r_hat',
                statistic=0.0,
                message="Need multiple chains for R-hat"
            )]
        
        results = []
        
        for dim in range(self.n_dims):
            param_name = self.parameter_names[dim]
            
            # Extract parameter chains
            param_chains = [chain[:, dim] if chain.ndim > 1 else chain 
                           for chain in self.chains]
            
            # Compute R-hat
            r_hat = potential_scale_reduction_factor(param_chains)
            converged = r_hat < threshold
            
            result = ConvergenceResult(
                converged=converged,
                diagnostic_name='r_hat',
                statistic=r_hat,
                critical_value=threshold,
                message=f"{'✓' if converged else '✗'} {param_name}: R̂={r_hat:.4f}",
                details={'threshold': threshold}
            )
            results.append(result)
        
        return results
    
    def between_chain_test(self, alpha: float = 0.05) -> List[ConvergenceResult]:
        """
        Test if chains have converged to same distribution.
        
        Uses Kruskal-Wallis test to compare chain distributions.
        
        Args:
            alpha: Significance level
            
        Returns:
            List of convergence results
        """
        if self.n_chains < 2:
            return []
        
        results = []
        
        for dim in range(self.n_dims):
            param_name = self.parameter_names[dim]
            
            # Extract parameter chains
            param_chains = [chain[:, dim] if chain.ndim > 1 else chain 
                           for chain in self.chains]
            
            # Kruskal-Wallis test
            try:
                statistic, p_value = stats.kruskal(*param_chains)
                converged = p_value > alpha  # Want high p-value (chains same distribution)
                
                result = ConvergenceResult(
                    converged=converged,
                    diagnostic_name='between_chain',
                    statistic=statistic,
                    p_value=p_value,
                    message=f"{'✓' if converged else '✗'} {param_name}: KW={statistic:.3f}, p={p_value:.4f}",
                    details={'test': 'kruskal_wallis'}
                )
            except Exception as e:
                result = ConvergenceResult(
                    converged=False,
                    diagnostic_name='between_chain',
                    statistic=0.0,
                    message=f"✗ {param_name}: Test failed ({str(e)})"
                )
            
            results.append(result)
        
        return results
    
    def effective_sample_size_check(self, min_ess: int = 100) -> List[ConvergenceResult]:
        """
        Check if effective sample size is sufficient.
        
        Args:
            min_ess: Minimum required effective sample size
            
        Returns:
            List of convergence results
        """
        results = []
        
        for dim in range(self.n_dims):
            param_name = self.parameter_names[dim]
            
            # Compute ESS for each chain
            ess_values = []
            for chain in self.chains:
                param_chain = chain[:, dim] if chain.ndim > 1 else chain
                ess = effective_sample_size(param_chain)
                ess_values.append(ess)
            
            mean_ess = np.mean(ess_values)
            converged = mean_ess >= min_ess
            
            result = ConvergenceResult(
                converged=converged,
                diagnostic_name='ess_check',
                statistic=mean_ess,
                critical_value=min_ess,
                message=f"{'✓' if converged else '✗'} {param_name}: ESS={mean_ess:.1f} (min={min_ess})",
                details={'ess_per_chain': ess_values}
            )
            results.append(result)
        
        return results
    
    def autocorrelation_test(self, max_autocorr: float = 0.1) -> List[ConvergenceResult]:
        """
        Test autocorrelation at reasonable lag.
        
        Args:
            max_autocorr: Maximum acceptable autocorrelation
            
        Returns:
            List of convergence results
        """
        results = []
        test_lag = min(50, self.n_samples // 10)  # Test at reasonable lag
        
        for dim in range(self.n_dims):
            param_name = self.parameter_names[dim]
            
            autocorr_values = []
            for chain in self.chains:
                param_chain = chain[:, dim] if chain.ndim > 1 else chain
                
                # Compute autocorrelation at test lag
                autocorr = self._compute_autocorrelation_lag(param_chain, test_lag)
                autocorr_values.append(abs(autocorr))
            
            mean_autocorr = np.mean(autocorr_values)
            converged = mean_autocorr < max_autocorr
            
            result = ConvergenceResult(
                converged=converged,
                diagnostic_name='autocorrelation',
                statistic=mean_autocorr,
                critical_value=max_autocorr,
                message=f"{'✓' if converged else '✗'} {param_name}: ρ({test_lag})={mean_autocorr:.4f}",
                details={'test_lag': test_lag, 'autocorr_per_chain': autocorr_values}
            )
            results.append(result)
        
        return results
    
    def _compute_autocorrelation_lag(self, chain: np.ndarray, lag: int) -> float:
        """Compute autocorrelation at specific lag."""
        if lag >= len(chain) or lag <= 0:
            return 0.0
        
        try:
            x = chain - np.mean(chain)
            autocorr = np.corrcoef(x[:-lag], x[lag:])[0, 1]
            return autocorr if not np.isnan(autocorr) else 0.0
        except Exception:
            return 0.0
    
    def _print_summary(self, results: Dict[str, List[ConvergenceResult]]):
        """Print convergence diagnostic summary."""
        print("\n" + "="*60)
        print("CONVERGENCE DIAGNOSTIC SUMMARY")
        print("="*60)
        
        all_converged = True
        
        for diagnostic_name, diagnostic_results in results.items():
            print(f"\n{diagnostic_name.upper()}:")
            
            param_converged = [r.converged for r in diagnostic_results]
            n_converged = sum(param_converged)
            n_total = len(param_converged)
            
            print(f"  Converged: {n_converged}/{n_total} parameters")
            
            if n_converged < n_total:
                all_converged = False
            
            # Print details for each parameter
            for result in diagnostic_results:
                print(f"  {result.message}")
        
        print(f"\n{'='*60}")
        if all_converged:
            print("🎉 ALL DIAGNOSTICS PASSED - CHAINS HAVE CONVERGED")
        else:
            print("⚠️  SOME DIAGNOSTICS FAILED - CHAINS MAY NOT HAVE CONVERGED")
        print(f"{'='*60}\n")
    
    def convergence_summary(self) -> pd.DataFrame:
        """Generate convergence summary DataFrame."""
        # Run all diagnostics
        results = self.run_all_diagnostics(verbose=False)
        
        # Create summary DataFrame
        summary_data = []
        
        for param_idx, param_name in enumerate(self.parameter_names):
            row = {'Parameter': param_name}
            
            # Extract results for this parameter
            for diagnostic_name, diagnostic_results in results.items():
                if param_idx < len(diagnostic_results):
                    result = diagnostic_results[param_idx]
                    row[f'{diagnostic_name}_converged'] = result.converged
                    row[f'{diagnostic_name}_statistic'] = result.statistic
                    if result.p_value is not None:
                        row[f'{diagnostic_name}_pvalue'] = result.p_value
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def recommend_burnin(self, method: str = 'geweke') -> int:
        """
        Recommend burnin length based on convergence diagnostics.
        
        Args:
            method: Method to use for recommendation
            
        Returns:
            Recommended burnin samples
        """
        if method == 'geweke':
            return self._recommend_burnin_geweke()
        elif method == 'heidelberger_welch':
            return self._recommend_burnin_hw()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _recommend_burnin_geweke(self) -> int:
        """Recommend burnin using Geweke diagnostic."""
        max_burnin = 0
        
        for dim in range(self.n_dims):
            for chain in self.chains:
                param_chain = chain[:, dim] if chain.ndim > 1 else chain
                
                # Test different burnin lengths
                for burnin_frac in np.arange(0.1, 0.6, 0.1):
                    burnin_samples = int(burnin_frac * len(param_chain))
                    remaining_chain = param_chain[burnin_samples:]
                    
                    if len(remaining_chain) < 50:
                        continue
                    
                    z_score = geweke_diagnostic(remaining_chain)
                    
                    if abs(z_score) < 2.0:  # Converged
                        max_burnin = max(max_burnin, burnin_samples)
                        break
        
        return min(max_burnin, self.n_samples // 2)  # Cap at 50% of samples
    
    def _recommend_burnin_hw(self) -> int:
        """Recommend burnin using Heidelberger-Welch test."""
        max_burnin = 0
        
        for dim in range(self.n_dims):
            for chain in self.chains:
                param_chain = chain[:, dim] if chain.ndim > 1 else chain
                test_result = heidelberger_welch_test(param_chain)
                
                if test_result.get('converged', False):
                    burnin = test_result.get('burnin', 0)
                    max_burnin = max(max_burnin, burnin)
        
        return min(max_burnin, self.n_samples // 2)


def compare_multiple_runs(run_results: List[Dict[str, Any]], 
                         parameter_names: List[str] = None) -> Dict[str, Any]:
    """
    Compare convergence across multiple independent runs.
    
    Args:
        run_results: List of results from different runs
        parameter_names: Names of parameters
        
    Returns:
        Comparison results
    """
    n_runs = len(run_results)
    if n_runs < 2:
        raise ValueError("Need at least 2 runs for comparison")
    
    # Extract final samples from each run
    final_samples = []
    for result in run_results:
        if 'samples' in result:
            final_samples.append(result['samples'])
        else:
            raise ValueError("Run results must contain 'samples' key")
    
    # Check consistency of dimensions
    shapes = [samples.shape for samples in final_samples]
    if not all(shape[1:] == shapes[0][1:] for shape in shapes):
        raise ValueError("All runs must have same parameter dimensions")
    
    n_params = shapes[0][1] if len(shapes[0]) > 1 else 1
    param_names = parameter_names or [f'param_{i}' for i in range(n_params)]
    
    comparison = {}
    
    # For each parameter, test if runs converged to same distribution
    for param_idx in range(n_params):
        param_name = param_names[param_idx]
        
        # Extract parameter samples from each run
        param_samples = []
        for samples in final_samples:
            if samples.ndim == 1:
                param_samples.append(samples)
            else:
                param_samples.append(samples[:, param_idx])
        
        # Statistical tests
        try:
            # Kruskal-Wallis test
            kw_stat, kw_p = stats.kruskal(*param_samples)
            
            # ANOVA F-test
            f_stat, f_p = stats.f_oneway(*param_samples)
            
            # Means and standard deviations
            means = [np.mean(samples) for samples in param_samples]
            stds = [np.std(samples) for samples in param_samples]
            
            comparison[param_name] = {
                'kruskal_wallis': {'statistic': kw_stat, 'p_value': kw_p},
                'f_test': {'statistic': f_stat, 'p_value': f_p},
                'means': means,
                'stds': stds,
                'mean_of_means': np.mean(means),
                'std_of_means': np.std(means),
                'converged': kw_p > 0.05 and f_p > 0.05  # High p-value = same distribution
            }
            
        except Exception as e:
            comparison[param_name] = {
                'error': str(e),
                'converged': False
            }
    
    # Overall assessment
    all_converged = all(
        result.get('converged', False) 
        for result in comparison.values() 
        if 'converged' in result
    )
    
    comparison['overall'] = {
        'all_converged': all_converged,
        'n_runs': n_runs,
        'n_parameters': n_params
    }
    
    return comparison