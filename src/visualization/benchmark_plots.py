"""
Comprehensive visualization suite for MCMC benchmark results.

This module provides publication-quality plots for:
- Performance comparison across samplers
- Convergence diagnostics visualization
- Hessian analysis plots
- Dimensional scaling analysis
- Cost vs accuracy trade-off analysis
"""

import numpy as np
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle
    import seaborn as sns
    HAS_PLOTTING = True
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("Matplotlib/Seaborn not available. Plotting will be disabled.")

try:
    from ..benchmarks.performance_metrics import effective_sample_size, integrated_autocorr_time
except ImportError:
    from benchmarks.performance_metrics import effective_sample_size, integrated_autocorr_time


def _check_plotting():
    """Check if plotting libraries are available."""
    if not HAS_PLOTTING:
        raise ImportError("Plotting requires matplotlib and seaborn")


def plot_ess_comparison(benchmark_results: Dict[str, Any], 
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Create bar plot comparing effective sample sizes across samplers.
    
    Args:
        benchmark_results: Results from SamplerBenchmark.run_benchmark()
        save_path: Path to save plot
        figsize: Figure size
    """
    _check_plotting()
    
    # Extract data
    data_for_plot = []
    for dist_name, sampler_results in benchmark_results.items():
        for sampler_name, result in sampler_results.items():
            if hasattr(result, 'effective_sample_size') and result.effective_sample_size:
                data_for_plot.append({
                    'Distribution': dist_name,
                    'Sampler': sampler_name,
                    'ESS': result.effective_sample_size,
                    'ESS_per_second': result.ess_per_second or 0
                })
    
    if not data_for_plot:
        warnings.warn("No ESS data available for plotting")
        return
    
    df = pd.DataFrame(data_for_plot)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ESS comparison
    sns.barplot(data=df, x='Distribution', y='ESS', hue='Sampler', ax=ax1)
    ax1.set_title('Effective Sample Size by Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Effective Sample Size', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # ESS per second comparison
    sns.barplot(data=df, x='Distribution', y='ESS_per_second', hue='Sampler', ax=ax2)
    ax2.set_title('ESS per Second by Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('ESS per Second', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ESS comparison plot saved to {save_path}")
    else:
        plt.show()


def plot_convergence_traces(chains_dict: Dict[str, np.ndarray], 
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (14, 10),
                          max_dims: int = 4) -> None:
    """
    Create trace plots for visual convergence assessment.
    
    Args:
        chains_dict: Dictionary of {sampler_name: samples_array}
        save_path: Path to save plot
        figsize: Figure size
        max_dims: Maximum dimensions to plot
    """
    _check_plotting()
    
    if not chains_dict:
        warnings.warn("No chains provided for trace plots")
        return
    
    # Determine dimensions to plot
    first_chain = next(iter(chains_dict.values()))
    n_dims = first_chain.shape[1] if first_chain.ndim > 1 else 1
    n_dims_plot = min(n_dims, max_dims)
    
    n_samplers = len(chains_dict)
    
    fig, axes = plt.subplots(n_dims_plot, n_samplers, figsize=figsize)
    if n_dims_plot == 1 and n_samplers == 1:
        axes = np.array([[axes]])
    elif n_dims_plot == 1:
        axes = axes.reshape(1, -1)
    elif n_samplers == 1:
        axes = axes.reshape(-1, 1)
    
    for sampler_idx, (sampler_name, samples) in enumerate(chains_dict.items()):
        for dim in range(n_dims_plot):
            ax = axes[dim, sampler_idx]
            
            if samples.ndim == 1:
                trace_data = samples
            else:
                trace_data = samples[:, dim]
            
            ax.plot(trace_data, alpha=0.8, linewidth=0.8)
            
            # Add running mean
            window = max(50, len(trace_data) // 50)
            running_mean = pd.Series(trace_data).rolling(window=window, center=True).mean()
            ax.plot(running_mean, 'r-', linewidth=2, alpha=0.8, label='Running mean')
            
            if dim == 0:
                ax.set_title(f'{sampler_name}', fontsize=12, fontweight='bold')
            if sampler_idx == 0:
                ax.set_ylabel(f'Parameter {dim}', fontsize=10)
            if dim == n_dims_plot - 1:
                ax.set_xlabel('Iteration', fontsize=10)
            
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('MCMC Trace Plots', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trace plots saved to {save_path}")
    else:
        plt.show()


def plot_autocorrelation_functions(chains_dict: Dict[str, np.ndarray], 
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 8),
                                 max_lag: int = 100) -> None:
    """
    Plot autocorrelation functions for different samplers.
    
    Args:
        chains_dict: Dictionary of {sampler_name: samples_array}
        save_path: Path to save plot
        figsize: Figure size
        max_lag: Maximum lag to compute
    """
    _check_plotting()
    
    if not chains_dict:
        warnings.warn("No chains provided for ACF plots")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for sampler_name, samples in chains_dict.items():
        # Use first dimension
        if samples.ndim > 1:
            chain = samples[:, 0]
        else:
            chain = samples
        
        # Compute autocorrelation
        autocorr = _compute_autocorr_function(chain, max_lag)
        lags = np.arange(len(autocorr))
        
        ax.plot(lags, autocorr, label=sampler_name, linewidth=2, alpha=0.8)
    
    # Add significance bounds
    n = len(next(iter(chains_dict.values())))
    significance_bound = 1.96 / np.sqrt(n)
    ax.axhline(y=significance_bound, color='red', linestyle='--', alpha=0.7, 
               label='95% significance')
    ax.axhline(y=-significance_bound, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_title('Autocorrelation Functions', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ACF plot saved to {save_path}")
    else:
        plt.show()


def _compute_autocorr_function(chain: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation function."""
    chain = chain - np.mean(chain)
    n = len(chain)
    max_lag = min(max_lag, n // 2)
    
    autocorr = np.zeros(max_lag)
    variance = np.var(chain)
    
    if variance == 0:
        return autocorr
    
    for lag in range(max_lag):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            autocorr[lag] = np.mean(chain[:-lag] * chain[lag:]) / variance
    
    return autocorr


def plot_hessian_conditioning(hessian_history: List[np.ndarray], 
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot Hessian condition numbers over sampling iterations.
    
    Args:
        hessian_history: List of Hessian matrices over time
        save_path: Path to save plot
        figsize: Figure size
    """
    _check_plotting()
    
    if not hessian_history:
        warnings.warn("No Hessian history provided")
        return
    
    # Compute condition numbers
    condition_numbers = []
    for hessian in hessian_history:
        try:
            cond = np.linalg.cond(hessian)
            condition_numbers.append(cond)
        except:
            condition_numbers.append(np.nan)
    
    iterations = np.arange(len(condition_numbers))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Linear scale
    ax1.plot(iterations, condition_numbers, 'b-', alpha=0.8, linewidth=1.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Condition Number')
    ax1.set_title('Hessian Conditioning (Linear Scale)')
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    valid_cond = [c for c in condition_numbers if not np.isnan(c) and c > 0]
    if valid_cond:
        ax2.semilogy(iterations[:len(valid_cond)], valid_cond, 'r-', alpha=0.8, linewidth=1.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Condition Number (log scale)')
        ax2.set_title('Hessian Conditioning (Log Scale)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Hessian conditioning plot saved to {save_path}")
    else:
        plt.show()


def plot_eigenvalue_spectrum(hessian_matrices: List[np.ndarray], 
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Analyze and plot eigenvalue distributions of Hessian matrices.
    
    Args:
        hessian_matrices: List of Hessian matrices
        save_path: Path to save plot
        figsize: Figure size
    """
    _check_plotting()
    
    if not hessian_matrices:
        warnings.warn("No Hessian matrices provided")
        return
    
    # Compute eigenvalues
    all_eigenvalues = []
    for hessian in hessian_matrices:
        try:
            eigenvals = np.linalg.eigvals(hessian)
            eigenvals = eigenvals[eigenvals > 1e-12]  # Remove near-zero eigenvalues
            all_eigenvalues.extend(eigenvals.real)
        except:
            continue
    
    if not all_eigenvalues:
        warnings.warn("No valid eigenvalues computed")
        return
    
    all_eigenvalues = np.array(all_eigenvalues)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Histogram of eigenvalues
    axes[0, 0].hist(all_eigenvalues, bins=50, alpha=0.7, density=True, edgecolor='black')
    axes[0, 0].set_xlabel('Eigenvalue')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Eigenvalue Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Log-histogram
    log_eigenvals = np.log10(all_eigenvalues[all_eigenvalues > 0])
    if len(log_eigenvals) > 0:
        axes[0, 1].hist(log_eigenvals, bins=50, alpha=0.7, density=True, edgecolor='black')
        axes[0, 1].set_xlabel('log₁₀(Eigenvalue)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Log Eigenvalue Distribution')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Condition number evolution
    condition_numbers = []
    for hessian in hessian_matrices:
        try:
            cond = np.linalg.cond(hessian)
            condition_numbers.append(cond)
        except:
            condition_numbers.append(np.nan)
    
    axes[1, 0].plot(condition_numbers, 'g-', alpha=0.8)
    axes[1, 0].set_xlabel('Matrix Index')
    axes[1, 0].set_ylabel('Condition Number')
    axes[1, 0].set_title('Condition Number Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Eigenvalue spectrum (sorted)
    if len(hessian_matrices) > 0:
        # Use last matrix as representative
        try:
            eigenvals = np.linalg.eigvals(hessian_matrices[-1])
            eigenvals = np.sort(eigenvals.real)[::-1]  # Sort descending
            
            axes[1, 1].semilogy(eigenvals, 'ro-', alpha=0.7, markersize=4)
            axes[1, 1].set_xlabel('Eigenvalue Index')
            axes[1, 1].set_ylabel('Eigenvalue (log scale)')
            axes[1, 1].set_title('Eigenvalue Spectrum (Last Matrix)')
            axes[1, 1].grid(True, alpha=0.3)
        except:
            axes[1, 1].text(0.5, 0.5, 'Eigenvalue computation failed', 
                           transform=axes[1, 1].transAxes, ha='center', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Eigenvalue spectrum plot saved to {save_path}")
    else:
        plt.show()


def plot_dimensional_scaling(results_by_dim: Dict[int, Dict[str, Any]], 
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (14, 10)) -> None:
    """
    Plot performance vs dimension for scaling analysis.
    
    Args:
        results_by_dim: Dictionary of {dimension: benchmark_results}
        save_path: Path to save plot
        figsize: Figure size
    """
    _check_plotting()
    
    if not results_by_dim:
        warnings.warn("No dimensional results provided")
        return
    
    # Extract data
    scaling_data = []
    for dim, benchmark_results in results_by_dim.items():
        for dist_name, sampler_results in benchmark_results.items():
            if isinstance(sampler_results, dict):  # Handle nested structure
                for sampler_name, result in sampler_results.items():
                    if hasattr(result, 'effective_sample_size'):
                        scaling_data.append({
                            'Dimension': dim,
                            'Distribution': dist_name,
                            'Sampler': sampler_name,
                            'ESS': result.effective_sample_size or 0,
                            'ESS_per_second': result.ess_per_second or 0,
                            'Time': result.sampling_time or 0,
                            'Acceptance_rate': result.acceptance_rate or 0
                        })
    
    if not scaling_data:
        warnings.warn("No scaling data available for plotting")
        return
    
    df = pd.DataFrame(scaling_data)
    dimensions = sorted(df['Dimension'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # ESS vs Dimension
    for sampler in df['Sampler'].unique():
        sampler_data = df[df['Sampler'] == sampler]
        mean_ess = sampler_data.groupby('Dimension')['ESS'].mean()
        std_ess = sampler_data.groupby('Dimension')['ESS'].std()
        
        axes[0, 0].errorbar(mean_ess.index, mean_ess.values, yerr=std_ess.values,
                           marker='o', label=sampler, alpha=0.8, capsize=5)
    
    axes[0, 0].set_xlabel('Dimension')
    axes[0, 0].set_ylabel('Effective Sample Size')
    axes[0, 0].set_title('ESS vs Dimension')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # ESS per second vs Dimension
    for sampler in df['Sampler'].unique():
        sampler_data = df[df['Sampler'] == sampler]
        mean_ess_per_sec = sampler_data.groupby('Dimension')['ESS_per_second'].mean()
        std_ess_per_sec = sampler_data.groupby('Dimension')['ESS_per_second'].std()
        
        axes[0, 1].errorbar(mean_ess_per_sec.index, mean_ess_per_sec.values, 
                           yerr=std_ess_per_sec.values, marker='s', label=sampler, 
                           alpha=0.8, capsize=5)
    
    axes[0, 1].set_xlabel('Dimension')
    axes[0, 1].set_ylabel('ESS per Second')
    axes[0, 1].set_title('ESS/sec vs Dimension')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Sampling time vs Dimension
    for sampler in df['Sampler'].unique():
        sampler_data = df[df['Sampler'] == sampler]
        mean_time = sampler_data.groupby('Dimension')['Time'].mean()
        std_time = sampler_data.groupby('Dimension')['Time'].std()
        
        axes[1, 0].errorbar(mean_time.index, mean_time.values, yerr=std_time.values,
                           marker='^', label=sampler, alpha=0.8, capsize=5)
    
    axes[1, 0].set_xlabel('Dimension')
    axes[1, 0].set_ylabel('Sampling Time (s)')
    axes[1, 0].set_title('Time vs Dimension')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Acceptance rate vs Dimension
    for sampler in df['Sampler'].unique():
        sampler_data = df[df['Sampler'] == sampler]
        mean_accept = sampler_data.groupby('Dimension')['Acceptance_rate'].mean()
        std_accept = sampler_data.groupby('Dimension')['Acceptance_rate'].std()
        
        axes[1, 1].errorbar(mean_accept.index, mean_accept.values, yerr=std_accept.values,
                           marker='d', label=sampler, alpha=0.8, capsize=5)
    
    axes[1, 1].set_xlabel('Dimension')
    axes[1, 1].set_ylabel('Acceptance Rate')
    axes[1, 1].set_title('Acceptance Rate vs Dimension')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    plt.suptitle('Dimensional Scaling Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dimensional scaling plot saved to {save_path}")
    else:
        plt.show()


def plot_cost_vs_accuracy_tradeoff(benchmark_results: Dict[str, Any], 
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot Pareto frontier of cost vs accuracy trade-off.
    
    Args:
        benchmark_results: Results from SamplerBenchmark
        save_path: Path to save plot
        figsize: Figure size
    """
    _check_plotting()
    
    # Extract data
    plot_data = []
    for dist_name, sampler_results in benchmark_results.items():
        for sampler_name, result in sampler_results.items():
            if hasattr(result, 'sampling_time') and hasattr(result, 'effective_sample_size'):
                time_per_ess = (result.sampling_time / result.effective_sample_size 
                              if result.effective_sample_size > 0 else np.inf)
                
                accuracy = result.effective_sample_size / result.n_samples if result.n_samples > 0 else 0
                
                plot_data.append({
                    'Distribution': dist_name,
                    'Sampler': sampler_name,
                    'Time_per_ESS': time_per_ess,
                    'ESS_efficiency': accuracy,
                    'ESS': result.effective_sample_size or 0,
                    'Time': result.sampling_time or 0
                })
    
    if not plot_data:
        warnings.warn("No cost-accuracy data available")
        return
    
    df = pd.DataFrame(plot_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Cost vs Accuracy scatter
    for dist in df['Distribution'].unique():
        dist_data = df[df['Distribution'] == dist]
        for sampler in dist_data['Sampler'].unique():
            sampler_data = dist_data[dist_data['Sampler'] == sampler]
            
            ax1.scatter(sampler_data['ESS_efficiency'], sampler_data['Time_per_ESS'],
                       label=f'{sampler} ({dist})', alpha=0.7, s=100)
    
    ax1.set_xlabel('ESS Efficiency (ESS/N_samples)')
    ax1.set_ylabel('Time per ESS (seconds)')
    ax1.set_title('Cost vs Accuracy Trade-off')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # ESS vs Time scatter
    colors = plt.cm.tab10(np.linspace(0, 1, len(df['Sampler'].unique())))
    sampler_colors = {sampler: color for sampler, color in 
                     zip(df['Sampler'].unique(), colors)}
    
    for sampler in df['Sampler'].unique():
        sampler_data = df[df['Sampler'] == sampler]
        ax2.scatter(sampler_data['Time'], sampler_data['ESS'], 
                   label=sampler, alpha=0.7, s=100, color=sampler_colors[sampler])
    
    ax2.set_xlabel('Sampling Time (seconds)')
    ax2.set_ylabel('Effective Sample Size')
    ax2.set_title('ESS vs Sampling Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cost vs accuracy plot saved to {save_path}")
    else:
        plt.show()


def create_benchmark_dashboard(benchmark_results: Dict[str, Any],
                             chains_dict: Optional[Dict[str, np.ndarray]] = None,
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (20, 16)) -> None:
    """
    Create comprehensive dashboard with all benchmark visualizations.
    
    Args:
        benchmark_results: Results from SamplerBenchmark
        chains_dict: Optional chains for trace plots
        save_path: Path to save dashboard
        figsize: Figure size
    """
    _check_plotting()
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Extract data for plotting
    plot_data = []
    for dist_name, sampler_results in benchmark_results.items():
        for sampler_name, result in sampler_results.items():
            if hasattr(result, 'effective_sample_size'):
                plot_data.append({
                    'Distribution': dist_name,
                    'Sampler': sampler_name,
                    'ESS': result.effective_sample_size or 0,
                    'ESS_per_second': result.ess_per_second or 0,
                    'Time': result.sampling_time or 0,
                    'Acceptance': result.acceptance_rate or 0
                })
    
    if not plot_data:
        warnings.warn("No data available for dashboard")
        return
    
    df = pd.DataFrame(plot_data)
    
    # ESS comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.barplot(data=df, x='Distribution', y='ESS', hue='Sampler', ax=ax1)
    ax1.set_title('Effective Sample Size', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # ESS per second (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    sns.barplot(data=df, x='Distribution', y='ESS_per_second', hue='Sampler', ax=ax2)
    ax2.set_title('ESS per Second', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend().remove()
    
    # Acceptance rates (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    sns.barplot(data=df, x='Distribution', y='Acceptance', hue='Sampler', ax=ax3)
    ax3.set_title('Acceptance Rates', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend().remove()
    
    # Cost vs accuracy (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    for sampler in df['Sampler'].unique():
        sampler_data = df[df['Sampler'] == sampler]
        ax4.scatter(sampler_data['ESS'], sampler_data['Time'], 
                   label=sampler, alpha=0.7, s=60)
    ax4.set_xlabel('ESS')
    ax4.set_ylabel('Time (s)')
    ax4.set_title('ESS vs Time')
    ax4.legend()
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    
    # Performance heatmap (middle middle and right)
    ax5 = fig.add_subplot(gs[1, 1:])
    pivot_ess = df.pivot_table(values='ESS_per_second', index='Distribution', 
                              columns='Sampler', aggfunc='mean')
    sns.heatmap(pivot_ess, annot=True, fmt='.1f', cmap='viridis', ax=ax5)
    ax5.set_title('ESS/sec Heatmap', fontweight='bold')
    
    # Trace plots (bottom row)
    if chains_dict and len(chains_dict) > 0:
        n_samplers = min(3, len(chains_dict))
        for i, (sampler_name, samples) in enumerate(list(chains_dict.items())[:n_samplers]):
            ax = fig.add_subplot(gs[2:, i])
            
            if samples.ndim > 1:
                trace_data = samples[:, 0]  # First dimension
            else:
                trace_data = samples
            
            ax.plot(trace_data, alpha=0.8, linewidth=0.8)
            ax.set_title(f'{sampler_name} Trace')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('MCMC Sampler Benchmark Dashboard', fontsize=20, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Benchmark dashboard saved to {save_path}")
    else:
        plt.show()


def save_all_plots(benchmark_results: Dict[str, Any],
                  chains_dict: Optional[Dict[str, np.ndarray]] = None,
                  output_dir: str = "benchmark_plots",
                  hessian_history: Optional[List[np.ndarray]] = None) -> None:
    """
    Generate and save all benchmark plots.
    
    Args:
        benchmark_results: Results from SamplerBenchmark
        chains_dict: Optional chains for additional plots
        output_dir: Directory to save plots
        hessian_history: Optional Hessian matrices for analysis
    """
    _check_plotting()
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📊 Generating benchmark plots in {output_dir}/")
    
    # ESS comparison
    plot_ess_comparison(
        benchmark_results, 
        save_path=os.path.join(output_dir, "ess_comparison.png")
    )
    
    # Cost vs accuracy
    plot_cost_vs_accuracy_tradeoff(
        benchmark_results,
        save_path=os.path.join(output_dir, "cost_vs_accuracy.png")
    )
    
    # Dashboard
    create_benchmark_dashboard(
        benchmark_results,
        chains_dict=chains_dict,
        save_path=os.path.join(output_dir, "benchmark_dashboard.png")
    )
    
    if chains_dict:
        # Trace plots
        plot_convergence_traces(
            chains_dict,
            save_path=os.path.join(output_dir, "trace_plots.png")
        )
        
        # Autocorrelation
        plot_autocorrelation_functions(
            chains_dict,
            save_path=os.path.join(output_dir, "autocorrelation.png")
        )
    
    if hessian_history:
        # Hessian analysis
        plot_hessian_conditioning(
            hessian_history,
            save_path=os.path.join(output_dir, "hessian_conditioning.png")
        )
        
        plot_eigenvalue_spectrum(
            hessian_history,
            save_path=os.path.join(output_dir, "eigenvalue_spectrum.png")
        )
    
    print(f"✅ All plots saved to {output_dir}/")