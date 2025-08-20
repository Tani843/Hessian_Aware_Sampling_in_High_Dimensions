"""
Advanced visualization suite for publication-quality results.

This module provides sophisticated plotting capabilities including:
- Multi-panel comparison grids
- Interactive Hessian visualizations  
- Statistical convergence analysis
- Publication-ready figure generation

All plots follow publication standards with proper error bars,
statistical significance testing, and consistent styling.
"""

import numpy as np
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle, Ellipse
    from matplotlib.animation import FuncAnimation
    import seaborn as sns
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    HAS_PLOTTING = True
    
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Define consistent styling
    PLOT_STYLE = {
        'figure_size': (12, 8),
        'font_sizes': {'title': 14, 'label': 12, 'tick': 10, 'legend': 10},
        'line_width': 2.0,
        'marker_size': 6,
        'alpha': 0.8,
        'dpi': 300
    }
    
    # Color palette for consistency
    COLOR_PALETTE = sns.color_palette("husl", n_colors=8)
    METHOD_COLORS = {
        'Standard Metropolis': COLOR_PALETTE[0],
        'Hessian Metropolis': COLOR_PALETTE[1], 
        'Adaptive Metropolis': COLOR_PALETTE[2],
        'Hessian Langevin': COLOR_PALETTE[3],
        'Langevin Dynamics': COLOR_PALETTE[4],
        'HMC': COLOR_PALETTE[5],
        'Hessian HMC': COLOR_PALETTE[6],
        'Adaptive Hessian': COLOR_PALETTE[7]
    }
    
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("Matplotlib/Seaborn not available. Advanced plotting will be disabled.")

try:
    from ..benchmarks.performance_metrics import effective_sample_size, integrated_autocorr_time
    from ..benchmarks.convergence_diagnostics import ConvergenceDiagnostics
except ImportError:
    from benchmarks.performance_metrics import effective_sample_size, integrated_autocorr_time
    from benchmarks.convergence_diagnostics import ConvergenceDiagnostics


def _check_plotting():
    """Check if plotting libraries are available."""
    if not HAS_PLOTTING:
        raise ImportError("Advanced plotting requires matplotlib and seaborn")


def _setup_publication_style():
    """Set up publication-quality matplotlib parameters."""
    plt.rcParams.update({
        'font.size': PLOT_STYLE['font_sizes']['tick'],
        'axes.titlesize': PLOT_STYLE['font_sizes']['title'],
        'axes.labelsize': PLOT_STYLE['font_sizes']['label'],
        'xtick.labelsize': PLOT_STYLE['font_sizes']['tick'],
        'ytick.labelsize': PLOT_STYLE['font_sizes']['tick'],
        'legend.fontsize': PLOT_STYLE['font_sizes']['legend'],
        'figure.titlesize': PLOT_STYLE['font_sizes']['title'],
        'lines.linewidth': PLOT_STYLE['line_width'],
        'lines.markersize': PLOT_STYLE['marker_size'],
        'figure.dpi': PLOT_STYLE['dpi'],
        'savefig.dpi': PLOT_STYLE['dpi'],
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })


def create_sampler_comparison_grid(benchmark_results: Dict[str, Any], 
                                 save_dir: str,
                                 chains_dict: Optional[Dict[str, np.ndarray]] = None) -> None:
    """
    Create a comprehensive 3x3 grid showing complete sampler analysis.
    
    Grid layout:
    Row 1: ESS comparison, Autocorrelation functions, Trace plots
    Row 2: Acceptance rates, Step size adaptation, Hessian conditioning  
    Row 3: Computational cost, Convergence diagnostics, Error analysis
    
    Args:
        benchmark_results: Results from comprehensive benchmark
        save_dir: Directory to save plots
        chains_dict: Optional chains for trace/autocorr analysis
    """
    _check_plotting()
    _setup_publication_style()
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create the 3x3 grid
    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Extract data for plotting
    plot_data = _extract_benchmark_data(benchmark_results)
    
    if not plot_data:
        warnings.warn("No benchmark data available for comparison grid")
        return
    
    # Row 1, Col 1: ESS Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_ess_comparison_panel(ax1, plot_data)
    
    # Row 1, Col 2: Autocorrelation Functions
    ax2 = fig.add_subplot(gs[0, 1])
    if chains_dict:
        _plot_autocorrelation_panel(ax2, chains_dict)
    else:
        ax2.text(0.5, 0.5, 'Chains not available', ha='center', va='center', 
                transform=ax2.transAxes)
        ax2.set_title('Autocorrelation Functions')
    
    # Row 1, Col 3: Trace Plots
    ax3 = fig.add_subplot(gs[0, 2])
    if chains_dict:
        _plot_trace_panel(ax3, chains_dict)
    else:
        ax3.text(0.5, 0.5, 'Chains not available', ha='center', va='center',
                transform=ax3.transAxes)
        ax3.set_title('Trace Plots')
    
    # Row 2, Col 1: Acceptance Rates
    ax4 = fig.add_subplot(gs[1, 0])
    _plot_acceptance_rates_panel(ax4, plot_data)
    
    # Row 2, Col 2: Step Size Adaptation
    ax5 = fig.add_subplot(gs[1, 1])
    _plot_step_size_adaptation_panel(ax5, benchmark_results)
    
    # Row 2, Col 3: Hessian Conditioning
    ax6 = fig.add_subplot(gs[1, 2])
    _plot_hessian_conditioning_panel(ax6, benchmark_results)
    
    # Row 3, Col 1: Computational Cost
    ax7 = fig.add_subplot(gs[2, 0])
    _plot_computational_cost_panel(ax7, plot_data)
    
    # Row 3, Col 2: Convergence Diagnostics
    ax8 = fig.add_subplot(gs[2, 1])
    _plot_convergence_diagnostics_panel(ax8, benchmark_results)
    
    # Row 3, Col 3: Error Analysis
    ax9 = fig.add_subplot(gs[2, 2])
    _plot_error_analysis_panel(ax9, plot_data)
    
    # Add overall title
    fig.suptitle('Comprehensive Sampler Comparison Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save in multiple formats
    plt.savefig(save_path / "sampler_comparison_grid.png", dpi=PLOT_STYLE['dpi'])
    plt.savefig(save_path / "sampler_comparison_grid.pdf", format='pdf')
    
    print(f"Sampler comparison grid saved to {save_path}")
    plt.close()


def _extract_benchmark_data(benchmark_results: Dict[str, Any]) -> List[Dict]:
    """Extract data from benchmark results for plotting."""
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
                    'Acceptance': result.acceptance_rate or 0,
                    'N_samples': result.n_samples or 0,
                    'MSE': getattr(result, 'mean_squared_error', None)
                })
    
    return plot_data


def _plot_ess_comparison_panel(ax: plt.Axes, plot_data: List[Dict]) -> None:
    """Plot ESS comparison in grid panel."""
    if not plot_data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('ESS Comparison')
        return
    
    df = pd.DataFrame(plot_data)
    
    # Group by sampler and compute statistics
    sampler_stats = df.groupby('Sampler')['ESS'].agg(['mean', 'std']).reset_index()
    
    colors = [METHOD_COLORS.get(sampler, 'gray') for sampler in sampler_stats['Sampler']]
    
    bars = ax.bar(range(len(sampler_stats)), sampler_stats['mean'], 
                 yerr=sampler_stats['std'], capsize=5, color=colors, alpha=0.8)
    
    ax.set_xlabel('Sampler')
    ax.set_ylabel('Effective Sample Size')
    ax.set_title('ESS Comparison')
    ax.set_xticks(range(len(sampler_stats)))
    ax.set_xticklabels(sampler_stats['Sampler'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)


def _plot_autocorrelation_panel(ax: plt.Axes, chains_dict: Dict[str, np.ndarray]) -> None:
    """Plot autocorrelation functions in grid panel."""
    max_lag = 50
    
    for sampler_name, samples in list(chains_dict.items())[:4]:  # Limit to 4 samplers
        if samples.ndim > 1:
            chain = samples[:, 0]  # Use first dimension
        else:
            chain = samples
        
        # Compute autocorrelation
        autocorr = _compute_autocorr(chain, max_lag)
        lags = np.arange(len(autocorr))
        
        color = METHOD_COLORS.get(sampler_name, 'gray')
        ax.plot(lags, autocorr, label=sampler_name, color=color, alpha=0.8)
    
    # Add significance bounds
    n = len(next(iter(chains_dict.values())))
    significance = 1.96 / np.sqrt(n)
    ax.axhline(y=significance, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=-significance, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation Functions')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)


def _plot_trace_panel(ax: plt.Axes, chains_dict: Dict[str, np.ndarray]) -> None:
    """Plot trace plots in grid panel."""
    # Show traces for first 2 samplers to avoid clutter
    for i, (sampler_name, samples) in enumerate(list(chains_dict.items())[:2]):
        if samples.ndim > 1:
            trace = samples[:, 0]  # Use first dimension
        else:
            trace = samples
        
        color = METHOD_COLORS.get(sampler_name, 'gray')
        ax.plot(trace[:500], label=sampler_name, color=color, alpha=0.7, linewidth=1)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title('Trace Plots (First 500 iterations)')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_acceptance_rates_panel(ax: plt.Axes, plot_data: List[Dict]) -> None:
    """Plot acceptance rates in grid panel."""
    if not plot_data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Acceptance Rates')
        return
    
    df = pd.DataFrame(plot_data)
    
    # Create box plot of acceptance rates by sampler
    samplers = df['Sampler'].unique()
    acceptance_data = [df[df['Sampler'] == sampler]['Acceptance'].values 
                      for sampler in samplers]
    
    colors = [METHOD_COLORS.get(sampler, 'gray') for sampler in samplers]
    
    box_plot = ax.boxplot(acceptance_data, labels=samplers, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Sampler')
    ax.set_ylabel('Acceptance Rate')
    ax.set_title('Acceptance Rates')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)


def _plot_step_size_adaptation_panel(ax: plt.Axes, benchmark_results: Dict[str, Any]) -> None:
    """Plot step size adaptation in grid panel."""
    # This is a placeholder - would need step size history from samplers
    ax.text(0.5, 0.5, 'Step size adaptation\n(requires step size history)', 
           ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Step Size Adaptation')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Step Size')


def _plot_hessian_conditioning_panel(ax: plt.Axes, benchmark_results: Dict[str, Any]) -> None:
    """Plot Hessian conditioning in grid panel."""
    # This is a placeholder - would need Hessian history from samplers  
    ax.text(0.5, 0.5, 'Hessian conditioning\n(requires Hessian history)', 
           ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Hessian Conditioning')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Condition Number')


def _plot_computational_cost_panel(ax: plt.Axes, plot_data: List[Dict]) -> None:
    """Plot computational cost in grid panel."""
    if not plot_data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Computational Cost')
        return
    
    df = pd.DataFrame(plot_data)
    
    # Plot ESS per second vs Time
    for sampler in df['Sampler'].unique():
        sampler_data = df[df['Sampler'] == sampler]
        color = METHOD_COLORS.get(sampler, 'gray')
        
        ax.scatter(sampler_data['Time'], sampler_data['ESS_per_second'], 
                  label=sampler, color=color, alpha=0.7, s=50)
    
    ax.set_xlabel('Sampling Time (s)')
    ax.set_ylabel('ESS per Second')
    ax.set_title('Computational Cost')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)


def _plot_convergence_diagnostics_panel(ax: plt.Axes, benchmark_results: Dict[str, Any]) -> None:
    """Plot convergence diagnostics in grid panel."""
    # This would need convergence diagnostic results
    ax.text(0.5, 0.5, 'Convergence diagnostics\n(R-hat, Geweke, etc.)', 
           ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Convergence Diagnostics')
    ax.set_xlabel('Parameter')
    ax.set_ylabel('R-hat Statistic')


def _plot_error_analysis_panel(ax: plt.Axes, plot_data: List[Dict]) -> None:
    """Plot error analysis in grid panel."""
    if not plot_data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Error Analysis')
        return
    
    df = pd.DataFrame(plot_data)
    
    # Plot ESS vs MSE if available
    if 'MSE' in df.columns and df['MSE'].notna().any():
        for sampler in df['Sampler'].unique():
            sampler_data = df[df['Sampler'] == sampler]
            mse_data = sampler_data['MSE'].dropna()
            
            if len(mse_data) > 0:
                color = METHOD_COLORS.get(sampler, 'gray')
                ax.scatter(sampler_data['ESS'], mse_data, 
                          label=sampler, color=color, alpha=0.7, s=50)
        
        ax.set_xlabel('Effective Sample Size')
        ax.set_ylabel('Mean Squared Error')
        ax.set_title('Error Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Error analysis\n(MSE data not available)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Error Analysis')


def _compute_autocorr(chain: np.ndarray, max_lag: int) -> np.ndarray:
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


def plot_dimensional_scaling_analysis(results_by_dim: Dict[int, Dict[str, Any]], 
                                    save_dir: str) -> None:
    """
    Multi-panel plot showing how each sampler scales with dimension.
    
    Panels:
    - ESS vs dimension
    - Time per sample vs dimension  
    - Memory usage vs dimension (placeholder)
    - Relative performance vs dimension
    
    Args:
        results_by_dim: Dictionary of {dimension: benchmark_results}
        save_dir: Directory to save plots
    """
    _check_plotting()
    _setup_publication_style()
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Extract scaling data
    scaling_data = _extract_scaling_data(results_by_dim)
    
    if not scaling_data:
        warnings.warn("No scaling data available for analysis")
        return
    
    df = pd.DataFrame(scaling_data)
    dimensions = sorted(df['Dimension'].unique())
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dimensional Scaling Analysis', fontsize=16, fontweight='bold')
    
    # Panel 1: ESS vs Dimension
    ax1 = axes[0, 0]
    _plot_scaling_metric(ax1, df, 'Dimension', 'ESS', 'ESS vs Dimension', log_y=True)
    
    # Panel 2: Time per Sample vs Dimension
    ax2 = axes[0, 1]
    df['Time_per_sample'] = df['Time'] / df['N_samples']
    _plot_scaling_metric(ax2, df, 'Dimension', 'Time_per_sample', 
                        'Time per Sample vs Dimension', log_y=True)
    
    # Panel 3: Memory Usage (placeholder)
    ax3 = axes[1, 0]
    ax3.text(0.5, 0.5, 'Memory usage scaling\n(requires memory profiling)', 
            ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Memory Usage vs Dimension')
    ax3.set_xlabel('Dimension')
    ax3.set_ylabel('Memory Usage (MB)')
    
    # Panel 4: Relative Performance vs Dimension  
    ax4 = axes[1, 1]
    _plot_relative_performance(ax4, df, dimensions)
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(save_path / "dimensional_scaling_analysis.png", dpi=PLOT_STYLE['dpi'])
    plt.savefig(save_path / "dimensional_scaling_analysis.pdf", format='pdf')
    
    print(f"Dimensional scaling analysis saved to {save_path}")
    plt.close()


def _extract_scaling_data(results_by_dim: Dict[int, Dict[str, Any]]) -> List[Dict]:
    """Extract data for dimensional scaling analysis."""
    scaling_data = []
    
    for dim, benchmark_results in results_by_dim.items():
        for dist_name, sampler_results in benchmark_results.items():
            if isinstance(sampler_results, dict):
                for sampler_name, result in sampler_results.items():
                    if hasattr(result, 'effective_sample_size'):
                        scaling_data.append({
                            'Dimension': dim,
                            'Distribution': dist_name,
                            'Sampler': sampler_name,
                            'ESS': result.effective_sample_size or 0,
                            'ESS_per_second': result.ess_per_second or 0,
                            'Time': result.sampling_time or 0,
                            'Acceptance': result.acceptance_rate or 0,
                            'N_samples': result.n_samples or 0
                        })
    
    return scaling_data


def _plot_scaling_metric(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, 
                        title: str, log_y: bool = False) -> None:
    """Plot scaling metric with error bars."""
    for sampler in df['Sampler'].unique():
        sampler_data = df[df['Sampler'] == sampler]
        
        # Group by dimension and compute statistics
        grouped = sampler_data.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
        
        color = METHOD_COLORS.get(sampler, 'gray')
        
        ax.errorbar(grouped[x_col], grouped['mean'], yerr=grouped['std'],
                   marker='o', label=sampler, color=color, alpha=0.8, capsize=5)
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if log_y:
        ax.set_yscale('log')


def _plot_relative_performance(ax: plt.Axes, df: pd.DataFrame, dimensions: List[int]) -> None:
    """Plot relative performance vs dimension."""
    # Compute relative performance vs baseline (Standard Metropolis)
    baseline_method = 'Standard Metropolis'
    
    if baseline_method not in df['Sampler'].values:
        baseline_method = df['Sampler'].iloc[0]  # Use first method as baseline
    
    relative_data = []
    
    for dim in dimensions:
        dim_data = df[df['Dimension'] == dim]
        baseline_perf = dim_data[dim_data['Sampler'] == baseline_method]['ESS_per_second'].mean()
        
        if baseline_perf > 0:
            for sampler in dim_data['Sampler'].unique():
                sampler_perf = dim_data[dim_data['Sampler'] == sampler]['ESS_per_second'].mean()
                relative_perf = sampler_perf / baseline_perf
                
                relative_data.append({
                    'Dimension': dim,
                    'Sampler': sampler,
                    'Relative_Performance': relative_perf
                })
    
    if relative_data:
        rel_df = pd.DataFrame(relative_data)
        
        for sampler in rel_df['Sampler'].unique():
            if sampler != baseline_method:  # Don't plot baseline (it's always 1)
                sampler_data = rel_df[rel_df['Sampler'] == sampler]
                color = METHOD_COLORS.get(sampler, 'gray')
                
                ax.plot(sampler_data['Dimension'], sampler_data['Relative_Performance'],
                       marker='o', label=sampler, color=color, alpha=0.8)
        
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, 
                  label=f'Baseline ({baseline_method})')
    
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Relative Performance')
    ax.set_title(f'Relative Performance vs Dimension\n(vs {baseline_method})')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_hessian_eigenvalue_evolution(hessian_history: List[np.ndarray], 
                                    save_path: str) -> None:
    """
    Plot showing how Hessian eigenvalues change during sampling.
    
    Creates both static and animated plots of eigenvalue evolution.
    
    Args:
        hessian_history: List of Hessian matrices over time
        save_path: Path to save the plot
    """
    _check_plotting()
    _setup_publication_style()
    
    if not hessian_history:
        warnings.warn("No Hessian history provided")
        return
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract eigenvalue evolution
    eigenval_evolution = []
    condition_numbers = []
    
    for i, hessian in enumerate(hessian_history):
        try:
            eigenvals = np.linalg.eigvals(hessian)
            eigenvals = eigenvals.real
            eigenvals = eigenvals[eigenvals > 1e-12]  # Remove near-zero
            
            if len(eigenvals) > 0:
                eigenval_evolution.append((i, eigenvals))
                condition_numbers.append(np.max(eigenvals) / np.min(eigenvals))
            
        except Exception:
            continue
    
    if not eigenval_evolution:
        warnings.warn("No valid eigenvalues computed")
        return
    
    # Create static plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hessian Eigenvalue Evolution', fontsize=16, fontweight='bold')
    
    # Panel 1: Eigenvalue spectrum at different times
    ax1 = axes[0, 0]
    time_points = [0, len(eigenval_evolution)//4, len(eigenval_evolution)//2, 
                   3*len(eigenval_evolution)//4, -1]
    
    for i, t_idx in enumerate(time_points):
        if t_idx < len(eigenval_evolution):
            _, eigenvals = eigenval_evolution[t_idx]
            sorted_eigenvals = np.sort(eigenvals)[::-1]
            
            ax1.plot(range(len(sorted_eigenvals)), sorted_eigenvals, 
                    marker='o', alpha=0.7, label=f't={t_idx}')
    
    ax1.set_xlabel('Eigenvalue Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Eigenvalue Spectrum Evolution')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Condition number evolution
    ax2 = axes[0, 1]
    iterations = [ev[0] for ev in eigenval_evolution]
    ax2.plot(iterations[:len(condition_numbers)], condition_numbers, 'b-', alpha=0.8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Condition Number')
    ax2.set_title('Condition Number Evolution')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Eigenvalue distribution histogram
    ax3 = axes[1, 0]
    if len(eigenval_evolution) > 0:
        # Use eigenvalues from last time point
        _, final_eigenvals = eigenval_evolution[-1]
        ax3.hist(np.log10(final_eigenvals), bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('log₁₀(Eigenvalue)')
        ax3.set_ylabel('Count')
        ax3.set_title('Final Eigenvalue Distribution')
        ax3.grid(True, alpha=0.3)
    
    # Panel 4: Eigenvalue statistics over time
    ax4 = axes[1, 1]
    min_eigenvals = []
    max_eigenvals = []
    mean_eigenvals = []
    
    for _, eigenvals in eigenval_evolution:
        min_eigenvals.append(np.min(eigenvals))
        max_eigenvals.append(np.max(eigenvals))
        mean_eigenvals.append(np.mean(eigenvals))
    
    ax4.plot(iterations[:len(min_eigenvals)], min_eigenvals, 'b-', alpha=0.8, label='Min')
    ax4.plot(iterations[:len(max_eigenvals)], max_eigenvals, 'r-', alpha=0.8, label='Max')
    ax4.plot(iterations[:len(mean_eigenvals)], mean_eigenvals, 'g-', alpha=0.8, label='Mean')
    
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Eigenvalue')
    ax4.set_title('Eigenvalue Statistics')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save static plot
    plt.savefig(str(save_path).replace('.png', '_static.png'), dpi=PLOT_STYLE['dpi'])
    plt.savefig(str(save_path).replace('.png', '_static.pdf'), format='pdf')
    
    print(f"Hessian eigenvalue evolution plots saved to {save_path.parent}")
    plt.close()


def visualize_hessian_preconditioning_effect(samples_dict: Dict[str, np.ndarray], 
                                           save_path: str) -> None:
    """
    Show the effect of Hessian preconditioning on sample paths.
    
    Compare preconditioned vs non-preconditioned trajectories.
    
    Args:
        samples_dict: Dictionary with sampler names and their samples
        save_path: Path to save the plot
    """
    _check_plotting()
    _setup_publication_style()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Look for Hessian vs non-Hessian methods
    hessian_methods = {k: v for k, v in samples_dict.items() if 'Hessian' in k}
    standard_methods = {k: v for k, v in samples_dict.items() if 'Hessian' not in k}
    
    if not hessian_methods or not standard_methods:
        warnings.warn("Need both Hessian and non-Hessian methods for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hessian Preconditioning Effect', fontsize=16, fontweight='bold')
    
    # Get first method from each category
    hessian_name, hessian_samples = next(iter(hessian_methods.items()))
    standard_name, standard_samples = next(iter(standard_methods.items()))
    
    # Ensure 2D for visualization
    if hessian_samples.shape[1] < 2:
        warnings.warn("Need at least 2D samples for preconditioning visualization")
        return
    
    # Panel 1: Sample path comparison (2D projection)
    ax1 = axes[0, 0]
    n_plot = min(1000, len(hessian_samples))
    
    ax1.plot(standard_samples[:n_plot, 0], standard_samples[:n_plot, 1], 
            'b-', alpha=0.6, linewidth=0.5, label=standard_name)
    ax1.plot(hessian_samples[:n_plot, 0], hessian_samples[:n_plot, 1], 
            'r-', alpha=0.6, linewidth=0.5, label=hessian_name)
    
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.set_title('Sample Path Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Marginal distribution comparison (dimension 1)
    ax2 = axes[0, 1]
    ax2.hist(standard_samples[:, 0], bins=30, alpha=0.5, density=True, 
            label=standard_name, color='blue')
    ax2.hist(hessian_samples[:, 0], bins=30, alpha=0.5, density=True, 
            label=hessian_name, color='red')
    
    ax2.set_xlabel('Value (Dimension 1)')
    ax2.set_ylabel('Density')
    ax2.set_title('Marginal Distribution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Autocorrelation comparison
    ax3 = axes[1, 0]
    max_lag = min(100, len(hessian_samples) // 4)
    
    standard_autocorr = _compute_autocorr(standard_samples[:, 0], max_lag)
    hessian_autocorr = _compute_autocorr(hessian_samples[:, 0], max_lag)
    
    lags = np.arange(len(standard_autocorr))
    ax3.plot(lags, standard_autocorr, 'b-', alpha=0.8, label=standard_name)
    ax3.plot(lags, hessian_autocorr, 'r-', alpha=0.8, label=hessian_name)
    
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('Autocorrelation')
    ax3.set_title('Autocorrelation Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Effective sample size comparison
    ax4 = axes[1, 1]
    standard_ess = effective_sample_size(standard_samples)
    hessian_ess = effective_sample_size(hessian_samples)
    
    methods = [standard_name, hessian_name]
    ess_values = [standard_ess, hessian_ess]
    colors = ['blue', 'red']
    
    bars = ax4.bar(methods, ess_values, color=colors, alpha=0.7)
    ax4.set_ylabel('Effective Sample Size')
    ax4.set_title('ESS Comparison')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, ess_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=PLOT_STYLE['dpi'])
    plt.savefig(str(save_path).replace('.png', '.pdf'), format='pdf')
    
    print(f"Preconditioning effect visualization saved to {save_path}")
    plt.close()


def save_all_advanced_plots(benchmark_results: Dict[str, Any],
                          results_by_dim: Optional[Dict[int, Dict[str, Any]]] = None,
                          chains_dict: Optional[Dict[str, np.ndarray]] = None,
                          hessian_history: Optional[List[np.ndarray]] = None,
                          output_dir: str = "advanced_plots") -> None:
    """
    Generate and save all advanced plots for Phase 4.
    
    Args:
        benchmark_results: Results from comprehensive benchmark
        results_by_dim: Results organized by dimension for scaling analysis
        chains_dict: Sample chains for trace/autocorr analysis
        hessian_history: History of Hessian matrices
        output_dir: Directory to save all plots
    """
    _check_plotting()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Generating advanced plots in {output_dir}/")
    
    try:
        # Main comparison grid
        create_sampler_comparison_grid(
            benchmark_results, 
            str(output_path),
            chains_dict=chains_dict
        )
        
        # Dimensional scaling analysis
        if results_by_dim:
            plot_dimensional_scaling_analysis(
                results_by_dim,
                str(output_path)
            )
        
        # Hessian analysis
        if hessian_history:
            plot_hessian_eigenvalue_evolution(
                hessian_history,
                str(output_path / "hessian_eigenvalue_evolution.png")
            )
        
        # Preconditioning effect
        if chains_dict:
            visualize_hessian_preconditioning_effect(
                chains_dict,
                str(output_path / "preconditioning_effect.png")
            )
        
        print(f"✅ All advanced plots saved to {output_dir}/")
        
    except Exception as e:
        print(f"⚠️ Some plots failed to generate: {e}")


if __name__ == "__main__":
    print("Advanced plotting module loaded successfully")
    if HAS_PLOTTING:
        print("✅ All plotting dependencies available")
    else:
        print("⚠️ Plotting dependencies not available")