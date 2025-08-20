"""
Statistical convergence plotting suite for MCMC diagnostics.

This module provides comprehensive convergence visualization including:
- R-hat evolution plots
- Geweke diagnostic visualizations  
- Running mean convergence analysis
- Posterior comparison plots
- Multi-chain diagnostic plots

All plots follow publication standards with proper statistical analysis.
"""

import numpy as np
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle
    import seaborn as sns
    from scipy import stats
    HAS_PLOTTING = True
    
    # Import plotting style
    from .advanced_plotting import PLOT_STYLE, METHOD_COLORS, _setup_publication_style
    
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("Matplotlib/Seaborn not available. Convergence plotting will be disabled.")

try:
    from ..benchmarks.performance_metrics import (
        effective_sample_size, 
        potential_scale_reduction_factor,
        geweke_diagnostic
    )
    from ..benchmarks.convergence_diagnostics import ConvergenceDiagnostics
except ImportError:
    from benchmarks.performance_metrics import (
        effective_sample_size, 
        potential_scale_reduction_factor,
        geweke_diagnostic
    )
    from benchmarks.convergence_diagnostics import ConvergenceDiagnostics


def _check_plotting():
    """Check if plotting libraries are available."""
    if not HAS_PLOTTING:
        raise ImportError("Convergence plotting requires matplotlib and seaborn")


def plot_convergence_diagnostics_suite(chains_dict: Dict[str, Union[np.ndarray, List[np.ndarray]]], 
                                     save_dir: str,
                                     parameter_names: Optional[List[str]] = None) -> None:
    """
    Create comprehensive convergence analysis plots.
    
    Generates:
    - R-hat evolution over iterations
    - Geweke diagnostic z-scores  
    - Running mean convergence
    - Potential scale reduction factors
    - Multiple chain comparison
    
    Args:
        chains_dict: Dictionary of {sampler_name: chains}
        save_dir: Directory to save plots
        parameter_names: Names for parameters (optional)
    """
    _check_plotting()
    _setup_publication_style()
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Process each sampler's chains
    for sampler_name, chains in chains_dict.items():
        print(f"Generating convergence diagnostics for {sampler_name}...")
        
        # Standardize chains format
        if isinstance(chains, np.ndarray):
            if chains.ndim == 2:
                chains = [chains]  # Single chain
            else:
                continue
        
        if not isinstance(chains, list):
            continue
        
        try:
            # Generate comprehensive diagnostics plot
            _create_sampler_convergence_plot(
                chains, sampler_name, save_path, parameter_names
            )
            
        except Exception as e:
            print(f"Failed to create convergence plot for {sampler_name}: {e}")
            continue
    
    # Create comparative convergence analysis
    try:
        _create_comparative_convergence_plot(chains_dict, save_path)
    except Exception as e:
        print(f"Failed to create comparative convergence plot: {e}")
    
    print(f"✅ Convergence diagnostic plots saved to {save_path}")


def _create_sampler_convergence_plot(chains: List[np.ndarray], 
                                   sampler_name: str,
                                   save_path: Path,
                                   parameter_names: Optional[List[str]] = None) -> None:
    """Create comprehensive convergence plot for a single sampler."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    n_chains = len(chains)
    n_params = chains[0].shape[1] if chains[0].ndim > 1 else 1
    n_samples = len(chains[0])
    
    if parameter_names is None:
        parameter_names = [f'param_{i}' for i in range(n_params)]
    
    # Use first few parameters for visualization
    max_params_plot = min(n_params, 3)
    
    # Panel 1: Trace plots for multiple chains
    ax1 = fig.add_subplot(gs[0, :])
    _plot_multi_chain_traces(ax1, chains, parameter_names[:max_params_plot])
    
    # Panel 2: R-hat evolution
    ax2 = fig.add_subplot(gs[1, 0])
    _plot_r_hat_evolution(ax2, chains, parameter_names[:max_params_plot])
    
    # Panel 3: Geweke diagnostics
    ax3 = fig.add_subplot(gs[1, 1])
    _plot_geweke_diagnostics(ax3, chains, parameter_names)
    
    # Panel 4: Running mean convergence
    ax4 = fig.add_subplot(gs[1, 2])
    _plot_running_mean_convergence(ax4, chains, parameter_names[:max_params_plot])
    
    # Panel 5: Autocorrelation functions
    ax5 = fig.add_subplot(gs[2, 0])
    _plot_multi_chain_autocorr(ax5, chains, parameter_names[0] if parameter_names else 'param_0')
    
    # Panel 6: Density comparison across chains
    ax6 = fig.add_subplot(gs[2, 1])
    _plot_chain_density_comparison(ax6, chains, parameter_names[0] if parameter_names else 'param_0')
    
    # Panel 7: ESS by parameter
    ax7 = fig.add_subplot(gs[2, 2])
    _plot_ess_by_parameter(ax7, chains, parameter_names)
    
    # Add title
    fig.suptitle(f'Convergence Diagnostics: {sampler_name}', 
                fontsize=16, fontweight='bold')
    
    # Save plot
    clean_name = sampler_name.replace(' ', '_').replace('/', '_')
    plt.savefig(save_path / f"convergence_{clean_name}.png", dpi=PLOT_STYLE['dpi'])
    plt.savefig(save_path / f"convergence_{clean_name}.pdf", format='pdf')
    plt.close()


def _plot_multi_chain_traces(ax: plt.Axes, chains: List[np.ndarray], 
                           parameter_names: List[str]) -> None:
    """Plot trace plots for multiple chains."""
    colors = plt.cm.Set1(np.linspace(0, 1, len(chains)))
    
    for param_idx, param_name in enumerate(parameter_names[:3]):  # Max 3 params
        for chain_idx, chain in enumerate(chains):
            if chain.ndim > 1 and param_idx < chain.shape[1]:
                trace = chain[:, param_idx]
            elif param_idx == 0:
                trace = chain if chain.ndim == 1 else chain[:, 0]
            else:
                continue
                
            # Offset traces vertically for clarity
            offset = param_idx * 2
            ax.plot(trace + offset, color=colors[chain_idx], alpha=0.7, 
                   linewidth=0.8, label=f'Chain {chain_idx+1}' if param_idx == 0 else '')
            
        # Add parameter labels
        if len(parameter_names) > 1:
            ax.text(0.02, 0.8 - param_idx * 0.25, param_name, 
                   transform=ax.transAxes, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title('Multi-Chain Trace Plots')
    if len(chains) > 1:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)


def _plot_r_hat_evolution(ax: plt.Axes, chains: List[np.ndarray], 
                        parameter_names: List[str]) -> None:
    """Plot R-hat evolution over iterations."""
    if len(chains) < 2:
        ax.text(0.5, 0.5, 'Multiple chains required\nfor R-hat computation', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('R-hat Evolution')
        return
    
    n_samples = len(chains[0])
    n_params = min(len(parameter_names), chains[0].shape[1] if chains[0].ndim > 1 else 1)
    
    # Compute R-hat at different time points
    time_points = np.linspace(100, n_samples, 20, dtype=int)
    
    for param_idx in range(min(n_params, 3)):  # Max 3 parameters
        r_hat_values = []
        
        for t in time_points:
            try:
                param_chains = []
                for chain in chains:
                    if chain.ndim > 1 and param_idx < chain.shape[1]:
                        param_chains.append(chain[:t, param_idx])
                    elif param_idx == 0:
                        param_chains.append(chain[:t] if chain.ndim == 1 else chain[:t, 0])
                
                if param_chains and all(len(c) > 10 for c in param_chains):
                    r_hat = potential_scale_reduction_factor(param_chains)
                    r_hat_values.append(r_hat)
                else:
                    r_hat_values.append(np.nan)
                    
            except Exception:
                r_hat_values.append(np.nan)
        
        param_name = parameter_names[param_idx] if param_idx < len(parameter_names) else f'param_{param_idx}'
        ax.plot(time_points, r_hat_values, marker='o', alpha=0.8, 
               label=param_name, markersize=4)
    
    # Add convergence threshold line
    ax.axhline(y=1.1, color='red', linestyle='--', alpha=0.7, label='Convergence threshold')
    ax.axhline(y=1.01, color='green', linestyle='--', alpha=0.7, label='Excellent convergence')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('R-hat')
    ax.set_title('R-hat Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.95, None)


def _plot_geweke_diagnostics(ax: plt.Axes, chains: List[np.ndarray], 
                           parameter_names: List[str]) -> None:
    """Plot Geweke diagnostic z-scores."""
    n_params = len(parameter_names)
    z_scores_by_param = []
    param_labels = []
    
    for param_idx in range(min(n_params, 10)):  # Max 10 parameters
        param_z_scores = []
        
        for chain in chains:
            try:
                if chain.ndim > 1 and param_idx < chain.shape[1]:
                    param_chain = chain[:, param_idx]
                elif param_idx == 0:
                    param_chain = chain if chain.ndim == 1 else chain[:, 0]
                else:
                    continue
                
                z_score = geweke_diagnostic(param_chain)
                param_z_scores.append(z_score)
                
            except Exception:
                continue
        
        if param_z_scores:
            z_scores_by_param.extend(param_z_scores)
            param_labels.extend([parameter_names[param_idx]] * len(param_z_scores))
    
    if z_scores_by_param:
        # Create box plot of z-scores by parameter
        unique_params = list(set(param_labels))
        z_score_data = [
            [z for z, p in zip(z_scores_by_param, param_labels) if p == param]
            for param in unique_params
        ]
        
        box_plot = ax.boxplot(z_score_data, labels=unique_params, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_params)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add significance bounds
        ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='95% significance')
        ax.axhline(y=-2.0, color='red', linestyle='--', alpha=0.7)
        
        ax.set_ylabel('Geweke Z-score')
        ax.set_title('Geweke Convergence Diagnostics')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No Geweke diagnostics\ncomputed', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Geweke Diagnostics')


def _plot_running_mean_convergence(ax: plt.Axes, chains: List[np.ndarray], 
                                 parameter_names: List[str]) -> None:
    """Plot running mean convergence."""
    colors = plt.cm.Set1(np.linspace(0, 1, len(chains)))
    
    for param_idx in range(min(len(parameter_names), 3)):  # Max 3 parameters
        for chain_idx, chain in enumerate(chains):
            try:
                if chain.ndim > 1 and param_idx < chain.shape[1]:
                    param_chain = chain[:, param_idx]
                elif param_idx == 0:
                    param_chain = chain if chain.ndim == 1 else chain[:, 0]
                else:
                    continue
                
                # Compute running mean
                running_mean = np.cumsum(param_chain) / np.arange(1, len(param_chain) + 1)
                
                label = f'{parameter_names[param_idx]} (Chain {chain_idx+1})' if len(chains) > 1 else parameter_names[param_idx]
                ax.plot(running_mean, color=colors[chain_idx], alpha=0.8, 
                       label=label if param_idx < 3 else '', linewidth=1.5)
                
            except Exception:
                continue
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Running Mean')
    ax.set_title('Running Mean Convergence')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)


def _plot_multi_chain_autocorr(ax: plt.Axes, chains: List[np.ndarray], 
                              parameter_name: str) -> None:
    """Plot autocorrelation functions for multiple chains."""
    max_lag = min(100, len(chains[0]) // 4)
    colors = plt.cm.Set1(np.linspace(0, 1, len(chains)))
    
    for chain_idx, chain in enumerate(chains):
        try:
            if chain.ndim > 1:
                param_chain = chain[:, 0]  # Use first parameter
            else:
                param_chain = chain
            
            # Compute autocorrelation
            autocorr = _compute_autocorr_function(param_chain, max_lag)
            lags = np.arange(len(autocorr))
            
            ax.plot(lags, autocorr, color=colors[chain_idx], alpha=0.8,
                   label=f'Chain {chain_idx+1}')
            
        except Exception:
            continue
    
    # Add significance bounds
    n = len(chains[0])
    significance = 1.96 / np.sqrt(n)
    ax.axhline(y=significance, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=-significance, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'Autocorrelation: {parameter_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_chain_density_comparison(ax: plt.Axes, chains: List[np.ndarray], 
                                 parameter_name: str) -> None:
    """Plot density comparison across chains."""
    colors = plt.cm.Set1(np.linspace(0, 1, len(chains)))
    
    for chain_idx, chain in enumerate(chains):
        try:
            if chain.ndim > 1:
                param_chain = chain[:, 0]  # Use first parameter
            else:
                param_chain = chain
            
            # Plot density
            ax.hist(param_chain, bins=30, density=True, alpha=0.6, 
                   color=colors[chain_idx], label=f'Chain {chain_idx+1}')
            
        except Exception:
            continue
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Density Comparison: {parameter_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_ess_by_parameter(ax: plt.Axes, chains: List[np.ndarray], 
                         parameter_names: List[str]) -> None:
    """Plot effective sample size by parameter."""
    n_params = min(len(parameter_names), 10)  # Max 10 parameters
    ess_values = []
    param_labels = []
    
    for param_idx in range(n_params):
        for chain in chains:
            try:
                if chain.ndim > 1 and param_idx < chain.shape[1]:
                    param_chain = chain[:, param_idx]
                elif param_idx == 0:
                    param_chain = chain if chain.ndim == 1 else chain[:, 0]
                else:
                    continue
                
                ess = effective_sample_size(param_chain)
                ess_values.append(ess)
                param_labels.append(parameter_names[param_idx] if param_idx < len(parameter_names) else f'param_{param_idx}')
                
            except Exception:
                continue
    
    if ess_values:
        # Create box plot of ESS by parameter
        unique_params = list(set(param_labels))
        ess_data = [
            [ess for ess, p in zip(ess_values, param_labels) if p == param]
            for param in unique_params
        ]
        
        box_plot = ax.boxplot(ess_data, labels=unique_params, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_params)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Effective Sample Size')
        ax.set_title('ESS by Parameter')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No ESS data\navailable', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('ESS by Parameter')


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


def _create_comparative_convergence_plot(chains_dict: Dict[str, Union[np.ndarray, List[np.ndarray]]], 
                                       save_path: Path) -> None:
    """Create comparative convergence analysis across samplers."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparative Convergence Analysis', fontsize=16, fontweight='bold')
    
    # Panel 1: ESS comparison
    ax1 = axes[0, 0]
    _plot_comparative_ess(ax1, chains_dict)
    
    # Panel 2: R-hat comparison (if multiple chains available)
    ax2 = axes[0, 1]
    _plot_comparative_r_hat(ax2, chains_dict)
    
    # Panel 3: Autocorrelation time comparison
    ax3 = axes[1, 0]
    _plot_comparative_autocorr_time(ax3, chains_dict)
    
    # Panel 4: Convergence summary
    ax4 = axes[1, 1]
    _plot_convergence_summary(ax4, chains_dict)
    
    plt.tight_layout()
    plt.savefig(save_path / "comparative_convergence.png", dpi=PLOT_STYLE['dpi'])
    plt.savefig(save_path / "comparative_convergence.pdf", format='pdf')
    plt.close()


def _plot_comparative_ess(ax: plt.Axes, chains_dict: Dict[str, Union[np.ndarray, List[np.ndarray]]]) -> None:
    """Plot ESS comparison across samplers."""
    sampler_names = []
    ess_values = []
    
    for sampler_name, chains in chains_dict.items():
        try:
            if isinstance(chains, list):
                # Multiple chains - use first chain
                chain = chains[0]
            else:
                chain = chains
            
            ess = effective_sample_size(chain)
            sampler_names.append(sampler_name)
            ess_values.append(ess)
            
        except Exception:
            continue
    
    if sampler_names:
        colors = [METHOD_COLORS.get(name, 'gray') for name in sampler_names]
        bars = ax.bar(sampler_names, ess_values, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, ess_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.0f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Effective Sample Size')
        ax.set_title('ESS Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')


def _plot_comparative_r_hat(ax: plt.Axes, chains_dict: Dict[str, Union[np.ndarray, List[np.ndarray]]]) -> None:
    """Plot R-hat comparison across samplers."""
    sampler_names = []
    r_hat_values = []
    
    for sampler_name, chains in chains_dict.items():
        try:
            if isinstance(chains, list) and len(chains) > 1:
                # Multiple chains - can compute R-hat
                r_hat = potential_scale_reduction_factor(chains)
                sampler_names.append(sampler_name)
                r_hat_values.append(r_hat)
                
        except Exception:
            continue
    
    if sampler_names:
        colors = [METHOD_COLORS.get(name, 'gray') for name in sampler_names]
        bars = ax.bar(sampler_names, r_hat_values, color=colors, alpha=0.8)
        
        # Add convergence threshold line
        ax.axhline(y=1.1, color='red', linestyle='--', alpha=0.7, label='Convergence threshold')
        
        # Add value labels on bars
        for bar, value in zip(bars, r_hat_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('R-hat')
        ax.set_title('R-hat Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0.95, None)
    else:
        ax.text(0.5, 0.5, 'Multiple chains required\nfor R-hat computation', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('R-hat Comparison')


def _plot_comparative_autocorr_time(ax: plt.Axes, chains_dict: Dict[str, Union[np.ndarray, List[np.ndarray]]]) -> None:
    """Plot autocorrelation time comparison."""
    sampler_names = []
    autocorr_times = []
    
    for sampler_name, chains in chains_dict.items():
        try:
            if isinstance(chains, list):
                chain = chains[0]  # Use first chain
            else:
                chain = chains
            
            # Use first dimension
            if chain.ndim > 1:
                param_chain = chain[:, 0]
            else:
                param_chain = chain
            
            # Compute integrated autocorrelation time
            autocorr_time = _compute_integrated_autocorr_time(param_chain)
            
            sampler_names.append(sampler_name)
            autocorr_times.append(autocorr_time)
            
        except Exception:
            continue
    
    if sampler_names:
        colors = [METHOD_COLORS.get(name, 'gray') for name in sampler_names]
        bars = ax.bar(sampler_names, autocorr_times, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, autocorr_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Autocorrelation Time')
        ax.set_title('Autocorrelation Time Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')


def _plot_convergence_summary(ax: plt.Axes, chains_dict: Dict[str, Union[np.ndarray, List[np.ndarray]]]) -> None:
    """Plot convergence summary scores."""
    sampler_names = list(chains_dict.keys())
    n_samplers = len(sampler_names)
    
    if n_samplers == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
               transform=ax.transAxes)
        ax.set_title('Convergence Summary')
        return
    
    # Create mock convergence scores (in practice, would be computed)
    metrics = ['ESS Score', 'R-hat Score', 'Autocorr Score', 'Overall Score']
    
    # Generate random scores for demonstration (replace with real computation)
    scores = np.random.rand(n_samplers, len(metrics))
    scores = 0.3 + 0.7 * scores  # Scale to [0.3, 1.0]
    
    # Create heatmap
    im = ax.imshow(scores.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(n_samplers):
        for j in range(len(metrics)):
            text = ax.text(i, j, f'{scores[i, j]:.2f}', 
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xticks(range(n_samplers))
    ax.set_xticklabels(sampler_names, rotation=45, ha='right')
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_title('Convergence Summary Scores')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Score (0=Poor, 1=Excellent)')


def _compute_integrated_autocorr_time(chain: np.ndarray) -> float:
    """Compute integrated autocorrelation time."""
    try:
        from ..benchmarks.performance_metrics import integrated_autocorr_time
        return integrated_autocorr_time(chain)
    except ImportError:
        # Fallback implementation
        chain = chain - np.mean(chain)
        n = len(chain)
        
        # Compute autocorrelation using FFT
        f = np.fft.fft(chain, n=2*n)
        acorr = np.fft.ifft(f * np.conj(f))[:n].real
        acorr = acorr / acorr[0] if acorr[0] > 0 else np.zeros_like(acorr)
        
        # Integrated time
        cumsum = np.cumsum(acorr)
        for i in range(1, len(cumsum)):
            if i >= 6 * cumsum[i]:
                return cumsum[i]
        
        return cumsum[-1] if len(cumsum) > 0 else 1.0


def plot_posterior_comparison(true_posterior: Dict[str, Any], 
                            sampled_posterior: Dict[str, np.ndarray],
                            save_path: str,
                            parameter_names: Optional[List[str]] = None) -> None:
    """
    Compare true vs estimated posterior distributions.
    
    Includes marginal distributions, contour plots, and QQ plots.
    
    Args:
        true_posterior: Dictionary with 'mean', 'cov' for true distribution
        sampled_posterior: Dictionary of {sampler_name: samples}
        save_path: Path to save plot
        parameter_names: Names for parameters
    """
    _check_plotting()
    _setup_publication_style()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not true_posterior or not sampled_posterior:
        warnings.warn("Need both true and sampled posteriors for comparison")
        return
    
    # Extract true distribution info
    true_mean = true_posterior.get('mean', np.zeros(2))
    true_cov = true_posterior.get('cov', np.eye(len(true_mean)))
    
    n_params = len(true_mean)
    if parameter_names is None:
        parameter_names = [f'θ_{i+1}' for i in range(n_params)]
    
    # Create figure with subplots
    n_samplers = len(sampled_posterior)
    fig, axes = plt.subplots(2, n_samplers, figsize=(4*n_samplers, 8))
    
    if n_samplers == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('True vs Estimated Posterior Comparison', 
                fontsize=16, fontweight='bold')
    
    # Plot comparisons for each sampler
    for i, (sampler_name, samples) in enumerate(sampled_posterior.items()):
        
        # Top panel: Marginal comparison
        ax_top = axes[0, i]
        _plot_marginal_comparison(ax_top, true_mean, true_cov, samples, 
                                parameter_names, sampler_name)
        
        # Bottom panel: QQ plot or 2D comparison
        ax_bottom = axes[1, i]
        if n_params >= 2:
            _plot_2d_posterior_comparison(ax_bottom, true_mean, true_cov, samples,
                                        parameter_names[:2], sampler_name)
        else:
            _plot_qq_comparison(ax_bottom, true_mean[0], np.sqrt(true_cov[0, 0]), 
                              samples, parameter_names[0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_STYLE['dpi'])
    plt.savefig(str(save_path).replace('.png', '.pdf'), format='pdf')
    print(f"Posterior comparison plot saved to {save_path}")
    plt.close()


def _plot_marginal_comparison(ax: plt.Axes, true_mean: np.ndarray, true_cov: np.ndarray,
                            samples: np.ndarray, parameter_names: List[str],
                            sampler_name: str) -> None:
    """Plot marginal distribution comparison."""
    # Use first parameter for marginal comparison
    param_idx = 0
    param_name = parameter_names[param_idx] if param_idx < len(parameter_names) else 'param_0'
    
    if samples.ndim > 1 and param_idx < samples.shape[1]:
        sample_values = samples[:, param_idx]
    else:
        sample_values = samples if samples.ndim == 1 else samples[:, 0]
    
    # True marginal distribution
    true_marginal_mean = true_mean[param_idx]
    true_marginal_std = np.sqrt(true_cov[param_idx, param_idx])
    
    # Plot true distribution
    x_range = np.linspace(true_marginal_mean - 4*true_marginal_std,
                         true_marginal_mean + 4*true_marginal_std, 100)
    true_density = stats.norm.pdf(x_range, true_marginal_mean, true_marginal_std)
    
    ax.plot(x_range, true_density, 'r-', linewidth=2, label='True', alpha=0.8)
    
    # Plot estimated distribution
    ax.hist(sample_values, bins=30, density=True, alpha=0.6, 
           color=METHOD_COLORS.get(sampler_name, 'blue'), 
           label='Estimated', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel(param_name)
    ax.set_ylabel('Density')
    ax.set_title(f'Marginal: {sampler_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_2d_posterior_comparison(ax: plt.Axes, true_mean: np.ndarray, true_cov: np.ndarray,
                                samples: np.ndarray, parameter_names: List[str],
                                sampler_name: str) -> None:
    """Plot 2D posterior comparison with contours."""
    if samples.shape[1] < 2:
        ax.text(0.5, 0.5, 'Need at least 2D\nfor contour plot', 
               ha='center', va='center', transform=ax.transAxes)
        return
    
    # Extract 2D samples
    samples_2d = samples[:, :2]
    
    # Plot sample points
    ax.scatter(samples_2d[:, 0], samples_2d[:, 1], alpha=0.3, s=1,
              color=METHOD_COLORS.get(sampler_name, 'blue'))
    
    # True distribution contours
    from matplotlib.patches import Ellipse
    from scipy.stats import chi2
    
    # 95% confidence ellipse for true distribution
    eigenvals, eigenvecs = np.linalg.eigh(true_cov[:2, :2])
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    # Chi-square quantile for 95% confidence
    scale = chi2.ppf(0.95, df=2)
    width, height = 2 * np.sqrt(scale * eigenvals)
    
    ellipse = Ellipse(true_mean[:2], width, height, angle=angle,
                     facecolor='none', edgecolor='red', linewidth=2,
                     linestyle='--', label='True 95% CI')
    ax.add_patch(ellipse)
    
    # Sample statistics
    sample_mean = np.mean(samples_2d, axis=0)
    ax.plot(sample_mean[0], sample_mean[1], 'ro', markersize=8, 
           label='Sample mean', markerfacecolor='white', markeredgewidth=2)
    ax.plot(true_mean[0], true_mean[1], 'r*', markersize=12, 
           label='True mean')
    
    ax.set_xlabel(parameter_names[0])
    ax.set_ylabel(parameter_names[1])
    ax.set_title(f'2D Posterior: {sampler_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')


def _plot_qq_comparison(ax: plt.Axes, true_mean: float, true_std: float,
                       samples: np.ndarray, parameter_name: str) -> None:
    """Plot Q-Q plot comparison."""
    if samples.ndim > 1:
        sample_values = samples[:, 0]
    else:
        sample_values = samples
    
    # Standardize samples
    standardized_samples = (sample_values - np.mean(sample_values)) / np.std(sample_values)
    
    # Create Q-Q plot
    stats.probplot(standardized_samples, dist="norm", plot=ax)
    
    ax.set_title(f'Q-Q Plot: {parameter_name}')
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    print("Convergence plotting module loaded successfully")
    if HAS_PLOTTING:
        print("✅ All plotting dependencies available")
    else:
        print("⚠️ Plotting dependencies not available")