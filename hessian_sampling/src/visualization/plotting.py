"""
Visualization and plotting utilities for Hessian-aware sampling.

This module provides comprehensive plotting functions for analyzing
sampling results, convergence diagnostics, and Hessian properties.
"""

from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib not available. Plotting functionality will be limited.")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from ..core.sampling_base import SamplingResults


def check_plotting_available():
    """Check if plotting libraries are available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for plotting. Install with: pip install matplotlib")


def plot_sampling_results(results: SamplingResults,
                         dimensions: Optional[List[int]] = None,
                         max_dims_to_plot: int = 6,
                         figsize: Optional[Tuple[int, int]] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive visualization of sampling results.
    
    Args:
        results: SamplingResults object
        dimensions: Specific dimensions to plot (None for auto-selection)
        max_dims_to_plot: Maximum dimensions to include in plots
        figsize: Figure size (width, height)
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
        
    Raises:
        ImportError: If matplotlib not available
    """
    check_plotting_available()
    
    samples = results.samples
    n_samples, n_dims = samples.shape
    
    # Select dimensions to plot
    if dimensions is None:
        dimensions = list(range(min(n_dims, max_dims_to_plot)))
    else:
        dimensions = [d for d in dimensions if 0 <= d < n_dims]
    
    n_plot_dims = len(dimensions)
    
    if figsize is None:
        figsize = (15, 3 * n_plot_dims)
    
    # Create subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_plot_dims, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    for i, dim in enumerate(dimensions):
        # Trace plot
        ax_trace = fig.add_subplot(gs[i, 0])
        ax_trace.plot(samples[:, dim], alpha=0.7, linewidth=0.5)
        ax_trace.set_title(f'Trace - Dimension {dim}')
        ax_trace.set_xlabel('Iteration')
        ax_trace.set_ylabel('Value')
        ax_trace.grid(True, alpha=0.3)
        
        # Histogram
        ax_hist = fig.add_subplot(gs[i, 1])
        ax_hist.hist(samples[:, dim], bins=50, density=True, alpha=0.7, edgecolor='black')
        ax_hist.set_title(f'Histogram - Dimension {dim}')
        ax_hist.set_xlabel('Value')
        ax_hist.set_ylabel('Density')
        ax_hist.grid(True, alpha=0.3)
        
        # Autocorrelation
        ax_autocorr = fig.add_subplot(gs[i, 2])
        autocorr = compute_autocorrelation(samples[:, dim])
        max_lag = min(200, len(autocorr) - 1)
        ax_autocorr.plot(range(max_lag), autocorr[:max_lag])
        ax_autocorr.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax_autocorr.set_title(f'Autocorrelation - Dimension {dim}')
        ax_autocorr.set_xlabel('Lag')
        ax_autocorr.set_ylabel('Autocorrelation')
        ax_autocorr.grid(True, alpha=0.3)
        
        # Running average
        ax_running = fig.add_subplot(gs[i, 3])
        running_mean = np.cumsum(samples[:, dim]) / np.arange(1, n_samples + 1)
        ax_running.plot(running_mean)
        if results.diagnostics and 'mean' in results.diagnostics:
            final_mean = results.diagnostics['mean'][dim]
            ax_running.axhline(y=final_mean, color='r', linestyle='--', 
                             label=f'Final mean: {final_mean:.3f}')
            ax_running.legend()
        ax_running.set_title(f'Running Mean - Dimension {dim}')
        ax_running.set_xlabel('Iteration')
        ax_running.set_ylabel('Running Mean')
        ax_running.grid(True, alpha=0.3)
    
    plt.suptitle(f'Sampling Results (Acceptance Rate: {results.acceptance_rate:.3f})', 
                 fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_pairwise_scatter(samples: np.ndarray,
                         dimensions: Optional[List[int]] = None,
                         max_dims: int = 5,
                         alpha: float = 0.6,
                         figsize: Optional[Tuple[int, int]] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Create pairwise scatter plots of samples.
    
    Args:
        samples: Sample array (n_samples × n_dims)
        dimensions: Dimensions to plot
        max_dims: Maximum dimensions for pairwise plots
        alpha: Point transparency
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    check_plotting_available()
    
    n_samples, n_dims = samples.shape
    
    if dimensions is None:
        dimensions = list(range(min(n_dims, max_dims)))
    else:
        dimensions = [d for d in dimensions if 0 <= d < n_dims]
    
    n_plot_dims = len(dimensions)
    
    if figsize is None:
        figsize = (3 * n_plot_dims, 3 * n_plot_dims)
    
    fig, axes = plt.subplots(n_plot_dims, n_plot_dims, figsize=figsize)
    
    if n_plot_dims == 1:
        axes = np.array([[axes]])
    elif n_plot_dims == 2:
        axes = axes.reshape(2, 2)
    
    for i, dim_i in enumerate(dimensions):
        for j, dim_j in enumerate(dimensions):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                ax.hist(samples[:, dim_i], bins=50, density=True, alpha=0.7)
                ax.set_xlabel(f'Dimension {dim_i}')
                ax.set_ylabel('Density')
            elif i > j:
                # Lower triangle: scatter plot
                ax.scatter(samples[:, dim_j], samples[:, dim_i], 
                          alpha=alpha, s=1)
                ax.set_xlabel(f'Dimension {dim_j}')
                ax.set_ylabel(f'Dimension {dim_i}')
            else:
                # Upper triangle: remove axis
                ax.remove()
            
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_hessian_properties(hessian: np.ndarray,
                          title: str = "Hessian Properties",
                          figsize: Optional[Tuple[int, int]] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize properties of Hessian matrix.
    
    Args:
        hessian: Hessian matrix
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    check_plotting_available()
    
    if figsize is None:
        figsize = (15, 5)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Hessian heatmap
    im1 = axes[0].imshow(hessian, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('Hessian Matrix')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Dimension')
    plt.colorbar(im1, ax=axes[0])
    
    # Eigenvalue spectrum
    eigenvals = np.linalg.eigvals(hessian)
    eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
    
    axes[1].plot(eigenvals, 'o-', markersize=4)
    axes[1].set_title('Eigenvalue Spectrum')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Eigenvalue')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('symlog')  # Handle negative eigenvalues
    
    # Condition number and statistics
    axes[2].axis('off')
    
    pos_eigenvals = eigenvals[eigenvals > 0]
    condition_number = np.max(pos_eigenvals) / np.min(pos_eigenvals) if len(pos_eigenvals) > 0 else np.inf
    
    stats_text = f"""
    Matrix Size: {hessian.shape[0]} × {hessian.shape[1]}
    Condition Number: {condition_number:.2e}
    Max Eigenvalue: {np.max(eigenvals):.2e}
    Min Eigenvalue: {np.min(eigenvals):.2e}
    Num Positive: {np.sum(eigenvals > 0)}
    Num Negative: {np.sum(eigenvals < 0)}
    Num Zero: {np.sum(np.abs(eigenvals) < 1e-12)}
    Trace: {np.trace(hessian):.2e}
    Determinant: {np.linalg.det(hessian):.2e}
    """
    
    axes[2].text(0.1, 0.9, stats_text, transform=axes[2].transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    axes[2].set_title('Statistics')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_step_size_adaptation(step_sizes: np.ndarray,
                            acceptance_rates: Optional[np.ndarray] = None,
                            target_acceptance: float = 0.574,
                            figsize: Optional[Tuple[int, int]] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot step size adaptation over iterations.
    
    Args:
        step_sizes: Array of step sizes over iterations
        acceptance_rates: Array of acceptance rates (optional)
        target_acceptance: Target acceptance rate line
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    check_plotting_available()
    
    if figsize is None:
        figsize = (12, 6)
    
    if acceptance_rates is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    
    # Step size plot
    iterations = np.arange(len(step_sizes))
    ax1.plot(iterations, step_sizes, linewidth=1)
    ax1.set_ylabel('Step Size')
    ax1.set_title('Step Size Adaptation')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Acceptance rate plot
    if acceptance_rates is not None:
        ax2.plot(iterations[:len(acceptance_rates)], acceptance_rates, linewidth=1, color='orange')
        ax2.axhline(y=target_acceptance, color='red', linestyle='--', 
                   label=f'Target: {target_acceptance:.3f}')
        ax2.set_ylabel('Acceptance Rate')
        ax2.set_xlabel('Iteration')
        ax2.set_title('Acceptance Rate')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1)
    else:
        ax1.set_xlabel('Iteration')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_convergence_diagnostics(results: SamplingResults,
                                dimension: int = 0,
                                window_size: int = 100,
                                figsize: Optional[Tuple[int, int]] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot convergence diagnostics for a specific dimension.
    
    Args:
        results: SamplingResults object
        dimension: Dimension to analyze
        window_size: Window size for rolling statistics
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    check_plotting_available()
    
    if figsize is None:
        figsize = (15, 10)
    
    samples = results.samples[:, dimension]
    n_samples = len(samples)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Trace plot with rolling mean
    axes[0, 0].plot(samples, alpha=0.7, linewidth=0.5, label='Samples')
    
    # Rolling mean
    rolling_mean = np.convolve(samples, np.ones(window_size)/window_size, mode='valid')
    rolling_x = np.arange(window_size-1, n_samples)
    axes[0, 0].plot(rolling_x, rolling_mean, color='red', linewidth=2, label=f'Rolling Mean (window={window_size})')
    
    axes[0, 0].set_title(f'Trace Plot - Dimension {dimension}')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rolling variance
    rolling_var = np.convolve(samples**2, np.ones(window_size)/window_size, mode='valid') - rolling_mean**2
    axes[0, 1].plot(rolling_x, rolling_var, color='green', linewidth=2)
    axes[0, 1].set_title(f'Rolling Variance - Dimension {dimension}')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Variance')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Autocorrelation function
    autocorr = compute_autocorrelation(samples)
    max_lag = min(200, len(autocorr) - 1)
    axes[1, 0].plot(range(max_lag), autocorr[:max_lag])
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5% threshold')
    axes[1, 0].set_title(f'Autocorrelation - Dimension {dimension}')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('Autocorrelation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Potential scale reduction factor (R-hat) - simplified version
    # Split chain into two halves
    mid = n_samples // 2
    chain1 = samples[:mid]
    chain2 = samples[mid:]
    
    # Compute R-hat approximation
    r_hat_values = []
    min_length = 50
    
    for end in range(min_length, min(len(chain1), len(chain2)) + 1, 10):
        var1 = np.var(chain1[:end])
        var2 = np.var(chain2[:end])
        mean1 = np.mean(chain1[:end])
        mean2 = np.mean(chain2[:end])
        
        W = (var1 + var2) / 2  # Within-chain variance
        B = end * (mean1 - mean2)**2 / 2  # Between-chain variance
        
        if W > 0:
            V_hat = ((end - 1) * W + B) / end
            r_hat = np.sqrt(V_hat / W)
        else:
            r_hat = 1.0
        
        r_hat_values.append(r_hat)
    
    r_hat_x = np.arange(min_length, min_length + len(r_hat_values) * 10, 10)
    axes[1, 1].plot(r_hat_x, r_hat_values, linewidth=2)
    axes[1, 1].axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect convergence')
    axes[1, 1].axhline(y=1.1, color='r', linestyle='--', alpha=0.5, label='Good convergence')
    axes[1, 1].set_title(f'R-hat Approximation - Dimension {dimension}')
    axes[1, 1].set_xlabel('Sample Size')
    axes[1, 1].set_ylabel('R-hat')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compute_autocorrelation(x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """
    Compute autocorrelation function using FFT.
    
    Args:
        x: Input time series
        max_lag: Maximum lag to compute (None for all)
        
    Returns:
        Autocorrelation function
    """
    x = x - np.mean(x)
    n = len(x)
    
    if max_lag is None:
        max_lag = n
    
    # Zero-pad for FFT
    x_padded = np.zeros(2 * n)
    x_padded[:n] = x
    
    # Compute autocorrelation via FFT
    X = np.fft.fft(x_padded)
    autocorr = np.fft.ifft(X * np.conj(X)).real[:n]
    
    # Normalize
    autocorr = autocorr / autocorr[0]
    
    return autocorr[:max_lag]


def plot_comparison(results_dict: Dict[str, SamplingResults],
                   dimension: int = 0,
                   figsize: Optional[Tuple[int, int]] = None,
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare results from different samplers.
    
    Args:
        results_dict: Dictionary of {sampler_name: SamplingResults}
        dimension: Dimension to compare
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    check_plotting_available()
    
    if figsize is None:
        figsize = (15, 10)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for i, (name, results) in enumerate(results_dict.items()):
        color = colors[i]
        samples = results.samples[:, dimension]
        
        # Trace plots
        axes[0, 0].plot(samples, alpha=0.7, linewidth=0.5, 
                       label=f'{name} (AR: {results.acceptance_rate:.3f})', color=color)
        
        # Histograms
        axes[0, 1].hist(samples, bins=50, density=True, alpha=0.5, 
                       label=name, color=color, edgecolor='black')
        
        # Autocorrelations
        autocorr = compute_autocorrelation(samples)
        max_lag = min(100, len(autocorr) - 1)
        axes[1, 0].plot(range(max_lag), autocorr[:max_lag], 
                       label=name, color=color)
        
        # Running means
        running_mean = np.cumsum(samples) / np.arange(1, len(samples) + 1)
        axes[1, 1].plot(running_mean, label=name, color=color)
    
    axes[0, 0].set_title(f'Trace Plots - Dimension {dimension}')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title(f'Density Comparison - Dimension {dimension}')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title(f'Autocorrelation Comparison - Dimension {dimension}')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('Autocorrelation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title(f'Running Mean Comparison - Dimension {dimension}')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Running Mean')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig