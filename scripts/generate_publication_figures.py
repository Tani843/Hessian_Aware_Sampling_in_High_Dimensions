#!/usr/bin/env python3
"""
Publication Figure Generator

Generates publication-ready figures from benchmark results.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication figures
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.figsize': (10, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def load_benchmark_results(results_dir="benchmark_results"):
    """Load all benchmark results from CSV files."""
    results = {}
    
    for result_dir in Path(results_dir).glob("*_d*"):
        if result_dir.is_dir():
            dist_dim = result_dir.name
            csv_file = result_dir / "detailed_results.csv"
            
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                results[dist_dim] = df
    
    return results

def create_figure_1_comparison(results, save_path="assets/images/plots/fig1_comparison.png"):
    """Figure 1: Method comparison across distributions."""
    
    # Extract key metrics
    methods = ['Standard Metropolis', 'Adaptive Metropolis', 'Langevin Dynamics', 'HMC']
    metrics = ['ESS_per_second', 'Acceptance_rate', 'ESS']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sampling Method Performance Comparison', fontsize=16, fontweight='bold')
    
    # ESS per second comparison
    ax = axes[0, 0]
    data_for_plot = []
    
    for dist_dim, df in results.items():
        if 'd10' in dist_dim:  # Focus on 10D for clarity
            for method in methods:
                method_data = df[df['Sampler'] == method]
                if len(method_data) > 0:
                    ess_per_sec = method_data['ESS_per_second'].mean()
                    data_for_plot.append({
                        'Method': method,
                        'Distribution': dist_dim.replace('_d10', ''),
                        'ESS_per_second': ess_per_sec
                    })
    
    plot_df = pd.DataFrame(data_for_plot)
    if len(plot_df) > 0:
        pivot_df = plot_df.pivot(index='Distribution', columns='Method', values='ESS_per_second')
        pivot_df.plot(kind='bar', ax=ax)
        ax.set_title('Effective Sample Size per Second')
        ax.set_ylabel('ESS/second')
        ax.set_xlabel('Distribution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Acceptance rates
    ax = axes[0, 1]
    data_for_plot = []
    
    for dist_dim, df in results.items():
        if 'd10' in dist_dim:
            for method in methods:
                method_data = df[df['Sampler'] == method]
                if len(method_data) > 0:
                    acc_rate = method_data['Acceptance_rate'].mean()
                    data_for_plot.append({
                        'Method': method,
                        'Distribution': dist_dim.replace('_d10', ''),
                        'Acceptance_rate': acc_rate
                    })
    
    plot_df = pd.DataFrame(data_for_plot)
    if len(plot_df) > 0:
        pivot_df = plot_df.pivot(index='Distribution', columns='Method', values='Acceptance_rate')
        pivot_df.plot(kind='bar', ax=ax)
        ax.set_title('Acceptance Rates')
        ax.set_ylabel('Acceptance Rate')
        ax.set_xlabel('Distribution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    # ESS comparison
    ax = axes[1, 0]
    data_for_plot = []
    
    for dist_dim, df in results.items():
        if 'd10' in dist_dim:
            for method in methods:
                method_data = df[df['Sampler'] == method]
                if len(method_data) > 0:
                    avg_ess = method_data['ESS'].mean()
                    data_for_plot.append({
                        'Method': method,
                        'Distribution': dist_dim.replace('_d10', ''),
                        'ESS': avg_ess
                    })
    
    plot_df = pd.DataFrame(data_for_plot)
    if len(plot_df) > 0:
        pivot_df = plot_df.pivot(index='Distribution', columns='Method', values='ESS')
        pivot_df.plot(kind='bar', ax=ax)
        ax.set_title('Average Effective Sample Size')
        ax.set_ylabel('Average ESS')
        ax.set_xlabel('Distribution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Performance vs dimension
    ax = axes[1, 1]
    for method in methods:
        dims = []
        perfs = []
        
        for dim in [10, 50, 100]:
            perf_values = []
            for dist_dim, df in results.items():
                if f'd{dim}' in dist_dim and 'Gaussian_Easy' in dist_dim:
                    method_data = df[df['Sampler'] == method]
                    if len(method_data) > 0:
                        perf_values.append(method_data['ESS_per_second'].mean())
            
            if perf_values:
                dims.append(dim)
                perfs.append(np.mean(perf_values))
        
        if dims:
            ax.plot(dims, perfs, 'o-', label=method, linewidth=2, markersize=8)
    
    ax.set_xlabel('Dimension')
    ax.set_ylabel('ESS per Second')
    ax.set_title('Scaling with Dimension')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 1 saved to {save_path}")

def create_figure_2_scaling(results, save_path="assets/images/plots/fig2_scaling.png"):
    """Figure 2: Dimensional scaling analysis."""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Dimensional Scaling Analysis', fontsize=16, fontweight='bold')
    
    methods = ['Standard Metropolis', 'Adaptive Metropolis', 'Langevin Dynamics', 'HMC']
    colors = sns.color_palette("husl", len(methods))
    
    # Scaling with dimension
    ax = axes[0]
    for i, method in enumerate(methods):
        dims = []
        perfs = []
        errors = []
        
        for dim in [10, 50, 100]:
            perf_values = []
            for dist_dim, df in results.items():
                if f'd{dim}' in dist_dim and 'Gaussian_Easy' in dist_dim:
                    method_data = df[df['Sampler'] == method]
                    if len(method_data) > 0:
                        perf_values.extend(method_data['ESS_per_second'].values)
            
            if perf_values:
                dims.append(dim)
                perfs.append(np.mean(perf_values))
                errors.append(np.std(perf_values) / np.sqrt(len(perf_values)))
        
        if dims and len(dims) > 1:
            ax.errorbar(dims, perfs, yerr=errors, 
                       color=colors[i], label=method, 
                       linewidth=2, markersize=8, capsize=5,
                       marker='o', capthick=2)
    
    ax.set_xlabel('Dimension')
    ax.set_ylabel('ESS per Second')
    ax.set_title('Performance vs Dimension')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Condition number effect
    ax = axes[1]
    cond_data = {}
    
    for dist_dim, df in results.items():
        if 'd50' in dist_dim:  # Focus on 50D
            for method in methods:
                method_data = df[df['Sampler'] == method]
                if len(method_data) > 0:
                    if method not in cond_data:
                        cond_data[method] = {}
                    
                    if 'Easy' in dist_dim:
                        cond_data[method]['Low'] = method_data['ESS_per_second'].mean()
                    elif 'Hard' in dist_dim:
                        cond_data[method]['High'] = method_data['ESS_per_second'].mean()
    
    # Plot condition number comparison
    for i, method in enumerate(methods):
        if method in cond_data:
            conditions = list(cond_data[method].keys())
            values = list(cond_data[method].values())
            if len(conditions) >= 2:
                x_pos = np.arange(len(conditions)) + i * 0.2
                ax.bar(x_pos, values, width=0.2, label=method, color=colors[i])
    
    ax.set_xlabel('Condition Number')
    ax.set_ylabel('ESS per Second')
    ax.set_title('Effect of Conditioning')
    ax.set_xticks(np.arange(2) + 0.3)
    ax.set_xticklabels(['Low (κ=10)', 'High (κ=1000)'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 2 saved to {save_path}")

def create_figure_3_hessian_analysis(save_path="assets/images/plots/fig3_hessian.png"):
    """Figure 3: Hessian eigenvalue analysis (theoretical)."""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Hessian-Based Preconditioning Analysis', fontsize=16, fontweight='bold')
    
    # Simulated eigenvalue distribution
    ax = axes[0]
    dims = [10, 50, 100]
    
    for dim in dims:
        # Simulate eigenvalues for ill-conditioned Gaussian
        condition_number = 1000
        eigenvals = np.logspace(0, np.log10(condition_number), dim)
        eigenvals = eigenvals / np.max(eigenvals)  # Normalize
        
        ax.semilogy(range(1, dim+1), sorted(eigenvals, reverse=True), 
                   'o-', label=f'dim={dim}', linewidth=2, markersize=4)
    
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Hessian Eigenvalue Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Preconditioning effect
    ax = axes[1]
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Original ill-conditioned distribution
    kappa = 100  # condition number
    Z1 = np.exp(-0.5 * (X**2 + kappa * Y**2))
    
    # Hessian-preconditioned distribution  
    Z2 = np.exp(-0.5 * (X**2 + Y**2))
    
    # Plot contours
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    ax.contour(X, Y, Z1, levels=levels, colors='red', alpha=0.7, 
              linestyles='--', linewidths=2)
    ax.contour(X, Y, Z2, levels=levels, colors='blue', alpha=0.7, 
              linestyles='-', linewidths=2)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, 
               label='Original (ill-conditioned)'),
        Line2D([0], [0], color='blue', linestyle='-', linewidth=2, 
               label='Hessian-preconditioned')
    ]
    ax.legend(handles=legend_elements)
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Preconditioning Effect')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 3 saved to {save_path}")

def create_figure_4_cost_accuracy(results, save_path="assets/images/plots/fig4_cost_accuracy.png"):
    """Figure 4: Cost vs accuracy tradeoff."""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Computational Cost vs Accuracy Analysis', fontsize=16, fontweight='bold')
    
    methods = ['Standard Metropolis', 'Adaptive Metropolis', 'Langevin Dynamics', 'HMC']
    colors = sns.color_palette("husl", len(methods))
    
    # ESS vs Time per sample
    ax = axes[0]
    for i, method in enumerate(methods):
        ess_values = []
        time_values = []
        
        for dist_dim, df in results.items():
            if 'd50' in dist_dim:  # Focus on 50D
                method_data = df[df['Sampler'] == method]
                if len(method_data) > 0:
                    ess_values.extend(method_data['ESS'].values)
                    time_values.extend(method_data['Sampling_time'].values)
        
        if ess_values and time_values:
            ax.scatter(time_values, ess_values, 
                      c=[colors[i]], label=method, s=100, alpha=0.7)
    
    ax.set_xlabel('Time per Sample (seconds)')
    ax.set_ylabel('Average ESS')
    ax.set_title('ESS vs Computational Cost')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Efficiency frontier
    ax = axes[1]
    efficiency_data = []
    
    for method in methods:
        method_efficiency = []
        
        for dist_dim, df in results.items():
            method_data = df[df['Sampler'] == method]
            if len(method_data) > 0:
                ess_per_sec = method_data['ESS_per_second'].mean()
                method_efficiency.append(ess_per_sec)
        
        if method_efficiency:
            efficiency_data.append({
                'Method': method,
                'Mean_Efficiency': np.mean(method_efficiency),
                'Std_Efficiency': np.std(method_efficiency)
            })
    
    if efficiency_data:
        eff_df = pd.DataFrame(efficiency_data)
        bars = ax.bar(eff_df['Method'], eff_df['Mean_Efficiency'], 
                     yerr=eff_df['Std_Efficiency'], capsize=5, 
                     color=colors[:len(eff_df)])
        
        ax.set_ylabel('Average ESS per Second')
        ax.set_title('Overall Efficiency Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, eff_df['Mean_Efficiency']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 4 saved to {save_path}")

def copy_existing_plots():
    """Copy plots from benchmark_results to assets directory."""
    
    src_dir = Path("benchmark_results/plots")
    dst_dir = Path("assets/images/plots")
    
    if src_dir.exists():
        os.makedirs(dst_dir, exist_ok=True)
        
        for plot_file in src_dir.glob("*.png"):
            dst_file = dst_dir / plot_file.name
            import shutil
            shutil.copy2(plot_file, dst_file)
            print(f"✓ Copied {plot_file.name} to {dst_file}")

def main():
    """Generate all publication figures."""
    
    print("🎨 Generating Publication Figures")
    print("=" * 50)
    
    # Load benchmark results
    results = load_benchmark_results()
    
    if not results:
        print("❌ No benchmark results found!")
        print("Please run benchmarks first:")
        print("  python examples/comprehensive_benchmark.py")
        return
    
    print(f"✓ Loaded results from {len(results)} benchmark runs")
    
    # Create output directory
    os.makedirs("assets/images/plots", exist_ok=True)
    os.makedirs("assets/images/diagrams", exist_ok=True)
    
    # Generate publication figures
    try:
        create_figure_1_comparison(results)
        create_figure_2_scaling(results)
        create_figure_3_hessian_analysis()
        create_figure_4_cost_accuracy(results)
        
        # Copy existing plots
        copy_existing_plots()
        
        print("\n✅ All publication figures generated successfully!")
        print("\nGenerated files:")
        
        for plot_file in Path("assets/images/plots").glob("*.png"):
            print(f"  📊 {plot_file}")
            
    except Exception as e:
        print(f"❌ Error generating figures: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()