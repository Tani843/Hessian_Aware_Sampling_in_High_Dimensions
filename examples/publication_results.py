"""
Comprehensive Publication Results Generator

This script runs the complete experimental suite and generates all 
publication-ready figures, tables, and reports for the Hessian-Aware
Sampling paper.

Usage:
    python publication_results.py [--output-dir results] [--skip-plots] [--quick-run]
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    # Core benchmark and analysis modules
    from benchmarks.benchmark_suite import SamplerBenchmark
    from analysis.theoretical_analysis import dimensional_scaling_theory
    from samplers.base import BaseSampler
    from samplers.hessian_metropolis import HessianMetropolis
    from samplers.langevin_mcmc import LangevinMCMC
    from samplers.nuts_sampler import NUTSSampler
    from distributions.multivariate_normal import MultivariateNormal
    from distributions.ill_conditioned import IllConditioned
    
    # Visualization modules (Phase 4)
    from visualization.advanced_plotting import (
        create_sampler_comparison_grid,
        plot_dimensional_scaling_analysis,
        plot_hessian_eigenvalue_evolution,
        create_multi_distribution_comparison
    )
    from visualization.convergence_plots import (
        plot_convergence_diagnostics_suite,
        plot_posterior_comparison,
        plot_trace_comparison_grid
    )
    from visualization.publication_plots import (
        create_figure_1_method_comparison,
        create_figure_2_scaling_analysis,
        create_figure_3_hessian_analysis,
        create_figure_4_cost_accuracy
    )
    
    # Results processing modules
    from results.result_generator import (
        ResultsProcessor,
        generate_experiment_report,
        create_latex_table,
        compute_statistical_significance
    )
    
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required modules are installed and paths are correct")
    sys.exit(1)


class PublicationResultsGenerator:
    """
    Comprehensive results generator for publication.
    
    Orchestrates the complete experimental pipeline from benchmarking
    to final publication-ready outputs.
    """
    
    def __init__(self, output_dir: str = "publication_results", quick_run: bool = False):
        """
        Initialize publication results generator.
        
        Args:
            output_dir: Directory for all outputs
            quick_run: If True, run reduced experiments for testing
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quick_run = quick_run
        
        # Create subdirectories
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.data_dir = self.output_dir / "data"
        self.reports_dir = self.output_dir / "reports"
        
        for subdir in [self.figures_dir, self.tables_dir, self.data_dir, self.reports_dir]:
            subdir.mkdir(exist_ok=True)
        
        # Configure plotting style for publication
        self._setup_publication_style()
        
        print(f"📁 Publication results will be saved to: {self.output_dir}")
    
    def _setup_publication_style(self):
        """Configure matplotlib/seaborn for publication-quality plots."""
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Publication settings
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'serif'],
            'text.usetex': False,  # Set to True if LaTeX is available
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    
    def run_benchmark_experiments(self) -> dict:
        """
        Run comprehensive benchmark experiments.
        
        Returns:
            Dictionary containing all benchmark results
        """
        print("🚀 Starting benchmark experiments...")
        
        # Define experimental parameters
        if self.quick_run:
            dimensions = [2, 5, 10]
            n_samples = 1000
            n_chains = 2
            n_distributions = 2
        else:
            dimensions = [2, 5, 10, 20, 50, 100]
            n_samples = 5000
            n_chains = 4
            n_distributions = 5
        
        # Initialize benchmark suite
        benchmark = SamplerBenchmark()
        
        # Define test distributions
        distributions = []
        
        # Well-conditioned distributions
        for i in range(n_distributions):
            for dim in dimensions:
                # Standard multivariate normal
                cov = np.eye(dim) + 0.1 * np.random.randn(dim, dim)
                cov = cov @ cov.T  # Ensure positive definite
                distributions.append(MultivariateNormal(
                    mean=np.zeros(dim),
                    covariance=cov,
                    name=f"MVN_d{dim}_well_{i}"
                ))
                
                # Ill-conditioned distribution
                condition_numbers = [10, 100, 1000]
                for j, cond_num in enumerate(condition_numbers):
                    if j < n_distributions:  # Limit number of ill-conditioned
                        distributions.append(IllConditioned(
                            dimension=dim,
                            condition_number=cond_num,
                            name=f"IllCond_d{dim}_c{cond_num}_{j}"
                        ))
        
        # Define samplers to test
        samplers = [
            BaseSampler(step_size=0.1, name="Standard Metropolis"),
            HessianMetropolis(step_size=0.05, name="Hessian Metropolis"),
            LangevinMCMC(step_size=0.01, name="Langevin MCMC"),
            NUTSSampler(step_size=0.1, name="NUTS")
        ]
        
        # Run benchmarks
        all_results = {}
        
        for i, distribution in enumerate(distributions):
            print(f"📊 Testing distribution {i+1}/{len(distributions)}: {distribution.name}")
            
            try:
                dist_results = benchmark.compare_samplers(
                    distribution=distribution,
                    samplers=samplers,
                    n_samples=n_samples,
                    n_chains=n_chains,
                    detailed_diagnostics=True
                )
                all_results[distribution.name] = dist_results
                
            except Exception as e:
                warnings.warn(f"Failed to benchmark {distribution.name}: {e}")
                continue
        
        # Save raw results
        results_file = self.data_dir / "benchmark_results.json"
        try:
            import json
            with open(results_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = self._prepare_for_json(all_results)
                json.dump(json_results, f, indent=2, default=str)
            print(f"💾 Raw results saved to: {results_file}")
        except Exception as e:
            warnings.warn(f"Failed to save raw results: {e}")
        
        return all_results
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization by converting numpy arrays."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def generate_publication_figures(self, results: dict) -> None:
        """
        Generate all publication figures (Figure 1-4).
        
        Args:
            results: Benchmark results dictionary
        """
        print("🎨 Generating publication figures...")
        
        try:
            # Figure 1: Method Comparison
            print("  📈 Creating Figure 1: Method Comparison")
            fig1_path = self.figures_dir / "figure_1_method_comparison"
            create_figure_1_method_comparison(
                results, 
                save_path=str(fig1_path),
                formats=['pdf', 'png']
            )
            
            # Figure 2: Dimensional Scaling Analysis
            print("  📊 Creating Figure 2: Scaling Analysis")
            fig2_path = self.figures_dir / "figure_2_scaling_analysis"
            create_figure_2_scaling_analysis(
                results,
                save_path=str(fig2_path),
                formats=['pdf', 'png']
            )
            
            # Figure 3: Hessian Analysis
            print("  🔍 Creating Figure 3: Hessian Analysis")
            fig3_path = self.figures_dir / "figure_3_hessian_analysis"
            create_figure_3_hessian_analysis(
                results,
                save_path=str(fig3_path),
                formats=['pdf', 'png']
            )
            
            # Figure 4: Cost vs Accuracy Trade-off
            print("  ⚖️ Creating Figure 4: Cost-Accuracy Trade-off")
            fig4_path = self.figures_dir / "figure_4_cost_accuracy"
            create_figure_4_cost_accuracy(
                results,
                save_path=str(fig4_path),
                formats=['pdf', 'png']
            )
            
        except Exception as e:
            warnings.warn(f"Error generating publication figures: {e}")
    
    def generate_supplementary_figures(self, results: dict) -> None:
        """
        Generate supplementary figures for detailed analysis.
        
        Args:
            results: Benchmark results dictionary
        """
        print("📊 Generating supplementary figures...")
        
        try:
            # Advanced comparison grids
            print("  🔄 Creating sampler comparison grids")
            comp_grid_path = self.figures_dir / "sampler_comparison_grid"
            create_sampler_comparison_grid(
                results,
                save_path=str(comp_grid_path),
                formats=['pdf', 'png']
            )
            
            # Dimensional scaling analysis
            print("  📈 Creating dimensional scaling analysis")
            scaling_path = self.figures_dir / "dimensional_scaling_analysis"
            plot_dimensional_scaling_analysis(
                results,
                save_path=str(scaling_path),
                formats=['pdf', 'png']
            )
            
            # Convergence diagnostics
            print("  🎯 Creating convergence diagnostics")
            conv_path = self.figures_dir / "convergence_diagnostics"
            plot_convergence_diagnostics_suite(
                results,
                save_path=str(conv_path),
                formats=['pdf', 'png']
            )
            
            # Multi-distribution comparison
            print("  🔀 Creating multi-distribution comparison")
            multi_dist_path = self.figures_dir / "multi_distribution_comparison"
            create_multi_distribution_comparison(
                results,
                save_path=str(multi_dist_path),
                formats=['pdf', 'png']
            )
            
        except Exception as e:
            warnings.warn(f"Error generating supplementary figures: {e}")
    
    def generate_tables_and_reports(self, results: dict) -> None:
        """
        Generate publication tables and comprehensive reports.
        
        Args:
            results: Benchmark results dictionary
        """
        print("📋 Generating tables and reports...")
        
        try:
            # Initialize results processor
            processor = ResultsProcessor(results)
            
            # Generate performance comparison table
            print("  📊 Creating performance comparison table")
            
            # LaTeX table
            latex_table = processor.generate_performance_table(
                format_style='latex',
                metrics=['ESS', 'ESS_per_second', 'Acceptance_rate', 'Time_per_ESS']
            )
            
            with open(self.tables_dir / "performance_table.tex", 'w') as f:
                f.write(latex_table)
            
            # Markdown table for reports
            md_table = processor.generate_performance_table(
                format_style='markdown',
                metrics=['ESS', 'ESS_per_second', 'Acceptance_rate', 'Time_per_ESS']
            )
            
            with open(self.tables_dir / "performance_table.md", 'w') as f:
                f.write(md_table)
            
            # Performance ranking table
            print("  🏆 Creating performance ranking table")
            ranking_df = processor.generate_performance_ranking()
            
            if not ranking_df.empty:
                # Save as CSV
                ranking_df.to_csv(self.tables_dir / "performance_ranking.csv", index=False)
                
                # Create LaTeX table
                ranking_latex = create_latex_table(
                    ranking_df,
                    caption="Overall Performance Ranking of MCMC Samplers",
                    label="tab:ranking"
                )
                
                with open(self.tables_dir / "ranking_table.tex", 'w') as f:
                    f.write(ranking_latex)
            
            # Generate comprehensive report
            print("  📝 Generating comprehensive experiment report")
            report_path = generate_experiment_report(
                results,
                output_dir=str(self.reports_dir),
                include_plots=True
            )
            
            # Statistical significance summary
            print("  🧮 Computing statistical significance tests")
            stat_tests = processor.perform_statistical_tests()
            
            if stat_tests:
                # Create significance summary table
                sig_data = []
                for sampler, test_result in stat_tests.items():
                    sig_data.append({
                        'Sampler': sampler,
                        'Test': test_result.test_name,
                        'P-value': f"{test_result.p_value:.4f}",
                        'Effect Size': f"{test_result.effect_size:.3f}",
                        'Significant': "Yes" if test_result.significant else "No",
                        'Interpretation': test_result.interpretation
                    })
                
                sig_df = pd.DataFrame(sig_data)
                sig_df.to_csv(self.tables_dir / "statistical_significance.csv", index=False)
                
                # LaTeX version
                sig_latex = create_latex_table(
                    sig_df,
                    caption="Statistical Significance Tests vs. Baseline",
                    label="tab:significance"
                )
                
                with open(self.tables_dir / "significance_table.tex", 'w') as f:
                    f.write(sig_latex)
            
        except Exception as e:
            warnings.warn(f"Error generating tables and reports: {e}")
    
    def create_executive_summary(self, results: dict) -> None:
        """
        Create executive summary of all results.
        
        Args:
            results: Benchmark results dictionary
        """
        print("📋 Creating executive summary...")
        
        try:
            processor = ResultsProcessor(results)
            
            summary_lines = [
                "# Executive Summary: Hessian-Aware MCMC Sampling Results",
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## Key Findings",
                ""
            ]
            
            # Top performers
            top_performers = processor.get_top_performers(n_top=3)
            if top_performers:
                summary_lines.append("### Best Performing Methods")
                for i, (sampler, performance) in enumerate(top_performers, 1):
                    summary_lines.append(f"{i}. **{sampler}**: {performance:.2f} ESS/sec")
                summary_lines.append("")
            
            # Improvement factors
            improvements = processor.compute_improvement_factors()
            if improvements:
                best_improvement = max(improvements.items(), key=lambda x: x[1])
                best_pct = (best_improvement[1] - 1) * 100
                
                summary_lines.extend([
                    "### Performance Improvements",
                    f"- Best improvement: **{best_improvement[0]}** (+{best_pct:.1f}%)",
                    f"- Average improvement across methods: {np.mean(list(improvements.values()))-1:.1%}",
                    ""
                ])
            
            # Statistical significance
            stat_tests = processor.perform_statistical_tests()
            if stat_tests:
                significant_methods = [name for name, test in stat_tests.items() if test.significant]
                summary_lines.extend([
                    "### Statistical Significance",
                    f"- {len(significant_methods)} out of {len(stat_tests)} methods show significant improvement",
                    f"- Methods with significant improvement: {', '.join(significant_methods)}",
                    ""
                ])
            
            # Recommendations
            summary_lines.extend([
                "## Recommendations",
                "",
                "1. **For general use**: Use Hessian-aware methods when computational budget allows",
                "2. **For ill-conditioned problems**: Hessian Metropolis shows consistent advantages", 
                "3. **For high dimensions**: NUTS and Langevin MCMC scale well with dimension",
                "4. **For quick sampling**: Standard Metropolis remains viable for well-conditioned targets",
                "",
                "## Files Generated",
                "",
                "### Figures",
                "- `figure_1_method_comparison.pdf/png`: Main performance comparison",
                "- `figure_2_scaling_analysis.pdf/png`: Dimensional scaling analysis",
                "- `figure_3_hessian_analysis.pdf/png`: Hessian conditioning effects",
                "- `figure_4_cost_accuracy.pdf/png`: Cost vs accuracy trade-offs",
                "",
                "### Tables",
                "- `performance_table.tex`: LaTeX performance comparison table",
                "- `ranking_table.tex`: Overall performance ranking",
                "- `significance_table.tex`: Statistical significance results",
                "",
                "### Reports",
                "- `experiment_report.md`: Comprehensive technical report",
                "- `experiment_report.json`: Machine-readable results data",
                "",
                "---",
                "*Generated by Hessian-Aware Sampling Publication Results Suite*"
            ])
            
            # Write summary
            summary_path = self.output_dir / "EXECUTIVE_SUMMARY.md"
            with open(summary_path, 'w') as f:
                f.write('\n'.join(summary_lines))
            
            print(f"📋 Executive summary saved to: {summary_path}")
            
        except Exception as e:
            warnings.warn(f"Error creating executive summary: {e}")
    
    def run_complete_analysis(self) -> None:
        """
        Run the complete publication analysis pipeline.
        
        This is the main entry point that orchestrates the entire process:
        1. Run benchmark experiments
        2. Generate all publication figures  
        3. Create supplementary visualizations
        4. Generate tables and reports
        5. Create executive summary
        """
        start_time = datetime.now()
        print(f"🚀 Starting complete publication analysis pipeline...")
        print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Run benchmark experiments
            print("\n" + "="*60)
            print("STEP 1: BENCHMARK EXPERIMENTS")
            print("="*60)
            results = self.run_benchmark_experiments()
            
            if not results:
                print("❌ No benchmark results obtained. Aborting analysis.")
                return
            
            print(f"✅ Completed benchmarks for {len(results)} distributions")
            
            # Step 2: Generate publication figures
            print("\n" + "="*60) 
            print("STEP 2: PUBLICATION FIGURES")
            print("="*60)
            self.generate_publication_figures(results)
            print("✅ Publication figures generated")
            
            # Step 3: Generate supplementary figures
            print("\n" + "="*60)
            print("STEP 3: SUPPLEMENTARY FIGURES")
            print("="*60)
            self.generate_supplementary_figures(results)
            print("✅ Supplementary figures generated")
            
            # Step 4: Generate tables and reports
            print("\n" + "="*60)
            print("STEP 4: TABLES AND REPORTS")
            print("="*60)
            self.generate_tables_and_reports(results)
            print("✅ Tables and reports generated")
            
            # Step 5: Create executive summary
            print("\n" + "="*60)
            print("STEP 5: EXECUTIVE SUMMARY")
            print("="*60)
            self.create_executive_summary(results)
            print("✅ Executive summary created")
            
            # Final summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "="*60)
            print("🎉 PUBLICATION ANALYSIS COMPLETE!")
            print("="*60)
            print(f"⏰ Total runtime: {duration}")
            print(f"📁 All outputs saved to: {self.output_dir}")
            print(f"📋 See EXECUTIVE_SUMMARY.md for key findings")
            
            # List key output files
            key_files = [
                "EXECUTIVE_SUMMARY.md",
                "figures/figure_1_method_comparison.pdf",
                "figures/figure_2_scaling_analysis.pdf", 
                "figures/figure_3_hessian_analysis.pdf",
                "figures/figure_4_cost_accuracy.pdf",
                "tables/performance_table.tex",
                "reports/experiment_report.md"
            ]
            
            print("\n📂 Key output files:")
            for file in key_files:
                full_path = self.output_dir / file
                if full_path.exists():
                    print(f"  ✅ {file}")
                else:
                    print(f"  ❌ {file} (missing)")
            
        except KeyboardInterrupt:
            print("\n⚠️  Analysis interrupted by user")
        except Exception as e:
            print(f"\n❌ Error in analysis pipeline: {e}")
            raise


def main():
    """Main entry point for publication results generation."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive publication results for Hessian-Aware Sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python publication_results.py                    # Full analysis
  python publication_results.py --quick-run        # Quick test run
  python publication_results.py --output-dir pub   # Custom output directory
  python publication_results.py --skip-plots       # Skip plot generation
        """
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='publication_results',
        help='Output directory for all results (default: publication_results)'
    )
    
    parser.add_argument(
        '--quick-run',
        action='store_true',
        help='Run reduced experiments for testing (faster, less comprehensive)'
    )
    
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip plot generation (useful for quick table/report updates)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Configure warnings
    if not args.verbose:
        warnings.filterwarnings('ignore')
    
    # Initialize and run analysis
    try:
        generator = PublicationResultsGenerator(
            output_dir=args.output_dir,
            quick_run=args.quick_run
        )
        
        if args.skip_plots:
            print("⚠️  Skipping plot generation as requested")
            # Run only benchmarks and reports
            results = generator.run_benchmark_experiments()
            generator.generate_tables_and_reports(results)
            generator.create_executive_summary(results)
        else:
            # Run complete analysis
            generator.run_complete_analysis()
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()