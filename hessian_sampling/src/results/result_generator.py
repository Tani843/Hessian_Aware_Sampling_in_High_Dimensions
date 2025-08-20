"""
Automated results generation and processing.

This module provides comprehensive tools for:
- Processing benchmark results into publication-ready formats
- Computing statistical significance and improvement factors  
- Generating LaTeX tables and automated reports
- Creating executive summaries with recommendations
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime
from scipy import stats
from dataclasses import dataclass

try:
    from ..benchmarks.performance_metrics import (
        effective_sample_size,
        potential_scale_reduction_factor
    )
    from ..benchmarks.convergence_diagnostics import ConvergenceDiagnostics
    from ..analysis.theoretical_analysis import dimensional_scaling_theory
except ImportError:
    from benchmarks.performance_metrics import (
        effective_sample_size,
        potential_scale_reduction_factor
    )
    from benchmarks.convergence_diagnostics import ConvergenceDiagnostics
    from analysis.theoretical_analysis import dimensional_scaling_theory


@dataclass
class StatisticalSummary:
    """Container for statistical summary results."""
    mean: float
    std: float
    median: float
    ci_lower: float
    ci_upper: float
    sample_size: int


@dataclass 
class ComparisonResult:
    """Container for comparison test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    significant: bool
    interpretation: str


class ResultsProcessor:
    """
    Comprehensive processor for benchmark results.
    
    Processes raw benchmark data into publication-ready formats
    including tables, summaries, and statistical analyses.
    """
    
    def __init__(self, benchmark_data: Dict[str, Any]):
        """
        Initialize results processor.
        
        Args:
            benchmark_data: Raw benchmark results from SamplerBenchmark
        """
        self.benchmark_data = benchmark_data
        self.processed_results: Optional[pd.DataFrame] = None
        self.statistical_tests: Dict[str, Any] = {}
        self.improvement_factors: Dict[str, float] = {}
        
        self._process_raw_data()
    
    def _process_raw_data(self) -> None:
        """Process raw benchmark data into structured format."""
        processed_data = []
        
        if 'benchmark_results' in self.benchmark_data:
            benchmark_results = self.benchmark_data['benchmark_results']
        else:
            benchmark_results = self.benchmark_data
        
        for dist_name, sampler_results in benchmark_results.items():
            for sampler_name, result in sampler_results.items():
                if hasattr(result, 'effective_sample_size'):
                    processed_data.append({
                        'Distribution': dist_name,
                        'Sampler': sampler_name,
                        'Dimension': getattr(result, 'dimension', 'Unknown'),
                        'ESS': result.effective_sample_size or 0,
                        'ESS_per_second': result.ess_per_second or 0,
                        'Sampling_time': result.sampling_time or 0,
                        'Acceptance_rate': result.acceptance_rate or 0,
                        'N_samples': result.n_samples or 0,
                        'R_hat': getattr(result, 'r_hat', None),
                        'Autocorr_time': getattr(result, 'autocorr_time', None),
                        'MSE': getattr(result, 'mean_squared_error', None),
                        'Diagnostics': getattr(result, 'diagnostics', {})
                    })
        
        self.processed_results = pd.DataFrame(processed_data)
        
        # Add derived metrics
        if not self.processed_results.empty:
            self._compute_derived_metrics()
    
    def _compute_derived_metrics(self) -> None:
        """Compute derived performance metrics."""
        df = self.processed_results
        
        # Time per effective sample
        df['Time_per_ESS'] = np.where(df['ESS'] > 0, df['Sampling_time'] / df['ESS'], np.inf)
        
        # Efficiency (ESS / N_samples)
        df['Efficiency'] = np.where(df['N_samples'] > 0, df['ESS'] / df['N_samples'], 0)
        
        # Performance score (composite metric)
        # Normalize ESS_per_second and efficiency, then take geometric mean
        if len(df) > 1:
            ess_per_sec_norm = df['ESS_per_second'] / df['ESS_per_second'].max()
            efficiency_norm = df['Efficiency'] / df['Efficiency'].max()
            df['Performance_score'] = np.sqrt(ess_per_sec_norm * efficiency_norm)
        else:
            df['Performance_score'] = 1.0
    
    def generate_performance_table(self, 
                                 metrics: List[str] = None,
                                 format_style: str = 'latex',
                                 round_digits: int = 3) -> Union[str, pd.DataFrame]:
        """
        Generate publication-ready performance comparison table.
        
        Args:
            metrics: Metrics to include in table
            format_style: Output format ('latex', 'html', 'markdown', 'dataframe')
            round_digits: Number of decimal places
            
        Returns:
            Formatted table as string or DataFrame
        """
        if self.processed_results is None or self.processed_results.empty:
            warnings.warn("No processed results available for table generation")
            return pd.DataFrame()
        
        if metrics is None:
            metrics = ['ESS', 'ESS_per_second', 'Acceptance_rate', 'Time_per_ESS']
        
        # Create pivot table
        table_data = []
        
        for dist in self.processed_results['Distribution'].unique():
            dist_data = self.processed_results[self.processed_results['Distribution'] == dist]
            
            for sampler in dist_data['Sampler'].unique():
                sampler_data = dist_data[dist_data['Sampler'] == sampler]
                
                row = {
                    'Distribution': dist,
                    'Sampler': sampler,
                    'Dimension': sampler_data['Dimension'].iloc[0]
                }
                
                # Add statistics for each metric
                for metric in metrics:
                    if metric in sampler_data.columns:
                        values = sampler_data[metric].dropna()
                        if len(values) > 0:
                            mean_val = values.mean()
                            std_val = values.std() if len(values) > 1 else 0
                            
                            if format_style == 'latex':
                                row[metric] = f"{mean_val:.{round_digits}f} $\\pm$ {std_val:.{round_digits}f}"
                            else:
                                row[metric] = f"{mean_val:.{round_digits}f} ± {std_val:.{round_digits}f}"
                        else:
                            row[metric] = 'N/A'
                    else:
                        row[metric] = 'N/A'
                
                table_data.append(row)
        
        df_table = pd.DataFrame(table_data)
        
        if format_style == 'dataframe':
            return df_table
        elif format_style == 'latex':
            return self._format_latex_table(df_table)
        elif format_style == 'html':
            return df_table.to_html(index=False)
        elif format_style == 'markdown':
            return df_table.to_markdown(index=False)
        else:
            return df_table
    
    def _format_latex_table(self, df: pd.DataFrame) -> str:
        """Format DataFrame as LaTeX table."""
        latex_str = "\\begin{table}[htbp]\n"
        latex_str += "\\centering\n"
        latex_str += "\\caption{Performance Comparison of MCMC Samplers}\n"
        latex_str += "\\label{tab:performance_comparison}\n"
        
        # Create column specification
        n_cols = len(df.columns)
        col_spec = "l" * n_cols
        latex_str += f"\\begin{{tabular}}{{{col_spec}}}\n"
        latex_str += "\\toprule\n"
        
        # Header
        headers = [col.replace('_', '\\_') for col in df.columns]
        latex_str += " & ".join(headers) + " \\\\\n"
        latex_str += "\\midrule\n"
        
        # Data rows
        for _, row in df.iterrows():
            row_str = " & ".join(str(val).replace('_', '\\_') for val in row.values)
            latex_str += row_str + " \\\\\n"
        
        latex_str += "\\bottomrule\n"
        latex_str += "\\end{tabular}\n"
        latex_str += "\\end{table}\n"
        
        return latex_str
    
    def compute_improvement_factors(self, 
                                  baseline_method: str = 'Standard Metropolis',
                                  metric: str = 'ESS_per_second') -> Dict[str, float]:
        """
        Compute relative improvements over baseline method.
        
        Args:
            baseline_method: Name of baseline sampler
            metric: Metric to compare
            
        Returns:
            Dictionary of improvement factors
        """
        if self.processed_results is None or self.processed_results.empty:
            return {}
        
        improvements = {}
        
        # Get baseline performance for each distribution
        for dist in self.processed_results['Distribution'].unique():
            dist_data = self.processed_results[self.processed_results['Distribution'] == dist]
            
            # Find baseline method performance
            baseline_data = dist_data[dist_data['Sampler'] == baseline_method]
            if baseline_data.empty:
                continue
            
            baseline_perf = baseline_data[metric].mean()
            if baseline_perf <= 0:
                continue
            
            # Compute improvements for each method
            for sampler in dist_data['Sampler'].unique():
                if sampler == baseline_method:
                    continue
                
                sampler_data = dist_data[dist_data['Sampler'] == sampler]
                sampler_perf = sampler_data[metric].mean()
                
                improvement = sampler_perf / baseline_perf
                key = f"{sampler}_{dist}"
                improvements[key] = improvement
        
        # Compute average improvements across distributions
        sampler_improvements = {}
        for sampler in self.processed_results['Sampler'].unique():
            if sampler == baseline_method:
                continue
            
            sampler_keys = [k for k in improvements.keys() if k.startswith(sampler)]
            if sampler_keys:
                avg_improvement = np.mean([improvements[k] for k in sampler_keys])
                sampler_improvements[sampler] = avg_improvement
        
        self.improvement_factors = sampler_improvements
        return sampler_improvements
    
    def generate_statistical_summary(self, 
                                   confidence_level: float = 0.95) -> Dict[str, Dict[str, StatisticalSummary]]:
        """
        Generate comprehensive statistical summary with confidence intervals.
        
        Args:
            confidence_level: Confidence level for intervals
            
        Returns:
            Nested dictionary of statistical summaries
        """
        if self.processed_results is None or self.processed_results.empty:
            return {}
        
        alpha = 1 - confidence_level
        summaries = {}
        
        metrics = ['ESS', 'ESS_per_second', 'Acceptance_rate', 'Time_per_ESS', 'Efficiency']
        
        for sampler in self.processed_results['Sampler'].unique():
            sampler_data = self.processed_results[self.processed_results['Sampler'] == sampler]
            summaries[sampler] = {}
            
            for metric in metrics:
                if metric in sampler_data.columns:
                    values = sampler_data[metric].dropna()
                    
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std()
                        median_val = values.median()
                        
                        # Confidence interval (assuming normal distribution)
                        if len(values) > 1:
                            ci_margin = stats.t.ppf(1 - alpha/2, len(values) - 1) * (std_val / np.sqrt(len(values)))
                            ci_lower = mean_val - ci_margin
                            ci_upper = mean_val + ci_margin
                        else:
                            ci_lower = ci_upper = mean_val
                        
                        summaries[sampler][metric] = StatisticalSummary(
                            mean=mean_val,
                            std=std_val,
                            median=median_val,
                            ci_lower=ci_lower,
                            ci_upper=ci_upper,
                            sample_size=len(values)
                        )
        
        return summaries
    
    def perform_statistical_tests(self, 
                                baseline_method: str = 'Standard Metropolis',
                                metric: str = 'ESS_per_second',
                                alpha: float = 0.05) -> Dict[str, ComparisonResult]:
        """
        Perform statistical significance tests comparing methods to baseline.
        
        Args:
            baseline_method: Baseline method for comparison
            metric: Metric to test
            alpha: Significance level
            
        Returns:
            Dictionary of comparison test results
        """
        if self.processed_results is None or self.processed_results.empty:
            return {}
        
        test_results = {}
        
        # Get baseline data
        baseline_data = self.processed_results[
            self.processed_results['Sampler'] == baseline_method
        ][metric].dropna()
        
        if len(baseline_data) < 2:
            warnings.warn(f"Insufficient baseline data for {baseline_method}")
            return {}
        
        # Test each method against baseline
        for sampler in self.processed_results['Sampler'].unique():
            if sampler == baseline_method:
                continue
            
            sampler_data = self.processed_results[
                self.processed_results['Sampler'] == sampler
            ][metric].dropna()
            
            if len(sampler_data) < 2:
                continue
            
            # Perform Mann-Whitney U test (non-parametric)
            try:
                statistic, p_value = stats.mannwhitneyu(
                    sampler_data, baseline_data, alternative='two-sided'
                )
                
                # Effect size (Cohen's d approximation)
                pooled_std = np.sqrt(
                    ((len(sampler_data) - 1) * sampler_data.var() + 
                     (len(baseline_data) - 1) * baseline_data.var()) /
                    (len(sampler_data) + len(baseline_data) - 2)
                )
                
                if pooled_std > 0:
                    effect_size = (sampler_data.mean() - baseline_data.mean()) / pooled_std
                else:
                    effect_size = 0.0
                
                # Interpret effect size
                if abs(effect_size) < 0.2:
                    interpretation = "negligible effect"
                elif abs(effect_size) < 0.5:
                    interpretation = "small effect"
                elif abs(effect_size) < 0.8:
                    interpretation = "medium effect"  
                else:
                    interpretation = "large effect"
                
                test_results[sampler] = ComparisonResult(
                    test_name='Mann-Whitney U',
                    statistic=statistic,
                    p_value=p_value,
                    effect_size=effect_size,
                    significant=p_value < alpha,
                    interpretation=interpretation
                )
                
            except Exception as e:
                warnings.warn(f"Statistical test failed for {sampler}: {e}")
                continue
        
        self.statistical_tests = test_results
        return test_results
    
    def get_top_performers(self, 
                         metric: str = 'ESS_per_second',
                         n_top: int = 3) -> List[Tuple[str, float]]:
        """
        Get top performing methods by metric.
        
        Args:
            metric: Performance metric
            n_top: Number of top performers to return
            
        Returns:
            List of (sampler_name, performance) tuples
        """
        if self.processed_results is None or self.processed_results.empty:
            return []
        
        # Compute average performance by sampler
        sampler_performance = self.processed_results.groupby('Sampler')[metric].mean()
        top_performers = sampler_performance.nlargest(n_top)
        
        return list(top_performers.items())
    
    def generate_performance_ranking(self, 
                                   metrics: List[str] = None) -> pd.DataFrame:
        """
        Generate comprehensive performance ranking across multiple metrics.
        
        Args:
            metrics: List of metrics to include in ranking
            
        Returns:
            DataFrame with rankings
        """
        if self.processed_results is None or self.processed_results.empty:
            return pd.DataFrame()
        
        if metrics is None:
            metrics = ['ESS_per_second', 'Efficiency', 'Performance_score']
        
        # Compute average performance by sampler
        rankings = []
        
        for sampler in self.processed_results['Sampler'].unique():
            sampler_data = self.processed_results[self.processed_results['Sampler'] == sampler]
            
            row = {'Sampler': sampler}
            
            for metric in metrics:
                if metric in sampler_data.columns:
                    values = sampler_data[metric].dropna()
                    if len(values) > 0:
                        row[f'{metric}_mean'] = values.mean()
                        row[f'{metric}_rank'] = 0  # Will be filled in
            
            rankings.append(row)
        
        ranking_df = pd.DataFrame(rankings)
        
        # Compute ranks (1 = best)
        for metric in metrics:
            mean_col = f'{metric}_mean'
            rank_col = f'{metric}_rank'
            
            if mean_col in ranking_df.columns:
                ranking_df[rank_col] = ranking_df[mean_col].rank(ascending=False, method='min')
        
        # Compute overall rank (average of individual ranks)
        rank_cols = [f'{metric}_rank' for metric in metrics if f'{metric}_rank' in ranking_df.columns]
        if rank_cols:
            ranking_df['Overall_rank'] = ranking_df[rank_cols].mean(axis=1)
            ranking_df = ranking_df.sort_values('Overall_rank')
        
        return ranking_df


def generate_experiment_report(results: Dict[str, Any], 
                             output_dir: str,
                             include_plots: bool = True) -> str:
    """
    Generate complete experimental report.
    
    Creates a comprehensive report including:
    - Executive summary of results
    - Statistical significance tests
    - Performance improvement quantification  
    - Recommendations for usage
    
    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save report
        include_plots: Whether to include plot references
        
    Returns:
        Path to generated report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize results processor
    processor = ResultsProcessor(results)
    
    # Generate report content
    report_lines = []
    
    # Header
    report_lines.extend([
        "# Comprehensive MCMC Sampler Evaluation Report",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        ""
    ])
    
    # Performance overview
    if processor.processed_results is not None and not processor.processed_results.empty:
        n_samplers = processor.processed_results['Sampler'].nunique()
        n_distributions = processor.processed_results['Distribution'].nunique()
        total_experiments = len(processor.processed_results)
        
        report_lines.extend([
            f"This report presents results from {total_experiments} experiments",
            f"comparing {n_samplers} MCMC samplers across {n_distributions} test distributions.",
            ""
        ])
        
        # Top performers
        top_performers = processor.get_top_performers(n_top=3)
        if top_performers:
            report_lines.append("### Top Performing Methods (ESS per Second)")
            for i, (sampler, performance) in enumerate(top_performers, 1):
                report_lines.append(f"{i}. **{sampler}**: {performance:.2f}")
            report_lines.append("")
    
    # Statistical analysis
    report_lines.extend([
        "## Statistical Analysis",
        ""
    ])
    
    # Performance improvements
    improvements = processor.compute_improvement_factors()
    if improvements:
        report_lines.append("### Performance Improvements Over Baseline")
        for sampler, improvement in sorted(improvements.items(), 
                                         key=lambda x: x[1], reverse=True):
            improvement_pct = (improvement - 1) * 100
            report_lines.append(f"- **{sampler}**: {improvement_pct:+.1f}% improvement")
        report_lines.append("")
    
    # Statistical significance tests
    stat_tests = processor.perform_statistical_tests()
    if stat_tests:
        report_lines.append("### Statistical Significance Tests")
        report_lines.append("Compared to Standard Metropolis baseline:")
        report_lines.append("")
        
        for sampler, test_result in stat_tests.items():
            significance = "**Significant**" if test_result.significant else "Not significant"
            report_lines.append(
                f"- **{sampler}**: {significance} "
                f"(p={test_result.p_value:.4f}, {test_result.interpretation})"
            )
        report_lines.append("")
    
    # Performance ranking
    ranking_df = processor.generate_performance_ranking()
    if not ranking_df.empty:
        report_lines.extend([
            "### Overall Performance Ranking",
            ""
        ])
        
        for i, (_, row) in enumerate(ranking_df.iterrows(), 1):
            report_lines.append(f"{i}. **{row['Sampler']}** (Rank: {row['Overall_rank']:.1f})")
        report_lines.append("")
    
    # Performance table
    perf_table = processor.generate_performance_table(format_style='markdown')
    if isinstance(perf_table, str) and perf_table.strip():
        report_lines.extend([
            "## Detailed Performance Results",
            "",
            perf_table,
            ""
        ])
    
    # Recommendations
    report_lines.extend([
        "## Recommendations",
        ""
    ])
    
    if improvements:
        best_method = max(improvements.items(), key=lambda x: x[1])
        best_improvement = (best_method[1] - 1) * 100
        
        report_lines.extend([
            f"1. **Best Overall Method**: {best_method[0]} shows {best_improvement:.1f}% improvement",
            "2. **For High-Dimensional Problems**: Consider Hessian-aware methods for ill-conditioned targets",
            "3. **For Well-Conditioned Problems**: Standard methods may be sufficient",
            "4. **Computational Budget**: Balance accuracy vs computational cost based on requirements",
            ""
        ])
    
    # Technical details
    report_lines.extend([
        "## Technical Details",
        "",
        "### Metrics Computed",
        "- **ESS**: Effective Sample Size",
        "- **ESS/sec**: Effective samples per second",
        "- **Acceptance Rate**: Fraction of proposals accepted",
        "- **Time per ESS**: Computational cost per effective sample",
        "",
        "### Statistical Tests",
        "- Mann-Whitney U test for non-parametric comparison",
        "- Effect size computed using Cohen's d approximation",
        "- Significance level: α = 0.05",
        ""
    ])
    
    if include_plots:
        report_lines.extend([
            "### Figures",
            "- Performance comparison plots: `sampler_comparison_grid.png`",
            "- Dimensional scaling analysis: `dimensional_scaling_analysis.png`", 
            "- Convergence diagnostics: `convergence_*.png`",
            ""
        ])
    
    # Footer
    report_lines.extend([
        "---",
        "*Report generated by Hessian-Aware Sampling Benchmark Suite*"
    ])
    
    # Write report
    report_text = "\n".join(report_lines)
    report_path = output_path / "experiment_report.md"
    
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    # Also save as JSON for programmatic access
    report_data = {
        'metadata': {
            'generated_on': datetime.now().isoformat(),
            'n_samplers': processor.processed_results['Sampler'].nunique() if processor.processed_results is not None else 0,
            'n_experiments': len(processor.processed_results) if processor.processed_results is not None else 0
        },
        'top_performers': top_performers if 'top_performers' in locals() else [],
        'improvements': improvements,
        'statistical_tests': {k: {
            'p_value': v.p_value,
            'effect_size': v.effect_size, 
            'significant': v.significant,
            'interpretation': v.interpretation
        } for k, v in stat_tests.items()},
        'performance_table': processor.generate_performance_table(format_style='dataframe').to_dict('records') if processor.processed_results is not None else []
    }
    
    with open(output_path / "experiment_report.json", 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    print(f"📋 Experiment report generated: {report_path}")
    return str(report_path)


def create_latex_table(data: pd.DataFrame, 
                      caption: str = "Performance Comparison",
                      label: str = "tab:performance") -> str:
    """
    Create publication-ready LaTeX table.
    
    Args:
        data: DataFrame to convert
        caption: Table caption
        label: LaTeX label
        
    Returns:
        LaTeX table string
    """
    processor = ResultsProcessor({})  # Empty processor just for formatting
    
    latex_str = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{'l' * len(data.columns)}}}
\\toprule
{' & '.join(data.columns)} \\\\
\\midrule
"""
    
    for _, row in data.iterrows():
        row_str = ' & '.join(str(val) for val in row.values)
        latex_str += row_str + " \\\\\n"
    
    latex_str += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex_str


def compute_statistical_significance(group1: np.ndarray, 
                                   group2: np.ndarray,
                                   test_type: str = 'mann_whitney') -> Dict[str, Any]:
    """
    Compute statistical significance between two groups.
    
    Args:
        group1: First group of values
        group2: Second group of values  
        test_type: Type of statistical test
        
    Returns:
        Dictionary with test results
    """
    if len(group1) < 2 or len(group2) < 2:
        return {'error': 'Insufficient data for statistical test'}
    
    try:
        if test_type == 'mann_whitney':
            statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            test_name = 'Mann-Whitney U'
            
        elif test_type == 'welch_t':
            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            test_name = "Welch's t-test"
            
        elif test_type == 'ks':
            statistic, p_value = stats.ks_2samp(group1, group2)
            test_name = 'Kolmogorov-Smirnov'
            
        else:
            return {'error': f'Unknown test type: {test_type}'}
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + 
                             (len(group2) - 1) * group2.var()) /
                            (len(group1) + len(group2) - 2))
        
        effect_size = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0.0
        
        return {
            'test_name': test_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'effect_size': float(effect_size),
            'significant': p_value < 0.05,
            'group1_stats': {
                'mean': float(group1.mean()),
                'std': float(group1.std()),
                'n': len(group1)
            },
            'group2_stats': {
                'mean': float(group2.mean()),
                'std': float(group2.std()),
                'n': len(group2)
            }
        }
        
    except Exception as e:
        return {'error': f'Statistical test failed: {str(e)}'}


if __name__ == "__main__":
    print("Results generator module loaded successfully")