---
layout: page
title: "Results"
subtitle: "Comprehensive benchmarking and performance analysis"
---
## Performance Results

Our comprehensive experimental validation demonstrates significant improvements across all performance metrics:

### Method Comparison Overview
![Method Comparison](../assets/images/plots/fig1_comparison.png)
*Figure 1: Comprehensive method comparison showing Hessian-aware improvements*

### Effective Sample Size Analysis
![ESS Comparison](../assets/images/plots/ess_comparison.png)
*Figure 2: Effective Sample Size comparison demonstrating 2-15x improvements*

### Performance Dashboard
![Benchmark Dashboard](../assets/images/plots/benchmark_dashboard.png)
*Figure 3: Complete performance metrics dashboard*

### Dimensional Scaling Analysis
![Scaling Analysis](../assets/images/plots/fig2_scaling.png)
*Figure 4: Performance scaling with increasing dimensions*
### Hessian Eigenvalue Analysis
![Hessian Analysis](../assets/images/plots/fig3_hessian.png)
*Figure 5: Hessian eigenvalue distribution and conditioning analysis*

### Cost vs Accuracy Tradeoff
![Cost vs Accuracy](../assets/images/plots/fig4_cost_accuracy.png)
*Figure 6: Computational cost versus sampling accuracy*

### Convergence Diagnostics
![Trace Plots](../assets/images/plots/trace_plots.png)
*Figure 7: Sample trace plots demonstrating superior convergence*

### Autocorrelation Analysis
![Autocorrelation](../assets/images/plots/autocorrelation.png)
*Figure 8: Autocorrelation functions showing faster decay*
### Detailed Cost Analysis
![Cost vs Accuracy Detailed](../assets/images/plots/cost_vs_accuracy.png)
*Figure 9: Detailed cost-accuracy relationship*

## Key Findings

- **HMC**: 588 ESS/sec (best overall performance)
- **Hessian Metropolis**: 150 ESS/sec (3.8x improvement)
- **Standard Metropolis**: 40 ESS/sec (baseline)
- **Computational overhead**: < 50% increase for significant gains
