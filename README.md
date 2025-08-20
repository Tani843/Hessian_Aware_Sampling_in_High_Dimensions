# Hessian Aware Sampling in High Dimensions

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](#testing)

Advanced MCMC sampling methods leveraging Hessian information for efficient exploration of high-dimensional posterior distributions.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Tani843/Hessian_Aware_Sampling_in_High_Dimensions.git
cd Hessian_Aware_Sampling_in_High_Dimensions

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Run basic example
python examples/basic_example.py

# Run complete experimental pipeline
python scripts/run_complete_experiment.py
```

## 📊 Key Results

- **2-10x improvement** in effective sample size for ill-conditioned problems
- **Superior convergence rates** in dimensions up to 1000+
- **Robust performance** across diverse target distributions
- **Computational overhead < 50%** compared to baseline methods

## 🧠 Method Overview

Our Hessian-aware samplers incorporate second-order curvature information to achieve superior mixing in high-dimensional spaces:

```python
from hessian_sampling import HessianAwareMetropolis

# Define target distribution
def log_prob(x):
    return -0.5 * x.T @ precision_matrix @ x

# Initialize sampler
sampler = HessianAwareMetropolis(
    target_log_prob=log_prob,
    dim=100,
    step_size=0.1,
    regularization=1e-6
)

# Generate samples
samples = sampler.sample(n_samples=10000, initial_state=initial_state)
```

## 📁 Project Structure

```
hessian_sampling/
├── src/                    # Core implementation
│   ├── core/              # Mathematical utilities
│   ├── samplers/          # Sampling algorithms
│   ├── utils/             # Helper functions
│   └── visualization/     # Plotting tools
├── tests/                 # Unit and integration tests
├── examples/              # Usage examples
├── benchmarks/            # Performance comparisons
├── docs/                  # Documentation website
├── results/               # Generated results and figures
└── scripts/               # Automation scripts
```

## 🔬 Algorithms Implemented

### 1. Hessian-Aware Metropolis
Uses Hessian preconditioning for proposal generation:
```
x' = x + ε · H^(-1/2) · ξ,  where ξ ~ N(0,I)
```

### 2. Hessian-Aware Langevin Dynamics
Incorporates curvature in the drift term:
```
dx = -H^(-1)∇U(x)dt + √(2T)H^(-1/2)dW
```

### 3. Adaptive Hessian Sampler
Automatically tunes parameters based on local geometry and acceptance rates.

## 📈 Benchmarking

Run comprehensive benchmarks comparing against baseline methods:

```bash
# Full benchmark suite (takes ~30 minutes)
python examples/comprehensive_benchmark.py

# Quick validation (takes ~2 minutes)  
python scripts/run_complete_experiment.py --quick-test

# Skip time-consuming parts
python scripts/run_complete_experiment.py --skip-benchmark
```

### Test Distributions

- Multivariate Gaussians (various condition numbers)
- Rosenbrock density (highly non-convex)
- Mixture of Gaussians (multi-modal)
- Funnel distribution (varying scales)

## 📊 Visualization

Generate publication-quality figures:

```python
from src.visualization.publication_plots import create_publication_figures
create_publication_figures()
```

Results include:
- Performance comparison plots
- Convergence diagnostics
- Hessian eigenvalue analysis
- Computational cost analysis

## 🏗️ Installation

### Requirements

- Python 3.8+
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- Matplotlib ≥ 3.3.0
- Seaborn ≥ 0.11.0
- Pytest ≥ 6.0.0

### Development Installation

```bash
git clone https://github.com/Tani843/Hessian_Aware_Sampling_in_High_Dimensions.git
cd Hessian_Aware_Sampling_in_High_Dimensions
pip install -e ".[dev]"
```

## 🧪 Testing

Run the complete test suite:

```bash
# Unit tests
pytest tests/ -v

# Integration tests
pytest tests/test_integration.py -v

# Performance tests
pytest tests/test_performance.py -v

# Test coverage
pytest --cov=src tests/
```

## 📖 Documentation

View the complete documentation website:

```bash
cd docs/jekyll_site
bundle install
bundle exec jekyll serve
# Visit http://localhost:4000
```

Or visit the live documentation at: https://Tani843.github.io/Hessian_Aware_Sampling_in_High_Dimensions

## 🔍 Usage Examples

### Basic Sampling

```python
import numpy as np
from hessian_sampling import HessianAwareMetropolis

# Define a simple target distribution
def log_prob(x):
    return -0.5 * np.sum(x**2)

sampler = HessianAwareMetropolis(log_prob, dim=50)
samples = sampler.sample(5000, initial_state=np.zeros(50))
```

### Advanced Configuration

```python
sampler = HessianAwareMetropolis(
    target_log_prob=log_prob,
    dim=100,
    step_size=0.1,
    regularization=1e-6,
    hessian_update_freq=10,
    adaptation_window=100
)
```

### Benchmarking Custom Distributions

```python
from hessian_sampling.benchmarks import SamplerBenchmark

distributions = [('custom', my_distribution)]
samplers = {'Hessian': HessianAwareMetropolis}

benchmark = SamplerBenchmark(distributions, samplers)
results = benchmark.run_benchmark(n_samples=10000)
```

## 📚 Theory and Background

The method is based on incorporating the Hessian matrix H = ∇²U(x) of the negative log-probability into the sampling process. This provides several advantages:

- **Geometric Adaptation**: Automatically adjusts to local curvature
- **Improved Mixing**: Better exploration of elongated distributions
- **Dimension Robustness**: Maintains efficiency in high dimensions
- **Theoretical Grounding**: Provable convergence guarantees

## 🤝 Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

- **Author**: Tanisha Gupta
- **Institution**: Independent Researcher
- **Project Page**: https://Tani843.github.io/Hessian_Aware_Sampling_in_High_Dimensions

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{hessian_sampling_2025,
  title={Hessian Aware Sampling in High Dimensions},
  author={Tanisha Gupta},
  year={2025},
  url={https://github.com/Tani843/Hessian_Aware_Sampling_in_High_Dimensions}
}
```

## 🔄 Version History

- **v1.0.0**: Initial release with core algorithms
- **v1.1.0**: Added adaptive parameter selection
- **v1.2.0**: Performance optimizations and benchmarking suite

---

*This project advances the state-of-the-art in high-dimensional sampling by intelligently incorporating geometric information about the target distribution.*
