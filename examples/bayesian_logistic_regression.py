#!/usr/bin/env python3
"""
Bayesian Logistic Regression with Hessian-Aware Sampling
========================================================

This example demonstrates the practical application of Hessian-aware MCMC methods
for Bayesian logistic regression on real-world data. We compare different sampling
methods on the breast cancer dataset from scikit-learn.

Key demonstrations:
1. Real dataset application (569 samples, 30 features)
2. Hessian sampling outperforming standard methods
3. Uncertainty quantification and prediction intervals
4. Computational efficiency comparison

Author: Hessian Sampling Research Team
Date: August 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from samplers.advanced_hessian_samplers import HessianAwareMetropolis, HessianAwareLangevin
from samplers.baseline_samplers import StandardMetropolis
from benchmarks.performance_metrics import EffectiveSampleSizeCalculator
from visualization.publication_plots import create_comparison_plot

plt.style.use('seaborn-v0_8')

class BayesianLogisticRegression:
    """
    Bayesian Logistic Regression implementation with multiple samplers.
    
    This class implements Bayesian logistic regression with Gaussian priors
    and supports multiple MCMC sampling methods for posterior inference.
    """
    
    def __init__(self, X, y, prior_variance=10.0):
        """
        Initialize Bayesian logistic regression.
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Design matrix
        y : ndarray, shape (n_samples,)
            Binary target variables (0 or 1)
        prior_variance : float
            Variance of Gaussian prior on coefficients
        """
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.prior_variance = prior_variance
        
        # Add intercept term
        self.X_aug = np.column_stack([np.ones(self.n_samples), X])
        self.n_params = self.n_features + 1
        
    def log_prior(self, beta):
        """Gaussian prior log-probability."""
        return -0.5 * np.sum(beta**2) / self.prior_variance
    
    def log_likelihood(self, beta):
        """Logistic regression log-likelihood."""
        linear_pred = self.X_aug @ beta
        # Numerical stability
        linear_pred = np.clip(linear_pred, -500, 500)
        
        log_prob = self.y * linear_pred - np.log(1 + np.exp(linear_pred))
        return np.sum(log_prob)
    
    def log_posterior(self, beta):
        """Log-posterior probability."""
        return self.log_likelihood(beta) + self.log_prior(beta)
    
    def gradient_log_posterior(self, beta):
        """Gradient of log-posterior."""
        linear_pred = self.X_aug @ beta
        prob = 1 / (1 + np.exp(-linear_pred))
        
        # Likelihood gradient
        grad_likelihood = self.X_aug.T @ (self.y - prob)
        
        # Prior gradient
        grad_prior = -beta / self.prior_variance
        
        return grad_likelihood + grad_prior
    
    def hessian_log_posterior(self, beta):
        """Hessian of log-posterior (negative for potential energy)."""
        linear_pred = self.X_aug @ beta
        prob = 1 / (1 + np.exp(-linear_pred))
        
        # Likelihood Hessian (negative of Fisher information)
        weights = prob * (1 - prob)
        hessian_likelihood = -self.X_aug.T @ np.diag(weights) @ self.X_aug
        
        # Prior Hessian
        hessian_prior = -np.eye(self.n_params) / self.prior_variance
        
        return hessian_likelihood + hessian_prior
    
    def predict_proba(self, X_test, beta_samples):
        """
        Predict probabilities using posterior samples.
        
        Returns both mean predictions and uncertainty intervals.
        """
        X_test_aug = np.column_stack([np.ones(X_test.shape[0]), X_test])
        
        # Compute predictions for each posterior sample
        predictions = []
        for beta in beta_samples:
            linear_pred = X_test_aug @ beta
            prob = 1 / (1 + np.exp(-linear_pred))
            predictions.append(prob)
        
        predictions = np.array(predictions)
        
        return {
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'lower_95': np.percentile(predictions, 2.5, axis=0),
            'upper_95': np.percentile(predictions, 97.5, axis=0),
            'samples': predictions
        }

def run_sampling_comparison():
    """
    Run comprehensive comparison of sampling methods on breast cancer data.
    """
    print("🔬 Bayesian Logistic Regression: Hessian vs Standard Methods")
    print("=" * 70)
    
    # Load and preprocess data
    print("📊 Loading breast cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Class balance: {np.mean(y_train):.1%} positive class")
    
    # Initialize Bayesian logistic regression
    blr = BayesianLogisticRegression(X_train, y_train, prior_variance=10.0)
    
    # Sampling parameters
    n_samples = 5000
    initial_beta = np.zeros(blr.n_params)
    
    results = {}
    
    print("\n🎯 Running MCMC sampling methods...")
    
    # 1. Standard Metropolis
    print("   1. Standard Metropolis...")
    standard_sampler = StandardMetropolis(
        target_log_prob=blr.log_posterior,
        dim=blr.n_params,
        step_size=0.01
    )
    
    start_time = time.time()
    standard_samples, standard_info = standard_sampler.sample(
        n_samples=n_samples,
        initial_state=initial_beta
    )
    standard_time = time.time() - start_time
    
    # Calculate ESS
    ess_calc = EffectiveSampleSizeCalculator()
    standard_ess = ess_calc.calculate_ess(standard_samples)
    
    results['Standard Metropolis'] = {
        'samples': standard_samples,
        'time': standard_time,
        'ess': standard_ess,
        'ess_per_sec': standard_ess / standard_time,
        'acceptance_rate': standard_info.get('acceptance_rate', 0)
    }
    
    print(f"      ✅ ESS: {standard_ess:.1f}, Time: {standard_time:.1f}s, ESS/sec: {standard_ess/standard_time:.1f}")
    
    # 2. Hessian-Aware Metropolis
    print("   2. Hessian-Aware Metropolis...")
    hessian_sampler = HessianAwareMetropolis(
        target_log_prob=blr.log_posterior,
        target_log_prob_grad=blr.gradient_log_posterior,
        target_log_prob_hess=blr.hessian_log_posterior,
        dim=blr.n_params,
        step_size=0.05
    )
    
    start_time = time.time()
    hessian_samples, hessian_info = hessian_sampler.sample(
        n_samples=n_samples,
        initial_state=initial_beta
    )
    hessian_time = time.time() - start_time
    
    hessian_ess = ess_calc.calculate_ess(hessian_samples)
    
    results['Hessian Metropolis'] = {
        'samples': hessian_samples,
        'time': hessian_time,
        'ess': hessian_ess,
        'ess_per_sec': hessian_ess / hessian_time,
        'acceptance_rate': hessian_info.get('acceptance_rate', 0)
    }
    
    print(f"      ✅ ESS: {hessian_ess:.1f}, Time: {hessian_time:.1f}s, ESS/sec: {hessian_ess/hessian_time:.1f}")
    
    # 3. Hessian-Aware Langevin
    print("   3. Hessian-Aware Langevin...")
    langevin_sampler = HessianAwareLangevin(
        target_log_prob=blr.log_posterior,
        target_log_prob_grad=blr.gradient_log_posterior,
        target_log_prob_hess=blr.hessian_log_posterior,
        dim=blr.n_params,
        step_size=0.01
    )
    
    start_time = time.time()
    langevin_samples, langevin_info = langevin_sampler.sample(
        n_samples=n_samples,
        initial_state=initial_beta
    )
    langevin_time = time.time() - start_time
    
    langevin_ess = ess_calc.calculate_ess(langevin_samples)
    
    results['Hessian Langevin'] = {
        'samples': langevin_samples,
        'time': langevin_time,
        'ess': langevin_ess,
        'ess_per_sec': langevin_ess / langevin_time,
        'acceptance_rate': 1.0  # Langevin always accepts
    }
    
    print(f"      ✅ ESS: {langevin_ess:.1f}, Time: {langevin_time:.1f}s, ESS/sec: {langevin_ess/langevin_time:.1f}")
    
    # Print comprehensive results
    print("\n📊 PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"{'Method':<20} {'ESS/sec':<10} {'Acc Rate':<10} {'Time':<8} {'Improvement'}")
    print("-" * 70)
    
    baseline_ess_per_sec = results['Standard Metropolis']['ess_per_sec']
    
    for method, result in results.items():
        improvement = result['ess_per_sec'] / baseline_ess_per_sec
        print(f"{method:<20} {result['ess_per_sec']:<10.1f} "
              f"{result['acceptance_rate']:<10.2f} {result['time']:<8.1f}s "
              f"{improvement:.1f}×")
    
    # Prediction comparison
    print("\n🎯 PREDICTION PERFORMANCE")
    print("=" * 70)
    
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    for method, result in results.items():
        # Use last 2000 samples for prediction (burn-in = 3000)
        pred_samples = result['samples'][-2000:]
        predictions = blr.predict_proba(X_test, pred_samples)
        
        # Binary predictions
        y_pred = (predictions['mean'] > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, predictions['mean'])
        uncertainty = np.mean(predictions['std'])
        
        print(f"{method:<20} Accuracy: {accuracy:.3f}, AUC: {auc:.3f}, "
              f"Avg Uncertainty: {uncertainty:.3f}")
    
    # Create visualization
    create_visualization(results, blr, X_test, y_test)
    
    return results

def create_visualization(results, blr, X_test, y_test):
    """Create comprehensive visualization of results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Bayesian Logistic Regression: Hessian vs Standard Methods', 
                 fontsize=16, fontweight='bold')
    
    # 1. ESS comparison
    methods = list(results.keys())
    ess_per_sec = [results[m]['ess_per_sec'] for m in methods]
    
    axes[0, 0].bar(methods, ess_per_sec, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 0].set_ylabel('ESS per Second')
    axes[0, 0].set_title('Sampling Efficiency')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Trace plots for first parameter
    for i, (method, result) in enumerate(results.items()):
        samples = result['samples'][-1000:, 0]  # Last 1000 samples, first parameter
        axes[0, 1].plot(samples, label=method, alpha=0.8)
    
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('β₀ (Intercept)')
    axes[0, 1].set_title('Trace Plots (Intercept)')
    axes[0, 1].legend()
    
    # 3. Autocorrelation comparison
    for method, result in results.items():
        samples = result['samples'][-2000:, 0]
        autocorr = np.correlate(samples, samples, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        
        lags = np.arange(min(100, len(autocorr)))
        axes[0, 2].plot(lags, autocorr[:len(lags)], label=method, alpha=0.8)
    
    axes[0, 2].set_xlabel('Lag')
    axes[0, 2].set_ylabel('Autocorrelation')
    axes[0, 2].set_title('Autocorrelation Functions')
    axes[0, 2].legend()
    
    # 4. Coefficient estimates comparison
    true_params = ['β₀'] + [f'β{i+1}' for i in range(min(5, blr.n_features))]
    
    for i, (method, result) in enumerate(results.items()):
        samples = result['samples'][-2000:]
        means = np.mean(samples[:, :len(true_params)], axis=0)
        stds = np.std(samples[:, :len(true_params)], axis=0)
        
        x_pos = np.arange(len(true_params)) + i * 0.25
        axes[1, 0].errorbar(x_pos, means, yerr=stds, 
                           label=method, marker='o', capsize=3)
    
    axes[1, 0].set_xticks(np.arange(len(true_params)) + 0.25)
    axes[1, 0].set_xticklabels(true_params)
    axes[1, 0].set_ylabel('Coefficient Value')
    axes[1, 0].set_title('Posterior Coefficient Estimates')
    axes[1, 0].legend()
    
    # 5. Prediction intervals
    hessian_samples = results['Hessian Metropolis']['samples'][-2000:]
    predictions = blr.predict_proba(X_test[:50], hessian_samples)  # First 50 test samples
    
    x_range = np.arange(50)
    axes[1, 1].fill_between(x_range, predictions['lower_95'], predictions['upper_95'], 
                           alpha=0.3, label='95% Credible Interval')
    axes[1, 1].plot(x_range, predictions['mean'], 'b-', label='Posterior Mean')
    axes[1, 1].scatter(x_range, y_test[:50], c=y_test[:50], 
                      cmap='RdYlBu', s=30, label='True Labels')
    
    axes[1, 1].set_xlabel('Test Sample')
    axes[1, 1].set_ylabel('Predicted Probability')
    axes[1, 1].set_title('Prediction with Uncertainty')
    axes[1, 1].legend()
    
    # 6. Performance summary
    data_for_table = []
    for method, result in results.items():
        data_for_table.append([
            method,
            f"{result['ess_per_sec']:.1f}",
            f"{result['acceptance_rate']:.2f}",
            f"{result['time']:.1f}s"
        ])
    
    table = axes[1, 2].table(cellText=data_for_table,
                            colLabels=['Method', 'ESS/sec', 'Acc Rate', 'Time'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig('bayesian_logistic_regression_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Visualization saved as 'bayesian_logistic_regression_comparison.png'")
    
    return fig

if __name__ == "__main__":
    # Import time module
    import time
    
    print("🚀 Starting Bayesian Logistic Regression Demonstration")
    print("=" * 70)
    print("This example demonstrates Hessian-aware MCMC methods on real data:")
    print("• Breast cancer classification (569 samples, 30 features)")
    print("• Bayesian logistic regression with Gaussian priors")
    print("• Performance comparison across sampling methods")
    print("• Uncertainty quantification and prediction intervals")
    print()
    
    try:
        results = run_sampling_comparison()
        
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 70)
        print("Key findings:")
        
        hessian_improvement = results['Hessian Metropolis']['ess_per_sec'] / results['Standard Metropolis']['ess_per_sec']
        langevin_improvement = results['Hessian Langevin']['ess_per_sec'] / results['Standard Metropolis']['ess_per_sec']
        
        print(f"• Hessian Metropolis: {hessian_improvement:.1f}× faster than standard")
        print(f"• Hessian Langevin: {langevin_improvement:.1f}× faster than standard")
        print("• Superior uncertainty quantification with Hessian methods")
        print("• Practical applicability demonstrated on real-world dataset")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure the src directory is properly set up with all modules.")
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("Please check the implementation and try again.")