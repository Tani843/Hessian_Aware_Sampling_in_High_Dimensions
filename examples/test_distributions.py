"""
Test target distributions for evaluating Hessian-aware sampling.

This module provides various challenging distributions with known
properties to test the effectiveness of Hessian-aware sampling
algorithms in different scenarios.
"""

from typing import Callable, Tuple, Optional, Dict, Any
import numpy as np
import warnings
from scipy.stats import multivariate_normal, norm
from scipy.special import logsumexp


class TestDistribution:
    """Base class for test distributions."""
    
    def __init__(self, dim: int, name: str):
        self.dim = dim
        self.name = name
    
    def log_prob(self, x: np.ndarray) -> float:
        """Compute log probability density."""
        raise NotImplementedError
    
    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of log probability."""
        raise NotImplementedError
    
    def hessian_log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute Hessian of log probability."""
        raise NotImplementedError
    
    def sample_exact(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate exact samples (if possible)."""
        raise NotImplementedError
    
    def true_mean(self) -> np.ndarray:
        """True mean of the distribution (if known)."""
        raise NotImplementedError
    
    def true_cov(self) -> np.ndarray:
        """True covariance of the distribution (if known)."""
        raise NotImplementedError


class MultivariateGaussian(TestDistribution):
    """
    Multivariate Gaussian distribution with configurable condition number.
    
    Good for testing basic functionality and comparing with theoretical results.
    """
    
    def __init__(self, dim: int, condition_number: float = 10.0, seed: Optional[int] = None):
        super().__init__(dim, f"MultivariateGaussian_d{dim}_cond{condition_number}")
        
        self.condition_number = condition_number
        
        # Generate covariance matrix with specified condition number
        np.random.seed(seed)
        
        # Create random orthogonal matrix
        Q, _ = np.linalg.qr(np.random.randn(dim, dim))
        
        # Create eigenvalues with specified condition number
        eigenvals = np.logspace(0, np.log10(condition_number), dim)
        eigenvals = eigenvals / np.mean(eigenvals)  # Normalize
        
        # Construct covariance matrix
        self.cov = Q @ np.diag(eigenvals) @ Q.T
        self.mean = np.zeros(dim)
        
        # Precompute useful quantities
        self.cov_inv = np.linalg.inv(self.cov)
        self.log_det_cov = np.log(np.linalg.det(self.cov))
        self.log_norm_const = -0.5 * (dim * np.log(2 * np.pi) + self.log_det_cov)
    
    def log_prob(self, x: np.ndarray) -> float:
        """Compute log probability density."""
        diff = x - self.mean
        quad_form = diff @ self.cov_inv @ diff
        return self.log_norm_const - 0.5 * quad_form
    
    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of log probability."""
        diff = x - self.mean
        return -self.cov_inv @ diff
    
    def hessian_log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute Hessian of log probability."""
        return -self.cov_inv
    
    def sample_exact(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate exact samples."""
        np.random.seed(seed)
        return multivariate_normal.rvs(mean=self.mean, cov=self.cov, size=n_samples)
    
    def true_mean(self) -> np.ndarray:
        """True mean of the distribution."""
        return self.mean.copy()
    
    def true_cov(self) -> np.ndarray:
        """True covariance of the distribution."""
        return self.cov.copy()


class RosenbrockDensity(TestDistribution):
    """
    Rosenbrock density - challenging banana-shaped distribution.
    
    log p(x) ∝ -∑_{i=1}^{d-1} [a(x_{i+1} - x_i^2)^2 + b(1 - x_i)^2]
    
    Highly curved with narrow valley - excellent test for Hessian-aware methods.
    """
    
    def __init__(self, dim: int, a: float = 1.0, b: float = 100.0):
        super().__init__(dim, f"Rosenbrock_d{dim}_a{a}_b{b}")
        
        if dim < 2:
            raise ValueError("Rosenbrock density requires at least 2 dimensions")
        
        self.a = a
        self.b = b
    
    def log_prob(self, x: np.ndarray) -> float:
        """Compute log probability density."""
        log_prob = 0.0
        
        for i in range(self.dim - 1):
            term1 = self.a * (x[i+1] - x[i]**2)**2
            term2 = self.b * (1 - x[i])**2
            log_prob -= (term1 + term2)
        
        return log_prob
    
    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of log probability."""
        grad = np.zeros_like(x)
        
        for i in range(self.dim - 1):
            # Gradient w.r.t. x[i]
            grad[i] -= 4 * self.a * x[i] * (x[i]**2 - x[i+1])
            grad[i] -= 2 * self.b * (x[i] - 1)
            
            # Gradient w.r.t. x[i+1]
            grad[i+1] -= 2 * self.a * (x[i+1] - x[i]**2)
        
        return grad
    
    def hessian_log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute Hessian of log probability."""
        H = np.zeros((self.dim, self.dim))
        
        for i in range(self.dim - 1):
            # Diagonal terms
            H[i, i] -= 4 * self.a * (3 * x[i]**2 - x[i+1]) + 2 * self.b
            H[i+1, i+1] -= 2 * self.a
            
            # Off-diagonal terms
            H[i, i+1] -= -4 * self.a * x[i]
            H[i+1, i] -= -4 * self.a * x[i]
        
        return H
    
    def sample_exact(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """No exact sampling available - use approximate samples."""
        warnings.warn("No exact sampling available for Rosenbrock density")
        np.random.seed(seed)
        # Return samples near the mode (approximately at [1, 1, ...])
        return np.random.normal(1.0, 0.5, size=(n_samples, self.dim))
    
    def true_mean(self) -> np.ndarray:
        """Approximate mean (mode is at [1, 1, ...])."""
        return np.ones(self.dim)
    
    def true_cov(self) -> np.ndarray:
        """No analytical covariance available."""
        warnings.warn("No analytical covariance available for Rosenbrock density")
        return np.eye(self.dim)


class MixtureOfGaussians(TestDistribution):
    """
    Mixture of Gaussians - multimodal distribution.
    
    Tests ability to explore multiple modes and handle multimodality.
    """
    
    def __init__(self, 
                 dim: int, 
                 n_components: int = 3,
                 separation: float = 5.0,
                 component_var: float = 1.0,
                 seed: Optional[int] = None):
        super().__init__(dim, f"MixtureGaussians_d{dim}_comp{n_components}")
        
        self.n_components = n_components
        self.separation = separation
        self.component_var = component_var
        
        np.random.seed(seed)
        
        # Create component means (spread out in first few dimensions)
        self.means = []
        for i in range(n_components):
            mean = np.zeros(dim)
            if dim >= 2:
                angle = 2 * np.pi * i / n_components
                mean[0] = separation * np.cos(angle)
                mean[1] = separation * np.sin(angle)
            else:
                mean[0] = separation * (i - n_components // 2)
            self.means.append(mean)
        
        # Equal mixing weights
        self.weights = np.ones(n_components) / n_components
        
        # Same covariance for all components
        self.cov = component_var * np.eye(dim)
        self.cov_inv = np.linalg.inv(self.cov)
        self.log_det_cov = np.log(np.linalg.det(self.cov))
    
    def log_prob(self, x: np.ndarray) -> float:
        """Compute log probability density."""
        log_probs = []
        
        for i in range(self.n_components):
            diff = x - self.means[i]
            quad_form = diff @ self.cov_inv @ diff
            log_norm_const = -0.5 * (self.dim * np.log(2 * np.pi) + self.log_det_cov)
            log_prob_i = log_norm_const - 0.5 * quad_form + np.log(self.weights[i])
            log_probs.append(log_prob_i)
        
        return logsumexp(log_probs)
    
    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of log probability."""
        # Compute posterior probabilities
        log_probs = []
        for i in range(self.n_components):
            diff = x - self.means[i]
            quad_form = diff @ self.cov_inv @ diff
            log_norm_const = -0.5 * (self.dim * np.log(2 * np.pi) + self.log_det_cov)
            log_prob_i = log_norm_const - 0.5 * quad_form + np.log(self.weights[i])
            log_probs.append(log_prob_i)
        
        log_probs = np.array(log_probs)
        posterior_probs = np.exp(log_probs - logsumexp(log_probs))
        
        # Weighted gradient
        grad = np.zeros_like(x)
        for i in range(self.n_components):
            diff = x - self.means[i]
            grad -= posterior_probs[i] * self.cov_inv @ diff
        
        return grad
    
    def hessian_log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute Hessian of log probability."""
        # This is complex for mixtures - approximate with dominant component
        # Find closest component
        distances = [np.linalg.norm(x - mean) for mean in self.means]
        closest_idx = np.argmin(distances)
        
        # Use Hessian of closest component
        return -self.cov_inv
    
    def sample_exact(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate exact samples."""
        np.random.seed(seed)
        
        samples = []
        component_counts = np.random.multinomial(n_samples, self.weights)
        
        for i, count in enumerate(component_counts):
            if count > 0:
                component_samples = multivariate_normal.rvs(
                    mean=self.means[i], cov=self.cov, size=count
                )
                if count == 1:
                    component_samples = component_samples.reshape(1, -1)
                samples.append(component_samples)
        
        all_samples = np.vstack(samples)
        np.random.shuffle(all_samples)
        
        return all_samples
    
    def true_mean(self) -> np.ndarray:
        """True mean of the mixture."""
        mean = np.zeros(self.dim)
        for i in range(self.n_components):
            mean += self.weights[i] * self.means[i]
        return mean
    
    def true_cov(self) -> np.ndarray:
        """True covariance of the mixture."""
        mean = self.true_mean()
        
        cov = np.zeros((self.dim, self.dim))
        for i in range(self.n_components):
            diff = self.means[i] - mean
            cov += self.weights[i] * (self.cov + np.outer(diff, diff))
        
        return cov


class FunnelDistribution(TestDistribution):
    """
    Neal's funnel distribution - challenging geometry.
    
    x[0] ~ N(0, σ²)
    x[i] ~ N(0, exp(x[0])) for i > 0
    
    Creates a funnel shape that's difficult to sample efficiently.
    """
    
    def __init__(self, dim: int, sigma: float = 3.0):
        super().__init__(dim, f"Funnel_d{dim}_sigma{sigma}")
        
        self.sigma = sigma
    
    def log_prob(self, x: np.ndarray) -> float:
        """Compute log probability density."""
        # First coordinate: N(0, σ²)
        log_prob = -0.5 * (x[0] / self.sigma)**2 - 0.5 * np.log(2 * np.pi * self.sigma**2)
        
        # Other coordinates: N(0, exp(x[0]))
        if self.dim > 1:
            var_others = np.exp(x[0])
            for i in range(1, self.dim):
                log_prob -= 0.5 * x[i]**2 / var_others
                log_prob -= 0.5 * np.log(2 * np.pi * var_others)
        
        return log_prob
    
    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of log probability."""
        grad = np.zeros_like(x)
        
        # Gradient w.r.t. x[0]
        grad[0] = -x[0] / self.sigma**2
        
        if self.dim > 1:
            var_others = np.exp(x[0])
            sum_x_squared = np.sum(x[1:]**2)
            
            grad[0] += 0.5 * sum_x_squared / var_others - 0.5 * (self.dim - 1)
            
            # Gradient w.r.t. x[i] for i > 0
            for i in range(1, self.dim):
                grad[i] = -x[i] / var_others
        
        return grad
    
    def hessian_log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute Hessian of log probability."""
        H = np.zeros((self.dim, self.dim))
        
        # Hessian w.r.t. x[0]
        H[0, 0] = -1.0 / self.sigma**2
        
        if self.dim > 1:
            var_others = np.exp(x[0])
            sum_x_squared = np.sum(x[1:]**2)
            
            H[0, 0] -= 0.5 * sum_x_squared / var_others
            
            # Mixed partial derivatives
            for i in range(1, self.dim):
                H[0, i] = -x[i] / var_others
                H[i, 0] = -x[i] / var_others
                
                # Pure second derivatives
                H[i, i] = -1.0 / var_others
        
        return H
    
    def sample_exact(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate exact samples."""
        np.random.seed(seed)
        
        samples = np.zeros((n_samples, self.dim))
        
        # Sample first coordinate
        samples[:, 0] = np.random.normal(0, self.sigma, n_samples)
        
        # Sample other coordinates conditionally
        for i in range(n_samples):
            var_others = np.exp(samples[i, 0])
            std_others = np.sqrt(var_others)
            
            for j in range(1, self.dim):
                samples[i, j] = np.random.normal(0, std_others)
        
        return samples
    
    def true_mean(self) -> np.ndarray:
        """True mean of the distribution."""
        return np.zeros(self.dim)
    
    def true_cov(self) -> np.ndarray:
        """True covariance (approximate)."""
        cov = np.zeros((self.dim, self.dim))
        
        # Variance of first coordinate
        cov[0, 0] = self.sigma**2
        
        # Variance of other coordinates (marginal)
        # E[exp(x[0])] where x[0] ~ N(0, σ²)
        expected_var = np.exp(0.5 * self.sigma**2)
        
        for i in range(1, self.dim):
            cov[i, i] = expected_var
        
        return cov


def get_test_distribution(name: str, dim: int, **kwargs) -> TestDistribution:
    """
    Factory function to create test distributions.
    
    Args:
        name: Distribution name
        dim: Dimensionality
        **kwargs: Additional parameters
        
    Returns:
        TestDistribution instance
    """
    name_lower = name.lower()
    
    if name_lower in ['gaussian', 'multivariate_gaussian', 'mvn']:
        return MultivariateGaussian(dim, **kwargs)
    elif name_lower in ['rosenbrock', 'banana']:
        return RosenbrockDensity(dim, **kwargs)
    elif name_lower in ['mixture', 'mixture_gaussian', 'gmm']:
        return MixtureOfGaussians(dim, **kwargs)
    elif name_lower in ['funnel', 'neals_funnel']:
        return FunnelDistribution(dim, **kwargs)
    else:
        available = ['gaussian', 'rosenbrock', 'mixture', 'funnel']
        raise ValueError(f"Unknown distribution '{name}'. Available: {available}")


def create_test_suite(dim: int) -> Dict[str, TestDistribution]:
    """
    Create a suite of test distributions for benchmarking.
    
    Args:
        dim: Dimensionality
        
    Returns:
        Dictionary of test distributions
    """
    distributions = {
        'gaussian_easy': MultivariateGaussian(dim, condition_number=5.0),
        'gaussian_hard': MultivariateGaussian(dim, condition_number=100.0),
        'rosenbrock': RosenbrockDensity(dim),
        'mixture_2': MixtureOfGaussians(dim, n_components=2),
        'mixture_3': MixtureOfGaussians(dim, n_components=3),
        'funnel': FunnelDistribution(dim)
    }
    
    return distributions


# Example usage functions
def demo_distribution(dist_name: str, dim: int = 10):
    """Demonstrate a test distribution."""
    dist = get_test_distribution(dist_name, dim)
    
    print(f"\nDistribution: {dist.name}")
    print(f"Dimension: {dist.dim}")
    
    # Test point
    x = np.random.randn(dim)
    
    print(f"\nTest point: {x[:5]}..." if dim > 5 else f"\nTest point: {x}")
    print(f"Log probability: {dist.log_prob(x):.4f}")
    
    # Gradient
    try:
        grad = dist.grad_log_prob(x)
        print(f"Gradient norm: {np.linalg.norm(grad):.4f}")
    except NotImplementedError:
        print("Gradient: Not implemented")
    
    # Hessian
    try:
        hess = dist.hessian_log_prob(x)
        eigenvals = np.linalg.eigvals(hess)
        print(f"Hessian condition number: {np.max(eigenvals) / np.min(eigenvals):.2e}")
    except NotImplementedError:
        print("Hessian: Not implemented")
    
    # True statistics
    try:
        mean = dist.true_mean()
        print(f"True mean: {mean[:5]}..." if dim > 5 else f"True mean: {mean}")
    except NotImplementedError:
        print("True mean: Not available")


if __name__ == "__main__":
    # Demo all distributions
    dim = 5
    
    for dist_name in ['gaussian', 'rosenbrock', 'mixture', 'funnel']:
        demo_distribution(dist_name, dim)