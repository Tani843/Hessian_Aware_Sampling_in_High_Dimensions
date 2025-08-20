#!/usr/bin/env python3
"""
Create algorithmic diagrams for the Hessian Aware Sampling project.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
import os

# Set style for diagrams
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'figure.figsize': (12, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def create_algorithm_flowchart(save_path="assets/images/diagrams/algorithm_flowchart.png"):
    """Create flowchart for Hessian-aware sampling algorithm."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    colors = {
        'start': '#4CAF50',
        'process': '#2196F3', 
        'decision': '#FF9800',
        'hessian': '#9C27B0',
        'update': '#F44336'
    }
    
    # Helper function to add boxes
    def add_box(x, y, width, height, text, color, text_color='white'):
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                            boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black',
                            linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', 
               fontsize=9, fontweight='bold', color=text_color,
               wrap=True)
    
    # Helper function to add arrows
    def add_arrow(x1, y1, x2, y2, text=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        if text:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.2, mid_y, text, fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Create flowchart
    add_box(5, 11, 2.5, 0.8, "Initialize\nθ₀, ε, regularization", colors['start'])
    
    add_box(5, 9.5, 2.5, 0.8, "Current state: θₜ", colors['process'])
    add_arrow(5, 10.6, 5, 9.9)
    
    add_box(5, 8, 3, 0.8, "Compute ∇log π(θₜ)", colors['process'])
    add_arrow(5, 9.1, 5, 8.4)
    
    add_box(5, 6.5, 3.5, 0.8, "Compute Hessian ∇²log π(θₜ)", colors['hessian'])
    add_arrow(5, 7.6, 5, 6.9)
    
    add_box(2.5, 5, 2.5, 1, "H = ∇²log π(θₜ)\n+ λI", colors['hessian'])
    add_box(7.5, 5, 2.5, 1, "Cholesky: L = chol(H)\nH⁻¹/² = L⁻ᵀ", colors['hessian'])
    add_arrow(4.25, 6.1, 3.5, 5.5)
    add_arrow(5.75, 6.1, 6.5, 5.5)
    
    add_box(5, 3.5, 4, 1, "Proposal:\nθ' = θₜ + ε·H⁻¹/²·ξ\nwhere ξ ~ N(0,I)", colors['update'])
    add_arrow(2.5, 4.5, 4, 4)
    add_arrow(7.5, 4.5, 6, 4)
    
    add_box(5, 2, 3, 0.8, "Accept/Reject\nusing MH criterion", colors['decision'])
    add_arrow(5, 3, 5, 2.4)
    
    add_box(8, 0.8, 1.5, 0.6, "Accept\nθₜ₊₁ = θ'", colors['start'])
    add_box(2, 0.8, 1.5, 0.6, "Reject\nθₜ₊₁ = θₜ", colors['update'])
    
    add_arrow(6.2, 1.7, 7.3, 1.2, "α > u")
    add_arrow(3.8, 1.7, 2.7, 1.2, "α ≤ u")
    
    # Return arrow
    ax.annotate('', xy=(1, 9.5), xytext=(1, 1.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray', linestyle='--'))
    ax.text(0.7, 5.5, 't = t+1', rotation=90, va='center', 
           fontsize=8, color='gray')
    
    add_arrow(2, 9.5, 3.8, 9.5)
    
    plt.title("Hessian-Aware Metropolis Algorithm", fontsize=16, fontweight='bold', pad=20)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Algorithm flowchart saved to {save_path}")

def create_preconditioning_diagram(save_path="assets/images/diagrams/hessian_preconditioning_diagram.png"):
    """Create diagram showing Hessian preconditioning effect."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Hessian Preconditioning Effect', fontsize=16, fontweight='bold')
    
    # Create data
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # 1. Original ill-conditioned distribution
    ax = axes[0]
    kappa = 100
    Z1 = np.exp(-0.5 * (X**2 + kappa * Y**2))
    
    contours = ax.contour(X, Y, Z1, levels=10, colors='red', alpha=0.8)
    ax.set_title('Original Distribution\n(Ill-conditioned)', fontsize=12, fontweight='bold')
    ax.set_xlabel('θ₁')
    ax.set_ylabel('θ₂')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add sample trajectory (poor mixing)
    np.random.seed(42)
    n_steps = 100
    trajectory = np.zeros((n_steps, 2))
    current = np.array([2.0, 0.1])
    
    for i in range(n_steps):
        # Simulate poor mixing with small steps
        proposal = current + 0.1 * np.random.randn(2) * np.array([1, 0.1])
        if np.random.rand() < 0.3:  # Low acceptance
            current = proposal
        trajectory[i] = current
    
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.7, linewidth=2, label='Sample path')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, zorder=5, label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, zorder=5, label='End')
    ax.legend(fontsize=8)
    
    # 2. Hessian matrix visualization
    ax = axes[1] 
    H = np.array([[1, 0], [0, kappa]])
    
    # Create Hessian visualization
    eigenvals, eigenvecs = np.linalg.eig(H)
    
    # Plot eigenvalue ellipse
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = np.sqrt(eigenvals[0]) * np.cos(theta)
    ellipse_y = np.sqrt(eigenvals[1]) * np.sin(theta)
    
    # Transform by eigenvectors
    ellipse = np.vstack([ellipse_x, ellipse_y])
    ellipse_transformed = eigenvecs @ ellipse
    
    ax.plot(ellipse_transformed[0], ellipse_transformed[1], 'purple', linewidth=3, label='Hessian ellipse')
    ax.arrow(0, 0, eigenvecs[0,0]*3, eigenvecs[1,0]*3, head_width=0.2, head_length=0.2, 
             fc='blue', ec='blue', label=f'λ₁={eigenvals[0]:.1f}')
    ax.arrow(0, 0, eigenvecs[0,1]*0.3, eigenvecs[1,1]*0.3, head_width=0.2, head_length=0.2, 
             fc='red', ec='red', label=f'λ₂={eigenvals[1]:.1f}')
    
    ax.set_title('Hessian Matrix\nEigenstructure', fontsize=12, fontweight='bold')
    ax.set_xlabel('θ₁')
    ax.set_ylabel('θ₂') 
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    
    # 3. Preconditioned distribution
    ax = axes[2]
    Z2 = np.exp(-0.5 * (X**2 + Y**2))  # Well-conditioned
    
    contours = ax.contour(X, Y, Z2, levels=10, colors='blue', alpha=0.8)
    ax.set_title('Preconditioned Distribution\n(Well-conditioned)', fontsize=12, fontweight='bold')
    ax.set_xlabel('θ₁')
    ax.set_ylabel('θ₂')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add improved sample trajectory
    trajectory2 = np.zeros((n_steps, 2))
    current = np.array([2.0, 2.0])
    
    for i in range(n_steps):
        # Simulate good mixing with reasonable steps
        proposal = current + 0.5 * np.random.randn(2)
        if np.random.rand() < 0.7:  # Higher acceptance
            current = proposal
        trajectory2[i] = current
    
    ax.plot(trajectory2[:, 0], trajectory2[:, 1], 'g-', alpha=0.7, linewidth=2, label='Sample path')
    ax.scatter(trajectory2[0, 0], trajectory2[0, 1], c='green', s=100, zorder=5, label='Start')
    ax.scatter(trajectory2[-1, 0], trajectory2[-1, 1], c='red', s=100, zorder=5, label='End')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Preconditioning diagram saved to {save_path}")

def create_sampling_architecture(save_path="assets/images/diagrams/sampling_architecture.png"):
    """Create system architecture diagram."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors for different components
    colors = {
        'input': '#E3F2FD',
        'core': '#1976D2',
        'hessian': '#7B1FA2',
        'sampler': '#388E3C',
        'output': '#F57C00',
        'util': '#795548'
    }
    
    def add_component(x, y, width, height, title, details, color, text_color='black'):
        # Main box
        rect = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black',
                             linewidth=2)
        ax.add_patch(rect)
        
        # Title
        ax.text(x + width/2, y + height - 0.3, title,
               ha='center', va='center', fontsize=11, fontweight='bold',
               color=text_color)
        
        # Details
        for i, detail in enumerate(details):
            ax.text(x + width/2, y + height - 0.8 - i*0.25, f"• {detail}",
                   ha='center', va='center', fontsize=8, color=text_color)
    
    def add_arrow(x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.2, label, ha='center', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Input layer
    add_component(0.5, 8, 3, 1.5, "Target Distribution",
                 ["log π(θ)", "∇ log π(θ)", "∇² log π(θ)"], 
                 colors['input'])
    
    # Core Hessian utilities
    add_component(5, 8.5, 3.5, 1, "Hessian Utils",
                 ["Automatic differentiation", "Finite differences", "Regularization"],
                 colors['hessian'], 'white')
    
    add_component(10, 8, 3, 1.5, "Math Utils", 
                 ["Cholesky decomposition", "Matrix square root", "Condition number"],
                 colors['util'], 'white')
    
    # Samplers
    add_component(1, 5.5, 2.5, 2, "Hessian Metropolis",
                 ["Preconditioned proposals", "MH acceptance", "Adaptive step size"],
                 colors['sampler'], 'white')
    
    add_component(4.5, 5.5, 2.5, 2, "Hessian Langevin",
                 ["Gradient drift", "Hessian preconditioning", "Euler-Maruyama"],
                 colors['sampler'], 'white')
    
    add_component(8, 5.5, 2.5, 2, "Adaptive Hessian",
                 ["Online adaptation", "Regularization tuning", "Step size control"],
                 colors['sampler'], 'white')
    
    # Core sampling base
    add_component(5, 3, 4, 1.5, "Base Sampler Interface",
                 ["Sample generation", "Diagnostics", "Convergence monitoring", "Result storage"],
                 colors['core'], 'white')
    
    # Output components
    add_component(0.5, 0.5, 3, 1.5, "Performance Metrics",
                 ["ESS calculation", "Acceptance rates", "Convergence tests"],
                 colors['output'], 'white')
    
    add_component(5, 0.5, 3.5, 1.5, "Visualization",
                 ["Trace plots", "ACF analysis", "Benchmark plots"],
                 colors['output'], 'white')
    
    add_component(10, 0.5, 3, 1.5, "Results Storage",
                 ["Sample chains", "Diagnostics", "Metadata"],
                 colors['output'], 'white')
    
    # Add arrows showing data flow
    add_arrow(2, 8, 2.25, 7.5)  # Input to Hessian Metropolis
    add_arrow(2, 8, 5.75, 7.5)  # Input to Hessian Langevin
    add_arrow(2, 8, 9.25, 7.5)  # Input to Adaptive Hessian
    
    add_arrow(6.75, 8.5, 6.75, 4.5)  # Hessian Utils to Base
    add_arrow(11.5, 8, 11.5, 6.5)   # Math Utils to samplers
    add_arrow(11.5, 6.5, 9, 6.5)    # Math Utils connection
    
    add_arrow(2.25, 5.5, 6, 4.5)    # Samplers to Base
    add_arrow(5.75, 5.5, 7, 4.5)
    add_arrow(9.25, 5.5, 8, 4.5)
    
    add_arrow(6, 3, 2, 2)            # Base to Metrics
    add_arrow(7, 3, 6.75, 2)         # Base to Visualization
    add_arrow(8, 3, 11.5, 2)         # Base to Results
    
    plt.title("Hessian-Aware Sampling System Architecture", 
             fontsize=16, fontweight='bold', pad=20)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Architecture diagram saved to {save_path}")

def main():
    """Generate all algorithmic diagrams."""
    
    print("🎨 Creating Algorithmic Diagrams")
    print("=" * 40)
    
    # Create output directory
    os.makedirs("assets/images/diagrams", exist_ok=True)
    
    # Generate diagrams
    create_algorithm_flowchart()
    create_preconditioning_diagram()
    create_sampling_architecture()
    
    print("\n✅ All diagrams created successfully!")

if __name__ == "__main__":
    main()