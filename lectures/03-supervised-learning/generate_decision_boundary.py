import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))

# Generate sample data for two classes
np.random.seed(42)
n_points = 30

# Class 0 (Not Spam) - bottom left
x0_class0 = np.random.randn(n_points) * 0.8 + 2
x1_class0 = np.random.randn(n_points) * 0.8 + 2

# Class 1 (Spam) - top right
x0_class1 = np.random.randn(n_points) * 0.8 + 5
x1_class1 = np.random.randn(n_points) * 0.8 + 5

# Plot points
ax.scatter(x0_class0, x1_class0, c='blue', s=80, label='Not Spam (y=0)', alpha=0.7, edgecolors='black')
ax.scatter(x0_class1, x1_class1, c='red', s=80, label='Spam (y=1)', alpha=0.7, edgecolors='black')

# Decision boundary: θ₀ + θ₁x₁ + θ₂x₂ = 0
# Simplified: x₂ = -x₁ + 7 (diagonal line)
x_boundary = np.linspace(0, 8, 100)
y_boundary = -x_boundary + 7

ax.plot(x_boundary, y_boundary, 'g-', linewidth=3, label='Decision Boundary')

# Shade regions
ax.fill_between(x_boundary, y_boundary, 8, alpha=0.15, color='red')
ax.fill_between(x_boundary, 0, y_boundary, alpha=0.15, color='blue')

# Add region labels
ax.text(1.5, 6.5, 'Predict: Spam\n(P > 0.5)', fontsize=12, ha='center', color='darkred')
ax.text(5.5, 1.5, 'Predict: Not Spam\n(P ≤ 0.5)', fontsize=12, ha='center', color='darkblue')

# Add equation for decision boundary
ax.text(6.5, 5.8, r'$\boldsymbol{\theta}^\top \mathbf{x} = 0$', fontsize=14,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Labels and styling
ax.set_xlabel(r'$x_1$ (Exclamation marks)', fontsize=13)
ax.set_ylabel(r'$x_2$ (Has "FREE")', fontsize=13)
ax.set_title('Linear Decision Boundary in Logistic Regression', fontsize=14, fontweight='bold')
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/decision_boundary_matplotlib.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved decision_boundary_matplotlib.png")
