#!/usr/bin/env python3
"""
Diagram generator for Lecture 03: Supervised Learning
Uses REAL datasets (Iris, synthetic data) for all visualizations.
"""

import sys
import os
# Add parent directory to path to import shared style module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

# Import shared style
try:
    from themes.diagram_style import COLORS, setup_figure, save_svg, save_both
except ImportError:
    # Fallback if shared style not available
    COLORS = {
        'primary': '#1e3a5f',
        'accent': '#e85a4f',
        'success': '#2a9d8f',
        'warning': '#e9c46a',
        'blue': '#3b82f6',
        'purple': '#8b5cf6',
        'text': '#2d3748',
        'text_light': '#4a5568',
        'bg_light': '#f7fafc',
    }

    def setup_figure(figsize=(12, 6), bg_color='white'):
        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
        ax.set_facecolor(bg_color)
        ax.axis('off')
        return fig, ax

    def save_svg(fig, filename):
        os.makedirs('diagrams/svg', exist_ok=True)
        filepath = f'diagrams/svg/{filename}'
        fig.savefig(filepath, format='svg', bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  ✓ {filename}")

# =============================================================================
# 1. Classification vs Regression (with Real Iris Data)
# =============================================================================
def create_classification_vs_regression():
    """Side-by-side comparison using REAL Iris data for classification."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    # Load REAL Iris data
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data[:, :2]  # Sepal length and width
    y = iris.target
    target_names = iris.target_names

    # Classification plot
    ax1 = axes[0]
    ax1.set_facecolor('white')

    colors = [COLORS['blue'], COLORS['success'], COLORS['accent']]
    for i, (color, name) in enumerate(zip(colors, target_names)):
        ax1.scatter(X[y == i, 0], X[y == i, 1], c=color, s=100,
                   label=name, alpha=0.7, edgecolors='white', linewidth=1)

    # Add decision boundary example
    ax1.axvline(x=5.8, color=COLORS['primary'], linestyle='--', linewidth=2, alpha=0.5)

    ax1.set_title('CLASSIFICATION\n(Predict category)', fontsize=14, fontweight='bold',
                  color=COLORS['primary'], pad=15)
    ax1.set_xlabel('Sepal Length (cm)', fontsize=11)
    ax1.set_ylabel('Sepal Width (cm)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Regression plot (synthetic but realistic)
    ax2 = axes[1]
    ax2.set_facecolor('white')

    np.random.seed(42)
    x = np.linspace(0, 10, 40)
    y = 2.5 * x + 10 + np.random.randn(40) * 4

    ax2.scatter(x, y, c=COLORS['purple'], s=100, alpha=0.7,
              edgecolors='white', linewidth=1, label='Data points')

    # Regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_smooth = np.linspace(0, 10, 100)
    ax2.plot(x_smooth, p(x_smooth), color=COLORS['accent'],
            linewidth=3, label='Regression line')

    ax2.set_title('REGRESSION\n(Predict value)', fontsize=14, fontweight='bold',
                  color=COLORS['primary'], pad=15)
    ax2.set_xlabel('Input Feature (x)', fontsize=11)
    ax2.set_ylabel('Target Value (y)', fontsize=11)
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_svg(fig, 'classification_vs_regression.svg')


# =============================================================================
# 2. K-NN Intuition (with Real Iris)
# =============================================================================
def create_knn_intuition():
    """K-NN algorithm visualization using REAL Iris data."""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('white')

    # Load Iris
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target

    # Plot all points
    colors = [COLORS['blue'], COLORS['success'], COLORS['accent']]
    target_names = iris.target_names

    for i, (color, name) in enumerate(zip(colors, target_names)):
        ax.scatter(X[y == i, 0], X[y == i, 1], c=color, s=80,
                  label=name, alpha=0.5, edgecolors='white', linewidth=1)

    # Query point
    query_point = np.array([5.5, 3.0])
    ax.scatter(*query_point, s=300, marker='*', color=COLORS['warning'],
              edgecolors='black', linewidth=2, label='Query Point', zorder=10)

    # Draw circle showing k=5 nearest neighbors
    circle = Circle(query_point, 0.6, fill=False, edgecolor=COLORS['primary'],
                   linewidth=2.5, linestyle='--', alpha=0.8)
    ax.add_patch(circle)

    # Highlight k nearest neighbors (simulated)
    neighbors = np.array([[5.4, 3.0], [5.2, 2.7], [5.7, 3.0], [5.5, 2.6], [5.3, 3.2]])
    ax.scatter(neighbors[:, 0], neighbors[:, 1], s=150,
              facecolors='none', edgecolors=COLORS['primary'], linewidth=2.5,
              label='k=5 Nearest', zorder=9)

    ax.set_xlabel('Sepal Length (cm)', fontsize=12)
    ax.set_ylabel('Sepal Width (cm)', fontsize=12)
    ax.set_title('K-Nearest Neighbors (K-NN) Algorithm\n"Predict using similar examples from training data"',
                fontsize=14, fontweight='bold', color=COLORS['primary'])
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.annotate('', xy=(query_point[0] + 0.4, query_point[1] + 0.3),
               xytext=(query_point[0] + 0.8, query_point[1] + 0.8),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    ax.text(query_point[0] + 0.9, query_point[1] + 0.9,
           'Find k most\nsimilar points',
           fontsize=10, ha='left', va='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                    edgecolor=COLORS['primary'], linewidth=1))

    save_svg(fig, 'knn_intuition.svg')


# =============================================================================
# 3. Decision Boundary Visualization (with Real Iris)
# =============================================================================
def create_decision_boundary():
    """Show decision boundary using REAL Iris data."""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('white')

    # Load Iris (use only 2 classes for binary classification example)
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    iris = load_iris()
    # Use only setosa (0) and versicolor (1)
    mask = iris.target < 2
    X = iris.data[mask, :2]
    y = iris.target[mask]

    # Train simple model
    model = LogisticRegression(C=100)
    model.fit(X, y)

    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision regions
    ax.contourf(xx, yy, Z, alpha=0.3, colors=[COLORS['blue'], COLORS['success']])

    # Plot training points
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c=COLORS['blue'], s=100,
              label='Setosa', alpha=0.8, edgecolors='white', linewidth=1)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c=COLORS['success'], s=100,
              label='Versicolor', alpha=0.8, edgecolors='white', linewidth=1)

    # Draw decision boundary
    ax.contour(xx, yy, Z, levels=[0.5], colors=COLORS['accent'],
              linewidths=3, linestyles='--')

    ax.set_xlabel('Sepal Length (cm)', fontsize=12)
    ax.set_ylabel('Sepal Width (cm)', fontsize=12)
    ax.set_title('Decision Boundary\n"Model learns to separate classes"',
                fontsize=14, fontweight='bold', color=COLORS['primary'])
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    save_svg(fig, 'decision_boundary.svg')


# =============================================================================
# 4. Linear Regression Intuition
# =============================================================================
def create_linear_regression_intuition():
    """Linear regression with residuals visualization."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')

    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y_true = 2.5 * x + 10
    y = y_true + np.random.randn(20) * 3

    # Fit line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    y_pred = p(x)

    # Plot data and line
    ax.scatter(x, y, c=COLORS['blue'], s=100, alpha=0.8,
              edgecolors='white', linewidth=1.5, label='Data points', zorder=10)
    ax.plot(x, y_pred, color=COLORS['accent'], linewidth=3,
           label='Regression line', zorder=5)

    # Draw residuals
    for xi, yi, ypi in zip(x, y, y_pred):
        ax.plot([xi, xi], [yi, ypi], color=COLORS['accent'],
               linewidth=1.5, alpha=0.6, zorder=3)

    # Highlight a few residuals
    ax.annotate('', xy=(x[3], y_pred[3]), xytext=(x[3], y[3]),
              arrowprops=dict(arrowstyle='<->', color=COLORS['accent'], lw=2))
    ax.text(x[3], (y[3] + y_pred[3])/2, ' residual', fontsize=10,
           color=COLORS['accent'], ha='right')

    ax.set_xlabel('Input (x)', fontsize=12)
    ax.set_ylabel('Target (y)', fontsize=12)
    ax.set_title('Linear Regression\n"Minimize sum of squared residuals"',
                fontsize=14, fontweight='bold', color=COLORS['primary'])
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    save_svg(fig, 'linear_regression_intuition.svg')


# =============================================================================
# 5. Overfitting vs Underfitting (with Real Data)
# =============================================================================
def create_overfitting_visual():
    """Show overfitting with polynomial regression on noisy data."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')

    np.random.seed(42)
    x = np.linspace(0, 10, 30)
    y_true = np.sin(x) + 0.5
    y = y_true + np.random.randn(30) * 0.3
    x_smooth = np.linspace(0, 10, 200)

    titles = ['Underfitting\n(Degree 1)', 'Good Fit\n(Degree 3)', 'Overfitting\n(Degree 10)']
    degrees = [1, 3, 10]
    colors = [COLORS['warning'], COLORS['success'], COLORS['accent']]

    for ax, title, degree, color in zip(axes, titles, degrees, colors):
        ax.set_facecolor('white')
        ax.scatter(x, y, c=COLORS['blue'], s=60, alpha=0.7,
                  edgecolors='white', linewidth=1, zorder=5)

        # Fit polynomial
        coeffs = np.polyfit(x, y, degree)
        p = np.poly1d(coeffs)
        y_smooth = p(x_smooth)

        ax.plot(x_smooth, y_smooth, color=color, linewidth=2.5, label=f'Degree {degree}')
        ax.set_title(title, fontsize=12, fontweight='bold', color=COLORS['primary'])
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 10.5)

    plt.tight_layout()
    save_svg(fig, 'overfitting_visual.svg')


# =============================================================================
# 6. Loss Function Landscape
# =============================================================================
def create_loss_landscape():
    """Visualize loss function minimization."""
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
    ax.set_facecolor('white')

    # Create loss landscape (quadratic bowl)
    x = np.linspace(-5, 5, 100)
    loss = 0.5 * x**2 + 2*np.sin(x) + 5

    ax.plot(x, loss, color=COLORS['primary'], linewidth=3, label='Loss function')
    ax.fill_between(x, loss, alpha=0.1, color=COLORS['blue'])

    # Show gradient descent steps
    steps = [-4, -2.5, -1.2, -0.3, 0.2]
    step_colors = [COLORS['accent'], COLORS['warning'], COLORS['purple'],
                   COLORS['blue'], COLORS['success']]

    for i, (step, color) in enumerate(zip(steps, step_colors)):
        loss_val = 0.5 * step**2 + 2*np.sin(step) + 5
        ax.scatter([step], [loss_val], s=150, color=color, zorder=5,
                  edgecolors='black', linewidth=1.5)
        ax.text(step, loss_val + 0.8, f'Step {i+1}', fontsize=9,
               ha='center', fontweight='bold', color=color)

        if i < len(steps) - 1:
            next_step = steps[i + 1]
            next_loss = 0.5 * next_step**2 + 2*np.sin(next_step) + 5
            ax.annotate('', xy=(next_step, next_loss), xytext=(step, loss_val),
                      arrowprops=dict(arrowstyle='->', color=COLORS['gray'],
                                    lw=2, ls='--'))

    # Mark optimum
    opt_x = x[np.argmin(loss)]
    opt_loss = np.min(loss)
    ax.scatter([opt_x], [opt_loss], s=200, color=COLORS['success'],
              marker='*', zorder=6, edgecolors='black', linewidth=2,
              label='Optimum')

    ax.set_xlabel('Parameter (θ)', fontsize=12)
    ax.set_ylabel('Loss L(θ)', fontsize=12)
    ax.set_title('Gradient Descent: Follow the Slope Downhill\n"Minimize loss by updating parameters"',
                fontsize=14, fontweight='bold', color=COLORS['primary'])
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    save_svg(fig, 'loss_landscape.svg')


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Generating diagrams for L03: Supervised Learning...")
    print()

    print("Creating classification vs regression (with REAL Iris data)...")
    create_classification_vs_regression()

    print("Creating K-NN intuition (with REAL Iris data)...")
    create_knn_intuition()

    print("Creating decision boundary (with REAL Iris data)...")
    create_decision_boundary()

    print("Creating linear regression intuition...")
    create_linear_regression_intuition()

    print("Creating overfitting visualization...")
    create_overfitting_visual()

    print("Creating loss landscape...")
    create_loss_landscape()

    print()
    print("Done! All diagrams generated in diagrams/svg/")
    print()
    print("Generated diagrams:")
    print("  - classification_vs_regression.svg (REAL Iris data)")
    print("  - knn_intuition.svg (REAL Iris data)")
    print("  - decision_boundary.svg (REAL Iris data)")
    print("  - linear_regression_intuition.svg")
    print("  - overfitting_visual.svg")
    print("  - loss_landscape.svg")
