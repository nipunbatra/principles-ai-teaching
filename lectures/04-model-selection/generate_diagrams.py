#!/usr/bin/env python3
"""
Diagram generator for Lecture 04: Model Selection
Uses REAL datasets for validation and model comparison visualizations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Import shared style
try:
    from themes.diagram_style import COLORS, setup_figure, save_svg
except ImportError:
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
# 1. Train/Validation/Test Split (Visual with Real Data)
# =============================================================================
def create_train_val_test_split():
    """Visualize train/val/test split with REAL Iris data."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X = iris.data[:, :2]  # Use sepal features
    y = iris.target

    # Simulate split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')

    titles = ['Training Set (60%)', 'Validation Set (20%)',
              'Test Set (20%)', 'Full Dataset']
    datasets = [(X_train, y_train), (X_val, y_val), (X_test, y_test), (X, y)]
    colors_list = [COLORS['blue'], COLORS['warning'], COLORS['accent'], COLORS['success']]

    for ax, (X_d, y_d), title, color in zip(axes.flat, datasets, titles, colors_list):
        ax.set_facecolor('white')

        for i, c in enumerate(['blue', 'green', 'red']):
            mask = y_d == i
            ax.scatter(X_d[mask, 0], X_d[mask, 1], c=color, s=60,
                      alpha=0.6, edgecolors='white', linewidth=0.5)

        ax.set_title(title, fontsize=13, fontweight='bold', color=COLORS['primary'])
        ax.set_xlabel('Sepal Length', fontsize=10)
        ax.set_ylabel('Sepal Width', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add sample count
        ax.text(0.02, 0.98, f'n={len(X_d)}', transform=ax.transAxes,
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                        edgecolor=color, linewidth=2),
               va='top')

    plt.suptitle('Train/Validation/Test Split\n"Never use test data for training or tuning!"',
                fontsize=15, fontweight='bold', color=COLORS['primary'], y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_svg(fig, 'train_val_test_split.svg')


# =============================================================================
# 2. Cross-Validation Visualization
# =============================================================================
def create_cross_validation():
    """Show k-fold cross-validation process."""
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
    ax.set_facecolor(COLORS['bg_light'])

    k = 5  # 5-fold CV
    n_samples = 20

    # Create fold visualization
    for fold in range(k):
        y_pos = k - fold - 1

        for i in range(n_samples):
            # Determine if this sample is in validation set for this fold
            val_start = fold * (n_samples // k)
            val_end = (fold + 1) * (n_samples // k)

            if val_start <= i < val_end:
                color = COLORS['accent']  # Validation
                alpha = 0.8
            else:
                color = COLORS['blue']  # Training
                alpha = 0.4

            rect = FancyBboxPatch((i * 0.8, y_pos - 0.35), 0.7, 0.7,
                                 boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor='white',
                                 linewidth=1, alpha=alpha)
            ax.add_patch(rect)

        ax.text(-1.5, y_pos, f'Fold {fold + 1}', fontsize=11, ha='right',
               va='center', fontweight='bold', color=COLORS['primary'])

    # Labels
    ax.text(n_samples * 0.8 / 2, k + 0.5, 'Data Samples', fontsize=13,
           ha='center', fontweight='bold', color=COLORS['primary'])
    ax.text(-3, k / 2, 'Folds', fontsize=13, ha='center', va='center',
           fontweight='bold', color=COLORS['primary'], rotation=90)

    # Legend
    ax.scatter([], [], c=COLORS['blue'], s=200, alpha=0.5, edgecolors='white',
              label='Training', linewidth=1.5)
    ax.scatter([], [], c=COLORS['accent'], s=200, alpha=0.8, edgecolors='white',
              label='Validation', linewidth=1.5)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)

    ax.set_xlim(-4, n_samples * 0.8 + 1)
    ax.set_ylim(-0.5, k + 0.5)
    ax.axis('off')
    ax.set_title('5-Fold Cross-Validation\n"Each sample used for validation exactly once"',
                fontsize=15, fontweight='bold', color=COLORS['primary'], pad=20)

    save_svg(fig, 'cross_validation.svg')


# =============================================================================
# 3. Overfitting Curve (Learning Curve)
# =============================================================================
def create_learning_curve():
    """Show learning curve with overfitting (using synthetic but realistic data)."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')

    # Training sizes
    train_sizes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    n_samples = 1000

    # Simulate learning curves
    train_score = 1 - np.exp(-5 * train_sizes) + 0.05 * train_sizes  # Starts high, improves
    val_score = 1 - np.exp(-3 * train_sizes) - 0.1 * np.sin(train_sizes * 10)  # Improves then plateaus

    ax.plot(train_sizes * n_samples, train_score, 'o-', color=COLORS['blue'],
           linewidth=2.5, markersize=8, label='Training Score')
    ax.plot(train_sizes * n_samples, val_score, 's-', color=COLORS['accent'],
           linewidth=2.5, markersize=8, label='Validation Score')

    # Highlight gap
    ax.fill_between(train_sizes * n_samples, train_score, val_score,
                   alpha=0.2, color=COLORS['warning'], label='Overfitting Gap')

    ax.set_xlabel('Training Set Size', fontsize=13)
    ax.set_ylabel('Model Score (R² / Accuracy)', fontsize=13)
    ax.set_title('Learning Curve\n"More data → better generalization"',
                fontsize=15, fontweight='bold', color=COLORS['primary'])
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)

    # Annotation
    ax.annotate('', xy=(800, 0.92), xytext=(800, 0.98),
              arrowprops=dict(arrowstyle='<->', color=COLORS['warning'], lw=2))
    ax.text(820, 0.95, 'Overfitting\n  gap', fontsize=10, color=COLORS['warning'],
           fontweight='bold')

    save_svg(fig, 'learning_curve.svg')


# =============================================================================
# 4. Bias-Variance Tradeoff
# =============================================================================
def create_bias_variance_tradeoff():
    """Show the bias-variance tradeoff."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')

    # Model complexity
    complexity = np.linspace(0, 10, 100)

    # Simulate bias (decreases with complexity)
    bias = 0.5 * np.exp(-0.5 * complexity) + 0.05

    # Simulate variance (increases with complexity)
    variance = 0.02 * complexity**1.5 + 0.05

    # Total error (bias + variance + irreducible)
    total_error = bias + variance + 0.1

    ax.plot(complexity, bias, '--', color=COLORS['blue'], linewidth=2.5,
           label='Bias² (Underfitting)', alpha=0.8)
    ax.plot(complexity, variance, '--', color=COLORS['accent'], linewidth=2.5,
           label='Variance (Overfitting)', alpha=0.8)
    ax.plot(complexity, total_error, '-', color=COLORS['primary'], linewidth=3.5,
           label='Total Error', alpha=0.9)

    # Mark optimal point
    optimal_idx = np.argmin(total_error)
    optimal_complexity = complexity[optimal_idx]
    ax.scatter([optimal_complexity], [total_error[optimal_idx]], s=200,
              color=COLORS['success'], marker='*', zorder=10, edgecolors='black',
              linewidth=1.5, label='Optimal Complexity')

    ax.set_xlabel('Model Complexity', fontsize=13)
    ax.set_ylabel('Error', fontsize=13)
    ax.set_title('Bias-Variance Tradeoff\n"Find the sweet spot between too simple and too complex"',
                fontsize=15, fontweight='bold', color=COLORS['primary'])
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Annotations
    ax.text(1, 0.5, 'High Bias\n(Underfit)', fontsize=10, color=COLORS['blue'],
           ha='center', style='italic')
    ax.text(9, 0.6, 'High Variance\n(Overfit)', fontsize=10, color=COLORS['accent'],
           ha='center', style='italic')

    save_svg(fig, 'bias_variance_tradeoff.svg')


# =============================================================================
# 5. Model Comparison (with Real Data)
# =============================================================================
def create_model_comparison():
    """Compare different models using REAL Iris data."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    iris = load_iris()
    X, y = iris.data, iris.target

    # Train different models
    models = {
        'Logistic\nRegression': LogisticRegression(max_iter=200, random_state=42),
        'Decision\nTree': DecisionTreeClassifier(random_state=42),
        'Random\nForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'K-NN\n(k=5)': KNeighborsClassifier(n_neighbors=5),
    }

    # Get cross-validation scores
    cv_scores = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5)
        cv_scores[name] = scores

    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')

    # Plot box plots
    positions = np.arange(len(models))
    bp = ax.boxplot([cv_scores[name] for name in models.keys()],
                    positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True)

    # Color boxes
    colors = [COLORS['blue'], COLORS['success'], COLORS['purple'], COLORS['accent']]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add individual points
    for i, (name, color) in enumerate(zip(models.keys(), colors)):
        x = np.random.normal(i + 1, 0.04, size=5)
        ax.scatter(x, cv_scores[name], c=color, s=50, alpha=0.8, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(models.keys(), fontsize=11)
    ax.set_ylabel('Accuracy Score (5-fold CV)', fontsize=13)
    ax.set_title('Model Comparison on Iris Dataset\n"All models perform well on this simple task"',
                fontsize=15, fontweight='bold', color=COLORS['primary'])
    ax.set_ylim(0.85, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color=COLORS['success'], linestyle='--', alpha=0.5)

    save_svg(fig, 'model_comparison.svg')


# =============================================================================
# 6. Hyperparameter Tuning Visualization
# =============================================================================
def create_hyperparameter_tuning():
    """Show hyperparameter tuning (K in K-NN using REAL Iris data)."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    iris = load_iris()
    X, y = iris.data, iris.target

    # Test different K values
    k_values = range(1, 31)
    train_scores = []
    val_scores = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        # Simple train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        knn.fit(X_train, y_train)
        train_scores.append(knn.score(X_train, y_train))
        val_scores.append(knn.score(X_val, y_val))

    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')

    ax.plot(k_values, train_scores, 'o-', color=COLORS['blue'], linewidth=2,
           markersize=6, label='Training Accuracy', alpha=0.8)
    ax.plot(k_values, val_scores, 's-', color=COLORS['accent'], linewidth=2.5,
           markersize=7, label='Validation Accuracy', alpha=0.9)

    # Mark best K
    best_k = np.argmax(val_scores) + 1
    best_score = max(val_scores)
    ax.scatter([best_k], [best_score], s=300, color=COLORS['success'],
              marker='*', zorder=10, edgecolors='black', linewidth=1.5,
              label=f'Best K={best_k}')

    ax.axvline(x=best_k, color=COLORS['success'], linestyle='--', alpha=0.5)

    ax.set_xlabel('K (Number of Neighbors)', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title(f'Hyperparameter Tuning: K in K-NN (Iris Dataset)\n"Best K={best_k} gives validation accuracy of {best_score:.1%}"',
                fontsize=15, fontweight='bold', color=COLORS['primary'])
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 31)

    save_svg(fig, 'hyperparameter_tuning.svg')


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Generating diagrams for L04: Model Selection...")
    print()

    print("Creating train/validation/test split (with REAL Iris data)...")
    create_train_val_test_split()

    print("Creating cross-validation visualization...")
    create_cross_validation()

    print("Creating learning curve...")
    create_learning_curve()

    print("Creating bias-variance tradeoff...")
    create_bias_variance_tradeoff()

    print("Creating model comparison (with REAL Iris data)...")
    create_model_comparison()

    print("Creating hyperparameter tuning (with REAL Iris data)...")
    create_hyperparameter_tuning()

    print()
    print("Done! All diagrams generated in diagrams/svg/")
    print()
    print("Generated diagrams:")
    print("  - train_val_test_split.svg (REAL Iris data)")
    print("  - cross_validation.svg")
    print("  - learning_curve.svg")
    print("  - bias_variance_tradeoff.svg")
    print("  - model_comparison.svg (REAL Iris data)")
    print("  - hyperparameter_tuning.svg (REAL Iris data)")
