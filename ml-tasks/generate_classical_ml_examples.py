#!/usr/bin/env python3
"""
Generate classical ML algorithm visualizations for lecture slides.
Shows what each algorithm actually DOES with real examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "examples"
OUTPUT_DIR.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11


def generate_decision_tree_example():
    """Generate decision tree visualization with sklearn."""
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    # Load data
    iris = load_iris()
    X, y = iris.data[:, :2], iris.target  # Use only first 2 features for visualization

    # Train tree
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)

    # Create figure with tree and decision boundary
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Decision tree structure
    plot_tree(clf,
              feature_names=['Sepal Length', 'Sepal Width'],
              class_names=iris.target_names,
              filled=True,
              rounded=True,
              ax=axes[0],
              fontsize=10)
    axes[0].set_title("Decision Tree Structure", fontsize=14, fontweight='bold')

    # Right: Decision boundary
    ax = axes[1]
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black', s=50)
    ax.set_xlabel('Sepal Length (cm)', fontsize=12)
    ax.set_ylabel('Sepal Width (cm)', fontsize=12)
    ax.set_title("Decision Boundary on Iris Data", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "decision_tree_example.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'decision_tree_example.png'}")


def generate_linear_regression_example():
    """Generate linear regression visualization."""
    from sklearn.linear_model import LinearRegression

    # Generate synthetic house data
    np.random.seed(42)
    sqft = np.random.uniform(500, 3000, 50)
    price = 50000 + 150 * sqft + np.random.normal(0, 30000, 50)

    # Fit model
    X = sqft.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(X, price)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Data + regression line
    ax = axes[0]
    ax.scatter(sqft, price / 1000, alpha=0.7, s=50, c='#2196F3', edgecolors='black')
    x_line = np.linspace(400, 3100, 100)
    y_line = (reg.intercept_ + reg.coef_[0] * x_line) / 1000
    ax.plot(x_line, y_line, 'r-', linewidth=3, label='Fitted line')
    ax.set_xlabel('Square Feet', fontsize=12)
    ax.set_ylabel('Price ($K)', fontsize=12)
    ax.set_title('House Price Prediction', fontsize=14, fontweight='bold')
    ax.legend()

    # Right: The formula
    ax = axes[1]
    ax.axis('off')
    ax.text(0.5, 0.8, "Linear Regression Formula:", ha='center', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.6, r"$\hat{y} = w_0 + w_1 \cdot x$", ha='center', fontsize=24,
            fontfamily='serif')
    ax.text(0.5, 0.4, f"Fitted: price = ${reg.intercept_:,.0f} + ${reg.coef_[0]:.0f} * sqft",
            ha='center', fontsize=14)
    ax.text(0.5, 0.2, "Each extra sqft adds ~$150 to price!", ha='center', fontsize=12,
            style='italic', color='#666')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "linear_regression_example.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'linear_regression_example.png'}")


def generate_logistic_regression_example():
    """Generate logistic regression decision boundary."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    # Generate 2D classification data
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                                n_redundant=0, n_clusters_per_class=1,
                                random_state=42)

    # Fit model
    clf = LogisticRegression()
    clf.fit(X, y)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Decision boundary
    ax = axes[0]
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Probability contours
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.7)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='#F44336', edgecolors='black', s=50, label='Class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='#2196F3', edgecolors='black', s=50, label='Class 1')
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title('Logistic Regression Decision Boundary', fontsize=14, fontweight='bold')
    ax.legend()
    plt.colorbar(contour, ax=ax, label='P(Class 1)')

    # Right: Sigmoid function
    ax = axes[1]
    z = np.linspace(-6, 6, 100)
    sigmoid = 1 / (1 + np.exp(-z))
    ax.plot(z, sigmoid, 'b-', linewidth=3)
    ax.axhline(0.5, color='gray', linestyle='--', label='Decision boundary')
    ax.axvline(0, color='gray', linestyle='--')
    ax.fill_between(z, 0, sigmoid, where=z > 0, alpha=0.3, color='#2196F3', label='Predict Class 1')
    ax.fill_between(z, 0, sigmoid, where=z < 0, alpha=0.3, color='#F44336', label='Predict Class 0')
    ax.set_xlabel('z = w*x + b', fontsize=12)
    ax.set_ylabel('P(y=1)', fontsize=12)
    ax.set_title('Sigmoid Function', fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "logistic_regression_example.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'logistic_regression_example.png'}")


def generate_kmeans_example():
    """Generate K-Means clustering before/after visualization."""
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    # Generate clustered data
    X, y_true = make_blobs(n_samples=200, centers=4, cluster_std=0.8, random_state=42)

    # Fit K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Before clustering (unlabeled)
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c='gray', s=50, alpha=0.6, edgecolors='black')
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title('Before: Unlabeled Data', fontsize=14, fontweight='bold')
    ax.text(0.5, 0.95, "We see patterns, but no labels!", transform=ax.transAxes,
            ha='center', fontsize=11, style='italic')

    # Right: After clustering
    ax = axes[1]
    colors = ['#F44336', '#2196F3', '#4CAF50', '#FF9800']
    for i in range(4):
        mask = y_pred == i
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=50, edgecolors='black',
                   label=f'Cluster {i + 1}')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c='black', marker='X', s=200, edgecolors='white', linewidths=2,
               label='Centroids')
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title('After: K-Means Clustering (k=4)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kmeans_example.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'kmeans_example.png'}")


def generate_neural_network_diagram():
    """Generate simple neural network visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Network structure
    layers = [3, 4, 4, 2]  # Input, Hidden1, Hidden2, Output
    layer_names = ['Input\n(features)', 'Hidden 1', 'Hidden 2', 'Output\n(prediction)']
    colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#E8F5E9']

    # Draw neurons and connections
    for i, (n_neurons, name, color) in enumerate(zip(layers, layer_names, colors)):
        x = i * 2
        y_positions = np.linspace(0, 4, n_neurons)
        y_positions = y_positions - y_positions.mean() + 2

        # Draw neurons
        for y in y_positions:
            circle = plt.Circle((x, y), 0.25, color=color, ec='black', linewidth=2, zorder=3)
            ax.add_patch(circle)

        # Draw connections to next layer
        if i < len(layers) - 1:
            next_y = np.linspace(0, 4, layers[i + 1])
            next_y = next_y - next_y.mean() + 2
            for y1 in y_positions:
                for y2 in next_y:
                    ax.plot([x + 0.25, x + 2 - 0.25], [y1, y2], 'gray', alpha=0.3, linewidth=0.5, zorder=1)

        # Label
        ax.text(x, -0.8, name, ha='center', fontsize=11, fontweight='bold')

    # Add title and annotations
    ax.set_title('Neural Network: Universal Function Approximator', fontsize=16, fontweight='bold')
    ax.text(3, 4.5, "Each connection has a learnable weight", ha='center', fontsize=10, style='italic')
    ax.text(3, -1.5, "Forward: x -> h1 -> h2 -> y_hat    |    Backward: Update weights using gradient descent",
            ha='center', fontsize=10)

    ax.set_xlim(-1, 7)
    ax.set_ylim(-2, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "neural_network_diagram.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'neural_network_diagram.png'}")


def generate_gradient_descent():
    """Generate gradient descent visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: 1D loss curve
    ax = axes[0]
    x = np.linspace(-3, 3, 100)
    y = x ** 2 + 0.5  # Simple parabola

    ax.plot(x, y, 'b-', linewidth=3, label='Loss function')

    # Gradient descent steps
    x_gd = [2.5]
    lr = 0.3
    for _ in range(5):
        grad = 2 * x_gd[-1]  # Derivative of x^2
        x_new = x_gd[-1] - lr * grad
        x_gd.append(x_new)

    y_gd = [xi ** 2 + 0.5 for xi in x_gd]
    ax.scatter(x_gd, y_gd, c='red', s=100, zorder=5, edgecolors='black')
    for i in range(len(x_gd) - 1):
        ax.annotate('', xy=(x_gd[i + 1], y_gd[i + 1]), xytext=(x_gd[i], y_gd[i]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_xlabel('Weight (w)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Gradient Descent: Finding the Minimum', fontsize=14, fontweight='bold')
    ax.axhline(0.5, color='green', linestyle='--', alpha=0.5)
    ax.text(0, 0.8, 'Optimal!', ha='center', fontsize=10, color='green')

    # Right: Update formula
    ax = axes[1]
    ax.axis('off')
    ax.text(0.5, 0.8, "Gradient Descent Update Rule:", ha='center', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.55, r"$w_{new} = w_{old} - \eta \cdot \nabla L$", ha='center', fontsize=24,
            fontfamily='serif')
    ax.text(0.5, 0.35, r"$\eta$ = learning rate (step size)", ha='center', fontsize=14)
    ax.text(0.5, 0.2, r"$\nabla L$ = gradient (direction of steepest ascent)", ha='center', fontsize=14)
    ax.text(0.5, 0.05, "We go OPPOSITE to gradient to minimize loss!", ha='center', fontsize=12,
            style='italic', color='#D62828')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gradient_descent.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'gradient_descent.png'}")


def generate_softmax_example():
    """Generate softmax visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # Raw logits
    logits = np.array([2.0, 1.0, 0.1])
    classes = ['Cat', 'Dog', 'Bird']

    # Softmax
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()

    # Create bar chart
    x = np.arange(3)
    width = 0.35

    bars1 = ax.bar(x - width / 2, logits, width, label='Raw Scores (logits)', color='#BBDEFB', edgecolor='black')
    bars2 = ax.bar(x + width / 2, probs, width, label='Softmax Probabilities', color='#C8E6C9', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Softmax: Converting Scores to Probabilities', fontsize=14, fontweight='bold')
    ax.legend()

    # Add value labels
    for bar, val in zip(bars1, logits):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f'{val:.1f}',
                ha='center', fontsize=10)
    for bar, val in zip(bars2, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{val:.1%}',
                ha='center', fontsize=10, fontweight='bold')

    # Add formula
    ax.text(2.5, 1.8, r"$P(class_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$", fontsize=14, fontfamily='serif')
    ax.text(2.5, 1.5, "Sum of probabilities = 1.0", fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "softmax_example.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'softmax_example.png'}")


def main():
    print("=" * 60)
    print("Generating Classical ML Examples")
    print("=" * 60)

    generate_decision_tree_example()
    generate_linear_regression_example()
    generate_logistic_regression_example()
    generate_kmeans_example()
    generate_neural_network_diagram()
    generate_gradient_descent()
    generate_softmax_example()

    print()
    print("=" * 60)
    print(f"Done! Examples saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
