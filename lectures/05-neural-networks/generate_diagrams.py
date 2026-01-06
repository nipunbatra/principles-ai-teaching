#!/usr/bin/env python3
"""
Diagram generator for Lecture 05: Neural Networks
Visualizes perceptrons, activation functions, backpropagation, and deep architectures.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
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
# 1. Single Perceptron (The Neuron)
# =============================================================================
def create_perceptron():
    """Show a single perceptron with inputs, weights, and output."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor(COLORS['bg_light'])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)

    # Input labels
    inputs = ['x₁', 'x₂', 'x₃', 'bias']
    weights = ['w₁', 'w₂', 'w₃', 'b']
    y_positions = [6, 4.5, 3, 1.5]

    for i, (inp, w, y) in enumerate(zip(inputs, weights, y_positions)):
        # Input node
        circle = Circle((1, y), 0.4, facecolor=COLORS['blue'],
                      edgecolor=COLORS['primary'], linewidth=2, alpha=0.7)
        ax.add_patch(circle)
        ax.text(1, y, inp, fontsize=12, ha='center', va='center',
               fontweight='bold', color='white')

        # Weight label
        ax.text(3, y + 0.6, w, fontsize=12, ha='center',
               color=COLORS['accent'], fontweight='bold')

        # Arrow to sum
        arrow = FancyArrowPatch((1.5, y), (6, 4),
                              arrowstyle='->', mutation_scale=20,
                              color=COLORS['primary'], linewidth=1.5, alpha=0.5)
        ax.add_patch(arrow)

    # Sum circle
    circle = Circle((6, 4), 0.6, facecolor=COLORS['warning'],
                   edgecolor=COLORS['primary'], linewidth=2, alpha=0.8)
    ax.add_patch(circle)
    ax.text(6, 4, 'Σ', fontsize=20, ha='center', va='center',
           fontweight='bold', color=COLORS['primary'])

    # Activation function
    rect = FancyBboxPatch((7.5, 3.2), 1.5, 1.6, boxstyle="round,pad=0.1",
                         facecolor=COLORS['purple'], edgecolor=COLORS['primary'],
                         linewidth=2, alpha=0.6)
    ax.add_patch(rect)
    ax.text(8.25, 4, 'σ(z)', fontsize=14, ha='center', va='center',
           fontweight='bold', color='white')

    # Arrow to activation
    arrow = FancyArrowPatch((6.7, 4), (7.4, 4),
                          arrowstyle='->', mutation_scale=20,
                          color=COLORS['primary'], linewidth=2)
    ax.add_patch(arrow)

    # Output
    circle = Circle((11, 4), 0.5, facecolor=COLORS['success'],
                   edgecolor=COLORS['primary'], linewidth=2, alpha=0.8)
    ax.add_patch(circle)
    ax.text(11, 4, 'y', fontsize=14, ha='center', va='center',
           fontweight='bold', color='white')

    # Arrow to output
    arrow = FancyArrowPatch((9.1, 4), (10.4, 4),
                          arrowstyle='->', mutation_scale=20,
                          color=COLORS['primary'], linewidth=2)
    ax.add_patch(arrow)

    # Labels
    ax.text(6, 2.5, 'Sum weighted\ninputs', fontsize=10, ha='center',
           color=COLORS['text_light'])
    ax.text(8.25, 2.5, 'Apply\nactivation', fontsize=10, ha='center',
           color=COLORS['text_light'])

    ax.set_title('A Single Perceptron\n"The basic building block of neural networks"',
                fontsize=15, fontweight='bold', color=COLORS['primary'], pad=20)

    save_svg(fig, 'perceptron.svg')


# =============================================================================
# 2. Activation Functions Comparison
# =============================================================================
def create_activation_functions():
    """Compare different activation functions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')

    x = np.linspace(-5, 5, 200)

    # Sigmoid
    ax = axes[0, 0]
    ax.set_facecolor('white')
    y = 1 / (1 + np.exp(-x))
    ax.plot(x, y, color=COLORS['blue'], linewidth=3)
    ax.axhline(y=0.5, color=COLORS['gray'], linestyle='--', alpha=0.5)
    ax.axvline(x=0, color=COLORS['gray'], linestyle='--', alpha=0.5)
    ax.set_title('Sigmoid σ(x) = 1/(1+e⁻ˣ)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Output', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.text(-4.5, 0.9, '• Outputs 0 to 1\n• Probabilities', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Tanh
    ax = axes[0, 1]
    ax.set_facecolor('white')
    y = np.tanh(x)
    ax.plot(x, y, color=COLORS['success'], linewidth=3)
    ax.axhline(y=0, color=COLORS['gray'], linestyle='--', alpha=0.5)
    ax.axvline(x=0, color=COLORS['gray'], linestyle='--', alpha=0.5)
    ax.set_title('Tanh tanh(x)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(-4.5, 0.7, '• Outputs -1 to 1\n• Zero-centered', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ReLU
    ax = axes[1, 0]
    ax.set_facecolor('white')
    y = np.maximum(0, x)
    ax.plot(x, y, color=COLORS['accent'], linewidth=3)
    ax.axhline(y=0, color=COLORS['gray'], linestyle='--', alpha=0.5)
    ax.axvline(x=0, color=COLORS['gray'], linestyle='--', alpha=0.5)
    ax.set_title('ReLU max(0, x)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Input', fontsize=11)
    ax.set_ylabel('Output', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.text(-4.5, 4, '• Default choice\n• No vanishing', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Leaky ReLU
    ax = axes[1, 1]
    ax.set_facecolor('white')
    y = np.where(x > 0, x, 0.01 * x)
    ax.plot(x, y, color=COLORS['purple'], linewidth=3)
    ax.axhline(y=0, color=COLORS['gray'], linestyle='--', alpha=0.5)
    ax.axvline(x=0, color=COLORS['gray'], linestyle='--', alpha=0.5)
    ax.set_title('Leaky ReLU max(0.01x, x)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Input', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.text(-4.5, 4, '• Fixes dead neurons\n• Small gradient for x<0', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Activation Functions: Introduce Non-Linearity',
                fontsize=15, fontweight='bold', color=COLORS['primary'])
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_svg(fig, 'activation_functions.svg')


# =============================================================================
# 3. Multi-Layer Perceptron (Deep Network)
# =============================================================================
def create_mlp_architecture():
    """Show a multi-layer perceptron architecture."""
    fig, ax = plt.subplots(figsize=(14, 9), facecolor='white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Layer configurations
    layers = [
        ('Input\nLayer', 2, 4, 1.5, COLORS['blue']),      # 4 inputs (2 features shown)
        ('Hidden\nLayer 1', 4, 5, 4, COLORS['purple']),    # 5 neurons
        ('Hidden\nLayer 2', 7.5, 4, 7, COLORS['warning']),  # 4 neurons
        ('Output\nLayer', 10.5, 3, 9.5, COLORS['success']),  # 3 classes
    ]

    neurons = {}  # Store neuron positions

    for layer_idx, (name, x, n_neurons, y_start, color) in enumerate(layers):
        # Layer label
        ax.text(x, y_start + n_neurons * 0.8 + 0.5, name,
               fontsize=12, ha='center', fontweight='bold', color=color)

        # Draw neurons
        for i in range(n_neurons):
            y = y_start + i * 0.8
            circle = Circle((x, y), 0.3, facecolor=color,
                          edgecolor=COLORS['primary'], linewidth=2, alpha=0.7)
            ax.add_patch(circle)
            neurons[(layer_idx, i)] = (x, y)

    # Draw connections
    # Input to Hidden 1
    for in_i in range(4):
        for out_i in range(5):
            x1, y1 = neurons[(0, in_i)]
            x2, y2 = neurons[(1, out_i)]
            alpha = 0.3 if np.random.random() > 0.5 else 0.1
            ax.plot([x1, x2], [y1, y2], color=COLORS['primary'],
                   linewidth=0.5, alpha=alpha, zorder=0)

    # Hidden 1 to Hidden 2
    for in_i in range(5):
        for out_i in range(4):
            x1, y1 = neurons[(1, in_i)]
            x2, y2 = neurons[(2, out_i)]
            alpha = 0.3 if np.random.random() > 0.5 else 0.1
            ax.plot([x1, x2], [y1, y2], color=COLORS['primary'],
                   linewidth=0.5, alpha=alpha, zorder=0)

    # Hidden 2 to Output
    for in_i in range(4):
        for out_i in range(3):
            x1, y1 = neurons[(2, in_i)]
            x2, y2 = neurons[(3, out_i)]
            alpha = 0.3 if np.random.random() > 0.5 else 0.1
            ax.plot([x1, x2], [y1, y2], color=COLORS['primary'],
                   linewidth=0.5, alpha=alpha, zorder=0)

    # Labels
    ax.text(1.5, 0.8, 'Features', fontsize=10, ha='center', color=COLORS['text_light'])
    ax.text(10.5, 0.8, 'Classes', fontsize=10, ha='center', color=COLORS['text_light'])

    ax.set_title('Multi-Layer Perceptron (Deep Network)\n"Stack layers to learn complex patterns"',
                fontsize=15, fontweight='bold', color=COLORS['primary'], pad=20)

    save_svg(fig, 'mlp_architecture.svg')


# =============================================================================
# 4. Backpropagation Visualization
# =============================================================================
def create_backpropagation():
    """Visualize backpropagation - gradient flowing backward."""
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
    ax.set_facecolor(COLORS['bg_light'])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)

    # Simple 3-layer network
    layers = [
        ('Input', 2, 3, 4, COLORS['blue']),
        ('Hidden', 6, 4, 3, COLORS['purple']),
        ('Output', 10, 2, 4, COLORS['success']),
    ]

    neurons = {}
    for layer_idx, (name, x, n_neurons, y_start, color) in enumerate(layers):
        ax.text(x, y_start + n_neurons * 0.8 + 0.5, name,
               fontsize=11, ha='center', fontweight='bold', color=color)
        for i in range(n_neurons):
            y = y_start + i * 0.8
            circle = Circle((x, y), 0.25, facecolor=color,
                          edgecolor=COLORS['primary'], linewidth=2, alpha=0.7)
            ax.add_patch(circle)
            neurons[(layer_idx, i)] = (x, y)

    # Forward pass (solid arrows, blue)
    for in_i in range(3):
        for out_i in range(4):
            if np.random.random() > 0.3:
                x1, y1 = neurons[(0, in_i)]
                x2, y2 = neurons[(1, out_i)]
                arrow = FancyArrowPatch((x1 + 0.3, y1), (x2 - 0.3, y2),
                                      arrowstyle='->', mutation_scale=12,
                                      color=COLORS['blue'], linewidth=1.5, alpha=0.6)
                ax.add_patch(arrow)

    for in_i in range(4):
        for out_i in range(2):
            if np.random.random() > 0.3:
                x1, y1 = neurons[(1, in_i)]
                x2, y2 = neurons[(2, out_i)]
                arrow = FancyArrowPatch((x1 + 0.3, y1), (x2 - 0.3, y2),
                                      arrowstyle='->', mutation_scale=12,
                                      color=COLORS['blue'], linewidth=1.5, alpha=0.6)
                ax.add_patch(arrow)

    # Backward pass (dashed arrows, red/overlay)
    for in_i in range(4):
        for out_i in range(2):
            if np.random.random() > 0.3:
                x1, y1 = neurons[(2, out_i)]
                x2, y2 = neurons[(1, in_i)]
                # Offset slightly to show both directions
                arrow = FancyArrowPatch((x2 + 0.2, y2 + 0.2), (x1 - 0.2, y1 - 0.2),
                                      arrowstyle='->', mutation_scale=12,
                                      color=COLORS['accent'], linewidth=1.5,
                                      linestyle='--', alpha=0.8)
                ax.add_patch(arrow)

    for in_i in range(3):
        for out_i in range(4):
            if np.random.random() > 0.3:
                x1, y1 = neurons[(1, out_i)]
                x2, y2 = neurons[(0, in_i)]
                arrow = FancyArrowPatch((x2 + 0.2, y2 + 0.2), (x1 - 0.2, y1 - 0.2),
                                      arrowstyle='->', mutation_scale=12,
                                      color=COLORS['accent'], linewidth=1.5,
                                      linestyle='--', alpha=0.8)
                ax.add_patch(arrow)

    # Legend
    ax.plot([], [], color=COLORS['blue'], linewidth=2, label='Forward (prediction)')
    ax.plot([], [], color=COLORS['accent'], linewidth=2, linestyle='--',
           label='Backward (gradients)')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

    ax.set_title('Backpropagation\n"Gradients flow backward to update weights"',
                fontsize=15, fontweight='bold', color=COLORS['primary'], pad=20)

    save_svg(fig, 'backpropagation.svg')


# =============================================================================
# 5. Loss Landscape (Visual Optimization)
# =============================================================================
def create_loss_landscape_3d():
    """Show 3D loss landscape with gradient descent path."""
    fig = plt.figure(figsize=(14, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    # Create loss landscape
    w1 = np.linspace(-3, 3, 50)
    w2 = np.linspace(-3, 3, 50)
    W1, W2 = np.meshgrid(w1, w2)
    Loss = W1**2 + W2**2 + 0.5 * np.sin(3*W1) * np.sin(3*W2)

    # Plot surface
    surf = ax.plot_surface(W1, W2, Loss, cmap='viridis', alpha=0.8,
                          edgecolor='none', rstride=2, cstride=2)

    # Gradient descent path
    path_w1 = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0])
    path_w2 = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0])
    path_loss = [path_w1[i]**2 + path_w2[i]**2 + 0.5 * np.sin(3*path_w1[i]) * np.sin(3*path_w2[i])
                 for i in range(len(path_w1))]

    ax.plot(path_w1, path_w2, path_loss, 'o-', color=COLORS['accent'],
           linewidth=3, markersize=10, label='Gradient Descent')

    # Mark start and end
    ax.scatter([path_w1[0]], [path_w2[0]], [path_loss[0]], s=200,
              color=COLORS['warning'], marker='*', edgecolors='black',
              linewidth=1.5, label='Start', zorder=10)
    ax.scatter([path_w1[-1]], [path_w2[-1]], [path_loss[-1]], s=200,
              color=COLORS['success'], marker='*', edgecolors='black',
              linewidth=1.5, label='Optimum', zorder=10)

    ax.set_xlabel('Weight 1', fontsize=11)
    ax.set_ylabel('Weight 2', fontsize=11)
    ax.set_zlabel('Loss', fontsize=11)
    ax.set_title('Loss Landscape: Gradient Descent Path\n"Follow the slope downhill to minimize loss"',
                fontsize=14, fontweight='bold', color=COLORS['primary'], pad=20)
    ax.legend(loc='upper right', fontsize=10)

    save_svg(fig, 'loss_landscape_3d.svg')


# =============================================================================
# 6. Universal Approximation Visualization
# =============================================================================
def create_universal_approximation():
    """Show how neural networks can approximate any function."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')

    # True function
    x = np.linspace(0, 4, 100)
    y_true = np.sin(x * 3) + 0.5 * x

    # Plot true function on all
    for ax in axes:
        ax.set_facecolor('white')
        ax.plot(x, y_true, '--', color=COLORS['primary'], linewidth=2,
               label='True function', alpha=0.7)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('f(x)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 4)

    # Number of neurons
    configs = [(2, '1 neuron', COLORS['accent']),
               (5, '5 neurons', COLORS['warning']),
               (20, '20 neurons', COLORS['success'])]

    for ax, (n_neurons, title, color) in zip(axes, configs):
        # Simulate network output (sum of sigmoids)
        np.random.seed(42)
        y_pred = np.zeros_like(x)
        for _ in range(n_neurons):
            w = np.random.uniform(-3, 3)
            b = np.random.uniform(-3, 3)
            y_pred += np.random.uniform(-1, 1) * (1 / (1 + np.exp(-w * x + b)))

        # Normalize
        y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
        y_pred = y_pred * (y_true.max() - y_true.min()) + y_true.min()

        ax.plot(x, y_pred, '-', color=color, linewidth=2.5, label=f'NN ({n_neurons} neurons)')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)

    plt.suptitle('Universal Approximation Theorem\n"More neurons → better function approximation"',
                fontsize=14, fontweight='bold', color=COLORS['primary'])
    plt.tight_layout()

    save_svg(fig, 'universal_approximation.svg')


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Generating diagrams for L05: Neural Networks...")
    print()

    print("Creating perceptron diagram...")
    create_perceptron()

    print("Creating activation functions comparison...")
    create_activation_functions()

    print("Creating MLP architecture...")
    create_mlp_architecture()

    print("Creating backpropagation visualization...")
    create_backpropagation()

    print("Creating 3D loss landscape...")
    create_loss_landscape_3d()

    print("Creating universal approximation visualization...")
    create_universal_approximation()

    print()
    print("Done! All diagrams generated in diagrams/svg/")
    print()
    print("Generated diagrams:")
    print("  - perceptron.svg")
    print("  - activation_functions.svg")
    print("  - mlp_architecture.svg")
    print("  - backpropagation.svg")
    print("  - loss_landscape_3d.svg")
    print("  - universal_approximation.svg")
