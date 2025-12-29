"""
Consolidated diagram generator for ML Tasks lecture.
Uses matplotlib for all diagrams - simpler and more maintainable.

Diagrams that need visual representation:
1. bbox_regression - Shows image with bounding box coordinates
2. rl_loop - Cyclic agent-environment interaction
3. nn_binary_classification - Neural network architecture
4. nn_multiclass - Neural network architecture
5. nn_regression - Neural network architecture
6. nn_detection - Multi-output neural network
7. ml_decision_flowchart - Decision tree for choosing ML task

Other diagrams replaced with markdown tables in the slides.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os

# Create output directories
os.makedirs('diagrams/svg', exist_ok=True)

# Color scheme matching iitgn-modern theme
COLORS = {
    'primary': '#1e3a5f',
    'primary_light': '#2e5a8f',
    'accent': '#e85a4f',
    'success': '#2a9d8f',
    'warning': '#e9c46a',
    'blue': '#3b82f6',
    'text': '#2d3748',
    'text_light': '#4a5568',
    'bg_light': '#f7fafc',
    'white': '#ffffff',
    'gray': '#94a3b8',
}

def setup_figure(figsize=(10, 6), bg_color='white'):
    """Create a clean figure with consistent styling."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.axis('off')
    ax.set_aspect('equal')
    return fig, ax

def save_svg(fig, filename):
    """Save figure as SVG."""
    fig.savefig(f'diagrams/svg/{filename}', format='svg', bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"  ✓ {filename}")


# =============================================================================
# 1. Bounding Box Regression Diagram
# =============================================================================
def create_bbox_regression():
    """Shows how detection is regression: predicting 4 numbers."""
    fig, ax = setup_figure(figsize=(12, 5))

    # Left: Image with cat and bounding box
    img_rect = FancyBboxPatch((0.5, 0.5), 4, 4, boxstyle="round,pad=0.05",
                               facecolor='#e8f4ea', edgecolor=COLORS['text'], linewidth=2)
    ax.add_patch(img_rect)

    # Cat face (simple)
    cat_circle = Circle((2.5, 2.5), 1.2, facecolor=COLORS['warning'], edgecolor=COLORS['text'], linewidth=1.5)
    ax.add_patch(cat_circle)
    # Cat ears
    ax.fill([1.6, 1.8, 2.0], [3.4, 4.0, 3.4], color=COLORS['warning'], edgecolor=COLORS['text'], linewidth=1)
    ax.fill([3.0, 3.2, 3.4], [3.4, 4.0, 3.4], color=COLORS['warning'], edgecolor=COLORS['text'], linewidth=1)
    # Cat eyes
    ax.plot([2.1, 2.1], [2.6, 2.8], 'k-', linewidth=2)
    ax.plot([2.9, 2.9], [2.6, 2.8], 'k-', linewidth=2)
    # Cat nose
    ax.plot(2.5, 2.2, 'k^', markersize=8)

    # Bounding box with coordinates
    bbox = FancyBboxPatch((1.2, 1.2), 2.6, 2.6, boxstyle="square,pad=0",
                          facecolor='none', edgecolor=COLORS['accent'], linewidth=3)
    ax.add_patch(bbox)

    # Coordinate labels
    ax.annotate('x', (1.2, 1.0), fontsize=14, fontweight='bold', color=COLORS['accent'], ha='center')
    ax.annotate('y', (0.8, 1.2), fontsize=14, fontweight='bold', color=COLORS['accent'], ha='center')
    ax.annotate('w', (2.5, 0.85), fontsize=14, fontweight='bold', color=COLORS['accent'], ha='center')
    ax.annotate('h', (4.1, 2.5), fontsize=14, fontweight='bold', color=COLORS['accent'], ha='center')

    # Width arrow
    ax.annotate('', xy=(3.8, 1.1), xytext=(1.2, 1.1),
                arrowprops=dict(arrowstyle='<->', color=COLORS['accent'], lw=2))
    # Height arrow
    ax.annotate('', xy=(3.95, 3.8), xytext=(3.95, 1.2),
                arrowprops=dict(arrowstyle='<->', color=COLORS['accent'], lw=2))

    ax.text(2.5, 5.2, 'Input: Image', fontsize=14, ha='center', fontweight='bold', color=COLORS['primary'])

    # Arrow to right
    ax.annotate('', xy=(6.0, 2.5), xytext=(5.2, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=3))
    ax.text(5.6, 3.0, 'Neural\nNetwork', fontsize=10, ha='center', color=COLORS['text_light'])

    # Right: Output values
    output_box = FancyBboxPatch((6.5, 1.0), 4.5, 3, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['bg_light'], edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(output_box)

    ax.text(8.75, 3.7, 'Output: 4 Numbers', fontsize=14, ha='center', fontweight='bold', color=COLORS['primary'])

    # Output values
    outputs = [
        ('x = 120', 'pixels from left'),
        ('y = 85', 'pixels from top'),
        ('w = 200', 'width in pixels'),
        ('h = 180', 'height in pixels'),
    ]
    for i, (val, desc) in enumerate(outputs):
        y_pos = 3.3 - i * 0.6
        ax.text(7.2, y_pos, val, fontsize=13, fontweight='bold', color=COLORS['accent'], fontfamily='monospace')
        ax.text(9.0, y_pos, desc, fontsize=10, color=COLORS['text_light'])

    ax.text(8.75, 0.5, 'Detection = Regression!', fontsize=12, ha='center',
            fontweight='bold', color=COLORS['success'], style='italic')

    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5.5)
    save_svg(fig, 'bbox_regression.svg')


# =============================================================================
# 2. RL Loop Diagram
# =============================================================================
def create_rl_loop():
    """Shows the agent-environment interaction loop."""
    fig, ax = setup_figure(figsize=(10, 6))

    # Agent box
    agent_box = FancyBboxPatch((0.5, 2), 3, 2, boxstyle="round,pad=0.15",
                                facecolor=COLORS['primary'], edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(agent_box)
    ax.text(2, 3, 'Agent', fontsize=18, ha='center', va='center',
            fontweight='bold', color='white')
    ax.text(2, 2.4, '(Policy π)', fontsize=12, ha='center', va='center', color='#a0b4c7')

    # Environment box
    env_box = FancyBboxPatch((6.5, 2), 3, 2, boxstyle="round,pad=0.15",
                              facecolor=COLORS['success'], edgecolor=COLORS['success'], linewidth=2)
    ax.add_patch(env_box)
    ax.text(8, 3, 'Environment', fontsize=18, ha='center', va='center',
            fontweight='bold', color='white')
    ax.text(8, 2.4, '(World/Game)', fontsize=12, ha='center', va='center', color='#a0c7c0')

    # Action arrow (top)
    ax.annotate('', xy=(6.3, 3.5), xytext=(3.7, 3.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=3,
                               connectionstyle='arc3,rad=0.2'))
    ax.text(5, 4.5, 'Action aₜ', fontsize=14, ha='center', fontweight='bold', color=COLORS['accent'])
    ax.text(5, 4.1, '"move left"', fontsize=10, ha='center', color=COLORS['text_light'], style='italic')

    # State arrow (bottom left)
    ax.annotate('', xy=(3.7, 2.5), xytext=(6.3, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['blue'], lw=3,
                               connectionstyle='arc3,rad=0.2'))
    ax.text(5, 1.5, 'State sₜ₊₁', fontsize=14, ha='center', fontweight='bold', color=COLORS['blue'])
    ax.text(5, 1.1, '"new position"', fontsize=10, ha='center', color=COLORS['text_light'], style='italic')

    # Reward (shown below state)
    ax.text(5, 0.5, '+ Reward rₜ', fontsize=14, ha='center', fontweight='bold', color=COLORS['warning'])
    ax.text(5, 0.15, '"+10 points"', fontsize=10, ha='center', color=COLORS['text_light'], style='italic')

    # Goal text
    ax.text(5, 5.5, 'Goal: Maximize cumulative reward Σrₜ', fontsize=14, ha='center',
            fontweight='bold', color=COLORS['primary'])

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 6)
    save_svg(fig, 'rl_loop.svg')


# =============================================================================
# 3-6. Neural Network Diagrams
# =============================================================================
def draw_nn_layer(ax, x, y_center, n_neurons, color, label=None, radius=0.25):
    """Draw a layer of neurons."""
    neurons = []
    y_positions = np.linspace(y_center - (n_neurons-1)*0.7/2, y_center + (n_neurons-1)*0.7/2, n_neurons)
    for y in y_positions:
        circle = Circle((x, y), radius, facecolor=color, edgecolor=COLORS['text'], linewidth=1.5)
        ax.add_patch(circle)
        neurons.append((x, y))
    if label:
        ax.text(x, y_positions[-1] + 0.6, label, fontsize=10, ha='center', color=COLORS['text_light'])
    return neurons

def draw_connections(ax, layer1, layer2, color=COLORS['gray'], alpha=0.3):
    """Draw connections between two layers."""
    for x1, y1 in layer1:
        for x2, y2 in layer2:
            ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=0.8)

def create_nn_binary():
    """Neural network for binary classification."""
    fig, ax = setup_figure(figsize=(11, 5))

    # Draw layers
    input_layer = draw_nn_layer(ax, 1, 2.5, 4, '#e2e8f0', 'Input\n(features)')
    hidden1 = draw_nn_layer(ax, 3, 2.5, 5, COLORS['primary_light'])
    hidden2 = draw_nn_layer(ax, 5, 2.5, 5, COLORS['primary_light'])
    output_layer = draw_nn_layer(ax, 7, 2.5, 1, COLORS['accent'], radius=0.35)

    # Draw connections
    draw_connections(ax, input_layer, hidden1)
    draw_connections(ax, hidden1, hidden2)
    draw_connections(ax, hidden2, output_layer)

    # Labels
    ax.text(4, 4.2, 'Hidden Layers', fontsize=11, ha='center', color=COLORS['text_light'])
    ax.text(7, 3.3, 'σ(z)', fontsize=12, ha='center', color=COLORS['accent'], fontweight='bold')

    # Output annotation
    ax.annotate('', xy=(8.5, 2.5), xytext=(7.5, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=2))

    output_box = FancyBboxPatch((8.7, 1.8), 2, 1.4, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['bg_light'], edgecolor=COLORS['primary'], linewidth=1.5)
    ax.add_patch(output_box)
    ax.text(9.7, 2.7, 'p ∈ [0, 1]', fontsize=13, ha='center', fontweight='bold', color=COLORS['primary'])
    ax.text(9.7, 2.2, 'Sigmoid', fontsize=10, ha='center', color=COLORS['text_light'])

    # Example
    ax.text(5.5, 0.5, 'Example: p = 0.87 → "Spam" (threshold = 0.5)', fontsize=11,
            ha='center', color=COLORS['success'], fontweight='bold')

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.8)
    save_svg(fig, 'nn_binary_classification.svg')


def create_nn_multiclass():
    """Neural network for multi-class classification."""
    fig, ax = setup_figure(figsize=(12, 5.5))

    # Draw layers
    input_layer = draw_nn_layer(ax, 1, 2.5, 4, '#e2e8f0', 'Input')
    hidden1 = draw_nn_layer(ax, 3, 2.5, 5, COLORS['primary_light'])
    hidden2 = draw_nn_layer(ax, 5, 2.5, 5, COLORS['primary_light'])

    # Output layer - C classes
    output_neurons = []
    output_colors = [COLORS['accent'], COLORS['success'], COLORS['warning'], COLORS['blue']]
    y_positions = [3.5, 2.8, 2.2, 1.5]
    labels = ['Cat', 'Dog', 'Bird', 'Fish']

    for i, (y, color, label) in enumerate(zip(y_positions, output_colors, labels)):
        circle = Circle((7, y), 0.3, facecolor=color, edgecolor=COLORS['text'], linewidth=1.5)
        ax.add_patch(circle)
        output_neurons.append((7, y))
        ax.text(7.6, y, label, fontsize=10, va='center', color=COLORS['text'])

    # Draw connections
    draw_connections(ax, input_layer, hidden1)
    draw_connections(ax, hidden1, hidden2)
    draw_connections(ax, hidden2, output_neurons)

    ax.text(4, 4.3, 'Hidden Layers', fontsize=11, ha='center', color=COLORS['text_light'])
    ax.text(7, 4.3, 'C outputs', fontsize=11, ha='center', color=COLORS['text_light'])

    # Softmax annotation
    ax.annotate('', xy=(9, 2.5), xytext=(7.9, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=2))
    ax.text(8.4, 3.0, 'Softmax', fontsize=10, ha='center', color=COLORS['text_light'])

    # Output probabilities
    output_box = FancyBboxPatch((9.2, 1.2), 2.3, 2.6, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['bg_light'], edgecolor=COLORS['primary'], linewidth=1.5)
    ax.add_patch(output_box)

    probs = [('Cat:', '0.75'), ('Dog:', '0.15'), ('Bird:', '0.08'), ('Fish:', '0.02')]
    for i, (name, prob) in enumerate(probs):
        y_pos = 3.4 - i * 0.55
        ax.text(9.5, y_pos, name, fontsize=10, color=COLORS['text'])
        ax.text(10.9, y_pos, prob, fontsize=11, ha='right', fontweight='bold',
                color=COLORS['accent'] if i == 0 else COLORS['text_light'])

    ax.text(10.35, 0.85, 'Σ = 1.00', fontsize=10, ha='center', color=COLORS['success'], fontweight='bold')

    ax.set_xlim(0, 12)
    ax.set_ylim(0.5, 5)
    save_svg(fig, 'nn_multiclass.svg')


def create_nn_regression():
    """Neural network for regression."""
    fig, ax = setup_figure(figsize=(11, 5))

    # Draw layers
    input_layer = draw_nn_layer(ax, 1, 2.5, 4, '#e2e8f0', 'Input\n(features)')
    hidden1 = draw_nn_layer(ax, 3, 2.5, 5, COLORS['primary_light'])
    hidden2 = draw_nn_layer(ax, 5, 2.5, 5, COLORS['primary_light'])
    output_layer = draw_nn_layer(ax, 7, 2.5, 1, COLORS['success'], radius=0.35)

    # Draw connections
    draw_connections(ax, input_layer, hidden1)
    draw_connections(ax, hidden1, hidden2)
    draw_connections(ax, hidden2, output_layer)

    ax.text(4, 4.2, 'Hidden Layers', fontsize=11, ha='center', color=COLORS['text_light'])
    ax.text(7, 3.3, 'Linear', fontsize=12, ha='center', color=COLORS['success'], fontweight='bold')

    # Output annotation
    ax.annotate('', xy=(8.5, 2.5), xytext=(7.5, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=2))

    output_box = FancyBboxPatch((8.7, 1.8), 2, 1.4, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['bg_light'], edgecolor=COLORS['primary'], linewidth=1.5)
    ax.add_patch(output_box)
    ax.text(9.7, 2.7, 'ŷ ∈ ℝ', fontsize=13, ha='center', fontweight='bold', color=COLORS['primary'])
    ax.text(9.7, 2.2, 'Any real number', fontsize=9, ha='center', color=COLORS['text_light'])

    # Example
    ax.text(5.5, 0.5, 'Example: ŷ = $425,000 (predicted house price)', fontsize=11,
            ha='center', color=COLORS['success'], fontweight='bold')

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.8)
    save_svg(fig, 'nn_regression.svg')


def create_nn_detection():
    """Neural network for object detection (multi-task output)."""
    fig, ax = setup_figure(figsize=(13, 6.5))

    # Draw input and hidden layers
    input_layer = draw_nn_layer(ax, 0.8, 3, 4, '#e2e8f0', 'Input\n(image)')
    hidden1 = draw_nn_layer(ax, 2.5, 3, 6, COLORS['primary_light'])
    hidden2 = draw_nn_layer(ax, 4.2, 3, 6, COLORS['primary_light'])

    ax.text(3.35, 5.5, 'Backbone CNN', fontsize=11, ha='center', color=COLORS['text_light'])

    # Split into three output heads
    # Box regression head
    box_neurons = draw_nn_layer(ax, 6.5, 4.5, 4, COLORS['accent'], radius=0.22)
    draw_connections(ax, hidden2, box_neurons, alpha=0.2)

    # Box output
    box_box = FancyBboxPatch((7.5, 3.7), 2.8, 1.6, boxstyle="round,pad=0.08",
                              facecolor='#fef2f2', edgecolor=COLORS['accent'], linewidth=1.5)
    ax.add_patch(box_box)
    ax.text(8.9, 5.0, 'Box (Regression)', fontsize=10, ha='center', fontweight='bold', color=COLORS['accent'])
    ax.text(7.7, 4.5, 'x, y, w, h', fontsize=11, color=COLORS['text'], fontfamily='monospace')
    ax.text(7.7, 4.0, '4 real numbers', fontsize=9, color=COLORS['text_light'])
    ax.annotate('', xy=(7.4, 4.5), xytext=(7.0, 4.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5))

    # Objectness head
    obj_neurons = draw_nn_layer(ax, 6.5, 3, 1, COLORS['success'], radius=0.28)
    draw_connections(ax, hidden2, obj_neurons, alpha=0.2)

    # Objectness output
    obj_box = FancyBboxPatch((7.5, 2.2), 2.8, 1.3, boxstyle="round,pad=0.08",
                              facecolor='#f0fff4', edgecolor=COLORS['success'], linewidth=1.5)
    ax.add_patch(obj_box)
    ax.text(8.9, 3.2, 'Objectness', fontsize=10, ha='center', fontweight='bold', color=COLORS['success'])
    ax.text(7.7, 2.8, 'σ → [0,1]', fontsize=11, color=COLORS['text'], fontfamily='monospace')
    ax.text(7.7, 2.4, '"Is there an object?"', fontsize=9, color=COLORS['text_light'])
    ax.annotate('', xy=(7.4, 2.85), xytext=(7.0, 3),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1.5))

    # Class head
    class_neurons = draw_nn_layer(ax, 6.5, 1.3, 3, COLORS['blue'], radius=0.22)
    draw_connections(ax, hidden2, class_neurons, alpha=0.2)

    # Class output
    class_box = FancyBboxPatch((7.5, 0.3), 2.8, 1.6, boxstyle="round,pad=0.08",
                                facecolor='#eff6ff', edgecolor=COLORS['blue'], linewidth=1.5)
    ax.add_patch(class_box)
    ax.text(8.9, 1.6, 'Class (Softmax)', fontsize=10, ha='center', fontweight='bold', color=COLORS['blue'])
    ax.text(7.7, 1.1, 'C probabilities', fontsize=11, color=COLORS['text'], fontfamily='monospace')
    ax.text(7.7, 0.6, 'Σ = 1.0', fontsize=9, color=COLORS['text_light'])
    ax.annotate('', xy=(7.4, 1.3), xytext=(7.0, 1.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['blue'], lw=1.5))

    # Combined loss
    loss_box = FancyBboxPatch((10.8, 2.3), 2.0, 1.4, boxstyle="round,pad=0.1",
                               facecolor=COLORS['warning'], edgecolor=COLORS['warning'], linewidth=2, alpha=0.3)
    ax.add_patch(loss_box)
    ax.text(11.8, 3.3, 'Combined Loss', fontsize=10, ha='center', fontweight='bold', color=COLORS['text'])
    ax.text(11.8, 2.85, 'L = λ₁Lbox', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(11.8, 2.55, '  + λ₂Lobj', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(11.8, 2.25, '  + λ₃Lcls', fontsize=9, ha='center', color=COLORS['text'])

    # Arrows to loss
    for y in [4.5, 2.85, 1.1]:
        ax.annotate('', xy=(10.7, 3), xytext=(10.3, y),
                    arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1,
                                   connectionstyle='arc3,rad=0'))

    ax.set_xlim(0, 13.2)
    ax.set_ylim(-0.2, 6)
    save_svg(fig, 'nn_detection.svg')


# =============================================================================
# 7. ML Decision Flowchart
# =============================================================================
def create_ml_flowchart():
    """Decision flowchart for choosing ML task type."""
    fig, ax = setup_figure(figsize=(12, 8))

    # Start node
    start = FancyBboxPatch((4.5, 7), 3, 0.8, boxstyle="round,pad=0.1",
                            facecolor=COLORS['primary'], edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(start)
    ax.text(6, 7.4, 'Do you have labels?', fontsize=11, ha='center', fontweight='bold', color='white')

    # Yes/No arrows from start
    ax.annotate('', xy=(3.5, 6.2), xytext=(4.5, 7),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))
    ax.text(3.5, 6.7, 'Yes', fontsize=10, color=COLORS['success'], fontweight='bold')

    ax.annotate('', xy=(8.5, 6.2), xytext=(7.5, 7),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))
    ax.text(8.2, 6.7, 'No', fontsize=10, color=COLORS['accent'], fontweight='bold')

    # Supervised branch
    sup_box = FancyBboxPatch((1.5, 5.4), 4, 0.8, boxstyle="round,pad=0.1",
                              facecolor=COLORS['success'], edgecolor=COLORS['success'], alpha=0.2, linewidth=2)
    ax.add_patch(sup_box)
    ax.text(3.5, 5.8, 'SUPERVISED LEARNING', fontsize=10, ha='center', fontweight='bold', color=COLORS['success'])

    # Question under supervised
    q1 = FancyBboxPatch((1.5, 4.3), 4, 0.8, boxstyle="round,pad=0.1",
                         facecolor=COLORS['bg_light'], edgecolor=COLORS['text'], linewidth=1.5)
    ax.add_patch(q1)
    ax.text(3.5, 4.7, 'What is the output type?', fontsize=10, ha='center', color=COLORS['text'])
    ax.annotate('', xy=(3.5, 4.3), xytext=(3.5, 5.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))

    # Classification vs Regression
    cls_box = FancyBboxPatch((0.3, 2.8), 2.8, 1.2, boxstyle="round,pad=0.1",
                              facecolor='#dbeafe', edgecolor=COLORS['blue'], linewidth=2)
    ax.add_patch(cls_box)
    ax.text(1.7, 3.6, 'Category?', fontsize=10, ha='center', fontweight='bold', color=COLORS['blue'])
    ax.text(1.7, 3.1, 'Classification', fontsize=11, ha='center', color=COLORS['text'])

    reg_box = FancyBboxPatch((3.4, 2.8), 2.8, 1.2, boxstyle="round,pad=0.1",
                              facecolor='#d1fae5', edgecolor=COLORS['success'], linewidth=2)
    ax.add_patch(reg_box)
    ax.text(4.8, 3.6, 'Number?', fontsize=10, ha='center', fontweight='bold', color=COLORS['success'])
    ax.text(4.8, 3.1, 'Regression', fontsize=11, ha='center', color=COLORS['text'])

    ax.annotate('', xy=(1.7, 4.0), xytext=(2.8, 4.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))
    ax.annotate('', xy=(4.8, 4.0), xytext=(4.2, 4.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))

    # Sub-types under classification
    bin_box = FancyBboxPatch((0.1, 1.3), 1.4, 1.0, boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor=COLORS['blue'], linewidth=1)
    ax.add_patch(bin_box)
    ax.text(0.8, 2.0, 'Binary', fontsize=9, ha='center', fontweight='bold', color=COLORS['blue'])
    ax.text(0.8, 1.6, 'spam/not', fontsize=8, ha='center', color=COLORS['text_light'])

    multi_box = FancyBboxPatch((1.8, 1.3), 1.6, 1.0, boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor=COLORS['blue'], linewidth=1)
    ax.add_patch(multi_box)
    ax.text(2.6, 2.0, 'Multi-class', fontsize=9, ha='center', fontweight='bold', color=COLORS['blue'])
    ax.text(2.6, 1.6, 'cat/dog/bird', fontsize=8, ha='center', color=COLORS['text_light'])

    ax.annotate('', xy=(0.8, 2.3), xytext=(1.3, 2.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1))
    ax.annotate('', xy=(2.6, 2.3), xytext=(2.1, 2.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1))

    # Unsupervised branch
    unsup_box = FancyBboxPatch((6.5, 5.4), 4, 0.8, boxstyle="round,pad=0.1",
                                facecolor=COLORS['accent'], edgecolor=COLORS['accent'], alpha=0.2, linewidth=2)
    ax.add_patch(unsup_box)
    ax.text(8.5, 5.8, 'UNSUPERVISED LEARNING', fontsize=10, ha='center', fontweight='bold', color=COLORS['accent'])

    ax.annotate('', xy=(8.5, 5.4), xytext=(8.5, 6.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))

    # Unsupervised options
    clust_box = FancyBboxPatch((6.3, 3.8), 2, 1.2, boxstyle="round,pad=0.1",
                                facecolor='#fef3c7', edgecolor=COLORS['warning'], linewidth=2)
    ax.add_patch(clust_box)
    ax.text(7.3, 4.6, 'Find groups?', fontsize=9, ha='center', fontweight='bold', color=COLORS['warning'])
    ax.text(7.3, 4.1, 'Clustering', fontsize=10, ha='center', color=COLORS['text'])

    gen_box = FancyBboxPatch((8.7, 3.8), 2.2, 1.2, boxstyle="round,pad=0.1",
                              facecolor='#fce7f3', edgecolor='#ec4899', linewidth=2)
    ax.add_patch(gen_box)
    ax.text(9.8, 4.6, 'Create new?', fontsize=9, ha='center', fontweight='bold', color='#ec4899')
    ax.text(9.8, 4.1, 'Generative', fontsize=10, ha='center', color=COLORS['text'])

    ax.annotate('', xy=(7.3, 5.0), xytext=(7.8, 5.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))
    ax.annotate('', xy=(9.8, 5.0), xytext=(9.2, 5.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))

    # RL branch (separate)
    rl_q = FancyBboxPatch((4, 0.3), 4, 0.8, boxstyle="round,pad=0.1",
                           facecolor=COLORS['bg_light'], edgecolor=COLORS['primary'], linewidth=1.5)
    ax.add_patch(rl_q)
    ax.text(6, 0.7, 'Learning from rewards? → RL', fontsize=10, ha='center',
            color=COLORS['primary'], fontweight='bold')

    ax.set_xlim(-0.2, 11.5)
    ax.set_ylim(-0.2, 8)
    save_svg(fig, 'ml_decision_flowchart.svg')


# =============================================================================
# Additional Simple Diagrams (keeping as visuals but simplified)
# =============================================================================
def create_learning_paradigms():
    """Simple 3-column comparison of learning paradigms."""
    fig, ax = setup_figure(figsize=(12, 4))

    paradigms = [
        ('Supervised', 'Learn from labeled examples',
         ['X: Images', 'Y: Labels'], COLORS['success']),
        ('Unsupervised', 'Find patterns in data',
         ['X: Data only', 'Y: None'], COLORS['warning']),
        ('Reinforcement', 'Learn from rewards',
         ['Actions → Rewards', 'Trial & error'], COLORS['accent']),
    ]

    for i, (name, desc, items, color) in enumerate(paradigms):
        x = 2 + i * 4

        # Box
        box = FancyBboxPatch((x-1.5, 0.5), 3, 3, boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor=color, alpha=0.15, linewidth=2)
        ax.add_patch(box)

        # Title
        ax.text(x, 3.2, name, fontsize=14, ha='center', fontweight='bold', color=color)
        ax.text(x, 2.6, desc, fontsize=10, ha='center', color=COLORS['text_light'], style='italic')

        # Items
        for j, item in enumerate(items):
            ax.text(x, 1.8 - j*0.5, f'• {item}', fontsize=10, ha='center', color=COLORS['text'])

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    save_svg(fig, 'learning_paradigms.svg')


def create_ml_recipe():
    """Simple flow: Data → Model → Predictions."""
    fig, ax = setup_figure(figsize=(11, 3))

    steps = [
        ('Data\n(X, Y)', COLORS['blue']),
        ('Model\nf(x; θ)', COLORS['primary']),
        ('Predictions\nŷ = f(x)', COLORS['success']),
        ('Loss\nL(y, ŷ)', COLORS['accent']),
    ]

    for i, (text, color) in enumerate(steps):
        x = 1.5 + i * 2.8

        box = FancyBboxPatch((x-1, 0.6), 2, 1.5, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor=color, alpha=0.2, linewidth=2)
        ax.add_patch(box)
        ax.text(x, 1.35, text, fontsize=11, ha='center', va='center',
                fontweight='bold', color=color)

        if i < len(steps) - 1:
            ax.annotate('', xy=(x+1.2, 1.35), xytext=(x+0.9, 1.35),
                       arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))

    # Feedback arrow
    ax.annotate('', xy=(2.3, 0.4), xytext=(9.3, 0.4),
               arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2,
                              connectionstyle='arc3,rad=-0.3'))
    ax.text(5.8, -0.1, 'Update θ to minimize loss', fontsize=10, ha='center',
            color=COLORS['accent'], style='italic')

    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 2.5)
    save_svg(fig, 'ml_recipe.svg')


def create_vision_hierarchy():
    """Vision tasks from classification to segmentation."""
    fig, ax = setup_figure(figsize=(12, 5))

    levels = [
        ('Classification', 'One label\nfor image', COLORS['blue'], 1),
        ('Detection', 'Labels +\nboxes', COLORS['success'], 2),
        ('Semantic Seg.', 'Label every\npixel', COLORS['warning'], 3),
        ('Instance Seg.', 'Separate\neach object', COLORS['accent'], 4),
    ]

    for i, (name, desc, color, level) in enumerate(levels):
        x = 1.5 + i * 2.8
        y_base = 1 + level * 0.3

        # Box
        box = FancyBboxPatch((x-1.2, y_base), 2.4, 2, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor=color, alpha=0.15, linewidth=2)
        ax.add_patch(box)

        ax.text(x, y_base + 1.6, name, fontsize=11, ha='center', fontweight='bold', color=color)
        ax.text(x, y_base + 0.9, desc, fontsize=9, ha='center', color=COLORS['text_light'])
        ax.text(x, y_base + 0.3, f'Level {level}', fontsize=9, ha='center',
                color=color, fontweight='bold')

        if i < len(levels) - 1:
            ax.annotate('', xy=(x+1.4, y_base + 1.2), xytext=(x+1.1, y_base + 1),
                       arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))

    ax.text(6, 4.5, 'More information per prediction →', fontsize=11, ha='center',
            color=COLORS['primary'], fontweight='bold')
    ax.text(6, 0.3, 'More compute & data required →', fontsize=10, ha='center',
            color=COLORS['text_light'], style='italic')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    save_svg(fig, 'vision_tasks_hierarchy.svg')


def create_supervised_vs_unsupervised():
    """Side by side comparison."""
    fig, ax = setup_figure(figsize=(11, 4))

    # Supervised side
    sup_box = FancyBboxPatch((0.5, 0.5), 4.5, 3, boxstyle="round,pad=0.15",
                              facecolor=COLORS['success'], edgecolor=COLORS['success'], alpha=0.1, linewidth=2)
    ax.add_patch(sup_box)
    ax.text(2.75, 3.1, 'Supervised', fontsize=14, ha='center', fontweight='bold', color=COLORS['success'])
    ax.text(2.75, 2.5, '"Teacher shows correct answers"', fontsize=9, ha='center',
            color=COLORS['text_light'], style='italic')

    # Data points with labels
    points = [(1.5, 1.8, 'A'), (2.0, 1.2, 'B'), (3.0, 1.5, 'A'), (3.5, 2.0, 'B'), (4.0, 1.3, 'A')]
    for x, y, label in points:
        color = COLORS['blue'] if label == 'A' else COLORS['accent']
        ax.scatter(x, y, s=80, c=color, zorder=5)
        ax.text(x+0.15, y+0.15, label, fontsize=8, color=color, fontweight='bold')

    # Unsupervised side
    unsup_box = FancyBboxPatch((5.5, 0.5), 4.5, 3, boxstyle="round,pad=0.15",
                                facecolor=COLORS['warning'], edgecolor=COLORS['warning'], alpha=0.1, linewidth=2)
    ax.add_patch(unsup_box)
    ax.text(7.75, 3.1, 'Unsupervised', fontsize=14, ha='center', fontweight='bold', color=COLORS['warning'])
    ax.text(7.75, 2.5, '"Find patterns on your own"', fontsize=9, ha='center',
            color=COLORS['text_light'], style='italic')

    # Data points without labels, but visually clustered
    cluster1 = [(6.3, 1.8), (6.5, 1.5), (6.8, 1.7)]
    cluster2 = [(8.5, 1.5), (8.8, 1.2), (9.0, 1.6)]
    for x, y in cluster1:
        ax.scatter(x, y, s=80, c=COLORS['gray'], zorder=5)
    for x, y in cluster2:
        ax.scatter(x, y, s=80, c=COLORS['gray'], zorder=5)
    ax.text(6.5, 1.0, '?', fontsize=12, ha='center', color=COLORS['text_light'])
    ax.text(8.8, 0.9, '?', fontsize=12, ha='center', color=COLORS['text_light'])

    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4)
    save_svg(fig, 'supervised_vs_unsupervised.svg')


def create_generative_vs_discriminative():
    """Compare generative vs discriminative models."""
    fig, ax = setup_figure(figsize=(11, 4.5))

    # Discriminative
    disc_box = FancyBboxPatch((0.5, 0.5), 4.5, 3.5, boxstyle="round,pad=0.15",
                               facecolor=COLORS['blue'], edgecolor=COLORS['blue'], alpha=0.1, linewidth=2)
    ax.add_patch(disc_box)
    ax.text(2.75, 3.6, 'Discriminative', fontsize=14, ha='center', fontweight='bold', color=COLORS['blue'])
    ax.text(2.75, 3.1, 'P(label | input)', fontsize=11, ha='center', color=COLORS['text'])

    # Input → Label flow
    ax.text(1.5, 2.2, '🖼️', fontsize=24, ha='center')
    ax.annotate('', xy=(3.5, 2.2), xytext=(2.3, 2.2),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))
    ax.text(4.0, 2.2, '"Cat"', fontsize=12, ha='center', fontweight='bold', color=COLORS['blue'])
    ax.text(2.75, 1.2, 'Classifies existing data', fontsize=9, ha='center',
            color=COLORS['text_light'], style='italic')

    # Generative
    gen_box = FancyBboxPatch((5.5, 0.5), 4.5, 3.5, boxstyle="round,pad=0.15",
                              facecolor=COLORS['accent'], edgecolor=COLORS['accent'], alpha=0.1, linewidth=2)
    ax.add_patch(gen_box)
    ax.text(7.75, 3.6, 'Generative', fontsize=14, ha='center', fontweight='bold', color=COLORS['accent'])
    ax.text(7.75, 3.1, 'P(data) or P(data | text)', fontsize=11, ha='center', color=COLORS['text'])

    # Prompt → Image flow
    ax.text(6.2, 2.2, '"Cat"', fontsize=11, ha='center', color=COLORS['text'])
    ax.annotate('', xy=(8.7, 2.2), xytext=(7.0, 2.2),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))
    ax.text(9.3, 2.2, '🖼️', fontsize=24, ha='center')
    ax.text(7.75, 1.2, 'Creates new data', fontsize=9, ha='center',
            color=COLORS['text_light'], style='italic')

    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.3)
    save_svg(fig, 'generative_vs_discriminative.svg')


def create_multimodal():
    """Show multimodal inputs/outputs."""
    fig, ax = setup_figure(figsize=(11, 5))

    # Center: Multimodal Model
    model_box = FancyBboxPatch((4, 1.5), 3, 2, boxstyle="round,pad=0.15",
                                facecolor=COLORS['primary'], edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(model_box)
    ax.text(5.5, 2.7, 'Multimodal', fontsize=14, ha='center', fontweight='bold', color='white')
    ax.text(5.5, 2.2, 'Model', fontsize=14, ha='center', fontweight='bold', color='white')

    # Inputs (left)
    inputs = [('🖼️ Image', 3.8), ('📝 Text', 2.5), ('🔊 Audio', 1.2)]
    for text, y in inputs:
        ax.text(1.5, y, text, fontsize=12, ha='center', color=COLORS['text'])
        ax.annotate('', xy=(3.8, 2.5), xytext=(2.5, y),
                   arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1.5,
                                  connectionstyle='arc3,rad=0'))
    ax.text(1.5, 4.3, 'Inputs', fontsize=12, ha='center', fontweight='bold', color=COLORS['success'])

    # Outputs (right)
    outputs = [('Caption 📝', 3.8), ('Answer 💬', 2.5), ('Action 🎮', 1.2)]
    for text, y in outputs:
        ax.text(9.5, y, text, fontsize=12, ha='center', color=COLORS['text'])
        ax.annotate('', xy=(8.5, y), xytext=(7.2, 2.5),
                   arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5,
                                  connectionstyle='arc3,rad=0'))
    ax.text(9.5, 4.3, 'Outputs', fontsize=12, ha='center', fontweight='bold', color=COLORS['accent'])

    ax.text(5.5, 4.5, 'GPT-4, Claude, Gemini can process multiple modalities!',
            fontsize=10, ha='center', color=COLORS['text_light'], style='italic')

    ax.set_xlim(0, 11)
    ax.set_ylim(0.5, 5)
    save_svg(fig, 'multimodal.svg')


def create_deep_learning_revolution():
    """Before/After deep learning comparison."""
    fig, ax = setup_figure(figsize=(12, 4.5))

    # Before (Traditional ML)
    before_box = FancyBboxPatch((0.3, 0.5), 5, 3.5, boxstyle="round,pad=0.1",
                                 facecolor='#fef2f2', edgecolor=COLORS['accent'], linewidth=2)
    ax.add_patch(before_box)
    ax.text(2.8, 3.6, 'Traditional ML', fontsize=13, ha='center', fontweight='bold', color=COLORS['accent'])

    # Flow for traditional
    steps_before = ['Raw Data', 'Hand-crafted\nFeatures', 'Model', 'Output']
    x_positions = [0.8, 2.0, 3.5, 4.8]
    for i, (step, x) in enumerate(zip(steps_before, x_positions)):
        color = COLORS['accent'] if 'Hand' in step else COLORS['text']
        ax.text(x, 2.0, step, fontsize=9, ha='center', va='center', color=color,
                fontweight='bold' if 'Hand' in step else 'normal')
        if i < len(steps_before) - 1:
            ax.annotate('', xy=(x_positions[i+1]-0.4, 2.0), xytext=(x+0.4, 2.0),
                       arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))

    ax.text(2.8, 1.0, '👤 Human designs features', fontsize=9, ha='center',
            color=COLORS['accent'], style='italic')

    # After (Deep Learning)
    after_box = FancyBboxPatch((5.8, 0.5), 5.5, 3.5, boxstyle="round,pad=0.1",
                                facecolor='#f0fdf4', edgecolor=COLORS['success'], linewidth=2)
    ax.add_patch(after_box)
    ax.text(8.55, 3.6, 'Deep Learning', fontsize=13, ha='center', fontweight='bold', color=COLORS['success'])

    # Flow for deep learning
    steps_after = ['Raw Data', 'Neural Network\n(learns features)', 'Output']
    x_positions_after = [6.5, 8.55, 10.6]
    for i, (step, x) in enumerate(zip(steps_after, x_positions_after)):
        color = COLORS['success'] if 'learns' in step else COLORS['text']
        ax.text(x, 2.0, step, fontsize=9, ha='center', va='center', color=color,
                fontweight='bold' if 'learns' in step else 'normal')
        if i < len(steps_after) - 1:
            ax.annotate('', xy=(x_positions_after[i+1]-0.6, 2.0), xytext=(x+0.5, 2.0),
                       arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))

    ax.text(8.55, 1.0, '🤖 Model learns features from data', fontsize=9, ha='center',
            color=COLORS['success'], style='italic')

    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 4.3)
    save_svg(fig, 'deep_learning_revolution.svg')


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Generating ML Tasks diagrams...")

    # Core diagrams that need visuals
    print("\nCore diagrams:")
    create_bbox_regression()
    create_rl_loop()
    create_nn_binary()
    create_nn_multiclass()
    create_nn_regression()
    create_nn_detection()
    create_ml_flowchart()

    # Simple comparison diagrams (could be tables but visuals are nice)
    print("\nComparison diagrams:")
    create_learning_paradigms()
    create_ml_recipe()
    create_vision_hierarchy()
    create_supervised_vs_unsupervised()
    create_generative_vs_discriminative()
    create_multimodal()
    create_deep_learning_revolution()

    print("\n✓ All diagrams generated in diagrams/svg/")
