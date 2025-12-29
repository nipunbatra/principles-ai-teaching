#!/usr/bin/env python3
"""
Generate Neural Network Diagrams with actual nodes using matplotlib
Shows proper network structure for each task type
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import numpy as np

# Theme colors (matching iitgn-modern.css)
COLORS = {
    'primary': '#1e3a5f',
    'primary_light': '#2e5a8f',
    'success': '#2a9d8f',
    'warning': '#FF9800',
    'accent': '#e85a4f',
    'fill_primary': '#E3F2FD',
    'fill_success': '#E8F5E9',
    'fill_warning': '#FFF8E1',
    'fill_accent': '#FFEBEE',
    'text': '#2d3748',
}

def draw_layer(ax, x, y_positions, color, fill_color, node_labels=None, layer_label=None, layer_label_y=None):
    """Draw a layer of neurons"""
    for i, y in enumerate(y_positions):
        circle = Circle((x, y), 0.15, facecolor=fill_color, edgecolor=color, linewidth=2)
        ax.add_patch(circle)
        if node_labels and i < len(node_labels):
            ax.text(x, y, node_labels[i], ha='center', va='center', fontsize=8, color=color)

    if layer_label:
        label_y = layer_label_y if layer_label_y else max(y_positions) + 0.4
        ax.text(x, label_y, layer_label, ha='center', va='bottom', fontsize=10,
                fontweight='bold', color=color)

def draw_connections(ax, x1, y1_list, x2, y2_list, color, alpha=0.3):
    """Draw connections between two layers"""
    for y1 in y1_list:
        for y2 in y2_list:
            ax.plot([x1 + 0.15, x2 - 0.15], [y1, y2], color=color, alpha=alpha, linewidth=0.5)

def draw_output_box(ax, x, y, width, height, fill_color, stroke_color, title, content):
    """Draw output/result box"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=fill_color, edgecolor=stroke_color, linewidth=2
    )
    ax.add_patch(box)
    ax.text(x, y + height/4, title, ha='center', va='center',
            fontsize=10, fontweight='bold', color=stroke_color)
    ax.text(x, y - height/6, content, ha='center', va='center',
            fontsize=9, color=COLORS['text'])

def generate_binary_classification():
    """Binary Classification: 1 output neuron with sigmoid"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-1, 2.5)
    ax.axis('off')
    ax.set_aspect('equal')

    # Input layer (d features)
    input_y = [0.2, 0.7, 1.2, 1.7]
    draw_layer(ax, 0, input_y, COLORS['primary'], COLORS['fill_primary'],
               node_labels=['$x_1$', '$x_2$', '...', '$x_d$'], layer_label='Input\n(d features)')

    # Hidden layer 1
    hidden1_y = [0.3, 0.8, 1.3]
    draw_layer(ax, 1.5, hidden1_y, COLORS['warning'], COLORS['fill_warning'],
               layer_label='Hidden 1')

    # Hidden layer 2
    hidden2_y = [0.5, 1.0, 1.5]
    draw_layer(ax, 3, hidden2_y, COLORS['warning'], COLORS['fill_warning'],
               layer_label='Hidden 2')

    # Output layer - 1 neuron
    output_y = [1.0]
    draw_layer(ax, 4.5, output_y, COLORS['success'], COLORS['fill_success'],
               node_labels=['$\\sigma$'], layer_label='Output\n(1 neuron)')

    # Connections
    draw_connections(ax, 0, input_y, 1.5, hidden1_y, COLORS['primary'])
    draw_connections(ax, 1.5, hidden1_y, 3, hidden2_y, COLORS['warning'])
    draw_connections(ax, 3, hidden2_y, 4.5, output_y, COLORS['warning'])

    # Arrow to result
    ax.annotate('', xy=(5.8, 1.0), xytext=(4.65, 1.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))

    # Result box
    draw_output_box(ax, 6.8, 1.0, 1.8, 1.0, COLORS['fill_accent'], COLORS['accent'],
                   '$p = 0.87$', 'Disease? Yes')

    # Formulas
    ax.text(4.5, -0.3, r'$\sigma(z) = \frac{1}{1+e^{-z}}$', ha='center', va='top',
            fontsize=11, color=COLORS['success'])
    ax.text(6.8, 0.3, r'$p \in [0, 1]$', ha='center', va='top',
            fontsize=10, color=COLORS['accent'])

    plt.tight_layout()
    plt.savefig('svg/nn_binary_classification.svg', format='svg', bbox_inches='tight', dpi=150)
    plt.savefig('svg/nn_binary_classification.png', format='png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated: nn_binary_classification.svg/png")

def generate_multiclass():
    """Multi-class Classification: K output neurons with softmax"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-1, 3)
    ax.axis('off')
    ax.set_aspect('equal')

    # Input layer
    input_y = [0.5, 1.0, 1.5, 2.0]
    draw_layer(ax, 0, input_y, COLORS['primary'], COLORS['fill_primary'],
               node_labels=['$x_1$', '$x_2$', '...', '$x_d$'], layer_label='Input', layer_label_y=2.5)

    # Hidden layer
    hidden_y = [0.5, 1.0, 1.5, 2.0]
    draw_layer(ax, 1.5, hidden_y, COLORS['warning'], COLORS['fill_warning'],
               layer_label='Hidden', layer_label_y=2.5)

    # Output layer - K neurons (showing 4 for classes)
    output_y = [0.2, 0.8, 1.4, 2.0]
    draw_layer(ax, 3.5, output_y, COLORS['success'], COLORS['fill_success'],
               node_labels=['0', '1', '...', 'K-1'], layer_label='Output (K classes)', layer_label_y=2.5)

    # Softmax bracket
    ax.plot([4.0, 4.2, 4.2, 4.0], [0.0, 0.0, 2.2, 2.2], color=COLORS['success'], linewidth=2)
    ax.text(4.4, 1.1, 'softmax', ha='left', va='center', fontsize=10, color=COLORS['success'], rotation=90)

    # Connections
    draw_connections(ax, 0, input_y, 1.5, hidden_y, COLORS['primary'])
    draw_connections(ax, 1.5, hidden_y, 3.5, output_y, COLORS['warning'])

    # Arrow to result
    ax.annotate('', xy=(5.5, 1.1), xytext=(4.6, 1.1),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))

    # Result box with probabilities
    draw_output_box(ax, 6.6, 1.1, 2.0, 1.6, COLORS['fill_accent'], COLORS['accent'],
                   'Probabilities', 'Cat: 0.85\nDog: 0.10\nBird: 0.05')

    # Formula
    ax.text(3.5, -0.5, r'$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$', ha='center',
            fontsize=11, color=COLORS['success'])
    ax.text(6.6, -0.1, r'$\sum_i p_i = 1$', ha='center', fontsize=10, color=COLORS['accent'])

    plt.tight_layout()
    plt.savefig('svg/nn_multiclass.svg', format='svg', bbox_inches='tight', dpi=150)
    plt.savefig('svg/nn_multiclass.png', format='png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated: nn_multiclass.svg/png")

def generate_regression():
    """Regression: 1 output neuron, linear activation"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-1, 2.5)
    ax.axis('off')
    ax.set_aspect('equal')

    # Input layer
    input_y = [0.3, 0.8, 1.3, 1.8]
    draw_layer(ax, 0, input_y, COLORS['primary'], COLORS['fill_primary'],
               node_labels=['sqft', 'beds', 'age', '...'], layer_label='Input\n(features)')

    # Hidden layers
    hidden1_y = [0.5, 1.0, 1.5]
    draw_layer(ax, 1.5, hidden1_y, COLORS['warning'], COLORS['fill_warning'],
               layer_label='Hidden')

    hidden2_y = [0.6, 1.1]
    draw_layer(ax, 3, hidden2_y, COLORS['warning'], COLORS['fill_warning'],
               layer_label='Hidden')

    # Output - 1 neuron (linear)
    output_y = [0.85]
    draw_layer(ax, 4.5, output_y, COLORS['success'], COLORS['fill_success'],
               node_labels=['lin'], layer_label='Output\n(1 neuron)')

    # Connections
    draw_connections(ax, 0, input_y, 1.5, hidden1_y, COLORS['primary'])
    draw_connections(ax, 1.5, hidden1_y, 3, hidden2_y, COLORS['warning'])
    draw_connections(ax, 3, hidden2_y, 4.5, output_y, COLORS['warning'])

    # Arrow to result
    ax.annotate('', xy=(5.8, 0.85), xytext=(4.65, 0.85),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))

    # Result
    draw_output_box(ax, 6.8, 0.85, 1.8, 1.0, COLORS['fill_accent'], COLORS['accent'],
                   r'$\hat{y} = \$450K$', 'Price estimate')

    # Formula
    ax.text(4.5, -0.2, r'$\hat{y} = \mathbf{w}^T\mathbf{h} + b$', ha='center',
            fontsize=11, color=COLORS['success'])
    ax.text(6.8, 0.2, r'$\hat{y} \in \mathbb{R}$', ha='center',
            fontsize=10, color=COLORS['accent'])

    plt.tight_layout()
    plt.savefig('svg/nn_regression.svg', format='svg', bbox_inches='tight', dpi=150)
    plt.savefig('svg/nn_regression.png', format='png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated: nn_regression.svg/png")

def generate_detection():
    """Object Detection: Multiple outputs per detection (box + class + confidence)"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-1.5, 3.5)
    ax.axis('off')
    ax.set_aspect('equal')

    # Input (image represented as grid)
    ax.add_patch(FancyBboxPatch((-0.3, 0.2), 0.8, 1.6, boxstyle="round,pad=0.02",
                                facecolor=COLORS['fill_primary'], edgecolor=COLORS['primary'], lw=2))
    ax.text(0.1, 1.0, 'Image\n416×416', ha='center', va='center', fontsize=9, color=COLORS['primary'])
    ax.text(0.1, 2.1, 'Input', ha='center', fontsize=10, fontweight='bold', color=COLORS['primary'])

    # CNN Backbone
    hidden_y = [0.4, 0.9, 1.4]
    draw_layer(ax, 1.5, hidden_y, COLORS['warning'], COLORS['fill_warning'],
               layer_label='CNN\nBackbone')

    # More hidden
    hidden2_y = [0.5, 1.0, 1.5]
    draw_layer(ax, 2.8, hidden2_y, COLORS['warning'], COLORS['fill_warning'])

    # Detection head outputs (per anchor)
    # Box coordinates (4)
    box_y = [2.2, 2.6, 3.0]
    draw_layer(ax, 4.5, box_y, COLORS['success'], '#C8E6C9',
               node_labels=['x', 'y', 'w,h'])
    ax.text(4.5, 3.4, 'Box (4)', ha='center', fontsize=9, fontweight='bold', color=COLORS['success'])

    # Objectness (1)
    obj_y = [1.0]
    draw_layer(ax, 4.5, obj_y, COLORS['success'], '#C8E6C9',
               node_labels=['obj'])
    ax.text(4.5, 1.5, 'Obj (1)', ha='center', fontsize=9, fontweight='bold', color=COLORS['success'])

    # Classes (K)
    class_y = [-0.4, 0.0, 0.4]
    draw_layer(ax, 4.5, class_y, COLORS['success'], '#C8E6C9',
               node_labels=['c1', '...', 'cK'])
    ax.text(4.5, 0.9, 'Class (K)', ha='center', fontsize=9, fontweight='bold', color=COLORS['success'])

    # Connections
    draw_connections(ax, 0.5, [0.6, 1.0, 1.4], 1.5, hidden_y, COLORS['primary'], alpha=0.4)
    draw_connections(ax, 1.5, hidden_y, 2.8, hidden2_y, COLORS['warning'], alpha=0.4)
    draw_connections(ax, 2.8, hidden2_y, 4.5, box_y + obj_y + class_y, COLORS['warning'], alpha=0.2)

    # Bracket for all outputs
    ax.plot([5.0, 5.3, 5.3, 5.0], [-0.6, -0.6, 3.2, 3.2], color=COLORS['success'], linewidth=2)
    ax.text(5.5, 1.3, 'Per\nAnchor', ha='left', va='center', fontsize=9, color=COLORS['success'])

    # Arrow to result
    ax.annotate('', xy=(6.5, 1.0), xytext=(5.8, 1.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))

    # Result box
    draw_output_box(ax, 7.5, 1.0, 2.2, 1.8, COLORS['fill_accent'], COLORS['accent'],
                   'Detections', 'Dog: 95%\n[120, 80, 200, 150]\nCat: 87%\n[300, 100, 120, 80]')

    # Formula
    ax.text(4.5, -1.2, r'Total per cell: 4 + 1 + K = 5 + K values', ha='center',
            fontsize=10, color=COLORS['success'])

    plt.tight_layout()
    plt.savefig('svg/nn_detection.svg', format='svg', bbox_inches='tight', dpi=150)
    plt.savefig('svg/nn_detection.png', format='png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated: nn_detection.svg/png")

if __name__ == '__main__':
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'sans-serif'

    print("Generating NN diagrams with actual nodes...")
    generate_binary_classification()
    generate_multiclass()
    generate_regression()
    generate_detection()
    print("Done!")
