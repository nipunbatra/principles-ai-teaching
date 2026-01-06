#!/usr/bin/env python3
"""
Shared style module for all lecture diagram generation scripts.
Ensures consistent colors, fonts, and styling across all diagrams.

Usage:
    from themes.diagram_style import setup_figure, save_svg, COLORS
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os
from pathlib import Path

# =============================================================================
# COLOR PALETTE (matching iitgn-modern Marp theme)
# =============================================================================
COLORS = {
    # Primary colors
    'primary': '#1e3a5f',
    'primary_light': '#2e5a8f',

    # Accent colors
    'accent': '#e85a4f',        # Red
    'success': '#2a9d8f',       # Green
    'warning': '#e9c46a',       # Yellow

    # Additional colors
    'blue': '#3b82f6',
    'purple': '#8b5cf6',
    'orange': '#f59e0b',
    'green': '#10b981',

    # Text colors
    'text': '#2d3748',
    'text_light': '#4a5568',
    'text_dark': '#1a202c',

    # Background colors
    'bg_light': '#f7fafc',
    'bg_white': '#ffffff',
    'gray': '#94a3b8',
    'light_gray': '#e2e8f0',
}


# =============================================================================
# MATPLOTLIB RC PARAMS (default styling)
# =============================================================================
def get_mpl_style():
    """Return matplotlib rcParams for consistent styling."""
    return {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'text.color': COLORS['text'],
        'axes.labelcolor': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
    }


# =============================================================================
# FIGURE SETUP
# =============================================================================
def setup_figure(figsize=(12, 6), bg_color='white', style_dict=None):
    """
    Create a clean figure with consistent styling.

    Args:
        figsize: Tuple of (width, height) in inches
        bg_color: Background color for figure
        style_dict: Optional dict to override default rcParams

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    # Apply default styling
    style = get_mpl_style()
    if style_dict:
        style.update(style_dict)

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
        ax.set_facecolor(bg_color)
        ax.axis('off')

    return fig, ax


def setup_figure_grid(rows=1, cols=1, figsize=(12, 6), bg_color='white'):
    """
    Create a figure grid with multiple subplots.

    Args:
        rows: Number of rows
        cols: Number of columns
        figsize: Tuple of (width, height) in inches
        bg_color: Background color

    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    style = get_mpl_style()

    with plt.rc_context(style):
        fig, axes = plt.subplots(rows, cols, figsize=figsize, facecolor=bg_color)

        # Handle single subplot case
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        elif rows == 1 or cols == 1:
            axes = np.array(axes)

        for ax in axes.flat:
            ax.set_facecolor(bg_color)

    return fig, axes


# =============================================================================
# SAVE FUNCTIONS
# =============================================================================
def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_svg(fig, filename, output_dir='diagrams/svg'):
    """
    Save figure as SVG (vector graphics, best for slides).

    Args:
        fig: Matplotlib figure object
        filename: Name of output file (without extension)
        output_dir: Directory to save to
    """
    ensure_dir(output_dir)
    filepath = os.path.join(output_dir, f'{filename}.svg')
    fig.savefig(filepath, format='svg', bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
    plt.close(fig)
    return filepath


def save_png(fig, filename, output_dir='diagrams/png', dpi=300):
    """
    Save figure as PNG (raster, good fallback).

    Args:
        fig: Matplotlib figure object
        filename: Name of output file (without extension)
        output_dir: Directory to save to
        dpi: Resolution (dots per inch)
    """
    ensure_dir(output_dir)
    filepath = os.path.join(output_dir, f'{filename}.png')
    fig.savefig(filepath, format='png', bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none', dpi=dpi)
    plt.close(fig)
    return filepath


def save_both(fig, filename, output_dir='diagrams'):
    """
    Save figure as both SVG and PNG.

    Args:
        fig: Matplotlib figure object
        filename: Name of output file (without extension)
        output_dir: Base directory (will create svg/ and png/ subdirs)
    """
    svg_path = save_svg(fig, filename, os.path.join(output_dir, 'svg'))
    png_path = save_png(fig, filename, os.path.join(output_dir, 'png'))
    return svg_path, png_path


# =============================================================================
# CONVENIENCE DRAWING FUNCTIONS
# =============================================================================
def draw_box(ax, x, y, width, height, text='', color=COLORS['blue'],
             alpha=0.3, fontsize=12, **kwargs):
    """
    Draw a labeled box on the axes.

    Args:
        ax: Matplotlib axes
        x, y: Position
        width, height: Size
        text: Optional label text
        color: Face color
        alpha: Transparency
        fontsize: Text size
        **kwargs: Additional arguments for FancyBboxPatch
    """
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor=COLORS['primary'],
                         linewidth=2, alpha=alpha, **kwargs)
    ax.add_patch(box)

    if text:
        ax.text(x + width/2, y + height/2, text,
               fontsize=fontsize, ha='center', va='center',
               fontweight='bold', color=COLORS['primary'])

    return box


def draw_arrow(ax, x1, y1, x2, y2, color=COLORS['primary'], lw=2, **kwargs):
    """
    Draw an arrow from (x1, y1) to (x2, y2).

    Args:
        ax: Matplotlib axes
        x1, y1: Start position
        x2, y2: End position
        color: Arrow color
        lw: Line width
        **kwargs: Additional arguments for annotate
    """
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', color=color, lw=lw, **kwargs))


def draw_label(ax, text, x, y, fontsize=12, fontweight='normal',
               color=COLORS['text'], ha='center', va='center'):
    """Add text label at position."""
    ax.text(x, y, text, fontsize=fontsize, fontweight=fontweight,
           color=color, ha=ha, va=va)


# =============================================================================
# DATASET LOADING HELPERS
# =============================================================================
def load_mnist_sample(n_samples=5):
    """Load a few MNIST samples for visualization."""
    try:
        from torchvision import datasets
        mnist = datasets.MNIST('./data', download=True, train=True)
        return [mnist[i] for i in range(n_samples)]
    except ImportError:
        print("Warning: torchvision not available, cannot load MNIST")
        return None


def load_cifar10_sample(n_samples=5):
    """Load a few CIFAR-10 samples for visualization."""
    try:
        from torchvision import datasets
        cifar = datasets.CIFAR10('./data', download=True, train=True)
        return [cifar[i] for i in range(n_samples)]
    except ImportError:
        print("Warning: torchvision not available, cannot load CIFAR-10")
        return None


def load_iris_dataset():
    """Load Iris dataset for visualization."""
    try:
        from sklearn.datasets import load_iris
        return load_iris()
    except ImportError:
        print("Warning: sklearn not available, cannot load Iris")
        return None


# =============================================================================
# REAL DATASET EXAMPLES
# =============================================================================
def create_sample_classification_plot():
    """Create a sample classification plot using real dataset data."""
    data = load_iris_dataset()
    if data is None:
        return None

    fig, ax = setup_figure(figsize=(10, 8))
    ax.axis('on')

    # Plot Iris dataset
    X = data.data[:, :2]  # Use only first 2 features for 2D plot
    y = data.target

    colors = [COLORS['blue'], COLORS['success'], COLORS['accent']]
    target_names = data.target_names

    for i, (color, name) in enumerate(zip(colors, target_names)):
        ax.scatter(X[y == i, 0], X[y == i, 1], c=color, label=name,
                  s=80, alpha=0.7, edgecolors='white', linewidth=1)

    ax.set_xlabel('Sepal Length (cm)', fontsize=12)
    ax.set_ylabel('Sepal Width (cm)', fontsize=12)
    ax.set_title('Iris Dataset - Real Sample', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig


def create_sample_regression_plot():
    """Create a sample regression plot using synthetic but realistic data."""
    fig, ax = setup_figure(figsize=(10, 6))
    ax.axis('on')

    # Generate realistic regression data
    np.random.seed(42)
    n_samples = 50

    # Features
    X = np.linspace(0, 10, n_samples)
    # True relationship + noise
    y_true = 2.5 * X + 10
    y = y_true + np.random.randn(n_samples) * 5

    # Plot data
    ax.scatter(X, y, c=COLORS['blue'], s=80, alpha=0.7,
              edgecolors='white', linewidth=1, label='Data')

    # Plot true relationship
    ax.plot(X, y_true, c=COLORS['accent'], linewidth=2,
           linestyle='--', label='True relationship')

    ax.set_xlabel('Feature X', fontsize=12)
    ax.set_ylabel('Target y', fontsize=12)
    ax.set_title('Regression Example', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig


# =============================================================================
# MODULE INFO
# =============================================================================
__all__ = [
    'COLORS',
    'setup_figure',
    'setup_figure_grid',
    'save_svg',
    'save_png',
    'save_both',
    'draw_box',
    'draw_arrow',
    'draw_label',
    'load_mnist_sample',
    'load_cifar10_sample',
    'load_iris_dataset',
    'create_sample_classification_plot',
    'create_sample_regression_plot',
]


if __name__ == "__main__":
    # Test the style module
    print("Diagram Style Module")
    print("=" * 50)
    print(f"Available colors: {list(COLORS.keys())}")
    print()
    print("Example usage:")
    print("  from themes.diagram_style import setup_figure, save_svg, COLORS")
    print("  fig, ax = setup_figure(figsize=(12, 6))")
    print("  draw_box(ax, 0.5, 0.5, 2, 1, 'Hello', color=COLORS['blue'])")
    print("  save_svg(fig, 'test')")
