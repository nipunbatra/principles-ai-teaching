#!/usr/bin/env python3
"""
Generate all MLP / Neural Network diagrams for Lecture 05.
Outputs SVG to diagrams/svg/ and PNG (300 DPI) to diagrams/png/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arc, Wedge
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
SVG_DIR = os.path.join(BASE, "diagrams", "svg")
PNG_DIR = os.path.join(BASE, "diagrams", "png")
os.makedirs(SVG_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

# ── Style constants ──────────────────────────────────────────────────────────
RED = "#CC0000"
GREEN = "#00AA00"
BLUE = "#3366CC"
DARK = "#222222"
LIGHT_PINK = "#FFE0E0"
LIGHT_GREEN = "#E0FFE0"
LIGHT_BLUE = "#E0E8FF"
LIGHT_GRAY = "#F0F0F0"
NEURON_COLOR = "#FFFFCC"
NEURON_EDGE = "#888888"

plt.rcParams.update({
    'font.size': 14,
    'font.family': 'sans-serif',
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
})


def save(fig, name):
    fig.savefig(os.path.join(SVG_DIR, f"{name}.svg"), bbox_inches='tight')
    fig.savefig(os.path.join(PNG_DIR, f"{name}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {name}")


def draw_arrow(ax, xy_from, xy_to, **kw):
    """Draw a simple arrow between two points."""
    kw.setdefault('arrowstyle', '->')
    kw.setdefault('lw', 1.8)
    kw.setdefault('color', DARK)
    kw.setdefault('mutation_scale', 18)
    arr = FancyArrowPatch(xy_from, xy_to, **kw)
    ax.add_patch(arr)
    return arr


def draw_neuron(ax, center, radius=0.25, color=NEURON_COLOR):
    c = Circle(center, radius, fc=color, ec=NEURON_EDGE, lw=2, zorder=5)
    ax.add_patch(c)
    return c


def draw_split_neuron(ax, center, radius=0.35, left_label="$\\Sigma$", right_label="step"):
    """Draw a neuron split in half: left=sum, right=activation."""
    cx, cy = center
    # Left half (pink)
    left = Wedge(center, radius, 90, 270, fc=LIGHT_PINK, ec=NEURON_EDGE, lw=2, zorder=5)
    ax.add_patch(left)
    # Right half (green)
    right = Wedge(center, radius, 270, 90, fc=LIGHT_GREEN, ec=NEURON_EDGE, lw=2, zorder=5)
    ax.add_patch(right)
    # Dividing line
    ax.plot([cx, cx], [cy - radius, cy + radius], color=NEURON_EDGE, lw=2, zorder=6)
    # Labels
    ax.text(cx - radius * 0.45, cy, left_label, ha='center', va='center', fontsize=14, fontweight='bold', zorder=7)
    ax.text(cx + radius * 0.45, cy, right_label, ha='center', va='center', fontsize=10, fontweight='bold', zorder=7)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. paradigm_old  – Traditional ML pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def make_paradigm_old():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-1.5, 3.5)
    ax.axis('off')

    # Image box
    img_box = FancyBboxPatch((0, 0.5), 1.8, 2, boxstyle="round,pad=0.1",
                              fc=LIGHT_GRAY, ec=DARK, lw=2)
    ax.add_patch(img_box)
    ax.text(0.9, 1.5, '"2"', ha='center', va='center', fontsize=28, fontweight='bold')
    ax.text(0.9, 0.3, 'Image', ha='center', va='center', fontsize=10, color='gray')

    # Arrow to feature extractor
    draw_arrow(ax, (2.0, 1.5), (3.3, 1.5))
    ax.text(2.65, 1.9, 'Feature\nExtractor', ha='center', va='bottom', fontsize=10, color='gray')

    # Features box (tall rectangle)
    feat_box = FancyBboxPatch((3.5, -0.2), 2.5, 3.4, boxstyle="round,pad=0.1",
                               fc=LIGHT_BLUE, ec=DARK, lw=2)
    ax.add_patch(feat_box)
    features = ["Intensity", "# Horiz. Lines", "# Vert. Lines", "% Black", "..."]
    for i, f in enumerate(features):
        y = 2.6 - i * 0.6
        ax.text(4.75, y, f, ha='center', va='center', fontsize=11)

    # HAND CRAFTED label
    ax.text(4.75, -0.7, "HAND CRAFTED", ha='center', va='center', fontsize=13,
            fontweight='bold', color=RED)

    # Arrow to classifier
    draw_arrow(ax, (6.2, 1.5), (7.5, 1.5))

    # Classifier box
    cls_box = FancyBboxPatch((7.7, 0.5), 2.2, 2, boxstyle="round,pad=0.1",
                              fc=LIGHT_GREEN, ec=DARK, lw=2)
    ax.add_patch(cls_box)
    ax.text(8.8, 1.5, 'Classifier', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(8.8, -0.0, "TRAINABLE", ha='center', va='center', fontsize=13,
            fontweight='bold', color=RED)

    # Arrow to output
    draw_arrow(ax, (10.1, 1.5), (11.2, 1.5))
    ax.text(11.7, 1.5, r'$\hat{y}$', ha='center', va='center', fontsize=22, fontweight='bold')

    save(fig, "paradigm_old")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. paradigm_new  – Neural Network pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def make_paradigm_new():
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-2, 4)
    ax.axis('off')

    # Image box
    img_box = FancyBboxPatch((0, 0.5), 1.8, 2, boxstyle="round,pad=0.1",
                              fc=LIGHT_GRAY, ec=DARK, lw=2)
    ax.add_patch(img_box)
    ax.text(0.9, 1.5, '"2"', ha='center', va='center', fontsize=28, fontweight='bold')
    ax.text(0.9, 0.3, 'Image', ha='center', va='center', fontsize=10, color='gray')

    # Arrow
    draw_arrow(ax, (2.0, 1.5), (3.3, 1.5))

    # Big outer box
    outer = FancyBboxPatch((3.5, -0.2), 7.5, 3.4, boxstyle="round,pad=0.15",
                            fc='#FAFAFA', ec=DARK, lw=2.5)
    ax.add_patch(outer)

    # Inner boxes
    labels = ["Low Level\nFeatures", "...", "High Level\nFeatures", "Classifier"]
    colors = [LIGHT_BLUE, '#F8F8F8', LIGHT_BLUE, LIGHT_GREEN]
    xs = [3.9, 5.7, 7.2, 9.2]
    for i, (lbl, col, x) in enumerate(zip(labels, colors, xs)):
        w = 1.5 if lbl != "..." else 1.0
        box = FancyBboxPatch((x, 0.2), w, 2.6, boxstyle="round,pad=0.08",
                              fc=col, ec=NEURON_EDGE, lw=1.5)
        ax.add_patch(box)
        ax.text(x + w / 2, 1.5, lbl, ha='center', va='center', fontsize=11, fontweight='bold')
        # Arrows between inner boxes
        if i < len(labels) - 1:
            draw_arrow(ax, (x + w + 0.05, 1.5), (xs[i + 1] - 0.05, 1.5), color='gray')

    # TRAINABLE brace label
    ax.annotate('', xy=(3.5, -0.6), xytext=(11.0, -0.6),
                arrowprops=dict(arrowstyle='-', lw=2, color=RED))
    # Brace endpoints
    ax.plot([3.5, 3.5], [-0.4, -0.6], color=RED, lw=2)
    ax.plot([11.0, 11.0], [-0.4, -0.6], color=RED, lw=2)
    ax.text(7.25, -1.1, "TRAINABLE", ha='center', va='center', fontsize=15,
            fontweight='bold', color=RED)

    # Arrow to output
    draw_arrow(ax, (11.2, 1.5), (12.3, 1.5))
    ax.text(12.8, 1.5, r'$\hat{y}$', ha='center', va='center', fontsize=22, fontweight='bold')

    save(fig, "paradigm_new")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. perceptron_simple
# ═══════════════════════════════════════════════════════════════════════════════
def make_perceptron_simple():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 6)
    ax.axis('off')

    neuron_center = (4.5, 2.5)
    nr = 0.5

    # Draw neuron
    draw_neuron(ax, neuron_center, nr)

    # Inputs x1, x2, x3
    inputs = [("$x_1$", 4.5), ("$x_2$", 2.5), ("$x_3$", 0.5)]
    weights = ["$w_1$", "$w_2$", "$w_3$"]
    for (label, y), w in zip(inputs, weights):
        ax.text(0.5, y, label, ha='center', va='center', fontsize=18, fontweight='bold')
        draw_arrow(ax, (1.0, y), (neuron_center[0] - nr - 0.05, neuron_center[1] + (y - 2.5) * 0.35 + (y - 2.5) * 0.05))
        # Weight label
        mid_x = (1.0 + neuron_center[0] - nr) / 2
        mid_y = (y + neuron_center[1]) / 2
        ax.text(mid_x - 0.2, mid_y + 0.25, w, ha='center', va='center', fontsize=14, color=BLUE)

    # Bias input (from top)
    ax.text(4.5, 5.3, "1", ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc=LIGHT_GRAY, ec=DARK))
    draw_arrow(ax, (4.5, 4.9), (4.5, neuron_center[1] + nr + 0.05))
    ax.text(5.1, 4.2, "$b$ (bias)", ha='center', va='center', fontsize=13, color=BLUE)

    # Output
    draw_arrow(ax, (neuron_center[0] + nr + 0.05, neuron_center[1]), (7.0, neuron_center[1]))
    ax.text(7.3, neuron_center[1], r"o/p = $\hat{y}$", ha='left', va='center', fontsize=16, fontweight='bold')

    ax.set_title("Simple Perceptron", fontsize=18, fontweight='bold', pad=10)
    save(fig, "perceptron_simple")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. perceptron_components
# ═══════════════════════════════════════════════════════════════════════════════
def make_perceptron_components():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})
    ax = axes[0]
    ax.set_xlim(-1, 9)
    ax.set_ylim(-2, 6)
    ax.axis('off')

    nc = (4.5, 2.5)
    nr = 0.6

    # Draw split neuron
    draw_split_neuron(ax, nc, nr)

    # Inputs
    inputs = [("$x_1$", 4.5), ("$x_2$", 2.5), ("$x_3$", 0.5)]
    weights = ["$w_1$", "$w_2$", "$w_3$"]
    for (label, y), w in zip(inputs, weights):
        ax.text(0.5, y, label, ha='center', va='center', fontsize=18, fontweight='bold')
        target_y = nc[1] + (y - 2.5) * 0.3
        draw_arrow(ax, (1.0, y), (nc[0] - nr - 0.05, target_y))
        mid_x = (1.0 + nc[0] - nr) / 2
        mid_y = (y + target_y) / 2
        ax.text(mid_x - 0.3, mid_y + 0.25, w, ha='center', va='center', fontsize=14, color=BLUE)

    # Bias
    ax.text(4.5, 5.3, "1", ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc=LIGHT_GRAY, ec=DARK))
    draw_arrow(ax, (4.5, 4.9), (4.5, nc[1] + nr + 0.05))
    ax.text(5.3, 4.2, "$b$ (bias)", ha='center', va='center', fontsize=13, color=BLUE)

    # Output
    draw_arrow(ax, (nc[0] + nr + 0.05, nc[1]), (7.0, nc[1]))
    ax.text(7.3, nc[1], r"o/p = $\hat{y}$", ha='left', va='center', fontsize=16, fontweight='bold')

    # Label below
    ax.text(4.5, -1.2, "Neuron has 2 components:\n(1) Summation   (2) Activation: Step Function",
            ha='center', va='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc='#FFF8E0', ec='orange', lw=1.5))

    # Small step function plot
    ax2 = axes[1]
    z = np.linspace(-3, 3, 200)
    step = np.where(z >= 0, 1, 0)
    ax2.plot(z, step, color=GREEN, lw=3)
    ax2.set_title("Step Function", fontsize=14, fontweight='bold')
    ax2.set_xlabel("z", fontsize=13)
    ax2.set_ylabel("g(z)", fontsize=13)
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.axvline(0, color='gray', lw=0.5)
    ax2.set_ylim(-0.2, 1.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle("Perceptron Components", fontsize=18, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "perceptron_components")


# ═══════════════════════════════════════════════════════════════════════════════
# Gate diagrams helper
# ═══════════════════════════════════════════════════════════════════════════════
def draw_gate_neuron(ax, inputs, weights, bias_val, title, output_label="y",
                     xlim=(-1, 8.5), ylim=(-1, 5)):
    """
    inputs: list of (label, y_pos) for x inputs
    weights: list of weight labels
    bias_val: string for bias weight
    """
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis('off')

    nc = (4.5, 2.0)
    nr = 0.5

    draw_split_neuron(ax, nc, nr)

    # Bias from top
    ax.text(nc[0], 4.2, "1", ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc=LIGHT_GRAY, ec=DARK))
    draw_arrow(ax, (nc[0], 3.8), (nc[0], nc[1] + nr + 0.05))
    ax.text(nc[0] + 0.8, 3.2, f"$b={bias_val}$", ha='center', va='center', fontsize=13,
            fontweight='bold', color=RED)

    # Inputs
    for (label, y), w in zip(inputs, weights):
        ax.text(0.5, y, f"${label}$", ha='center', va='center', fontsize=16, fontweight='bold')
        target_y = nc[1] + (y - nc[1]) * 0.3
        draw_arrow(ax, (1.1, y), (nc[0] - nr - 0.05, target_y))
        mid_x = (1.1 + nc[0] - nr) / 2
        mid_y = (y + target_y) / 2
        ax.text(mid_x, mid_y + 0.3, f"$w={w}$", ha='center', va='center', fontsize=13,
                fontweight='bold', color=RED)

    # Output
    draw_arrow(ax, (nc[0] + nr + 0.05, nc[1]), (7.0, nc[1]))
    ax.text(7.3, nc[1], f"${output_label}$", ha='left', va='center', fontsize=18, fontweight='bold')

    ax.set_title(title, fontsize=17, fontweight='bold', pad=10)


# ═══════════════════════════════════════════════════════════════════════════════
# 5-7. Gate AND, OR, NOT
# ═══════════════════════════════════════════════════════════════════════════════
def make_gate_and():
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_gate_neuron(ax, [("x_1", 3.0), ("x_2", 1.0)], ["1", "1"], "-1.5", "AND Gate as Neuron")
    save(fig, "gate_and")

def make_gate_or():
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_gate_neuron(ax, [("x_1", 3.0), ("x_2", 1.0)], ["1", "1"], "-0.5", "OR Gate as Neuron")
    save(fig, "gate_or")

def make_gate_not():
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_gate_neuron(ax, [("x_1", 2.0)], ["-1"], "0.5", "NOT Gate as Neuron")
    save(fig, "gate_not")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. gate_nand_approach1
# ═══════════════════════════════════════════════════════════════════════════════
def make_gate_nand_approach1():
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_gate_neuron(ax, [("x_1", 3.0), ("x_2", 1.0)], ["-1", "-1"], "1.5",
                     "NAND Gate (Direct Approach)")
    save(fig, "gate_nand_approach1")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. gate_nand_approach2  – NAND as AND -> NOT composition
# ═══════════════════════════════════════════════════════════════════════════════
def make_gate_nand_approach2():
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 2]})

    # Top: logic gate version
    ax = axes[0]
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, 3)
    ax.axis('off')
    ax.set_title("NAND = AND followed by NOT (Logic Gate View)", fontsize=15, fontweight='bold')

    # x1, x2
    ax.text(1, 2, "$x_1$", ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(1, 1, "$x_2$", ha='center', va='center', fontsize=16, fontweight='bold')
    draw_arrow(ax, (1.5, 2), (3.0, 1.5))
    draw_arrow(ax, (1.5, 1), (3.0, 1.5))
    # AND box
    and_box = FancyBboxPatch((3.0, 0.8), 2, 1.4, boxstyle="round,pad=0.1",
                              fc=LIGHT_BLUE, ec=DARK, lw=2)
    ax.add_patch(and_box)
    ax.text(4, 1.5, "AND", ha='center', va='center', fontsize=14, fontweight='bold')
    # Arrow
    draw_arrow(ax, (5.2, 1.5), (6.5, 1.5))
    # NOT box
    not_box = FancyBboxPatch((6.5, 0.8), 2, 1.4, boxstyle="round,pad=0.1",
                              fc=LIGHT_GREEN, ec=DARK, lw=2)
    ax.add_patch(not_box)
    ax.text(7.5, 1.5, "NOT", ha='center', va='center', fontsize=14, fontweight='bold')
    draw_arrow(ax, (8.7, 1.5), (10, 1.5))
    ax.text(10.5, 1.5, r"$\hat{y}$", ha='center', va='center', fontsize=18, fontweight='bold')

    # Bottom: neuron version
    ax2 = axes[1]
    ax2.set_xlim(-0.5, 14)
    ax2.set_ylim(-1, 6)
    ax2.axis('off')
    ax2.set_title("NAND = AND followed by NOT (Neuron View)", fontsize=15, fontweight='bold')

    # x1, x2
    ax2.text(0.5, 4, "$x_1$", ha='center', va='center', fontsize=16, fontweight='bold')
    ax2.text(0.5, 1.5, "$x_2$", ha='center', va='center', fontsize=16, fontweight='bold')

    # AND neuron (dashed box)
    and_dashed = FancyBboxPatch((2.5, 0.3), 4, 5, boxstyle="round,pad=0.2",
                                 fc='none', ec=BLUE, lw=2, linestyle='dashed')
    ax2.add_patch(and_dashed)
    ax2.text(4.5, 5.5, "AND", ha='center', va='center', fontsize=13, fontweight='bold', color=BLUE)

    nc1 = (4.5, 2.8)
    draw_split_neuron(ax2, nc1, 0.5)

    # Bias for AND
    ax2.text(4.5, 4.8, "1", ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.15', fc=LIGHT_GRAY, ec=DARK))
    draw_arrow(ax2, (4.5, 4.4), (4.5, nc1[1] + 0.55))
    ax2.text(5.5, 3.9, "$b=-1.5$", ha='center', va='center', fontsize=11, color=RED, fontweight='bold')

    # Inputs to AND
    draw_arrow(ax2, (1.0, 4), (nc1[0] - 0.55, nc1[1] + 0.2))
    ax2.text(2.0, 3.9, "$w_1=1$", fontsize=11, color=RED, fontweight='bold')
    draw_arrow(ax2, (1.0, 1.5), (nc1[0] - 0.55, nc1[1] - 0.2))
    ax2.text(2.0, 1.8, "$w_2=1$", fontsize=11, color=RED, fontweight='bold')

    # NOT neuron (dashed box)
    not_dashed = FancyBboxPatch((7.5, 0.3), 4, 5, boxstyle="round,pad=0.2",
                                 fc='none', ec=GREEN, lw=2, linestyle='dashed')
    ax2.add_patch(not_dashed)
    ax2.text(9.5, 5.5, "NOT", ha='center', va='center', fontsize=13, fontweight='bold', color=GREEN)

    nc2 = (9.5, 2.8)
    draw_split_neuron(ax2, nc2, 0.5)

    # Arrow from AND to NOT
    draw_arrow(ax2, (nc1[0] + 0.55, nc1[1]), (nc2[0] - 0.55, nc2[1]))
    ax2.text(7.0, 3.2, "$w=-1$", fontsize=11, color=RED, fontweight='bold')

    # Bias for NOT
    ax2.text(9.5, 4.8, "1", ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.15', fc=LIGHT_GRAY, ec=DARK))
    draw_arrow(ax2, (9.5, 4.4), (9.5, nc2[1] + 0.55))
    ax2.text(10.6, 3.9, "$b=0.5$", ha='center', va='center', fontsize=11, color=RED, fontweight='bold')

    # Output
    draw_arrow(ax2, (nc2[0] + 0.55, nc2[1]), (12.5, nc2[1]))
    ax2.text(13.0, nc2[1], r"$\hat{y}$", ha='center', va='center', fontsize=18, fontweight='bold')

    fig.tight_layout()
    save(fig, "gate_nand_approach2")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. decision_boundary_and
# ═══════════════════════════════════════════════════════════════════════════════
def plot_decision_boundary(ax, points_green, points_red, boundary_fn=None,
                           title="", boundary_label="", xlim=(-0.3, 2), ylim=(-0.3, 2)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    for (x, y) in points_red:
        ax.plot(x, y, 'o', color=RED, markersize=14, markeredgecolor=DARK, markeredgewidth=1.5, zorder=5)
        ax.text(x + 0.1, y + 0.1, f"({x},{y})", fontsize=10, color='gray')
    for (x, y) in points_green:
        ax.plot(x, y, 'o', color=GREEN, markersize=14, markeredgecolor=DARK, markeredgewidth=1.5, zorder=5)
        ax.text(x + 0.1, y + 0.1, f"({x},{y})", fontsize=10, color='gray')

    if boundary_fn is not None:
        x_line = np.linspace(-0.5, 2.5, 100)
        y_line = boundary_fn(x_line)
        ax.plot(x_line, y_line, '--', color=BLUE, lw=2.5, label=boundary_label)
        if boundary_label:
            ax.legend(fontsize=11, loc='upper right')


def make_decision_boundary_and():
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_decision_boundary(ax,
        points_green=[(1, 1)],
        points_red=[(0, 0), (0, 1), (1, 0)],
        boundary_fn=lambda x: 1.5 - x,
        title="AND Gate Decision Boundary",
        boundary_label="$x_1 + x_2 = 1.5$")
    ax.text(1.5, 1.5, r"$\hat{y}=1$", fontsize=14, color=GREEN, fontweight='bold')
    ax.text(0.2, 0.2, r"$\hat{y}=0$", fontsize=14, color=RED, fontweight='bold')
    fig.tight_layout()
    save(fig, "decision_boundary_and")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. decision_boundary_and_or
# ═══════════════════════════════════════════════════════════════════════════════
def make_decision_boundary_and_or():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    plot_decision_boundary(ax1,
        points_green=[(1, 1)],
        points_red=[(0, 0), (0, 1), (1, 0)],
        boundary_fn=lambda x: 1.5 - x,
        title="AND Gate",
        boundary_label="$x_1 + x_2 = 1.5$")

    plot_decision_boundary(ax2,
        points_green=[(0, 1), (1, 0), (1, 1)],
        points_red=[(0, 0)],
        boundary_fn=lambda x: 0.5 - x,
        title="OR Gate",
        boundary_label="$x_1 + x_2 = 0.5$")

    fig.tight_layout()
    save(fig, "decision_boundary_and_or")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. decision_boundary_all  – AND + OR + XOR
# ═══════════════════════════════════════════════════════════════════════════════
def make_decision_boundary_all():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.5))

    plot_decision_boundary(ax1,
        points_green=[(1, 1)],
        points_red=[(0, 0), (0, 1), (1, 0)],
        boundary_fn=lambda x: 1.5 - x,
        title="AND Gate",
        boundary_label="$x_1 + x_2 = 1.5$")
    ax1.text(0.5, -0.15, "LINEARLY SEPARABLE", ha='center', fontsize=11,
             fontweight='bold', color=GREEN,
             bbox=dict(boxstyle='round,pad=0.3', fc=LIGHT_GREEN, ec=GREEN))

    plot_decision_boundary(ax2,
        points_green=[(0, 1), (1, 0), (1, 1)],
        points_red=[(0, 0)],
        boundary_fn=lambda x: 0.5 - x,
        title="OR Gate",
        boundary_label="$x_1 + x_2 = 0.5$")
    ax2.text(0.5, -0.15, "LINEARLY SEPARABLE", ha='center', fontsize=11,
             fontweight='bold', color=GREEN,
             bbox=dict(boxstyle='round,pad=0.3', fc=LIGHT_GREEN, ec=GREEN))

    plot_decision_boundary(ax3,
        points_green=[(0, 1), (1, 0)],
        points_red=[(0, 0), (1, 1)],
        title="XOR Gate")
    ax3.text(0.5, -0.15, "NOT SEPARABLE LINEARLY", ha='center', fontsize=11,
             fontweight='bold', color=RED,
             bbox=dict(boxstyle='round,pad=0.3', fc=LIGHT_PINK, ec=RED))

    fig.tight_layout()
    save(fig, "decision_boundary_all")


# ═══════════════════════════════════════════════════════════════════════════════
# 13. xor_classification – XOR with decision surface
# ═══════════════════════════════════════════════════════════════════════════════
def make_xor_classification():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.5, 1.8)
    ax.set_ylim(-0.5, 1.8)
    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$", fontsize=14)
    ax.set_title("XOR: Non-Linear Decision Surface", fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Shaded regions
    # Two lines: x2 = x1 + 0.5 and x2 = x1 - 0.5 (diagonal band)
    x_fill = np.linspace(-0.5, 2, 300)

    # Fill the diagonal band green (where XOR=1)
    ax.fill_between(x_fill, x_fill - 0.5, x_fill + 0.5,
                    alpha=0.15, color=GREEN, zorder=1)
    # Fill outside the band red
    ax.fill_between(x_fill, -0.5, x_fill - 0.5, alpha=0.12, color=RED, zorder=1)
    ax.fill_between(x_fill, x_fill + 0.5, 2.0, alpha=0.12, color=RED, zorder=1)

    # Boundary lines
    ax.plot(x_fill, x_fill + 0.5, '--', color=BLUE, lw=2.5, label="Boundary 1")
    ax.plot(x_fill, x_fill - 0.5, '--', color=BLUE, lw=2.5, label="Boundary 2")

    # Points
    for (x, y), c in [((0, 0), RED), ((1, 1), RED), ((0, 1), GREEN), ((1, 0), GREEN)]:
        ax.plot(x, y, 'o', color=c, markersize=16, markeredgecolor=DARK, markeredgewidth=2, zorder=5)
        ax.text(x + 0.08, y + 0.08, f"({x},{y})", fontsize=10, color='gray')

    ax.legend(fontsize=11)
    fig.tight_layout()
    save(fig, "xor_classification")


# ═══════════════════════════════════════════════════════════════════════════════
# 14. xor_neuron – XOR with hand-crafted x1*x2 feature
# ═══════════════════════════════════════════════════════════════════════════════
def make_xor_neuron():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(-1, 9)
    ax.set_ylim(-1.5, 6)
    ax.axis('off')

    nc = (5, 2.5)
    nr = 0.5

    draw_split_neuron(ax, nc, nr)

    # Inputs: 1 (bias), x1, x2, x1*x2
    inputs = [
        ("1", 5.0, "$b = -0.5$"),
        ("$x_1$", 4.0, "$w = 1$"),
        ("$x_2$", 2.5, "$w = 1$"),
        ("$x_1 x_2$", 1.0, "$w = -2$"),
    ]

    # Bias from top
    ax.text(nc[0], 5.3, "1", ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc=LIGHT_GRAY, ec=DARK))
    draw_arrow(ax, (nc[0], 4.9), (nc[0], nc[1] + nr + 0.05))
    ax.text(nc[0] + 1.0, 4.0, "$b = -0.5$", ha='center', va='center', fontsize=13,
            fontweight='bold', color=RED)

    # x1, x2, x1*x2
    x_inputs = [("$x_1$", 4.0, "$w = 1$"), ("$x_2$", 2.5, "$w = 1$"), ("$x_1 x_2$", 1.0, "$w = -2$")]
    for label, y, w in x_inputs:
        ax.text(0.5, y, label, ha='center', va='center', fontsize=16, fontweight='bold')
        target_y = nc[1] + (y - nc[1]) * 0.35
        draw_arrow(ax, (1.3, y), (nc[0] - nr - 0.05, target_y))
        mid_x = (1.3 + nc[0] - nr) / 2
        mid_y = (y + target_y) / 2
        ax.text(mid_x, mid_y + 0.3, w, ha='center', va='center', fontsize=13,
                fontweight='bold', color=RED)

    # Output
    draw_arrow(ax, (nc[0] + nr + 0.05, nc[1]), (7.5, nc[1]))
    ax.text(8.0, nc[1], "output", ha='left', va='center', fontsize=16, fontweight='bold')

    ax.set_title("XOR with Hand-Crafted Feature $x_1 x_2$", fontsize=18, fontweight='bold')
    fig.tight_layout()
    save(fig, "xor_neuron")


# ═══════════════════════════════════════════════════════════════════════════════
# 15. backprop_activations  – neuron with z and ? activation
# ═══════════════════════════════════════════════════════════════════════════════
def make_backprop_activations():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-1, 10)
    ax.set_ylim(-2, 6)
    ax.axis('off')

    nc = (5, 2.5)
    nr = 0.6

    # Split neuron: left = z formula, right = ?
    left = Wedge(nc, nr, 90, 270, fc=LIGHT_PINK, ec=NEURON_EDGE, lw=2, zorder=5)
    ax.add_patch(left)
    right = Wedge(nc, nr, 270, 90, fc='#FFFFAA', ec=NEURON_EDGE, lw=2, zorder=5)
    ax.add_patch(right)
    ax.plot([nc[0], nc[0]], [nc[1] - nr, nc[1] + nr], color=NEURON_EDGE, lw=2, zorder=6)
    ax.text(nc[0] - nr * 0.45, nc[1], "$z$", ha='center', va='center', fontsize=15, fontweight='bold', zorder=7)
    ax.text(nc[0] + nr * 0.45, nc[1], "?", ha='center', va='center', fontsize=18, fontweight='bold',
            color=RED, zorder=7)

    # Inputs
    ax.text(nc[0], 5.3, "1", ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc=LIGHT_GRAY, ec=DARK))
    draw_arrow(ax, (nc[0], 4.9), (nc[0], nc[1] + nr + 0.05))
    ax.text(nc[0] + 0.7, 4.2, "$b$", ha='center', va='center', fontsize=14, color=BLUE, fontweight='bold')

    for label, y, w in [("$x_1$", 4.0, "$w_1$"), ("$x_2$", 2.5, "$...$"), ("$x_d$", 1.0, "$w_d$")]:
        ax.text(0.5, y, label, ha='center', va='center', fontsize=16, fontweight='bold')
        target_y = nc[1] + (y - nc[1]) * 0.35
        draw_arrow(ax, (1.1, y), (nc[0] - nr - 0.05, target_y))
        mid_x = (1.1 + nc[0] - nr) / 2
        mid_y = (y + target_y) / 2
        ax.text(mid_x - 0.2, mid_y + 0.3, w, ha='center', va='center', fontsize=13, color=BLUE, fontweight='bold')

    # Output
    draw_arrow(ax, (nc[0] + nr + 0.05, nc[1]), (8.0, nc[1]))
    ax.text(8.5, nc[1], r"$\hat{y}$", ha='center', va='center', fontsize=20, fontweight='bold')

    # z formula
    ax.text(2.5, -0.8, r"$z = \sum w_i x_i + b$", ha='center', va='center', fontsize=15,
            fontweight='bold', color=DARK)

    # Key idea box
    ax.text(5, -1.5, "Key idea: Use activation similar to step but differentiable",
            ha='center', va='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc='#FFF8E0', ec='orange', lw=2))

    ax.set_title("Neuron with Unknown Activation", fontsize=17, fontweight='bold')
    fig.tight_layout()
    save(fig, "backprop_activations")


# ═══════════════════════════════════════════════════════════════════════════════
# 16. adding_nonlinearity – neuron with z and g(z)
# ═══════════════════════════════════════════════════════════════════════════════
def make_adding_nonlinearity():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-1, 10)
    ax.set_ylim(-2, 6)
    ax.axis('off')

    nc = (5, 2.5)
    nr = 0.6

    # Split neuron: left = z, right = g(z)
    left = Wedge(nc, nr, 90, 270, fc=LIGHT_PINK, ec=NEURON_EDGE, lw=2, zorder=5)
    ax.add_patch(left)
    right = Wedge(nc, nr, 270, 90, fc=LIGHT_GREEN, ec=NEURON_EDGE, lw=2, zorder=5)
    ax.add_patch(right)
    ax.plot([nc[0], nc[0]], [nc[1] - nr, nc[1] + nr], color=NEURON_EDGE, lw=2, zorder=6)
    ax.text(nc[0] - nr * 0.45, nc[1], "$z$", ha='center', va='center', fontsize=14, fontweight='bold', zorder=7)
    ax.text(nc[0] + nr * 0.42, nc[1], "$g(z)$", ha='center', va='center', fontsize=10, fontweight='bold', zorder=7)

    # Inputs
    ax.text(nc[0], 5.3, "1", ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc=LIGHT_GRAY, ec=DARK))
    draw_arrow(ax, (nc[0], 4.9), (nc[0], nc[1] + nr + 0.05))
    ax.text(nc[0] + 0.7, 4.2, "$b$", ha='center', va='center', fontsize=14, color=BLUE, fontweight='bold')

    for label, y, w in [("$x_1$", 4.0, "$w_1$"), ("$x_2$", 2.5, "$...$"), ("$x_d$", 1.0, "$w_d$")]:
        ax.text(0.5, y, label, ha='center', va='center', fontsize=16, fontweight='bold')
        target_y = nc[1] + (y - nc[1]) * 0.35
        draw_arrow(ax, (1.1, y), (nc[0] - nr - 0.05, target_y))
        mid_x = (1.1 + nc[0] - nr) / 2
        mid_y = (y + target_y) / 2
        ax.text(mid_x - 0.2, mid_y + 0.3, w, ha='center', va='center', fontsize=13, color=BLUE, fontweight='bold')

    # Output with a = g(z)
    draw_arrow(ax, (nc[0] + nr + 0.05, nc[1]), (8.0, nc[1]))
    ax.text(8.5, nc[1], r"$\hat{y}$", ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(7.0, nc[1] + 0.5, "$a = g(z)$", ha='center', va='center', fontsize=13, color=GREEN, fontweight='bold')

    # Note below
    ax.text(5, -1.2, "$g(z)$: non-linear transformation", ha='center', va='center',
            fontsize=15, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc=LIGHT_GREEN, ec=GREEN, lw=2))

    ax.set_title("Adding Non-Linearity to the Neuron", fontsize=17, fontweight='bold')
    fig.tight_layout()
    save(fig, "adding_nonlinearity")


# ═══════════════════════════════════════════════════════════════════════════════
# Activation function helpers
# ═══════════════════════════════════════════════════════════════════════════════
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.1):
    return np.where(z >= 0, z, alpha * z)

def plot_activation(ax, z, vals, title, formula, color=BLUE, ylim=None, note=None):
    ax.plot(z, vals, color=color, lw=3)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel("$z$", fontsize=13)
    ax.set_ylabel("$g(z)$", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # formula
    ax.text(0.5, 0.05, formula, ha='center', va='bottom', fontsize=12,
            transform=ax.transAxes, color=DARK,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.9))
    if note:
        ax.text(0.5, -0.22, note, ha='center', va='top', fontsize=10,
                transform=ax.transAxes, color=DARK, style='italic',
                bbox=dict(boxstyle='round,pad=0.2', fc='#FFF8E0', ec='orange', alpha=0.9))


# ═══════════════════════════════════════════════════════════════════════════════
# 17. activation_sigmoid
# ═══════════════════════════════════════════════════════════════════════════════
def make_activation_sigmoid():
    fig, ax = plt.subplots(figsize=(7, 5))
    z = np.linspace(-6, 6, 300)
    ax.plot(z, sigmoid(z), color=BLUE, lw=3)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.axhline(1, color='gray', lw=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel("$z$", fontsize=14)
    ax.set_ylabel("$g(z)$", fontsize=14)
    ax.set_title("Sigmoid Activation", fontsize=16, fontweight='bold')
    ax.set_ylim(-0.1, 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.5, 0.05, r"$g(z) = \frac{1}{1 + e^{-z}}$", ha='center', va='bottom',
            fontsize=15, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray'))
    fig.tight_layout()
    save(fig, "activation_sigmoid")


# ═══════════════════════════════════════════════════════════════════════════════
# 18. activation_sigmoid_tanh
# ═══════════════════════════════════════════════════════════════════════════════
def make_activation_sigmoid_tanh():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    z = np.linspace(-6, 6, 300)

    plot_activation(ax1, z, sigmoid(z), "Sigmoid", r"$g(z) = \frac{1}{1+e^{-z}}$",
                    color=BLUE, ylim=(-0.1, 1.2))
    plot_activation(ax2, z, tanh(z), "Tanh", r"$g(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$",
                    color='purple', ylim=(-1.3, 1.3))
    fig.suptitle("Activation Functions", fontsize=17, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "activation_sigmoid_tanh")


# ═══════════════════════════════════════════════════════════════════════════════
# 19. activation_all_three
# ═══════════════════════════════════════════════════════════════════════════════
def make_activation_all_three():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    z = np.linspace(-6, 6, 300)

    plot_activation(ax1, z, sigmoid(z), "Sigmoid", r"$g(z) = \frac{1}{1+e^{-z}}$",
                    color=BLUE, ylim=(-0.1, 1.2))
    plot_activation(ax2, z, tanh(z), "Tanh", r"$g(z) = \tanh(z)$",
                    color='purple', ylim=(-1.3, 1.3))
    z_relu = np.linspace(-3, 6, 300)
    plot_activation(ax3, z_relu, relu(z_relu), "ReLU",
                    r"$g(z) = \max(0, z)$", color=GREEN, ylim=(-1, 6.5))

    fig.suptitle("Activation Functions", fontsize=17, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "activation_all_three")


# ═══════════════════════════════════════════════════════════════════════════════
# 20. activation_all_four
# ═══════════════════════════════════════════════════════════════════════════════
def make_activation_all_four():
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    z = np.linspace(-6, 6, 300)
    z_relu = np.linspace(-3, 6, 300)

    plot_activation(axes[0], z, sigmoid(z), "Sigmoid", r"$g(z) = \frac{1}{1+e^{-z}}$",
                    color=BLUE, ylim=(-0.1, 1.2))
    plot_activation(axes[1], z, tanh(z), "Tanh", r"$g(z) = \tanh(z)$",
                    color='purple', ylim=(-1.3, 1.3))
    plot_activation(axes[2], z_relu, relu(z_relu), "ReLU",
                    r"$g(z) = \max(0, z)$", color=GREEN, ylim=(-1, 6.5))
    plot_activation(axes[3], z_relu, leaky_relu(z_relu), "Leaky ReLU",
                    r"$g(z) = \max(\alpha z, z),\ \alpha \to 0$", color='darkorange', ylim=(-1, 6.5))

    fig.suptitle("Activation Functions", fontsize=17, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "activation_all_four")


# ═══════════════════════════════════════════════════════════════════════════════
# 21. activation_all_four_notes
# ═══════════════════════════════════════════════════════════════════════════════
def make_activation_all_four_notes():
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    z = np.linspace(-6, 6, 300)
    z_relu = np.linspace(-3, 6, 300)

    notes = [
        "Useful for probabilistic\nestimates (btw 0 & 1)",
        "Useful if data transformed\nwith mean 0",
        "Game changer (Default).\nGood learning for |z| = high.",
        "Similar to ReLU.\nLearns for z < 0 also",
    ]

    plot_activation(axes[0], z, sigmoid(z), "Sigmoid", r"$g(z) = \frac{1}{1+e^{-z}}$",
                    color=BLUE, ylim=(-0.1, 1.2), note=notes[0])
    plot_activation(axes[1], z, tanh(z), "Tanh", r"$g(z) = \tanh(z)$",
                    color='purple', ylim=(-1.3, 1.3), note=notes[1])
    plot_activation(axes[2], z_relu, relu(z_relu), "ReLU",
                    r"$g(z) = \max(0, z)$", color=GREEN, ylim=(-1, 6.5), note=notes[2])
    plot_activation(axes[3], z_relu, leaky_relu(z_relu), "Leaky ReLU",
                    r"$g(z) = \max(\alpha z, z)$", color='darkorange', ylim=(-1, 6.5), note=notes[3])

    fig.suptitle("Activation Functions with Usage Notes", fontsize=17, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.subplots_adjust(bottom=0.18)
    save(fig, "activation_all_four_notes")


# ═══════════════════════════════════════════════════════════════════════════════
# 22. step_function
# ═══════════════════════════════════════════════════════════════════════════════
def make_step_function():
    fig, ax = plt.subplots(figsize=(7, 5))
    z = np.linspace(-4, 4, 300)
    step = np.where(z >= 0, 1, 0)
    ax.plot(z, step, color=BLUE, lw=3)
    # Dots at discontinuity
    ax.plot(0, 1, 'o', color=BLUE, markersize=8, zorder=5)
    ax.plot(0, 0, 'o', color='white', markersize=8, markeredgecolor=BLUE, markeredgewidth=2, zorder=5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel("$z$", fontsize=14)
    ax.set_ylabel("$g(z)$", fontsize=14)
    ax.set_ylim(-0.2, 1.4)
    ax.set_title("(Sign/Step) Activation", fontsize=16, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.5, 0.08, "$g(z) = 1$  if  $z \\geq 0$,   $g(z) = 0$  if  $z < 0$",
            ha='center', va='bottom', fontsize=14, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray'))
    fig.tight_layout()
    save(fig, "step_function")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating MLP diagrams...")
    make_paradigm_old()
    make_paradigm_new()
    make_perceptron_simple()
    make_perceptron_components()
    make_gate_and()
    make_gate_or()
    make_gate_not()
    make_gate_nand_approach1()
    make_gate_nand_approach2()
    make_decision_boundary_and()
    make_decision_boundary_and_or()
    make_decision_boundary_all()
    make_xor_classification()
    make_xor_neuron()
    make_backprop_activations()
    make_adding_nonlinearity()
    make_activation_sigmoid()
    make_activation_sigmoid_tanh()
    make_activation_all_three()
    make_activation_all_four()
    make_activation_all_four_notes()
    make_step_function()
    print("\nDone! All 22 diagrams generated.")
