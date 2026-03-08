#!/usr/bin/env python3
"""
Generate backward pass / autograd diagrams for Lecture 05 - Neural Networks.

Diagrams produced:
  1. backward_f9            - Backward through f9 (×-1) node
  2. backward_f8_log        - Backward through f8 (log) node
  3. backward_f7_inv        - Backward through f7 (1/x) node
  4. backward_f6_plus1      - Backward through f6 (+1) node
  5. backward_f5_exp        - Backward through f5 (exp) node
  6. backward_f4_neg        - Backward through f4 (×-1) node
  7. backward_f3_plus_theta0 - Backward through f3 (+θ0) node
  8. backward_f1_theta1     - Backward: gradient for θ1
  9. backward_f2_theta2     - Backward: gradient for θ2
  10. autodiff_local_grads   - What autodiff library needs to know
  11. simplified_comp_graph  - Simplified with sigmoid node
  12. simplified_comp_graph_sigmoid_grad - With σ'(z) formula
  13. simplified_nn_linear   - nn.Linear → C.E. Loss
  14. training_n_examples_1  - Training over N examples (single)
  15. training_n_examples_2  - Training over N examples (two, shared weights)

All saved as SVG (diagrams/svg/) and PNG 300 DPI (diagrams/png/).
"""

import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ---------- paths -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SVG_DIR = os.path.join(BASE_DIR, "diagrams", "svg")
PNG_DIR = os.path.join(BASE_DIR, "diagrams", "png")
os.makedirs(SVG_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

# ---------- style constants -------------------------------------------------
BG_COLOR = "white"
NODE_RADIUS = 0.30
NODE_COLOR = "#E8E8E8"
NODE_EDGE_COLOR = "#444444"
ARROW_COLOR = "#333333"
FWD_COLOR = "#2e7d32"      # green for forward values
BWD_COLOR = "#c62828"      # red for backward gradients
UPSTREAM_COLOR = "#c62828"
LOCAL_COLOR = "#ef6c00"
LABEL_FONT = 13
TITLE_FONT = 14
SMALL_FONT = 11
FORMULA_FONT = 12


def save(fig, name):
    """Save figure as SVG and PNG."""
    fig.savefig(os.path.join(SVG_DIR, f"{name}.svg"), bbox_inches="tight",
                facecolor=BG_COLOR)
    fig.savefig(os.path.join(PNG_DIR, f"{name}.png"), bbox_inches="tight",
                dpi=300, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  saved {name}")


# ============================================================================
# Helper drawing primitives
# ============================================================================

def draw_circle_node(ax, x, y, label, radius=NODE_RADIUS,
                     fc=NODE_COLOR, ec=NODE_EDGE_COLOR, fontsize=LABEL_FONT,
                     text_color="black", zorder=3):
    c = plt.Circle((x, y), radius, fc=fc, ec=ec, lw=1.8, zorder=zorder)
    ax.add_patch(c)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=zorder + 1)
    return (x, y)


def draw_arrow(ax, x0, y0, x1, y1, color=ARROW_COLOR, lw=1.5,
               shrinkA=0, shrinkB=0, style="->", zorder=2):
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, mutation_scale=14,
        color=color, lw=lw, shrinkA=shrinkA, shrinkB=shrinkB,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def arrow_node_to_node(ax, cx1, cy1, cx2, cy2, radius=NODE_RADIUS, **kw):
    dx, dy = cx2 - cx1, cy2 - cy1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    sx = cx1 + dx / dist * radius
    sy = cy1 + dy / dist * radius
    ex = cx2 - dx / dist * radius
    ey = cy2 - dy / dist * radius
    draw_arrow(ax, sx, sy, ex, ey, **kw)


def arrow_to_node(ax, x0, y0, cx, cy, radius=NODE_RADIUS, **kw):
    dx, dy = cx - x0, cy - y0
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    ex = cx - dx / dist * radius
    ey = cy - dy / dist * radius
    sx = x0 + dx / dist * 0.04
    sy = y0 + dy / dist * 0.04
    draw_arrow(ax, sx, sy, ex, ey, **kw)


def arrow_from_node(ax, cx, cy, x1, y1, radius=NODE_RADIUS, **kw):
    dx, dy = x1 - cx, y1 - cy
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    sx = cx + dx / dist * radius
    sy = cy + dy / dist * radius
    ex = x1 - dx / dist * 0.04
    ey = y1 - dy / dist * 0.04
    draw_arrow(ax, sx, sy, ex, ey, **kw)


# ============================================================================
# Generic backward-pass node diagram
# ============================================================================

def make_backward_node_diagram(
    title,
    node_label,
    fwd_in_val,
    fwd_out_val,
    upstream_val,
    local_grad_str,
    local_grad_val,
    downstream_val,
    formula_lines=None,
    in_label=None,
    out_label=None,
    extra_text=None,
    figsize=(10, 5),
):
    """Create a backward pass diagram for a single node.

    Parameters
    ----------
    title : str - slide title
    node_label : str - operation label inside circle
    fwd_in_val : str - forward value entering node
    fwd_out_val : str - forward value leaving node
    upstream_val : str - upstream gradient value
    local_grad_str : str - formula for local gradient
    local_grad_val : str - numeric value of local gradient
    downstream_val : str - computed downstream gradient
    formula_lines : list[str] or None - extra formula lines below
    in_label : str or None - label for input edge (e.g., "f5")
    out_label : str or None - label for output edge (e.g., "f6")
    extra_text : list[tuple(x, y, str, dict)] - additional annotations
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-3, 9)
    ax.set_ylim(-3.5, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    # Node
    nx, ny = 3, 0.5
    draw_circle_node(ax, nx, ny, node_label, fontsize=16)

    # Forward arrows (green)
    # Input arrow
    draw_arrow(ax, 0.5, ny, nx - NODE_RADIUS, ny, color=FWD_COLOR, lw=2)
    ax.text(0.5, ny + 0.35, fwd_in_val, fontsize=SMALL_FONT, ha="left",
            va="bottom", color=FWD_COLOR, fontweight="bold")
    if in_label:
        ax.text(0.3, ny - 0.35, in_label, fontsize=SMALL_FONT, ha="left",
                va="top", color="#555")

    # Output arrow
    draw_arrow(ax, nx + NODE_RADIUS, ny, 5.5, ny, color=FWD_COLOR, lw=2)
    ax.text(4.2, ny + 0.35, fwd_out_val, fontsize=SMALL_FONT, ha="left",
            va="bottom", color=FWD_COLOR, fontweight="bold")
    if out_label:
        ax.text(5.6, ny, out_label, fontsize=SMALL_FONT, ha="left",
                va="center", color="#555")

    # Backward arrows (red)
    # Upstream (from right)
    draw_arrow(ax, 5.5, ny - 0.6, nx + NODE_RADIUS + 0.1, ny - 0.6,
               color=BWD_COLOR, lw=2.5)
    ax.text(4.5, ny - 0.95, upstream_val, fontsize=SMALL_FONT, ha="center",
            va="top", color=BWD_COLOR, fontweight="bold")
    ax.text(5.8, ny - 0.6, "upstream", fontsize=9, ha="left",
            va="center", color=BWD_COLOR, fontstyle="italic")

    # Downstream (to left)
    draw_arrow(ax, nx - NODE_RADIUS - 0.1, ny - 0.6, 0.5, ny - 0.6,
               color=BWD_COLOR, lw=2.5)
    ax.text(1.3, ny - 0.95, downstream_val, fontsize=SMALL_FONT, ha="center",
            va="top", color=BWD_COLOR, fontweight="bold")

    # Formulas below
    y_formula = -1.8
    if local_grad_str:
        ax.text(3, y_formula, f"Local gradient = {local_grad_str} = {local_grad_val}",
                fontsize=FORMULA_FONT, ha="center", va="top", color=LOCAL_COLOR,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff3e0", ec=LOCAL_COLOR,
                          alpha=0.9))

    if formula_lines:
        for i, line in enumerate(formula_lines):
            ax.text(3, y_formula - 0.7 - i * 0.6, line,
                    fontsize=FORMULA_FONT, ha="center", va="top", color="#333")

    if extra_text:
        for (ex, ey, etxt, ekw) in extra_text:
            ax.text(ex, ey, etxt, **ekw)

    ax.set_title(title, fontsize=TITLE_FONT + 2, pad=12, fontweight="bold")
    return fig


# ============================================================================
# Individual backward pass diagrams
# ============================================================================

def backward_f9():
    """f9: × (-1), L = f9 * -1"""
    return make_backward_node_diagram(
        title="Backward Pass — f9 (× −1)",
        node_label=r"$\times$",
        fwd_in_val="−0.00247",
        fwd_out_val="L = 0.00247",
        upstream_val="1.0",
        local_grad_str=r"$\partial L / \partial f_9$",
        local_grad_val="−1",
        downstream_val="1.0 × (−1) = −1.0",
        formula_lines=[r"$L = f_9 \times (-1)$"],
        in_label="f9",
        out_label="L",
    )


def backward_f8():
    """f8: log, f9 = log(f8)"""
    return make_backward_node_diagram(
        title="Backward Pass — f8 (log)",
        node_label="log",
        fwd_in_val="0.9975",
        fwd_out_val="−0.00247",
        upstream_val="−1.0",
        local_grad_str=r"$1/f_8 = 1/0.9975$",
        local_grad_val="1.00247",
        downstream_val="−1.0 × 1.00247 = −1.00247",
        formula_lines=[r"$f_9 = \log(f_8)$",
                        r"$\partial f_9 / \partial f_8 = 1/f_8$"],
        in_label="f8",
        out_label="f9",
    )


def backward_f7():
    """f7: 1/x, f8 = 1/f7"""
    return make_backward_node_diagram(
        title="Backward Pass — f7 (1/x)",
        node_label=r"$\frac{1}{x}$",
        fwd_in_val="1.00247",
        fwd_out_val="0.9975",
        upstream_val="−1.00247",
        local_grad_str=r"$-1/f_7^2$",
        local_grad_val="−0.9951",
        downstream_val="(−1.00247)(−0.9951) = 0.9975",
        formula_lines=[r"$f_8 = 1/f_7$",
                        r"$\partial f_8 / \partial f_7 = -1/f_7^2$"],
        in_label="f7",
        out_label="f8",
    )


def backward_f6():
    """f6: +1, f7 = f6 + 1"""
    return make_backward_node_diagram(
        title="Backward Pass — f6 (+1)",
        node_label="+",
        fwd_in_val="0.00247",
        fwd_out_val="1.00247",
        upstream_val="0.9975",
        local_grad_str=r"$\partial f_7 / \partial f_6$",
        local_grad_val="1",
        downstream_val="0.9975 × 1 = 0.9975",
        formula_lines=[r"$f_7 = f_6 + 1$"],
        in_label="f6",
        out_label="f7",
    )


def backward_f5():
    """f5: exp, f6 = exp(f5)"""
    return make_backward_node_diagram(
        title="Backward Pass — f5 (exp)",
        node_label="exp",
        fwd_in_val="−6.0",
        fwd_out_val="0.00247",
        upstream_val="0.9975",
        local_grad_str=r"$e^{f_5} = e^{-6}$",
        local_grad_val="0.0025",
        downstream_val="0.9975 × 0.0025 = 0.00247",
        formula_lines=[r"$f_6 = e^{f_5}$",
                        r"$\partial f_6 / \partial f_5 = e^{f_5}$"],
        in_label="f5",
        out_label="f6",
    )


def backward_f4():
    """f4: ×(-1), f5 = f4 × (-1)"""
    return make_backward_node_diagram(
        title="Backward Pass — f4 (× −1)",
        node_label=r"$\times$",
        fwd_in_val="6.0",
        fwd_out_val="−6.0",
        upstream_val="0.00247",
        local_grad_str=r"$\partial f_5 / \partial f_4$",
        local_grad_val="−1",
        downstream_val="0.00247 × (−1) = −0.00247",
        formula_lines=[r"$f_5 = f_4 \times (-1)$"],
        in_label="f4",
        out_label="f5",
    )


def backward_f3():
    """f3: + θ0, f4 = f3 + θ0"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-2, 10)
    ax.set_ylim(-4, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    # Node
    nx, ny = 4, 0.5
    draw_circle_node(ax, nx, ny, "+", fontsize=18)

    # Forward: f3 from left
    draw_arrow(ax, 1, ny, nx - NODE_RADIUS, ny, color=FWD_COLOR, lw=2)
    ax.text(1.2, ny + 0.35, "5.0", fontsize=SMALL_FONT, color=FWD_COLOR,
            fontweight="bold")
    ax.text(0.5, ny, "f3", fontsize=SMALL_FONT, color="#555", ha="right")

    # Forward: θ0 from below
    draw_arrow(ax, nx, -1.2, nx, ny - NODE_RADIUS, color=FWD_COLOR, lw=2)
    ax.text(nx + 0.3, -1, r"$\theta_0 = 1.0$", fontsize=SMALL_FONT,
            color=FWD_COLOR, fontweight="bold")

    # Forward: output
    draw_arrow(ax, nx + NODE_RADIUS, ny, 7, ny, color=FWD_COLOR, lw=2)
    ax.text(5.5, ny + 0.35, "f4 = 6.0", fontsize=SMALL_FONT, color=FWD_COLOR,
            fontweight="bold")

    # Backward: upstream from right
    draw_arrow(ax, 7, ny - 0.6, nx + NODE_RADIUS + 0.1, ny - 0.6,
               color=BWD_COLOR, lw=2.5)
    ax.text(5.5, ny - 1.0, "−0.00247", fontsize=SMALL_FONT, color=BWD_COLOR,
            fontweight="bold", ha="center")

    # Backward: to f3 (left)
    draw_arrow(ax, nx - NODE_RADIUS - 0.1, ny - 0.6, 1, ny - 0.6,
               color=BWD_COLOR, lw=2.5)
    ax.text(2.2, ny - 1.0, "−0.00247", fontsize=SMALL_FONT, color=BWD_COLOR,
            fontweight="bold", ha="center")

    # Backward: to θ0 (down)
    draw_arrow(ax, nx - 0.4, ny - NODE_RADIUS - 0.3, nx - 0.4, -1.2,
               color=BWD_COLOR, lw=2.5)
    ax.text(nx - 0.8, -0.5, "−0.00247", fontsize=SMALL_FONT, color=BWD_COLOR,
            fontweight="bold", ha="right")

    # Highlight θ0
    rect = mpatches.FancyBboxPatch((nx - 0.1, -1.8), 1.5, 0.8,
                                    boxstyle="round,pad=0.1",
                                    fc="#e0f7fa", ec="#00838f", lw=2)
    ax.add_patch(rect)
    ax.text(nx + 0.65, -1.4, r"$\frac{\partial L}{\partial \theta_0} = -0.00247$",
            fontsize=FORMULA_FONT, ha="center", va="center", fontweight="bold",
            color="#00838f")

    # Formula
    ax.text(4, -3, r"Upstream grad = $-0.00247$;   Local grad $(\theta_0) = 1$;   Local grad $(f_3) = 1$",
            fontsize=FORMULA_FONT, ha="center", va="top", color=LOCAL_COLOR,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff3e0", ec=LOCAL_COLOR, alpha=0.9))

    ax.set_title(r"Backward Pass — $f_3$ ($+ \theta_0$)",
                 fontsize=TITLE_FONT + 2, pad=12, fontweight="bold")
    return fig


def backward_theta1():
    """Backward: gradient for θ1 via f1 = θ1 × x1"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-2, 10)
    ax.set_ylim(-4, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    # Node
    nx, ny = 4, 0.5
    draw_circle_node(ax, nx, ny, r"$\times$", fontsize=18)

    # x1 from top
    draw_arrow(ax, nx + 0.8, 2.2, nx + 0.2, ny + NODE_RADIUS, color=FWD_COLOR, lw=2)
    ax.text(nx + 1, 2.2, r"$x_1 = 1.0$", fontsize=SMALL_FONT, color=FWD_COLOR,
            fontweight="bold")

    # θ1 from left-top
    draw_arrow(ax, 1, 2.2, nx - 0.2, ny + NODE_RADIUS, color=FWD_COLOR, lw=2)
    ax.text(0.3, 2.2, r"$\theta_1 = 1.0$", fontsize=SMALL_FONT, color=FWD_COLOR,
            fontweight="bold")

    # Output f1
    draw_arrow(ax, nx + NODE_RADIUS, ny, 7, ny, color=FWD_COLOR, lw=2)
    ax.text(5.5, ny + 0.35, "f1 = 1.0", fontsize=SMALL_FONT, color=FWD_COLOR,
            fontweight="bold")

    # Upstream
    draw_arrow(ax, 7, ny - 0.6, nx + NODE_RADIUS + 0.1, ny - 0.6,
               color=BWD_COLOR, lw=2.5)
    ax.text(5.5, ny - 1.0, "−0.00247", fontsize=SMALL_FONT, color=BWD_COLOR,
            fontweight="bold", ha="center")

    # Downstream to θ1
    draw_arrow(ax, nx - NODE_RADIUS - 0.2, ny - 0.4, 1.2, -0.8,
               color=BWD_COLOR, lw=2.5)

    # Highlight θ1 gradient
    rect = mpatches.FancyBboxPatch((-0.5, -1.6), 3.5, 1.0,
                                    boxstyle="round,pad=0.1",
                                    fc="#e0f7fa", ec="#00838f", lw=2)
    ax.add_patch(rect)
    ax.text(1.25, -1.1, r"$\frac{\partial L}{\partial \theta_1} = -0.00247 \times x_1 = -0.00247$",
            fontsize=FORMULA_FONT, ha="center", va="center", fontweight="bold",
            color="#00838f")

    ax.set_title(r"Backward Pass — $\theta_1$ gradient",
                 fontsize=TITLE_FONT + 2, pad=12, fontweight="bold")
    return fig


def backward_theta2():
    """Backward: gradient for θ2 via f2 = θ2 × x2"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-2, 10)
    ax.set_ylim(-4, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    nx, ny = 4, 0.5
    draw_circle_node(ax, nx, ny, r"$\times$", fontsize=18)

    draw_arrow(ax, nx + 0.8, 2.2, nx + 0.2, ny + NODE_RADIUS, color=FWD_COLOR, lw=2)
    ax.text(nx + 1, 2.2, r"$x_2 = 2.0$", fontsize=SMALL_FONT, color=FWD_COLOR,
            fontweight="bold")

    draw_arrow(ax, 1, 2.2, nx - 0.2, ny + NODE_RADIUS, color=FWD_COLOR, lw=2)
    ax.text(0.3, 2.2, r"$\theta_2 = 2.0$", fontsize=SMALL_FONT, color=FWD_COLOR,
            fontweight="bold")

    draw_arrow(ax, nx + NODE_RADIUS, ny, 7, ny, color=FWD_COLOR, lw=2)
    ax.text(5.5, ny + 0.35, "f2 = 4.0", fontsize=SMALL_FONT, color=FWD_COLOR,
            fontweight="bold")

    draw_arrow(ax, 7, ny - 0.6, nx + NODE_RADIUS + 0.1, ny - 0.6,
               color=BWD_COLOR, lw=2.5)
    ax.text(5.5, ny - 1.0, "−0.00247", fontsize=SMALL_FONT, color=BWD_COLOR,
            fontweight="bold", ha="center")

    draw_arrow(ax, nx - NODE_RADIUS - 0.2, ny - 0.4, 1.2, -0.8,
               color=BWD_COLOR, lw=2.5)

    rect = mpatches.FancyBboxPatch((-0.5, -1.6), 4.2, 1.0,
                                    boxstyle="round,pad=0.1",
                                    fc="#e0f7fa", ec="#00838f", lw=2)
    ax.add_patch(rect)
    ax.text(1.6, -1.1, r"$\frac{\partial L}{\partial \theta_2} = -0.00247 \times x_2 = -0.0049$",
            fontsize=FORMULA_FONT, ha="center", va="center", fontweight="bold",
            color="#00838f")

    ax.set_title(r"Backward Pass — $\theta_2$ gradient",
                 fontsize=TITLE_FONT + 2, pad=12, fontweight="bold")
    return fig


# ============================================================================
# What autodiff library needs to know
# ============================================================================

def autodiff_local_grads():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    ax.set_title("What AutoDiff Library Needs to Know",
                 fontsize=TITLE_FONT + 4, pad=20, fontweight="bold")

    rules = [
        (r"(i)   $f = a \times b$", r"$\frac{\partial f}{\partial a} = b$",
         r"$\frac{\partial f}{\partial b} = a$"),
        (r"(ii)  $f = a + b$", r"$\frac{\partial f}{\partial a} = 1$",
         r"$\frac{\partial f}{\partial b} = 1$"),
        (r"(iii) $f = e^a$", r"$\frac{\partial f}{\partial a} = e^a$", ""),
        (r"(iv)  $f = \frac{1}{a}$", r"$\frac{\partial f}{\partial a} = \frac{-1}{a^2}$", ""),
        (r"(v)   $f = \log(a)$", r"$\frac{\partial f}{\partial a} = \frac{1}{a}$", ""),
    ]

    y = 7
    for func, grad1, grad2 in rules:
        ax.text(1, y, func, fontsize=FORMULA_FONT + 2, va="center",
                fontweight="bold")
        ax.text(5.5, y, grad1, fontsize=FORMULA_FONT + 2, va="center",
                color="#1565c0")
        if grad2:
            ax.text(8, y, grad2, fontsize=FORMULA_FONT + 2, va="center",
                    color="#1565c0")
        y -= 1.3

    ax.text(5, y - 0.3, r"$\vdots$", fontsize=20, ha="center", va="center")

    return fig


# ============================================================================
# Simplified computational graph (sigmoid as single node)
# ============================================================================

def simplified_comp_graph():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-2, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    # Input variables
    inputs = [(0, 3, r"$\theta_1$"), (0, 2, r"$x_1$"),
              (0, 0.5, r"$\theta_2$"), (0, -0.5, r"$x_2$"),
              (0, -1.5, r"$\theta_0$")]

    for x, y, label in inputs:
        ax.text(x, y, label, fontsize=LABEL_FONT, ha="center", va="center",
                fontweight="bold", color="#1a73e8")

    # Multiply nodes
    m1 = draw_circle_node(ax, 1.8, 2.5, r"$\times$", fontsize=14, radius=0.25)
    m2 = draw_circle_node(ax, 1.8, 0, r"$\times$", fontsize=14, radius=0.25)

    arrow_to_node(ax, 0.3, 3, 1.8, 2.5, radius=0.25)
    arrow_to_node(ax, 0.3, 2, 1.8, 2.5, radius=0.25)
    arrow_to_node(ax, 0.3, 0.5, 1.8, 0, radius=0.25)
    arrow_to_node(ax, 0.3, -0.5, 1.8, 0, radius=0.25)

    # Add node
    a1 = draw_circle_node(ax, 3.5, 1.25, "+", fontsize=14, radius=0.25)
    arrow_node_to_node(ax, 1.8, 2.5, 3.5, 1.25, radius=0.25)
    arrow_node_to_node(ax, 1.8, 0, 3.5, 1.25, radius=0.25)

    # Add θ0
    a2 = draw_circle_node(ax, 5, 1.25, "+", fontsize=14, radius=0.25)
    arrow_node_to_node(ax, 3.5, 1.25, 5, 1.25, radius=0.25)
    arrow_to_node(ax, 0.3, -1.5, 5, 1.25, radius=0.25)

    # Sigmoid (larger, distinctive)
    sig_x, sig_y = 7, 1.25
    ellipse = mpatches.Ellipse((sig_x, sig_y), 1.2, 0.8, fc="#bbdefb",
                                ec="#1565c0", lw=2, zorder=3)
    ax.add_patch(ellipse)
    ax.text(sig_x, sig_y, "sigmoid", fontsize=10, ha="center", va="center",
            fontweight="bold", color="#1565c0", zorder=4)
    arrow_node_to_node(ax, 5, 1.25, sig_x, sig_y, radius=0.4)

    # Log node
    log_x = 9
    log_node = draw_circle_node(ax, log_x, 1.25, "log", fontsize=11, radius=0.3)
    draw_arrow(ax, sig_x + 0.6, sig_y, log_x - 0.3, 1.25, color=ARROW_COLOR, lw=1.5)

    # ×(-1)
    mul_x = 11
    mul_node = draw_circle_node(ax, mul_x, 1.25, r"$\times$", fontsize=14, radius=0.25)
    arrow_node_to_node(ax, log_x, 1.25, mul_x, 1.25, radius=0.28)
    ax.text(mul_x, 2.2, "−1", fontsize=SMALL_FONT, ha="center", color="#555")
    draw_arrow(ax, mul_x, 1.9, mul_x, 1.25 + 0.25, color=ARROW_COLOR, lw=1.5)

    # Output L
    draw_arrow(ax, mul_x + 0.25, 1.25, 12.5, 1.25, color=ARROW_COLOR, lw=1.5)
    ax.text(12.8, 1.25, "L", fontsize=LABEL_FONT + 2, ha="left", va="center",
            fontweight="bold")

    ax.set_title("Simplifying Computational Graph",
                 fontsize=TITLE_FONT + 2, pad=12, fontweight="bold")
    return fig


def simplified_comp_graph_sigmoid_grad():
    """Same as above but with sigmoid derivative formula."""
    fig = simplified_comp_graph()
    ax = fig.axes[0]
    ax.text(7, -0.8,
            r"$\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$",
            fontsize=FORMULA_FONT + 4, ha="center", va="center",
            fontweight="bold", color="#1565c0",
            bbox=dict(boxstyle="round,pad=0.4", fc="#e8eaf6", ec="#1565c0",
                      alpha=0.95))
    ax.text(7, -1.7, "Exercise: show you get the same answer as before",
            fontsize=SMALL_FONT, ha="center", va="center", fontstyle="italic",
            color="#555")
    ax.set_title("Simplifying Computational Graph — Sigmoid Gradient",
                 fontsize=TITLE_FONT + 2, pad=12, fontweight="bold")
    return fig


# ============================================================================
# Further simplified: nn.Linear → C.E. Loss
# ============================================================================

def simplified_nn_linear():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    # x box
    x_box = mpatches.FancyBboxPatch((0.5, 1.2), 1.5, 1.2,
                                     boxstyle="round,pad=0.15",
                                     fc="#c8e6c9", ec="#2e7d32", lw=2)
    ax.add_patch(x_box)
    ax.text(1.25, 1.8, r"$\mathbf{x}$", fontsize=LABEL_FONT + 4, ha="center",
            va="center", fontweight="bold")

    # θ box
    t_box = mpatches.FancyBboxPatch((0.5, -0.3), 1.5, 1.0,
                                     boxstyle="round,pad=0.15",
                                     fc="#fff9c4", ec="#f9a825", lw=2)
    ax.add_patch(t_box)
    ax.text(1.25, 0.2, r"$\boldsymbol{\theta}$", fontsize=LABEL_FONT + 4,
            ha="center", va="center", fontweight="bold")

    # nn.Linear
    nn_box = mpatches.Ellipse((5, 1.5), 2.5, 1.5, fc="#e1bee7",
                               ec="#7b1fa2", lw=2.5, zorder=3)
    ax.add_patch(nn_box)
    ax.text(5, 1.5, "nn.Linear", fontsize=LABEL_FONT, ha="center",
            va="center", fontweight="bold", color="#4a148c", zorder=4)

    # Arrows to nn.Linear
    draw_arrow(ax, 2.1, 1.8, 3.8, 1.5, color=ARROW_COLOR, lw=2)
    draw_arrow(ax, 2.1, 0.5, 3.8, 1.2, color=ARROW_COLOR, lw=2)

    # Logits label
    ax.text(7.5, 2.3, r"$\mathbf{x}^\top \boldsymbol{\theta}$",
            fontsize=LABEL_FONT + 2, ha="center", va="center", color="#555")
    ax.text(7.5, 1.7, "logits", fontsize=SMALL_FONT, ha="center",
            va="center", fontstyle="italic", color="#888")

    # Arrow to C.E. Loss
    draw_arrow(ax, 6.3, 1.5, 8.5, 1.5, color=ARROW_COLOR, lw=2)

    # C.E. Loss
    ce_box = mpatches.Ellipse((10, 1.5), 2.5, 1.5, fc="#ffcdd2",
                               ec="#c62828", lw=2.5, zorder=3)
    ax.add_patch(ce_box)
    ax.text(10, 1.5, "C.E. Loss", fontsize=LABEL_FONT, ha="center",
            va="center", fontweight="bold", color="#b71c1c", zorder=4)

    # Output
    draw_arrow(ax, 11.3, 1.5, 12.5, 1.5, color=ARROW_COLOR, lw=2)
    ax.text(12.7, 1.5, "L", fontsize=LABEL_FONT + 2, ha="left",
            va="center", fontweight="bold")

    ax.set_title("Simplifying Computational Graph — Further",
                 fontsize=TITLE_FONT + 2, pad=12, fontweight="bold")
    return fig


# ============================================================================
# Training over N examples
# ============================================================================

def training_n_examples_single():
    """Single example: x_{1,1}, x_{1,2}, θ1, θ2, θ0 → logit → C.E. Loss → L1"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-2, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    # Inputs
    inputs = [(-0.5, 4, r"$x_{1,1}$"), (-0.5, 3, r"$\theta_1$"),
              (-0.5, 1.5, r"$x_{1,2}$"), (-0.5, 0.5, r"$\theta_2$"),
              (-0.5, -1, r"$\theta_0$")]
    for x, y, lbl in inputs:
        ax.text(x, y, lbl, fontsize=LABEL_FONT, ha="center", va="center",
                fontweight="bold", color="#1a73e8")

    # Multiply nodes
    draw_circle_node(ax, 1.5, 3.5, r"$\times$", fontsize=14, radius=0.25)
    draw_circle_node(ax, 1.5, 1.0, r"$\times$", fontsize=14, radius=0.25)
    arrow_to_node(ax, 0, 4, 1.5, 3.5, radius=0.25)
    arrow_to_node(ax, 0, 3, 1.5, 3.5, radius=0.25)
    arrow_to_node(ax, 0, 1.5, 1.5, 1.0, radius=0.25)
    arrow_to_node(ax, 0, 0.5, 1.5, 1.0, radius=0.25)

    # Add
    draw_circle_node(ax, 3.2, 2.25, "+", fontsize=14, radius=0.25)
    arrow_node_to_node(ax, 1.5, 3.5, 3.2, 2.25, radius=0.25)
    arrow_node_to_node(ax, 1.5, 1.0, 3.2, 2.25, radius=0.25)

    # Add θ0
    draw_circle_node(ax, 4.8, 2.25, "+", fontsize=14, radius=0.25)
    arrow_node_to_node(ax, 3.2, 2.25, 4.8, 2.25, radius=0.25)
    arrow_to_node(ax, 0, -1, 4.8, 2.25, radius=0.25)

    # Arrow to "Logits eg. 1"
    draw_arrow(ax, 5.05, 2.25, 7, 2.25, color=ARROW_COLOR, lw=2)
    ax.text(6, 2.65, "Logits\neg. 1", fontsize=SMALL_FONT, ha="center",
            va="bottom", color="#555", fontstyle="italic")

    # C.E. Loss
    ce = mpatches.Ellipse((8.5, 2.25), 2, 1.0, fc="#ffcdd2",
                           ec="#c62828", lw=2, zorder=3)
    ax.add_patch(ce)
    ax.text(8.5, 2.25, "C.E.\nLoss", fontsize=10, ha="center", va="center",
            fontweight="bold", color="#b71c1c", zorder=4)

    # L1
    draw_arrow(ax, 9.5, 2.25, 11, 2.25, color=ARROW_COLOR, lw=2)
    ax.text(11.3, 2.25, r"$L_1$", fontsize=LABEL_FONT + 2, ha="left",
            va="center", fontweight="bold")

    ax.set_title("Training Over N Examples — Single Example",
                 fontsize=TITLE_FONT + 2, pad=12, fontweight="bold")
    return fig


def training_n_examples_two():
    """Two examples with shared weights, → L = L1 + L2"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(-1.5, 16)
    ax.set_ylim(-4, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    # --- Example 1 (top) ---
    inputs1 = [(-1, 5.5, r"$x_{1,1}$"), (-1, 4.5, r"$\theta_1$"),
               (-1, 3, r"$x_{1,2}$"), (-1, 2, r"$\theta_2$")]
    for x, y, lbl in inputs1:
        ax.text(x, y, lbl, fontsize=LABEL_FONT - 1, ha="center", va="center",
                fontweight="bold", color="#1a73e8")

    draw_circle_node(ax, 1, 5.0, r"$\times$", fontsize=12, radius=0.22)
    draw_circle_node(ax, 1, 2.5, r"$\times$", fontsize=12, radius=0.22)
    arrow_to_node(ax, -0.5, 5.5, 1, 5.0, radius=0.22)
    arrow_to_node(ax, -0.5, 4.5, 1, 5.0, radius=0.22)
    arrow_to_node(ax, -0.5, 3, 1, 2.5, radius=0.22)
    arrow_to_node(ax, -0.5, 2, 1, 2.5, radius=0.22)

    draw_circle_node(ax, 2.8, 3.75, "+", fontsize=12, radius=0.22)
    arrow_node_to_node(ax, 1, 5.0, 2.8, 3.75, radius=0.22)
    arrow_node_to_node(ax, 1, 2.5, 2.8, 3.75, radius=0.22)

    draw_circle_node(ax, 4.3, 3.75, "+", fontsize=12, radius=0.22)
    arrow_node_to_node(ax, 2.8, 3.75, 4.3, 3.75, radius=0.22)

    draw_arrow(ax, 4.52, 3.75, 6.5, 3.75, color=ARROW_COLOR, lw=1.5)
    ax.text(5.5, 4.1, "Logits\neg. 1", fontsize=9, ha="center", color="#555",
            fontstyle="italic")

    ce1 = mpatches.Ellipse((7.8, 3.75), 1.8, 0.9, fc="#ffcdd2",
                            ec="#c62828", lw=1.5, zorder=3)
    ax.add_patch(ce1)
    ax.text(7.8, 3.75, "C.E.\nLoss", fontsize=9, ha="center", va="center",
            fontweight="bold", color="#b71c1c", zorder=4)

    draw_arrow(ax, 8.7, 3.75, 10, 3.75, color=ARROW_COLOR, lw=1.5)
    ax.text(10.2, 3.75, r"$L_1$", fontsize=LABEL_FONT, ha="left",
            va="center", fontweight="bold")

    # --- Example 2 (bottom, blue) ---
    inputs2 = [(-1, 0.5, r"$x_{2,1}$"), (-1, -0.5, r"$x_{2,2}$")]
    for x, y, lbl in inputs2:
        ax.text(x, y, lbl, fontsize=LABEL_FONT - 1, ha="center", va="center",
                fontweight="bold", color="#1565c0")

    ax.text(-1, -1.5, r"$\theta_0$", fontsize=LABEL_FONT - 1, ha="center",
            va="center", fontweight="bold", color="#1a73e8")

    draw_circle_node(ax, 1, 0.5, r"$\times$", fontsize=12, radius=0.22,
                     fc="#bbdefb", ec="#1565c0")
    draw_circle_node(ax, 1, -0.5, r"$\times$", fontsize=12, radius=0.22,
                     fc="#bbdefb", ec="#1565c0")

    # Shared weights (blue arrows from θ1, θ2)
    draw_arrow(ax, -0.5, 4.5, 1 - 0.22, 0.5 + 0.15, color="#1565c0", lw=1.5)
    draw_arrow(ax, -0.5, 2, 1 - 0.22, -0.5 + 0.15, color="#1565c0", lw=1.5)
    arrow_to_node(ax, -0.5, 0.5, 1, 0.5, radius=0.22, color="#1565c0")
    arrow_to_node(ax, -0.5, -0.5, 1, -0.5, radius=0.22, color="#1565c0")

    draw_circle_node(ax, 2.8, 0, "+", fontsize=12, radius=0.22,
                     fc="#bbdefb", ec="#1565c0")
    arrow_node_to_node(ax, 1, 0.5, 2.8, 0, radius=0.22, color="#1565c0")
    arrow_node_to_node(ax, 1, -0.5, 2.8, 0, radius=0.22, color="#1565c0")

    draw_circle_node(ax, 4.3, 0, "+", fontsize=12, radius=0.22,
                     fc="#bbdefb", ec="#1565c0")
    arrow_node_to_node(ax, 2.8, 0, 4.3, 0, radius=0.22, color="#1565c0")
    arrow_to_node(ax, -0.5, -1.5, 4.3, 0, radius=0.22, color="#1565c0")
    # θ0 also connects to example 1
    arrow_to_node(ax, -0.5, -1.5, 4.3, 3.75, radius=0.22, color=ARROW_COLOR)

    draw_arrow(ax, 4.52, 0, 6.5, 0, color="#1565c0", lw=1.5)
    ax.text(5.5, 0.35, "Logits\neg. 2", fontsize=9, ha="center",
            color="#1565c0", fontstyle="italic")

    ce2 = mpatches.Ellipse((7.8, 0), 1.8, 0.9, fc="#bbdefb",
                            ec="#1565c0", lw=1.5, zorder=3)
    ax.add_patch(ce2)
    ax.text(7.8, 0, "C.E.\nLoss", fontsize=9, ha="center", va="center",
            fontweight="bold", color="#0d47a1", zorder=4)

    draw_arrow(ax, 8.7, 0, 10, 0, color="#1565c0", lw=1.5)
    ax.text(10.2, 0, r"$L_2$", fontsize=LABEL_FONT, ha="left",
            va="center", fontweight="bold", color="#1565c0")

    # Sum node
    draw_circle_node(ax, 12, 1.9, "+", fontsize=16, radius=0.3)
    draw_arrow(ax, 10.5, 3.75, 12 - 0.3, 1.9 + 0.2, color=ARROW_COLOR, lw=2)
    draw_arrow(ax, 10.5, 0, 12 - 0.3, 1.9 - 0.2, color="#1565c0", lw=2)

    draw_arrow(ax, 12.3, 1.9, 14, 1.9, color=ARROW_COLOR, lw=2)
    ax.text(14.3, 1.9, r"$L$", fontsize=LABEL_FONT + 4, ha="left",
            va="center", fontweight="bold")

    ax.set_title("Training Over N Examples — Shared Weights",
                 fontsize=TITLE_FONT + 2, pad=12, fontweight="bold")
    return fig


# ============================================================================
# Main
# ============================================================================

def main():
    print("Generating backward pass / autograd diagrams...")

    diagrams = [
        ("backward_f9", backward_f9),
        ("backward_f8_log", backward_f8),
        ("backward_f7_inv", backward_f7),
        ("backward_f6_plus1", backward_f6),
        ("backward_f5_exp", backward_f5),
        ("backward_f4_neg", backward_f4),
        ("backward_f3_plus_theta0", backward_f3),
        ("backward_f1_theta1", backward_theta1),
        ("backward_f2_theta2", backward_theta2),
        ("autodiff_local_grads", autodiff_local_grads),
        ("simplified_comp_graph", simplified_comp_graph),
        ("simplified_comp_graph_sigmoid_grad", simplified_comp_graph_sigmoid_grad),
        ("simplified_nn_linear", simplified_nn_linear),
        ("training_n_examples_1", training_n_examples_single),
        ("training_n_examples_2", training_n_examples_two),
    ]

    for i, (name, func) in enumerate(diagrams, 1):
        print(f"[{i}/{len(diagrams)}] {name}")
        fig = func()
        save(fig, name)

    print(f"\nDone! {len(diagrams)} diagrams saved to:")
    print(f"  SVG: {SVG_DIR}")
    print(f"  PNG: {PNG_DIR}")


if __name__ == "__main__":
    main()
