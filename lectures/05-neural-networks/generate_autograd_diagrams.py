#!/usr/bin/env python3
"""
Generate autograd / computational graph diagrams for Lecture 05 - Neural Networks.

Diagrams produced:
  1. comp_graph_simple          - Simple (x*y)+z computational graph
  2. backprop_node              - Backprop through a single node
  3. logistic_comp_graph_step1  - Progressive build steps 1-10
     ...
     logistic_comp_graph_step10
  4. logistic_comp_graph_labeled - Full graph with f-labels
  5. logistic_comp_graph_values  - Full graph with numeric values

All saved as SVG (diagrams/svg/) and PNG 300 DPI (diagrams/png/).
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
LABEL_FONT = 13
TITLE_FONT = 14
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
    """Draw a circle operation node and return its centre."""
    c = plt.Circle((x, y), radius, fc=fc, ec=ec, lw=1.8, zorder=zorder)
    ax.add_patch(c)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=zorder + 1)
    return (x, y)


def draw_arrow(ax, x0, y0, x1, y1, color=ARROW_COLOR, lw=1.5,
               shrinkA=0, shrinkB=0, style="->", zorder=2):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, mutation_scale=14,
        color=color, lw=lw, shrinkA=shrinkA, shrinkB=shrinkB,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def arrow_to_node(ax, x0, y0, cx, cy, radius=NODE_RADIUS, **kw):
    """Arrow from (x0,y0) ending at the boundary of a circle at (cx,cy)."""
    dx, dy = cx - x0, cy - y0
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    # shrink target by radius
    ex = cx - dx / dist * radius
    ey = cy - dy / dist * radius
    # shrink source a tiny bit
    sx = x0 + dx / dist * 0.04
    sy = y0 + dy / dist * 0.04
    draw_arrow(ax, sx, sy, ex, ey, **kw)


def arrow_from_node(ax, cx, cy, x1, y1, radius=NODE_RADIUS, **kw):
    """Arrow from circle boundary at (cx,cy) to (x1,y1)."""
    dx, dy = x1 - cx, y1 - cy
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    sx = cx + dx / dist * radius
    sy = cy + dy / dist * radius
    ex = x1 - dx / dist * 0.04
    ey = y1 - dy / dist * 0.04
    draw_arrow(ax, sx, sy, ex, ey, **kw)


def arrow_node_to_node(ax, cx1, cy1, cx2, cy2, radius=NODE_RADIUS, **kw):
    """Arrow between two circle-node centres (shrink both ends by radius)."""
    dx, dy = cx2 - cx1, cy2 - cy1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    sx = cx1 + dx / dist * radius
    sy = cy1 + dy / dist * radius
    ex = cx2 - dx / dist * radius
    ey = cy2 - dy / dist * radius
    draw_arrow(ax, sx, sy, ex, ey, **kw)


# ============================================================================
# 1. comp_graph_simple  --  (x*y) + z
# ============================================================================

def diagram_comp_graph_simple():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(-1.5, 6.5)
    ax.set_ylim(-1.5, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    # Positions
    mul_x, mul_y = 1.5, 0.5
    add_x, add_y = 3.8, 0.5

    # Input labels
    ax.text(-0.8, 1.5, "x", fontsize=LABEL_FONT + 2, ha="center",
            va="center", fontstyle="italic", color="#1a73e8")
    ax.text(-0.8, -0.5, "y", fontsize=LABEL_FONT + 2, ha="center",
            va="center", fontstyle="italic", color="#1a73e8")
    ax.text(2.6, -0.9, "z", fontsize=LABEL_FONT + 2, ha="center",
            va="center", fontstyle="italic", color="#1a73e8")

    # Operation nodes
    draw_circle_node(ax, mul_x, mul_y, r"$\times$", fontsize=18)
    draw_circle_node(ax, add_x, add_y, "+", fontsize=18)

    # Arrows into multiply
    arrow_to_node(ax, -0.3, 1.5, mul_x, mul_y)
    arrow_to_node(ax, -0.3, -0.5, mul_x, mul_y)

    # Arrow multiply -> add
    arrow_node_to_node(ax, mul_x, mul_y, add_x, add_y)
    ax.text((mul_x + add_x) / 2, mul_y + 0.35, r"$x \cdot y$",
            fontsize=11, ha="center", va="bottom", color="#555")

    # Arrow z -> add
    arrow_to_node(ax, 2.8, -0.5, add_x, add_y)

    # Arrow add -> output
    arrow_from_node(ax, add_x, add_y, 5.8, add_y)
    ax.text(5.9, add_y, r"$(x \cdot y) + z$", fontsize=12,
            ha="left", va="center", color="#555")

    ax.set_title("Computational Graph:  $(x \\cdot y) + z$",
                 fontsize=TITLE_FONT + 2, pad=12, fontweight="bold")
    save(fig, "comp_graph_simple")


# ============================================================================
# 2. backprop_node  --  backprop through a single node
# ============================================================================

def diagram_backprop_node():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-5.5, 7)
    ax.set_ylim(-3.5, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    # Node position
    nx, ny = 0.0, 0.5
    R = 0.45

    # -- the f node (gray) --
    draw_circle_node(ax, nx, ny, "f", radius=R, fc="#CCCCCC",
                     fontsize=20, ec="#555555")

    # -- forward inputs --
    # x  (top-left)
    x_start = (-3.5, 2.5)
    ax.text(x_start[0] - 0.3, x_start[1], "x", fontsize=15, ha="center",
            va="center", fontstyle="italic", color="#1a73e8", fontweight="bold")
    arrow_to_node(ax, x_start[0], x_start[1], nx, ny, radius=R, color="#1a73e8", lw=2)

    # y  (bottom-left)
    y_start = (-3.5, -1.5)
    ax.text(y_start[0] - 0.3, y_start[1], "y", fontsize=15, ha="center",
            va="center", fontstyle="italic", color="#1a73e8", fontweight="bold")
    arrow_to_node(ax, y_start[0], y_start[1], nx, ny, radius=R, color="#1a73e8", lw=2)

    # -- forward output  z = f(x,y) --
    z_end = (3.0, ny)
    arrow_from_node(ax, nx, ny, z_end[0], z_end[1], radius=R, color="#333", lw=2)
    ax.text(1.5, ny + 0.30, r"$z = f(x, y)$", fontsize=13, ha="center",
            va="bottom", color="#333")

    # .... J(o/p)  further right
    ax.text(4.0, ny, r"$\cdots\;\; J$  (output)", fontsize=13, ha="left",
            va="center", color="#555")
    draw_arrow(ax, z_end[0] + 0.05, z_end[1], 3.8, z_end[1], color="#555", lw=1.5)

    # ===================== BACKWARD PASS =====================

    # -- upstream gradient (red arrow coming back from right) --
    up_start = (5.5, ny - 0.7)
    up_end = (R + 0.15, ny - 0.0)
    draw_arrow(ax, up_start[0], up_start[1], up_end[0] + 0.6, ny,
               color="#CC0000", lw=2.5, style="->")
    ax.text(3.8, ny - 0.95,
            r"$\dfrac{\partial J}{\partial z}$",
            fontsize=15, ha="center", va="top", color="#CC0000")
    ax.text(5.0, ny - 0.25, "Upstream\ngradient", fontsize=11,
            ha="center", va="top", color="#CC0000", fontstyle="italic")

    # -- downstream gradient to x  (blue arrow up-left) --
    dx_end = (-3.5, 2.5 - 0.6)
    draw_arrow(ax, -R - 0.1, ny + 0.25, dx_end[0] + 0.5, dx_end[1],
               color="#0055CC", lw=2.5, style="->")
    # formula with colored parts (no \color -- unsupported in mpl mathtext)
    # Background box first
    bx, by = -3.8, 3.6
    ax.add_patch(FancyBboxPatch((bx - 2.6, by - 0.55), 5.2, 1.1,
                 boxstyle="round,pad=0.15", fc="#EEF4FF", ec="#0055CC", lw=1,
                 zorder=1))
    ax.text(bx - 1.8, by, r"$\frac{\partial J}{\partial x}\;=$",
            fontsize=14, ha="center", va="center", color="#0055CC", zorder=2)
    ax.text(bx - 0.3, by, r"$\frac{\partial J}{\partial z}$",
            fontsize=14, ha="center", va="center", color="#E68A00", zorder=2)
    ax.text(bx + 0.5, by, r"$\times$",
            fontsize=14, ha="center", va="center", color="#0055CC", zorder=2)
    ax.text(bx + 1.5, by, r"$\frac{\partial z}{\partial x}$",
            fontsize=14, ha="center", va="center", color="#8B00CC", zorder=2)

    # labels for upstream / local
    ax.text(-2.0, 2.0, "upstream\ngrad.", fontsize=9, ha="center",
            va="center", color="#E68A00", fontstyle="italic", fontweight="bold")
    ax.text(-1.0, 2.9, "local\ngrad.", fontsize=9, ha="center",
            va="center", color="#8B00CC", fontstyle="italic", fontweight="bold")

    # -- downstream gradient to y  (blue arrow down-left) --
    dy_end = (-3.5, -1.5 + 0.6)
    draw_arrow(ax, -R - 0.1, ny - 0.25, dy_end[0] + 0.5, dy_end[1],
               color="#0055CC", lw=2.5, style="->")
    # formula for y gradient
    bx2, by2 = -3.8, -2.6
    ax.add_patch(FancyBboxPatch((bx2 - 2.6, by2 - 0.55), 5.2, 1.1,
                 boxstyle="round,pad=0.15", fc="#EEF4FF", ec="#0055CC", lw=1,
                 zorder=1))
    ax.text(bx2 - 1.8, by2, r"$\frac{\partial J}{\partial y}\;=$",
            fontsize=14, ha="center", va="center", color="#0055CC", zorder=2)
    ax.text(bx2 - 0.3, by2, r"$\frac{\partial J}{\partial z}$",
            fontsize=14, ha="center", va="center", color="#E68A00", zorder=2)
    ax.text(bx2 + 0.5, by2, r"$\times$",
            fontsize=14, ha="center", va="center", color="#0055CC", zorder=2)
    ax.text(bx2 + 1.5, by2, r"$\frac{\partial z}{\partial y}$",
            fontsize=14, ha="center", va="center", color="#8B00CC", zorder=2)

    ax.text(-2.0, -1.2, "upstream\ngrad.", fontsize=9, ha="center",
            va="center", color="#E68A00", fontstyle="italic", fontweight="bold")
    ax.text(-1.0, -2.1, "local\ngrad.", fontsize=9, ha="center",
            va="center", color="#8B00CC", fontstyle="italic", fontweight="bold")

    # -- bottom box: rule summary --
    box_text = "DOWNSTREAM GRADIENT  =  UPSTREAM GRADIENT  ×  LOCAL GRADIENT"
    ax.text(0, -3.2, box_text, fontsize=13, ha="center", va="center",
            fontweight="bold", color="#222",
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFF9E6", ec="#E6A800", lw=2))

    ax.set_title("Backpropagation Through a Single Node", fontsize=TITLE_FONT + 2,
                 pad=14, fontweight="bold")
    save(fig, "backprop_node")


# ============================================================================
# 3-5. Logistic regression computational graph  (progressive build)
# ============================================================================

# Node definitions for the full logistic regression graph
# Each node: (id, x, y, label_in_circle, display)
# We place nodes left to right.

# Layout coordinates
LY_TOP = 1.8     # top row for theta1, x1
LY_BOT = -1.8    # bottom row for theta2, x2
LY_MID = 0.0     # middle for combined path

NODES = [
    # id         x     y        op_label
    ("mul1",     2.0,  LY_TOP,  r"$\times$"),
    ("mul2",     2.0,  LY_BOT,  r"$\times$"),
    ("add12",    4.0,  LY_MID,  "+"),
    ("add_b",    6.0,  LY_MID,  "+"),
    ("neg1",     8.0,  LY_MID,  r"$\times$"),
    ("exp",     10.0,  LY_MID,  "exp"),
    ("add1",    12.0,  LY_MID,  "+"),
    ("inv",     14.0,  LY_MID,  r"$\frac{1}{x}$"),
    ("log",     16.0,  LY_MID,  "log"),
    ("neg2",    18.0,  LY_MID,  r"$\times$"),
]

# Input labels (name, x, y, anchor_node_id)
INPUTS = [
    (r"$\theta_1$",  -0.2, LY_TOP + 0.7,  "mul1", 2.0, LY_TOP),
    (r"$x_1$",       -0.2, LY_TOP - 0.7,  "mul1", 2.0, LY_TOP),
    (r"$\theta_2$",  -0.2, LY_BOT + 0.7,  "mul2", 2.0, LY_BOT),
    (r"$x_2$",       -0.2, LY_BOT - 0.7,  "mul2", 2.0, LY_BOT),
    (r"$\theta_0$",   6.0, LY_MID + 2.0,  "add_b", 6.0, LY_MID),
    (r"$-1$",         8.0, LY_MID + 2.0,  "neg1", 8.0, LY_MID),
    (r"$1$",         12.0, LY_MID + 2.0,  "add1", 12.0, LY_MID),
    (r"$-1$",        18.0, LY_MID + 2.0,  "neg2", 18.0, LY_MID),
]

# Edges (from_id, to_id)
EDGES = [
    ("mul1", "add12"),
    ("mul2", "add12"),
    ("add12", "add_b"),
    ("add_b", "neg1"),
    ("neg1", "exp"),
    ("exp", "add1"),
    ("add1", "inv"),
    ("inv", "log"),
    ("log", "neg2"),
]

# Which step each node is introduced (1-indexed)
NODE_STEP = {
    "mul1": 1, "mul2": 2, "add12": 3, "add_b": 4, "neg1": 5,
    "exp": 6, "add1": 7, "inv": 8, "log": 9, "neg2": 10,
}

# Which step each input is introduced
INPUT_STEP = {
    r"$\theta_1$": 1, r"$x_1$": 1,
    r"$\theta_2$": 2, r"$x_2$": 2,
    r"$\theta_0$": 4, r"$-1$": 5,   # first -1
    r"$1$": 7,
    # second -1 at step 10 -- handled specially
}

# Which step each edge appears
EDGE_STEP = {
    ("mul1", "add12"): 3,
    ("mul2", "add12"): 3,
    ("add12", "add_b"): 4,
    ("add_b", "neg1"): 5,
    ("neg1", "exp"): 6,
    ("exp", "add1"): 7,
    ("add1", "inv"): 8,
    ("inv", "log"): 9,
    ("log", "neg2"): 10,
}

# f-labels for the labeled version
F_LABELS = {
    "mul1": r"$f_1$", "mul2": r"$f_2$", "add12": r"$f_3$",
    "add_b": r"$f_4$", "neg1": r"$f_5$", "exp": r"$f_6$",
    "add1": r"$f_7$", "inv": r"$f_8$", "log": r"$f_9$",
    "neg2": r"$f_{10}$",
}

FORMULA = r"Loss $= -\log\!\left(\dfrac{1}{1+e^{-(\theta_0 + \theta_1 x_1 + \theta_2 x_2)}}\right)$"


def _node_dict():
    return {n[0]: n for n in NODES}


def draw_logistic_graph(ax, step=10, show_f_labels=False, show_values=False):
    """Draw the logistic comp graph up to the given step."""
    nd = _node_dict()
    R = NODE_RADIUS

    # Draw edges
    for (src, dst) in EDGES:
        s = EDGE_STEP.get((src, dst), 99)
        if s > step:
            continue
        sx, sy = nd[src][1], nd[src][2]
        dx, dy = nd[dst][1], nd[dst][2]
        arrow_node_to_node(ax, sx, sy, dx, dy, radius=R, lw=1.8)

    # Draw nodes
    for nid, nx, ny, lbl in NODES:
        s = NODE_STEP[nid]
        if s > step:
            continue
        # Highlight newest node
        fc = "#D4EDDA" if s == step and step <= 10 else NODE_COLOR
        draw_circle_node(ax, nx, ny, lbl, radius=R, fc=fc, fontsize=14)

        # f-labels
        if show_f_labels:
            ax.text(nx, ny - R - 0.30, F_LABELS.get(nid, ""),
                    fontsize=11, ha="center", va="top", color="#007766",
                    fontweight="bold")

    # Draw input arrows & labels
    for (ilabel, ix, iy, target_id, tx, ty) in INPUTS:
        ts = NODE_STEP[target_id]
        if ts > step:
            continue
        # Determine if this input should show at this step
        # The second -1 (for neg2) appears at step 10
        if ilabel == r"$-1$" and target_id == "neg2" and step < 10:
            continue
        if ilabel == r"$-1$" and target_id == "neg1" and step < 5:
            continue
        if ilabel == r"$1$" and step < 7:
            continue
        if ilabel == r"$\theta_0$" and step < 4:
            continue
        if ilabel in (r"$\theta_2$", r"$x_2$") and step < 2:
            continue

        # Position text
        ax.text(ix, iy, ilabel, fontsize=13, ha="center", va="center",
                color="#1a73e8", fontweight="bold")
        # Arrow from label to node
        arrow_to_node(ax, ix, iy, tx, ty, radius=R, color="#1a73e8", lw=1.5)

    # Output arrow & L label at the end
    if step >= 10:
        last_x, last_y = nd["neg2"][1], nd["neg2"][2]
        arrow_from_node(ax, last_x, last_y, last_x + 1.8, last_y, radius=R, lw=2)
        ax.text(last_x + 2.0, last_y, r"$L$", fontsize=16, ha="left",
                va="center", fontweight="bold", color="#CC0000")

    # Show numeric values if requested
    if show_values:
        val_color = "#CC0000"
        vfs = 10
        # theta1=1, x1=1 => f1=1
        ax.text(nd["mul1"][1], nd["mul1"][2] + R + 0.25,
                r"$1 \times 1 = 1$", fontsize=vfs, ha="center", color=val_color)
        # theta2=2, x2=2 => f2=4
        ax.text(nd["mul2"][1], nd["mul2"][2] - R - 0.25,
                r"$2 \times 2 = 4$", fontsize=vfs, ha="center", va="top", color=val_color)
        # f3 = 1+4 = 5
        ax.text(nd["add12"][1], nd["add12"][2] + R + 0.25,
                r"$1+4=5$", fontsize=vfs, ha="center", color=val_color)
        # f4 = 5+1 = 6
        ax.text(nd["add_b"][1], nd["add_b"][2] - R - 0.25,
                r"$5+1=6$", fontsize=vfs, ha="center", va="top", color=val_color)
        # f5 = -6
        ax.text(nd["neg1"][1], nd["neg1"][2] + R + 0.25,
                r"$-6$", fontsize=vfs, ha="center", color=val_color)
        # f6 = e^{-6} ~ 0.0025
        ax.text(nd["exp"][1], nd["exp"][2] - R - 0.25,
                r"$e^{-6}\!\approx\!0.0025$", fontsize=vfs, ha="center", va="top", color=val_color)
        # f7 = 1.0025
        ax.text(nd["add1"][1], nd["add1"][2] + R + 0.25,
                r"$1.0025$", fontsize=vfs, ha="center", color=val_color)
        # f8 = 0.9975
        ax.text(nd["inv"][1], nd["inv"][2] - R - 0.25,
                r"$0.9975$", fontsize=vfs, ha="center", va="top", color=val_color)
        # log = -0.0025
        ax.text(nd["log"][1], nd["log"][2] + R + 0.25,
                r"$-0.0025$", fontsize=vfs, ha="center", color=val_color)
        # L = 0.0025
        ax.text(nd["neg2"][1], nd["neg2"][2] - R - 0.25,
                r"$L\!=\!0.0025$", fontsize=vfs, ha="center", va="top",
                color=val_color, fontweight="bold")

        # Show input values
        ax.text(-0.2, LY_TOP + 1.2, r"$\theta_1\!=\!1$", fontsize=vfs,
                ha="center", color=val_color)
        ax.text(-0.2, LY_TOP - 1.2, r"$x_1\!=\!1$", fontsize=vfs,
                ha="center", color=val_color)
        ax.text(-0.2, LY_BOT + 1.2, r"$\theta_2\!=\!2$", fontsize=vfs,
                ha="center", color=val_color)
        ax.text(-0.2, LY_BOT - 1.2, r"$x_2\!=\!2$", fontsize=vfs,
                ha="center", color=val_color)
        ax.text(6.0, LY_MID + 2.5, r"$\theta_0\!=\!1$", fontsize=vfs,
                ha="center", color=val_color)


def make_logistic_figure(step=10, show_f_labels=False, show_values=False):
    fig, ax = plt.subplots(figsize=(18, 6))
    fig.patch.set_facecolor(BG_COLOR)

    # Axis limits depend on how many nodes are visible
    max_x = 20.5
    ax.set_xlim(-1.5, max_x)
    ax.set_ylim(-3.5, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title with formula
    title_extra = ""
    if show_f_labels:
        title_extra = "  (with function labels)"
    elif show_values:
        title_extra = r"  ($\theta_1\!=\!1,\; x_1\!=\!1,\; \theta_2\!=\!2,\; x_2\!=\!2,\; \theta_0\!=\!1$)"
    ax.set_title(FORMULA + title_extra,
                 fontsize=TITLE_FONT, pad=14, fontweight="bold")

    draw_logistic_graph(ax, step=step, show_f_labels=show_f_labels,
                        show_values=show_values)
    return fig


# ============================================================================
# Main
# ============================================================================

def main():
    print("Generating autograd diagrams...")

    # 1. Simple computational graph
    print("[1/14] comp_graph_simple")
    diagram_comp_graph_simple()

    # 2. Backprop through single node
    print("[2/14] backprop_node")
    diagram_backprop_node()

    # 3-12. Progressive logistic regression comp graph
    for step in range(1, 11):
        name = f"logistic_comp_graph_step{step}"
        print(f"[{step + 2}/14] {name}")
        fig = make_logistic_figure(step=step)
        save(fig, name)

    # 13. Full graph with f-labels
    print("[13/14] logistic_comp_graph_labeled")
    fig = make_logistic_figure(step=10, show_f_labels=True)
    save(fig, "logistic_comp_graph_labeled")

    # 14. Full graph with numeric values
    print("[14/14] logistic_comp_graph_values")
    fig = make_logistic_figure(step=10, show_values=True)
    save(fig, "logistic_comp_graph_values")

    print("\nDone! All diagrams saved to:")
    print(f"  SVG: {SVG_DIR}")
    print(f"  PNG: {PNG_DIR}")


if __name__ == "__main__":
    main()
