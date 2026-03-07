"""Generate all diagrams for Lecture 05: Neural Networks.

Run: python generate_diagrams.py
Output: diagrams/png/*.png and diagrams/svg/*.svg
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

os.makedirs('diagrams/png', exist_ok=True)
os.makedirs('diagrams/svg', exist_ok=True)

plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'sans-serif'


def save(fig, name):
    """Save as both PNG and SVG."""
    fig.savefig(f'diagrams/png/{name}.png', dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(f'diagrams/svg/{name}.svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  {name}")


# ============================================================
# 1. Biological vs Artificial Neuron
# ============================================================
def plot_bio_vs_artificial():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.set_xlim(0, 10); ax1.set_ylim(0, 8); ax1.set_aspect('equal')

    for angle in [150, 170, 190, 210]:
        rad = np.radians(angle)
        x0, y0 = 3, 4
        for length in [1.5, 2.0, 2.5]:
            dx = length * np.cos(rad) + np.random.uniform(-0.3, 0.3)
            dy = length * np.sin(rad) + np.random.uniform(-0.3, 0.3)
            ax1.annotate('', xy=(x0+dx*0.3, y0+dy*0.3), xytext=(x0+dx, y0+dy),
                         arrowprops=dict(arrowstyle='->', color='#2a9d8f', lw=2))

    ax1.add_patch(plt.Circle((3, 4), 1.0, color='#264653', alpha=0.8))
    ax1.text(3, 4, 'Cell\nBody', ha='center', va='center', color='white', fontsize=11, fontweight='bold')
    ax1.annotate('', xy=(7.5, 4), xytext=(4, 4), arrowprops=dict(arrowstyle='->', color='#e76f51', lw=3))
    ax1.text(5.5, 4.4, 'Axon', fontsize=12, ha='center', color='#e76f51')
    for dy in [-0.8, 0, 0.8]:
        ax1.plot([7.5, 8.5], [4, 4+dy], color='#e76f51', lw=2)
        ax1.plot(8.5, 4+dy, 'o', color='#e76f51', markersize=8)
    ax1.text(1.0, 7, 'Dendrites\n(inputs)', fontsize=11, color='#2a9d8f', ha='center')
    ax1.text(8.5, 6, 'Output\nterminals', fontsize=11, color='#e76f51', ha='center')
    ax1.set_title('Biological Neuron', fontsize=16, fontweight='bold')
    ax1.axis('off')

    ax2.set_xlim(0, 10); ax2.set_ylim(0, 8); ax2.set_aspect('equal')
    inputs = [(1, 6.5, '$x_1$'), (1, 4, '$x_2$'), (1, 1.5, '$x_3$')]
    for x, y, label in inputs:
        ax2.add_patch(plt.Circle((x, y), 0.5, color='#2a9d8f', alpha=0.8))
        ax2.text(x, y, label, ha='center', va='center', color='white', fontsize=13, fontweight='bold')

    weights = ['$w_1$', '$w_2$', '$w_3$']
    for (x, y, _), w in zip(inputs, weights):
        ax2.annotate('', xy=(4.2, 4), xytext=(1.5, y),
                    arrowprops=dict(arrowstyle='->', color='#264653', lw=2))
        mid_x, mid_y = (1.5 + 4.2) / 2, (y + 4) / 2
        ax2.text(mid_x - 0.3, mid_y + 0.3, w, fontsize=12, color='#264653', fontweight='bold')

    ax2.add_patch(plt.Circle((5, 4), 0.8, color='#264653', alpha=0.8))
    ax2.text(5, 4, '$\\Sigma$', ha='center', va='center', color='white', fontsize=20, fontweight='bold')
    ax2.annotate('', xy=(7.3, 4), xytext=(5.8, 4), arrowprops=dict(arrowstyle='->', color='#264653', lw=2))
    ax2.add_patch(patches.FancyBboxPatch((7.0, 3.0), 1.5, 2, boxstyle="round,pad=0.2",
                                          facecolor='#e76f51', alpha=0.8))
    ax2.text(7.75, 4, '$g(z)$', ha='center', va='center', color='white', fontsize=14, fontweight='bold')
    ax2.annotate('', xy=(9.5, 4), xytext=(8.5, 4), arrowprops=dict(arrowstyle='->', color='#e76f51', lw=3))
    ax2.text(9.5, 4, '$\\hat{y}$', ha='center', va='center', fontsize=16, fontweight='bold', color='#e76f51')
    ax2.annotate('', xy=(5, 3.2), xytext=(5, 1.5), arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax2.text(5, 1.0, '+b (bias)', ha='center', fontsize=12, color='gray')
    ax2.set_title('Artificial Neuron', fontsize=16, fontweight='bold')
    ax2.axis('off')

    plt.tight_layout()
    save(fig, 'bio_vs_artificial_neuron')


# ============================================================
# 2. Decision boundaries for AND, OR, XOR
# ============================================================
def plot_decision_boundaries():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def plot_gate(ax, title, data, w1=None, w2=None, b=None, impossible=False):
        if not impossible and w1 is not None:
            xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 200), np.linspace(-0.3, 1.3, 200))
            Z = w1 * xx + w2 * yy + b
            ax.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['#fde8e4', '#d4edda'], alpha=0.5)
            x_line = np.linspace(-0.3, 1.3, 100)
            if abs(w2) > 0.01:
                y_line = -(w1 * x_line + b) / w2
                mask = (y_line >= -0.3) & (y_line <= 1.3)
                ax.plot(x_line[mask], y_line[mask], 'k--', lw=2, alpha=0.7)

        for (x1, x2), y in data:
            color = '#2a9d8f' if y == 1 else '#e85a4f'
            marker = '^' if y == 1 else 'o'
            ax.scatter(x1, x2, c=color, s=300, edgecolors='black', linewidth=2, zorder=5, marker=marker)
            ax.annotate(f'{y}', (x1, x2), textcoords="offset points", xytext=(12, 8),
                        fontsize=14, fontweight='bold')
        ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.3, 1.3)
        ax.set_xlabel('$x_1$', fontsize=14); ax.set_ylabel('$x_2$', fontsize=14)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    and_data = [((0,0),0), ((0,1),0), ((1,0),0), ((1,1),1)]
    or_data  = [((0,0),0), ((0,1),1), ((1,0),1), ((1,1),1)]
    xor_data = [((0,0),0), ((0,1),1), ((1,0),1), ((1,1),0)]

    plot_gate(axes[0], 'AND: One line works', and_data, w1=1, w2=1, b=-1.5)
    plot_gate(axes[1], 'OR: One line works', or_data, w1=1, w2=1, b=-0.5)
    plot_gate(axes[2], 'XOR: No single line!', xor_data, impossible=True)

    x_line = np.linspace(-0.3, 1.3, 50)
    axes[2].plot(x_line, -x_line + 0.5, 'r--', lw=1.5, alpha=0.5, label='Attempt 1')
    axes[2].plot(x_line, -x_line + 1.5, 'b--', lw=1.5, alpha=0.5, label='Attempt 2')
    axes[2].legend(fontsize=10)

    plt.tight_layout()
    save(fig, 'decision_boundaries_gates')


# ============================================================
# 3. XOR solved in hidden space
# ============================================================
def plot_xor_hidden_space():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    xor_data = [((0,0),0), ((0,1),1), ((1,0),1), ((1,1),0)]

    for (x1, x2), y in xor_data:
        color = '#2a9d8f' if y == 1 else '#e85a4f'
        marker = '^' if y == 1 else 'o'
        axes[0].scatter(x1, x2, c=color, s=300, edgecolors='black', linewidth=2, zorder=5, marker=marker)
        axes[0].annotate(f'({x1},{x2})->{y}', (x1, x2), textcoords="offset points", xytext=(10, 10), fontsize=12)
    axes[0].set_title('Input Space\n(NOT separable)', fontsize=14, fontweight='bold', color='#e85a4f')
    axes[0].set_xlabel('$x_1$'); axes[0].set_ylabel('$x_2$')
    axes[0].set_xlim(-0.3, 1.5); axes[0].set_ylim(-0.3, 1.5)
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)

    axes[1].text(0.5, 0.5, 'Hidden Layer\nTransforms\nthe Data!', ha='center', va='center',
                 fontsize=16, fontweight='bold', color='#264653',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#264653', lw=2))
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1); axes[1].axis('off')

    hidden_map = {(0,0): (0,0,0), (0,1): (1,0,1), (1,0): (1,0,1), (1,1): (1,1,0)}
    plotted = set()
    for (x1,x2), (h1, h2, y) in hidden_map.items():
        color = '#2a9d8f' if y == 1 else '#e85a4f'
        marker = '^' if y == 1 else 'o'
        axes[2].scatter(h1, h2, c=color, s=300, edgecolors='black', linewidth=2, zorder=5, marker=marker)
        offset_y = 10 if (h1,h2) not in plotted else -20
        axes[2].annotate(f'({x1},{x2})->{y}', (h1, h2), textcoords="offset points",
                         xytext=(10, offset_y), fontsize=11)
        plotted.add((h1,h2))

    h1_line = np.linspace(-0.3, 1.5, 100)
    h2_line = (h1_line - 0.5) / 2
    mask = (h2_line >= -0.3) & (h2_line <= 1.5)
    axes[2].plot(h1_line[mask], h2_line[mask], 'k--', lw=2, alpha=0.7)
    axes[2].fill_between(h1_line[mask], h2_line[mask], -0.3, alpha=0.1, color='#2a9d8f')
    axes[2].fill_between(h1_line[mask], h2_line[mask], 1.5, alpha=0.1, color='#e85a4f')
    axes[2].set_title('Hidden Space\n(Separable!)', fontsize=14, fontweight='bold', color='#2a9d8f')
    axes[2].set_xlabel('$h_1$ (OR)'); axes[2].set_ylabel('$h_2$ (AND)')
    axes[2].set_xlim(-0.3, 1.5); axes[2].set_ylim(-0.3, 1.5)
    axes[2].set_aspect('equal'); axes[2].grid(True, alpha=0.3)

    plt.suptitle('The Hidden Layer Transforms XOR into a Linearly Separable Problem!',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save(fig, 'xor_hidden_space')


# ============================================================
# 4. Activation functions comparison
# ============================================================
def plot_activations():
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    z = np.linspace(-5, 5, 300)

    axes[0].plot(z, (z > 0).astype(float), lw=3, color='#264653')
    axes[0].set_title('Step (Original Perceptron)', fontsize=13, fontweight='bold')
    axes[0].set_ylim(-0.2, 1.3); axes[0].text(2, 0.3, 'Binary:\n0 or 1', fontsize=11, color='gray')

    axes[1].plot(z, 1 / (1 + np.exp(-z)), lw=3, color='#2a9d8f')
    axes[1].axhline(0.5, color='gray', lw=0.5, ls='--')
    axes[1].set_title('Sigmoid', fontsize=13, fontweight='bold')
    axes[1].set_ylim(-0.2, 1.3); axes[1].text(1.5, 0.2, 'Smooth\n(0, 1)', fontsize=11, color='gray')

    axes[2].plot(z, np.tanh(z), lw=3, color='#e9c46a')
    axes[2].set_title('Tanh', fontsize=13, fontweight='bold')
    axes[2].set_ylim(-1.3, 1.3); axes[2].text(1.5, -0.7, 'Centered\n(-1, 1)', fontsize=11, color='gray')

    axes[3].plot(z, np.maximum(0, z), lw=3, color='#e76f51')
    axes[3].set_title('ReLU (Modern Default!)', fontsize=13, fontweight='bold', color='#e76f51')
    axes[3].set_ylim(-1, 5.5); axes[3].text(1.5, 1, 'Simple!\nmax(0,z)', fontsize=11, color='gray')

    for ax in axes:
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlabel('z'); ax.set_ylabel('g(z)'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save(fig, 'activation_functions_comparison')


# ============================================================
# 5. Why non-linearity matters
# ============================================================
def plot_nonlinearity():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.linspace(-3, 3, 100)
    axes[0].plot(x, 2*x + 1, lw=2, alpha=0.5, color='blue', label='Layer 1: y = 2x + 1')
    axes[0].plot(x, 0.5*(2*x+1) - 0.3, lw=2, alpha=0.5, color='red', label='Layer 2: y = 0.5h - 0.3')
    axes[0].plot(x, x + 0.2, lw=3, color='purple', ls='--', label='Combined: y = x + 0.2 (still linear!)')
    axes[0].set_title('Without Activation:\n100 layers = 1 line', fontsize=14, fontweight='bold', color='#e85a4f')
    axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')

    x2 = np.linspace(-3, 3, 300)
    h = np.maximum(0, 2*x2 + 1)
    axes[1].plot(x2, h, lw=2, alpha=0.5, color='blue', label='Layer 1: h = ReLU(2x + 1)')
    axes[1].plot(x2, 0.5 * h - 0.3, lw=3, color='#2a9d8f', label='Layer 2: y = 0.5h - 0.3')
    axes[1].set_title('With ReLU Activation:\nCan model non-linear functions!', fontsize=14, fontweight='bold', color='#2a9d8f')
    axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')

    plt.tight_layout()
    save(fig, 'why_nonlinearity')


# ============================================================
# 6. MLP Architecture diagram
# ============================================================
def plot_mlp_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    layer_sizes = [3, 4, 4, 2]
    layer_names = ['Input\nLayer', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Output\nLayer']
    layer_colors = ['#2a9d8f', '#264653', '#264653', '#e76f51']
    x_positions = [1, 3.5, 6, 8.5]

    neuron_positions = {}
    for l, (n, x_pos, name, color) in enumerate(zip(layer_sizes, x_positions, layer_names, layer_colors)):
        y_offset = (max(layer_sizes) - n) / 2
        for i in range(n):
            y = i + y_offset + 0.5
            ax.add_patch(plt.Circle((x_pos, y), 0.3, color=color, alpha=0.85, zorder=5))
            neuron_positions[(l, i)] = (x_pos, y)
            if l == 0:
                ax.text(x_pos, y, ['$x_1$','$x_2$','$x_3$'][i], ha='center', va='center',
                        color='white', fontsize=12, fontweight='bold', zorder=6)
            elif l == len(layer_sizes) - 1:
                ax.text(x_pos, y, ['$\\hat{y}_1$','$\\hat{y}_2$'][i], ha='center', va='center',
                        color='white', fontsize=11, fontweight='bold', zorder=6)
        ax.text(x_pos, -0.5, name, ha='center', fontsize=12, fontweight='bold', color=color)

    for l in range(len(layer_sizes) - 1):
        for i in range(layer_sizes[l]):
            for j in range(layer_sizes[l+1]):
                x1, y1 = neuron_positions[(l, i)]
                x2, y2 = neuron_positions[(l+1, j)]
                ax.plot([x1+0.3, x2-0.3], [y1, y2], color='gray', alpha=0.3, lw=1, zorder=1)

    ax.set_xlim(-0.5, 10); ax.set_ylim(-1.5, 5); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Multi-Layer Perceptron (MLP)', fontsize=16, fontweight='bold', pad=20)
    save(fig, 'mlp_architecture_clean')


# ============================================================
# 7. Old vs New paradigm
# ============================================================
def plot_paradigm_change():
    fig, axes = plt.subplots(2, 1, figsize=(16, 7))

    boxes_old = [(0.5, 'Image', '#bbb'), (2.5, 'Human designs\nfeatures', '#e85a4f'),
                 (5.0, 'Count edges\n# loops\npixel stats', '#e85a4f'),
                 (7.5, 'Classifier\n(SVM/LR)', '#2a9d8f'), (10.0, 'Prediction', '#bbb')]
    for x, text, color in boxes_old:
        axes[0].add_patch(patches.FancyBboxPatch((x-0.8, 0.1), 1.6, 1.3,
                          boxstyle="round,pad=0.15", facecolor=color, alpha=0.8))
        axes[0].text(x, 0.75, text, ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white' if color != '#bbb' else 'black')
    for i in range(len(boxes_old)-1):
        axes[0].annotate('', xy=(boxes_old[i+1][0]-0.8, 0.75), xytext=(boxes_old[i][0]+0.8, 0.75),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2))
    axes[0].set_xlim(-0.5, 11.5); axes[0].set_ylim(-0.3, 2)
    axes[0].set_title('Old Way: Hand-Crafted Features', fontsize=14, fontweight='bold', color='#e85a4f')
    axes[0].axis('off')

    boxes_new = [(0.5, 'Image', '#bbb'),
                 (3.5, 'Neural Network\n(learns features AND classifier)', '#2a9d8f'),
                 (7.5, 'Prediction', '#bbb')]
    for x, text, color in boxes_new:
        w = 1.6 if x != 3.5 else 3.5
        axes[1].add_patch(patches.FancyBboxPatch((x-w/2, 0.1), w, 1.3,
                          boxstyle="round,pad=0.15", facecolor=color, alpha=0.8))
        axes[1].text(x, 0.75, text, ha='center', va='center', fontsize=11, fontweight='bold',
                    color='white' if color != '#bbb' else 'black')
    axes[1].annotate('', xy=(1.75, 0.75), xytext=(1.3, 0.75),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    axes[1].annotate('', xy=(6.7, 0.75), xytext=(5.25, 0.75),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    axes[1].set_xlim(-0.5, 11.5); axes[1].set_ylim(-0.3, 2)
    axes[1].set_title('New Way: End-to-End Learning', fontsize=14, fontweight='bold', color='#2a9d8f')
    axes[1].axis('off')

    plt.tight_layout()
    save(fig, 'paradigm_change')


# ============================================================
# 8. Forward pass step-by-step visualization
# ============================================================
def plot_forward_pass():
    fig, ax = plt.subplots(figsize=(18, 8))

    positions = {
        'x1': (0, 5), 'x2': (0, 2),
        'z1': (4, 7), 'z2': (4, 4), 'z3': (4, 1),
        'h1': (7, 7), 'h2': (7, 4), 'h3': (7, 1),
        'z_out': (10, 4), 'y_hat': (13, 4),
    }
    values = {
        'x1': '1.0', 'x2': '0.5', 'z1': '0.5', 'z2': '-0.55', 'z3': '0.2',
        'h1': '0.5', 'h2': '0.0', 'h3': '0.2', 'z_out': '0.35', 'y_hat': '0.587',
    }
    colors_map = {
        'x1': '#2a9d8f', 'x2': '#2a9d8f',
        'z1': '#bbb', 'z2': '#bbb', 'z3': '#bbb',
        'h1': '#264653', 'h2': '#999', 'h3': '#264653',
        'z_out': '#bbb', 'y_hat': '#e76f51',
    }

    for name, (x, y) in positions.items():
        c = colors_map[name]
        ax.add_patch(plt.Circle((x, y), 0.6, color=c, alpha=0.85, zorder=5))
        ax.text(x, y, values[name], ha='center', va='center', color='white',
                fontsize=11, fontweight='bold', zorder=6)
        ax.text(x, y+0.9, name.replace('_', '\n'), ha='center', fontsize=10, color=c)

    for src, dst, w in [('x1','z1','0.2'),('x1','z2','-0.5'),('x1','z3','0.3'),
                         ('x2','z1','0.4'),('x2','z2','0.1'),('x2','z3','-0.2')]:
        x1, y1 = positions[src]; x2, y2 = positions[dst]
        ax.annotate('', xy=(x2-0.6, y2), xytext=(x1+0.6, y1),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.6))

    for src, dst in [('z1','h1'), ('z2','h2'), ('z3','h3')]:
        x1, y1 = positions[src]; x2, y2 = positions[dst]
        ax.annotate('', xy=(x2-0.6, y2), xytext=(x1+0.6, y1),
                   arrowprops=dict(arrowstyle='->', color='#e76f51', lw=2))

    ax.text(5.5, 8, 'ReLU', fontsize=12, color='#e76f51', fontweight='bold', ha='center')

    for src in ['h1', 'h2', 'h3']:
        x1, y1 = positions[src]; x2, y2 = positions['z_out']
        ax.annotate('', xy=(x2-0.6, y2), xytext=(x1+0.6, y1),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.6))

    ax.annotate('', xy=(positions['y_hat'][0]-0.6, 4), xytext=(positions['z_out'][0]+0.6, 4),
               arrowprops=dict(arrowstyle='->', color='#e76f51', lw=2))
    ax.text(11.5, 5, '$\\sigma$', fontsize=16, color='#e76f51', fontweight='bold', ha='center')

    ax.text(0, -0.5, 'Input', fontsize=13, ha='center', fontweight='bold', color='#2a9d8f')
    ax.text(4, -0.5, 'Step 1:\nWeighted Sum', fontsize=11, ha='center', color='gray')
    ax.text(7, -0.5, 'Step 2:\nReLU', fontsize=11, ha='center', color='#e76f51')
    ax.text(10, -0.5, 'Step 3:\nWeighted Sum', fontsize=11, ha='center', color='gray')
    ax.text(13, -0.5, 'Step 4:\nSigmoid', fontsize=11, ha='center', color='#e76f51')

    ax.annotate('Dead neuron!\n(ReLU killed it)', xy=(7, 4), xytext=(9, 2),
               fontsize=10, color='#e85a4f', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#e85a4f', lw=1.5))

    ax.set_xlim(-1.5, 14.5); ax.set_ylim(-1.5, 9); ax.axis('off')
    ax.set_title('Forward Pass: Step by Step (2->3->1 network)', fontsize=16, fontweight='bold', pad=20)
    save(fig, 'forward_pass_visual')


# ============================================================
# 9. Softmax visualization
# ============================================================
def plot_softmax():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    digits = list(range(10))
    raw = [0.5, 0.2, 3.1, 1.0, -0.5, 0.1, -0.3, 0.8, 0.3, -1.0]
    probs = np.exp(raw) / np.exp(raw).sum()

    c1 = ['#264653']*10; c1[2] = '#e76f51'
    ax1.barh(digits, raw, color=c1, edgecolor='white')
    ax1.set_xlabel('Raw Score (logit)'); ax1.set_ylabel('Digit')
    ax1.set_title('Before Softmax\n(Raw Scores)', fontsize=14, fontweight='bold')
    ax1.set_yticks(digits); ax1.axvline(0, color='gray', lw=0.5)

    c2 = ['#2a9d8f']*10; c2[2] = '#e76f51'
    ax2.barh(digits, probs, color=c2, edgecolor='white')
    ax2.set_xlabel('Probability'); ax2.set_ylabel('Digit')
    ax2.set_title('After Softmax\n(Probabilities)', fontsize=14, fontweight='bold')
    ax2.set_yticks(digits)
    for i, p in enumerate(probs):
        ax2.text(p + 0.01, i, f'{p:.1%}', va='center', fontsize=10, fontweight='bold')

    plt.suptitle('Softmax: Raw Scores -> Probabilities (sum to 1)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    save(fig, 'softmax_visualization')


# ============================================================
# 10. Parameter counting visual
# ============================================================
def plot_parameter_counting():
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (x, name, s) in enumerate(zip([1,5,9],
            ['Input\n(784)','Hidden\n(128)','Output\n(10)'], [784,128,10])):
        h = max(0.5, s / 200)
        ax.add_patch(patches.FancyBboxPatch((x-0.8, 2-h/2), 1.6, h, boxstyle="round,pad=0.1",
                     facecolor=['#2a9d8f','#264653','#e76f51'][i], alpha=0.85))
        ax.text(x, 2, name, ha='center', va='center', color='white', fontsize=12, fontweight='bold')

    ax.annotate('', xy=(4.2, 2), xytext=(1.8, 2), arrowprops=dict(arrowstyle='->', lw=3, color='gray'))
    ax.text(3, 2.8, '784 x 128 = 100,352\nweights', ha='center', fontsize=11, fontweight='bold')
    ax.text(3, 1.2, '+ 128 biases\n= 100,480', ha='center', fontsize=10, color='gray')

    ax.annotate('', xy=(8.2, 2), xytext=(5.8, 2), arrowprops=dict(arrowstyle='->', lw=3, color='gray'))
    ax.text(7, 2.8, '128 x 10 = 1,280\nweights', ha='center', fontsize=11, fontweight='bold')
    ax.text(7, 1.2, '+ 10 biases\n= 1,290', ha='center', fontsize=10, color='gray')

    ax.text(5, 4, 'Total: 101,770 parameters!', ha='center', fontsize=16, fontweight='bold', color='#264653',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='#264653', lw=2))
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(0, 5); ax.axis('off')
    ax.set_title('MNIST Classifier: Parameter Count', fontsize=15, fontweight='bold')
    save(fig, 'parameter_counting')


# ============================================================
# 11. Loss landscape 3D
# ============================================================
def plot_loss_landscape():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    w1 = np.linspace(-3, 3, 50)
    w2 = np.linspace(-3, 3, 50)
    W1, W2 = np.meshgrid(w1, w2)
    Loss = W1**2 + W2**2 + 0.5 * np.sin(3*W1) * np.sin(3*W2)
    ax.plot_surface(W1, W2, Loss, cmap='viridis', alpha=0.8, edgecolor='none', rstride=2, cstride=2)

    path_w = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0])
    path_l = [w**2 + w**2 + 0.5*np.sin(3*w)*np.sin(3*w) for w in path_w]
    ax.plot(path_w, path_w, path_l, 'o-', color='#e85a4f', linewidth=3, markersize=10)
    ax.set_xlabel('Weight 1'); ax.set_ylabel('Weight 2'); ax.set_zlabel('Loss')
    ax.set_title('Loss Landscape: Gradient Descent', fontsize=14, fontweight='bold')
    save(fig, 'loss_landscape_3d')


# ============================================================
# 12. Universal approximation
# ============================================================
def plot_universal_approximation():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.linspace(0, 4, 200)
    y_true = np.sin(x * 3) + 0.5 * x

    for ax, (n, title, color) in zip(axes,
            [(2, '2 neurons', '#e85a4f'), (10, '10 neurons', '#e9c46a'), (50, '50 neurons', '#2a9d8f')]):
        ax.plot(x, y_true, '--', color='#264653', lw=2, label='Target', alpha=0.7)
        np.random.seed(42)
        y_pred = np.zeros_like(x)
        for _ in range(n):
            w, b, a = np.random.randn()*2, np.random.randn()*2, np.random.randn()
            y_pred += a * (1 / (1 + np.exp(-w * x + b)))
        y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
        y_pred = y_pred * (y_true.max() - y_true.min()) + y_true.min()
        ax.plot(x, y_pred, '-', color=color, lw=2.5, label=f'NN ({n} neurons)')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        ax.set_xlabel('x'); ax.set_ylabel('f(x)')

    plt.suptitle('Universal Approximation: More Neurons = Better Fit', fontsize=15, fontweight='bold')
    plt.tight_layout()
    save(fig, 'universal_approximation')


if __name__ == '__main__':
    np.random.seed(42)
    print("Generating diagrams for Lecture 05...")
    plot_bio_vs_artificial()
    plot_decision_boundaries()
    plot_xor_hidden_space()
    plot_activations()
    plot_nonlinearity()
    plot_mlp_architecture()
    plot_paradigm_change()
    plot_forward_pass()
    plot_softmax()
    plot_parameter_counting()
    plot_loss_landscape()
    plot_universal_approximation()
    print("Done! Output in diagrams/png/ and diagrams/svg/")
