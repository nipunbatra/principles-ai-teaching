import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Create directory if it doesn't exist
os.makedirs('diagrams', exist_ok=True)

# Common Style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
COLORS = {
    'blue': '#2E86AB',
    'green': '#06A77D',
    'red': '#E63946',
    'yellow': '#F6AE2D',
    'gray': '#6c757d',
    'light_bg': '#f8f9fa'
}

def create_figure(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(COLORS['light_bg'])
    fig.patch.set_facecolor(COLORS['light_bg'])
    ax.axis('off') # Most plots will be conceptual, turn off axes by default
    return fig, ax

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f'diagrams/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

# 1. Embeddings Visual (Scatter Plot)
def plot_embeddings():
    fig, ax = create_figure(figsize=(8, 8))
    ax.axis('on') # Turn on axes for scatter plot
    
    # Mock data illustrating clusters
    # Vowels
    vowels = {'a': (2, 2), 'e': (2.5, 2.2), 'i': (2.2, 1.8), 'o': (2.8, 2), 'u': (2.5, 1.5)}
    # Consonants (some)
    consonants = {'b': (5, 5), 'c': (5.2, 4.8), 'd': (4.8, 5.2), 'k': (5.5, 5), 'p': (5.1, 5.3)}
    # End token
    special = {'.': (8, 8)}

    for char, (x, y) in vowels.items():
        ax.scatter(x, y, c=COLORS['red'], s=200, alpha=0.6)
        ax.text(x, y, char, fontsize=16, fontweight='bold', ha='center', va='center', color='white')

    for char, (x, y) in consonants.items():
        ax.scatter(x, y, c=COLORS['blue'], s=200, alpha=0.6)
        ax.text(x, y, char, fontsize=16, fontweight='bold', ha='center', va='center', color='white')
        
    for char, (x, y) in special.items():
        ax.scatter(x, y, c=COLORS['gray'], s=200, alpha=0.6)
        ax.text(x, y, char, fontsize=16, fontweight='bold', ha='center', va='center', color='white')

    ax.set_title("Learned Embeddings (2D Projection)", fontsize=16)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, linestyle='--', alpha=0.3)
    
    ax.text(2.5, 3, "Vowels cluster together", color=COLORS['red'], fontsize=10, ha='center')
    ax.text(5, 4, "Consonants cluster together", color=COLORS['blue'], fontsize=10, ha='center')

    save_plot('ntp_embeddings.png')

# 2. Temperature Visual (Bar Chart)
def plot_temperature():
    fig, ax = create_figure(figsize=(10, 5))
    
    logits = np.array([2.0, 1.0, 0.1, 0.5])
    labels = ['a', 'b', 'c', 'd']
    
    def softmax(x, T):
        e_x = np.exp(x / T)
        return e_x / e_x.sum()

    temps = [0.1, 1.0, 2.0]
    positions = [0, 5, 10]
    titles = ["Low Temp (0.1)\nConservative", "Temp (1.0)\nNormal", "High Temp (2.0)\nCreative/Random"]
    colors = [COLORS['blue'], COLORS['green'], COLORS['red']]

    for i, (T, pos, title, col) in enumerate(zip(temps, positions, titles, colors)):
        probs = softmax(logits, T)
        
        # Draw bars
        for j, p in enumerate(probs):
            rect = patches.Rectangle((pos + j, 0), 0.8, p*3, facecolor=col, alpha=0.8)
            ax.add_patch(rect)
            ax.text(pos + j + 0.4, -0.3, labels[j], ha='center', fontsize=12)
            if p > 0.1:
                ax.text(pos + j + 0.4, p*3 + 0.1, f"{p:.2f}", ha='center', fontsize=9)

        ax.text(pos + 1.5, 3.5, title, ha='center', fontweight='bold', fontsize=12)

    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 4)
    save_plot('ntp_temperature.png')

# 3. Bigram Heatmap (Heatmap)
def plot_bigram_heatmap():
    fig, ax = create_figure(figsize=(8, 8))
    ax.axis('on') # Turn on axes for heatmap
    
    # Mock Bigram counts for 'a', 'b', '.'
    data = np.array([
        [10, 50, 5],  # a -> a, b, .
        [5, 5, 40],   # b -> a, b, .
        [50, 5, 5]    # . -> a, b, .
    ])
    
    # Normalize to probabilities
    data = data / data.sum(axis=1, keepdims=True)
    
    im = ax.imshow(data, cmap='Blues')
    
    labels = ['a', 'b', '.']
    
    # Axis labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_yticklabels(labels, fontsize=16)
    
    ax.set_title("Bigram Probabilities\nP(Next | Current)", fontsize=18, pad=20)
    ax.set_xlabel("Next Character", fontsize=14)
    ax.set_ylabel("Current Character", fontsize=14)
    
    # Text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f"{data[i, j]:.2f}",
                           ha="center", va="center", color="black" if data[i, j] < 0.5 else "white", fontsize=14)

    save_plot('ntp_bigram.png')

# 4. Loss Descent (Curve Plot)
def plot_loss_descent():
    fig, ax = create_figure(figsize=(8, 6))
    ax.axis('on') # Turn on axes for curve plot
    
    x = np.linspace(-2, 2, 100)
    y = x**2
    ax.plot(x, y, color='black', lw=2)
    
    # Ball positions
    steps = [(-1.5, 2.25), (-1.0, 1.0), (-0.2, 0.04)]
    colors = [COLORS['red'], COLORS['yellow'], COLORS['green']]
    
    for i, (bx, by) in enumerate(steps):
        circle = patches.Circle((bx, by), 0.1, color=colors[i])
        ax.add_patch(circle)
        ax.text(bx+0.2, by, f"Step {i+1}", fontsize=10)
        
        if i < 2:
            next_x = steps[i+1][0]
            next_y = steps[i+1][1]
            ax.arrow(bx, by, (next_x-bx)*0.8, (next_y-by)*0.8, head_width=0.1, color='gray', ls='--')

    ax.set_title("Gradient Descent: Minimizing Loss", fontsize=14)
    ax.set_xlabel("Model Parameters", fontsize=12)
    ax.set_ylabel("Loss Function (Error)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    save_plot('ntp_descent.png')

# Run all the Matplotlib diagrams
if __name__ == "__main__":
    plot_embeddings()
    plot_temperature()
    plot_bigram_heatmap()
    plot_loss_descent()