import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Create directories if they don't exist
os.makedirs('diagrams', exist_ok=True)
os.makedirs('diagrams/svg', exist_ok=True)

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

# 5. Embedding Clusters (Word2Vec style visualization)
def plot_embedding_clusters():
    fig, ax = create_figure(figsize=(12, 10))
    ax.axis('on')

    # Mock data for word embeddings projected to 2D
    clusters = {
        'Countries': {
            'words': ['India', 'China', 'Japan', 'France', 'Germany'],
            'positions': [(7, 8), (7.5, 8.3), (7.2, 7.5), (6.8, 8.1), (7.3, 7.8)],
            'color': COLORS['blue']
        },
        'Animals': {
            'words': ['dog', 'cat', 'tiger', 'lion', 'bear'],
            'positions': [(2, 6), (2.3, 6.4), (2.5, 5.8), (2.8, 6.1), (2.1, 5.5)],
            'color': COLORS['red']
        },
        'Sports': {
            'words': ['football', 'cricket', 'basketball', 'tennis'],
            'positions': [(9, 5), (9.3, 5.4), (9.5, 4.8), (8.8, 5.2)],
            'color': COLORS['green']
        },
        'Food': {
            'words': ['pizza', 'burger', 'pasta', 'rice'],
            'positions': [(3, 2), (3.3, 2.5), (3.6, 1.8), (2.8, 2.2)],
            'color': COLORS['yellow']
        },
        'Technology': {
            'words': ['computer', 'phone', 'laptop', 'tablet'],
            'positions': [(8, 2), (8.3, 2.3), (8.5, 1.8), (7.8, 2.5)],
            'color': '#9b59b6'  # purple
        },
        'Colors': {
            'words': ['red', 'blue', 'green', 'yellow'],
            'positions': [(5, 4), (5.2, 4.3), (5.4, 3.8), (4.8, 4.1)],
            'color': '#e67e22'  # orange
        }
    }

    for cluster_name, data in clusters.items():
        for word, (x, y) in zip(data['words'], data['positions']):
            ax.scatter(x, y, c=data['color'], s=150, alpha=0.7, edgecolors='white', linewidth=1)
            ax.annotate(word, (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=10, color=data['color'], fontweight='bold')

        # Add cluster label
        cx = np.mean([p[0] for p in data['positions']])
        cy = np.max([p[1] for p in data['positions']]) + 0.6
        ax.text(cx, cy, cluster_name, fontsize=12, fontweight='bold',
                color=data['color'], ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=data['color']))

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10)
    ax.set_title("Word Embeddings in 2D (t-SNE projection)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Save in multiple formats
    plt.tight_layout()
    plt.savefig('diagrams/svg/embedding_clusters.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/embedding_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6. Neural Network Architecture
def plot_neural_net_architecture():
    fig, ax = create_figure(figsize=(10, 8))

    # Layer positions (y-coordinates, going down)
    layers = [
        ("Input: 'a' 'a' 'b'", 0.85, COLORS['blue']),
        ("Embedding Layer\na→[0.2,0.8]  b→[-0.5,0.1]", 0.68, COLORS['green']),
        ("Concatenate → [0.2, 0.8, 0.2, 0.8, -0.5, 0.1]", 0.51, '#607D8B'),
        ("Hidden Layer (100 neurons, ReLU)", 0.34, COLORS['yellow']),
        ("Output + Softmax → 27 probabilities", 0.17, '#FF8F00'),
    ]

    box_width = 0.7
    box_height = 0.11

    for label, y, color in layers:
        rect = patches.FancyBboxPatch((0.15, y), box_width, box_height,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color, edgecolor='black',
                                       alpha=0.8, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, y + box_height/2, label, ha='center', va='center',
                fontsize=14, fontweight='bold', color='white' if color != COLORS['yellow'] else 'black')

    # Arrows between layers
    for i in range(len(layers) - 1):
        y1 = layers[i][1]
        y2 = layers[i+1][1] + box_height
        ax.annotate('', xy=(0.5, y2), xytext=(0.5, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Output label
    ax.text(0.5, 0.05, "P('a')=0.05  P('i')=0.45  P('d')=0.30 ...",
            ha='center', fontsize=12, style='italic',
            bbox=dict(boxstyle='round', facecolor='#FFCCBC', edgecolor='#E64A19'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Neural Network for Next Character Prediction", fontsize=18, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('diagrams/svg/neural_net_architecture.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/neural_net_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

# 7. RNN Sequence Diagram
def plot_rnn_sequence():
    fig, ax = create_figure(figsize=(12, 6))

    words = ['I', 'love', 'pizza']
    hidden_labels = ['h0', 'h1', 'h2', 'h3']

    # Draw RNN cells
    cell_width = 0.18
    cell_height = 0.25
    start_x = 0.15
    y_pos = 0.4

    for i, word in enumerate(words):
        x = start_x + i * 0.28

        # Input word
        ax.text(x + cell_width/2, y_pos - 0.18, f'"{word}"', ha='center', va='center',
                fontsize=16, fontweight='bold', color=COLORS['blue'])

        # RNN cell
        rect = patches.FancyBboxPatch((x, y_pos), cell_width, cell_height,
                                       boxstyle="round,pad=0.02",
                                       facecolor=COLORS['green'], edgecolor='black',
                                       alpha=0.8, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + cell_width/2, y_pos + cell_height/2, 'RNN', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')

        # Arrow from input to cell
        ax.annotate('', xy=(x + cell_width/2, y_pos), xytext=(x + cell_width/2, y_pos - 0.12),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

        # Hidden state arrow (horizontal)
        if i < len(words) - 1:
            ax.annotate('', xy=(x + 0.28, y_pos + cell_height/2),
                       xytext=(x + cell_width + 0.02, y_pos + cell_height/2),
                       arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2))

        # Hidden state label above
        ax.text(x + cell_width/2, y_pos + cell_height + 0.08, hidden_labels[i+1],
                ha='center', va='center', fontsize=14, color=COLORS['red'], fontweight='bold')

    # Initial hidden state
    ax.text(start_x - 0.08, y_pos + cell_height/2, hidden_labels[0],
            ha='center', va='center', fontsize=14, color=COLORS['gray'])
    ax.annotate('', xy=(start_x, y_pos + cell_height/2),
               xytext=(start_x - 0.05, y_pos + cell_height/2),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))

    # Output arrow
    last_x = start_x + 2 * 0.28
    ax.annotate('', xy=(last_x + cell_width/2, y_pos + cell_height + 0.18),
               xytext=(last_x + cell_width/2, y_pos + cell_height + 0.02),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(last_x + cell_width/2, y_pos + cell_height + 0.25, 'Predict next',
            ha='center', fontsize=12, style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("RNN: Hidden State Carries Information Forward", fontsize=18, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('diagrams/svg/rnn_sequence.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/rnn_sequence.png', dpi=300, bbox_inches='tight')
    plt.close()

# 8. Attention Q, K, V
def plot_attention_qkv():
    fig, ax = create_figure(figsize=(12, 8))

    # Query box
    query_rect = patches.FancyBboxPatch((0.35, 0.75), 0.3, 0.15,
                                         boxstyle="round,pad=0.02",
                                         facecolor=COLORS['blue'], edgecolor='black',
                                         alpha=0.8, linewidth=2)
    ax.add_patch(query_rect)
    ax.text(0.5, 0.825, "QUERY (Q)\n'What opens doors?'", ha='center', va='center',
            fontsize=13, fontweight='bold', color='white')

    # Key-Value pairs
    keys = [("Alice", 0.1, "#ECEFF1", 0.1), ("key", 0.4, "#C8E6C9", 0.8), ("door", 0.7, "#ECEFF1", 0.3)]

    for name, x, color, score in keys:
        # Key box
        key_rect = patches.FancyBboxPatch((x, 0.4), 0.2, 0.12,
                                           boxstyle="round,pad=0.02",
                                           facecolor=color, edgecolor='black',
                                           alpha=0.9, linewidth=2 if name == "key" else 1)
        ax.add_patch(key_rect)
        ax.text(x + 0.1, 0.46, f"KEY: '{name}'", ha='center', va='center',
                fontsize=12, fontweight='bold')

        # Score
        ax.text(x + 0.1, 0.35, f"Score: {score}", ha='center', fontsize=11,
                color=COLORS['red'] if score > 0.5 else COLORS['gray'])

        # Value box
        val_rect = patches.FancyBboxPatch((x, 0.18), 0.2, 0.1,
                                           boxstyle="round,pad=0.02",
                                           facecolor=color, edgecolor='black',
                                           alpha=0.7, linewidth=1)
        ax.add_patch(val_rect)
        ax.text(x + 0.1, 0.23, f"VALUE\n[{name} embed]", ha='center', va='center',
                fontsize=10)

        # Arrow from query
        ax.annotate('', xy=(x + 0.1, 0.52), xytext=(0.5, 0.75),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1, alpha=0.5))

    # Output box
    out_rect = patches.FancyBboxPatch((0.3, 0.02), 0.4, 0.1,
                                       boxstyle="round,pad=0.02",
                                       facecolor='#FFECB3', edgecolor='#FF8F00',
                                       alpha=0.9, linewidth=2)
    ax.add_patch(out_rect)
    ax.text(0.5, 0.07, "OUTPUT: Weighted sum (mostly 'key')", ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Arrow to output
    ax.annotate('', xy=(0.5, 0.12), xytext=(0.5, 0.18),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Attention = Query-Key Matching + Value Retrieval", fontsize=18, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('diagrams/svg/attention_qkv.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/attention_qkv.png', dpi=300, bbox_inches='tight')
    plt.close()

# 9. Transformer Block
def plot_transformer_block():
    fig, ax = create_figure(figsize=(8, 10))

    components = [
        ("Input Embeddings", 0.88, '#E3F2FD', COLORS['blue']),
        ("Multi-Head\nSelf-Attention", 0.72, '#C8E6C9', COLORS['green']),
        ("Add & LayerNorm", 0.58, '#ECEFF1', '#607D8B'),
        ("Feed-Forward\nNetwork (MLP)", 0.42, '#FFF9C4', COLORS['yellow']),
        ("Add & LayerNorm", 0.28, '#ECEFF1', '#607D8B'),
        ("Output", 0.12, '#FFCCBC', COLORS['red']),
    ]

    box_width = 0.6
    box_heights = [0.08, 0.12, 0.08, 0.12, 0.08, 0.08]

    for i, (label, y, fill, edge) in enumerate(components):
        rect = patches.FancyBboxPatch((0.2, y), box_width, box_heights[i],
                                       boxstyle="round,pad=0.02",
                                       facecolor=fill, edgecolor=edge,
                                       alpha=0.9, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, y + box_heights[i]/2, label, ha='center', va='center',
                fontsize=13, fontweight='bold')

        # Arrows
        if i < len(components) - 1:
            next_y = components[i+1][1] + box_heights[i+1]
            ax.annotate('', xy=(0.5, next_y), xytext=(0.5, y),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Residual connections (curved arrows)
    # First residual
    ax.annotate('', xy=(0.82, 0.58 + 0.04), xytext=(0.82, 0.88),
               arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5,
                              connectionstyle='arc3,rad=0.3'))
    ax.text(0.92, 0.73, '+', fontsize=16, color=COLORS['red'], fontweight='bold')

    # Second residual
    ax.annotate('', xy=(0.82, 0.28 + 0.04), xytext=(0.82, 0.58),
               arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5,
                              connectionstyle='arc3,rad=0.3'))
    ax.text(0.92, 0.43, '+', fontsize=16, color=COLORS['red'], fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("One Transformer Block", fontsize=18, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('diagrams/svg/transformer_block.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/transformer_block.png', dpi=300, bbox_inches='tight')
    plt.close()

# 10. Gradient Descent (improved)
def plot_gradient_descent():
    fig, ax = create_figure(figsize=(10, 7))
    ax.axis('on')

    # Loss landscape
    x = np.linspace(-2.5, 2.5, 200)
    y = 0.5 * x**2 + 0.3 * np.sin(3*x)  # Non-convex for interest
    ax.plot(x, y, color='black', lw=3, label='Loss function')
    ax.fill_between(x, y, alpha=0.1, color='blue')

    # Gradient descent steps
    steps = [(-2.0, 0.5*(-2)**2 + 0.3*np.sin(-6)),
             (-1.2, 0.5*(-1.2)**2 + 0.3*np.sin(-3.6)),
             (-0.5, 0.5*(-0.5)**2 + 0.3*np.sin(-1.5)),
             (0.0, 0.5*(0)**2 + 0.3*np.sin(0))]
    colors = [COLORS['red'], COLORS['yellow'], COLORS['green'], COLORS['blue']]

    for i, ((bx, by), color) in enumerate(zip(steps, colors)):
        circle = plt.Circle((bx, by), 0.12, color=color, zorder=5)
        ax.add_patch(circle)
        ax.text(bx, by + 0.4, f"Step {i+1}", ha='center', fontsize=12, fontweight='bold')

        if i < len(steps) - 1:
            next_x, next_y = steps[i+1]
            ax.annotate('', xy=(next_x, next_y), xytext=(bx, by),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))

    ax.text(0, -0.8, "θ* (optimal)", ha='center', fontsize=14, fontweight='bold', color=COLORS['blue'])
    ax.set_xlabel("Parameters (θ)", fontsize=14)
    ax.set_ylabel("Loss L(θ)", fontsize=14)
    ax.set_title("Gradient Descent: Follow the Slope Downhill", fontsize=18, fontweight='bold')
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-1.2, 3.5)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig('diagrams/svg/gradient_descent.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/gradient_descent.png', dpi=300, bbox_inches='tight')
    plt.close()

# 11. Embedding Space (King-Queen analogy)
def plot_embedding_space():
    fig, ax = create_figure(figsize=(10, 8))
    ax.axis('on')

    # Word positions in embedding space
    words = {
        'king': (0.8, 0.9),
        'queen': (0.8, 0.2),
        'man': (0.2, 0.8),
        'woman': (0.2, 0.2),
    }

    colors = {'king': COLORS['blue'], 'queen': COLORS['red'],
              'man': COLORS['blue'], 'woman': COLORS['red']}

    # Plot words
    for word, (x, y) in words.items():
        ax.scatter(x, y, s=400, c=colors[word], zorder=5, edgecolors='black', linewidth=2)
        ax.text(x, y, word, ha='center', va='center', fontsize=14,
                fontweight='bold', color='white')

    # Draw arrows showing the relationships
    # man -> woman (gender direction)
    ax.annotate('', xy=(0.2, 0.25), xytext=(0.2, 0.75),
               arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=3))
    ax.text(0.08, 0.5, '−male\n+female', fontsize=11, ha='center', color=COLORS['green'])

    # king -> queen (same direction!)
    ax.annotate('', xy=(0.8, 0.25), xytext=(0.8, 0.85),
               arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=3))
    ax.text(0.92, 0.55, '−male\n+female', fontsize=11, ha='center', color=COLORS['green'])

    # man -> king (royalty direction)
    ax.annotate('', xy=(0.75, 0.85), xytext=(0.25, 0.8),
               arrowprops=dict(arrowstyle='->', color=COLORS['yellow'], lw=3))
    ax.text(0.5, 0.92, '+royalty', fontsize=11, ha='center', color=COLORS['yellow'])

    # woman -> queen (same direction!)
    ax.annotate('', xy=(0.75, 0.2), xytext=(0.25, 0.2),
               arrowprops=dict(arrowstyle='->', color=COLORS['yellow'], lw=3))
    ax.text(0.5, 0.08, '+royalty', fontsize=11, ha='center', color=COLORS['yellow'])

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Dimension 1 (royalty?)", fontsize=14)
    ax.set_ylabel("Dimension 2 (gender?)", fontsize=14)
    ax.set_title("Word Embeddings: king − man + woman ≈ queen", fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('diagrams/svg/embedding_space.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/embedding_space.png', dpi=300, bbox_inches='tight')
    plt.close()

# 12. Sampling Tree
def plot_sampling_tree():
    fig, ax = create_figure(figsize=(12, 8))

    # Tree structure
    levels = {
        0: [("The", 0.5, 0.9)],
        1: [("cat", 0.25, 0.7), ("dog", 0.75, 0.7)],
        2: [("sat", 0.12, 0.5), ("ran", 0.38, 0.5), ("barked", 0.62, 0.5), ("slept", 0.88, 0.5)],
    }

    colors = [COLORS['blue'], COLORS['green'], COLORS['yellow']]

    for level, nodes in levels.items():
        for word, x, y in nodes:
            # Draw node
            circle = plt.Circle((x, y), 0.06, color=colors[level], zorder=5, ec='black', lw=2)
            ax.add_patch(circle)
            ax.text(x, y, word, ha='center', va='center', fontsize=11, fontweight='bold', color='white')

            # Draw edges to children
            if level < 2:
                child_nodes = levels[level + 1]
                start_idx = 0 if level == 0 else (nodes.index((word, x, y)) * 2)
                for i in range(2):
                    if start_idx + i < len(child_nodes):
                        cx, cy = child_nodes[start_idx + i][1], child_nodes[start_idx + i][2]
                        ax.plot([x, cx], [y - 0.06, cy + 0.06], 'k-', lw=1.5, alpha=0.5)
                        # Add probability
                        mid_x, mid_y = (x + cx) / 2, (y + cy) / 2 - 0.03
                        prob = 0.6 if i == 0 else 0.4
                        ax.text(mid_x, mid_y, f'{prob:.1f}', fontsize=9, ha='center', color='gray')

    ax.set_xlim(0, 1)
    ax.set_ylim(0.35, 1)
    ax.set_title("Sampling Creates Different Outputs Each Time", fontsize=18, fontweight='bold', pad=20)
    ax.text(0.5, 0.38, "Each path = different generated text", ha='center', fontsize=12, style='italic')

    plt.tight_layout()
    plt.savefig('diagrams/svg/sampling_tree.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/sampling_tree.png', dpi=300, bbox_inches='tight')
    plt.close()

# 13. Bigram Heatmap (improved with real character counts)
def plot_bigram_heatmap_visual():
    fig, ax = create_figure(figsize=(10, 9))
    ax.axis('on')

    # Characters including start/end token
    chars = ['.', 'a', 'b', 'i', 'd', 'n', 'p', 'r', 'y', 'z']

    # Mock transition counts based on names like "aabid", "priya", "nipun", "zeel"
    # Rows = current char, Cols = next char
    data = np.array([
        [0.0, 0.4, 0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.3],  # . -> start of name
        [0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 0.0],  # a
        [0.0, 0.1, 0.0, 0.6, 0.0, 0.0, 0.0, 0.1, 0.0, 0.2],  # b
        [0.1, 0.1, 0.0, 0.0, 0.3, 0.1, 0.2, 0.0, 0.2, 0.0],  # i
        [0.5, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4],  # d -> often ends
        [0.2, 0.0, 0.0, 0.4, 0.0, 0.0, 0.2, 0.0, 0.0, 0.2],  # n
        [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.5, 0.3, 0.0],  # p
        [0.0, 0.1, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0],  # r
        [0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],  # y -> often ends
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # z (placeholder)
    ])

    im = ax.imshow(data, cmap='Blues', aspect='auto')

    ax.set_xticks(np.arange(len(chars)))
    ax.set_yticks(np.arange(len(chars)))
    ax.set_xticklabels(chars, fontsize=14, fontweight='bold')
    ax.set_yticklabels(chars, fontsize=14, fontweight='bold')

    # Text annotations
    for i in range(len(chars)):
        for j in range(len(chars)):
            if data[i, j] > 0.01:
                color = 'white' if data[i, j] > 0.4 else 'black'
                ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center',
                       fontsize=10, color=color, fontweight='bold')

    ax.set_xlabel("Next Character", fontsize=14, fontweight='bold')
    ax.set_ylabel("Current Character", fontsize=14, fontweight='bold')
    ax.set_title("Bigram Probability Table\nP(Next | Current)", fontsize=18, fontweight='bold', pad=15)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Probability', fontsize=12)

    plt.tight_layout()
    plt.savefig('diagrams/svg/bigram_heatmap.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/bigram_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# 14. Temperature Distribution Comparison
def plot_temperature_visual():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS['light_bg'])

    words = ['pizza', 'pasta', 'salad', 'cake', 'fruit']
    logits = np.array([2.5, 2.0, 1.0, 0.5, 0.2])

    def softmax(x, T):
        e_x = np.exp(x / T)
        return e_x / e_x.sum()

    temps = [0.2, 1.0, 2.0]
    titles = ["T = 0.2 (Focused)", "T = 1.0 (Balanced)", "T = 2.0 (Creative)"]
    colors_list = [COLORS['blue'], COLORS['green'], COLORS['red']]

    for ax, T, title, color in zip(axes, temps, titles, colors_list):
        ax.set_facecolor(COLORS['light_bg'])
        probs = softmax(logits, T)

        bars = ax.barh(words, probs, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Probability", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', color=color)

        for bar, p in zip(bars, probs):
            if p > 0.05:
                ax.text(p + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{p:.2f}', va='center', fontsize=10)

        ax.grid(True, axis='x', alpha=0.3)

    plt.suptitle('Temperature Controls "Creativity" in Sampling', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('diagrams/svg/temperature_visual.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/temperature_visual.png', dpi=300, bbox_inches='tight')
    plt.close()

# 15. BPE Tokenization Visualization
def plot_bpe_tokenization():
    fig, ax = create_figure(figsize=(14, 8))

    # Show BPE merge process
    ax.text(0.5, 0.95, "Byte Pair Encoding (BPE) Tokenization", ha='center',
            fontsize=20, fontweight='bold')

    # Original text
    ax.text(0.05, 0.82, "Original:", fontsize=12, fontweight='bold')
    original = "unhappiness"
    x_pos = 0.2
    for i, char in enumerate(original):
        rect = patches.FancyBboxPatch((x_pos, 0.78), 0.045, 0.08,
                                       boxstyle="round,pad=0.01",
                                       facecolor=COLORS['gray'], alpha=0.3)
        ax.add_patch(rect)
        ax.text(x_pos + 0.0225, 0.82, char, ha='center', fontsize=14, fontweight='bold')
        x_pos += 0.05

    # After merges
    stages = [
        ("Step 1:", ["u", "n", "h", "a", "pp", "i", "n", "e", "ss"], 0.65, "'pp' merged"),
        ("Step 2:", ["un", "h", "a", "pp", "i", "n", "e", "ss"], 0.50, "'un' merged"),
        ("Step 3:", ["un", "happ", "i", "n", "e", "ss"], 0.35, "'happ' merged"),
        ("Final:", ["un", "happiness"], 0.18, "Common subwords!"),
    ]

    token_colors = [COLORS['blue'], COLORS['green'], COLORS['red'], COLORS['yellow']]

    for label, tokens, y, note in stages:
        ax.text(0.05, y + 0.04, label, fontsize=11, fontweight='bold')
        x_pos = 0.2
        for i, tok in enumerate(tokens):
            width = 0.04 + len(tok) * 0.025
            color = token_colors[i % len(token_colors)]
            rect = patches.FancyBboxPatch((x_pos, y), width, 0.08,
                                           boxstyle="round,pad=0.01",
                                           facecolor=color, alpha=0.6)
            ax.add_patch(rect)
            ax.text(x_pos + width/2, y + 0.04, tok, ha='center', fontsize=12, fontweight='bold', color='white')
            x_pos += width + 0.015

        ax.text(0.85, y + 0.04, note, fontsize=10, style='italic', color='gray')

    # Key insight
    insight_box = patches.FancyBboxPatch((0.1, 0.02), 0.8, 0.1,
                                          boxstyle="round,pad=0.02",
                                          facecolor='#E3F2FD', edgecolor=COLORS['blue'], linewidth=2)
    ax.add_patch(insight_box)
    ax.text(0.5, 0.07, 'BPE finds common subwords: "un" (prefix) + "happiness" (common word)',
            ha='center', fontsize=12, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('diagrams/svg/bpe_tokenization.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/bpe_tokenization.png', dpi=300, bbox_inches='tight')
    plt.close()

# 16. Sliding Window Training Data
def plot_sliding_window():
    fig, ax = create_figure(figsize=(12, 10))

    ax.text(0.5, 0.95, "Creating Training Data: Sliding Window", ha='center',
            fontsize=22, fontweight='bold')

    text = "aabid"

    # Show the text at top
    ax.text(0.05, 0.85, "Text:", fontsize=16, fontweight='bold')
    x_pos = 0.18
    for i, char in enumerate(text):
        rect = patches.FancyBboxPatch((x_pos, 0.80), 0.08, 0.10,
                                       boxstyle="round,pad=0.01",
                                       facecolor=COLORS['blue'], alpha=0.7)
        ax.add_patch(rect)
        ax.text(x_pos + 0.04, 0.85, char, ha='center', fontsize=22, fontweight='bold', color='white')
        x_pos += 0.10

    # Show sliding windows
    windows = [
        (['.', '.', '.'], 'a', 0.68),
        (['.', '.', 'a'], 'a', 0.54),
        (['.', 'a', 'a'], 'b', 0.40),
        (['a', 'a', 'b'], 'i', 0.26),
        (['a', 'b', 'i'], 'd', 0.12),
    ]

    ax.text(0.05, 0.75, "Context (Input)", fontsize=14, fontweight='bold', color=COLORS['green'])
    ax.text(0.58, 0.75, "Target", fontsize=14, fontweight='bold', color=COLORS['red'])

    for context, target, y in windows:
        # Context boxes
        x_pos = 0.18
        for char in context:
            rect = patches.FancyBboxPatch((x_pos, y - 0.05), 0.08, 0.10,
                                           boxstyle="round,pad=0.01",
                                           facecolor=COLORS['green'], alpha=0.6)
            ax.add_patch(rect)
            ax.text(x_pos + 0.04, y, char, ha='center', fontsize=20, fontweight='bold', color='white')
            x_pos += 0.10

        # Arrow
        ax.annotate('', xy=(0.55, y), xytext=(0.48, y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=3))

        # Target box
        rect = patches.FancyBboxPatch((0.58, y - 0.05), 0.08, 0.10,
                                       boxstyle="round,pad=0.01",
                                       facecolor=COLORS['red'], alpha=0.7)
        ax.add_patch(rect)
        ax.text(0.62, y, target, ha='center', fontsize=20, fontweight='bold', color='white')

        # Predict label
        ax.text(0.70, y, "← Predict!", fontsize=14, va='center', color='gray', fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('diagrams/svg/sliding_window.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/sliding_window.png', dpi=300, bbox_inches='tight')
    plt.close()

# 17. LLM Recipe Overview
def plot_llm_recipe():
    fig, ax = create_figure(figsize=(12, 10))

    ax.text(0.5, 0.95, "The LLM Recipe", ha='center', fontsize=22, fontweight='bold')

    ingredients = [
        ("1. DATA", 0.78, COLORS['blue'], "Trillions of tokens from internet, books, code"),
        ("2. ARCHITECTURE", 0.58, COLORS['green'], "Transformer: Attention + Feed-Forward, stacked"),
        ("3. TRAINING", 0.38, COLORS['yellow'], "Next token prediction on massive compute"),
        ("4. ALIGNMENT", 0.18, COLORS['red'], "SFT + RLHF to follow instructions safely"),
    ]

    for label, y, color, desc in ingredients:
        # Main box
        rect = patches.FancyBboxPatch((0.08, y - 0.07), 0.25, 0.14,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color, edgecolor='black', alpha=0.8, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.2, y, label, ha='center', va='center', fontsize=14, fontweight='bold', color='white')

        # Description
        ax.text(0.38, y, desc, va='center', fontsize=12)

    # Arrow connecting all
    ax.annotate('', xy=(0.2, 0.25), xytext=(0.2, 0.71),
               arrowprops=dict(arrowstyle='->', color='gray', lw=3, ls='--'))

    # Result
    result_box = patches.FancyBboxPatch((0.25, 0.02), 0.5, 0.08,
                                         boxstyle="round,pad=0.02",
                                         facecolor='#9b59b6', edgecolor='black', alpha=0.8, linewidth=2)
    ax.add_patch(result_box)
    ax.text(0.5, 0.06, "= ChatGPT / Claude / Gemini", ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('diagrams/svg/llm_recipe.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/llm_recipe.png', dpi=300, bbox_inches='tight')
    plt.close()

# 18. LLM Training Pipeline
def plot_llm_training_pipeline():
    fig, ax = create_figure(figsize=(16, 6))

    stages = [
        ("Pre-training", 0.12, COLORS['blue'], "Internet text\n→ Base model"),
        ("SFT", 0.38, COLORS['green'], "Instruction data\n→ Follows orders"),
        ("RLHF/DPO", 0.62, COLORS['yellow'], "Human feedback\n→ Safe & helpful"),
        ("ChatGPT!", 0.88, COLORS['red'], "Ready to\ndeploy"),
    ]

    for label, x, color, desc in stages:
        rect = patches.FancyBboxPatch((x - 0.10, 0.35), 0.20, 0.30,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color, edgecolor='black', alpha=0.8, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 0.55, label, ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        ax.text(x, 0.42, desc, ha='center', va='center', fontsize=10, color='white')

    # Arrows
    for i in range(len(stages) - 1):
        x1 = stages[i][1] + 0.10
        x2 = stages[i+1][1] - 0.10
        ax.annotate('', xy=(x2, 0.5), xytext=(x1, 0.5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=3))

    # Labels below
    ax.text(0.12, 0.2, "~$100M\nMonths", ha='center', fontsize=10, color='gray')
    ax.text(0.38, 0.2, "~$1M\nDays", ha='center', fontsize=10, color='gray')
    ax.text(0.62, 0.2, "~$10M\nWeeks", ha='center', fontsize=10, color='gray')

    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 0.8)
    ax.set_title("LLM Training Pipeline: From Text to Assistant", fontsize=18, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('diagrams/svg/llm_training_pipeline.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/llm_training_pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()

# 19. RLHF Pipeline
def plot_rlhf_pipeline():
    fig, ax = create_figure(figsize=(14, 8))

    ax.text(0.5, 0.95, "RLHF: Reinforcement Learning from Human Feedback", ha='center',
            fontsize=18, fontweight='bold')

    steps = [
        ("1. Generate", 0.15, 0.65, COLORS['blue'], "LLM produces\nmultiple responses"),
        ("2. Rank", 0.45, 0.65, COLORS['green'], "Humans rank:\nA > B > C"),
        ("3. Reward Model", 0.75, 0.65, COLORS['yellow'], "Train model to\npredict rankings"),
        ("4. Optimize", 0.45, 0.25, COLORS['red'], "Use PPO to maximize\nreward while staying\nclose to base model"),
    ]

    for label, x, y, color, desc in steps:
        rect = patches.FancyBboxPatch((x - 0.12, y - 0.12), 0.24, 0.24,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color, edgecolor='black', alpha=0.8, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y + 0.04, label, ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        ax.text(x, y - 0.04, desc, ha='center', va='center', fontsize=9, color='white')

    # Arrows
    ax.annotate('', xy=(0.33, 0.65), xytext=(0.27, 0.65),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(0.63, 0.65), xytext=(0.57, 0.65),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(0.57, 0.35), xytext=(0.75, 0.53),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(0.27, 0.53), xytext=(0.33, 0.35),
               arrowprops=dict(arrowstyle='->', color='black', lw=2, ls='--'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1)

    plt.tight_layout()
    plt.savefig('diagrams/svg/rlhf_pipeline.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/rlhf_pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()

# Run all the Matplotlib diagrams
if __name__ == "__main__":
    print("Generating diagrams...")
    plot_embeddings()
    print("  - ntp_embeddings.png")
    plot_temperature()
    print("  - ntp_temperature.png")
    plot_bigram_heatmap()
    print("  - ntp_bigram.png")
    plot_loss_descent()
    print("  - ntp_descent.png")
    plot_embedding_clusters()
    print("  - embedding_clusters.svg/png")
    plot_neural_net_architecture()
    print("  - neural_net_architecture.svg/png")
    plot_rnn_sequence()
    print("  - rnn_sequence.svg/png")
    plot_attention_qkv()
    print("  - attention_qkv.svg/png")
    plot_transformer_block()
    print("  - transformer_block.svg/png")
    plot_gradient_descent()
    print("  - gradient_descent.svg/png")
    plot_embedding_space()
    print("  - embedding_space.svg/png")
    plot_sampling_tree()
    print("  - sampling_tree.svg/png")
    # New diagrams
    plot_bigram_heatmap_visual()
    print("  - bigram_heatmap.svg/png")
    plot_temperature_visual()
    print("  - temperature_visual.svg/png")
    plot_bpe_tokenization()
    print("  - bpe_tokenization.svg/png")
    plot_sliding_window()
    print("  - sliding_window.svg/png")
    plot_llm_recipe()
    print("  - llm_recipe.svg/png")
    plot_llm_training_pipeline()
    print("  - llm_training_pipeline.svg/png")
    plot_rlhf_pipeline()
    print("  - rlhf_pipeline.svg/png")
    print("Done!")