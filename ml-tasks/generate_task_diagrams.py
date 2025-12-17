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
    'purple': '#9D4EDD',
    'gray': '#6c757d',
    'light_bg': '#f8f9fa'
}

def create_figure(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(COLORS['light_bg'])
    fig.patch.set_facecolor(COLORS['light_bg'])
    ax.axis('off')
    return fig, ax

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f'diagrams/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

# 1. CV Taxonomy: Classification vs Detection vs Segmentation
def plot_cv_taxonomy():
    fig, ax = create_figure(figsize=(12, 4))
    
    # 1. Classification
    ax.text(2, 3.5, "Classification", fontweight='bold', ha='center', fontsize=12)
    rect = patches.Rectangle((1, 1), 2, 2, facecolor='white', edgecolor='black')
    ax.add_patch(rect)
    # Cat face
    circle = patches.Circle((2, 2), 0.6, facecolor=COLORS['yellow'])
    ax.add_patch(circle)
    ax.text(2, 0.5, "Input: Image\nOutput: 'Cat'", ha='center', fontsize=10)

    # 2. Detection
    ax.text(6, 3.5, "Object Detection", fontweight='bold', ha='center', fontsize=12)
    rect = patches.Rectangle((5, 1), 2, 2, facecolor='white', edgecolor='black')
    ax.add_patch(rect)
    # Cat face
    circle = patches.Circle((6, 2), 0.6, facecolor=COLORS['yellow'])
    ax.add_patch(circle)
    # Box
    box = patches.Rectangle((5.3, 1.3), 1.4, 1.4, facecolor='none', edgecolor=COLORS['red'], lw=2)
    ax.add_patch(box)
    ax.text(6, 0.5, "Input: Image\nOutput: Box (x,y,w,h)", ha='center', fontsize=10)

    # 3. Segmentation
    ax.text(10, 3.5, "Segmentation", fontweight='bold', ha='center', fontsize=12)
    rect = patches.Rectangle((9, 1), 2, 2, facecolor='white', edgecolor='black')
    ax.add_patch(rect)
    # Cat face (pixel wise)
    circle = patches.Circle((10, 2), 0.6, facecolor=COLORS['purple'], alpha=0.8) # Mask
    ax.add_patch(circle)
    ax.text(10, 0.5, "Input: Image\nOutput: Pixel Mask", ha='center', fontsize=10)

    save_plot('task_cv_types.png')

# 2. NLP: NER Visualization
def plot_ner():
    fig, ax = create_figure(figsize=(10, 3))
    
    sentence = [("Sundar", "PER"), ("Pichai", "PER"), ("works", "O"), ("at", "O"), ("Google", "ORG"), ("in", "O"), ("California", "LOC")]
    
    x_pos = 0.5
    for word, tag in sentence:
        if tag == "O":
            ax.text(x_pos, 1.5, word, fontsize=14)
            width = len(word) * 0.2 + 0.2
        else:
            color = COLORS['blue'] if tag == "PER" else (COLORS['green'] if tag == "ORG" else COLORS['yellow'])
            
            # Draw highlight box
            width = len(word) * 0.2 + 0.4
            rect = patches.Rectangle((x_pos-0.1, 1.2), width, 0.8, facecolor=color, alpha=0.3, edgecolor=color)
            ax.add_patch(rect)
            
            ax.text(x_pos, 1.5, word, fontsize=14, fontweight='bold')
            ax.text(x_pos, 1.0, tag, fontsize=8, color=color, fontweight='bold')
            
        x_pos += width + 0.1

    ax.set_xlim(0, x_pos)
    ax.set_ylim(0, 3)
    ax.text(x_pos/2, 2.5, "Named Entity Recognition (NER)", ha='center', fontweight='bold', fontsize=16)
    save_plot('task_nlp_ner.png')

# 3. Unsupervised: Clustering vs Anomaly
def plot_unsupervised():
    fig, ax = create_figure(figsize=(10, 5))
    
    # Clustering
    ax.text(2.5, 4.5, "Clustering (K-Means)", fontweight='bold', ha='center')
    
    # Cluster 1
    x1 = np.random.normal(1.5, 0.3, 20)
    y1 = np.random.normal(2.5, 0.3, 20)
    ax.scatter(x1, y1, c=COLORS['blue'], s=30)
    
    # Cluster 2
    x2 = np.random.normal(3.5, 0.3, 20)
    y2 = np.random.normal(2.5, 0.3, 20)
    ax.scatter(x2, y2, c=COLORS['red'], s=30)
    
    ax.text(2.5, 1, "Group similar data\nwithout labels", ha='center', fontsize=10, style='italic')

    # Divider
    ax.plot([5, 5], [0, 5], color='gray', linestyle='--')

    # Anomaly
    ax.text(7.5, 4.5, "Anomaly Detection", fontweight='bold', ha='center')
    
    # Normal data
    xn = np.random.normal(7.5, 0.4, 40)
    yn = np.random.normal(2.5, 0.4, 40)
    ax.scatter(xn, yn, c=COLORS['green'], s=30, alpha=0.5, label='Normal')
    
    # Anomaly
    ax.scatter([8.5], [4.0], c=COLORS['red'], s=100, marker='x', lw=3, label='Outlier')
    ax.text(8.7, 4.1, "Fraud!", color=COLORS['red'])
    
    ax.text(7.5, 1, "Find the odd one out", ha='center', fontsize=10, style='italic')

    save_plot('task_unsupervised.png')

# 4. Generative: Image Inpainting
def plot_inpainting():
    fig, ax = create_figure(figsize=(8, 4))
    
    # Input
    ax.text(2, 3.5, "Input: Damaged", ha='center', fontweight='bold')
    rect = patches.Rectangle((1, 1), 2, 2, facecolor='white', edgecolor='black')
    ax.add_patch(rect)
    # Scene
    rect_sky = patches.Rectangle((1, 2), 2, 1, facecolor=COLORS['blue'], alpha=0.5)
    ax.add_patch(rect_sky)
    rect_gnd = patches.Rectangle((1, 1), 2, 1, facecolor=COLORS['green'], alpha=0.5)
    ax.add_patch(rect_gnd)
    # Hole
    rect_hole = patches.Rectangle((1.5, 1.5), 1, 1, facecolor='black')
    ax.add_patch(rect_hole)
    
    # Arrow
    ax.arrow(3.5, 2, 1, 0, head_width=0.2, color='gray')
    ax.text(4, 2.2, "Generative\nModel", ha='center', fontsize=8)
    
    # Output
    ax.text(6, 3.5, "Output: Restored", ha='center', fontweight='bold')
    rect = patches.Rectangle((5, 1), 2, 2, facecolor='white', edgecolor='black')
    ax.add_patch(rect)
    # Scene restored
    rect_sky2 = patches.Rectangle((5, 2), 2, 1, facecolor=COLORS['blue'], alpha=0.5)
    ax.add_patch(rect_sky2)
    rect_gnd2 = patches.Rectangle((5, 1), 2, 1, facecolor=COLORS['green'], alpha=0.5)
    ax.add_patch(rect_gnd2)
    # Tree (imagined)
    circle = patches.Circle((6, 2), 0.4, facecolor=COLORS['green'])
    ax.add_patch(circle)
    rect_trunk = patches.Rectangle((5.9, 1.5), 0.2, 0.5, facecolor='brown')
    ax.add_patch(rect_trunk)
    
    ax.text(4, 0.5, "Model 'hallucinates' missing parts", ha='center', fontsize=10)
    save_plot('task_inpainting.png')

# 5. RL Loop
def plot_rl_loop():
    fig, ax = create_figure(figsize=(8, 5))
    
    # Agent
    circle = patches.Circle((2, 2.5), 0.8, facecolor=COLORS['blue'], alpha=0.3, edgecolor=COLORS['blue'])
    ax.add_patch(circle)
    ax.text(2, 2.5, "Agent\n(Model)", ha='center', va='center', fontweight='bold')
    
    # Environment
    rect = patches.Rectangle((6, 1.5), 2, 2, facecolor=COLORS['green'], alpha=0.3, edgecolor=COLORS['green'])
    ax.add_patch(rect)
    ax.text(7, 2.5, "Environment\n(Game/World)", ha='center', va='center', fontweight='bold')
    
    # Arrows
    # Action (Top)
    ax.annotate("", xy=(6, 3), xytext=(2.8, 3), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", lw=2))
    ax.text(4.4, 3.8, "Action", ha='center', fontweight='bold')
    
    # Reward/State (Bottom)
    ax.annotate("", xy=(2.8, 2), xytext=(6, 2), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", lw=2, linestyle="--"))
    ax.text(4.4, 1.2, "State + Reward", ha='center', fontweight='bold')
    
    save_plot('task_rl.png')

if __name__ == "__main__":
    plot_cv_taxonomy()
    plot_ner()
    plot_unsupervised()
    plot_inpainting()
    plot_rl_loop()
