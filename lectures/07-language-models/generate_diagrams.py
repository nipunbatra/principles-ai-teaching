#!/usr/bin/env python3
"""
Diagram generator for Lecture 07: Language Models
Visualizations for next token prediction, tokenization, embeddings, attention.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Import shared style
try:
    from themes.diagram_style import COLORS, setup_figure, save_svg
except ImportError:
    COLORS = {
        'primary': '#1e3a5f',
        'accent': '#e85a4f',
        'success': '#2a9d8f',
        'warning': '#e9c46a',
        'blue': '#3b82f6',
        'purple': '#8b5cf6',
        'text': '#2d3748',
        'text_light': '#4a5568',
        'bg_light': '#f7fafc',
    }

    def setup_figure(figsize=(12, 6), bg_color='white'):
        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
        ax.set_facecolor(bg_color)
        ax.axis('off')
        return fig, ax

    def save_svg(fig, filename):
        os.makedirs('diagrams/svg', exist_ok=True)
        filepath = f'diagrams/svg/{filename}'
        fig.savefig(filepath, format='svg', bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  ✓ {filename}")


# =============================================================================
# 1. Next Token Prediction Visualization
# =============================================================================
def create_next_token_prediction():
    """Show the core LLM task: predict next token."""
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
    ax.set_facecolor('white')

    # Input tokens
    tokens = ["The", "cat", "sat", "on", "the", "___"]
    colors = [COLORS['blue']] * 5 + [COLORS['warning']]

    # Draw token boxes
    x = 0.5
    for i, (token, color) in enumerate(zip(tokens, colors)):
        width = len(token) * 0.08 + 0.15
        rect = FancyBboxPatch((x, 0.55), width, 0.25,
                              boxstyle="round,pad=0.03,rounding_size=0.05",
                              facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + width/2, 0.675, token, ha='center', va='center',
               fontsize=14, fontweight='bold', color='white')
        x += width + 0.08

    # Arrow pointing to prediction
    ax.annotate('', xy=(x + 0.3, 0.45), xytext=(x - 0.1, 0.55),
               arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['accent']))

    # Probability distribution
    words = ['mat', 'floor', 'bed', 'table', 'chair']
    probs = [0.42, 0.25, 0.15, 0.10, 0.08]

    bar_x = 0.5
    bar_width = 0.12
    for word, prob in zip(words, probs):
        bar_height = prob * 0.5
        color = COLORS['success'] if word == 'mat' else COLORS['blue']
        alpha = 1.0 if word == 'mat' else 0.5
        rect = FancyBboxPatch((bar_x, 0.08), bar_width, bar_height,
                              boxstyle="round,pad=0.01",
                              facecolor=color, alpha=alpha, edgecolor='white')
        ax.add_patch(rect)
        ax.text(bar_x + bar_width/2, 0.03, word, ha='center', va='top',
               fontsize=10, rotation=45)
        ax.text(bar_x + bar_width/2, 0.08 + bar_height + 0.02, f'{prob:.0%}',
               ha='center', fontsize=9, fontweight='bold', color=color)
        bar_x += bar_width + 0.04

    # Labels
    ax.text(0.5, 0.9, 'Input: "The cat sat on the ___"', fontsize=16,
           fontweight='bold', color=COLORS['primary'])
    ax.text(0.5, 0.4, 'Model predicts probability of each word:', fontsize=12,
           color=COLORS['text'])

    ax.set_xlim(0.3, 1.5)
    ax.set_ylim(0, 1)
    ax.axis('off')

    save_svg(fig, 'next_token_prediction.svg')


# =============================================================================
# 2. Tokenization Visualization
# =============================================================================
def create_tokenization_visual():
    """Show how text becomes tokens."""
    fig, ax = plt.subplots(figsize=(14, 5), facecolor='white')
    ax.set_facecolor('white')

    # Original text
    ax.text(0.1, 0.85, 'Original Text:', fontsize=12, fontweight='bold',
           color=COLORS['text'])
    ax.text(0.1, 0.75, '"Hello, how are you doing today?"', fontsize=14,
           fontfamily='monospace', color=COLORS['primary'])

    # Arrow
    ax.annotate('', xy=(0.5, 0.55), xytext=(0.5, 0.65),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['accent']))
    ax.text(0.55, 0.60, 'Tokenize', fontsize=11, va='center', color=COLORS['accent'])

    # Tokens with IDs
    tokens = ['Hello', ',', ' how', ' are', ' you', ' doing', ' today', '?']
    ids = [15496, 11, 703, 527, 499, 3815, 3432, 30]
    colors_tok = [COLORS['blue'], COLORS['warning'], COLORS['success'],
                  COLORS['purple'], COLORS['accent'], COLORS['blue'],
                  COLORS['success'], COLORS['warning']]

    x = 0.1
    for token, tok_id, color in zip(tokens, ids, colors_tok):
        width = max(len(token) * 0.025 + 0.04, 0.06)

        # Token box
        rect = FancyBboxPatch((x, 0.35), width, 0.12,
                              boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + width/2, 0.41, token.replace(' ', '␣'),
               ha='center', va='center', fontsize=10, color='white',
               fontfamily='monospace', fontweight='bold')

        # ID below
        ax.text(x + width/2, 0.28, f'[{tok_id}]', ha='center',
               fontsize=8, color=COLORS['text_light'])

        x += width + 0.01

    # Explanation
    ax.text(0.1, 0.15, 'Each token gets a unique ID (index in vocabulary)',
           fontsize=11, color=COLORS['text'])
    ax.text(0.1, 0.07, 'Common words = single token, rare words = split into subwords',
           fontsize=10, color=COLORS['text_light'], style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    save_svg(fig, 'tokenization.svg')


# =============================================================================
# 3. Temperature Effect on Sampling
# =============================================================================
def create_temperature_comparison():
    """Show how temperature affects output distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor='white')

    words = ['mat', 'floor', 'bed', 'table', 'chair']
    base_logits = np.array([2.0, 1.2, 0.8, 0.4, 0.2])

    temperatures = [0.3, 1.0, 2.0]
    titles = ['Low Temperature (0.3)\nMore Focused',
              'Default (1.0)\nBalanced',
              'High Temperature (2.0)\nMore Random']

    for ax, temp, title in zip(axes, temperatures, titles):
        ax.set_facecolor('white')

        # Apply temperature
        scaled = base_logits / temp
        probs = np.exp(scaled) / np.exp(scaled).sum()

        # Color based on probability
        colors = [COLORS['success'] if p == max(probs) else COLORS['blue']
                  for p in probs]
        alphas = [0.9 if p == max(probs) else 0.5 for p in probs]

        bars = ax.bar(words, probs, color=colors, edgecolor='white', linewidth=2)
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)

        ax.set_ylim(0, 0.9)
        ax.set_title(title, fontsize=11, fontweight='bold', color=COLORS['primary'])
        ax.set_ylabel('Probability')
        ax.tick_params(axis='x', rotation=30)

        # Add values on bars
        for i, (word, prob) in enumerate(zip(words, probs)):
            ax.text(i, prob + 0.02, f'{prob:.0%}', ha='center', fontsize=9)

    plt.tight_layout()
    save_svg(fig, 'temperature_comparison.svg')


# =============================================================================
# 4. Embedding Space Visualization
# =============================================================================
def create_embedding_space():
    """Show words as points in embedding space."""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')

    # Simulated 2D embeddings for word groups
    np.random.seed(42)

    # Animals
    animals = ['cat', 'dog', 'tiger', 'lion']
    animal_coords = np.array([[2, 3], [2.3, 3.2], [2.5, 2.8], [2.2, 2.5]])
    animal_coords += np.random.randn(*animal_coords.shape) * 0.2

    # Countries
    countries = ['France', 'Germany', 'Italy', 'Spain']
    country_coords = np.array([[-2, 2], [-1.8, 2.3], [-2.2, 1.8], [-1.5, 2.1]])
    country_coords += np.random.randn(*country_coords.shape) * 0.2

    # Actions
    actions = ['run', 'walk', 'jump', 'swim']
    action_coords = np.array([[0, -2], [0.3, -2.2], [-0.2, -2.5], [0.5, -1.8]])
    action_coords += np.random.randn(*action_coords.shape) * 0.2

    # Plot each group
    for words, coords, color, label in [
        (animals, animal_coords, COLORS['success'], 'Animals'),
        (countries, country_coords, COLORS['blue'], 'Countries'),
        (actions, action_coords, COLORS['accent'], 'Actions')
    ]:
        ax.scatter(coords[:, 0], coords[:, 1], s=200, c=color,
                  edgecolors='white', linewidth=2, alpha=0.8, label=label)
        for word, (x, y) in zip(words, coords):
            ax.annotate(word, (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=11, fontweight='bold')

    # Draw circles around groups
    from matplotlib.patches import Circle
    ax.add_patch(Circle((2.2, 2.9), 0.8, fill=False, edgecolor=COLORS['success'],
                        linewidth=2, linestyle='--', alpha=0.5))
    ax.add_patch(Circle((-1.9, 2.0), 0.7, fill=False, edgecolor=COLORS['blue'],
                        linewidth=2, linestyle='--', alpha=0.5))
    ax.add_patch(Circle((0.1, -2.1), 0.7, fill=False, edgecolor=COLORS['accent'],
                        linewidth=2, linestyle='--', alpha=0.5))

    ax.set_xlabel('Embedding Dimension 1', fontsize=12)
    ax.set_ylabel('Embedding Dimension 2', fontsize=12)
    ax.set_title('Word Embeddings: Similar Words Are Close Together',
                fontsize=14, fontweight='bold', color=COLORS['primary'])
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    save_svg(fig, 'embedding_space.svg')


# =============================================================================
# 5. Attention Mechanism Intuition
# =============================================================================
def create_attention_intuition():
    """Visualize attention - which words to focus on."""
    fig, ax = plt.subplots(figsize=(14, 5), facecolor='white')
    ax.set_facecolor('white')

    # Sentence
    words = ['The', 'cat', 'sat', 'because', 'it', 'was', 'tired']

    # Attention weights when processing "it"
    # "it" should attend most to "cat"
    attention_weights = [0.05, 0.55, 0.10, 0.08, 0.0, 0.12, 0.10]

    # Draw word boxes
    y_top = 0.7
    x_positions = []
    x = 0.1

    for i, word in enumerate(words):
        width = len(word) * 0.025 + 0.06
        color = COLORS['accent'] if word == 'it' else COLORS['blue']
        alpha = 1.0 if word == 'it' else 0.6

        rect = FancyBboxPatch((x, y_top), width, 0.15,
                              boxstyle="round,pad=0.02",
                              facecolor=color, alpha=alpha, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + width/2, y_top + 0.075, word,
               ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        x_positions.append(x + width/2)
        x += width + 0.03

    # Draw attention arrows from "it" (index 4) to other words
    it_idx = 4
    for i, (weight, x_pos) in enumerate(zip(attention_weights, x_positions)):
        if i != it_idx and weight > 0.05:
            # Line width based on attention weight
            lw = weight * 8
            alpha = min(weight * 2, 1.0)

            # Draw curved arrow
            ax.annotate('', xy=(x_pos, y_top),
                       xytext=(x_positions[it_idx], y_top - 0.18),
                       arrowprops=dict(arrowstyle='->', color=COLORS['success'],
                                      lw=lw, alpha=alpha,
                                      connectionstyle='arc3,rad=-0.3'))

            # Show weight
            mid_x = (x_pos + x_positions[it_idx]) / 2
            ax.text(mid_x, y_top - 0.35, f'{weight:.0%}', ha='center',
                   fontsize=9, color=COLORS['success'], fontweight='bold')

    # Legend / explanation
    ax.text(0.1, 0.2, 'When processing "it", the model attends to "cat" (55%)',
           fontsize=12, color=COLORS['text'])
    ax.text(0.1, 0.1, 'Attention helps the model understand: "it" refers to "the cat"',
           fontsize=11, color=COLORS['text_light'], style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Self-Attention: Which Words Should I Focus On?',
                fontsize=14, fontweight='bold', color=COLORS['primary'], y=0.95)

    save_svg(fig, 'attention_intuition.svg')


# =============================================================================
# 6. Transformer Block (Simplified)
# =============================================================================
def create_transformer_block():
    """Simplified transformer block diagram."""
    fig, ax = plt.subplots(figsize=(8, 10), facecolor='white')
    ax.set_facecolor('white')

    # Block positions
    y_positions = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
    labels = ['Output', 'Feed Forward', 'Add & Norm', 'Self-Attention', 'Add & Norm', 'Input Embeddings']
    colors = [COLORS['success'], COLORS['purple'], COLORS['warning'],
              COLORS['blue'], COLORS['warning'], COLORS['primary']]

    for y, label, color in zip(y_positions, labels, colors):
        rect = FancyBboxPatch((0.25, y), 0.5, 0.1,
                              boxstyle="round,pad=0.02,rounding_size=0.02",
                              facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, y + 0.05, label, ha='center', va='center',
               fontsize=11, fontweight='bold', color='white')

        # Arrows between blocks
        if y < 0.85:
            ax.annotate('', xy=(0.5, y + 0.1), xytext=(0.5, y + 0.15),
                       arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['text_light']))

    # Skip connections (residual)
    ax.annotate('', xy=(0.2, 0.55), xytext=(0.2, 0.4),
               arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['accent'],
                              connectionstyle='arc3,rad=0.3'))
    ax.annotate('', xy=(0.2, 0.7), xytext=(0.2, 0.55),
               arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['accent'],
                              connectionstyle='arc3,rad=0.3'))
    ax.text(0.12, 0.47, 'skip', fontsize=8, color=COLORS['accent'], rotation=90)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Transformer Block (Simplified)',
                fontsize=14, fontweight='bold', color=COLORS['primary'])

    save_svg(fig, 'transformer_block.svg')


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Generating diagrams for L07: Language Models...")
    print()

    print("Creating next token prediction visualization...")
    create_next_token_prediction()

    print("Creating tokenization visualization...")
    create_tokenization_visual()

    print("Creating temperature comparison...")
    create_temperature_comparison()

    print("Creating embedding space visualization...")
    create_embedding_space()

    print("Creating attention intuition...")
    create_attention_intuition()

    print("Creating transformer block diagram...")
    create_transformer_block()

    print()
    print("Done! All diagrams generated in diagrams/svg/")
    print()
    print("Generated diagrams:")
    print("  - next_token_prediction.svg")
    print("  - tokenization.svg")
    print("  - temperature_comparison.svg")
    print("  - embedding_space.svg")
    print("  - attention_intuition.svg")
    print("  - transformer_block.svg")
