#!/usr/bin/env python3
"""
Diagram generator for Lecture 08: Generative AI
Creates professional diagrams for SFT, RLHF, and Diffusion processes.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os

# Create output directory
os.makedirs('diagrams/svg', exist_ok=True)

# Consistent color scheme matching iitgn-modern theme
COLORS = {
    'primary': '#1e3a5f',
    'primary_light': '#2e5a8f',
    'accent': '#e85a4f',
    'success': '#2a9d8f',
    'warning': '#e9c46a',
    'blue': '#3b82f6',
    'purple': '#8b5cf6',
    'text': '#2d3748',
    'text_light': '#4a5568',
    'bg_light': '#f7fafc',
    'white': '#ffffff',
    'gray': '#94a3b8',
    'light_gray': '#e2e8f0',
    'green': '#10b981',
    'orange': '#f59e0b',
}


def setup_figure(figsize=(12, 6), bg_color='white'):
    """Create a clean figure with consistent styling."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.axis('off')
    return fig, ax


def save_svg(fig, filename):
    """Save figure as SVG."""
    filepath = f'diagrams/svg/{filename}'
    fig.savefig(filepath, format='svg', bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
    plt.close(fig)
    print(f"  ✓ {filename}")


# =============================================================================
# 1. SFT (Supervised Fine-Tuning) Pipeline
# =============================================================================
def create_sft_pipeline():
    """Create SFT training pipeline diagram."""
    fig, ax = setup_figure(figsize=(14, 8))

    # Title
    ax.text(7, 7.8, 'Supervised Fine-Tuning (SFT) Pipeline',
           fontsize=18, ha='center', fontweight='bold', color=COLORS['primary'])

    # Base Model Box
    ax.add_patch(FancyBboxPatch((0.5, 5.5), 4, 2, boxstyle="round,pad=0.15",
                                 facecolor=COLORS['blue'], edgecolor=COLORS['primary'],
                                 linewidth=2.5, alpha=0.3))
    ax.text(2.5, 6.8, 'BASE MODEL', fontsize=13, ha='center', fontweight='bold',
           color=COLORS['primary'])
    ax.text(2.5, 6.3, '(Pre-trained on', fontsize=10, ha='center', color=COLORS['text'])
    ax.text(2.5, 6.0, 'internet text)', fontsize=10, ha='center', color=COLORS['text'])
    ax.text(2.5, 5.7, '"Text completer"', fontsize=9, ha='center', style='italic',
           color=COLORS['text_light'])

    # SFT Data Box
    ax.add_patch(FancyBboxPatch((6, 5.5), 4, 2, boxstyle="round,pad=0.15",
                                 facecolor=COLORS['success'], edgecolor=COLORS['primary'],
                                 linewidth=2.5, alpha=0.3))
    ax.text(8, 6.8, 'SFT DATA', fontsize=13, ha='center', fontweight='bold',
           color=COLORS['primary'])
    ax.text(8, 6.3, '(Instruction, Response)', fontsize=10, ha='center', color=COLORS['text'])
    ax.text(8, 6.0, 'pairs', fontsize=10, ha='center', color=COLORS['text'])
    ax.text(8, 5.7, '10K-100K examples', fontsize=9, ha='center', style='italic',
           color=COLORS['text_light'])

    # Fine-tuning Process Box
    ax.add_patch(FancyBboxPatch((2.5, 2.5), 8, 2, boxstyle="round,pad=0.15",
                                 facecolor=COLORS['warning'], edgecolor=COLORS['primary'],
                                 linewidth=3, alpha=0.4))
    ax.text(6.5, 4.0, 'SUPERVISED FINE-TUNING', fontsize=14, ha='center',
           fontweight='bold', color=COLORS['primary'])
    ax.text(6.5, 3.5, 'Continue training on instruction-response pairs', fontsize=11,
           ha='center', color=COLORS['text'])
    ax.text(6.5, 3.0, 'Same architecture, new objective', fontsize=10,
           ha='center', style='italic', color=COLORS['text_light'])

    # Result Box
    ax.add_patch(FancyBboxPatch((2.5, 0.3), 8, 1.5, boxstyle="round,pad=0.15",
                                 facecolor=COLORS['success'], edgecolor=COLORS['primary'],
                                 linewidth=3, alpha=0.4))
    ax.text(6.5, 1.3, 'INSTRUCTION-TUNED MODEL', fontsize=14, ha='center',
           fontweight='bold', color=COLORS['primary'])
    ax.text(6.5, 0.7, 'Follows instructions, helpful assistant', fontsize=11,
           ha='center', color=COLORS['text'])

    # Arrows
    # From base model to fine-tuning
    ax.annotate('', xy=(4.5, 3.5), xytext=(2.5, 5.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=3))

    # From SFT data to fine-tuning
    ax.annotate('', xy=(8.5, 3.5), xytext=(8, 5.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=3))

    # From fine-tuning to result
    ax.annotate('', xy=(6.5, 1.8), xytext=(6.5, 2.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=3))

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)

    save_svg(fig, 'sft_pipeline.svg')


# =============================================================================
# 2. RLHF Pipeline
# =============================================================================
def create_rlhf_pipeline():
    """Create RLHF pipeline diagram."""
    fig, ax = setup_figure(figsize=(14, 9))

    # Title
    ax.text(7, 8.8, 'RLHF: Reinforcement Learning from Human Feedback',
           fontsize=18, ha='center', fontweight='bold', color=COLORS['primary'])

    # Instruction-tuned model (top)
    ax.add_patch(FancyBboxPatch((4, 7.5), 6, 1.2, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['success'], edgecolor=COLORS['primary'],
                                 linewidth=2.5, alpha=0.4))
    ax.text(7, 8.1, 'INSTRUCTION-TUNED MODEL', fontsize=12, ha='center',
           fontweight='bold', color=COLORS['primary'])

    # Step 1: Generate responses
    ax.add_patch(FancyBboxPatch((0.5, 5.5), 4, 1.8, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['blue'], edgecolor=COLORS['primary'],
                                 linewidth=2, alpha=0.3))
    ax.text(2.5, 6.7, '1. GENERATE', fontsize=11, ha='center', fontweight='bold',
           color=COLORS['primary'])
    ax.text(2.5, 6.3, 'Model produces', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(2.5, 6.0, 'multiple responses', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(2.5, 5.7, '"A, B, C..."', fontsize=9, ha='center', style='italic',
           color=COLORS['text_light'])

    # Step 2: Human ranking
    ax.add_patch(FancyBboxPatch((5.5, 5.5), 3.5, 1.8, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['purple'], edgecolor=COLORS['primary'],
                                 linewidth=2, alpha=0.3))
    ax.text(7.25, 6.7, '2. RANK', fontsize=11, ha='center', fontweight='bold',
           color=COLORS['primary'])
    ax.text(7.25, 6.3, 'Humans rank:', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(7.25, 6.0, 'A > B > C', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(7.25, 5.7, '(A is best)', fontsize=9, ha='center', style='italic',
           color=COLORS['text_light'])

    # Step 3: Reward model
    ax.add_patch(FancyBboxPatch((9.5, 5.5), 4, 1.8, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['warning'], edgecolor=COLORS['primary'],
                                 linewidth=2, alpha=0.4))
    ax.text(11.5, 6.7, '3. REWARD MODEL', fontsize=11, ha='center',
           fontweight='bold', color=COLORS['primary'])
    ax.text(11.5, 6.3, 'Train model to', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(11.5, 6.0, 'predict rankings', fontsize=9, ha='center', color=COLORS['text'])

    # Step 4: Optimize (bottom center)
    ax.add_patch(FancyBboxPatch((4, 3), 6, 1.8, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['accent'], edgecolor=COLORS['primary'],
                                 linewidth=2.5, alpha=0.4))
    ax.text(7, 4.3, '4. OPTIMIZE (PPO/DPO)', fontsize=12, ha='center',
           fontweight='bold', color=COLORS['white'])
    ax.text(7, 3.8, 'Maximize reward while', fontsize=10, ha='center',
           color=COLORS['white'])
    ax.text(7, 3.4, 'staying close to base model', fontsize=10, ha='center',
           color=COLORS['white'])

    # Final result
    ax.add_patch(FancyBboxPatch((4, 0.5), 6, 1.5, boxstyle="round,pad=0.15",
                                 facecolor=COLORS['green'], edgecolor=COLORS['primary'],
                                 linewidth=3, alpha=0.5))
    ax.text(7, 1.5, 'AI ASSISTANT', fontsize=14, ha='center', fontweight='bold',
           color=COLORS['white'])
    ax.text(7, 0.9, 'Helpful, harmless, honest', fontsize=11, ha='center',
           color=COLORS['white'])

    # Arrows
    # From instruction model to generate
    ax.annotate('', xy=(2.5, 7.3), xytext=(5.5, 8.1),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2.5,
                              connectionstyle='arc3,rad=-0.3'))

    # From instruction model to rank
    ax.annotate('', xy=(7.25, 7.3), xytext=(7.75, 8.1),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2.5,
                              connectionstyle='arc3,rad=0'))

    # Horizontal arrows between steps
    ax.annotate('', xy=(5.5, 6.4), xytext=(4.5, 6.4),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2.5))
    ax.annotate('', xy=(9.5, 6.4), xytext=(9, 6.4),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2.5))

    # From reward model to optimize
    ax.annotate('', xy=(9, 3.9), xytext=(11.5, 5.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2.5,
                              connectionstyle='arc3,rad=0.3'))

    # From optimize to result
    ax.annotate('', xy=(7, 2), xytext=(7, 3),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=3))

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.5)

    save_svg(fig, 'rlhf_pipeline.svg')


# =============================================================================
# 3. Diffusion Model Process
# =============================================================================
def create_diffusion_process():
    """Create diffusion model denoising process diagram."""
    fig, ax = setup_figure(figsize=(14, 7))

    # Title
    ax.text(7, 6.5, 'Diffusion Models: Learning to Denoise',
           fontsize=18, ha='center', fontweight='bold', color=COLORS['primary'])

    # Process steps
    steps = [
        ('NOISY IMAGE', 'Random noise\n(Gaussian)', COLORS['accent'], 0),
        ('STEP 1', 'Predict &\nremove noise', COLORS['orange'], 1),
        ('STEP 2', 'Predict &\nremove noise', COLORS['warning'], 2),
        ('STEP 3', 'Predict &\nremove noise', COLORS['blue'], 3),
        ('STEP N', 'Predict &\nremove noise', COLORS['blue'], 4),
        ('CLEAN IMAGE', 'Original!\n(Or new sample)', COLORS['success'], 5),
    ]

    box_width = 1.8
    box_height = 2.2

    for name, desc, color, idx in steps:
        x = 1 + idx * 2.2
        y = 1.5

        # Box
        ax.add_patch(FancyBboxPatch((x, y), box_width, box_height,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor=COLORS['primary'],
                                    linewidth=2, alpha=0.4))

        # Title
        ax.text(x + box_width/2, y + 1.7, name, fontsize=10, ha='center',
               fontweight='bold', color=COLORS['primary'])

        # Description
        ax.text(x + box_width/2, y + 1.0, desc, fontsize=8, ha='center',
               color=COLORS['text'])

        # Arrow to next
        if idx < len(steps) - 1:
            ax.annotate('', xy=(x + box_width + 0.2, y + box_height/2),
                       xytext=(x + box_width - 0.1, y + box_height/2),
                       arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))

    # Training goal annotation
    ax.add_patch(FancyBboxPatch((3, 4), 8, 1.2, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['bg_light'], edgecolor=COLORS['accent'],
                                 linewidth=2, linestyle='--'))
    ax.text(7, 4.9, 'Training Goal: Learn to predict the EXACT noise added at each step',
           fontsize=12, ha='center', color=COLORS['accent'], fontweight='bold')

    # Generation annotation
    ax.text(7, 0.5, 'Generation: Start from pure random noise → iteratively denoise → sample!',
           fontsize=11, ha='center', style='italic', color=COLORS['text_light'])

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)

    save_svg(fig, 'diffusion_process.svg')


# =============================================================================
# 4. Full LLM Training Pipeline (Bonus)
# =============================================================================
def create_full_llm_pipeline():
    """Create the complete LLM training pipeline from pre-training to assistant."""
    fig, ax = setup_figure(figsize=(16, 7))

    # Title
    ax.text(8, 6.5, 'The Complete LLM Journey',
           fontsize=18, ha='center', fontweight='bold', color=COLORS['primary'])

    stages = [
        ('1. PRE-TRAINING', 'Internet text\n→ Base model',
         'Months, $100M\nPredict next token', COLORS['blue']),
        ('2. SFT', 'Instruction data\n→ Instruction model',
         'Days, $1M\nFollow instructions', COLORS['success']),
        ('3. RLHF/DPO', 'Human feedback\n→ Aligned model',
         'Weeks, $10M\nHelpful & safe', COLORS['warning']),
        ('4. DEPLOYMENT', 'ChatGPT/Claude\n→ Users!',
         'Ongoing\nReal assistants', COLORS['purple']),
    ]

    box_width = 3.2
    box_height = 2.5

    for name, desc, details, color in stages:
        idx = stages.index((name, desc, details, color))
        x = 0.8 + idx * 4

        # Main box
        ax.add_patch(FancyBboxPatch((x, 2.5), box_width, box_height,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor=COLORS['primary'],
                                    linewidth=2.5, alpha=0.4))

        # Stage name
        ax.text(x + box_width/2, 4.6, name, fontsize=11, ha='center',
               fontweight='bold', color=COLORS['primary'])

        # Description
        ax.text(x + box_width/2, 3.8, desc, fontsize=9, ha='center',
               color=COLORS['text'])

        # Details
        ax.text(x + box_width/2, 3.1, details, fontsize=8, ha='center',
               style='italic', color=COLORS['text_light'])

        # Arrow to next
        if idx < len(stages) - 1:
            ax.annotate('', xy=(x + box_width + 0.2, 3.75),
                       xytext=(x + box_width - 0.1, 3.75),
                       arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2.5))

    # Data flow annotation
    ax.text(8, 1.2, '→ More data, compute, and human feedback at each stage',
           fontsize=11, ha='center', style='italic', color=COLORS['text_light'])

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)

    save_svg(fig, 'full_llm_pipeline.svg')


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Generating diagrams for L08: Generative AI...")
    print()

    print("Creating SFT pipeline diagram...")
    create_sft_pipeline()

    print("Creating RLHF pipeline diagram...")
    create_rlhf_pipeline()

    print("Creating Diffusion process diagram...")
    create_diffusion_process()

    print("Creating Full LLM pipeline diagram...")
    create_full_llm_pipeline()

    print()
    print("Done! All diagrams generated in diagrams/svg/")
    print()
    print("Generated diagrams:")
    print("  - sft_pipeline.svg")
    print("  - rlhf_pipeline.svg")
    print("  - diffusion_process.svg")
    print("  - full_llm_pipeline.svg")
