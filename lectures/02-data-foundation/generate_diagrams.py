"""
Diagram generator for Data Foundation lecture.
Uses matplotlib for all diagrams.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
import numpy as np
import os

# Create output directory
os.makedirs('diagrams/svg', exist_ok=True)

# Color scheme matching iitgn-modern theme
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
# 1. Traditional Programming vs ML
# =============================================================================
def create_traditional_vs_ml():
    fig, ax = setup_figure(figsize=(14, 5))

    # Left side: Traditional Programming
    ax.add_patch(FancyBboxPatch((0.5, 3), 2.5, 1.2, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['light_gray'], edgecolor=COLORS['primary'], linewidth=2))
    ax.text(1.75, 3.6, 'RULES', fontsize=14, ha='center', fontweight='bold', color=COLORS['primary'])
    ax.text(1.75, 3.2, '(Human writes)', fontsize=10, ha='center', color=COLORS['text_light'])

    ax.add_patch(FancyBboxPatch((0.5, 1.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['blue'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.3))
    ax.text(1.75, 2.1, 'DATA', fontsize=14, ha='center', fontweight='bold', color=COLORS['primary'])

    ax.add_patch(FancyBboxPatch((0.5, 0), 2.5, 1.2, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['success'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.3))
    ax.text(1.75, 0.6, 'OUTPUT', fontsize=14, ha='center', fontweight='bold', color=COLORS['primary'])

    # Arrows
    ax.annotate('', xy=(1.75, 2.7), xytext=(1.75, 3),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    ax.annotate('', xy=(1.75, 1.2), xytext=(1.75, 1.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))

    ax.text(1.75, 4.5, 'Traditional Programming', fontsize=16, ha='center', fontweight='bold', color=COLORS['primary'])

    # Divider
    ax.axvline(x=5, ymin=0.1, ymax=0.9, color=COLORS['gray'], linestyle='--', linewidth=2)
    ax.text(5, 2.5, 'VS', fontsize=18, ha='center', fontweight='bold', color=COLORS['gray'],
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor=COLORS['gray']))

    # Right side: Machine Learning
    ax.add_patch(FancyBboxPatch((6.5, 3), 2.5, 1.2, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['blue'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.3))
    ax.text(7.75, 3.6, 'DATA', fontsize=14, ha='center', fontweight='bold', color=COLORS['primary'])

    ax.add_patch(FancyBboxPatch((6.5, 1.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['warning'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.5))
    ax.text(7.75, 2.1, 'LABELS', fontsize=14, ha='center', fontweight='bold', color=COLORS['primary'])
    ax.text(7.75, 1.7, '(Desired outputs)', fontsize=10, ha='center', color=COLORS['text_light'])

    ax.add_patch(FancyBboxPatch((6.5, 0), 2.5, 1.2, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['accent'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.5))
    ax.text(7.75, 0.6, 'MODEL', fontsize=14, ha='center', fontweight='bold', color=COLORS['primary'])
    ax.text(7.75, 0.2, '(Learned rules!)', fontsize=10, ha='center', color=COLORS['text_light'])

    # Arrows
    ax.annotate('', xy=(7.75, 2.7), xytext=(7.75, 3),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    ax.annotate('', xy=(7.75, 1.2), xytext=(7.75, 1.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))

    ax.text(7.75, 4.5, 'Machine Learning', fontsize=16, ha='center', fontweight='bold', color=COLORS['accent'])

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 5)

    save_svg(fig, 'traditional_vs_ml.svg')


# =============================================================================
# 2. Learning Paradigms
# =============================================================================
def create_learning_paradigms():
    fig, ax = setup_figure(figsize=(14, 5))

    # Supervised
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 3.5, 3.5, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['blue'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.2))
    ax.text(2.25, 3.5, 'SUPERVISED', fontsize=14, ha='center', fontweight='bold', color=COLORS['primary'])
    ax.text(2.25, 2.8, 'Data + Labels', fontsize=11, ha='center', color=COLORS['text'])
    ax.text(2.25, 2.3, '↓', fontsize=16, ha='center', color=COLORS['primary'])
    ax.text(2.25, 1.8, 'Predict labels', fontsize=11, ha='center', color=COLORS['text'])
    ax.text(2.25, 1.2, 'Ex: Spam detection', fontsize=10, ha='center', color=COLORS['text_light'], style='italic')

    # Unsupervised
    ax.add_patch(FancyBboxPatch((4.5, 0.5), 3.5, 3.5, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['success'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.2))
    ax.text(6.25, 3.5, 'UNSUPERVISED', fontsize=14, ha='center', fontweight='bold', color=COLORS['primary'])
    ax.text(6.25, 2.8, 'Data only', fontsize=11, ha='center', color=COLORS['text'])
    ax.text(6.25, 2.3, '↓', fontsize=16, ha='center', color=COLORS['primary'])
    ax.text(6.25, 1.8, 'Find structure', fontsize=11, ha='center', color=COLORS['text'])
    ax.text(6.25, 1.2, 'Ex: Customer segments', fontsize=10, ha='center', color=COLORS['text_light'], style='italic')

    # Reinforcement
    ax.add_patch(FancyBboxPatch((8.5, 0.5), 3.5, 3.5, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['warning'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.3))
    ax.text(10.25, 3.5, 'REINFORCEMENT', fontsize=14, ha='center', fontweight='bold', color=COLORS['primary'])
    ax.text(10.25, 2.8, 'Actions + Rewards', fontsize=11, ha='center', color=COLORS['text'])
    ax.text(10.25, 2.3, '↓', fontsize=16, ha='center', color=COLORS['primary'])
    ax.text(10.25, 1.8, 'Maximize reward', fontsize=11, ha='center', color=COLORS['text'])
    ax.text(10.25, 1.2, 'Ex: Game playing', fontsize=10, ha='center', color=COLORS['text_light'], style='italic')

    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.5)

    save_svg(fig, 'learning_paradigms.svg')


# =============================================================================
# 3. Classification vs Regression
# =============================================================================
def create_classification_vs_regression():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')

    # Classification
    ax1 = axes[0]
    ax1.set_facecolor('white')
    np.random.seed(42)

    # Two classes
    x1 = np.random.randn(20) - 1.5
    y1 = np.random.randn(20) + 1
    x2 = np.random.randn(20) + 1.5
    y2 = np.random.randn(20) - 1

    ax1.scatter(x1, y1, c=COLORS['blue'], s=100, label='Class A', alpha=0.7, edgecolors='white')
    ax1.scatter(x2, y2, c=COLORS['accent'], s=100, label='Class B', alpha=0.7, edgecolors='white')

    # Decision boundary
    ax1.axvline(x=0, color=COLORS['primary'], linestyle='--', linewidth=2, label='Decision boundary')

    ax1.set_title('CLASSIFICATION\n(Discrete categories)', fontsize=14, fontweight='bold', color=COLORS['primary'])
    ax1.set_xlabel('Feature 1', fontsize=11)
    ax1.set_ylabel('Feature 2', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Regression
    ax2 = axes[1]
    ax2.set_facecolor('white')

    x = np.linspace(0, 10, 30)
    y = 2 * x + 5 + np.random.randn(30) * 2

    ax2.scatter(x, y, c=COLORS['success'], s=100, alpha=0.7, edgecolors='white', label='Data points')

    # Regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax2.plot(x, p(x), color=COLORS['accent'], linewidth=3, label='Regression line')

    ax2.set_title('REGRESSION\n(Continuous values)', fontsize=14, fontweight='bold', color=COLORS['primary'])
    ax2.set_xlabel('Input (X)', fontsize=11)
    ax2.set_ylabel('Output (Y)', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_svg(fig, 'classification_vs_regression.svg')


# =============================================================================
# 4. Features and Labels
# =============================================================================
def create_features_and_labels():
    fig, ax = setup_figure(figsize=(12, 6))

    # Table structure
    cell_width = 1.8
    cell_height = 0.7
    start_x = 1
    start_y = 4

    # Headers
    headers = ['sqft', 'beds', 'baths', 'garage', '|', 'price']
    for i, header in enumerate(headers):
        x = start_x + i * cell_width
        if header == '|':
            ax.axvline(x=x + cell_width/2, ymin=0.1, ymax=0.85, color=COLORS['accent'], linewidth=3)
        else:
            color = COLORS['blue'] if i < 4 else COLORS['success']
            ax.add_patch(FancyBboxPatch((x, start_y), cell_width-0.1, cell_height,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor=COLORS['primary'], linewidth=1.5, alpha=0.3))
            ax.text(x + cell_width/2 - 0.05, start_y + cell_height/2, header,
                   fontsize=11, ha='center', va='center', fontweight='bold', color=COLORS['primary'])

    # Data rows
    data = [
        ['1500', '3', '2', 'Yes', '|', '$300K'],
        ['2000', '4', '3', 'Yes', '|', '$450K'],
        ['1200', '2', '1', 'No', '|', '$200K'],
    ]

    for row_idx, row in enumerate(data):
        y = start_y - (row_idx + 1) * cell_height - 0.2
        for col_idx, value in enumerate(row):
            x = start_x + col_idx * cell_width
            if value == '|':
                continue
            ax.add_patch(FancyBboxPatch((x, y), cell_width-0.1, cell_height-0.1,
                                        boxstyle="round,pad=0.02",
                                        facecolor='white', edgecolor=COLORS['gray'], linewidth=1))
            ax.text(x + cell_width/2 - 0.05, y + cell_height/2 - 0.05, value,
                   fontsize=10, ha='center', va='center', color=COLORS['text'])

    # Labels
    ax.text(start_x + 2*cell_width, start_y + 1.2, 'Features (X)', fontsize=14, ha='center',
           fontweight='bold', color=COLORS['blue'])
    ax.annotate('', xy=(start_x + 0.5*cell_width, start_y + 0.9), xytext=(start_x + 3.5*cell_width, start_y + 0.9),
                arrowprops=dict(arrowstyle='<->', color=COLORS['blue'], lw=2))

    ax.text(start_x + 5*cell_width, start_y + 1.2, 'Label (y)', fontsize=14, ha='center',
           fontweight='bold', color=COLORS['success'])

    ax.set_xlim(0, 13)
    ax.set_ylim(0.5, 6)

    save_svg(fig, 'features_and_labels.svg')


# =============================================================================
# 5. Feature Types
# =============================================================================
def create_feature_types():
    fig, ax = setup_figure(figsize=(14, 5))

    types = [
        ('Numerical', '25, 3.14, -5', COLORS['blue']),
        ('Categorical', 'Red, Blue, Green', COLORS['success']),
        ('Binary', 'Yes/No, 0/1', COLORS['warning']),
        ('Ordinal', 'S < M < L', COLORS['purple']),
        ('Text', '"Hello world"', COLORS['accent']),
    ]

    box_width = 2.2
    box_height = 2.5

    for i, (name, example, color) in enumerate(types):
        x = 0.5 + i * (box_width + 0.3)
        ax.add_patch(FancyBboxPatch((x, 0.5), box_width, box_height,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor=COLORS['primary'],
                                    linewidth=2, alpha=0.25))
        ax.text(x + box_width/2, 2.4, name, fontsize=12, ha='center',
               fontweight='bold', color=COLORS['primary'])
        ax.text(x + box_width/2, 1.5, example, fontsize=10, ha='center',
               color=COLORS['text'], style='italic')

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.5)

    save_svg(fig, 'feature_types.svg')


# =============================================================================
# 6. Data Types
# =============================================================================
def create_data_types():
    fig, ax = setup_figure(figsize=(14, 5))

    types = [
        ('Tabular', '┌───┬───┐\n│ A │ B │\n├───┼───┤\n│ 1 │ 2 │\n└───┴───┘', COLORS['blue']),
        ('Image', '█▓▒░░▒▓█\n▓▒░░░░▒▓\n▒░░░░░░▒\n░░░░░░░░', COLORS['success']),
        ('Text', '"The quick\nbrown fox\njumps..."', COLORS['warning']),
        ('Time Series', '╱╲╱╲\n  ╱  ╲\n╱    ╲', COLORS['accent']),
    ]

    box_width = 2.8
    box_height = 3

    for i, (name, example, color) in enumerate(types):
        x = 0.5 + i * (box_width + 0.3)
        ax.add_patch(FancyBboxPatch((x, 0.5), box_width, box_height,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor=COLORS['primary'],
                                    linewidth=2, alpha=0.2))
        ax.text(x + box_width/2, 3.1, name, fontsize=13, ha='center',
               fontweight='bold', color=COLORS['primary'])
        ax.text(x + box_width/2, 1.8, example, fontsize=9, ha='center',
               va='center', color=COLORS['text'], family='monospace')

    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4)

    save_svg(fig, 'data_types.svg')


# =============================================================================
# 7. Data Requirements
# =============================================================================
def create_data_requirements():
    fig, ax = setup_figure(figsize=(12, 5))

    # Model complexity vs data
    models = ['Linear\nRegression', 'Decision\nTree', 'Random\nForest', 'Neural\nNetwork', 'Deep\nLearning']
    data_needed = [100, 500, 2000, 10000, 100000]

    x = np.arange(len(models))
    bars = ax.bar(x, np.log10(data_needed), color=[COLORS['blue'], COLORS['success'],
                                                    COLORS['warning'], COLORS['accent'], COLORS['purple']],
                  alpha=0.7, edgecolor=COLORS['primary'], linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('Data Needed (log scale)', fontsize=12)
    ax.set_title('Data Requirements by Model Complexity', fontsize=14, fontweight='bold', color=COLORS['primary'])

    # Add value labels
    for bar, val in zip(bars, data_needed):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:,}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylim(0, 6)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axis('on')

    save_svg(fig, 'data_requirements.svg')


# =============================================================================
# 8. Data Quality Issues
# =============================================================================
def create_data_quality_issues():
    fig, ax = setup_figure(figsize=(14, 4))

    issues = [
        ('Missing\nValues', '? ? ?', COLORS['accent']),
        ('Outliers', '• • • ★', COLORS['warning']),
        ('Noise', '~∿~', COLORS['blue']),
        ('Imbalance', '●●●●●○', COLORS['success']),
        ('Duplicates', '▣ ▣', COLORS['purple']),
    ]

    box_width = 2.2
    box_height = 2.2

    for i, (name, symbol, color) in enumerate(issues):
        x = 0.5 + i * (box_width + 0.3)
        ax.add_patch(FancyBboxPatch((x, 0.5), box_width, box_height,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor=COLORS['primary'],
                                    linewidth=2, alpha=0.25))
        ax.text(x + box_width/2, 2.2, name, fontsize=11, ha='center',
               fontweight='bold', color=COLORS['primary'])
        ax.text(x + box_width/2, 1.3, symbol, fontsize=16, ha='center',
               color=COLORS['text'])

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.2)

    save_svg(fig, 'data_quality_issues.svg')


# =============================================================================
# 9. Data Lifecycle
# =============================================================================
def create_data_lifecycle():
    fig, ax = setup_figure(figsize=(14, 4))

    steps = ['Collect', 'Clean', 'Explore', 'Transform', 'Split', 'Use']
    colors = [COLORS['blue'], COLORS['accent'], COLORS['success'],
              COLORS['warning'], COLORS['purple'], COLORS['primary']]

    box_width = 1.8
    box_height = 1.5

    for i, (step, color) in enumerate(zip(steps, colors)):
        x = 0.5 + i * (box_width + 0.5)
        ax.add_patch(FancyBboxPatch((x, 1), box_width, box_height,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor=COLORS['primary'],
                                    linewidth=2, alpha=0.3))
        ax.text(x + box_width/2, 1.75, f'{i+1}. {step}', fontsize=11, ha='center',
               fontweight='bold', color=COLORS['primary'])

        # Arrow to next
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + box_width + 0.35, 1.75), xytext=(x + box_width + 0.15, 1.75),
                       arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))

    ax.set_xlim(0, 14)
    ax.set_ylim(0.5, 3)

    save_svg(fig, 'data_lifecycle.svg')


# =============================================================================
# 10. Train/Test Split
# =============================================================================
def create_train_test_split():
    fig, ax = setup_figure(figsize=(12, 4))

    # All data bar
    ax.add_patch(FancyBboxPatch((1, 2), 10, 1.2, boxstyle="round,pad=0.05",
                                 facecolor=COLORS['light_gray'], edgecolor=COLORS['primary'], linewidth=2))
    ax.text(6, 2.6, 'ALL YOUR DATA (100%)', fontsize=12, ha='center', fontweight='bold', color=COLORS['primary'])

    # Training portion
    ax.add_patch(FancyBboxPatch((1, 0.5), 8, 1.2, boxstyle="round,pad=0.05",
                                 facecolor=COLORS['blue'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.4))
    ax.text(5, 1.1, 'Training Set (80%)', fontsize=12, ha='center', fontweight='bold', color=COLORS['primary'])

    # Test portion
    ax.add_patch(FancyBboxPatch((9.1, 0.5), 1.9, 1.2, boxstyle="round,pad=0.05",
                                 facecolor=COLORS['accent'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.4))
    ax.text(10.05, 1.1, 'Test (20%)', fontsize=11, ha='center', fontweight='bold', color=COLORS['primary'])

    # Arrow
    ax.annotate('', xy=(6, 1.7), xytext=(6, 2),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.8)

    save_svg(fig, 'train_test_split.svg')


# =============================================================================
# 11. Overfitting Visual
# =============================================================================
def create_overfitting_visual():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor='white')

    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y = np.sin(x) + np.random.randn(20) * 0.3
    x_smooth = np.linspace(0, 10, 100)

    titles = ['Underfitting\n(Too simple)', 'Good Fit\n(Just right)', 'Overfitting\n(Too complex)']
    colors = [COLORS['warning'], COLORS['success'], COLORS['accent']]

    for idx, (ax, title, color) in enumerate(zip(axes, titles, colors)):
        ax.set_facecolor('white')
        ax.scatter(x, y, c=COLORS['blue'], s=80, alpha=0.7, edgecolors='white', zorder=5)

        if idx == 0:  # Underfitting
            ax.axhline(y=np.mean(y), color=color, linewidth=3)
        elif idx == 1:  # Good fit
            from scipy.interpolate import UnivariateSpline
            spl = UnivariateSpline(x, y, s=2)
            ax.plot(x_smooth, spl(x_smooth), color=color, linewidth=3)
        else:  # Overfitting
            from scipy.interpolate import interp1d
            f = interp1d(x, y, kind='cubic')
            x_interp = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_interp, f(x_interp), color=color, linewidth=3)

        ax.set_title(title, fontsize=12, fontweight='bold', color=COLORS['primary'])
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 10.5)

    plt.tight_layout()
    save_svg(fig, 'overfitting_visual.svg')


# =============================================================================
# 12. Three Way Split
# =============================================================================
def create_three_way_split():
    fig, ax = setup_figure(figsize=(12, 4))

    # All data bar
    ax.add_patch(FancyBboxPatch((1, 2), 10, 1.2, boxstyle="round,pad=0.05",
                                 facecolor=COLORS['light_gray'], edgecolor=COLORS['primary'], linewidth=2))
    ax.text(6, 2.6, 'ALL YOUR DATA (100%)', fontsize=12, ha='center', fontweight='bold', color=COLORS['primary'])

    # Training portion
    ax.add_patch(FancyBboxPatch((1, 0.5), 6, 1.2, boxstyle="round,pad=0.05",
                                 facecolor=COLORS['blue'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.4))
    ax.text(4, 1.1, 'Train (60%)', fontsize=11, ha='center', fontweight='bold', color=COLORS['primary'])

    # Validation portion
    ax.add_patch(FancyBboxPatch((7.1, 0.5), 2, 1.2, boxstyle="round,pad=0.05",
                                 facecolor=COLORS['warning'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.4))
    ax.text(8.1, 1.1, 'Val (20%)', fontsize=11, ha='center', fontweight='bold', color=COLORS['primary'])

    # Test portion
    ax.add_patch(FancyBboxPatch((9.2, 0.5), 1.8, 1.2, boxstyle="round,pad=0.05",
                                 facecolor=COLORS['accent'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.4))
    ax.text(10.1, 1.1, 'Test (20%)', fontsize=10, ha='center', fontweight='bold', color=COLORS['primary'])

    # Arrow
    ax.annotate('', xy=(6, 1.7), xytext=(6, 2),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.8)

    save_svg(fig, 'three_way_split.svg')


# =============================================================================
# 13. ML Recipe
# =============================================================================
def create_ml_recipe():
    fig, ax = setup_figure(figsize=(14, 4))

    steps = [
        ('1. GET\nDATA', COLORS['blue']),
        ('2. PREPARE\nDATA', COLORS['success']),
        ('3. CHOOSE\nMODEL', COLORS['warning']),
        ('4. TRAIN', COLORS['accent']),
        ('5. EVALUATE', COLORS['purple']),
        ('6. DEPLOY', COLORS['primary']),
    ]

    box_width = 1.7
    box_height = 1.8

    for i, (step, color) in enumerate(steps):
        x = 0.5 + i * (box_width + 0.4)
        ax.add_patch(FancyBboxPatch((x, 0.8), box_width, box_height,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor=COLORS['primary'],
                                    linewidth=2, alpha=0.35))
        ax.text(x + box_width/2, 1.7, step, fontsize=10, ha='center',
               fontweight='bold', color=COLORS['primary'])

        # Arrow to next
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + box_width + 0.3, 1.7), xytext=(x + box_width + 0.1, 1.7),
                       arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))

    ax.set_xlim(0, 13.5)
    ax.set_ylim(0.3, 3)

    save_svg(fig, 'ml_recipe.svg')


# =============================================================================
# 14. sklearn API
# =============================================================================
def create_sklearn_api():
    fig, ax = setup_figure(figsize=(12, 5))

    # Central box
    ax.add_patch(FancyBboxPatch((3.5, 1.5), 5, 2.5, boxstyle="round,pad=0.2",
                                 facecolor=COLORS['primary'], edgecolor=COLORS['primary'],
                                 linewidth=2, alpha=0.15))
    ax.text(6, 3.5, 'ANY sklearn Model', fontsize=14, ha='center', fontweight='bold', color=COLORS['primary'])

    # Methods
    methods = [
        ('fit(X, y)', 'Train', COLORS['blue']),
        ('predict(X)', 'Predict', COLORS['success']),
        ('score(X, y)', 'Evaluate', COLORS['accent']),
    ]

    for i, (method, desc, color) in enumerate(methods):
        y = 2.8 - i * 0.6
        ax.add_patch(FancyBboxPatch((4, y-0.2), 2, 0.5, boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor=COLORS['primary'],
                                     linewidth=1, alpha=0.4))
        ax.text(5, y, method, fontsize=10, ha='center', fontweight='bold',
               color=COLORS['primary'], family='monospace')
        ax.text(7, y, f'→ {desc}', fontsize=10, ha='left', color=COLORS['text'])

    ax.set_xlim(2, 10)
    ax.set_ylim(1, 4.5)

    save_svg(fig, 'sklearn_api.svg')


# =============================================================================
# 15. Other diagrams (simplified versions)
# =============================================================================
def create_exam_analogy():
    fig, ax = setup_figure(figsize=(12, 4))

    # Study from same questions
    ax.add_patch(FancyBboxPatch((0.5, 1), 5, 2, boxstyle="round,pad=0.1",
                 facecolor=COLORS['warning'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.3))
    ax.text(3, 2.5, 'Practice Questions', fontsize=12, ha='center', fontweight='bold', color=COLORS['primary'])
    ax.text(3, 1.8, 'Q1, Q2, Q3, Q4, Q5', fontsize=10, ha='center', color=COLORS['text'])
    ax.text(3, 1.3, '(Study material)', fontsize=9, ha='center', color=COLORS['text_light'], style='italic')

    # Exam has new questions
    ax.add_patch(FancyBboxPatch((6.5, 1), 5, 2, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['accent'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.3))
    ax.text(9, 2.5, 'Exam Questions', fontsize=12, ha='center', fontweight='bold', color=COLORS['primary'])
    ax.text(9, 1.8, 'Q6, Q7, Q8, Q9, Q10', fontsize=10, ha='center', color=COLORS['text'])
    ax.text(9, 1.3, '(Never seen before!)', fontsize=9, ha='center', color=COLORS['text_light'], style='italic')

    ax.annotate('', xy=(6.3, 2), xytext=(5.7, 2),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))

    ax.set_xlim(0, 12)
    ax.set_ylim(0.5, 3.5)

    save_svg(fig, 'exam_analogy.svg')


def create_ml_task_flowchart():
    fig, ax = setup_figure(figsize=(12, 6))

    # Start
    ax.add_patch(Circle((2, 5), 0.5, facecolor=COLORS['primary'], edgecolor=COLORS['primary']))
    ax.text(2, 5, 'Start', fontsize=10, ha='center', va='center', color='white', fontweight='bold')

    # Question 1
    ax.add_patch(FancyBboxPatch((3.5, 4.3), 4, 1.4, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['warning'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.4))
    ax.text(5.5, 5, 'Have labels?', fontsize=11, ha='center', fontweight='bold', color=COLORS['primary'])

    ax.annotate('', xy=(3.5, 5), xytext=(2.5, 5),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))

    # Yes branch - supervised
    ax.text(6, 4.2, 'Yes', fontsize=9, color=COLORS['success'])
    ax.add_patch(FancyBboxPatch((4, 2.3), 3.5, 1.4, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['blue'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.3))
    ax.text(5.75, 3, 'Predicting\ncategory or number?', fontsize=10, ha='center', color=COLORS['primary'])

    ax.annotate('', xy=(5.5, 4.3), xytext=(5.5, 3.7),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))

    # Category
    ax.add_patch(FancyBboxPatch((2.5, 0.5), 2.5, 1, boxstyle="round,pad=0.1",
                 facecolor=COLORS['success'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.4))
    ax.text(3.75, 1, 'Classification', fontsize=10, ha='center', fontweight='bold', color=COLORS['primary'])

    # Number
    ax.add_patch(FancyBboxPatch((5.5, 0.5), 2.5, 1, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['accent'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.4))
    ax.text(6.75, 1, 'Regression', fontsize=10, ha='center', fontweight='bold', color=COLORS['primary'])

    ax.text(4, 2.1, 'Category', fontsize=8, color=COLORS['text_light'])
    ax.text(6.5, 2.1, 'Number', fontsize=8, color=COLORS['text_light'])

    ax.annotate('', xy=(3.75, 1.5), xytext=(4.5, 2.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1.5))
    ax.annotate('', xy=(6.75, 1.5), xytext=(6.5, 2.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1.5))

    # No branch - unsupervised
    ax.text(8.2, 4.8, 'No', fontsize=9, color=COLORS['accent'])
    ax.add_patch(FancyBboxPatch((9, 4.3), 2.5, 1.4, boxstyle="round,pad=0.1",
                                 facecolor=COLORS['purple'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.3))
    ax.text(10.25, 5, 'Unsupervised', fontsize=10, ha='center', fontweight='bold', color=COLORS['primary'])

    ax.annotate('', xy=(9, 5), xytext=(7.5, 5),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    save_svg(fig, 'ml_task_flowchart.svg')


def create_vision_hierarchy():
    fig, ax = setup_figure(figsize=(14, 4))

    tasks = [
        ('Classification', '"What is it?"', 'Cat', COLORS['blue']),
        ('Detection', '"What & where?"', 'Boxes', COLORS['success']),
        ('Segmentation', '"Exact shape?"', 'Masks', COLORS['warning']),
        ('Pose', '"Keypoints?"', 'Skeleton', COLORS['accent']),
    ]

    box_width = 2.8
    box_height = 2.2

    for i, (name, question, output, color) in enumerate(tasks):
        x = 0.5 + i * (box_width + 0.4)
        ax.add_patch(FancyBboxPatch((x, 0.5), box_width, box_height,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor=COLORS['primary'],
                                    linewidth=2, alpha=0.3))
        ax.text(x + box_width/2, 2.3, name, fontsize=12, ha='center',
               fontweight='bold', color=COLORS['primary'])
        ax.text(x + box_width/2, 1.7, question, fontsize=10, ha='center',
               color=COLORS['text'], style='italic')
        ax.text(x + box_width/2, 1.1, f'→ {output}', fontsize=10, ha='center',
               color=COLORS['text_light'])

        # Arrow
        if i < len(tasks) - 1:
            ax.annotate('', xy=(x + box_width + 0.3, 1.6), xytext=(x + box_width + 0.1, 1.6),
                       arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))

    ax.text(7, 3.2, 'More Information →', fontsize=11, ha='center',
           color=COLORS['text_light'], style='italic')

    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 3.5)

    save_svg(fig, 'vision_hierarchy.svg')


def create_model_selection_guide():
    fig, ax = setup_figure(figsize=(12, 5))

    # Simple text-based guide
    ax.text(6, 4, 'Model Selection Guide', fontsize=16, ha='center', fontweight='bold', color=COLORS['primary'])

    guides = [
        ('Start Here →', 'Linear/Logistic Regression', COLORS['blue']),
        ('Need Explanation →', 'Decision Tree', COLORS['success']),
        ('General Purpose →', 'Random Forest', COLORS['warning']),
        ('Maximum Accuracy →', 'XGBoost / Neural Net', COLORS['accent']),
    ]

    for i, (condition, model, color) in enumerate(guides):
        y = 3 - i * 0.8
        ax.text(2, y, condition, fontsize=11, ha='left', color=COLORS['text'])
        ax.add_patch(FancyBboxPatch((6, y-0.25), 4.5, 0.5, boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor=COLORS['primary'], linewidth=1.5, alpha=0.3))
        ax.text(8.25, y, model, fontsize=11, ha='center', fontweight='bold', color=COLORS['primary'])

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.5)

    save_svg(fig, 'model_selection_guide.svg')


def create_skills_progression():
    fig, ax = setup_figure(figsize=(12, 4))

    skills = [
        ('Week 1-2', 'pandas\nsklearn', COLORS['blue']),
        ('Week 3-4', 'Algorithms\nEvaluation', COLORS['success']),
        ('Week 5-6', 'PyTorch\nCNNs', COLORS['warning']),
        ('Week 7-8', 'Transformers\nLLMs', COLORS['accent']),
    ]

    box_width = 2.5
    box_height = 2

    for i, (week, skill, color) in enumerate(skills):
        x = 0.5 + i * (box_width + 0.5)
        ax.add_patch(FancyBboxPatch((x, 0.5), box_width, box_height,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor=COLORS['primary'],
                                    linewidth=2, alpha=0.3))
        ax.text(x + box_width/2, 2.1, week, fontsize=11, ha='center',
               fontweight='bold', color=COLORS['primary'])
        ax.text(x + box_width/2, 1.3, skill, fontsize=10, ha='center', color=COLORS['text'])

        if i < len(skills) - 1:
            ax.annotate('', xy=(x + box_width + 0.35, 1.5), xytext=(x + box_width + 0.15, 1.5),
                       arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))

    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 3)

    save_svg(fig, 'skills_progression.svg')


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Generating diagrams for L02: Data Foundation...")

    create_traditional_vs_ml()
    create_learning_paradigms()
    create_classification_vs_regression()
    create_features_and_labels()
    create_feature_types()
    create_data_types()
    create_data_requirements()
    create_data_quality_issues()
    create_data_lifecycle()
    create_train_test_split()
    create_overfitting_visual()
    create_three_way_split()
    create_ml_recipe()
    create_sklearn_api()
    create_exam_analogy()
    create_ml_task_flowchart()
    create_vision_hierarchy()
    create_model_selection_guide()
    create_skills_progression()

    print("\nDone! All diagrams generated in diagrams/svg/")
