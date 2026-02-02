"""Generate FC vs CNN comparison diagram."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# LEFT: Fully Connected (messy)
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_aspect('equal')
ax1.axis('off')

# Draw input image grid (4x4)
for i in range(5):
    ax1.plot([1, 1+2], [2+i*1.2, 2+i*1.2], 'gray', linewidth=0.5)
    ax1.plot([1+i*0.5, 1+i*0.5], [2, 2+4.8], 'gray', linewidth=0.5)

# Draw "pixels"
for i in range(4):
    for j in range(4):
        gray = np.random.uniform(0.3, 0.9)
        rect = patches.Rectangle((1+j*0.5, 2+i*1.2), 0.5, 1.2,
                                  facecolor=str(gray), edgecolor='gray')
        ax1.add_patch(rect)

ax1.text(2, 1.3, '224×224×3\n= 150,528 inputs', ha='center', fontsize=10)

# Draw hidden layer neurons
for i in range(6):
    circle = plt.Circle((6, 3+i*0.8), 0.3, color='lightblue', ec='blue')
    ax1.add_patch(circle)
ax1.text(6, 2, '1000 neurons', ha='center', fontsize=10)

# Draw MANY connections (messy)
np.random.seed(42)
for _ in range(50):
    x1 = np.random.uniform(2.8, 3.2)
    y1 = np.random.uniform(2, 7)
    y2 = np.random.uniform(3, 7.5)
    ax1.plot([x1, 5.7], [y1, y2], 'red', alpha=0.3, linewidth=0.5)

ax1.text(5, 8.5, '150 MILLION\nweights!', ha='center', fontsize=14,
         fontweight='bold', color='red')
ax1.text(5, 0.5, '❌ Fully Connected', ha='center', fontsize=12, fontweight='bold')

# RIGHT: CNN (clean)
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_aspect('equal')
ax2.axis('off')

# Draw input image
for i in range(5):
    ax2.plot([1, 3], [2+i*1.2, 2+i*1.2], 'gray', linewidth=0.5)
for j in range(5):
    ax2.plot([1+j*0.5, 1+j*0.5], [2, 6.8], 'gray', linewidth=0.5)

for i in range(4):
    for j in range(4):
        gray = np.random.uniform(0.3, 0.9)
        rect = patches.Rectangle((1+j*0.5, 2+i*1.2), 0.5, 1.2,
                                  facecolor=str(gray), edgecolor='gray')
        ax2.add_patch(rect)

ax2.text(2, 1.3, 'Image', ha='center', fontsize=10)

# Draw 3x3 filter
filter_rect = patches.Rectangle((4.5, 4), 1.5, 1.8,
                                  facecolor='lightgreen', edgecolor='green', linewidth=2)
ax2.add_patch(filter_rect)
ax2.text(5.25, 4.9, '3×3\nfilter', ha='center', fontsize=10, fontweight='bold')

# Draw arrow
ax2.annotate('', xy=(4.3, 5), xytext=(3.2, 5),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))

# Draw feature map output
for i in range(3):
    for j in range(3):
        rect = patches.Rectangle((7+j*0.5, 3.5+i*0.6), 0.5, 0.6,
                                  facecolor='lightblue', edgecolor='blue')
        ax2.add_patch(rect)

ax2.text(7.75, 2.8, 'Feature map', ha='center', fontsize=10)

ax2.text(5, 8.5, 'Only 27\nweights!', ha='center', fontsize=14,
         fontweight='bold', color='green')
ax2.text(5, 7.5, '(3×3×3 filter)', ha='center', fontsize=10, color='green')
ax2.text(5, 0.5, '✓ Convolutional', ha='center', fontsize=12, fontweight='bold')

plt.suptitle('Why Not Use Fully Connected Networks for Images?',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('diagrams/fc_vs_cnn_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved fc_vs_cnn_comparison.png")
plt.close()
