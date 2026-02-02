import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left plot: Cross-entropy loss curves
ax1 = axes[0]
p = np.linspace(0.01, 0.99, 100)

# When y=1: loss = -log(p)
loss_y1 = -np.log(p)
# When y=0: loss = -log(1-p)
loss_y0 = -np.log(1 - p)

ax1.plot(p, loss_y1, 'b-', linewidth=2.5, label=r'$y=1$: $-\log(\hat{y})$')
ax1.plot(p, loss_y0, 'r-', linewidth=2.5, label=r'$y=0$: $-\log(1-\hat{y})$')

ax1.set_xlabel(r'Predicted probability $\hat{y}$', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Cross-Entropy Loss', fontsize=13, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 5)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Annotate key points
ax1.annotate('High loss!\n(wrong & confident)', xy=(0.1, 2.3), fontsize=10, color='blue')
ax1.annotate('High loss!\n(wrong & confident)', xy=(0.7, 2.3), fontsize=10, color='red')

# Right plot: 4 cases table as visualization
ax2 = axes[1]
ax2.axis('off')

# Create table data
cases = [
    ['Case', 'True (y)', 'Pred (ŷ)', 'Loss', 'Interpretation'],
    ['✓ Correct', '1 (Spam)', '0.95', '0.05', 'Low loss - good!'],
    ['✓ Correct', '0 (Not)', '0.05', '0.05', 'Low loss - good!'],
    ['✗ Wrong', '1 (Spam)', '0.10', '2.30', 'High loss - BAD!'],
    ['✗ Wrong', '0 (Not)', '0.90', '2.30', 'High loss - BAD!'],
]

# Colors for rows
colors = [['lightgray']*5, ['lightgreen']*5, ['lightgreen']*5, ['lightcoral']*5, ['lightcoral']*5]

table = ax2.table(cellText=cases, cellColours=colors, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

ax2.set_title('The 4 Cases: Confident & Wrong = High Loss!', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('diagrams/cross_entropy_explained.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved cross_entropy_explained.png")
