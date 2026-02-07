"""Generate a realistic 10x10 MNIST confusion matrix diagram."""
import numpy as np
import matplotlib.pyplot as plt

# Realistic MNIST confusion matrix (97% accuracy model)
np.random.seed(42)
cm = np.zeros((10, 10), dtype=int)
samples_per_class = 100

for i in range(10):
    correct = np.random.randint(92, 99)
    cm[i, i] = correct
    remaining = samples_per_class - correct
    # Distribute errors to other classes
    error_classes = np.random.choice([j for j in range(10) if j != i], size=remaining)
    for ec in error_classes:
        cm[i, ec] += 1

# Common confusions: 4↔9, 3↔5, 7↔1, 3↔8
cm[4, 9] += 3; cm[4, 4] -= 3
cm[9, 4] += 2; cm[9, 9] -= 2
cm[3, 5] += 2; cm[3, 3] -= 2
cm[5, 3] += 2; cm[5, 5] -= 2
cm[7, 1] += 2; cm[7, 7] -= 2
cm[3, 8] += 2; cm[3, 3] -= 2

fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax, shrink=0.8)

# Add text annotations
for i in range(10):
    for j in range(10):
        color = 'white' if cm[i, j] > 50 else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color=color, fontsize=11, fontweight='bold' if i == j else 'normal')

ax.set_xlabel('Predicted Digit', fontsize=14)
ax.set_ylabel('Actual Digit', fontsize=14)
ax.set_title('MNIST Digit Classification — Confusion Matrix', fontsize=15)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(range(10), fontsize=12)
ax.set_yticklabels(range(10), fontsize=12)

plt.tight_layout()
plt.savefig('diagrams/mnist_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Saved diagrams/mnist_confusion_matrix.png")
