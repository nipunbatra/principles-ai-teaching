import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 5))

# Sigmoid function
z = np.linspace(-8, 8, 200)
sigmoid = 1 / (1 + np.exp(-z))

# Plot sigmoid
ax.plot(z, sigmoid, 'b-', linewidth=3, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$')

# Add horizontal lines at 0 and 1
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

# Add vertical line at z=0
ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

# Mark key points
key_points = [(-6, 1/(1+np.exp(6))), (0, 0.5), (6, 1/(1+np.exp(-6)))]
for z_val, sig_val in key_points:
    ax.plot(z_val, sig_val, 'ro', markersize=10)
    if z_val == 0:
        ax.annotate(f'(0, 0.5)', (z_val, sig_val), xytext=(1, 0.35), fontsize=12)
    elif z_val < 0:
        ax.annotate(f'z → -∞\nσ → 0', (z_val, sig_val), xytext=(-7, 0.15), fontsize=11)
    else:
        ax.annotate(f'z → +∞\nσ → 1', (z_val, sig_val), xytext=(4.5, 0.75), fontsize=11)

# Labels and styling
ax.set_xlabel(r'$z = \boldsymbol{\theta}^\top \mathbf{x}$', fontsize=14)
ax.set_ylabel(r'$\sigma(z)$ = Probability', fontsize=14)
ax.set_title('The Sigmoid Function: Squash Any Number to (0, 1)', fontsize=14, fontweight='bold')
ax.set_xlim(-8, 8)
ax.set_ylim(-0.1, 1.1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.legend(loc='lower right', fontsize=14)
ax.grid(True, alpha=0.3)

# Add region labels
ax.fill_between(z[z < -2], 0, sigmoid[z < -2], alpha=0.2, color='red', label='Predict 0')
ax.fill_between(z[z > 2], 0, sigmoid[z > 2], alpha=0.2, color='green', label='Predict 1')

plt.tight_layout()
plt.savefig('diagrams/sigmoid_function_matplotlib.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved sigmoid_function_matplotlib.png")
