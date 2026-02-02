import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Simple 1D loss function: L(theta) = theta^2
theta_range = np.linspace(-3, 3, 100)
loss = theta_range**2

# Simulate gradient descent with different learning rates
def gradient_descent_1d(start, lr, steps):
    path = [start]
    theta = start
    for _ in range(steps):
        grad = 2 * theta  # derivative of theta^2
        theta = theta - lr * grad
        path.append(theta)
        if abs(theta) > 10:  # Divergence check
            break
    return path

# Case 1: Too small (lr = 0.05)
ax1 = axes[0]
ax1.plot(theta_range, loss, 'b-', linewidth=2)
path1 = gradient_descent_1d(2.5, 0.05, 25)
loss_path1 = [t**2 for t in path1]
ax1.plot(path1, loss_path1, 'ro-', markersize=6, linewidth=1.5, alpha=0.7)
ax1.set_xlabel(r'$\theta$', fontsize=12)
ax1.set_ylabel(r'Loss $\mathcal{L}(\theta)$', fontsize=12)
ax1.set_title(r'$\eta$ = 0.05 (Too Small)', fontsize=14, fontweight='bold', color='orange')
ax1.set_xlim(-3, 3)
ax1.set_ylim(0, 9)
ax1.axhline(y=0, color='green', linestyle='--', alpha=0.5)
ax1.text(0.5, 7, 'Slow\nconvergence', fontsize=11, ha='center', color='orange')
ax1.grid(True, alpha=0.3)

# Case 2: Just right (lr = 0.3)
ax2 = axes[1]
ax2.plot(theta_range, loss, 'b-', linewidth=2)
path2 = gradient_descent_1d(2.5, 0.3, 10)
loss_path2 = [t**2 for t in path2]
ax2.plot(path2, loss_path2, 'go-', markersize=8, linewidth=2)
ax2.set_xlabel(r'$\theta$', fontsize=12)
ax2.set_ylabel(r'Loss $\mathcal{L}(\theta)$', fontsize=12)
ax2.set_title(r'$\eta$ = 0.3 (Just Right)', fontsize=14, fontweight='bold', color='green')
ax2.set_xlim(-3, 3)
ax2.set_ylim(0, 9)
ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5)
ax2.text(0.5, 7, 'Fast\nconvergence!', fontsize=11, ha='center', color='green')
ax2.grid(True, alpha=0.3)

# Case 3: Too large (lr = 1.1)
ax3 = axes[2]
ax3.plot(theta_range, loss, 'b-', linewidth=2)
path3 = gradient_descent_1d(0.5, 1.1, 8)
loss_path3 = [min(t**2, 10) for t in path3]  # Cap for visualization
ax3.plot(path3[:len(loss_path3)], loss_path3, 'ro-', markersize=8, linewidth=2)
ax3.set_xlabel(r'$\theta$', fontsize=12)
ax3.set_ylabel(r'Loss $\mathcal{L}(\theta)$', fontsize=12)
ax3.set_title(r'$\eta$ = 1.1 (Too Large)', fontsize=14, fontweight='bold', color='red')
ax3.set_xlim(-3, 3)
ax3.set_ylim(0, 9)
ax3.axhline(y=0, color='green', linestyle='--', alpha=0.5)
ax3.text(0, 7, 'Diverges!\n(overshoots)', fontsize=11, ha='center', color='red')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/learning_rate_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved learning_rate_comparison.png")
