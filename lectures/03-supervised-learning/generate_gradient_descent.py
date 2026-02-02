import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create figure with two subplots
fig = plt.figure(figsize=(14, 5))

# Left plot: 3D surface
ax1 = fig.add_subplot(121, projection='3d')

# Create loss surface (bowl shape)
theta0 = np.linspace(-3, 3, 50)
theta1 = np.linspace(-3, 3, 50)
T0, T1 = np.meshgrid(theta0, theta1)
L = T0**2 + T1**2  # Simple quadratic loss

# Plot surface
ax1.plot_surface(T0, T1, L, cmap='viridis', alpha=0.7, edgecolor='none')

# Gradient descent path
path_t0 = [2.5, 1.8, 1.2, 0.7, 0.4, 0.2, 0.1, 0.0]
path_t1 = [2.0, 1.4, 0.9, 0.5, 0.3, 0.15, 0.05, 0.0]
path_L = [t0**2 + t1**2 for t0, t1 in zip(path_t0, path_t1)]

# Plot path
ax1.plot(path_t0, path_t1, path_L, 'ro-', markersize=8, linewidth=2, label='Gradient descent path')
ax1.scatter([2.5], [2.0], [2.5**2 + 2.0**2], c='red', s=150, marker='o', label='Start (random θ)')
ax1.scatter([0], [0], [0], c='green', s=150, marker='*', label='Minimum (θ*)')

ax1.set_xlabel(r'$\theta_0$', fontsize=14)
ax1.set_ylabel(r'$\theta_1$', fontsize=14)
ax1.set_zlabel(r'Loss $\mathcal{L}(\theta)$', fontsize=12)
ax1.set_title('Gradient Descent on Loss Surface', fontsize=14, fontweight='bold')
ax1.view_angle = 30
ax1.legend(loc='upper left', fontsize=9)

# Right plot: 2D contour with arrows
ax2 = fig.add_subplot(122)

# Contour plot
contour = ax2.contour(T0, T1, L, levels=15, cmap='viridis')
ax2.contourf(T0, T1, L, levels=15, cmap='viridis', alpha=0.3)

# Plot gradient descent path with arrows
for i in range(len(path_t0)-1):
    ax2.annotate('', xy=(path_t0[i+1], path_t1[i+1]), xytext=(path_t0[i], path_t1[i]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax2.plot(path_t0, path_t1, 'ro', markersize=8)
ax2.scatter([0], [0], c='green', s=200, marker='*', zorder=5)

# Labels
ax2.set_xlabel(r'$\theta_0$', fontsize=14)
ax2.set_ylabel(r'$\theta_1$', fontsize=14)
ax2.set_title('Contour View: Following Negative Gradient', fontsize=14, fontweight='bold')

# Add update rule box
textstr = r'$\boldsymbol{\theta}_{new} = \boldsymbol{\theta}_{old} - \eta \nabla_{\theta}\mathcal{L}$'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

# Add "Start" and "Minimum" labels
ax2.annotate('Start', (2.5, 2.0), fontsize=11, xytext=(2.7, 2.3))
ax2.annotate('Minimum', (0, 0), fontsize=11, xytext=(0.3, -0.5))

plt.tight_layout()
plt.savefig('diagrams/gradient_descent_theta.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved gradient_descent_theta.png")
