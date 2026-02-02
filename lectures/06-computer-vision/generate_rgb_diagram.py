"""Generate RGB channels 3D diagram for lecture slides."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# Create figure with two subplots
fig = plt.figure(figsize=(14, 6))

# Left: 3D exploded view of RGB channels
ax1 = fig.add_subplot(121, projection='3d')

# Create sample data for visualization
height, width = 8, 10

# Create colored channel representations
r_channel = np.ones((height, width, 4))
r_channel[:, :, :3] = [1, 0.3, 0.3]  # Red tint
r_channel[:, :, 3] = 0.7

g_channel = np.ones((height, width, 4))
g_channel[:, :, :3] = [0.3, 1, 0.3]  # Green tint
g_channel[:, :, 3] = 0.7

b_channel = np.ones((height, width, 4))
b_channel[:, :, :3] = [0.3, 0.3, 1]  # Blue tint
b_channel[:, :, 3] = 0.7

# Plot each channel as a surface at different z levels
x = np.arange(width + 1)
y = np.arange(height + 1)
X, Y = np.meshgrid(x, y)

# Z positions for each channel (exploded view)
z_positions = [0, 3, 6]
colors = ['#ff6b6b', '#51cf66', '#339af0']
labels = ['Red (R)', 'Green (G)', 'Blue (B)']

for z_pos, color, label in zip(z_positions, colors, labels):
    Z = np.ones_like(X) * z_pos
    ax1.plot_surface(X, Y, Z, alpha=0.7, color=color, edgecolor='white', linewidth=0.5)

# Add grid lines on each surface
for z_pos in z_positions:
    for i in range(height + 1):
        ax1.plot([0, width], [i, i], [z_pos, z_pos], 'white', linewidth=0.3, alpha=0.5)
    for j in range(width + 1):
        ax1.plot([j, j], [0, height], [z_pos, z_pos], 'white', linewidth=0.3, alpha=0.5)

# Labels
ax1.set_xlabel('Width', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('Height', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_zlabel('Channels', fontsize=12, fontweight='bold', labelpad=10)

# Set z-ticks for channels
ax1.set_zticks([0, 3, 6])
ax1.set_zticklabels(['R', 'G', 'B'])

ax1.set_title('Color Image = 3 Channels\n(Height × Width × 3)', fontsize=14, fontweight='bold', pad=20)

# Adjust view angle
ax1.view_init(elev=25, azim=-60)

# Right: Show how one pixel works
ax2 = fig.add_subplot(122)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_aspect('equal')
ax2.axis('off')

# Title
ax2.text(5, 9.5, 'One Pixel = 3 Numbers', fontsize=14, fontweight='bold', ha='center')

# Draw three boxes for R, G, B values
box_width = 2
box_height = 1.5
y_pos = 6
x_positions = [1.5, 4, 6.5]
colors_rgb = ['#ff6b6b', '#51cf66', '#339af0']
values = ['255', '128', '64']
labels_rgb = ['Red', 'Green', 'Blue']

for x, color, val, label in zip(x_positions, colors_rgb, values, labels_rgb):
    rect = mpatches.FancyBboxPatch((x, y_pos), box_width, box_height,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor='black', linewidth=2)
    ax2.add_patch(rect)
    ax2.text(x + box_width/2, y_pos + box_height/2, val,
             fontsize=16, fontweight='bold', ha='center', va='center', color='white')
    ax2.text(x + box_width/2, y_pos - 0.4, label,
             fontsize=11, ha='center', va='top', color=color, fontweight='bold')

# Arrow pointing to combined color
ax2.annotate('', xy=(5, 4.5), xytext=(5, 5.8),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Combined color box
combined_color = (255/255, 128/255, 64/255)  # RGB normalized
rect = mpatches.FancyBboxPatch((3.5, 2.5), 3, 1.8,
                                boxstyle="round,pad=0.05",
                                facecolor=combined_color, edgecolor='black', linewidth=2)
ax2.add_patch(rect)
ax2.text(5, 3.4, '(255, 128, 64)', fontsize=12, fontweight='bold',
         ha='center', va='center', color='white')
ax2.text(5, 1.8, 'Combined = Orange pixel', fontsize=11, ha='center', va='top')

# Shape notation
ax2.text(5, 0.5, 'Image shape: (Height, Width, 3)', fontsize=12,
         ha='center', style='italic',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='gray'))

plt.tight_layout()
plt.savefig('diagrams/rgb_channels_diagram.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('diagrams/rgb_channels_diagram.svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved rgb_channels_diagram.png and .svg")
plt.close()
