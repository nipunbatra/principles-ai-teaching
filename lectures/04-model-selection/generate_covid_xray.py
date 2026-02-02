import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: What we thought the model learned
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)

# Draw simplified X-ray (rectangle with lung shapes)
ax1.add_patch(patches.Rectangle((1, 1), 8, 8, fill=True, facecolor='#1a1a2e', edgecolor='white', linewidth=2))
# Lungs
ax1.add_patch(patches.Ellipse((3.5, 5), 2.5, 5, fill=True, facecolor='#4a4a6a', edgecolor='#8a8aaa', linewidth=1))
ax1.add_patch(patches.Ellipse((6.5, 5), 2.5, 5, fill=True, facecolor='#4a4a6a', edgecolor='#8a8aaa', linewidth=1))

# Highlight "lung features"
ax1.annotate('', xy=(3.5, 6.5), xytext=(1, 9.5),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax1.text(0.5, 9.7, 'Lung\nfeatures?', fontsize=10, color='green', fontweight='bold')

ax1.set_title('What We Thought\nModel Learned', fontsize=14, fontweight='bold', color='green')
ax1.axis('off')

# Right: What the model actually learned
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

# Draw X-ray
ax2.add_patch(patches.Rectangle((1, 1), 8, 8, fill=True, facecolor='#1a1a2e', edgecolor='white', linewidth=2))
# Lungs (same)
ax2.add_patch(patches.Ellipse((3.5, 5), 2.5, 5, fill=True, facecolor='#4a4a6a', edgecolor='#8a8aaa', linewidth=1))
ax2.add_patch(patches.Ellipse((6.5, 5), 2.5, 5, fill=True, facecolor='#4a4a6a', edgecolor='#8a8aaa', linewidth=1))

# Hospital ID in corner (what model actually learned)
ax2.text(7.5, 8.2, 'HOSP-A', fontsize=9, color='white', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))

# Arrow pointing to hospital ID
ax2.annotate('', xy=(7.5, 8.2), xytext=(9.5, 9.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=3))
ax2.text(8.5, 9.7, 'Hospital ID!', fontsize=11, color='red', fontweight='bold')

# Add explanation
ax2.text(5, 0.3, 'COVID cases came from Hospital A\nNon-COVID from Hospital B',
         fontsize=9, ha='center', style='italic', color='gray')

ax2.set_title('What Model Actually\nLearned', fontsize=14, fontweight='bold', color='red')
ax2.axis('off')

plt.suptitle('COVID X-Ray Detection: A Cautionary Tale', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('diagrams/covid_xray_shortcut.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved covid_xray_shortcut.png")
