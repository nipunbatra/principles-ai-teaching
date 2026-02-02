import matplotlib.pyplot as plt
import numpy as np

# Set up the figure
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# Actual noisy data points from the slide
sizes = np.array([1000, 1500, 2000, 2500])
actual_prices = np.array([42, 58, 83, 97])

# The "perfect" line: price = 0.04 * size
line_x = np.linspace(800, 2700, 100)
line_y = 0.04 * line_x

# Plot the "ideal" line
plt.plot(line_x, line_y, 'b-', linewidth=2, label='Ideal line (y = 0.04x)', alpha=0.7)

# Plot actual data points
plt.scatter(sizes, actual_prices, s=150, c='red', zorder=5, label='Actual data (noisy)')

# Draw vertical lines showing the "error" between points and line
for size, actual in zip(sizes, actual_prices):
    ideal = 0.04 * size
    plt.vlines(size, min(actual, ideal), max(actual, ideal),
               colors='gray', linestyles='dashed', alpha=0.7)
    # Annotate the difference
    mid_y = (actual + ideal) / 2
    diff = actual - ideal
    plt.annotate(f'{diff:+.0f}', (size + 30, mid_y), fontsize=11, color='gray')

# Annotate data points
for size, actual in zip(sizes, actual_prices):
    plt.annotate(f'({size}, {actual})', (size - 80, actual + 3), fontsize=10)

# Labels and title
plt.xlabel('House Size (sqft)', fontsize=14)
plt.ylabel('Price (Rs. lakhs)', fontsize=14)
plt.title('Real Data Has Noise!', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='lower right')

# Set axis limits
plt.xlim(800, 2700)
plt.ylim(30, 110)

plt.tight_layout()
plt.savefig('diagrams/noisy_data.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('diagrams/noisy_data.svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved noisy_data.png and noisy_data.svg")
