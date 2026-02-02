import matplotlib.pyplot as plt
import numpy as np

# Set up the figure
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# Data points from the slide
sizes = np.array([1000, 1500, 2000, 2500])
actual_prices = np.array([42, 58, 83, 97])
predicted_prices = np.array([40, 60, 80, 100])  # From the line y = 0.04x

# The line of best fit
line_x = np.linspace(800, 2700, 100)
line_y = 0.04 * line_x

# Plot the line of best fit
plt.plot(line_x, line_y, 'b-', linewidth=2.5, label='Line of Best Fit: y = 0.04x')

# Plot actual data points
plt.scatter(sizes, actual_prices, s=150, c='red', zorder=5, label='Actual Data')

# Plot predicted points on the line
plt.scatter(sizes, predicted_prices, s=100, c='blue', marker='s', zorder=4, alpha=0.7, label='Predicted (on line)')

# Draw residual lines (vertical lines from actual to predicted)
for size, actual, predicted in zip(sizes, actual_prices, predicted_prices):
    residual = actual - predicted
    color = 'green' if residual > 0 else 'orange'
    plt.vlines(size, predicted, actual, colors=color, linewidth=3, alpha=0.8)

    # Label the residual
    mid_y = (actual + predicted) / 2
    plt.annotate(f'Residual\n= {residual:+d}',
                 (size + 40, mid_y),
                 fontsize=10,
                 color=color,
                 fontweight='bold')

# Labels and title
plt.xlabel('House Size (sqft)', fontsize=14)
plt.ylabel('Price (Rs. lakhs)', fontsize=14)
plt.title('Residuals: Difference Between Actual and Predicted', fontsize=16, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')

# Set axis limits
plt.xlim(800, 2700)
plt.ylim(30, 110)

# Add annotation explaining goal
plt.text(1400, 100, 'Goal: Minimize sum of squared residuals!',
         fontsize=12, style='italic',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('diagrams/residuals_explained.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved residuals_explained.png")
