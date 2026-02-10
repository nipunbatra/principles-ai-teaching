"""Generate plot showing noisy data with 3 candidate lines."""
import matplotlib.pyplot as plt
import numpy as np

# Data points (from the slide)
sizes = np.array([1000, 1500, 2000, 2500])
prices = np.array([42, 58, 83, 97])

# Three candidate lines
# Line A: y = 0.04x  (the "ideal" line from slide)
# Line B: y = 0.038x + 3  (slightly different slope + intercept)
# Line C: y = 0.035x + 10  (worse fit)
x_range = np.linspace(800, 2700, 100)

lines = [
    ("Line A: y = 0.04x", lambda x: 0.04 * x, "#2196F3"),
    ("Line B: y = 0.038x + 3", lambda x: 0.038 * x + 3, "#FF9800"),
    ("Line C: y = 0.035x + 10", lambda x: 0.035 * x + 10, "#E91E63"),
]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot data points
ax.scatter(sizes, prices, s=120, color="black", zorder=5, label="Actual data")

# Plot each line
for label, fn, color in lines:
    ax.plot(x_range, fn(x_range), color=color, linewidth=2.5, label=label)
    # Draw residuals for this line
    for xi, yi in zip(sizes, prices):
        pred = fn(xi)
        ax.plot([xi, xi], [yi, pred], color=color, linewidth=1, linestyle="--", alpha=0.5)

# Compute SSE for each and annotate with vertical offset to avoid overlap
offsets = [12, -2, -16]
for (label, fn, color), yoff in zip(lines, offsets):
    sse = sum((yi - fn(xi))**2 for xi, yi in zip(sizes, prices))
    ax.annotate(f"SSE = {sse:.0f}", xy=(2550, fn(2550)),
                fontsize=12, color=color, fontweight="bold",
                xytext=(15, yoff), textcoords="offset points")

ax.set_xlabel("Size (sq ft)", fontsize=14)
ax.set_ylabel("Price (lakhs)", fontsize=14)
ax.set_title("Which Line Fits Best?", fontsize=16, fontweight="bold")
ax.legend(fontsize=12, loc="upper left")
ax.set_xlim(800, 2900)
ax.set_ylim(25, 115)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig("diagrams/noisy_data_lines.png", dpi=150, bbox_inches="tight")
print("Saved diagrams/noisy_data_lines.png")
