import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# Create realistic data: temperature vs ice cream sales
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Generate data with nonlinear relationship
X = np.linspace(15, 40, 25).reshape(-1, 1)  # Temperature (°C)
y_true = 0.5 * (X.flatten() - 20)**2 + 10   # Quadratic relationship
y = y_true + np.random.randn(25) * 8        # Add noise

# Plot 1: Linear fit (underfitting)
ax1 = axes[0]
ax1.scatter(X, y, c='blue', s=60, alpha=0.7, label='Data')

model_lin = LinearRegression()
model_lin.fit(X, y)
X_plot = np.linspace(12, 43, 100).reshape(-1, 1)
y_lin = model_lin.predict(X_plot)
ax1.plot(X_plot, y_lin, 'r-', linewidth=2, label='Degree 1 (Linear)')

ax1.set_xlabel('Temperature (°C)', fontsize=11)
ax1.set_ylabel('Ice Cream Sales', fontsize=11)
ax1.set_title('Degree 1: Underfitting', fontsize=12, fontweight='bold', color='red')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(12, 43)

# Plot 2: Quadratic fit (just right)
ax2 = axes[1]
ax2.scatter(X, y, c='blue', s=60, alpha=0.7, label='Data')

poly2 = PolynomialFeatures(degree=2)
X_poly2 = poly2.fit_transform(X)
model_quad = LinearRegression()
model_quad.fit(X_poly2, y)
y_quad = model_quad.predict(poly2.transform(X_plot))
ax2.plot(X_plot, y_quad, 'g-', linewidth=2, label='Degree 2 (Quadratic)')

ax2.set_xlabel('Temperature (°C)', fontsize=11)
ax2.set_ylabel('Ice Cream Sales', fontsize=11)
ax2.set_title('Degree 2: Just Right ✓', fontsize=12, fontweight='bold', color='green')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(12, 43)

# Plot 3: High degree fit (overfitting)
ax3 = axes[2]
ax3.scatter(X, y, c='blue', s=60, alpha=0.7, label='Data')

poly10 = PolynomialFeatures(degree=10)
X_poly10 = poly10.fit_transform(X)
model_high = LinearRegression()
model_high.fit(X_poly10, y)
y_high = model_high.predict(poly10.transform(X_plot))
ax3.plot(X_plot, y_high, 'orange', linewidth=2, label='Degree 10')

ax3.set_xlabel('Temperature (°C)', fontsize=11)
ax3.set_ylabel('Ice Cream Sales', fontsize=11)
ax3.set_title('Degree 10: Overfitting', fontsize=12, fontweight='bold', color='orange')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(12, 43)
ax3.set_ylim(-20, 250)

plt.tight_layout()
plt.savefig('diagrams/polynomial_regression.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved polynomial_regression.png")


# Second figure: Feature transformation visualization
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

# Left: Original 1D feature
ax_left = axes2[0]
ax_left.scatter(X, y, c='blue', s=60, alpha=0.7)
ax_left.set_xlabel(r'$x$ (Temperature)', fontsize=12)
ax_left.set_ylabel('Sales', fontsize=12)
ax_left.set_title('Original Feature Space', fontsize=12, fontweight='bold')
ax_left.grid(True, alpha=0.3)

# Right: Transformed features (x vs x²) with color = y
ax_right = axes2[1]
X_sq = X.flatten()**2
scatter = ax_right.scatter(X.flatten(), X_sq, c=y, cmap='coolwarm', s=60, alpha=0.7)
ax_right.set_xlabel(r'$x$', fontsize=12)
ax_right.set_ylabel(r'$x^2$', fontsize=12)
ax_right.set_title('Expanded Feature Space: [x, x²]', fontsize=12, fontweight='bold')
ax_right.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax_right, label='Sales')

plt.tight_layout()
plt.savefig('diagrams/feature_expansion.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved feature_expansion.png")
