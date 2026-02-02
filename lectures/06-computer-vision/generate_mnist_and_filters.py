"""Generate MNIST visualization and filter examples for lecture slides."""
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
import os

# Create diagrams directory if needed
os.makedirs('diagrams', exist_ok=True)

# ============================================================================
# 1. MNIST Real Examples
# ============================================================================
print("Generating MNIST visualization...")

# Load MNIST
mnist = datasets.MNIST('./data', train=True, download=True)

# Create figure showing multiple digits
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

# Show 10 different digits
for i in range(10):
    # Find an example of digit i
    for idx in range(len(mnist)):
        if mnist.targets[idx] == i:
            img = mnist.data[idx].numpy()
            break

    ax = axes[i // 5, i % 5]
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Label: {i}', fontsize=14, fontweight='bold')
    ax.axis('off')

plt.suptitle('MNIST Dataset: Handwritten Digits (28×28 pixels)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('diagrams/mnist_real_examples.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved mnist_real_examples.png")
plt.close()

# ============================================================================
# 2. MNIST as Numbers
# ============================================================================
print("Generating MNIST as numbers visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Get a digit 7
for idx in range(len(mnist)):
    if mnist.targets[idx] == 7:
        img = mnist.data[idx].numpy()
        break

# Left: the image
axes[0].imshow(img, cmap='gray')
axes[0].set_title('What We See: Digit "7"', fontsize=14, fontweight='bold')
axes[0].axis('off')

# Right: show the numbers (center 10x10 region for clarity)
center_region = img[9:19, 9:19]
axes[1].imshow(center_region, cmap='gray')

# Overlay the actual numbers
for i in range(10):
    for j in range(10):
        val = center_region[i, j]
        color = 'white' if val < 128 else 'black'
        axes[1].text(j, i, f'{val}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

axes[1].set_title('What Computer Sees: Numbers (0-255)\n(center 10×10 region)', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.suptitle('MNIST: Images Are Just Numbers!', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('diagrams/mnist_as_numbers.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved mnist_as_numbers.png")
plt.close()

# ============================================================================
# 3. Filter Examples: Vertical and Horizontal Edge Detection
# ============================================================================
print("Generating filter examples...")

# Create a simple test image with clear edges
test_img = np.zeros((8, 8))
test_img[:, 4:] = 200  # Right half is white = vertical edge in middle
test_img[4:, :] = np.maximum(test_img[4:, :], 150)  # Bottom half is gray = horizontal edge

# Define filters
vertical_filter = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

horizontal_filter = np.array([
    [ 1,  1,  1],
    [ 0,  0,  0],
    [-1, -1, -1]
])

def apply_conv(img, kernel):
    """Apply convolution manually."""
    h, w = img.shape
    kh, kw = kernel.shape
    output = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(img[i:i+kh, j:j+kw] * kernel)
    return output

# Apply filters
vertical_output = apply_conv(test_img, vertical_filter)
horizontal_output = apply_conv(test_img, horizontal_filter)

# Create visualization
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: Vertical edge detection
axes[0, 0].imshow(test_img, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Input Image', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Show vertical filter with values
axes[0, 1].imshow(vertical_filter, cmap='RdBu', vmin=-2, vmax=2)
for i in range(3):
    for j in range(3):
        axes[0, 1].text(j, i, f'{vertical_filter[i,j]:+d}', ha='center', va='center',
                       fontsize=16, fontweight='bold', color='black')
axes[0, 1].set_title('Vertical Edge Filter', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].text(0.5, 0.5, '✱', fontsize=40, ha='center', va='center', transform=axes[0, 2].transAxes)
axes[0, 2].text(0.5, 0.2, 'Convolve', fontsize=12, ha='center', transform=axes[0, 2].transAxes)
axes[0, 2].axis('off')

axes[0, 3].imshow(np.abs(vertical_output), cmap='hot')
axes[0, 3].set_title('Vertical Edges Detected!', fontsize=12, fontweight='bold', color='darkred')
axes[0, 3].axis('off')

# Row 2: Horizontal edge detection
axes[1, 0].imshow(test_img, cmap='gray', vmin=0, vmax=255)
axes[1, 0].set_title('Same Input Image', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

# Show horizontal filter with values
axes[1, 1].imshow(horizontal_filter, cmap='RdBu', vmin=-2, vmax=2)
for i in range(3):
    for j in range(3):
        axes[1, 1].text(j, i, f'{horizontal_filter[i,j]:+d}', ha='center', va='center',
                       fontsize=16, fontweight='bold', color='black')
axes[1, 1].set_title('Horizontal Edge Filter', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].text(0.5, 0.5, '✱', fontsize=40, ha='center', va='center', transform=axes[1, 2].transAxes)
axes[1, 2].text(0.5, 0.2, 'Convolve', fontsize=12, ha='center', transform=axes[1, 2].transAxes)
axes[1, 2].axis('off')

axes[1, 3].imshow(np.abs(horizontal_output), cmap='hot')
axes[1, 3].set_title('Horizontal Edges Detected!', fontsize=12, fontweight='bold', color='darkred')
axes[1, 3].axis('off')

plt.suptitle('Different Filters Detect Different Patterns', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('diagrams/filter_examples.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved filter_examples.png")
plt.close()

# ============================================================================
# 4. Real Image Edge Detection Demo
# ============================================================================
print("Generating real edge detection demo...")

# Create a more realistic test image (simple shapes)
real_img = np.zeros((32, 32))
# Add a rectangle
real_img[8:24, 8:24] = 200
# Add a smaller square inside
real_img[12:20, 12:20] = 100

# Sobel filters (better for real edges)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

edges_x = apply_conv(real_img, sobel_x)
edges_y = apply_conv(real_img, sobel_y)
edges_combined = np.sqrt(edges_x**2 + edges_y**2)

fig, axes = plt.subplots(1, 4, figsize=(14, 4))

axes[0].imshow(real_img, cmap='gray')
axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(np.abs(edges_x), cmap='hot')
axes[1].set_title('Vertical Edges', fontsize=12, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(np.abs(edges_y), cmap='hot')
axes[2].set_title('Horizontal Edges', fontsize=12, fontweight='bold')
axes[2].axis('off')

axes[3].imshow(edges_combined, cmap='hot')
axes[3].set_title('All Edges Combined', fontsize=12, fontweight='bold')
axes[3].axis('off')

plt.suptitle('Edge Detection on a Real Image', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('diagrams/edge_detection_demo.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved edge_detection_demo.png")
plt.close()

print("\nAll diagrams generated successfully!")
