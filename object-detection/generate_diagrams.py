import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Create directory if it doesn't exist
os.makedirs('diagrams', exist_ok=True)

# Common Style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
COLORS = {
    'blue': '#2E86AB',
    'green': '#06A77D',
    'red': '#E63946',
    'yellow': '#F6AE2D',
    'gray': '#6c757d',
    'light_bg': '#f8f9fa'
}

def create_figure(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(COLORS['light_bg'])
    fig.patch.set_facecolor(COLORS['light_bg'])
    return fig, ax

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f'diagrams/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

# 1. Bounding Box Anatomy
def plot_bbox_anatomy():
    fig, ax = create_figure(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Image background
    image_rect = patches.Rectangle((0.5, 0.5), 9, 8, facecolor='#cccccc', edgecolor='black', alpha=0.3)
    ax.add_patch(image_rect)
    ax.text(5, 8.8, "Image (W, H)", ha='center', fontsize=12)

    # Object in image
    ax.text(2.5, 5, "Object", ha='center', fontsize=16, color=COLORS['gray'])

    # Bounding box
    bbox = patches.Rectangle((1.5, 2), 6, 5, edgecolor=COLORS['red'], facecolor='none', linewidth=3)
    ax.add_patch(bbox)

    # Coordinates
    ax.plot([1.5, 1.5], [1.8, 2], color=COLORS['red'], linestyle='--', linewidth=1)
    ax.plot([1.5, 1.5], [0.5, 0.8], color=COLORS['red'], linestyle='--', linewidth=1)
    ax.text(1.5, 1.5, 'x_min', ha='center', color=COLORS['red'])

    ax.plot([7.5, 7.5], [1.8, 2], color=COLORS['red'], linestyle='--', linewidth=1)
    ax.text(7.5, 1.5, 'x_max', ha='center', color=COLORS['red'])

    ax.plot([1.2, 1.5], [2, 2], color=COLORS['red'], linestyle='--', linewidth=1)
    ax.text(0.8, 2, 'y_min', va='center', color=COLORS['red'])

    ax.plot([1.2, 1.5], [7, 7], color=COLORS['red'], linestyle='--', linewidth=1)
    ax.text(0.8, 7, 'y_max', va='center', color=COLORS['red'])

    # Width and Height
    ax.annotate('', xy=(1.5, 8), xytext=(7.5, 8), arrowprops=dict(arrowstyle='|->|', color=COLORS['red'], linewidth=2))
    ax.text(4.5, 8.2, 'width', ha='center', color=COLORS['red'])

    ax.annotate('', xy=(8, 7), xytext=(8, 2), arrowprops=dict(arrowstyle='|->|', color=COLORS['red'], linewidth=2))
    ax.text(8.2, 4.5, 'height', va='center', rotation=90, color=COLORS['red'])

    ax.text(5, 0.8, "(x_min, y_min, x_max, y_max) or (x_center, y_center, width, height)", ha='center', fontsize=10)
    save_plot('bbox_anatomy.png')

# 2. IoU Examples
def plot_iou_examples():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor(COLORS['light_bg'])
    
    for i, ax in enumerate(axes):
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.axis('off')
        
        # Ground Truth (Blue)
        gt_box = patches.Rectangle((1, 1), 2, 2, edgecolor=COLORS['blue'], facecolor=COLORS['blue'], alpha=0.3, linewidth=2)
        ax.add_patch(gt_box)
        ax.text(2, 3.2, 'GT', color=COLORS['blue'], fontsize=12, fontweight='bold')

        if i == 0: # Low IoU
            pred_box = patches.Rectangle((2.5, 2.5), 1.5, 1.5, edgecolor=COLORS['red'], facecolor=COLORS['red'], alpha=0.3, linewidth=2)
            ax.add_patch(pred_box)
            ax.text(3.25, 2.2, 'Pred', color=COLORS['red'], fontsize=12, fontweight='bold')
            ax.set_title('Low IoU (0.1)', fontsize=14)
        elif i == 1: # Medium IoU
            pred_box = patches.Rectangle((1.5, 1.5), 2.5, 2.5, edgecolor=COLORS['red'], facecolor=COLORS['red'], alpha=0.3, linewidth=2)
            ax.add_patch(pred_box)
            ax.text(2.75, 2.2, 'Pred', color=COLORS['red'], fontsize=12, fontweight='bold')
            ax.set_title('Medium IoU (0.5)', fontsize=14)
        else: # High IoU
            pred_box = patches.Rectangle((1.1, 1.1), 1.9, 1.9, edgecolor=COLORS['red'], facecolor=COLORS['red'], alpha=0.3, linewidth=2)
            ax.add_patch(pred_box)
            ax.text(2.05, 2.2, 'Pred', color=COLORS['red'], fontsize=12, fontweight='bold')
            ax.set_title('High IoU (0.8)', fontsize=14)

    save_plot('iou_examples.png')

# 3. NMS Process Visualization (Still Matplotlib for visual steps)
def plot_nms_visual():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(COLORS['light_bg'])

    for i, ax in enumerate(axes):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Base car image (simplified)
        car = patches.Rectangle((2, 3), 5, 3, facecolor=COLORS['blue'], edgecolor='none', alpha=0.7)
        ax.add_patch(car)
        ax.text(4.5, 4.5, 'Car', color='white', fontsize=14, ha='center', va='center')

        if i == 0: # Initial detections
            ax.set_title('1. Initial Detections', fontsize=16)
            boxes = [
                ((2.1, 2.9), 5.2, 3.2, 0.95),  # Best
                ((2.5, 3.5), 4.5, 2.5, 0.90),  # Good overlap
                ((1.8, 2.5), 4.8, 2.8, 0.80),  # Medium overlap
                ((6, 6), 2, 2, 0.40)           # Low confidence, different object
            ]
            for (x,y), w, h, conf in boxes:
                color = COLORS['red'] if conf > 0.5 else COLORS['gray']
                rect = patches.Rectangle((x,y), w, h, edgecolor=color, facecolor='none', linewidth=2)
                ax.add_patch(rect)
                ax.text(x + w/2, y + h + 0.5, f'{conf:.2f}', color=color, fontsize=10, ha='center')

        elif i == 1: # After picking best, suppressing others
            ax.set_title('2. Pick Best & Suppress', fontsize=16)
            # Best box (retained)
            rect_best = patches.Rectangle((2.1, 2.9), 5.2, 3.2, edgecolor=COLORS['green'], facecolor='none', linewidth=3)
            ax.add_patch(rect_best)
            ax.text(2.1 + 5.2/2, 2.9 + 3.2 + 0.5, '0.95', color=COLORS['green'], fontsize=10, ha='center')

            # Suppressed boxes (faded)
            boxes_suppressed = [
                ((2.5, 3.5), 4.5, 2.5, 0.90),
                ((1.8, 2.5), 4.8, 2.8, 0.80)
            ]
            for (x,y), w, h, conf in boxes_suppressed:
                rect = patches.Rectangle((x,y), w, h, edgecolor=COLORS['red'], facecolor='none', linewidth=1, alpha=0.3, linestyle='--')
                ax.add_patch(rect)
                ax.text(x + w/2, y + h + 0.5, f'{conf:.2f}', color=COLORS['red'], fontsize=10, ha='center', alpha=0.3)

            # Low confidence box (already filtered or ignored)
            rect_low_conf = patches.Rectangle((6, 6), 2, 2, edgecolor=COLORS['gray'], facecolor='none', linewidth=1, alpha=0.3)
            ax.add_patch(rect_low_conf)
            ax.text(6 + 2/2, 6 + 2 + 0.5, '0.40', color=COLORS['gray'], fontsize=10, ha='center', alpha=0.3)

        else: # Final result
            ax.set_title('3. Final Detection', fontsize=16)
            rect_final = patches.Rectangle((2.1, 2.9), 5.2, 3.2, edgecolor=COLORS['green'], facecolor=COLORS['green'], alpha=0.3, linewidth=3)
            ax.add_patch(rect_final)
            ax.text(2.1 + 5.2/2, 2.9 + 3.2 + 0.5, '0.95 (Final)', color=COLORS['green'], fontsize=12, ha='center', fontweight='bold')

    save_plot('nms_visual.png')

# 4. YOLO Grid
def plot_yolo_grid():
    fig, ax = create_figure(figsize=(8, 8))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    ax.set_xticks(np.arange(0.5, 7.5, 1), minor=True)
    ax.set_yticks(np.arange(0.5, 7.5, 1), minor=True)
    ax.set_xticks(np.arange(0, 8, 1))
    ax.set_yticks(np.arange(0, 8, 1))
    ax.grid(which='minor', color=COLORS['gray'], linestyle='-', linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(which='both', length=0)
    ax.axis('on')
    ax.set_title("YOLO: Dividing the Image into a Grid", fontsize=16)

    # Example object
    car = patches.Rectangle((2.5, 3.5), 2, 1, facecolor=COLORS['blue'], alpha=0.7)
    ax.add_patch(car)
    ax.text(3.5, 4, 'Object', color='white', ha='center', va='center', fontsize=12)

    # Cell responsible (center of object)
    ax.plot([3.5], [4], 'o', color=COLORS['red'], markersize=10, label='Object Center')
    ax.text(3.5, 4.3, 'Responsible Cell', color=COLORS['red'], ha='center', fontsize=10)
    
    # Grid labels (optional)
    # for i in range(7):
    #     for j in range(7):
    #         ax.text(i+0.5, j+0.5, f'({i},{j})', ha='center', va='center', fontsize=8, color='gray', alpha=0.5)

    save_plot('yolo_grid.png')

# 5. Loss Function Breakdown (Conceptual, Matplotlib)
def plot_loss_breakdown():
    fig, ax = create_figure(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Loss types
    ax.text(2, 4, "Total Loss", ha='center', fontsize=16, fontweight='bold', color=COLORS['blue'])
    
    ax.text(1.5, 3, "=", fontsize=20, ha='center', va='center')

    ax.text(1.5, 2, "Bounding Box Loss (L_bbox)", ha='center', fontsize=12, color=COLORS['red'])
    ax.text(1.5, 1.5, "How far off is the predicted box? (IoU, L1/L2)", ha='center', fontsize=8, color='gray')
    
    ax.text(4.5, 2, "Class Loss (L_cls)", ha='center', fontsize=12, color=COLORS['green'])
    ax.text(4.5, 1.5, "Is the object correctly classified? (Cross-Entropy)", ha='center', fontsize=8, color='gray')

    ax.text(7.5, 2, "Objectness Loss (L_obj)", ha='center', fontsize=12, color=COLORS['yellow'])
    ax.text(7.5, 1.5, "Does this box contain any object at all? (Binary Cross-Entropy)", ha='center', fontsize=8, color='gray')
    
    ax.text(3, 2, "+", fontsize=16, ha='center', va='center')
    ax.text(6, 2, "+", fontsize=16, ha='center', va='center')

    save_plot('training_loss.png')

# 6. Evaluation Metrics Visualization (Conceptual, Matplotlib for curves)
def plot_metrics_visualization():
    fig, ax = create_figure(figsize=(10, 6))
    
    # Precision-Recall Curve (conceptual)
    x = np.linspace(0, 1, 100)
    y = 1 - x**2 # Example curve
    ax.plot(x, y, color=COLORS['blue'], linewidth=2, label='Precision-Recall Curve')
    
    ax.fill_between(x, 0, y, color=COLORS['blue'], alpha=0.1, label='Area = AP')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Average Precision (AP) for one class", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower left', fontsize=10)
    
    # mAP explanation
    ax.text(0.5, 0.9, "mAP is the mean of AP over all classes\nand usually across multiple IoU thresholds", ha='center', fontsize=10, style='italic')
    
    save_plot('metrics_visualization.png')

# Run all the Matplotlib diagrams
if __name__ == "__main__":
    plot_bbox_anatomy()
    plot_iou_examples()
    plot_nms_visual()
    plot_yolo_grid()
    plot_loss_breakdown()
    plot_metrics_visualization()