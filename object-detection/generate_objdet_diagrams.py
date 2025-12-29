import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Create directories if they don't exist
os.makedirs('diagrams', exist_ok=True)
os.makedirs('diagrams/svg', exist_ok=True)

# Common Style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
COLORS = {
    'blue': '#2E86AB',
    'green': '#06A77D',
    'red': '#E63946',
    'yellow': '#F6AE2D',
    'orange': '#FF8F00',
    'purple': '#9b59b6',
    'gray': '#6c757d',
    'light_bg': '#f8f9fa'
}

def create_figure(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(COLORS['light_bg'])
    fig.patch.set_facecolor(COLORS['light_bg'])
    ax.axis('off')
    return fig, ax

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f'diagrams/svg/{filename}.svg', format='svg', bbox_inches='tight')
    plt.savefig(f'diagrams/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 1. Vision Task Hierarchy
def plot_vision_hierarchy():
    fig, ax = create_figure(figsize=(12, 8))

    levels = [
        ("Classification", 0.75, COLORS['blue'], "What is in this image?\n→ One label: 'dog'"),
        ("Detection", 0.45, COLORS['green'], "What + Where?\n→ Multiple boxes with labels"),
        ("Segmentation", 0.15, COLORS['red'], "Exact shape of each object\n→ Pixel-perfect masks"),
    ]

    box_width = 0.7
    box_height = 0.18

    for label, y, color, description in levels:
        # Main box
        rect = patches.FancyBboxPatch((0.15, y), box_width, box_height,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color, edgecolor='black',
                                       alpha=0.8, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.25, y + box_height/2, label, ha='left', va='center',
                fontsize=18, fontweight='bold', color='white')
        ax.text(0.55, y + box_height/2, description, ha='left', va='center',
                fontsize=12, color='white')

    # Arrows
    for i in range(2):
        y1 = levels[i][1]
        y2 = levels[i+1][1] + box_height
        ax.annotate('', xy=(0.5, y2), xytext=(0.5, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Vision Task Hierarchy", fontsize=20, fontweight='bold', pad=20)

    save_plot('vision_hierarchy')

# 2. Bounding Box Formats
def plot_bbox_formats():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS['light_bg'])

    formats = [
        ("Corner Format\n(x1, y1, x2, y2)", "corners", COLORS['blue']),
        ("Center Format\n(cx, cy, w, h)", "center", COLORS['green']),
        ("Corner+Size\n(x, y, w, h)", "xywh", COLORS['orange']),
    ]

    for ax, (title, fmt, color) in zip(axes, formats):
        ax.set_facecolor(COLORS['light_bg'])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')

        # Draw box
        x1, y1, x2, y2 = 20, 30, 80, 70
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=3, edgecolor=color,
                                  facecolor=color, alpha=0.3)
        ax.add_patch(rect)

        if fmt == "corners":
            ax.plot(x1, y1, 'ko', markersize=10)
            ax.plot(x2, y2, 'ko', markersize=10)
            ax.text(x1-5, y1-5, f"({x1},{y1})", fontsize=10, ha='right')
            ax.text(x2+5, y2+5, f"({x2},{y2})", fontsize=10, ha='left')
        elif fmt == "center":
            cx, cy = (x1+x2)/2, (y1+y2)/2
            ax.plot(cx, cy, 'ko', markersize=12)
            ax.text(cx, cy+5, f"center\n({cx:.0f},{cy:.0f})", fontsize=9, ha='center', va='bottom')
            # Width/height arrows
            ax.annotate('', xy=(x2, cy), xytext=(x1, cy),
                       arrowprops=dict(arrowstyle='<->', color='black', lw=2))
            ax.text(cx, cy-8, f"w={x2-x1}", fontsize=9, ha='center')
        else:  # xywh
            ax.plot(x1, y1, 'ko', markersize=10)
            ax.text(x1-5, y1-5, f"({x1},{y1})", fontsize=10, ha='right')
            ax.annotate('', xy=(x2, y1), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))
            ax.annotate('', xy=(x1, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))
            ax.text((x1+x2)/2, y1-5, f"w={x2-x1}", fontsize=9, ha='center')
            ax.text(x1-5, (y1+y2)/2, f"h={y2-y1}", fontsize=9, ha='right', va='center')

        ax.set_title(title, fontsize=14, fontweight='bold', color=color)
        ax.axis('off')

    plt.suptitle("Bounding Box Format Comparison", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('diagrams/svg/bbox_formats.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/bbox_formats.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. IoU Visualization
def plot_iou_visual():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS['light_bg'])

    scenarios = [
        ("IoU = 0.0\nNo Overlap", [(10, 10, 40, 40), (60, 60, 90, 90)], 0.0),
        ("IoU ≈ 0.5\nGood Match", [(20, 20, 60, 60), (40, 30, 80, 70)], 0.5),
        ("IoU = 1.0\nPerfect", [(25, 25, 75, 75), (25, 25, 75, 75)], 1.0),
    ]

    for ax, (title, boxes, iou) in zip(axes, scenarios):
        ax.set_facecolor(COLORS['light_bg'])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')

        # Box 1 (Ground Truth)
        b1 = boxes[0]
        rect1 = patches.Rectangle((b1[0], b1[1]), b1[2]-b1[0], b1[3]-b1[1],
                                   linewidth=3, edgecolor=COLORS['blue'],
                                   facecolor=COLORS['blue'], alpha=0.3)
        ax.add_patch(rect1)

        # Box 2 (Prediction)
        b2 = boxes[1]
        rect2 = patches.Rectangle((b2[0], b2[1]), b2[2]-b2[0], b2[3]-b2[1],
                                   linewidth=3, edgecolor=COLORS['red'],
                                   facecolor=COLORS['red'], alpha=0.3)
        ax.add_patch(rect2)

        # Intersection (if any)
        if iou > 0:
            ix1 = max(b1[0], b2[0])
            iy1 = max(b1[1], b2[1])
            ix2 = min(b1[2], b2[2])
            iy2 = min(b1[3], b2[3])
            if ix2 > ix1 and iy2 > iy1:
                rect_i = patches.Rectangle((ix1, iy1), ix2-ix1, iy2-iy1,
                                           linewidth=0, facecolor=COLORS['purple'], alpha=0.6)
                ax.add_patch(rect_i)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend([patches.Patch(color=COLORS['blue'], alpha=0.5),
                   patches.Patch(color=COLORS['red'], alpha=0.5)],
                  ['Ground Truth', 'Prediction'], loc='upper right', fontsize=9)
        ax.axis('off')

    plt.suptitle("Intersection over Union (IoU)", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('diagrams/svg/iou_visual.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/iou_visual.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. IoU Formula Diagram
def plot_iou_formula():
    fig, ax = create_figure(figsize=(10, 8))

    # Two overlapping boxes
    b1 = (0.1, 0.3, 0.5, 0.7)
    b2 = (0.3, 0.2, 0.7, 0.6)

    # Box A
    rect1 = patches.Rectangle((b1[0], b1[1]), b1[2]-b1[0], b1[3]-b1[1],
                               linewidth=3, edgecolor=COLORS['blue'],
                               facecolor=COLORS['blue'], alpha=0.3)
    ax.add_patch(rect1)
    ax.text(b1[0]+0.02, b1[3]-0.02, "A", fontsize=16, fontweight='bold', color=COLORS['blue'])

    # Box B
    rect2 = patches.Rectangle((b2[0], b2[1]), b2[2]-b2[0], b2[3]-b2[1],
                               linewidth=3, edgecolor=COLORS['red'],
                               facecolor=COLORS['red'], alpha=0.3)
    ax.add_patch(rect2)
    ax.text(b2[2]-0.05, b2[1]+0.02, "B", fontsize=16, fontweight='bold', color=COLORS['red'])

    # Intersection
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    rect_i = patches.Rectangle((ix1, iy1), ix2-ix1, iy2-iy1,
                               linewidth=2, edgecolor='black',
                               facecolor=COLORS['purple'], alpha=0.7)
    ax.add_patch(rect_i)
    ax.text((ix1+ix2)/2, (iy1+iy2)/2, "Overlap", fontsize=12, fontweight='bold',
            ha='center', va='center', color='white')

    # Formula box
    formula_box = patches.FancyBboxPatch((0.1, 0.78), 0.8, 0.18,
                                          boxstyle="round,pad=0.02",
                                          facecolor='white', edgecolor='black',
                                          alpha=0.9, linewidth=2)
    ax.add_patch(formula_box)
    ax.text(0.5, 0.87, "IoU = Intersection / Union",
            ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.82, "= Area of Overlap / (Area A + Area B − Overlap)",
            ha='center', va='center', fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("IoU Calculation", fontsize=20, fontweight='bold', pad=20)

    save_plot('iou_formula')

# 5. NMS Before/After
def plot_nms_visual():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(COLORS['light_bg'])

    # Before NMS - multiple overlapping boxes
    ax1 = axes[0]
    ax1.set_facecolor('#E8F5E9')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_aspect('equal')

    # Draw overlapping boxes with different confidences
    boxes = [
        (25, 30, 65, 75, 0.95),  # Best
        (22, 28, 62, 72, 0.91),
        (28, 32, 68, 78, 0.88),
        (20, 25, 60, 70, 0.85),
        (30, 35, 70, 80, 0.82),
    ]

    for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
        alpha = 0.3 + (conf - 0.8) * 2
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor=COLORS['red'],
                                  facecolor=COLORS['red'], alpha=alpha)
        ax1.add_patch(rect)
        ax1.text(x2+2, y2, f"{conf:.2f}", fontsize=9, va='top')

    ax1.set_title("Before NMS: 5 overlapping boxes", fontsize=14, fontweight='bold')
    ax1.axis('off')

    # After NMS - single best box
    ax2 = axes[1]
    ax2.set_facecolor('#E8F5E9')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.set_aspect('equal')

    rect = patches.Rectangle((25, 30), 40, 45,
                              linewidth=4, edgecolor=COLORS['green'],
                              facecolor=COLORS['green'], alpha=0.4)
    ax2.add_patch(rect)
    ax2.text(67, 77, "0.95", fontsize=12, fontweight='bold', color=COLORS['green'])
    ax2.text(45, 52, "Dog", fontsize=16, fontweight='bold', ha='center', va='center')

    ax2.set_title("After NMS: 1 clean detection", fontsize=14, fontweight='bold')
    ax2.axis('off')

    plt.suptitle("Non-Maximum Suppression (NMS)", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('diagrams/svg/nms_visual.svg', format='svg', bbox_inches='tight')
    plt.savefig('diagrams/nms_visual.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6. YOLO Grid
def plot_yolo_grid():
    fig, ax = create_figure(figsize=(10, 10))

    grid_size = 7
    cell_size = 1.0 / grid_size

    # Draw grid
    for i in range(grid_size + 1):
        ax.axhline(y=i*cell_size, color='gray', linewidth=1, alpha=0.5)
        ax.axvline(x=i*cell_size, color='gray', linewidth=1, alpha=0.5)

    # Draw a bounding box spanning multiple cells
    bbox = patches.Rectangle((0.35, 0.25), 0.35, 0.45,
                              linewidth=4, edgecolor=COLORS['red'],
                              facecolor=COLORS['red'], alpha=0.2)
    ax.add_patch(bbox)

    # Highlight the center cell (responsible cell)
    center_x, center_y = 0.35 + 0.35/2, 0.25 + 0.45/2
    cell_i, cell_j = int(center_x * grid_size), int(center_y * grid_size)
    responsible_cell = patches.Rectangle((cell_i*cell_size, cell_j*cell_size),
                                          cell_size, cell_size,
                                          linewidth=3, edgecolor=COLORS['green'],
                                          facecolor=COLORS['green'], alpha=0.4)
    ax.add_patch(responsible_cell)

    # Mark the center
    ax.plot(center_x, center_y, 'ko', markersize=12)
    ax.plot(center_x, center_y, 'yo', markersize=8)

    # Label
    ax.text(0.5, -0.08, "The cell containing the object's CENTER is 'responsible' for detecting it",
            ha='center', fontsize=12, style='italic')

    ax.text(center_x + 0.02, center_y + 0.05, "Center", fontsize=10, fontweight='bold')

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.12, 1.02)
    ax.set_title(f"YOLO {grid_size}×{grid_size} Grid Division", fontsize=18, fontweight='bold', pad=20)

    save_plot('yolo_grid')

# 7. Detector Types (Two-stage vs One-stage)
def plot_detector_types():
    fig, ax = create_figure(figsize=(14, 7))

    # Two-stage detector (left)
    ax.text(0.25, 0.95, "Two-Stage Detectors", ha='center', fontsize=16, fontweight='bold', color=COLORS['blue'])
    ax.text(0.25, 0.88, "(Accurate but slower)", ha='center', fontsize=11, color='gray')

    two_stage = [
        ("Image", 0.25, 0.75, COLORS['blue']),
        ("Region Proposal\nNetwork (RPN)", 0.25, 0.55, COLORS['blue']),
        ("ROI Pooling", 0.25, 0.35, COLORS['blue']),
        ("Classification\n+ Refinement", 0.25, 0.15, COLORS['blue']),
    ]

    for label, x, y, color in two_stage:
        rect = patches.FancyBboxPatch((x-0.12, y-0.06), 0.24, 0.12,
                                       boxstyle="round,pad=0.01",
                                       facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    for i in range(len(two_stage)-1):
        ax.annotate('', xy=(0.25, two_stage[i+1][2]+0.06), xytext=(0.25, two_stage[i][2]-0.06),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # One-stage detector (right)
    ax.text(0.75, 0.95, "One-Stage Detectors", ha='center', fontsize=16, fontweight='bold', color=COLORS['green'])
    ax.text(0.75, 0.88, "(Fast, good enough)", ha='center', fontsize=11, color='gray')

    one_stage = [
        ("Image", 0.75, 0.70, COLORS['green']),
        ("Backbone +\nDetection Head", 0.75, 0.45, COLORS['green']),
        ("All Detections\n(one pass!)", 0.75, 0.20, COLORS['green']),
    ]

    for label, x, y, color in one_stage:
        rect = patches.FancyBboxPatch((x-0.12, y-0.08), 0.24, 0.16,
                                       boxstyle="round,pad=0.01",
                                       facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    for i in range(len(one_stage)-1):
        ax.annotate('', xy=(0.75, one_stage[i+1][2]+0.08), xytext=(0.75, one_stage[i][2]-0.08),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Examples
    ax.text(0.25, 0.02, "Examples: R-CNN, Faster R-CNN, Mask R-CNN\n~5-10 FPS",
            ha='center', fontsize=10, style='italic')
    ax.text(0.75, 0.02, "Examples: YOLO, SSD, RetinaNet\n~30-100+ FPS",
            ha='center', fontsize=10, style='italic')

    # Dividing line
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Two Families of Object Detectors", fontsize=20, fontweight='bold', pad=20)

    save_plot('detector_types')

# 8. Precision-Recall Curve
def plot_precision_recall():
    fig, ax = create_figure(figsize=(10, 8))
    ax.axis('on')

    # Sample PR curve
    recall = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    precision = np.array([1.0, 0.98, 0.95, 0.92, 0.88, 0.82, 0.75, 0.65, 0.50, 0.30, 0.10])

    # Fill under curve
    ax.fill_between(recall, precision, alpha=0.3, color=COLORS['blue'])
    ax.plot(recall, precision, 'o-', color=COLORS['blue'], linewidth=3, markersize=8)

    # AP annotation
    ax.text(0.5, 0.5, f"AP = Area Under Curve\n≈ 0.72", ha='center', va='center',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))

    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Precision-Recall Curve", fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3)

    save_plot('precision_recall')

# 9. Detection Pipeline
def plot_detection_pipeline():
    fig, ax = create_figure(figsize=(14, 6))

    stages = [
        ("Input\nImage", 0.08, COLORS['blue']),
        ("Backbone\n(Extract Features)", 0.28, COLORS['green']),
        ("Neck\n(FPN/PAN)", 0.48, COLORS['yellow']),
        ("Head\n(Predict Boxes)", 0.68, COLORS['orange']),
        ("NMS\n(Clean Up)", 0.88, COLORS['red']),
    ]

    for label, x, color in stages:
        rect = patches.FancyBboxPatch((x-0.08, 0.35), 0.16, 0.3,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color, edgecolor='black', alpha=0.8, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 0.5, label, ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Arrows
    for i in range(len(stages)-1):
        ax.annotate('', xy=(stages[i+1][1]-0.08, 0.5), xytext=(stages[i][1]+0.08, 0.5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=3))

    # Output label
    ax.text(0.5, 0.15, "Output: List of [class, confidence, x1, y1, x2, y2]",
            ha='center', fontsize=12, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.8)
    ax.set_title("Object Detection Pipeline", fontsize=18, fontweight='bold', pad=20)

    save_plot('detection_pipeline')

# 10. Anchor Boxes
def plot_anchor_boxes():
    fig, ax = create_figure(figsize=(12, 8))

    # Draw a grid cell
    cell_rect = patches.Rectangle((0.35, 0.35), 0.3, 0.3,
                                   linewidth=2, edgecolor='gray',
                                   facecolor='white', alpha=0.5)
    ax.add_patch(cell_rect)
    ax.text(0.5, 0.8, "One Grid Cell", ha='center', fontsize=12, fontweight='bold')

    # Draw anchor boxes (different shapes)
    anchors = [
        (0.5, 0.5, 0.15, 0.3, COLORS['blue'], "Tall\n(person)"),   # tall
        (0.5, 0.5, 0.25, 0.15, COLORS['green'], "Wide\n(car)"),   # wide
        (0.5, 0.5, 0.2, 0.2, COLORS['red'], "Square\n(ball)"),    # square
    ]

    offsets = [-0.25, 0, 0.25]

    for i, (cx, cy, w, h, color, label) in enumerate(anchors):
        offset = offsets[i]
        rect = patches.Rectangle((cx - w/2 + offset*0.8, cy - h/2), w, h,
                                  linewidth=3, edgecolor=color,
                                  facecolor=color, alpha=0.3)
        ax.add_patch(rect)
        ax.text(cx + offset*0.8, cy - h/2 - 0.05, label, ha='center', fontsize=10, color=color, fontweight='bold')

    ax.text(0.5, 0.15, "Each anchor predicts offsets: predicted_w = anchor_w × exp(tw)",
            ha='center', fontsize=11, style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Anchor Boxes: Pre-defined Shapes", fontsize=18, fontweight='bold', pad=20)

    save_plot('anchor_boxes')

# 11. Loss Function Components
def plot_loss_function():
    fig, ax = create_figure(figsize=(12, 8))

    # Total loss equation at top
    ax.text(0.5, 0.92, "Total Loss = w1*Lbox + w2*Lobj + w3*Lclass",
            ha='center', fontsize=18, fontweight='bold')

    components = [
        ("Lbox\nLocalization Loss", 0.2, 0.55, COLORS['blue'], "Is the box in the\nright place?"),
        ("Lobj\nObjectness Loss", 0.5, 0.55, COLORS['green'], "Is there an\nobject here?"),
        ("Lclass\nClassification Loss", 0.8, 0.55, COLORS['red'], "What class\nis it?"),
    ]

    for label, x, y, color, desc in components:
        rect = patches.FancyBboxPatch((x-0.12, y-0.12), 0.24, 0.24,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color, edgecolor='black', alpha=0.8, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y+0.02, label, ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        ax.text(x, y+0.22, desc, ha='center', va='bottom', fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1)
    ax.set_title("Detection Loss Function", fontsize=20, fontweight='bold', pad=20)

    save_plot('loss_function')

# 12. mAP Metric Visualization
def plot_map_metric():
    fig, ax = create_figure(figsize=(12, 7))

    # Class AP values
    classes = ['person', 'car', 'dog', 'cat', 'bicycle']
    aps = [0.85, 0.92, 0.78, 0.71, 0.68]
    colors = [COLORS['blue'], COLORS['green'], COLORS['red'], COLORS['yellow'], COLORS['orange']]

    y_positions = np.arange(len(classes))

    # Horizontal bar chart
    ax.axis('on')
    bars = ax.barh(y_positions, aps, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(classes, fontsize=12)
    ax.set_xlabel("Average Precision (AP)", fontsize=12)
    ax.set_xlim(0, 1.1)

    # Add value labels
    for i, (bar, ap) in enumerate(zip(bars, aps)):
        ax.text(ap + 0.02, i, f'{ap:.2f}', va='center', fontsize=11, fontweight='bold')

    # mAP line
    mean_ap = np.mean(aps)
    ax.axvline(x=mean_ap, color='black', linestyle='--', linewidth=2, label=f'mAP = {mean_ap:.2f}')
    ax.legend(loc='lower right', fontsize=12)

    ax.set_title("Mean Average Precision (mAP) Calculation", fontsize=18, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    save_plot('map_metric')

# Run all diagrams
if __name__ == "__main__":
    print("Generating object detection diagrams...")
    plot_vision_hierarchy()
    print("  - vision_hierarchy.svg/png")
    plot_bbox_formats()
    print("  - bbox_formats.svg/png")
    plot_iou_visual()
    print("  - iou_visual.svg/png")
    plot_iou_formula()
    print("  - iou_formula.svg/png")
    plot_nms_visual()
    print("  - nms_visual.svg/png")
    plot_yolo_grid()
    print("  - yolo_grid.svg/png")
    plot_detector_types()
    print("  - detector_types.svg/png")
    plot_precision_recall()
    print("  - precision_recall.svg/png")
    plot_detection_pipeline()
    print("  - detection_pipeline.svg/png")
    plot_anchor_boxes()
    print("  - anchor_boxes.svg/png")
    plot_loss_function()
    print("  - loss_function.svg/png")
    plot_map_metric()
    print("  - map_metric.svg/png")
    print("Done!")
