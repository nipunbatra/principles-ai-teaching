#!/usr/bin/env python3
"""
Generate realistic ML task examples using actual datasets.
Creates visualization images for use in lecture slides.

Uses real datasets:
- CIFAR-10: Classification
- COCO: Detection, Segmentation, Pose
- MNIST: Digit recognition
- Hugging Face datasets for NLP examples

Run: python generate_realistic_examples.py
"""

import os
import urllib.request
from pathlib import Path
import json

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "examples"
OUTPUT_DIR.mkdir(exist_ok=True)

# Sample images directory
SAMPLE_DIR = OUTPUT_DIR / "samples"
SAMPLE_DIR.mkdir(exist_ok=True)


def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    try:
        from ultralytics import YOLO
    except ImportError:
        missing.append("ultralytics")
    try:
        import supervision
    except ImportError:
        missing.append("supervision")
    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")

    if missing:
        print("Missing packages. Install with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True


def generate_classification_example():
    """Generate a classification visualization showing ImageNet-style prediction."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Example 1: Dog classification
    ax = axes[0]
    # Create a simple dog-like shape
    img = np.random.rand(224, 224, 3) * 0.3 + 0.3
    ax.imshow(img)
    ax.set_title("Input: Dog Image", fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add annotation
    ax.text(112, 240, "Class: Golden Retriever\nConfidence: 94.2%",
            ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Example 2: Bar chart of predictions
    ax = axes[1]
    classes = ['Golden Retriever', 'Labrador', 'Beagle', 'Bulldog', 'Poodle']
    probs = [0.942, 0.031, 0.015, 0.008, 0.004]
    colors = ['#4CAF50', '#81C784', '#A5D6A7', '#C8E6C9', '#E8F5E9']
    ax.barh(classes, probs, color=colors)
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title('Top-5 Predictions', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    for i, (cls, prob) in enumerate(zip(classes, probs)):
        ax.text(prob + 0.02, i, f'{prob:.1%}', va='center')

    # Example 3: Multi-class concept
    ax = axes[2]
    ax.text(0.5, 0.8, "Classification Output:", ha='center', fontsize=14, fontweight='bold')
    ax.text(0.5, 0.6, "1 image -> 1 label", ha='center', fontsize=16)
    ax.text(0.5, 0.35, "Softmax: probabilities sum to 1", ha='center', fontsize=12)
    ax.text(0.5, 0.15, "P(dog) + P(cat) + ... = 1.0", ha='center', fontsize=12,
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "classification_example.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'classification_example.png'}")


def generate_detection_example():
    """Generate detection visualization using YOLO on sample images."""
    try:
        from ultralytics import YOLO
        import supervision as sv
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        import urllib.request

        # Download a sample image
        sample_url = "https://ultralytics.com/images/bus.jpg"
        sample_path = OUTPUT_DIR / "sample_bus.jpg"

        if not sample_path.exists():
            print("Downloading sample image...")
            urllib.request.urlretrieve(sample_url, sample_path)

        # Load YOLO model
        print("Loading YOLOv8n model...")
        model = YOLO('yolov8n.pt')

        # Run detection
        results = model(str(sample_path), verbose=False)

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Original image
        img = Image.open(sample_path)
        axes[0].imshow(img)
        axes[0].set_title("Input Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Detection result
        result_img = results[0].plot()
        axes[1].imshow(result_img)
        axes[1].set_title("YOLO Detection Output", fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # Add detection summary
        boxes = results[0].boxes
        summary = f"Detected: {len(boxes)} objects"
        for box in boxes[:5]:  # Show first 5
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            summary += f"\n  - {name}: {conf:.0%}"

        fig.text(0.5, 0.02, summary, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "detection_example.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Created: {OUTPUT_DIR / 'detection_example.png'}")

    except Exception as e:
        print(f"Detection example failed: {e}")
        print("Creating placeholder...")
        generate_detection_placeholder()


def generate_detection_placeholder():
    """Create a placeholder detection diagram."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Background
    ax.set_facecolor('#f0f0f0')

    # Simulated image area
    ax.add_patch(patches.Rectangle((0.05, 0.1), 0.9, 0.8,
                                    facecolor='#e0e0e0', edgecolor='black', linewidth=2))

    # Detection boxes
    detections = [
        {'label': 'Person', 'conf': 0.95, 'box': (0.1, 0.2, 0.25, 0.6), 'color': '#4CAF50'},
        {'label': 'Car', 'conf': 0.91, 'box': (0.4, 0.4, 0.35, 0.35), 'color': '#2196F3'},
        {'label': 'Dog', 'conf': 0.88, 'box': (0.55, 0.6, 0.2, 0.2), 'color': '#FF9800'},
    ]

    for det in detections:
        x, y, w, h = det['box']
        rect = patches.Rectangle((x, y), w, h, linewidth=3,
                                  edgecolor=det['color'], facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y + h + 0.02, f"{det['label']} {det['conf']:.0%}",
                fontsize=11, fontweight='bold', color=det['color'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Object Detection: What + Where", fontsize=16, fontweight='bold')
    ax.axis('off')

    # Add legend
    ax.text(0.5, 0.02, "Output: List of (class, confidence, x, y, width, height)",
            ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.savefig(OUTPUT_DIR / "detection_example.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'detection_example.png'}")


def generate_segmentation_comparison():
    """Generate comparison of semantic vs instance segmentation."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Create base "scene"
    H, W = 200, 300

    # Semantic segmentation
    ax = axes[0]
    semantic = np.zeros((H, W, 3))
    # Sky (light blue)
    semantic[:50, :] = [0.53, 0.81, 0.92]
    # Road (gray)
    semantic[150:, :] = [0.5, 0.5, 0.5]
    # Car 1 (red)
    semantic[80:130, 30:100] = [0.9, 0.2, 0.2]
    # Car 2 (red - same color!)
    semantic[80:130, 180:250] = [0.9, 0.2, 0.2]
    # Person (green)
    semantic[60:140, 130:160] = [0.2, 0.8, 0.2]

    ax.imshow(semantic)
    ax.set_title("Semantic Segmentation", fontsize=14, fontweight='bold')
    ax.text(150, 220, "Same class = Same color\nCan't distinguish cars!",
            ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.axis('off')

    # Instance segmentation
    ax = axes[1]
    instance = np.zeros((H, W, 3))
    # Sky
    instance[:50, :] = [0.53, 0.81, 0.92]
    # Road
    instance[150:, :] = [0.5, 0.5, 0.5]
    # Car 1 (red)
    instance[80:130, 30:100] = [0.9, 0.2, 0.2]
    # Car 2 (blue - different!)
    instance[80:130, 180:250] = [0.2, 0.2, 0.9]
    # Person (green)
    instance[60:140, 130:160] = [0.2, 0.8, 0.2]

    ax.imshow(instance)
    ax.set_title("Instance Segmentation", fontsize=14, fontweight='bold')
    ax.text(150, 220, "Each object = Unique ID\nCar #1 vs Car #2!",
            ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.axis('off')

    # Summary
    ax = axes[2]
    ax.text(0.5, 0.85, "Key Difference", ha='center', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.65, "Semantic: pixel -> class", ha='center', fontsize=14)
    ax.text(0.5, 0.50, "Instance: pixel -> class + object ID", ha='center', fontsize=14)
    ax.text(0.5, 0.30, "Self-driving cars need Instance\nto track individual vehicles!",
            ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "segmentation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'segmentation_comparison.png'}")


def generate_iou_visualization():
    """Generate IoU (Intersection over Union) visualization."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    cases = [
        ("IoU = 0.0", (0.1, 0.2), (0.6, 0.2), 0.0),
        ("IoU = 0.3", (0.15, 0.2), (0.4, 0.2), 0.3),
        ("IoU = 0.7", (0.2, 0.2), (0.3, 0.2), 0.7),
        ("IoU = 1.0", (0.2, 0.2), (0.2, 0.2), 1.0),
    ]

    for ax, (title, box1_xy, box2_xy, iou) in zip(axes, cases):
        # Box dimensions
        w, h = 0.3, 0.5

        # Draw boxes
        rect1 = patches.Rectangle(box1_xy, w, h, linewidth=3,
                                   edgecolor='#2196F3', facecolor='#BBDEFB', alpha=0.7)
        rect2 = patches.Rectangle(box2_xy, w, h, linewidth=3,
                                   edgecolor='#4CAF50', facecolor='#C8E6C9', alpha=0.7)

        ax.add_patch(rect1)
        ax.add_patch(rect2)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        # Add labels
        ax.text(box1_xy[0] + w/2, box1_xy[1] - 0.05, "Ground Truth", ha='center', fontsize=10, color='#2196F3')
        ax.text(box2_xy[0] + w/2, box2_xy[1] + h + 0.05, "Prediction", ha='center', fontsize=10, color='#4CAF50')

        # Quality label
        if iou >= 0.5:
            label = "Correct (TP)"
            color = '#4CAF50'
        else:
            label = "Wrong (FP)"
            color = '#F44336'
        ax.text(0.5, 0.02, label, ha='center', fontsize=11, fontweight='bold', color=color)

    plt.suptitle("IoU (Intersection over Union) - Measuring Detection Quality", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "iou_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'iou_visualization.png'}")


def generate_bbox_formats():
    """Generate bounding box format comparison."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    formats = [
        {
            'name': 'Corner Format (x1, y1, x2, y2)',
            'values': '[100, 50, 300, 200]',
            'desc': 'Top-left and bottom-right corners\nUsed by: PyTorch, PASCAL VOC',
            'points': [(0.2, 0.7), (0.8, 0.3)],
        },
        {
            'name': 'Center Format (cx, cy, w, h)',
            'values': '[200, 125, 200, 150]',
            'desc': 'Center point + dimensions\nUsed by: YOLO',
            'points': [(0.5, 0.5)],  # center
        },
        {
            'name': 'COCO Format (x, y, w, h)',
            'values': '[100, 50, 200, 150]',
            'desc': 'Top-left corner + dimensions\nUsed by: COCO dataset',
            'points': [(0.2, 0.7)],  # top-left
        },
    ]

    for ax, fmt in zip(axes, formats):
        # Draw box
        rect = patches.Rectangle((0.2, 0.3), 0.6, 0.4, linewidth=3,
                                   edgecolor='#2196F3', facecolor='#E3F2FD', alpha=0.8)
        ax.add_patch(rect)

        # Draw reference points
        if fmt['name'].startswith('Corner'):
            # Top-left
            ax.plot(0.2, 0.7, 'o', markersize=12, color='#4CAF50')
            ax.text(0.15, 0.75, '(x1, y1)', fontsize=10, color='#4CAF50')
            # Bottom-right
            ax.plot(0.8, 0.3, 'o', markersize=12, color='#F44336')
            ax.text(0.82, 0.35, '(x2, y2)', fontsize=10, color='#F44336')
        elif fmt['name'].startswith('Center'):
            # Center
            ax.plot(0.5, 0.5, 'o', markersize=12, color='#9C27B0')
            ax.text(0.52, 0.55, '(cx, cy)', fontsize=10, color='#9C27B0')
            # Width/height arrows
            ax.annotate('', xy=(0.8, 0.5), xytext=(0.2, 0.5),
                       arrowprops=dict(arrowstyle='<->', color='#FF9800', lw=2))
            ax.text(0.5, 0.45, 'w', fontsize=10, ha='center', color='#FF9800')
        else:
            # Top-left
            ax.plot(0.2, 0.7, 'o', markersize=12, color='#4CAF50')
            ax.text(0.15, 0.75, '(x, y)', fontsize=10, color='#4CAF50')
            # Width/height
            ax.annotate('', xy=(0.8, 0.7), xytext=(0.2, 0.7),
                       arrowprops=dict(arrowstyle='<->', color='#FF9800', lw=2))
            ax.text(0.5, 0.73, 'w', fontsize=10, ha='center', color='#FF9800')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(fmt['name'], fontsize=12, fontweight='bold')
        ax.text(0.5, 0.15, fmt['values'], ha='center', fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
        ax.text(0.5, 0.05, fmt['desc'], ha='center', fontsize=9)
        ax.axis('off')

    plt.suptitle("Bounding Box Formats - Know Your Coordinates!", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bbox_formats.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'bbox_formats.png'}")


def generate_ml_paradigms():
    """Generate ML paradigms comparison (supervised/unsupervised/self-supervised)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    paradigms = [
        {
            'name': 'Supervised Learning',
            'color': '#4CAF50',
            'bg': '#E8F5E9',
            'data': 'X (features) + Y (labels)',
            'example': 'Image -> "Cat"\nEmail -> "Spam"\nHouse features -> $500K',
            'tasks': ['Classification', 'Regression', 'Detection', 'Segmentation'],
        },
        {
            'name': 'Unsupervised Learning',
            'color': '#FF9800',
            'bg': '#FFF3E0',
            'data': 'X only (no labels!)',
            'example': 'Find customer groups\nFind anomalies\nCompress data',
            'tasks': ['Clustering', 'Dim. Reduction', 'Anomaly Detection'],
        },
        {
            'name': 'Self-Supervised Learning',
            'color': '#2196F3',
            'bg': '#E3F2FD',
            'data': 'X creates its own Y',
            'example': '"The cat sat" -> "on"\nPredict next word\nMask and predict',
            'tasks': ['Next Token (GPT)', 'Masked LM (BERT)', 'Contrastive Learning'],
        },
    ]

    for ax, p in zip(axes, paradigms):
        ax.set_facecolor(p['bg'])

        ax.text(0.5, 0.95, p['name'], ha='center', fontsize=16, fontweight='bold', color=p['color'])

        ax.text(0.5, 0.78, "Data:", ha='center', fontsize=12, fontweight='bold')
        ax.text(0.5, 0.70, p['data'], ha='center', fontsize=11)

        ax.text(0.5, 0.55, "Examples:", ha='center', fontsize=12, fontweight='bold')
        ax.text(0.5, 0.40, p['example'], ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.text(0.5, 0.20, "Tasks:", ha='center', fontsize=12, fontweight='bold')
        tasks_str = ', '.join(p['tasks'])
        ax.text(0.5, 0.08, tasks_str, ha='center', fontsize=9, wrap=True)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.suptitle("The Three Learning Paradigms", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ml_paradigms.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'ml_paradigms.png'}")


def generate_vision_hierarchy():
    """Generate vision task hierarchy visualization."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    tasks = [
        {
            'name': 'Classification',
            'output': '"Dog"',
            'desc': 'What is in the image?',
            'color': '#4CAF50',
        },
        {
            'name': 'Detection',
            'output': 'Box + "Dog"',
            'desc': 'What + Where?',
            'color': '#2196F3',
        },
        {
            'name': 'Segmentation',
            'output': 'Pixel mask',
            'desc': 'Exact boundaries',
            'color': '#FF9800',
        },
        {
            'name': 'Pose Estimation',
            'output': 'Keypoints',
            'desc': 'Body parts',
            'color': '#9C27B0',
        },
    ]

    for ax, task in zip(axes, tasks):
        # Title
        ax.text(0.5, 0.95, task['name'], ha='center', fontsize=14, fontweight='bold', color=task['color'])

        # Simulated "image"
        ax.add_patch(patches.Rectangle((0.1, 0.25), 0.8, 0.55,
                                         facecolor='#f0f0f0', edgecolor='black', linewidth=2))

        if task['name'] == 'Classification':
            ax.text(0.5, 0.52, '[Dog]', ha='center', fontsize=16)
        elif task['name'] == 'Detection':
            ax.add_patch(patches.Rectangle((0.25, 0.35), 0.5, 0.35,
                                             facecolor='none', edgecolor=task['color'], linewidth=3))
            ax.text(0.5, 0.72, 'Dog', ha='center', fontsize=12, color=task['color'])
        elif task['name'] == 'Segmentation':
            # Filled shape
            ax.add_patch(patches.Ellipse((0.5, 0.52), 0.4, 0.35,
                                           facecolor=task['color'], alpha=0.5))
        else:  # Pose
            # Stick figure keypoints
            points = [(0.5, 0.7), (0.5, 0.55), (0.35, 0.5), (0.65, 0.5), (0.4, 0.35), (0.6, 0.35)]
            for px, py in points:
                ax.plot(px, py, 'o', markersize=8, color=task['color'])
            # Lines
            ax.plot([0.5, 0.5], [0.7, 0.55], '-', color=task['color'], lw=2)
            ax.plot([0.35, 0.65], [0.5, 0.5], '-', color=task['color'], lw=2)
            ax.plot([0.5, 0.4], [0.55, 0.35], '-', color=task['color'], lw=2)
            ax.plot([0.5, 0.6], [0.55, 0.35], '-', color=task['color'], lw=2)

        ax.text(0.5, 0.12, f'Output: {task["output"]}', ha='center', fontsize=11, fontweight='bold')
        ax.text(0.5, 0.03, task['desc'], ha='center', fontsize=10)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    # Arrow showing progression
    plt.suptitle("Vision Tasks: Increasing Precision", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vision_hierarchy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'vision_hierarchy.png'}")


def download_sample_images():
    """Download sample images for demonstrations."""
    samples = {
        'bus.jpg': 'https://ultralytics.com/images/bus.jpg',
        'zidane.jpg': 'https://ultralytics.com/images/zidane.jpg',
        'dog.jpg': 'https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=640',
        'cat.jpg': 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=640',
        'street.jpg': 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=640',
    }

    for name, url in samples.items():
        path = SAMPLE_DIR / name
        if not path.exists():
            try:
                print(f"Downloading {name}...")
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"Could not download {name}: {e}")


def generate_cifar10_example():
    """Generate CIFAR-10 classification example with real images."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Try to load actual CIFAR-10
        try:
            from torchvision import datasets, transforms

            cifar = datasets.CIFAR10(root=str(SAMPLE_DIR / 'cifar10'), train=False, download=True)
            classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']

            fig, axes = plt.subplots(2, 5, figsize=(12, 5))
            axes = axes.flatten()

            for i, ax in enumerate(axes):
                img, label = cifar[i * 100]
                ax.imshow(img)
                ax.set_title(f"{classes[label]}", fontsize=12, fontweight='bold')
                ax.axis('off')

            plt.suptitle("CIFAR-10: Real Image Classification Dataset", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "cifar10_examples.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Created: {OUTPUT_DIR / 'cifar10_examples.png'}")

        except Exception as e:
            print(f"Could not load CIFAR-10: {e}")
            # Create placeholder
            generate_classification_example()

    except ImportError:
        print("matplotlib not available for CIFAR-10 example")


def generate_mnist_example():
    """Generate MNIST digit recognition example."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        try:
            from torchvision import datasets

            mnist = datasets.MNIST(root=str(SAMPLE_DIR / 'mnist'), train=False, download=True)

            fig, axes = plt.subplots(2, 5, figsize=(10, 4))
            axes = axes.flatten()

            for i, ax in enumerate(axes):
                img, label = mnist[i * 100]
                ax.imshow(np.array(img), cmap='gray')
                ax.set_title(f"Label: {label}", fontsize=12, fontweight='bold')
                ax.axis('off')

            plt.suptitle("MNIST: Handwritten Digit Recognition", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "mnist_examples.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Created: {OUTPUT_DIR / 'mnist_examples.png'}")

        except Exception as e:
            print(f"Could not load MNIST: {e}")

    except ImportError:
        print("Required packages not available for MNIST example")


def generate_coco_detection_example():
    """Generate COCO detection example with real images."""
    try:
        from ultralytics import YOLO
        import matplotlib.pyplot as plt
        from PIL import Image

        # Download sample images first
        download_sample_images()

        # Load model
        model = YOLO('yolov8n.pt')

        # Process multiple images
        sample_images = ['bus.jpg', 'zidane.jpg']
        available_images = [SAMPLE_DIR / img for img in sample_images if (SAMPLE_DIR / img).exists()]

        if not available_images:
            print("No sample images available")
            generate_detection_placeholder()
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for idx, img_path in enumerate(available_images[:2]):
            # Original
            img = Image.open(img_path)
            axes[idx, 0].imshow(img)
            axes[idx, 0].set_title("Input Image", fontsize=12, fontweight='bold')
            axes[idx, 0].axis('off')

            # Detection
            results = model(str(img_path), verbose=False)
            result_img = results[0].plot()
            axes[idx, 1].imshow(result_img)
            axes[idx, 1].set_title("YOLOv8 Detection", fontsize=12, fontweight='bold')
            axes[idx, 1].axis('off')

        plt.suptitle("Object Detection on Real Images (COCO-trained YOLOv8)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "coco_detection_examples.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Created: {OUTPUT_DIR / 'coco_detection_examples.png'}")

    except Exception as e:
        print(f"Could not generate COCO detection: {e}")
        generate_detection_placeholder()


def generate_coco_segmentation_example():
    """Generate segmentation example using YOLOv8-seg."""
    try:
        from ultralytics import YOLO
        import matplotlib.pyplot as plt
        from PIL import Image

        # Download sample images first
        download_sample_images()

        # Load segmentation model
        model = YOLO('yolov8n-seg.pt')

        img_path = SAMPLE_DIR / 'bus.jpg'
        if not img_path.exists():
            print("Sample image not available")
            return

        # Run segmentation
        results = model(str(img_path), verbose=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Original
        img = Image.open(img_path)
        axes[0].imshow(img)
        axes[0].set_title("Input Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Segmentation
        result_img = results[0].plot()
        axes[1].imshow(result_img)
        axes[1].set_title("Instance Segmentation (YOLOv8-seg)", fontsize=14, fontweight='bold')
        axes[1].axis('off')

        plt.suptitle("Instance Segmentation: Pixel-Perfect Object Boundaries", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "instance_segmentation_example.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Created: {OUTPUT_DIR / 'instance_segmentation_example.png'}")

    except Exception as e:
        print(f"Could not generate segmentation: {e}")


def generate_pose_estimation_example():
    """Generate pose estimation example using YOLOv8-pose."""
    try:
        from ultralytics import YOLO
        import matplotlib.pyplot as plt
        from PIL import Image

        download_sample_images()

        # Load pose model
        model = YOLO('yolov8n-pose.pt')

        img_path = SAMPLE_DIR / 'zidane.jpg'
        if not img_path.exists():
            print("Sample image not available for pose estimation")
            return

        results = model(str(img_path), verbose=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Original
        img = Image.open(img_path)
        axes[0].imshow(img)
        axes[0].set_title("Input Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Pose
        result_img = results[0].plot()
        axes[1].imshow(result_img)
        axes[1].set_title("Pose Estimation (YOLOv8-pose)", fontsize=14, fontweight='bold')
        axes[1].axis('off')

        plt.suptitle("Human Pose Estimation: 17 Body Keypoints", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "pose_estimation_example.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Created: {OUTPUT_DIR / 'pose_estimation_example.png'}")

    except Exception as e:
        print(f"Could not generate pose estimation: {e}")


def generate_ner_example():
    """Generate Named Entity Recognition example."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))

        # Example sentence with entities highlighted
        text = "Elon Musk announced that Tesla will open a factory in Berlin by March 2025."
        entities = [
            ("Elon Musk", "PERSON", "#4CAF50"),
            ("Tesla", "ORG", "#2196F3"),
            ("Berlin", "LOC", "#FF9800"),
            ("March 2025", "DATE", "#9C27B0"),
        ]

        ax.text(0.5, 0.8, "Named Entity Recognition (NER)", ha='center', fontsize=18, fontweight='bold')
        ax.text(0.5, 0.6, text, ha='center', fontsize=14, wrap=True)

        # Legend
        y_pos = 0.35
        for entity, label, color in entities:
            ax.text(0.15 + entities.index((entity, label, color)) * 0.2, y_pos,
                   f"{entity}\n[{label}]", ha='center', fontsize=12, fontweight='bold', color=color,
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

        ax.text(0.5, 0.1, "Input: Raw text -> Output: Text with labeled entities", ha='center', fontsize=11)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "ner_example.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Created: {OUTPUT_DIR / 'ner_example.png'}")

    except Exception as e:
        print(f"Could not generate NER example: {e}")


def generate_sentiment_example():
    """Generate sentiment analysis example."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        examples = [
            ("This movie was absolutely amazing!", "POSITIVE", 0.95, "#4CAF50"),
            ("The food was okay, nothing special.", "NEUTRAL", 0.72, "#FF9800"),
            ("Worst experience ever. Never going back.", "NEGATIVE", 0.98, "#F44336"),
        ]

        ax.text(0.5, 0.92, "Sentiment Analysis", ha='center', fontsize=18, fontweight='bold')

        for i, (text, sentiment, conf, color) in enumerate(examples):
            y = 0.7 - i * 0.25
            ax.text(0.05, y, f'"{text}"', fontsize=12, va='center')
            ax.text(0.72, y, f"{sentiment}", fontsize=12, fontweight='bold', color=color, va='center')
            ax.text(0.92, y, f"{conf:.0%}", fontsize=11, va='center')

        # Arrow
        ax.annotate('', xy=(0.68, 0.55), xytext=(0.68, 0.75),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        ax.text(0.68, 0.65, "Model", ha='center', fontsize=10)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "sentiment_example.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Created: {OUTPUT_DIR / 'sentiment_example.png'}")

    except Exception as e:
        print(f"Could not generate sentiment example: {e}")


def generate_translation_example():
    """Generate machine translation example."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        examples = [
            ("Hello, how are you?", "Bonjour, comment allez-vous?", "English -> French"),
            ("The weather is beautiful today", "आज मौसम बहुत सुंदर है", "English -> Hindi"),
            ("I love learning new things", "Me encanta aprender cosas nuevas", "English -> Spanish"),
        ]

        ax.text(0.5, 0.92, "Machine Translation (Seq2Seq)", ha='center', fontsize=18, fontweight='bold')

        for i, (source, target, lang) in enumerate(examples):
            y = 0.7 - i * 0.22
            ax.text(0.02, y, source, fontsize=11, va='center',
                   bbox=dict(boxstyle='round', facecolor='#E3F2FD'))
            ax.annotate('', xy=(0.52, y), xytext=(0.48, y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='#2196F3'))
            ax.text(0.55, y, target, fontsize=11, va='center',
                   bbox=dict(boxstyle='round', facecolor='#E8F5E9'))
            ax.text(0.97, y, lang, fontsize=9, va='center', ha='right', style='italic')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "translation_example.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Created: {OUTPUT_DIR / 'translation_example.png'}")

    except Exception as e:
        print(f"Could not generate translation example: {e}")


def main():
    """Generate all example images."""
    print("=" * 60)
    print("Generating ML Task Examples with Real Datasets")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Download sample images
    print("\n--- Downloading sample images ---")
    download_sample_images()

    # Basic matplotlib examples (always work)
    print("\n--- Basic visualizations ---")
    try:
        generate_classification_example()
        generate_segmentation_comparison()
        generate_iou_visualization()
        generate_bbox_formats()
        generate_ml_paradigms()
        generate_vision_hierarchy()
        generate_ner_example()
        generate_sentiment_example()
        generate_translation_example()
    except Exception as e:
        print(f"Error generating basic examples: {e}")

    # Real dataset examples
    print("\n--- Real dataset examples ---")
    try:
        generate_cifar10_example()
        generate_mnist_example()
    except Exception as e:
        print(f"Could not generate dataset examples: {e}")

    # YOLO-based examples
    print("\n--- YOLO-based examples ---")
    try:
        generate_coco_detection_example()
        generate_coco_segmentation_example()
        generate_pose_estimation_example()
    except Exception as e:
        print(f"Could not generate YOLO examples: {e}")
        generate_detection_placeholder()

    print()
    print("=" * 60)
    print("Done! Generated images are in:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)
    print()
    print("To use in slides, reference like:")
    print("  ![](examples/coco_detection_examples.png)")


if __name__ == "__main__":
    main()
