#!/usr/bin/env python3
"""
Generate realistic object detection examples using:
- supervision library for annotations
- ultralytics YOLO for detection
- COCO sample images

Output: PNG images showing detection, segmentation, IoU examples, NMS examples
"""

import os
from pathlib import Path

# Install dependencies if needed
try:
    import supervision as sv
    from ultralytics import YOLO
    import numpy as np
    from PIL import Image
    import requests
    import cv2
except ImportError:
    print("Installing required packages...")
    os.system("pip install supervision ultralytics pillow requests opencv-python")
    import supervision as sv
    from ultralytics import YOLO
    import numpy as np
    from PIL import Image
    import requests
    import cv2

# Output directory
OUTPUT_DIR = Path(__file__).parent / "diagrams" / "realistic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sample COCO images (public domain / CC licensed)
SAMPLE_IMAGES = {
    "street": "https://ultralytics.com/images/bus.jpg",
    "dogs": "https://ultralytics.com/images/zidane.jpg",
}


def download_image(url: str, name: str) -> Path:
    """Download image if not cached."""
    cache_dir = OUTPUT_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)
    filepath = cache_dir / f"{name}.jpg"

    if not filepath.exists():
        print(f"Downloading {name}...")
        response = requests.get(url)
        filepath.write_bytes(response.content)

    return filepath


def generate_detection_example():
    """Generate a clean object detection example."""
    print("Generating detection example...")

    # Load model
    model = YOLO("yolov8n.pt")

    # Download and load image
    img_path = download_image(SAMPLE_IMAGES["street"], "street")

    # Run detection
    results = model(str(img_path))[0]

    # Use supervision for beautiful annotations
    detections = sv.Detections.from_ultralytics(results)

    # Create annotators
    box_annotator = sv.BoxAnnotator(thickness=3)
    label_annotator = sv.LabelAnnotator(text_scale=0.8, text_thickness=2)

    # Load image
    image = cv2.imread(str(img_path))

    # Annotate
    labels = [
        f"{results.names[class_id]} {conf:.2f}"
        for class_id, conf in zip(detections.class_id, detections.confidence)
    ]

    annotated = box_annotator.annotate(image.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections, labels)

    # Save
    output_path = OUTPUT_DIR / "detection_example.png"
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved: {output_path}")

    return output_path


def generate_segmentation_example():
    """Generate instance segmentation example."""
    print("Generating segmentation example...")

    # Load segmentation model
    model = YOLO("yolov8n-seg.pt")

    # Download and load image
    img_path = download_image(SAMPLE_IMAGES["street"], "street")

    # Run segmentation
    results = model(str(img_path))[0]

    # Use supervision
    detections = sv.Detections.from_ultralytics(results)

    # Create mask annotator
    mask_annotator = sv.MaskAnnotator(opacity=0.5)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.6)

    # Load image
    image = cv2.imread(str(img_path))

    # Annotate with masks
    labels = [
        f"{results.names[class_id]}"
        for class_id in detections.class_id
    ]

    annotated = mask_annotator.annotate(image.copy(), detections)
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels)

    # Save
    output_path = OUTPUT_DIR / "segmentation_example.png"
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved: {output_path}")

    return output_path


def generate_iou_example():
    """Generate IoU visualization with two overlapping boxes."""
    print("Generating IoU example...")

    # Create blank image
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255

    # Define ground truth box (blue)
    gt_box = np.array([[100, 100, 300, 300]])
    # Define predicted box (green) - overlapping
    pred_box = np.array([[180, 120, 380, 320]])

    # Calculate IoU
    x1 = max(gt_box[0, 0], pred_box[0, 0])
    y1 = max(gt_box[0, 1], pred_box[0, 1])
    x2 = min(gt_box[0, 2], pred_box[0, 2])
    y2 = min(gt_box[0, 3], pred_box[0, 3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    gt_area = (gt_box[0, 2] - gt_box[0, 0]) * (gt_box[0, 3] - gt_box[0, 1])
    pred_area = (pred_box[0, 2] - pred_box[0, 0]) * (pred_box[0, 3] - pred_box[0, 1])
    union = gt_area + pred_area - intersection
    iou = intersection / union

    # Draw intersection area (yellow fill)
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
    cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

    # Draw ground truth box (blue)
    cv2.rectangle(image, (gt_box[0, 0], gt_box[0, 1]), (gt_box[0, 2], gt_box[0, 3]), (255, 100, 100), 4)

    # Draw predicted box (green)
    cv2.rectangle(image, (pred_box[0, 0], pred_box[0, 1]), (pred_box[0, 2], pred_box[0, 3]), (100, 200, 100), 4)

    # Add labels
    cv2.putText(image, "Ground Truth", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
    cv2.putText(image, "Prediction", (280, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 100), 2)
    cv2.putText(image, f"IoU = {iou:.2f}", (220, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(image, "Intersection", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 150), 2)

    # Save
    output_path = OUTPUT_DIR / "iou_example.png"
    cv2.imwrite(str(output_path), image)
    print(f"Saved: {output_path}")

    return output_path


def generate_nms_example():
    """Generate NMS before/after example."""
    print("Generating NMS example...")

    # Create side-by-side image
    image = np.ones((400, 800, 3), dtype=np.uint8) * 255

    # Left side: Before NMS (multiple overlapping boxes)
    boxes_before = [
        (50, 100, 200, 300, 0.95),  # Best box
        (60, 110, 210, 310, 0.88),
        (55, 95, 195, 295, 0.82),
        (70, 120, 220, 320, 0.75),
    ]

    # Right side: After NMS (single best box)
    boxes_after = [
        (450, 100, 600, 300, 0.95),
    ]

    # Draw before boxes
    for x1, y1, x2, y2, conf in boxes_before:
        alpha = conf
        color = (int(100 + 155 * alpha), int(200 * alpha), int(100 * alpha))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw after box
    for x1, y1, x2, y2, conf in boxes_after:
        cv2.rectangle(image, (x1, y1), (x2, y2), (100, 200, 100), 4)
        cv2.putText(image, f"{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 100), 2)

    # Add titles
    cv2.putText(image, "Before NMS", (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(image, "After NMS", (480, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # Add arrow
    cv2.arrowedLine(image, (320, 200), (380, 200), (0, 0, 0), 3, tipLength=0.3)
    cv2.putText(image, "NMS", (330, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Add explanation
    cv2.putText(image, "4 overlapping boxes", (60, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    cv2.putText(image, "1 best box kept", (470, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

    # Save
    output_path = OUTPUT_DIR / "nms_example.png"
    cv2.imwrite(str(output_path), image)
    print(f"Saved: {output_path}")

    return output_path


def generate_classification_vs_detection():
    """Generate side-by-side classification vs detection example."""
    print("Generating classification vs detection comparison...")

    # Load model and image
    model = YOLO("yolov8n.pt")
    img_path = download_image(SAMPLE_IMAGES["street"], "street")
    image = cv2.imread(str(img_path))

    # Get detections
    results = model(str(img_path))[0]
    detections = sv.Detections.from_ultralytics(results)

    # Create side-by-side
    h, w = image.shape[:2]
    combined = np.ones((h, w * 2 + 50, 3), dtype=np.uint8) * 255

    # Left: Classification style (just label)
    left_img = image.copy()
    # Add a single label at top
    classes_found = list(set([results.names[c] for c in detections.class_id]))
    label_text = ", ".join(classes_found[:3])
    cv2.rectangle(left_img, (10, 10), (w - 10, 60), (240, 240, 240), -1)
    cv2.putText(left_img, f"Classes: {label_text}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(left_img, "CLASSIFICATION", (w // 2 - 100, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    # Right: Detection style (boxes + labels)
    box_annotator = sv.BoxAnnotator(thickness=3)
    label_annotator = sv.LabelAnnotator(text_scale=0.6)
    labels = [f"{results.names[c]}" for c in detections.class_id]

    right_img = box_annotator.annotate(image.copy(), detections)
    right_img = label_annotator.annotate(right_img, detections, labels)
    cv2.putText(right_img, "DETECTION", (w // 2 - 70, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    # Combine
    combined[:, :w] = left_img
    combined[:, w + 50:] = right_img

    # Add arrow
    cv2.arrowedLine(combined, (w + 10, h // 2), (w + 40, h // 2), (0, 0, 0), 3, tipLength=0.4)

    # Save
    output_path = OUTPUT_DIR / "classification_vs_detection.png"
    cv2.imwrite(str(output_path), combined)
    print(f"Saved: {output_path}")

    return output_path


if __name__ == "__main__":
    print("Generating realistic object detection examples...\n")

    generate_detection_example()
    generate_segmentation_example()
    generate_iou_example()
    generate_nms_example()
    generate_classification_vs_detection()

    print(f"\nAll examples saved to: {OUTPUT_DIR}")
