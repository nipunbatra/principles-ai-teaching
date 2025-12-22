---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 23px;
    padding: 35px;
    color: #333;
  }
  h1 { color: #2E86AB; font-size: 1.7em; margin-bottom: 0.2em; }
  h2 { color: #06A77D; font-size: 1.1em; margin-top: 0; }
  h3 { color: #457B9D; font-size: 1.0em; }
  strong { color: #D62828; }
  code {
    background: #f4f4f4;
    color: #2E86AB;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Consolas', 'Monaco', monospace;
  }
  pre {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px;
    font-size: 0.78em;
    line-height: 1.3;
    overflow: hidden;
  }
  .example {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    border-left: 4px solid #06A77D;
    padding: 10px 12px;
    margin: 8px 0;
    border-radius: 0 6px 6px 0;
  }
  .insight {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 10px 12px;
    margin: 8px 0;
    border-radius: 0 6px 6px 0;
  }
  .warning {
    background: #ffebee;
    border-left: 4px solid #D62828;
    padding: 10px 12px;
    margin: 8px 0;
    border-radius: 0 6px 6px 0;
  }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  table { font-size: 0.85em; width: 100%; }
  th { background: #2E86AB; color: white; padding: 6px; }
  td { padding: 6px; border-bottom: 1px solid #dee2e6; }
---

# Object Detection Basics
## Deep Learning for Computer Vision

**Nipun Batra** · IIT Gandhinagar
*Inspired by Andrew Ng's teaching style*

---

# What You Will Learn Today

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Part 1: The Core Problem                                      │
│           Classification vs Detection vs Segmentation           │
│                                                                 │
│   Part 2: Bounding Boxes                                        │
│           How we represent object locations                     │
│                                                                 │
│   Part 3: IoU (Intersection over Union)                         │
│           How we measure detection quality                      │
│                                                                 │
│   Part 4: NMS (Non-Maximum Suppression)                         │
│           How we clean up duplicate detections                  │
│                                                                 │
│   Part 5: Architectures (YOLO, Faster R-CNN)                    │
│           How modern detectors work                             │
│                                                                 │
│   Part 6: Training & Metrics (mAP)                              │
│           How we train and evaluate detectors                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# Part 1: The Core Problem
## What IS Object Detection?

---

# Classification vs Detection

```
CLASSIFICATION:                    DETECTION:
"What is in this image?"           "What is here AND where?"

┌─────────────────────┐            ┌─────────────────────┐
│                     │            │  ┌────────┐         │
│     🐕              │            │  │ DOG    │         │
│                     │  ───►      │  │ 0.95   │         │
│                     │            │  └────────┘         │
└─────────────────────┘            └─────────────────────┘
        │                                  │
        ▼                                  ▼
   Output: "Dog"                  Output: "Dog" at (10,20,80,90)
   (one label)                    (label + bounding box)
```

<div class="insight">
Detection = Classification + Localization
</div>

---

# The Full Vision Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│ CLASSIFICATION                                                  │
│                                                                 │
│   Input: Image                Output: Single Label              │
│   ┌─────────────┐                                               │
│   │  🐕  🐈     │  ───────►   ["Dog", "Cat"]                   │
│   └─────────────┘                                               │
├─────────────────────────────────────────────────────────────────┤
│ DETECTION                                                       │
│                                                                 │
│   Input: Image                Output: Labels + Boxes            │
│   ┌─────────────┐             [("Dog", 10,20,80,90),           │
│   │ ┌──┐  ┌──┐  │  ───────►    ("Cat", 100,30,60,70)]          │
│   │ │🐕│  │🐈│  │                                               │
│   │ └──┘  └──┘  │                                               │
│   └─────────────┘                                               │
├─────────────────────────────────────────────────────────────────┤
│ SEGMENTATION                                                    │
│                                                                 │
│   Input: Image                Output: Pixel-wise Labels         │
│   ┌─────────────┐             ┌─────────────┐                   │
│   │  🐕  🐈     │  ───────►   │ DDDD CCCCC  │  D=Dog, C=Cat    │
│   └─────────────┘             │ DDDD CCCCC  │                   │
│                               └─────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

# Real-World Detection Examples

<div class="columns">
<div>

**Self-Driving Cars:**
```
┌─────────────────────────┐
│   🚗  🚶  🚦            │
│  [car] [person] [light] │
│   0.97   0.89    0.95   │
└─────────────────────────┘
Detect: cars, pedestrians,
traffic signs, lanes
```

**Retail:**
```
┌─────────────────────────┐
│  📦  📦  📦             │
│ [milk][bread][eggs]     │
└─────────────────────────┘
Detect: products on shelves
for inventory management
```

</div>
<div>

**Medical Imaging:**
```
┌─────────────────────────┐
│                         │
│      ○ [tumor 0.87]     │
│                         │
└─────────────────────────┘
Detect: tumors, lesions,
abnormalities in X-rays
```

**Security:**
```
┌─────────────────────────┐
│  👤  👤  👤  📦        │
│ [face] [face] [package] │
└─────────────────────────┘
Detect: faces, unattended
bags, suspicious activity
```

</div>
</div>

---

# Part 2: Bounding Boxes
## How We Represent Locations

---

# Bounding Box Basics

A bounding box is a **rectangle** that tightly contains an object.

```
Image Coordinate System:

    (0,0) ─────────────────────────────────► x (width)
      │
      │    ┌─────────────────────┐
      │    │                     │
      │    │     🐕              │ ← Object inside box
      │    │                     │
      │    └─────────────────────┘
      │
      ▼
    y (height)

Box is defined by 4 numbers: Where does it start? How big is it?
```

---

# Bounding Box Formats

<div class="warning">
Different datasets/frameworks use different formats!
</div>

```
FORMAT 1: (x, y, width, height)              FORMAT 2: (x1, y1, x2, y2)
"Top-left corner + size"                     "Two opposite corners"

   (x, y)                                       (x1, y1)
     ┌─────────────────┐                          ┌─────────────────┐
     │                 │                          │                 │
     │      🐕         │ height                   │      🐕         │
     │                 │                          │                 │
     └─────────────────┘                          └─────────────────┘
           width                                                  (x2, y2)

Example: (50, 100, 200, 150)                 Example: (50, 100, 250, 250)

FORMAT 3: (cx, cy, width, height)
"Center point + size" (used by YOLO)

              width
         ←───────────→
       ┌─────────────────┐  ↑
       │                 │  │
       │    ● (cx,cy)    │  │ height
       │                 │  │
       └─────────────────┘  ↓
```

---

# Normalized vs Absolute Coordinates

```
ABSOLUTE (Pixels):                    NORMALIZED (0-1 range):

Image: 640×480 pixels                 Image: Any size → values 0-1

Box: (100, 50, 200, 150)              Box: (0.156, 0.104, 0.312, 0.312)
     ↑    ↑   ↑    ↑                       ↑      ↑      ↑      ↑
     │    │   │    └─ height 150px         │      │      │      └─ h/H
     │    │   └─ width 200px               │      │      └─ w/W
     │    └─ y = 50px                      │      └─ y/H
     └─ x = 100px                          └─ x/W

Conversion:
x_norm = x_abs / image_width
y_norm = y_abs / image_height
```

<div class="insight">
Normalized coordinates are **resolution-independent** — the same box
works for any image size!
</div>

---

# Part 3: IoU
## How Good Is a Detection?

---

# The Problem: When Is a Box "Correct"?

```
Ground Truth (what a human labeled):    Model Prediction:

┌───────────────────────┐               ┌───────────────────────┐
│                       │               │   ┌────────────┐      │
│  ┌─────────────┐      │               │   │            │      │
│  │    🐕       │      │               │   │   🐕       │      │
│  │             │      │               │   │            │      │
│  └─────────────┘      │               │   └────────────┘      │
│                       │               │                       │
└───────────────────────┘               └───────────────────────┘

Is this prediction "correct"?

- Boxes aren't identical, but they overlap a lot
- We need a NUMBER to measure how good this is
- That number is IoU (Intersection over Union)
```

---

# IoU: The Formula

```
                    Area of Overlap
        IoU = ─────────────────────────
                 Area of Union

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Ground Truth        Prediction        Overlap (Intersection) │
│   ┌────────┐          ┌────────┐        ┌────────┐             │
│   │████████│    +     │▓▓▓▓▓▓▓▓│   =    │▒▒▒▒▒▒▒▒│             │
│   │████████│          │▓▓▓▓▓▓▓▓│        └────────┘             │
│   └────────┘          └────────┘                                │
│                                                                 │
│   Union = All area covered by EITHER box (not double-counted)  │
│                                                                 │
│   Union = Area(GT) + Area(Pred) - Intersection                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# IoU: Visual Examples

```
CASE 1: Perfect Match              CASE 2: Good Overlap
        IoU = 1.0                          IoU ≈ 0.7

    ┌───────────┐                     ┌───────────┐
    │███████████│ ← Boxes             │███████│▓▓▓│
    │███████████│   identical         │███████│▓▓▓│
    └───────────┘                     └───────┴───┘
                                            ↑
                                      Partial overlap

CASE 3: Poor Overlap               CASE 4: No Overlap
        IoU ≈ 0.2                          IoU = 0.0

    ┌───────┐                         ┌───────┐    ┌───────┐
    │███│▓▓▓│▓▓▓│                     │███████│    │▓▓▓▓▓▓▓│
    │███│▓▓▓│▓▓▓│                     │███████│    │▓▓▓▓▓▓▓│
    └───┴───┴───┘                     └───────┘    └───────┘
           ↑                                 ↑           ↑
     Small overlap                     Ground Truth  Prediction
```

---

# IoU Thresholds in Practice

```
┌───────────────┬─────────────────────────────────────────────────┐
│   IoU Value   │   Interpretation                                │
├───────────────┼─────────────────────────────────────────────────┤
│   1.0         │   Perfect match (never happens in practice)     │
│   0.75+       │   Excellent detection                           │
│   0.50        │   Standard threshold for "correct" (COCO)       │
│   0.25        │   Loose match (used in some old benchmarks)     │
│   0.0         │   No overlap at all                             │
└───────────────┴─────────────────────────────────────────────────┘

Common Rule:
    If IoU ≥ 0.5 → Detection is a TRUE POSITIVE (TP) ✓
    If IoU < 0.5 → Detection is a FALSE POSITIVE (FP) ✗
```

<div class="insight">
Different competitions use different thresholds:
- PASCAL VOC: IoU ≥ 0.5
- COCO: Multiple thresholds (0.5, 0.55, ..., 0.95)
</div>

---

# Part 4: Non-Maximum Suppression
## Cleaning Up Duplicate Detections

---

# The Problem: Too Many Boxes!

```
What the detector produces:              What we actually want:

┌───────────────────────────┐           ┌───────────────────────────┐
│ ┌──────────┐              │           │                           │
│ │┌─────────┴┐             │           │  ┌─────────────┐          │
│ ││┌─────────┴┐            │           │  │    🐕       │          │
│ │││   🐕     │ 0.95       │   ───►    │  │    0.95     │          │
│ ││└──────────┤ 0.90       │    NMS    │  └─────────────┘          │
│ │└───────────┤ 0.85       │           │                           │
│ └────────────┘ 0.80       │           │                           │
└───────────────────────────┘           └───────────────────────────┘

Detector found the SAME dog 4 times with slightly different boxes!
We want to keep only the BEST one.
```

---

# NMS Algorithm: Step by Step

```
STEP 1: Sort all boxes by confidence (highest first)
        ┌────────────────────────────────────────────┐
        │  Box A: 0.95, Box B: 0.90, Box C: 0.85     │
        └────────────────────────────────────────────┘

STEP 2: Pick the highest confidence box → KEEP IT
        ┌────────────────────────────────────────────┐
        │  KEEP: Box A (0.95)                        │
        │  Remaining: [Box B, Box C]                 │
        └────────────────────────────────────────────┘

STEP 3: Remove all remaining boxes that overlap
        too much with Box A (IoU > threshold)
        ┌────────────────────────────────────────────┐
        │  IoU(A, B) = 0.85 > 0.5 → REMOVE Box B     │
        │  IoU(A, C) = 0.70 > 0.5 → REMOVE Box C     │
        │  Remaining: []                             │
        └────────────────────────────────────────────┘

STEP 4: Repeat until no boxes remain

OUTPUT: [Box A] — just one clean detection!
```

---

# NMS: Visual Example

```
BEFORE NMS:                           AFTER NMS:

Person Detections:                    Final Detections:
┌───────────────────────────────┐    ┌───────────────────────────────┐
│ ┌────────┐    ┌────────┐      │    │                               │
│ │ Person │    │ Person │      │    │ ┌────────┐    ┌────────┐      │
│ │  0.95  │    │  0.93  │      │    │ │ Person │    │ Person │      │
│ ├────────┴──┐ ├────────┴──┐   │    │ │  0.95  │    │  0.93  │      │
│ │ Person    │ │ Person    │   │    │ └────────┘    └────────┘      │
│ │   0.88    │ │   0.85    │   │    │                               │
│ └───────────┘ └───────────┘   │    │  Two people → Two boxes!      │
│                               │    │                               │
│  4 overlapping boxes          │    │                               │
└───────────────────────────────┘    └───────────────────────────────┘

NMS keeps the best box per object, removes duplicates.
Boxes of DIFFERENT objects (low IoU) are both kept!
```

---

# NMS: The Python Pseudocode

```python
def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: List of (x1, y1, x2, y2)
    scores: Confidence for each box
    """
    # Sort by confidence (descending)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        # Pick the best box
        i = order[0]
        keep.append(i)

        # Compute IoU with remaining boxes
        ious = compute_iou(boxes[i], boxes[order[1:]])

        # Keep only boxes with IoU below threshold
        remaining = np.where(ious <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep
```

---

# Part 5: How Detectors Work
## Two Main Approaches

---

# One-Stage vs Two-Stage Detectors

```
┌─────────────────────────────────────────────────────────────────┐
│ TWO-STAGE (Faster R-CNN)                                        │
│                                                                 │
│   Image ──► Region Proposal ──► Classify & Refine ──► Output   │
│             Network (RPN)        each region                    │
│                                                                 │
│   Step 1: "Where might objects be?" (propose ~2000 regions)    │
│   Step 2: "What is each region?" (classify each one)           │
│                                                                 │
│   ✓ More accurate   ✗ Slower                                   │
├─────────────────────────────────────────────────────────────────┤
│ ONE-STAGE (YOLO, SSD)                                           │
│                                                                 │
│   Image ──► Single Network ──► All boxes + classes at once!    │
│                                                                 │
│   Look at image once, predict everything in one pass            │
│                                                                 │
│   ✓ Very fast (real-time)   ✗ Slightly less accurate          │
└─────────────────────────────────────────────────────────────────┘
```

---

# YOLO: You Only Look Once

**Core Idea:** Divide image into grid, predict boxes for each cell.

```
Input Image:              Grid (7×7):              Predictions per cell:
┌─────────────────┐      ┌───┬───┬───┬───┐       ┌─────────────────────┐
│                 │      │   │   │ ● │   │       │ For each grid cell: │
│      🐕         │  ──► ├───┼───┼───┼───┤  ──►  │ • 2 bounding boxes  │
│                 │      │   │   │   │   │       │ • Confidence scores │
│                 │      ├───┼───┼───┼───┤       │ • Class probs       │
└─────────────────┘      │   │   │   │   │       └─────────────────────┘
                         └───┴───┴───┴───┘
                               ▲
                         Dog's center falls
                         in this cell → this
                         cell predicts the dog!
```

Each cell is "responsible" for objects whose **center** falls inside it.

---

# YOLO: What Each Cell Predicts

```
Each grid cell outputs:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   BOX 1:  [x, y, w, h, confidence]    (5 numbers)              │
│   BOX 2:  [x, y, w, h, confidence]    (5 numbers)              │
│   CLASS:  [P(dog), P(cat), P(car), ...] (C numbers)            │
│                                                                 │
│   Total per cell: 2×5 + C = 10 + C numbers                     │
│   For 7×7 grid with 20 classes: 7 × 7 × (10 + 20) = 1470       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Confidence = P(object exists) × IoU(pred, truth)

If confidence is low → "nothing interesting here"
If confidence is high → "I found something!"
```

---

# YOLO Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       YOLO Architecture                         │
│                                                                 │
│   Input: 448×448×3 (RGB image)                                  │
│      │                                                          │
│      ▼                                                          │
│   ┌─────────────────────────────────────────┐                   │
│   │ BACKBONE (Feature Extractor)            │                   │
│   │ 24 Convolutional Layers                 │                   │
│   │ Extract patterns: edges → shapes → parts│                   │
│   └────────────────────┬────────────────────┘                   │
│                        │                                        │
│                        ▼                                        │
│   ┌─────────────────────────────────────────┐                   │
│   │ DETECTION HEAD                          │                   │
│   │ 2 Fully Connected Layers                │                   │
│   │ Output: 7×7×30 tensor                   │                   │
│   └────────────────────┬────────────────────┘                   │
│                        │                                        │
│                        ▼                                        │
│   Output: 7×7 grid, each cell predicts 2 boxes + 20 classes    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# Anchor Boxes: Better Shape Priors

**Problem:** A grid cell might contain multiple objects.
**Solution:** Use multiple "anchor boxes" of different shapes.

```
Without Anchors:                   With Anchors:

Cell predicts ONE box shape        Cell predicts MULTIPLE shapes
                                   matched to anchor templates
     ┌───┐
     │   │ ← Only one shape       ┌───┐ ├───────┐ ╭───────╮
     │   │                        │   │ │       │ │       │
     └───┘                        └───┘ ├───────┘ ╰───────╯
                                    ↑       ↑         ↑
                                  Tall   Wide     Square

Common anchor ratios: 1:1, 1:2, 2:1
Common scales: small, medium, large
```

<div class="insight">
Modern YOLO (v3+) uses 9 anchors: 3 scales × 3 aspect ratios!
</div>

---

# Part 6: Training & Evaluation
## How We Measure Success

---

# The Loss Function

Detection models optimize multiple objectives simultaneously:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Total Loss = λ₁ × Box Loss                                    │
│              + λ₂ × Objectness Loss                             │
│              + λ₃ × Classification Loss                         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   BOX LOSS:           "How accurate are the coordinates?"       │
│   (x, y, w, h)        MSE or IoU-based loss                     │
│                                                                 │
│   OBJECTNESS LOSS:    "Does this cell contain an object?"       │
│   (confidence)        Binary cross-entropy                      │
│                                                                 │
│   CLASSIFICATION:     "What class is this object?"              │
│   (class probs)       Cross-entropy over classes                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# Precision and Recall

```
PRECISION: "When I say 'dog', am I right?"

                True Positives
Precision = ─────────────────────────
            True Positives + False Positives

   TP: Correctly detected dogs
   FP: Non-dogs mistakenly called dogs

RECALL: "Did I find all the dogs?"

             True Positives
Recall = ─────────────────────────────
         True Positives + False Negatives

   TP: Correctly detected dogs
   FN: Dogs that I missed
```

<div class="insight">
High precision = Few false alarms
High recall = Few missed objects
We want BOTH to be high!
</div>

---

# Precision-Recall Tradeoff

```
Confidence Threshold: "Only report if confidence > threshold"

HIGH THRESHOLD (0.9):                LOW THRESHOLD (0.3):
┌───────────────────────────┐       ┌───────────────────────────┐
│                           │       │ ┌───┐┌───┐┌───┐┌───┐┌───┐ │
│       ┌───┐               │       │ │dog││dog││dog││???││???│ │
│       │dog│               │       │ │.95││.80││.60││.40││.35│ │
│       │.95│               │       │ └───┘└───┘└───┘└───┘└───┘ │
│       └───┘               │       │                           │
└───────────────────────────┘       └───────────────────────────┘
Precision: High (few FP)            Precision: Low (some FP)
Recall: Low (missed some)           Recall: High (found most)

                    Precision
                    ↑
               1.0  │●
                    │ ╲
                    │  ╲      ← PR Curve
                    │   ╲
                    │    ●
                    │     ╲
                    └──────────► Recall
                         1.0
```

---

# Mean Average Precision (mAP)

```
STEP 1: For each class, compute Precision-Recall curve

        Precision
           ↑
      1.0  │●──●
           │    ╲
           │     ●──●
           │         ╲
           │          ●──●
           └────────────────► Recall
                          1.0

STEP 2: Compute Area Under Curve (AP) for this class
        AP = Average precision across all recall levels

STEP 3: Average across all classes
        mAP = (AP_dog + AP_cat + AP_car + ...) / num_classes

mAP@0.5 = mAP computed with IoU threshold 0.5
mAP@0.5:0.95 = Average mAP across thresholds 0.5, 0.55, ..., 0.95
```

---

# mAP: A Concrete Example

```
Dataset: 100 dog images, model makes predictions

At confidence threshold = 0.9:
├─ Found: 30 dogs correctly (TP = 30)
├─ Missed: 70 dogs (FN = 70)
├─ False alarms: 2 (FP = 2)
├─ Precision = 30/(30+2) = 0.94
└─ Recall = 30/(30+70) = 0.30

At confidence threshold = 0.5:
├─ Found: 80 dogs correctly (TP = 80)
├─ Missed: 20 dogs (FN = 20)
├─ False alarms: 15 (FP = 15)
├─ Precision = 80/(80+15) = 0.84
└─ Recall = 80/(80+20) = 0.80

AP ≈ Area under the precision-recall curve created by
     varying the confidence threshold from 1.0 to 0.0
```

---

# Common Benchmarks & Scores

```
┌──────────────────┬────────────────────────────────────────────┐
│  Dataset         │  Details                                   │
├──────────────────┼────────────────────────────────────────────┤
│  PASCAL VOC      │  20 classes, ~10K images                   │
│                  │  Uses mAP@0.5                              │
├──────────────────┼────────────────────────────────────────────┤
│  MS COCO         │  80 classes, ~120K images                  │
│                  │  Uses mAP@0.5:0.95 (stricter)              │
├──────────────────┼────────────────────────────────────────────┤
│  ImageNet Det    │  200 classes, ~400K images                 │
│                  │  Large-scale benchmark                      │
└──────────────────┴────────────────────────────────────────────┘

Modern YOLO scores:
├─ YOLOv5s: ~36 mAP on COCO (fast, small)
├─ YOLOv5x: ~50 mAP on COCO (accurate, large)
└─ YOLOv8x: ~53 mAP on COCO (latest)
```

---

# Data Augmentation for Detection

```
IMPORTANT: When you transform the IMAGE, also transform the BOXES!

┌─────────────────────────────────────────────────────────────────┐
│ HORIZONTAL FLIP:                                                │
│                                                                 │
│   Original:              Flipped:                               │
│   ┌──────────────┐       ┌──────────────┐                       │
│   │ ┌───┐        │       │        ┌───┐ │                       │
│   │ │ 🐕│        │  ──►  │        │ 🐕│ │                       │
│   │ └───┘        │       │        └───┘ │                       │
│   └──────────────┘       └──────────────┘                       │
│   Box: (10, 20, 60, 80)  Box: (W-70, 20, W-10, 80)             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Other augmentations:                                            │
│  • Random crop (with box adjustment)                            │
│  • Color jitter (no box change needed)                          │
│  • Mosaic: Combine 4 images (complex box handling)              │
│  • Mixup: Blend two images                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

# Summary: The Detection Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    OBJECT DETECTION PIPELINE                    │
│                                                                 │
│   1. INPUT IMAGE                                                │
│      └─► Resize to fixed size (e.g., 640×640)                  │
│                                                                 │
│   2. BACKBONE (Feature Extraction)                              │
│      └─► CNN extracts visual features                          │
│                                                                 │
│   3. DETECTION HEAD                                             │
│      └─► Predict boxes + classes for each anchor               │
│                                                                 │
│   4. POST-PROCESSING                                            │
│      ├─► Filter by confidence threshold                        │
│      └─► Apply NMS to remove duplicates                        │
│                                                                 │
│   5. OUTPUT                                                     │
│      └─► List of (class, confidence, x1, y1, x2, y2)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# Key Takeaways

1. **Detection = Classification + Localization**
   Predict WHAT and WHERE

2. **Bounding Box formats vary**
   Always check: (x,y,w,h) vs (x1,y1,x2,y2) vs normalized

3. **IoU measures overlap quality**
   IoU ≥ 0.5 usually means "correct" detection

4. **NMS removes duplicate boxes**
   Keep best, remove overlapping

5. **YOLO is fast (one-stage)**
   Real-time detection on video

6. **mAP is the gold standard metric**
   Combines precision and recall across thresholds

---

# Getting Started: Try YOLO!

```python
# Install
pip install ultralytics

# Run in 3 lines!
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load pretrained model
results = model('your_image.jpg')  # Run detection
results[0].show()  # Display results

# Each detection has:
for box in results[0].boxes:
    print(f"Class: {box.cls}, Confidence: {box.conf}")
    print(f"Box: {box.xyxy}")  # x1, y1, x2, y2
```

---

# Thank You!

**"AI is the new electricity."** — Andrew Ng

The same ideas that power self-driving cars work for detecting
anything: faces, products, medical anomalies, defects...

## Questions?
