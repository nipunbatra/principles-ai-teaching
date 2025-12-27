# Lecture Improvement Plan

## Course Overview: 4 Lectures for 2nd Semester BTech

**Goal:** Intuition-first, intro-level, examples-rich content that piques interest in AI/ML.

---

## Lecture Structure

### Lecture 1: Videos & Demos (Show, Don't Tell)
- **Purpose:** Hook students with exciting AI applications
- **Content:** Video demonstrations of AI in action
- **File:** Already exists in `video-demo-suggestions.md`
- **No changes needed** - this is a demo/video lecture

### Lecture 2: ML Tasks & The Zoo (Combined)
- **Files:** `ml-tasks/ml-tasks-taxonomy.md` + `ml-tasks/ml-tasks-zoo.md`
- **Purpose:** Supervised vs Unsupervised, Classification/Regression/Detection/Segmentation
- **Ends with:** Neural Networks and Deep Learning intro

### Lecture 3: SLM (Small Language Models)
- **File:** `next-token-prediction/next-token-prediction.md`
- **Purpose:** Next token prediction, pretraining, SFT, alignment
- **Already well-structured**

### Lecture 4: Object Detection
- **File:** `object-detection/object-detection-basics.md`
- **Purpose:** Deep dive into detection as a practical application

---

## Improvements Needed

### Category A: ASCII to D2 Diagrams

The following ASCII diagrams should be converted to D2 for professional rendering:

#### ml-tasks-taxonomy.md
1. "YOUR DAILY LIFE" box (lines 72-87) - Convert to styled table/diagram
2. "THE BIG INSIGHT" centered box (lines 97-103)
3. "THE MASTER TAXONOMY" tree (lines 138-158) - Already have D2 version, reference it
4. Classification examples box (lines 171-183)
5. Email spam detection training/inference (lines 201-226)
6. Binary vs Multi-class boxes (lines 237-275)
7. Softmax flow diagram (lines 317-338)
8. Regression examples box (lines 369-383)
9. House price prediction (lines 399-418)
10. Bounding box detection (lines 425-444)
11. Classification vs Regression comparison (lines 450-472)
12. Vision hierarchy examples (lines 521-637)
13. Sequence importance examples (lines 649-669)
14. Supervised vs Unsupervised comparison (lines 733-748)
15. Clustering visualization (lines 753-776)
16. Dimensionality reduction (lines 783-805)
17. Anomaly detection (lines 812-832)
18. Generative vs Discriminative (lines 843-864)
19. Generative AI revolution examples (lines 870-896)
20. Multimodal examples (lines 929-941)
21. VQA example (lines 952-969)
22. RL examples (lines 986-1006)
23. Decision flowchart (lines 1018-1044)

#### ml-tasks-zoo.md
1. ML Task Safari Map (lines 83-105)
2. Universal ML Recipe (lines 113-131)
3. ML Tasks by Input/Output (lines 163-186)
4. Vision Task Hierarchy (lines 199-222)
5. Same Image Different Tasks (lines 229-247)
6. YOLO grid diagram (lines 472-493)
7. NER example (lines 1022-1038)

#### next-token-prediction.md
1. Journey to GPT box (lines 76-103) - Already have D2
2. Core problem box (lines 122-135)
3. You already know this examples (lines 145-171)
4. Mathematical view probabilities (lines 208-224)
5. The One Algorithm (lines 250-268)
6. Bigram counting table (lines 378-396)
7. Generation demo (lines 430-457)
8. Context problem examples (lines 507-528)
9. RNN passing the baton (lines 1019-1041)
10. Vanishing gradient (lines 1072-1092)
11. LSTM memory cell (lines 1101-1119)
12. Library analogy for attention (lines 1179-1202)
13. Attention scores visualization (lines 1243-1259)
14. Transformer block (lines 1350-1378) - Already have D2
15. Stacking transformers (lines 1384-1409)
16. Toy model vs ChatGPT comparison (lines 1419-1436)
17. Tokenization examples (lines 1450-1519)
18. Temperature comparison (lines 1559-1601)
19. Sampling methods (lines 1608-1632)
20. Pre-training to fine-tuning (lines 1682-1706)
21. 5 Big Ideas summary (lines 1739-1759)

#### object-detection-basics.md
1. Why Object Detection driving scene (lines 76-94)
2. Learning outline (lines 102-124)
3. Vision task hierarchy (lines 135-159)
4. Classification vs Detection comparison (lines 166-184)
5. Real-world applications (lines 214-235)
6. Instance segmentation comparison (lines 240-258)
7. Bounding box coordinate system (lines 272-295)
8. Three main formats (lines 299-350)
9. Format conversion cheat sheet (lines 355-379)
10. Absolute vs Normalized (lines 385-408)
11. IoU formula (lines 467-491)
12. IoU visual examples (lines 497-515)
13. IoU threshold guide (lines 551-567)
14. NMS problem (lines 631-654)
15. NMS step-by-step (lines 712-737)
16. Detector families (lines 836-861)
17. YOLO vs traditional (lines 868-887)
18. YOLO grid division (lines 892-912)
19. Cell predictions (lines 918-941)
20. YOLO architecture (lines 969-996)
21. Anchor boxes (lines 1002-1027, 1036-1058)
22. YOLO evolution (lines 1064-1087)
23. Multi-scale features (lines 1093-1117)
24. Model variants table (lines 1124-1145)
25. Loss function (lines 1156-1184)
26. Box loss (lines 1190-1210)
27. Precision/Recall (lines 1217-1244)
28. P-R tradeoff (lines 1250-1278)
29. AP curve (lines 1284-1306)
30. mAP calculation (lines 1312-1365)
31. Augmentation for detection (lines 1372-1398)
32. Mosaic augmentation (lines 1404-1425)
33. Training tips (lines 1431-1458)
34. Benchmarks table (lines 1465-1492)
35. SOTA results (lines 1498-1515)
36. Complete pipeline (lines 1521-1558)
37. Summary (lines 1637-1664)
38. Pitfalls (lines 1735-1760)
39. IoU loss variants (lines 1767-1792)

### Category B: ASCII to Markdown Tables

Many ASCII boxes should become proper markdown tables for better rendering:

1. "YOUR DAILY LIFE" examples -> styled list or table
2. Classification examples -> markdown table
3. Real-world examples tables -> proper markdown
4. Comparison tables (supervised vs unsupervised, etc.)

### Category C: Real Data Examples with Python Code

Replace ASCII art with actual images from datasets:

#### Object Detection (Priority - needs real examples)
1. **COCO dataset examples** for detection, segmentation
   - Show actual images with bounding boxes
   - Use supervision library for visualization
   - Create `generate_realistic_examples.py` script

2. **Segmentation examples**
   - Semantic: ADE20K or Cityscapes
   - Instance: COCO with masks
   - Panoptic: Same datasets

3. **Pose estimation**
   - COCO keypoints visualization

#### ML Tasks Zoo (needs real examples for each task)
1. **Classification**: Show actual ImageNet/CIFAR images
2. **Detection**: COCO images with boxes
3. **Segmentation**: Cityscapes driving scenes
4. **Face Recognition**: Example face embeddings visualization
5. **Depth Estimation**: Show RGB -> depth pairs

### Category D: Code-Generated Figures

Create Python scripts to generate actual visualizations:

1. **t-SNE/UMAP word embeddings** - real word2vec visualization
2. **Attention heatmaps** - actual attention weights
3. **Precision-Recall curves** - real mAP calculations
4. **IoU visualization** - animated or static examples
5. **NMS step-by-step** - actual boxes being suppressed
6. **YOLO grid overlay** - real image with grid

---

## Implementation Priority

### Phase 1: Essential D2 Diagrams (High Impact)
Create/improve these D2 diagrams:
- [x] ML Family Tree (exists)
- [x] Classification Pipeline (exists)
- [x] Vision Tasks Hierarchy (exists)
- [ ] Daily Life ML Applications (new)
- [ ] Learning Types Comparison (supervised/unsupervised/self-supervised)
- [ ] Detection Pipeline with NMS
- [ ] YOLO Grid Concept
- [ ] Transformer Full Pipeline

### Phase 2: Real Dataset Examples
Create scripts in each folder:
- [ ] `ml-tasks/generate_realistic_examples.py` - CV task examples
- [ ] `object-detection/generate_realistic_examples.py` - Detection examples (exists, enhance)
- [ ] `next-token-prediction/generate_ntp_examples.py` - Token prediction visualization

### Phase 3: Convert Tables
- [ ] Convert ASCII comparison boxes to markdown tables
- [ ] Add proper styling in Marp

### Phase 4: Polish
- [ ] Rebuild all Marp presentations
- [ ] Generate PDFs
- [ ] Test in classroom setting

---

## File Changes Summary

### New Files to Create
1. `ml-tasks/diagrams/daily_life_ml.d2`
2. `ml-tasks/diagrams/learning_types.d2`
3. `ml-tasks/generate_realistic_examples.py`
4. `object-detection/diagrams/nms_process.d2`
5. `object-detection/diagrams/yolo_inference.d2`

### Files to Modify
1. `ml-tasks/ml-tasks-taxonomy.md` - Replace ASCII with D2/SVG references
2. `ml-tasks/ml-tasks-zoo.md` - Add real dataset examples
3. `next-token-prediction/next-token-prediction.md` - Polish ASCII
4. `object-detection/object-detection-basics.md` - Add COCO examples

### Existing D2 Files (28 total)
- ml-tasks/diagrams/: 7 files
- next-token-prediction/diagrams/: 9 files
- object-detection/diagrams/: 12 files

---

## Technical Stack

- **Slides**: Marp (markdown to presentation)
- **Diagrams**: D2 with Elk layout, theme 200
- **Python**: matplotlib, supervision, ultralytics, huggingface datasets
- **Output**: SVG (for slides), PNG (for docs), PDF (for print)

---

## Notes

1. **ASCII diagrams serve a purpose** - In a terminal/code context, ASCII is actually more appropriate. But in slides, professional diagrams are better.

2. **Some ASCII is fine to keep** - Simple code examples, terminal output simulations, etc.

3. **Real data is king** - For 2nd year BTech students, seeing real COCO images with detections is much more impactful than ASCII boxes saying "[Dog]".

4. **Balance theory and practice** - The lectures are already theory-heavy. Adding real examples makes them practical.
