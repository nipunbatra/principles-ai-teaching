# Lecture Structure V2 - Comprehensive Plan

## Course: 2nd Semester BTech - 4 Lectures
**Goal:** Intuition-first, examples-rich, pique interest, show real systems

---

## Lecture 1: Demo Day (Videos & Live Demos)
**File:** video-demo-suggestions.md (already exists)
**Purpose:** Hook students with mind-blowing AI demos
**No slides needed** - it's a demo session

---

## Lecture 2: ML Tasks, Zoo & Classical ML (COMBINED)
**Merge:** ml-tasks-taxonomy.md + ml-tasks-zoo.md + NEW classical ML examples
**Duration:** ~60 mins

### Structure:
1. **Hook (5 min)**
   - "ML powers your daily life" table
   - The one big question: "What are you trying to predict?"

2. **The Three Learning Paradigms (10 min)**
   - Supervised: Has labels (X->Y)
   - Unsupervised: No labels (find patterns)
   - Self-supervised: Create own labels (GPT, BERT)
   - ![D2: learning_paradigms diagram]

3. **Classification Deep Dive (15 min)**
   - Real CIFAR-10 examples
   - Show actual decision tree visualization
   - Show actual logistic regression formula
   - Binary vs Multi-class
   - Softmax visualization

4. **Regression Deep Dive (10 min)**
   - Real house price example
   - Show y_hat = w0 + w1*x1 + w2*x2 + ...
   - Linear regression visualization

5. **Vision Hierarchy (15 min)**
   - Classification -> Detection -> Segmentation
   - Real COCO examples for each
   - Detection = Classification + Regression
   - Instance vs Semantic Segmentation

6. **The Zoo of Tasks (10 min)**
   - NLP: NER, Sentiment, Translation
   - Audio: Speech-to-Text, Text-to-Speech
   - Generative: Images, Text, Music
   - Multimodal: VQA, Image Captioning

7. **End with Neural Networks (5 min)**
   - "All these tasks use the same tool: Neural Networks"
   - Teaser for deep learning

### Key Additions Needed:
- [ ] Decision tree visualization with sklearn
- [ ] Linear regression formula + plot
- [ ] Logistic regression decision boundary
- [ ] Real code snippets showing fit/predict

---

## Lecture 3: Small Language Models (SLM)
**File:** next-token-prediction.md (enhanced)
**Duration:** ~60 mins

### Structure:
1. **Hook: The Autocomplete Game (5 min)**
   - "The cat sat on the ___"
   - Phone keyboard prediction
   - Google search suggestions

2. **Core Idea: Next Token Prediction (15 min)**
   - P(next_word | context)
   - Bigram -> N-gram -> Neural
   - Show actual probability tables

3. **Neural Language Models (10 min)**
   - Embeddings: Words as vectors
   - RNNs: Remember history
   - Problem: Long-range dependencies

4. **Attention & Transformers (15 min)**
   - The library analogy
   - Query, Key, Value
   - Multi-head attention
   - The Transformer block

5. **From GPT to ChatGPT (NEW - 15 min)**
   - **Pre-training:** Learn language from massive text (next token)
   - **Supervised Fine-Tuning (SFT):** Learn to be helpful (instruction following)
   - **RLHF/DPO:** Learn to be harmless (alignment)
   - Show the full pipeline diagram

6. **Practical: Temperature, Top-k, Top-p (5 min)**
   - Why does ChatGPT sometimes give different answers?
   - Sampling strategies

### Key Additions Needed:
- [ ] Full LLM training pipeline diagram (pretrain -> SFT -> RLHF)
- [ ] Comparison table: GPT-1/2/3/4
- [ ] RLHF explanation with reward model
- [ ] Constitutional AI / DPO mention

---

## Lecture 4: Object Detection
**File:** object-detection-basics.md (enhanced)
**Duration:** ~60 mins

### Structure:
1. **Hook: Self-Driving Cars (5 min)**
   - Real image with detections
   - Why classification is not enough

2. **Bounding Boxes (10 min)**
   - Different formats (corner, center, COCO)
   - Absolute vs Normalized
   - Real examples with code

3. **IoU: Measuring Correctness (10 min)**
   - Formula: Intersection / Union
   - Visual examples
   - Thresholds in practice

4. **NMS: Handling Duplicates (10 min)**
   - Before/After visualization
   - Step-by-step algorithm
   - Code implementation

5. **YOLO: Real-Time Detection (15 min)**
   - One-stage vs Two-stage
   - Grid-based detection
   - Anchor boxes
   - YOLO evolution (v5 -> v8)

6. **Training & Metrics (10 min)**
   - Loss function components
   - mAP calculation
   - Precision-Recall

### Already has real examples, just needs polish.

---

## Code Examples to Generate

### For Lecture 2 (ML Tasks):
```python
# Decision Tree Visualization
from sklearn.tree import DecisionTreeClassifier, plot_tree
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)
plot_tree(clf, feature_names=..., class_names=...)

# Linear Regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
print(f"y = {reg.intercept_:.2f} + {reg.coef_[0]:.2f}*x1 + ...")

# Logistic Regression Decision Boundary
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
# Plot decision boundary

# Clustering (K-Means)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
# Plot clusters
```

### For Lecture 3 (SLM):
```python
# Bigram language model
# P(next | prev) = count(prev, next) / count(prev)

# Simple neural language model
# Embedding -> LSTM -> Linear -> Softmax

# Temperature sampling
def sample_with_temperature(logits, temperature):
    scaled = logits / temperature
    probs = softmax(scaled)
    return np.random.choice(len(probs), p=probs)
```

---

## New D2 Diagrams Needed:

1. **llm_training_pipeline.d2** - Pretrain -> SFT -> RLHF -> Deployment
2. **decision_tree_example.d2** - Visual decision tree for iris/mushroom
3. **linear_regression.d2** - y = mx + b with data points
4. **clustering_example.d2** - Before/after clustering
5. **rlhf_pipeline.d2** - Reward model + PPO

---

## Python Scripts to Create:

1. **generate_classical_ml_examples.py**
   - Decision tree on Iris
   - Linear regression on housing
   - Logistic regression on digits
   - K-means on synthetic data

2. **generate_llm_examples.py**
   - Bigram probability table
   - Attention heatmap
   - Temperature comparison
   - Token probability distribution

---

## Summary of Changes:

| Lecture | Current State | Needed Changes |
|---------|--------------|----------------|
| L1: Videos | Good | None |
| L2: ML Tasks | ASCII-heavy | Add classical ML examples, merge zoo |
| L3: SLM | Good foundation | Add pretrain/SFT/RLHF pipeline |
| L4: OD | Good | Real examples added, polish |

---

## Next Steps:

1. Create generate_classical_ml_examples.py
2. Create D2 diagrams for LLM pipeline
3. Merge ml-tasks-taxonomy + ml-tasks-zoo
4. Add RLHF section to next-token-prediction
5. Rebuild all Marp slides
