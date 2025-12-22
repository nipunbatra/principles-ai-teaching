---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 24px;
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
    padding: 15px;
    font-size: 0.85em;
    line-height: 1.4;
    overflow: hidden;
  }
  .example {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-left: 4px solid #06A77D;
    padding: 12px 15px;
    margin: 10px 0;
    border-radius: 0 8px 8px 0;
  }
  .insight {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 12px 15px;
    margin: 10px 0;
    border-radius: 0 8px 8px 0;
  }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 25px; }
  table { font-size: 0.9em; width: 100%; }
  th { background: #2E86AB; color: white; padding: 8px; }
  td { padding: 8px; border-bottom: 1px solid #dee2e6; }
---

# The Machine Learning Taxonomy
## Organizing 40+ Tasks by their Mathematical Roots

**Nipun Batra** · IIT Gandhinagar

---

# The Big Insight

Every ML task boils down to **one question**:

```
What are you predicting?
```

<div class="columns">
<div>

**Predicting a Category?**
→ Classification

**Predicting a Number?**
→ Regression

</div>
<div>

**Predicting a Sequence?**
→ Seq2Seq

**Predicting a Distribution?**
→ Generative

</div>
</div>

<div class="insight">
Once you know the "output type", you know which family the task belongs to!
</div>

---

# Section 1: Classification
## "Which Bucket Does This Belong To?"

---

# Classification: The Core Idea

```
                    ┌─────────────┐
   Input            │   Model     │         Output
  ───────────────►  │   f(x)      │  ──────────────►
  (Image, Text,     │             │   One of K classes
   Audio, etc.)     └─────────────┘
```

<div class="example">

**Example 1: Email Spam Detection**
```
Input:  "You won $1,000,000! Click here NOW!!!"
Output: SPAM (class 1 of 2)
```

**Example 2: Handwritten Digit**
```
Input:  [28x28 pixel image of "7"]
Output: "7" (class 7 of 10)
```

</div>

---

# Classification: Real-World Examples

| Task | Input | Output | # Classes |
|------|-------|--------|-----------|
| Cat vs Dog | Photo | "cat" or "dog" | 2 |
| ImageNet | Photo | Object name | 1000 |
| Sentiment | Movie review | Positive/Negative | 2-5 |
| Medical Diagnosis | X-ray | Disease type | varies |

<div class="insight">
Binary (2 classes) vs Multi-class (K classes) — same algorithm, different output layer!
</div>

---

# Classification: The Math

```
Input x ──► Neural Network ──► Softmax ──► Probabilities
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │ Cat:   0.85     │
                                    │ Dog:   0.10     │
                                    │ Bird:  0.05     │
                                    └─────────────────┘
                                              │
                                              ▼
                                        Pick highest
                                              │
                                              ▼
                                           "Cat"
```

The model outputs **probabilities** for each class, then picks the highest.

---

# Section 2: Regression
## "How Much? How Many?"

---

# Regression: The Core Idea

Instead of discrete classes, we predict a **continuous number**.

```
                    ┌─────────────┐
   Input            │   Model     │         Output
  ───────────────►  │   f(x)      │  ──────────────►
  (Features)        │             │   A real number
                    └─────────────┘
```

<div class="example">

**Example: House Price**
```
Input:  [3 beds, 2 baths, 1500 sqft, good location]
Output: $425,000
```

**Example: Age Estimation**
```
Input:  [Face photo]
Output: 27.3 years
```

</div>

---

# Regression: Real-World Examples

| Task | Input | Output | Unit |
|------|-------|--------|------|
| House Price | Features | $425,000 | Dollars |
| Temperature | Historical data | 32.5°C | Celsius |
| Stock Price | Market data | $147.23 | Dollars |
| Age from Face | Photo | 27.3 | Years |
| Bounding Box | Image region | (x, y, w, h) | Pixels |

<div class="insight">
Bounding box prediction is just **4 regression problems** solved together!
</div>

---

# Classification vs Regression: Side by Side

```
┌──────────────────────────────────────────────────────────────────┐
│                      CLASSIFICATION                              │
│                                                                  │
│   Input ───► Model ───► [0.1, 0.2, 0.7] ───► Class "C"          │
│                              ▲                                   │
│                         Probabilities                            │
│                         must sum to 1                            │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                        REGRESSION                                │
│                                                                  │
│   Input ───► Model ───► 425000.00 ───► $425,000                 │
│                              ▲                                   │
│                       Any real number                            │
│                       (no constraints)                           │
└──────────────────────────────────────────────────────────────────┘
```

---

# Section 3: Vision Hierarchy
## From Labels to Pixels

---

# The Computer Vision Ladder

```
Level 1: CLASSIFICATION          "There is a cat in this image"
         ────────────────────────────────────────────────────
              One label for the whole image

Level 2: DETECTION               "Cat at position (50,30,200,180)"
         ────────────────────────────────────────────────────
              Label + Bounding box

Level 3: SEGMENTATION            "These exact pixels are cat"
         ────────────────────────────────────────────────────
              Label for EVERY pixel
```

<div class="insight">
Each level builds on the previous. More precision = More complexity.
</div>

---

# Level 1: Image Classification

```
┌────────────────────────┐
│                        │
│    🐱  (somewhere)     │     ───►    "Cat"
│                        │
└────────────────────────┘
      Input Image              Single Label
```

**Use Cases:**
- Photo organization (Google Photos)
- Medical imaging (Is this cancerous?)
- Quality control (Defective or OK?)

---

# Level 2: Object Detection

```
┌────────────────────────┐
│    ┌──────┐            │
│    │ 🐱   │            │     ───►    "Cat" at (10,15,80,90)
│    └──────┘            │             "Dog" at (120,40,90,85)
│          ┌──────┐      │
│          │ 🐕   │      │
│          └──────┘      │
└────────────────────────┘
      Input Image              Labels + Bounding Boxes
```

**Detection = Classification + Regression (for box coordinates)**

---

# Level 3: Semantic Segmentation

```
┌────────────────────────┐        ┌────────────────────────┐
│░░░░░░░░░░░░░░░░░░░░░░░░│        │ SSSSSSSSSSSSSSSSSSSSSS │  S = Sky
│░░░░░░░░░░░░░░░░░░░░░░░░│        │ SSSSSSSSSSSSSSSSSSSSSS │  T = Tree
│░░░░TTTTTTTT░░░░░░░░░░░░│   ►    │ SSSSTTTTTTTTSSSSSSSSSS │  R = Road
│░░░░TTTTTTTT░░░░░░░░░░░░│        │ SSSSTTTTTTTTSSSSSSSSSS │  C = Car
│RRRRRRRRRRRRRRRRRRRRCCCC│        │ RRRRRRRRRRRRRRRRRRRRCCC │
│RRRRRRRRRRRRRRRRRRRRRRRR│        │ RRRRRRRRRRRRRRRRRRRRRR │
└────────────────────────┘        └────────────────────────┘
      Input Image                    Pixel-wise Labels
```

**Every pixel gets a class label!**

---

# Instance vs Semantic Segmentation

```
SEMANTIC SEGMENTATION:               INSTANCE SEGMENTATION:
┌────────────────────┐               ┌────────────────────┐
│                    │               │                    │
│   CCCCC    CCCCC   │               │   111111   222222  │
│   CCCCC    CCCCC   │               │   111111   222222  │
│                    │               │                    │
└────────────────────┘               └────────────────────┘
   Both are "Car"                    Car #1 vs Car #2

Semantic: "What class is each pixel?"
Instance: "What class AND which object?"
```

<div class="insight">
Self-driving cars need Instance Segmentation — they must track individual vehicles!
</div>

---

# Section 4: Sequence Tasks
## When Order Matters

---

# Sequence-to-Sequence (Seq2Seq)

```
       Input Sequence                    Output Sequence
┌───┬───┬───┬───┬───┐              ┌───┬───┬───┬───┐
│ H │ e │ l │ l │ o │    ────►     │ 你 │ 好 │   │   │
└───┴───┴───┴───┴───┘              └───┴───┴───┴───┘
      "Hello"                           "Ni Hao"
      (English)                         (Chinese)
```

**Key insight:** Input and output can have **different lengths**!

---

# Seq2Seq Examples

| Task | Input | Output |
|------|-------|--------|
| Translation | "Hello" (EN) | "Bonjour" (FR) |
| Summarization | Long article | Short summary |
| Speech-to-Text | Audio waveform | Text transcript |
| Text-to-Speech | Text | Audio waveform |
| Chatbot | Question | Answer |

<div class="example">

**Translation:**
```
Input:  "The cat sat on the mat"
Output: "Le chat s'est assis sur le tapis"
```

</div>

---

# Token-Level Classification (Tagging)

Sometimes we classify **each element** in the sequence:

```
Input:    "Sundar  Pichai   visited  New    York   yesterday"
           │       │        │        │      │      │
           ▼       ▼        ▼        ▼      ▼      ▼
Output:   PER     PER       O       LOC    LOC     O

PER = Person, LOC = Location, O = Other
```

<div class="insight">
Named Entity Recognition (NER) is like "semantic segmentation for text"!
</div>

---

# Section 5: Unsupervised Learning
## Finding Patterns Without Labels

---

# The Unsupervised Setting

```
SUPERVISED:                        UNSUPERVISED:
┌─────────────────────────┐        ┌─────────────────────────┐
│ Data: X                 │        │ Data: X                 │
│ Labels: Y               │        │ Labels: ???             │
│                         │        │                         │
│ Learn: f(X) → Y         │        │ Find: patterns in X     │
└─────────────────────────┘        └─────────────────────────┘
```

**No one tells the model what to look for — it discovers structure!**

---

# Clustering

Group similar items together **without predefined categories**.

```
Before Clustering:               After Clustering:
        •    •                       ○    ○
    •      •   •                 ○      ○   ○
      •  •                         ○  ○

        ▲  ▲                         △  △
    ▲        ▲                   △        △
      ▲    ▲                       △    △

      ■  ■                           □  □
        ■    ■                         □    □
      ■                              □
```

**Example:** Customer segmentation — find groups of similar shoppers.

---

# Dimensionality Reduction

Compress data while preserving structure.

```
1000 Dimensions                    2 Dimensions
     (Hard to visualize)               (Easy to plot!)
           │                               │
           │                               │
           ▼                               ▼
    ┌─────────────┐                    •  •
    │ 0.23, 0.11, │                   •    ••
    │ 0.87, 0.45, │    ─────────►        •
    │ 0.32, ...   │      PCA/t-SNE    ▲  ▲
    │ (1000 nums) │                  ▲  ▲ ▲
    └─────────────┘                    ■ ■
                                        ■
```

**Use case:** Visualizing word embeddings, gene expression data.

---

# Anomaly Detection

Find the **weird ones**.

```
Normal Data Points:          Anomaly:

    •  •  •  •  •                           ★ ← ALERT!
    •  •  •  •  •
    •  •  •  •  •
    •  •  •  •  •
```

**Use cases:**
- Credit card fraud detection
- Network intrusion detection
- Manufacturing defect detection

---

# Section 6: Generative Models
## Creating New Data

---

# Generative vs Discriminative

```
DISCRIMINATIVE (Classification):
┌─────────┐
│  Image  │ ────► Model ────► "Cat" or "Dog"
└─────────┘
   Given X, predict Y

GENERATIVE:
┌─────────┐
│  Noise  │ ────► Model ────► [New realistic image]
│ or Text │
└─────────┘
   Create new X from scratch
```

---

# Generative Task Examples

```
┌─────────────────────────────────────────────────────────────────┐
│ TEXT-TO-IMAGE                                                   │
│                                                                 │
│ "A cat wearing                    ┌────────────┐               │
│  a tiny hat,        ────────►     │  🐱 + 🎩   │               │
│  oil painting"                    └────────────┘               │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ IMAGE INPAINTING                                                │
│                                                                 │
│ ┌──────────┐                      ┌──────────┐                 │
│ │ 🏔️  ??? │        ────────►     │ 🏔️  🌅  │                 │
│ └──────────┘                      └──────────┘                 │
│   Missing part                      Filled in                   │
├─────────────────────────────────────────────────────────────────┤
│ TEXT GENERATION (LLMs)                                          │
│                                                                 │
│ "Once upon a"       ────────►     "Once upon a time, there     │
│                                    lived a dragon..."           │
└─────────────────────────────────────────────────────────────────┘
```

---

# Section 7: Complex & Multimodal
## Combining Everything

---

# Multimodal Tasks

These tasks combine **multiple input/output types**:

```
┌─────────────────────────────────────────────────────────────┐
│ VISUAL QUESTION ANSWERING (VQA)                             │
│                                                             │
│   Image: [Photo of red car]                                 │
│   Question: "What color is the car?"    ────►   "Red"       │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ IMAGE CAPTIONING                                            │
│                                                             │
│   Image: [Dog running on beach]                             │
│                                         ────►   "A dog      │
│                                                  running    │
│                                                  on a       │
│                                                  sandy      │
│                                                  beach"     │
└─────────────────────────────────────────────────────────────┘
```

---

# Reinforcement Learning

A different paradigm: **Learning through interaction**.

```
                    ┌─────────────────┐
                    │   Environment   │
                    │   (Game/World)  │
                    └────────┬────────┘
                             │
              State ◄────────┴────────► Reward
                │                         ▲
                ▼                         │
         ┌─────────────┐                  │
         │    Agent    │ ─── Action ──────┘
         │   (Model)   │
         └─────────────┘
```

**Goal:** Maximize total reward over time.

**Examples:** Game playing (Chess, Go), Robot control, Trading bots

---

# Summary: The ML Family Tree

```
                           Machine Learning
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
      Supervised            Unsupervised           Reinforcement
           │                      │                      │
     ┌─────┴─────┐          ┌─────┴─────┐          State → Action
     │           │          │           │
Classification Regression Clustering  Dim. Red.
     │           │
  ┌──┴──┐     ┌──┴──┐
  │     │     │     │
Image  Text  Price  Box
Class  Class Pred.  Pred.
```

---

# Key Takeaways

1. **Classification** → Predict a category (discrete)
2. **Regression** → Predict a number (continuous)
3. **Detection** → Classification + Box Regression
4. **Segmentation** → Classification for every pixel
5. **Seq2Seq** → Sequence in, sequence out
6. **Unsupervised** → Find patterns without labels
7. **Generative** → Create new data

<div class="insight">
Understanding the output type tells you which family of techniques to use!
</div>

---

# Thank You!

**"All models are wrong, but some are useful."** — George Box

## Questions?
