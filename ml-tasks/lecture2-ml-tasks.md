---
marp: true
theme: default
paginate: true
style: |
  /* ===== IITGN Modern Slide Theme ===== */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  :root {
    --primary: #1e3a5f;
    --primary-light: #2e5a8f;
    --accent: #e85a4f;
    --accent-soft: #ff8c7f;
    --success: #2a9d8f;
    --warning: #e9c46a;
    --text: #2d3748;
    --text-light: #4a5568;
    --bg-light: #f7fafc;
    --bg-card: #ffffff;
  }

  section {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 21px;
    padding: 0;
    color: var(--text);
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  }

  /* Modern title bar */
  section h1 {
    color: var(--primary);
    font-size: 1.7em;
    font-weight: 700;
    margin: 0;
    padding: 28px 45px 18px 45px;
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
    color: white;
    flex-shrink: 0;
    letter-spacing: -0.02em;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }

  section h2 {
    color: var(--accent);
    font-size: 1.0em;
    font-weight: 500;
    margin: 0;
    padding: 8px 45px 18px 45px;
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
    flex-shrink: 0;
  }

  section h1:has(+ h2) {
    padding-bottom: 8px;
  }

  section h3 {
    color: var(--primary);
    font-size: 1.05em;
    font-weight: 600;
    margin-top: 18px;
    padding: 0 45px;
  }

  /* Content area with better spacing */
  section > *:not(h1):not(h2):not(h3) {
    padding-left: 45px;
    padding-right: 45px;
  }

  section > p:first-of-type, section > ul:first-of-type,
  section > ol:first-of-type, section > table:first-of-type,
  section > div:first-of-type, section > pre:first-of-type {
    margin-top: 25px;
  }

  /* Typography */
  strong {
    color: var(--accent);
    font-weight: 600;
  }

  em {
    color: var(--text-light);
    font-style: italic;
  }

  /* Modern code styling */
  code {
    background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%);
    color: var(--primary);
    padding: 3px 8px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.9em;
  }

  pre {
    background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
    color: #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    font-size: 0.78em;
    line-height: 1.5;
    overflow: hidden;
    margin: 15px 45px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  }

  pre code {
    background: transparent;
    color: #e2e8f0;
    padding: 0;
  }

  /* Modern callout boxes */
  .example {
    background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
    border-left: 5px solid var(--success);
    padding: 16px 20px;
    margin: 15px 45px;
    border-radius: 0 12px 12px 0;
    box-shadow: 0 2px 4px rgba(42,157,143,0.1);
  }

  .insight {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border-left: 5px solid var(--warning);
    padding: 16px 20px;
    margin: 15px 45px;
    border-radius: 0 12px 12px 0;
    box-shadow: 0 2px 4px rgba(233,196,106,0.15);
  }

  .realworld {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-left: 5px solid #3b82f6;
    padding: 16px 20px;
    margin: 15px 45px;
    border-radius: 0 12px 12px 0;
    box-shadow: 0 2px 4px rgba(59,130,246,0.1);
  }

  /* Layout helpers */
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 30px;
    margin: 0 45px;
  }
  .columns3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 20px;
    margin: 0 45px;
  }

  /* Modern tables */
  table {
    font-size: 0.85em;
    width: calc(100% - 90px);
    margin: 15px 45px;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }
  th {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
    color: white;
    padding: 12px 10px;
    font-weight: 600;
    text-align: left;
  }
  td {
    padding: 10px;
    border-bottom: 1px solid #e2e8f0;
    background: white;
  }
  tr:last-child td {
    border-bottom: none;
  }
  tr:hover td {
    background: #f7fafc;
  }

  /* Images */
  img {
    margin: 15px auto;
    display: block;
    border-radius: 8px;
  }

  /* Lists */
  ul, ol {
    margin: 12px 0;
    padding-left: 25px;
  }
  li {
    margin: 10px 0;
    line-height: 1.5;
  }
  li::marker {
    color: var(--accent);
  }

  /* Section divider slides - dramatic gradient */
  section.section-divider {
    justify-content: center;
    align-items: center;
    text-align: center;
    background: linear-gradient(135deg, var(--primary) 0%, #0f2744 50%, var(--primary-light) 100%);
  }
  section.section-divider h1 {
    background: transparent;
    color: white;
    font-size: 2.5em;
    padding: 25px;
    text-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }
  section.section-divider h2 {
    background: transparent;
    color: var(--accent-soft);
    font-size: 1.4em;
    font-weight: 400;
  }

  /* Page numbers */
  section::after {
    color: var(--text-light);
    font-size: 0.75em;
    font-weight: 500;
  }

  /* ===== TITLE SLIDE ===== */
  section.title-slide {
    justify-content: center;
    align-items: center;
    text-align: center;
    background: linear-gradient(135deg, var(--primary) 0%, #0f2744 40%, var(--primary-light) 100%);
    position: relative;
    overflow: hidden;
  }
  section.title-slide::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 70%, rgba(232,90,79,0.15) 0%, transparent 50%),
                radial-gradient(circle at 70% 30%, rgba(46,90,143,0.2) 0%, transparent 50%);
    animation: none;
  }
  section.title-slide h1 {
    background: transparent;
    color: white;
    font-size: 2.8em;
    padding: 0 60px 15px 60px;
    text-shadow: 0 4px 20px rgba(0,0,0,0.4);
    letter-spacing: -0.03em;
    line-height: 1.2;
    position: relative;
    z-index: 1;
  }
  section.title-slide h2 {
    background: transparent;
    color: var(--accent-soft);
    font-size: 1.5em;
    font-weight: 400;
    padding: 0 60px 30px 60px;
    border: none;
    position: relative;
    z-index: 1;
  }
  section.title-slide p {
    color: rgba(255,255,255,0.9);
    font-size: 1.1em;
    margin-top: 20px;
    position: relative;
    z-index: 1;
  }
  section.title-slide strong {
    color: white;
    font-weight: 600;
  }

  /* Summary slide */
  section.summary-slide {
    background: linear-gradient(180deg, #f0f4f8 0%, #e2e8f0 100%);
  }
  section.summary-slide h1 {
    background: linear-gradient(135deg, var(--success) 0%, #1e8678 100%);
  }
---

<!-- _class: title-slide -->
<!-- _paginate: false -->

# Machine Learning
# Tasks, Taxonomy & Beyond
## From Classification to Deep Learning

**Nipun Batra** | IIT Gandhinagar

---

# ML is Everywhere in Your Daily Life

| Time | What You Do | ML Behind It |
|------|-------------|--------------|
| Morning | Phone unlocks with your face | Face Recognition |
| Commute | Google Maps predicts traffic | Time Series Prediction |
| Email | Gmail filters spam, suggests replies | Text Classification + Generation |
| Music | Spotify recommends songs | Recommendation Systems |
| Shopping | Amazon shows "You might also like..." | Collaborative Filtering |
| Photos | Google Photos groups by faces | Clustering + Image Classification |
| Evening | Netflix suggests what to watch | Recommendation Systems |
| Chat | You ask ChatGPT a question | Language Models (Generative AI) |

**Each of these is a different ML task!**

---

# ML Notation: The Basics

| Symbol | Meaning | Example |
|--------|---------|---------|
| **X** (bold, uppercase) | Input features (matrix) | All training images |
| **x** (bold, lowercase) | Single input sample | One image |
| **y** (bold, lowercase) | Output/Target | Label, price, category |
| **f(x; theta)** | Model with parameters | Neural network |
| **y_hat** | Model prediction | y_hat = f(x) |

| Dataset Term | What it is | Typical Split |
|--------------|-----------|---------------|
| **Training set** | Data to learn from | ~70-80% |
| **Validation set** | Data to tune hyperparameters | ~10-15% |
| **Test set** | Final evaluation (never peek!) | ~10-15% |

<div class="example">

**Example:** Spam detection
- **x** = email text ("Buy now! Limited offer!")
- **y** = label (spam=1, not spam=0)
- Model learns: f(**x**) -> **y_hat** (predicted probability)

</div>

---

# The Big Question

Every ML task boils down to **one question**:

## "What are you trying to PREDICT?"

<div class="columns">
<div>

**Predicting a Category?**
Classification
*"Is this email spam?"*

**Predicting a Number?**
Regression
*"What will be the price?"*

</div>
<div>

**Predicting a Sequence?**
Seq2Seq
*"How do you say this in French?"*

**Creating Something New?**
Generative
*"Draw me a cat in space"*

</div>
</div>

<div class="insight">
Once you know the "output type", you know which family the task belongs to!
</div>

---

# The Three Learning Paradigms

![w:1000 center](diagrams/svg/learning_paradigms.svg)

---

# The Universal ML Recipe

![w:900 center](diagrams/svg/ml_recipe.svg)

What changes between tasks:
- What **X** looks like (image, text, audio, numbers)
- What **Y** looks like (label, number, sequence, image)
- How we **measure success** (accuracy, MSE, IoU, BLEU)

<div class="insight">
The same Transformer architecture powers ChatGPT, DALL-E, and self-driving cars!
</div>

---

<!-- _class: section-divider -->
# Part 1: Classification
## "Which Bucket Does This Belong To?"

---

# Classification: Real Examples from CIFAR-10

![w:800 center](examples/cifar10_examples.png)

You look at the input and pick **one category** from a fixed set. That's classification!

---

# Classification: How a Decision Tree Learns

![w:1000 center](examples/decision_tree_example.png)

A decision tree learns **if-then rules** from data:
"If sepal length > 5.5 AND sepal width < 3.0, then iris-versicolor"

---

# Classification: Logistic Regression

![w:900 center](examples/logistic_regression_example.png)

Logistic regression learns a **decision boundary** that separates classes.

---

# Binary vs Multi-Class Classification

<div class="columns">
<div>

**Binary Classification**
*Two possible outcomes*

- Spam / Not Spam
- Fraud / Legitimate
- Pass / Fail
- Tumor: Benign / Malignant

</div>
<div>

**Multi-Class Classification**
*Many possible outcomes*

- Digit recognition (0-9)
- ImageNet (1000 classes)
- Emotion detection (6+ emotions)
- Animal species identification

</div>
</div>

<div class="insight">
Same algorithm, just different number of outputs!
</div>

---

# The Math: Softmax Turns Scores into Probabilities

![w:900 center](examples/softmax_example.png)

**Softmax** converts raw scores (logits) to probabilities that sum to 1.

The model isn't just saying "Cat" - it's saying "85% sure it's a cat!"

---

<!-- _class: section-divider -->
# Part 2: Regression
## "How Much? How Many?"

---

# Regression: When the Answer is a Number

Classification: *"Which category?"* - Discrete answer
Regression: *"How much?"* - Continuous number

| Question | Answer |
|----------|--------|
| "How old is this person?" | 27.3 years |
| "What's this house worth?" | $425,000 |
| "How many units will sell?" | 1,247 units |
| "What temperature tomorrow?" | 28.5 C |
| "How long until the bus arrives?" | 7.2 minutes |

The output is **any number** on a continuous scale!

---

# Regression in Action: Linear Regression

![w:1000 center](examples/linear_regression_example.png)

The model learns: **Price = $50,000 + $150 * (square feet)**

---

# Regression is Hidden Everywhere!

Bounding box detection is actually **regression**:

![w:700 center](diagrams/svg/bbox_regression.svg)

---

<!-- _class: section-divider -->
# Part 3: Computer Vision Hierarchy
## From Labels to Pixels

---

# The Vision Task Ladder

![w:900 center](diagrams/svg/vision_tasks_hierarchy.svg)

Each level gives you **more information** but requires **more data and compute**!

---

# Level 1: Image Classification

**What:** Assign one label to an image.

![w:600 center](examples/cifar10_examples.png)

**Use Cases:**
- Google Photos: "Show me all photos with dogs"
- Medical: "Is this X-ray normal or abnormal?"
- Quality Control: "Is this product defective?"

---

# Level 2: Object Detection

**Detection = Classification (what) + Regression (where)**

![w:900 center](examples/coco_detection_examples.png)

Output: List of `(class_name, confidence, x, y, width, height)` for each object

---

# Level 3 & 4: Segmentation

![w:700 center](examples/segmentation_comparison.png)

---

# Instance Segmentation in Action

![w:900 center](examples/instance_segmentation_example.png)

<div class="insight">
Self-driving cars need Instance Segmentation - they must track WHICH car is doing what!
</div>

---

# Pose Estimation: Finding Body Keypoints

**What:** Find skeleton keypoints of humans or animals.

![w:800 center](examples/pose_estimation_example.png)

**Applications:** Fitness apps, motion capture, sign language, fall detection

---

<!-- _class: section-divider -->
# Part 4: Natural Language Processing
## Teaching Machines to Read & Write

---

# The NLP Task Landscape

| Task Type | What It Does | Example |
|-----------|--------------|---------|
| **Sentiment Analysis** | Classify emotion | "Great movie!" → Positive |
| **Named Entity Recognition** | Find names, places, dates | "Sundar Pichai visited NYC" |
| **Question Answering** | Find answers in text | "When was Einstein born?" |
| **Translation** | Convert between languages | English → Hindi |
| **Summarization** | Shorten long text | 1000 words → 50 words |
| **Text Generation** | Create new text | ChatGPT, Claude |

<div class="realworld">
Modern LLMs (GPT-4, Claude) can do ALL of these with a single model!
</div>

---

# Named Entity Recognition (NER)

Classify **each word** in the sequence:

```
   Input:    "Sundar  Pichai   visited  New    York   yesterday"
              │       │        │        │      │      │
              ▼       ▼        ▼        ▼      ▼      ▼
   Output:   PER     PER       O       LOC    LOC     O

   PER = Person Name
   LOC = Location
   O   = Other (not an entity)
```

<div class="insight">
Think of it as "semantic segmentation for text" - every word gets a label!
</div>

---

<!-- _class: section-divider -->
# Part 5: Unsupervised Learning
## Finding Patterns Without Labels

---

# Supervised vs Unsupervised

![w:900 center](diagrams/svg/supervised_vs_unsupervised.svg)

**No one tells the model what to look for - it discovers structure on its own!**

---

# Clustering: K-Means in Action

![w:1000 center](examples/kmeans_example.png)

**K-Means:** No labels needed! The algorithm discovers natural groupings.

<div class="realworld">
**Applications:** Customer segmentation, gene expression analysis, document clustering
</div>

---

# Dimensionality Reduction

**Problem:** High-dimensional data is hard to visualize.

```
Original: 1000-dimensional data
         (Can't visualize 1000 axes!)
                    │
               PCA / t-SNE
                    │
                    ▼
         Just 2D: [0.45, -0.23]
                    │
                    ▼
            Can now plot it!

           •  •  •         ← Cluster 1
          •    •
            ▲  ▲  ▲        ← Cluster 2
              ■ ■ ■        ← Cluster 3
```

---

<!-- _class: section-divider -->
# Part 6: Generative Models
## Creating New Data

---

# Generative vs Discriminative

![w:800 center](diagrams/svg/generative_vs_discriminative.svg)

---

# The Generative AI Revolution

| Domain | Tool | What It Does |
|--------|------|--------------|
| **Text** | ChatGPT, Claude | Write essays, code, poems |
| **Images** | DALL-E, Midjourney, Stable Diffusion | Generate any image from text |
| **Music** | Suno, Udio | Create full songs with lyrics |
| **Video** | Sora, Runway | Generate realistic video clips |
| **Code** | GitHub Copilot, Claude | Write and debug code |
| **Voice** | ElevenLabs | Clone and synthesize voices |

<div class="insight">
All of these generate NEW content that never existed before!
</div>

---

<!-- _class: section-divider -->
# Part 7: Multimodal AI
## Combining Everything

---

# Multimodal = Multiple Modalities

**Modalities:** Text, Image, Audio, Video, etc.

![w:900 center](diagrams/svg/multimodal.svg)

<div class="insight">
Modern AI (GPT-4, Claude, Gemini) is multimodal - it can see AND read AND hear!
</div>

---

# Visual Question Answering (VQA)

```
   Image:                      Questions & Answers:
   ┌───────────────────┐
   │                   │      Q: "What color is the car?"
   │    [Red car on    │      A: "Red"
   │     a road with   │
   │     trees]        │      Q: "Is it daytime or night?"
   │                   │      A: "Daytime"
   └───────────────────┘
                              Q: "How many trees are visible?"
   Requires BOTH:             A: "Four trees"
   - Understanding image
   - Understanding language
   - Reasoning about both!
```

---

<!-- _class: section-divider -->
# Part 8: Reinforcement Learning
## Learning Through Interaction

---

# RL: A Different Paradigm

![w:900 center](diagrams/svg/rl_loop.svg)

**Goal:** Maximize total reward over time through trial and error.

---

# RL Examples

| Domain | Example | What It Learned |
|--------|---------|-----------------|
| **Games** | AlphaGo | Beat world champion at Go |
| **Games** | AlphaStar | Grandmaster at StarCraft II |
| **Robotics** | Boston Dynamics | Walk, run, dance |
| **Infrastructure** | Google Data Centers | 40% energy reduction |
| **AI Alignment** | RLHF for ChatGPT | Be helpful and safe |

<div class="insight">
RLHF (Reinforcement Learning from Human Feedback) is how ChatGPT learns to be helpful!
</div>

---

<!-- _class: section-divider -->
# Part 9: The Common Thread
## Neural Networks & Deep Learning

---

# Neural Networks: The Universal Tool

![w:700 center](examples/neural_network_diagram.png)

**All these tasks use the same fundamental building block:** Neural networks!

---

# NN Output: Binary Classification

![w:900 center](diagrams/svg/nn_binary_classification.svg)

**Output:** 1 neuron with **Sigmoid** activation → probability p ∈ [0, 1]

**Loss:** Binary Cross-Entropy = -[y·log(p) + (1-y)·log(1-p)]

<div class="example">
Example: Disease prediction, spam detection, fraud detection
</div>

---

# NN Output: Multi-class Classification

![w:900 center](diagrams/svg/nn_multiclass.svg)

**Output:** C neurons with **Softmax** → probabilities sum to 1.0

**Loss:** Categorical Cross-Entropy = -Σ yᵢ·log(pᵢ)

<div class="example">
Example: Digit recognition (10 classes), ImageNet (1000 classes)
</div>

---

# NN Output: Regression

![w:900 center](diagrams/svg/nn_regression.svg)

**Output:** 1 neuron with **No activation** (linear) → any real number

**Loss:** Mean Squared Error (MSE) = (1/n)Σ(yᵢ - ŷᵢ)²

<div class="example">
Example: House prices, stock prediction, age estimation
</div>

---

# NN Output: Object Detection (Multi-task)

![w:900 center](diagrams/svg/nn_detection.svg)

**Output per detection:**
- 4 values: Box coordinates (x, y, w, h) - *regression*
- 1 value: Objectness score - *sigmoid*
- C values: Class probabilities - *softmax*

**Loss:** L = λ₁·L_box + λ₂·L_obj + λ₃·L_class

---

# NN Output: Summary

| Task | Output Neurons | Activation | Loss Function |
|------|---------------|------------|---------------|
| **Binary Classification** | 1 | Sigmoid | Binary Cross-Entropy |
| **Multi-class (C classes)** | C | Softmax | Categorical Cross-Entropy |
| **Multi-label** | C | Sigmoid (each) | Binary CE (per label) |
| **Regression** | 1 (or k) | None/Linear | MSE or MAE |
| **Detection** | B × (5 + C) | Mixed | Multi-part loss |

<div class="insight">
The output layer design tells you everything about the task type!
</div>

---

# How Neural Networks Learn: Gradient Descent

![w:800 center](examples/gradient_descent.png)

**Training = Finding the weights that minimize the loss function**

---

# The Deep Learning Revolution

![w:900 center](diagrams/svg/deep_learning_revolution.svg)

<div class="insight">
Deep learning learns the features automatically - no hand-engineering needed!
</div>

---

# The Decision Flowchart

![w:800 center](diagrams/svg/ml_decision_flowchart.svg)

---

# Key Takeaways

1. **Classification** - Predict a category (discrete)
2. **Regression** - Predict a number (continuous)
3. **Detection** - Classification + Box Regression
4. **Segmentation** - Classification for every pixel
5. **Seq2Seq** - Sequence in, sequence out
6. **Unsupervised** - Find patterns without labels
7. **Generative** - Create new data
8. **Multimodal** - Combine text, images, audio
9. **RL** - Learn from rewards through interaction

<div class="insight">
Understanding the output type tells you which family of techniques to use!
</div>

---

# Coming Up Next

**Lecture 3:** Language Models
- Next Token Prediction
- Pre-training, SFT, RLHF
- From GPT to ChatGPT

**Lecture 4:** Object Detection
- YOLO and beyond
- Real-time detection

---

<!-- _class: section-divider -->

# Thank You!

**"All models are wrong, but some are useful."**
*— George Box*

### Key Takeaway
Match the **output type** to the right **task formulation**

Questions?

