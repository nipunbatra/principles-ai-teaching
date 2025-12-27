---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  /* PowerPoint-like slide layout with fixed title area */
  section {
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 22px;
    padding: 0;
    color: #333;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
  }
  /* Fixed title area at top */
  section h1 {
    color: #2E86AB;
    font-size: 1.6em;
    margin: 0;
    padding: 25px 40px 15px 40px;
    border-bottom: 3px solid #2E86AB;
    background: linear-gradient(180deg, #f8fbfd 0%, #fff 100%);
    flex-shrink: 0;
  }
  section h2 {
    color: #06A77D;
    font-size: 1.0em;
    margin: -10px 0 0 0;
    padding: 0 40px 15px 40px;
    background: linear-gradient(180deg, #fff 0%, #fff 100%);
    border-bottom: 3px solid #2E86AB;
    flex-shrink: 0;
  }
  /* When h1 followed by h2, remove h1 border */
  section h1:has(+ h2) {
    border-bottom: none;
    padding-bottom: 5px;
  }
  section h3 { color: #457B9D; font-size: 1.0em; margin-top: 15px; }
  /* Content area */
  section > *:not(h1):not(h2) {
    padding-left: 40px;
    padding-right: 40px;
  }
  section > p:first-of-type, section > ul:first-of-type,
  section > ol:first-of-type, section > table:first-of-type,
  section > div:first-of-type, section > pre:first-of-type {
    margin-top: 20px;
  }
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
    font-size: 0.8em;
    line-height: 1.3;
    overflow: hidden;
    margin: 10px 40px;
  }
  .example {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-left: 4px solid #06A77D;
    padding: 12px 15px;
    margin: 10px 40px;
    border-radius: 0 8px 8px 0;
  }
  .insight {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 12px 15px;
    margin: 10px 40px;
    border-radius: 0 8px 8px 0;
  }
  .realworld {
    background: #e3f2fd;
    border-left: 4px solid #2196F3;
    padding: 12px 15px;
    margin: 10px 40px;
    border-radius: 0 8px 8px 0;
  }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin: 0 40px; }
  .columns3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin: 0 40px; }
  table { font-size: 0.85em; width: calc(100% - 80px); margin: 10px 40px; }
  th { background: #2E86AB; color: white; padding: 8px; }
  td { padding: 8px; border-bottom: 1px solid #dee2e6; }
  img { margin: 10px auto; display: block; }
  /* Section divider slides */
  section.section-divider {
    justify-content: center;
    align-items: center;
    text-align: center;
    background: linear-gradient(135deg, #2E86AB 0%, #1a5276 100%);
  }
  section.section-divider h1 {
    color: white;
    border: none;
    background: transparent;
    font-size: 2.2em;
    padding: 20px;
  }
  section.section-divider h2 {
    color: #a8d8ea;
    background: transparent;
    border: none;
    font-size: 1.3em;
  }
---

# Machine Learning: Tasks, Taxonomy & Beyond
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

```
                    ┌─────────────────────────────────────┐
                    │          EVERY ML TASK              │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────┐         ┌─────────────────┐         ┌─────────────┐
│   INPUT     │   ───►  │     MODEL       │   ───►  │   OUTPUT    │
│   (X)       │         │    f(X; θ)      │         │    (Y)      │
└─────────────┘         └─────────────────┘         └─────────────┘
```

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

```
   ┌──────────┐
   │   DOG    │  This box needs 4 numbers:
   │          │
   └──────────┘

   x = 50 (left edge)      ← Regression!
   y = 30 (top edge)       ← Regression!
   w = 100 (width)         ← Regression!
   h = 80 (height)         ← Regression!

   DETECTION = Classification (what?) + Regression (where?)
```

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

```
SUPERVISED:                        UNSUPERVISED:
┌─────────────────────────┐        ┌─────────────────────────┐
│                         │        │                         │
│ Data: X (features)      │        │ Data: X (features)      │
│ Labels: Y (answers)     │        │ Labels: NONE!           │
│                         │        │                         │
│ Learn: f(X) → Y         │        │ Find: patterns in X     │
│                         │        │                         │
│ "Teach by example"      │        │ "Learn by exploration"  │
│                         │        │                         │
└─────────────────────────┘        └─────────────────────────┘
```

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

```
DISCRIMINATIVE (Classification):
┌─────────────────────────────────────────────────────────────────┐
│   [Image of cat]  ───►  Model  ───►  "Cat" or "Dog"            │
│   Given X, predict Y. "What IS this?"                           │
└─────────────────────────────────────────────────────────────────┘

GENERATIVE (Creation):
┌─────────────────────────────────────────────────────────────────┐
│   "Draw a cat"   ───►  Model  ───►  [NEW image of a cat!]      │
│   Create NEW X from scratch. "Make something like this"         │
└─────────────────────────────────────────────────────────────────┘
```

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

```
SINGLE-MODAL:                    MULTI-MODAL:
┌─────────────────────┐          ┌─────────────────────────────────┐
│                     │          │                                 │
│ Image → Model → Cat │          │ Image + Question → Model → Answer│
│ (just images)       │          │                                 │
│                     │          │ [Photo of 3 dogs]               │
│ Text → Model → Sent │          │ "How many dogs?"                │
│ (just text)         │          │         ↓                       │
│                     │          │       "Three"                   │
└─────────────────────┘          └─────────────────────────────────┘
```

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

![w:900 center](examples/neural_network_diagram.png)

**All these tasks use the same fundamental building block:** Neural networks!

---

# How Neural Networks Learn: Gradient Descent

![w:1000 center](examples/gradient_descent.png)

**Training = Finding the weights that minimize the loss function**

---

# The Deep Learning Revolution

```
WHY "DEEP" LEARNING?

Traditional ML:
┌────────────┐     ┌──────────────┐     ┌────────┐
│ Raw Input  │ ──► │ Hand-crafted │ ──► │ Simple │ ──► Output
│ (image)    │     │  Features    │     │ Model  │
└────────────┘     └──────────────┘     └────────┘
                   (SIFT, HOG, etc.)

Deep Learning:
┌────────────┐     ┌───────────────────────────────┐
│ Raw Input  │ ──► │    Many Neural Network        │ ──► Output
│ (image)    │     │    Layers (Deep!)             │
└────────────┘     │    [Features learned          │
                   │     automatically!]           │
                   └───────────────────────────────┘
```

<div class="insight">
Deep learning learns the features automatically - no hand-engineering needed!
</div>

---

# The Decision Flowchart

```
START: What do you want to predict?
            │
    ┌───────┴───────┐
    │               │
Category?       Number?
    │               │
    ▼               ▼
Classification  Regression
                    │
    ┌───────────────┤
    │               │
Sequence?       Location?
    │               │
    ▼               ▼
Seq2Seq       Detection
              (Class + Reg)
                    │
            Exact pixels?
                    │
                    ▼
              Segmentation

No labels available?  →  Unsupervised (Clustering, etc.)
Want to create new data?  →  Generative
Learning from trial/error?  →  Reinforcement Learning
```

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

# Thank You!

**"All models are wrong, but some are useful."** - George Box

The key is matching the right model to the right task!

## Questions?

