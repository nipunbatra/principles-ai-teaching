# Principles of AI - Claude Assistant Guide

## Course Overview
**Course:** Principles of AI (BTech First Year)
**Lectures:** 8 sessions
**Emphasis:** Intuition, motivation, examples, visual learning

### Audience: First-Year BTech Students
- **No prior ML/AI background assumed**
- **Math level:** Basic calculus, linear algebra intro
- **Programming:** Python basics (loops, functions, numpy)
- **Goal:** Understand AI landscape, concepts, and how modern AI systems work

## Repository Structure
```
principles-ai-teaching/
├── lectures/
│   ├── 01-what-is-ai/          # AI overview, applications
│   ├── 02-data-foundation/      # Data representation, types, quality
│   ├── 03-supervised-learning/  # Classification & regression
│   ├── 04-model-selection/      # Train/test, validation, overfitting
│   ├── 05-neural-networks/      # Perceptrons to deep networks
│   ├── 06-computer-vision/      # CNNs, object detection, YOLO
│   ├── 07-language-models/      # Next token prediction, transformers
│   └── 08-generative-ai/        # SFT, RLHF, modern AI assistants
├── diagrams/                    # Generated diagram assets
├── ml-tasks/                    # ML task type visualizations
├── object-detection/            # Object detection materials
├── next-token-prediction/       # NTP visualizations
└── themes/                      # Marp themes
```

## Slide System
- **Format:** Marp markdown slides
- **Theme:** `iitgn-modern` (custom theme in themes/)
- **Build:** `marp slides.md -o slides.html` or `marp slides.md -o slides.pdf`
- **Math:** MathJax enabled

## Python Environment

**Use `uv` for all Python package management:**

```bash
# Python location
which python  # /Users/nipun/.uv/base/bin/python

# Install packages with uv
uv pip install matplotlib numpy pillow torchvision

# Run scripts
python generate_diagrams.py
```

**Never use system pip directly** - always use `uv pip install`.

---

## Diagram Generation Guidelines

### Preferred Python Libraries (in order of preference)

#### 1. **Graphviz / PyGraphviz** - Graph structures, flowcharts
```python
from graphviz import Digraph

dot = Digraph(comment='Neural Network')
dot.attr(rankdir='LR')
dot.node('I', 'Input')
dot.node('H', 'Hidden')
dot.node('O', 'Output')
dot.edges(['IH', 'HO'])
dot.render('nn_simple', format='svg')
```

#### 2. **NetworkX** - Network visualizations, graphs
```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_edges_from([('Data', 'Model'), ('Model', 'Prediction')])
nx.draw(G, with_labels=True, node_color='lightblue',
        node_size=2000, font_size=12, arrows=True)
plt.savefig('pipeline.svg', bbox_inches='tight')
```

#### 3. **Diagrams** - Cloud/ML architecture diagrams
```python
from diagrams import Diagram, Cluster
from diagrams.programming.language import Python
from diagrams.onprem.ml import PyTorch

with Diagram("ML Pipeline", show=False, filename="ml_pipeline"):
    data = Python("Data")
    model = PyTorch("Model")
    data >> model
```

#### 4. **Schemdraw** - Circuit-style diagrams, signal flow
```python
import schemdraw
from schemdraw import elements as elm

with schemdraw.Drawing() as d:
    d += elm.Dot().label('x')
    d += elm.Arrow().right()
    d += elm.Box().label('f(x)')
    d += elm.Arrow().right()
    d += elm.Dot().label('y')
    d.save('function.svg')
```

#### 5. **Blockdiag Family** - Block diagrams, sequence diagrams
```python
# blockdiag style (use subprocess or blockdiag library)
# seqdiag for sequence diagrams
# nwdiag for network diagrams
```

#### 6. **TorchViz / HiddenLayer** - Neural network architectures
```python
from torchviz import make_dot
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(10, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 1)
)
x = torch.randn(1, 10)
y = model(x)
make_dot(y, params=dict(model.named_parameters())).render("model", format="svg")
```

#### 7. **Matplotlib/Seaborn** - Data visualizations, plots
```python
import matplotlib.pyplot as plt
import numpy as np

# Use consistent styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
```

### Diagram Style Guidelines
- **Format:** Prefer SVG for scalability, PNG at 300 DPI for fallback
- **Colors:** Use consistent color palette across all diagrams
- **Fonts:** Sans-serif, minimum 12pt for readability
- **Arrows:** Clear direction, labeled when needed
- **Whitespace:** Generous padding, avoid clutter

### IMPORTANT: Use Real Images Over ASCII Art

**Always prefer actual images over ASCII diagrams:**
- Use real images from public datasets (COCO, MNIST, CIFAR, etc.)
- Use matplotlib/seaborn to generate proper visualizations
- Download sample images from Unsplash (free) when needed
- Create Python scripts that generate reproducible diagrams

**Toy Datasets to Use:**
| Task | Dataset | Size | Where |
|------|---------|------|-------|
| Image Classification | MNIST | 70K images | `torchvision.datasets` |
| Image Classification | CIFAR-10 | 60K images | `torchvision.datasets` |
| Object Detection | COCO (sample) | 330K images | `fiftyone` or download |
| Text | Shakespeare | 1MB | `datasets` library |
| Tabular | Iris, California Housing | Small | `sklearn.datasets` |

**Pattern for Each Lecture:**
```python
# Each lecture folder should have:
lectures/XX-topic/
├── slides.md              # The slides
├── generate_diagrams.py   # Python script to generate visuals
├── diagrams/
│   ├── svg/              # Vector graphics for slides
│   └── png/              # Raster fallback
└── examples/             # Code examples with real data
```

**Example: Loading Real Images**
```python
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Load real MNIST data
mnist = datasets.MNIST('./data', download=True, train=True)
img, label = mnist[0]

# Display
plt.imshow(img, cmap='gray')
plt.title(f'Label: {label}')
plt.savefig('diagrams/svg/mnist_example.svg')
```

**Example: COCO Detection Sample**
```python
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Load COCO annotation and display with bounding boxes
# (Much better than ASCII boxes!)
```

**Why Real Images Matter:**
1. Students see what actual data looks like
2. Diagrams are accurate and professional
3. Code is reproducible
4. Connects theory to practice

### Key Diagrams to Create

#### Object Detection Diagrams
```python
# Classification vs Localization vs Detection
# Show progression: single label → single box → multiple boxes

# Bounding Box and IoU
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots()
# Ground truth box (green)
ax.add_patch(patches.Rectangle((0.2, 0.2), 0.4, 0.4,
             linewidth=2, edgecolor='green', facecolor='none'))
# Predicted box (red)
ax.add_patch(patches.Rectangle((0.3, 0.25), 0.35, 0.35,
             linewidth=2, edgecolor='red', facecolor='none'))
# Highlight IoU overlap
ax.set_title('IoU = Intersection / Union')

# YOLO Grid Visualization
# Show image divided into SxS grid, each cell predicting boxes
```

#### LLM Pipeline Diagrams
```python
# The full journey: Pre-training → SFT → RLHF
from graphviz import Digraph

dot = Digraph('LLM_Pipeline')
dot.attr(rankdir='LR')

# Pre-training
dot.node('data', 'Internet Text\n(TB of data)')
dot.node('pretrain', 'Pre-training\n(Next Token)')
dot.node('base', 'Base Model\n"Text Completer"')

# SFT
dot.node('instruct', 'Instruction\nDataset')
dot.node('sft', 'SFT\n(Supervised)')
dot.node('sft_model', 'Instruction\nModel')

# RLHF
dot.node('human', 'Human\nPreferences')
dot.node('rlhf', 'RLHF/DPO')
dot.node('assistant', 'AI Assistant\n(ChatGPT/Claude)')

dot.edges([
    ('data', 'pretrain'), ('pretrain', 'base'),
    ('base', 'sft'), ('instruct', 'sft'), ('sft', 'sft_model'),
    ('sft_model', 'rlhf'), ('human', 'rlhf'), ('rlhf', 'assistant')
])
```

#### Next Token Prediction Visualization
```python
# Show: "The cat sat on the ___"
# With probability distribution over next words
# P(mat)=0.3, P(floor)=0.2, P(dog)=0.01, ...

import matplotlib.pyplot as plt
words = ['mat', 'floor', 'bed', 'roof', 'dog']
probs = [0.35, 0.25, 0.20, 0.15, 0.05]
plt.barh(words, probs, color='steelblue')
plt.xlabel('Probability')
plt.title('Next Token Prediction: "The cat sat on the ___"')
```

#### Attention Intuition Diagram
```python
# Show: "The animal didn't cross the street because it was too tired"
# Highlight which words "it" attends to
# Use heatmap or connection lines with varying thickness
```

---

## Mathematical Notation Conventions

### Vectors and Matrices
| Symbol | Meaning | LaTeX |
|--------|---------|-------|
| $\mathbf{x}$ | Vector (bold lowercase) | `\mathbf{x}` |
| $\mathbf{X}$ | Matrix (bold uppercase) | `\mathbf{X}` |
| $x_i$ | i-th element of vector | `x_i` |
| $X_{ij}$ | Element at row i, col j | `X_{ij}` |
| $\mathbf{x}^\top$ | Transpose | `\mathbf{x}^\top` |
| $\|\mathbf{x}\|$ | Norm | `\|\mathbf{x}\|` |

### Machine Learning Symbols
| Symbol | Meaning | LaTeX |
|--------|---------|-------|
| $\mathbf{X}$ | Feature matrix (n × d) | `\mathbf{X}` |
| $\mathbf{y}$ | Target vector | `\mathbf{y}` |
| $\hat{y}$ | Prediction | `\hat{y}` |
| $\mathbf{w}$ | Weights | `\mathbf{w}` |
| $b$ | Bias | `b` |
| $\theta$ | Parameters | `\theta` |
| $\mathcal{L}$ | Loss function | `\mathcal{L}` |
| $\nabla$ | Gradient | `\nabla` |
| $\eta$ | Learning rate | `\eta` |
| $\sigma$ | Activation (sigmoid) | `\sigma` |

### Sets and Probability
| Symbol | Meaning | LaTeX |
|--------|---------|-------|
| $\mathcal{D}$ | Dataset | `\mathcal{D}` |
| $\mathbb{R}$ | Real numbers | `\mathbb{R}` |
| $P(A)$ | Probability | `P(A)` |
| $P(A|B)$ | Conditional prob. | `P(A|B)` |
| $\mathbb{E}[X]$ | Expectation | `\mathbb{E}[X]` |

### Dimensions Convention
- **n** = number of samples/examples
- **d** = number of features/dimensions
- **k** = number of classes
- **m** = number of hidden units

---

## Teaching Philosophy

### Core Principles
1. **Intuition First:** Always start with WHY before HOW
2. **Visual Learning:** One diagram > 1000 words
3. **Real Examples:** Ground every concept in applications
4. **Build Up:** Simple → Complex, never skip steps
5. **Connect Dots:** Link new concepts to what students already know
6. **First-Year Friendly:** No jargon without explanation

### First-Year Accessibility Guidelines
```
DO:
✓ Use analogies to everyday life (Netflix recommendations, spam filters)
✓ Show before you explain (demo → intuition → math)
✓ Use simple numbers in examples (2x2 matrices, 3-feature datasets)
✓ Celebrate small wins ("you just built a neural network!")
✓ Connect to their world (Instagram filters, ChatGPT, YouTube recommendations)

DON'T:
✗ Assume linear algebra fluency (explain dot products when needed)
✗ Use Greek letters without introduction
✗ Show complex equations without visual buildup
✗ Skip the "why does anyone care?" question
✗ Assume familiarity with ML terminology
```

### Math Accessibility
When introducing math:
1. **Visual first:** Show the geometric/visual intuition
2. **Small example:** Work through with actual numbers
3. **Then notation:** Introduce the formal symbols
4. **Never:** Throw an equation without context

Example flow for dot product:
```
1. "How similar are these two lists of numbers?"
2. Show: [1,2] and [2,1] visually as arrows
3. Multiply and add: 1×2 + 2×1 = 4
4. Then: "We write this as x · y or x^T y"
```

### Slide Structure (per concept)
```
1. MOTIVATION (Why do we need this?)
   - Real-world problem or question
   - Relatable example

2. INTUITION (What's the big idea?)
   - Visual/diagram
   - Analogy to familiar concepts

3. FORMALIZATION (How do we express it?)
   - Mathematical notation
   - Key equations

4. EXAMPLE (How does it work?)
   - Worked example with numbers
   - Code snippet if relevant

5. CONTEXT (Where does this fit?)
   - Connection to other concepts
   - Limitations and extensions
```

### Example-Driven Teaching
Every major concept should have:
- **Toy Example:** Small numbers, hand-computable
- **Visual Example:** Diagram or plot
- **Real Example:** Actual application (ImageNet, GPT, etc.)
- **Interactive Example:** Code students can run

---

## Reference Materials

### Internal Resources
- **ml-teaching repo:** `~/git/ml-teaching/`
  - Supervised learning slides
  - Neural network materials
  - Optimization content
  - Mathematical foundations

### External Resources
- **MIT IntroDeepLearning:** http://introtodeeplearning.com/
  - Excellent visual explanations
  - Lab notebooks
  - Video lectures

### Video Demonstrations
Include video examples showing:
- AlphaFold protein folding
- Self-driving cars
- ChatGPT/Claude demos
- Image generation (DALL-E, Midjourney)
- ML failures (bias, adversarial examples)

---

## Lecture Topics Overview

### Lecture 1: What is AI?
- Historical context (Turing, Dartmouth)
- AI vs ML vs DL distinction
- Applications showcase (videos)
- ML definition (Mitchell)
- What we'll build: SLM preview

### Lecture 2: Data Foundation
- Data types (tabular, images, text, audio)
- Features and labels
- Data quality and bias
- Train/test concept intro

### Lecture 3: Supervised Learning
- Classification vs Regression
- K-NN intuition
- Linear models
- Decision boundaries

### Lecture 4: Model Selection
- Overfitting/underfitting
- Bias-variance tradeoff
- Cross-validation
- Evaluation metrics

### Lecture 5: Neural Networks
- Perceptron history
- Activation functions
- Backpropagation intuition (visual, not math-heavy)
- Universal approximation

### Lecture 6: Computer Vision
- Image as matrix
- Convolution intuition (sliding window demo)
- CNN architectures (LeNet → ResNet journey)
- **Object Detection:**
  - Classification vs Localization vs Detection
  - Bounding boxes and IoU
  - YOLO intuition (grid-based detection)
  - Real-time detection demos
- Transfer learning

### Lecture 7: Language Models (Part 1)
- **Motivation:** Why predict the next token?
- Text as sequences (characters → tokens)
- **Next Character Prediction:**
  - Shakespeare/simple text example
  - Character-level model intuition
  - Temperature and sampling
- **Next Token Prediction:**
  - Tokenization (BPE intuition)
  - From characters to subwords
- **Transformer Motivation (NOT deep math):**
  - Why attention? (looking at all words at once)
  - Self-attention visual intuition
  - "Attention is All You Need" → what it enabled

### Lecture 8: From Language Model to Assistant (Part 2)
- **Recap:** We have a next-token predictor
- **The Problem:** It just completes text, doesn't follow instructions
- **Supervised Fine-Tuning (SFT):**
  - Instruction-response pairs
  - Teaching the model to be helpful
  - Demo: Before vs After SFT
- **Alignment (RLHF intuition):**
  - Why SFT isn't enough
  - Human preferences
  - Reward models (simple intuition)
  - RLHF/DPO overview (conceptual, not math)
- **Putting It Together:**
  - Pre-training → SFT → Alignment pipeline
  - Why ChatGPT/Claude work the way they do
- **Ethics & Future:**
  - AI safety basics
  - Responsible AI use
  - What's next in AI?

---

## Code Style Guidelines (for diagram generation)

### Diagram Scripts
```python
# Keep diagram code clean and reproducible
import matplotlib.pyplot as plt

# Use consistent styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 14

# Save as SVG for slides
plt.savefig('diagram_name.svg', bbox_inches='tight')
```

---

## Build Commands

```bash
# Build single lecture slides
cd lectures/01-what-is-ai
marp slides.md -o slides.html
marp slides.md -o slides.pdf

# Build all lectures
for dir in lectures/*/; do
    (cd "$dir" && marp slides.md -o slides.html && marp slides.md -o slides.pdf)
done
```

---

## Quality Checklist

Before finalizing any lecture:
- [ ] Every concept has visual representation
- [ ] Math notation is consistent with conventions
- [ ] At least 2 real-world examples per major concept
- [ ] Diagrams are clear and properly labeled
- [ ] Flow: Motivation → Intuition → Formalization → Example
- [ ] No unexplained jargon for first-year students
- [ ] Connected to previous and next lectures
- [ ] First-year accessible (no assumed ML background)
- [ ] "Why should I care?" answered for every topic

## Key Intuitions to Nail

### Object Detection
- "Finding AND locating objects = Classification + Where"
- IoU: "How much do these two boxes overlap?"
- YOLO: "Divide image into grid, each cell votes on what's there"

### Next Token Prediction
- "The model is playing a very sophisticated fill-in-the-blank game"
- "Given 'The cat sat on the ___', what word comes next?"
- Temperature: "Low = safe/boring, High = creative/risky"

### SFT
- "Teaching the model to follow instructions by showing examples"
- "Like training a new employee: here's the question, here's how to answer"

### RLHF/Alignment
- "The model knows how to complete text, but which completion is GOOD?"
- "Humans rank responses, model learns what humans prefer"
- "This is why ChatGPT is helpful instead of just completing your sentence"

---

## Useful External Resources

### For Object Detection
- YOLO papers (visual explanations)
- Roboflow tutorials (practical demos)

### For LLM Pipeline
- Karpathy's "Let's build GPT" video
- Anthropic's Constitutional AI paper (for alignment intuition)
- OpenAI's InstructGPT paper (SFT + RLHF)

### For First-Year Friendly Explanations
- 3Blue1Brown neural network series
- MIT IntroDeepLearning (introtodeeplearning.com)
- StatQuest (Josh Starmer) for ML basics
