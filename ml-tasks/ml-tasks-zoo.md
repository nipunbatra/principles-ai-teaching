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
    padding: 12px;
    font-size: 0.8em;
    line-height: 1.35;
    overflow: hidden;
  }
  .task {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-left: 4px solid #2E86AB;
    padding: 10px 12px;
    margin: 8px 0;
    border-radius: 0 6px 6px 0;
  }
  .example {
    background: #e8f5e9;
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
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  table { font-size: 0.85em; width: 100%; }
  th { background: #2E86AB; color: white; padding: 6px; }
  td { padding: 6px; border-bottom: 1px solid #dee2e6; }
---

# The Machine Learning Task Zoo
## A Tour of 40+ Real-World Problems

**Nipun Batra** · IIT Gandhinagar

---

# How to Think About ML Tasks

Every task is defined by **what goes in** and **what comes out**:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   INPUT (X)            MODEL              OUTPUT (Y)            │
│   ─────────     ─────────────────     ─────────────            │
│   Image         │               │     "Cat"                     │
│   Text      ───►│   f(x; θ)     │───► 0.87                      │
│   Audio         │               │     [x, y, w, h]              │
│   Numbers       └───────────────┘     "Bonjour"                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

<div class="insight">
The same model architecture can solve many different tasks — what changes is the data!
</div>

---

# Domain 1: Computer Vision
## Teaching Machines to See

---

# The Vision Task Hierarchy

```
┌────────────────────────────────────────────────────────────────┐
│  LEVEL 1: Classification     "There's a dog somewhere here"   │
│  ────────────────────────────────────────────────────────────  │
│                                                                │
│  LEVEL 2: Detection          "Dog is HERE → [box]"            │
│  ────────────────────────────────────────────────────────────  │
│                                                                │
│  LEVEL 3: Segmentation       "These EXACT pixels are dog"     │
│  ────────────────────────────────────────────────────────────  │
│                                                                │
│  LEVEL 4: Pose Estimation    "Dog's legs are at (x₁,y₁)..."   │
└────────────────────────────────────────────────────────────────┘
              ▲
              │
        More precision, more data, more compute
```

---

# Task 1: Image Classification

<div class="columns">
<div>

**What:** Assign one label to an image.

```
┌─────────────────┐
│                 │
│   [Photo of     │──► "Golden Retriever"
│    a dog]       │
│                 │
└─────────────────┘
```

**Real-world uses:**
- Google Photos auto-tagging
- Medical X-ray diagnosis
- Quality control in factories

</div>
<div>

<div class="example">

**Example: MNIST Digits**
```
Input:  28×28 grayscale image
Output: 0, 1, 2, ..., or 9

    ████
   █    █
        █
       █
      █
     █
     █        → "7"
```

</div>

</div>
</div>

---

# Task 2: Object Detection

<div class="columns">
<div>

**What:** Find objects AND locate them.

```
┌─────────────────────────┐
│  ┌────┐                 │
│  │dog │   ┌──────┐      │
│  │0.95│   │person│      │
│  └────┘   │ 0.91 │      │
│           └──────┘      │
└─────────────────────────┘
```

**Output:** List of (class, confidence, x, y, w, h)

</div>
<div>

<div class="example">

**Example: Self-Driving Car**
```
Detections:
├─ Car      at (120, 80)  conf: 0.97
├─ Car      at (400, 90)  conf: 0.89
├─ Person   at (300, 150) conf: 0.92
└─ Traffic  at (250, 20)  conf: 0.99
   Light
```

</div>

</div>
</div>

---

# Task 3-4: Semantic vs Instance Segmentation

```
ORIGINAL IMAGE:              SEMANTIC SEG:              INSTANCE SEG:
┌──────────────────┐        ┌──────────────────┐      ┌──────────────────┐
│                  │        │ SSSSSSSSSSSSSSSS │      │ SSSSSSSSSSSSSSSS │
│   [Car] [Car]    │   ►    │ SSSSSSSSSSSSSSSS │  ►   │ SSSSSSSSSSSSSSSS │
│                  │        │                  │      │                  │
│  ████   ████     │        │  CCCC   CCCC     │      │  1111   2222     │
│  ████   ████     │        │  CCCC   CCCC     │      │  1111   2222     │
│══════════════════│        │══════════════════│      │══════════════════│
│ RRRRRRRRRRRRRRRR │        │ RRRRRRRRRRRRRRRR │      │ RRRRRRRRRRRRRRRR │
└──────────────────┘        └──────────────────┘      └──────────────────┘
                             S=Sky, C=Car, R=Road      Car #1 vs Car #2
```

<div class="insight">
Self-driving needs **Instance** — you must track which car is which!
</div>

---

# Task 5: Pose Estimation

**What:** Find body keypoints (skeleton).

```
Original:                    Detected Skeleton:

  ┌─────────┐                     ●  ← Head
  │         │                    /│\
  │  Person │         ──►       / │ \
  │ standing│                  ●  ●  ●  ← Shoulders, Elbows
  │         │                     │
  └─────────┘                    / \
                                ●   ●  ← Hips, Knees, Ankles
                               /     \
                              ●       ●
```

**Uses:** Fitness apps, motion capture, sports analytics, sign language

---

# Task 6: Depth Estimation

**What:** Predict distance of each pixel from camera.

```
RGB Image:                   Depth Map:
┌──────────────────┐        ┌──────────────────┐
│    🏔️ (far)      │        │ ░░░░░░░░░░░░░░░░ │  ░ = far (light)
│                  │        │                  │
│  🌳 (medium)     │   ►    │  ▒▒▒▒            │  ▒ = medium
│                  │        │                  │
│ 🚗 (close)       │        │ ████████         │  █ = close (dark)
└──────────────────┘        └──────────────────┘
```

**Uses:** AR/VR, robotics, 3D reconstruction from single camera

---

# Task 7: Optical Flow

**What:** Track pixel movement between video frames.

```
Frame t:                Frame t+1:              Flow Vectors:
┌──────────────┐       ┌──────────────┐        ┌──────────────┐
│              │       │              │        │              │
│   ●          │  ──►  │        ●     │   =    │   ───────►   │
│              │       │              │        │              │
│        ▲     │       │    ▲         │        │        ◄──   │
│              │       │              │        │              │
└──────────────┘       └──────────────┘        └──────────────┘
                                                (motion vectors)
```

**Uses:** Video compression, action recognition, visual odometry

---

# Task 8: Face Recognition

**What:** Identify WHO a face belongs to.

```
┌───────────┐      ┌───────────────────────────────────────┐
│           │      │                                       │
│   Face    │ ───► │   Embedding: [0.23, -0.41, 0.87, ...] │
│   Image   │      │   (128-dimensional vector)            │
│           │      │                                       │
└───────────┘      └───────────────────────────────────────┘
                                    │
                                    ▼
                          Compare with database
                                    │
                                    ▼
                          "Match: Nipun Batra"
```

**Note:** Face Detection (where) ≠ Face Recognition (who)

---

# Domain 2: Natural Language Processing
## Teaching Machines to Read & Write

---

# Task 9: Sentiment Analysis

**What:** Classify text by emotion/opinion.

<div class="columns">
<div>

```
┌─────────────────────────────┐
│ "This movie was absolutely  │
│  amazing! Best film of the  │
│  year!"                     │
└──────────────┬──────────────┘
               │
               ▼
         ┌───────────┐
         │ POSITIVE  │
         │  (0.96)   │
         └───────────┘
```

</div>
<div>

<div class="example">

**Use Cases:**
- Brand monitoring on Twitter
- Product review analysis
- Customer feedback triage

**Output Options:**
- Binary: Positive/Negative
- 5-class: ⭐ to ⭐⭐⭐⭐⭐
- Continuous: -1.0 to +1.0

</div>

</div>
</div>

---

# Task 10: Named Entity Recognition (NER)

**What:** Find and label names, places, dates, etc.

```
Input:  "Elon Musk announced that Tesla will open a factory
         in Berlin by March 2025."

Output:
        ┌────────┐                    ┌───────┐
        │  PER   │                    │  ORG  │
        └───┬────┘                    └───┬───┘
            │                             │
        "Elon Musk announced that Tesla will open a factory
                                                    ┌───────┐ ┌──────────┐
                                                    │  LOC  │ │   DATE   │
                                                    └───┬───┘ └────┬─────┘
                                                        │          │
         in Berlin by March 2025."
```

**Uses:** Information extraction, knowledge graphs, search engines

---

# Task 11: Machine Translation

**What:** Convert text from one language to another.

```
┌─────────────────────────────────┐
│ English:                        │
│ "The weather is beautiful today"│
└────────────────┬────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │  Transformer  │
         │   (Encoder-   │
         │    Decoder)   │
         └───────┬───────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Hindi:                          │
│ "आज मौसम बहुत सुंदर है"            │
└─────────────────────────────────┘
```

**Key challenge:** Different word order, idioms, context

---

# Task 12: Text Summarization

<div class="columns">
<div>

**Extractive:** Pick important sentences.
```
Long Article:
┌─────────────────────┐
│ Sentence 1          │ ← Selected
│ Sentence 2          │
│ Sentence 3          │
│ Sentence 4          │ ← Selected
│ Sentence 5          │
│ Sentence 6          │ ← Selected
│ ...                 │
└─────────────────────┘
```

</div>
<div>

**Abstractive:** Generate new text.
```
Long Article:
┌─────────────────────┐
│ [Original 1000      │
│  words...]          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ "New summary in     │
│  different words    │
│  (100 words)"       │
└─────────────────────┘
```

</div>
</div>

<div class="insight">
LLMs like GPT-4 do **abstractive** summarization — they paraphrase!
</div>

---

# Task 13: Question Answering

<div class="columns">
<div>

**Extractive QA:**
Find answer span in text.

```
Context: "Albert Einstein
was born in Ulm, Germany
on March 14, 1879."

Question: "Where was
Einstein born?"

Answer: "Ulm, Germany"
        ▲
        └── Highlight in text
```

</div>
<div>

**Generative QA:**
Generate free-form answer.

```
Question: "Explain
quantum entanglement
to a 5-year-old."

Answer: "Imagine two
magic coins that always
land the same way, no
matter how far apart..."
        ▲
        └── Generated new text
```

</div>
</div>

---

# Task 14: Text Generation (LLMs)

**What:** Predict and generate next tokens.

```
Prompt:  "The secret to happiness is"
           │
           ▼
    ┌─────────────────┐
    │      LLM        │
    │  (GPT, Claude)  │
    └────────┬────────┘
             │
             ▼
         "not"     (token 1)
             │
             ▼
         "in"      (token 2)
             │
             ▼
         "wealth"  (token 3)
             │
             ▼
         ...

Output: "The secret to happiness is not in wealth but in
         meaningful connections with others."
```

---

# Domain 3: Audio & Speech
## Teaching Machines to Hear

---

# Task 15: Speech-to-Text (ASR)

**What:** Convert spoken audio to text.

```
Audio Waveform:                        Text Output:
┌────────────────────────────────┐
│  ∿∿∿∿∿∿╱╲∿∿∿╱╲╲∿∿∿∿∿∿∿∿∿∿∿╱╲∿  │ ──►  "Hello, how are
│ ∿∿╱╲∿∿∿∿∿∿∿∿∿∿∿∿∿╱╲∿∿∿∿∿∿∿∿∿∿  │       you today?"
│∿∿∿∿∿∿∿∿╱╲╲╲∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿  │
└────────────────────────────────┘

Pipeline:
Audio → Spectrogram → Encoder → Decoder → Text
```

**Uses:** Siri, Alexa, YouTube captions, meeting transcription

---

# Task 16: Text-to-Speech (TTS)

**What:** Convert text to natural-sounding audio.

```
Text Input:                           Audio Output:
┌─────────────────────┐
│ "Welcome to the     │              ┌────────────────────┐
│  future of AI."     │    ──►       │  ∿∿∿╱╲∿∿∿╱╲╲∿∿∿∿∿∿  │
└─────────────────────┘              │ ∿╱╲∿∿∿∿∿∿∿∿╱╲∿∿∿∿∿  │
                                     └────────────────────┘

Modern TTS:
Text → Acoustic Model → Vocoder → Waveform
       (predicts      (generates
        mel-spectrogram) audio)
```

**Uses:** GPS navigation, screen readers, audiobooks, voice assistants

---

# Task 17-18: Speaker ID & Verification

<div class="columns">
<div>

**Speaker Identification:**
Who is speaking? (1-of-N)

```
Voice Sample
     │
     ▼
┌──────────┐
│  Model   │
└────┬─────┘
     │
     ▼
"Speaker: Alice"
(from database of N people)
```

</div>
<div>

**Speaker Verification:**
Is this person who they claim?

```
Voice + "I am Alice"
     │
     ▼
┌──────────┐
│  Model   │
└────┬─────┘
     │
     ▼
"Verified" or "Rejected"
(binary decision)
```

</div>
</div>

<div class="insight">
Your phone uses **verification** to unlock with "Hey Siri"!
</div>

---

# Domain 4: Unsupervised Learning
## Finding Patterns Without Labels

---

# Task 19: Clustering

**What:** Group similar items together automatically.

```
Before (unlabeled):                 After (clustered):

    •    ■                              ○    □
  •   •    ■ ■                        ○   ○    □ □
    •        ■                          ○        □

        ▲   ▲                              △   △
    ▲          ▲                       △          △
      ▲    ▲                             △    △
```

<div class="example">

**Example: Customer Segmentation**
```
Cluster 1: High spenders, infrequent visits  → "VIPs"
Cluster 2: Low spenders, frequent visits     → "Regulars"
Cluster 3: Bargain hunters                   → "Deal seekers"
```

</div>

---

# Task 20: Anomaly Detection

**What:** Find the outliers / unusual patterns.

```
Normal Transactions:          Anomaly Alert:
┌───────────────────────────────────────────────────────────┐
│                                                           │
│  $50  $120  $45  $200  $75  $90  $15000  $80  $110       │
│   ●     ●    ●     ●    ●    ●      ★      ●     ●       │
│                                     ▲                     │
│                                     │                     │
│                              FRAUD DETECTED!              │
└───────────────────────────────────────────────────────────┘
```

**Uses:**
- Credit card fraud detection
- Network intrusion detection
- Manufacturing defect spotting

---

# Task 21: Dimensionality Reduction

**What:** Compress high-dimensional data for visualization or efficiency.

```
Original: 784 dimensions (28×28 image)

    ┌─────────────────────────────────┐
    │ [0.12, 0.45, 0.00, 0.87, ....   │
    │  0.23, 0.00, 0.91, .... (784)]  │
    └────────────────┬────────────────┘
                     │  PCA / t-SNE / UMAP
                     ▼
    ┌─────────────────────────────────┐
    │           [0.45, -0.23]         │  ← Just 2D!
    └─────────────────────────────────┘
                     │
                     ▼
              Can now plot on screen!

         •  •                    ← Digit "0" cluster
        •    •
          ▲  ▲  ▲               ← Digit "1" cluster
            ■ ■ ■               ← Digit "7" cluster
```

---

# Domain 5: Generative Models
## Creating New Content

---

# Task 22: Image Generation

**What:** Create new images from noise or text.

```
Text-to-Image (Stable Diffusion, DALL-E):

Prompt: "A robot painting                  Generated Image:
         a sunset, oil                     ┌─────────────────┐
         painting style"                   │   🤖 🎨 🌅     │
              │                            │                 │
              └──────────────────────────► │  [Beautiful     │
                                           │   AI artwork]   │
                                           └─────────────────┘

Noise-to-Image (GAN, Diffusion):

Random Noise ──► Generator ──► Realistic Face
[z ~ N(0,1)]                   (that doesn't exist!)
```

---

# Task 23: Image Inpainting

**What:** Fill in missing or masked regions.

```
Original with mask:              Inpainted result:
┌─────────────────────┐         ┌─────────────────────┐
│                     │         │                     │
│  🏔️  ████████       │   ──►   │  🏔️  ☀️ clouds     │
│     ████████        │         │     beautiful sky   │
│                     │         │                     │
│  🌲  🏠  🌲         │         │  🌲  🏠  🌲         │
└─────────────────────┘         └─────────────────────┘
   (hole in image)                 (AI filled it in)
```

**Uses:** Remove unwanted objects, restore old photos, extend images

---

# Task 24: Style Transfer

**What:** Apply artistic style to content.

```
Content Image:          Style Image:           Result:
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│               │      │               │      │               │
│  [Photo of    │  +   │  [Van Gogh's  │  =   │  [Photo with  │
│   a bridge]   │      │   Starry      │      │   swirly      │
│               │      │   Night]      │      │   brushwork]  │
└───────────────┘      └───────────────┘      └───────────────┘
```

**The model separates "what" (content) from "how" (style)!**

---

# Task 25: Super Resolution

**What:** Upscale low-resolution images with detail.

```
Low Resolution (64×64):         High Resolution (512×512):
┌───────────────┐               ┌─────────────────────────┐
│               │               │                         │
│   [Blurry     │     ──►       │   [Sharp, detailed      │
│    face]      │    AI         │    face with realistic  │
│               │  upscale      │    skin texture, etc.]  │
└───────────────┘               │                         │
                                └─────────────────────────┘
```

**Uses:** Enhance old photos, upscale video games, restore security footage

---

# Domain 6: Self-Supervised Learning
## The Secret Sauce of Modern AI

---

# Task 26: Masked Language Modeling (BERT-style)

**What:** Predict the hidden word(s).

```
Input:  "The cat sat on the [MASK]."
                              │
                              ▼
                    ┌─────────────────┐
                    │      BERT       │
                    └────────┬────────┘
                             │
                             ▼
Predictions:    "mat"  (0.45)
                "floor" (0.22)
                "couch" (0.15)
                ...
```

<div class="insight">
This is how BERT learned language — by playing fill-in-the-blank billions of times!
</div>

---

# Task 27: Next Token Prediction (GPT-style)

**What:** Predict what comes next.

```
Input:  "The capital of France is"
                                  │
                                  ▼
                        ┌─────────────────┐
                        │      GPT        │
                        └────────┬────────┘
                                 │
                                 ▼
Next token:              "Paris"  (0.89)
                         "the"    (0.05)
                         ...
```

<div class="insight">
GPT, Claude, and all LLMs are trained with just this one task — repeated trillions of times!
</div>

---

# Task 28: Contrastive Learning

**What:** Learn that augmented versions of same image are "similar".

```
Original Image:
    ┌─────────┐
    │  🐱     │
    └────┬────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌─────────┐
│ 🐱      │ │    🐱   │     These should have
│(cropped)│ │(rotated)│     SIMILAR embeddings!
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           ▼
    "Pull embeddings together"

Meanwhile: Push embeddings of DIFFERENT images apart!
```

---

# Domain 7: Reinforcement Learning
## Learning by Doing

---

# Task 29: Game Playing

**What:** Learn optimal strategy through trial and error.

```
Game State (Chess):              Agent Decision:
┌─────────────────────┐
│ ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜   │          ┌─────────────────┐
│ ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟   │    ──►   │  Best move:     │
│ . . . . . . . .   │          │  e2 → e4        │
│ . . . . . . . .   │          │  (eval: +0.3)   │
│ . . . . . . . .   │          └─────────────────┘
│ . . . . . . . .   │
│ ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙   │
│ ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖   │
└─────────────────────┘

AlphaGo/AlphaZero: Learned by playing millions of games against itself!
```

---

# Task 30: Robot Control

**What:** Learn to move in physical world.

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    Environment (Real World / Simulation)                    │
│                                                             │
│         🤖 ─────────────────────────► 🎯                    │
│        Robot                         Goal                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                    ▲           │
                    │           │
              Reward (+1        Actions
              if closer,        (move left,
              -1 if falls)      move right, etc.)
                    │           │
                    │           ▼
              ┌─────────────────────────┐
              │     Policy Network      │
              │   (learns from trial)   │
              └─────────────────────────┘
```

---

# Domain 8: Multimodal Tasks
## Combining Vision + Language

---

# Task 31: Visual Question Answering (VQA)

**What:** Answer questions about images.

```
Image:                          Question & Answer:
┌─────────────────────┐
│                     │        Q: "How many people are
│    🧑‍🤝‍🧑  🐕            │            in the image?"
│                     │
│ People walking      │        A: "Two people"
│ a dog in park       │
│                     │        Q: "What animal is there?"
│    🌳     🌳        │
└─────────────────────┘        A: "A dog"
```

**Requires:** Understanding both image content AND language!

---

# Task 32: Image Captioning

**What:** Generate text description of an image.

```
Image:                              Generated Caption:
┌───────────────────────┐
│                       │
│   🏃‍♂️ 🏃‍♀️ 🏃           │   ──►   "A group of runners
│                       │          participating in a
│   [Marathon scene     │          marathon on a sunny
│    with crowds]       │          day with spectators
│                       │          cheering along the
│   👥 👥 👥 👥        │          street."
└───────────────────────┘
```

**The reverse of VQA:** Instead of answering, we generate description!

---

# Task 33: Text-to-Video

**What:** Generate video from text description.

```
Prompt: "A golden retriever running through a field
         of flowers on a sunny day"
                    │
                    ▼
            ┌─────────────────┐
            │  Video Model    │
            │  (Sora, etc.)   │
            └────────┬────────┘
                     │
                     ▼
    ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
    │Frame 1│ │Frame 2│ │Frame 3│ │Frame 4│ ...
    │  🐕   │ │  🐕   │ │  🐕   │ │  🐕   │
    │ 🌸🌸  │ │ 🌸🌸  │ │ 🌸🌸  │ │ 🌸🌸  │
    └───────┘ └───────┘ └───────┘ └───────┘
```

---

# Domain 9: Tabular & Time Series
## The Classic ML Tasks

---

# Task 34-35: Regression & Classification on Tables

<div class="columns">
<div>

**Tabular Regression:**
```
┌────────┬──────┬────────┐
│ Beds   │ SqFt │ Price  │
├────────┼──────┼────────┤
│ 3      │ 1500 │ ???    │
│ 4      │ 2200 │ ???    │
│ 2      │ 900  │ ???    │
└────────┴──────┴────────┘
          │
          ▼
   Predict: $425,000
```

</div>
<div>

**Tabular Classification:**
```
┌─────┬───────┬─────────┐
│ Age │ Income│ Default?│
├─────┼───────┼─────────┤
│ 35  │ 75K   │ ???     │
│ 52  │ 120K  │ ???     │
│ 28  │ 45K   │ ???     │
└─────┴───────┴─────────┘
          │
          ▼
   Predict: Yes/No
```

</div>
</div>

<div class="insight">
For tabular data, gradient boosting (XGBoost, LightGBM) often beats deep learning!
</div>

---

# Task 36: Time Series Forecasting

**What:** Predict future values from historical patterns.

```
Historical Data:                     Forecast:
                                            ?
Sales                                      ?
  ↑                                      ?
  │    ╱╲    ╱╲    ╱╲    ╱╲           ╱╲
  │   ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲        ╱  ╲
  │  ╱    ╲╱    ╲╱    ╲╱    ╲    ╱╱    ╲╲
  │ ╱                        ╲  ╱        ╲
  │╱                          ╲╱          ╲
  └────────────────────────────┼────────────►
  Jan  Mar  May  Jul  Sep  Nov │ Jan  Mar
                               │
                         Today │   Future
```

**Uses:** Stock prices, weather, energy demand, retail sales

---

# Task 37: Recommendation Systems

**What:** Predict what users will like.

```
User-Item Matrix:              Recommendations:
                               ┌──────────────────────┐
       Movie1 Movie2 Movie3    │ For User A:          │
User A   5      ?      3       │  • Movie2 (pred: 4.2)│
User B   4      5      ?       │  • Movie5 (pred: 4.0)│
User C   ?      4      5       │                      │
                               │ "Because you liked   │
                               │  Movie1 and Movie3"  │
                               └──────────────────────┘

Collaborative Filtering: "Users like you also liked..."
Content-Based: "Similar movies to ones you liked..."
```

---

# Summary: The ML Task Landscape

```
┌─────────────────────────────────────────────────────────────────┐
│                        ML TASK FAMILIES                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SUPERVISED          UNSUPERVISED       SELF-SUPERVISED         │
│  ──────────          ────────────       ───────────────         │
│  • Classification    • Clustering       • Masked LM (BERT)      │
│  • Regression        • Dim. Reduction   • Next Token (GPT)      │
│  • Detection         • Anomaly Det.     • Contrastive           │
│  • Segmentation                                                 │
│  • Seq2Seq                                                      │
│                                                                 │
│  GENERATIVE          REINFORTIC         MULTIMODAL              │
│  ──────────          ─────────          ──────────              │
│  • Image Gen         • Game Playing     • VQA                   │
│  • Text Gen          • Robotics         • Captioning            │
│  • Inpainting        • Trading          • Text-to-Image         │
│  • Style Transfer                       • Text-to-Video         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# Key Takeaways

1. **Every task = Input type + Output type**
2. **Same architectures** (Transformers) work across domains
3. **Self-supervised learning** powers modern AI (GPT, BERT)
4. **Start with the task** → then choose the model
5. **Real-world ML** often combines multiple tasks

<div class="insight">
Pick a task, find a dataset, and start building!
</div>

---

# Thank You!

**"The best way to predict the future is to invent it."** — Alan Kay

## Questions?
