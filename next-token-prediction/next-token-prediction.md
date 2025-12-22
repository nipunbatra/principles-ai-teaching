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

# Next Token Prediction
## Building ChatGPT from Scratch (Conceptually)

**Nipun Batra** · IIT Gandhinagar

---

# The Journey Ahead

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Level 1: Intuition          "The Autocomplete Game"          │
│      ↓                                                          │
│   Level 2: Counting           "Bigrams (Just Statistics)"      │
│      ↓                                                          │
│   Level 3: Representation     "Embeddings (Meaning as Vectors)"│
│      ↓                                                          │
│   Level 4: Learning           "Neural Networks (The Brain)"    │
│      ↓                                                          │
│   Level 5: Memory             "RNNs (Remembering Context)"     │
│      ↓                                                          │
│   Level 6: Attention          "Transformers (The Revolution)"  │
│      ↓                                                          │
│   Level 7: Scale              "ChatGPT (Putting it Together)"  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# Level 1: The Intuition
## What Are We Really Doing?

---

# The Core Problem

Every language model answers **one simple question**:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   "Given what I have seen so far, what word comes next?"        │
│                                                                 │
│   ┌───────────────────────────────────────┐                     │
│   │ The capital of France is ___          │                     │
│   └───────────────────────────────────────┘                     │
│                       │                                         │
│                       ▼                                         │
│                    "Paris"                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

That's it. Predict the next word. Repeat until done.

---

# Let's Play: The Autocomplete Game

<div class="example">

**Round 1:** "The Eiffel Tower is located in ___"

Your brain: **Paris** (very confident!)

**Round 2:** "I want to eat ___"

Your brain: **pizza? pasta? nothing?** (uncertain!)

**Round 3:** "Once upon a ___"

Your brain: **time** (almost certain!)

</div>

<div class="insight">
Your brain assigns **probabilities** to each possible next word.
Some contexts have obvious answers, others don't!
</div>

---

# It's JUST Prediction

You might think ChatGPT "understands" physics or history.
But all it does is predict the next word.

```
Prompt: "F = m"                  Prediction: "a"      ← Newton's Law!
Prompt: "To be or not to"        Prediction: "be"     ← Shakespeare!
Prompt: "E = mc"                 Prediction: "²"      ← Einstein!
Prompt: "print('Hello"           Prediction: "')"     ← Python!
```

<div class="insight">
If you predict well enough, you **appear** to understand everything.
The model has compressed patterns from human knowledge into its weights.
</div>

---

# The Magic of "Just Prediction"

```
┌─────────────────────────────────────────────────────────────────┐
│ Q: "What is 17 + 28?"                                           │
│                                                                 │
│ The model has seen THOUSANDS of math problems in training:      │
│   "2 + 2 = 4"                                                   │
│   "15 + 10 = 25"                                                │
│   "17 + 28 = 45"   ← Saw this pattern!                         │
│                                                                 │
│ So when asked "17 + 28 =", it predicts "45"                    │
│ Not because it "knows" math, but because that pattern exists!   │
└─────────────────────────────────────────────────────────────────┘
```

<div class="warning">
This is why LLMs can make math mistakes — they're pattern matching,
not actually computing!
</div>

---

# Level 2: The Counting Era
## Bigrams: The Simplest Language Model

---

# The Simplest Possible Model

**Idea:** Just count what letter usually follows each letter.

<div class="example">

**Training data:** Names like `aabid`, `zeel`, `priya`

Count transitions:
- After `a`: saw `a` (1 time), `b` (1 time)
- After `z`: saw `e` (1 time)
- After `e`: saw `e` (1 time), `l` (1 time)

</div>

This is called a **Bigram** model (looks at pairs of 2 characters).

---

# Bigram: The Counting Table

```
        Next Character →
      ┌─────┬─────┬─────┬─────┬─────┬─────┐
      │  a  │  b  │  e  │  i  │  l  │ ... │
   ───┼─────┼─────┼─────┼─────┼─────┼─────┤
C  a  │ 0.3 │ 0.2 │ 0.1 │ 0.2 │ 0.1 │ ... │  (probabilities)
u  b  │ 0.1 │ 0.0 │ 0.1 │ 0.5 │ 0.0 │ ... │
r  e  │ 0.2 │ 0.0 │ 0.3 │ 0.1 │ 0.2 │ ... │
r  i  │ 0.4 │ 0.1 │ 0.1 │ 0.0 │ 0.1 │ ... │
   ↓  │ ... │ ... │ ... │ ... │ ... │ ... │
   ───┴─────┴─────┴─────┴─────┴─────┴─────┘

To generate: Look up current letter → Sample from that row
```

**This table IS the model.** No neural network needed!

---

# Generating Names with Bigrams

```
Step 1: Start with "." (beginning token)
        Look up row "." → High prob for 'a', 's', 'm'
        Sample → Got 'a'

Step 2: Current = 'a'
        Look up row "a" → Moderate prob for 'a', 'b', 'n'
        Sample → Got 'b'

Step 3: Current = 'b'
        Look up row "b" → High prob for 'i', 'a', 'r'
        Sample → Got 'i'

Step 4: Current = 'i'
        Look up row "i" → High prob for 'd', 'n', 'a'
        Sample → Got 'd'

Step 5: Current = 'd'
        Look up row "d" → High prob for "." (end token)
        Sample → Got "."   (DONE!)

Result: "abid"  ← Looks like a real name!
```

---

# Why Bigrams Fail

<div class="columns">
<div>

**The Problem:**
```
Sentence: "The quick brown
           fox jumps over
           the lazy dog."

Question: After "dog",
          what comes next?

Bigram sees: "dog" → ?
             (forgot everything
              before "dog"!)
```

</div>
<div>

**Context is Lost:**
```
With context:
"The cat sat on the ___"
  → Probably "mat"

Without context:
"the ___"
  → Could be anything!

Bigram only sees 1 char!
```

</div>
</div>

<div class="warning">
Bigrams have **no memory**. They forget everything except the last character!
</div>

---

# The Curse of Dimensionality

Why not just count longer patterns?

```
1-gram (Unigram):   26 entries           ← Fits in memory
2-gram (Bigram):    26² = 676            ← Still fine
3-gram (Trigram):   26³ = 17,576         ← OK
4-gram:             26⁴ = 456,976        ← Getting big
5-gram:             26⁵ = 11,881,376     ← Very big
10-gram:            26¹⁰ ≈ 141 TRILLION  ← Impossible!
```

<div class="insight">
We can't just count longer patterns — we need to **generalize**.
This is where neural networks come in!
</div>

---

# Level 3: Representing Meaning
## Embeddings: Words as Vectors

---

# How Do Computers Read?

Computers only understand numbers. How do we convert letters?

**Option A: One-Hot Encoding**
```
'a' = [1, 0, 0, 0, ..., 0]    (26 dimensions)
'b' = [0, 1, 0, 0, ..., 0]
'c' = [0, 0, 1, 0, ..., 0]
...
```

**Problem:** These vectors are **orthogonal** (dot product = 0).
The computer thinks 'a' and 'b' are completely unrelated!

---

# The Problem with One-Hot

```
Distance between letters:

   'a' ●                    'b' ●

   Distance(a, b) = Distance(a, z) = √2

   Every letter is equally far from every other letter!

   But we KNOW:
   - 'a' and 'e' are both vowels (similar!)
   - 'a' and 'x' have nothing in common (different!)
```

We need a smarter representation.

---

# Dense Embeddings: Meaning as Coordinates

**Idea:** Represent each character as a point in space where **similar things are close**.

```
             ▲ Dimension 2 ("Common ending?")
             │
        0.9  │    ● e        ● a
             │        ● i
        0.5  │              ● o
             │                    ● u
        0.1  │
             │  ● b   ● c   ● d   ● f   ← Consonants cluster
        -0.5 │           ● x   ● z      ← Rare letters
             │
             └────────────────────────────► Dimension 1 ("Vowel?")
                  -1       0       1
```

Now `a` and `e` are **mathematically close**!

---

# Word Embeddings: The Famous Example

```
                    ▲
                    │
                    │    ● Queen
                    │
                    │    ● King
                    │
                    │    ● Woman
                    │
                    │    ● Man
                    │
                    └────────────────────────►

    King - Man + Woman = ???

    Move from "King" toward "Woman" direction:
    You land near "Queen"!

    The model learned: King is to Queen as Man is to Woman
```

---

# How Embeddings Are Learned

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Start: Random vectors for each word                           │
│                                                                 │
│  Training:                                                      │
│    "The cat sat on the mat"                                     │
│                                                                 │
│    "cat" often appears near "sat", "dog", "pet"                │
│    → Push these embeddings closer together                      │
│                                                                 │
│    "cat" rarely appears near "quantum", "fiscal"               │
│    → Push these embeddings apart                                │
│                                                                 │
│  After billions of examples:                                    │
│    Similar words → Similar vectors                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# Level 4: Learning Patterns
## Neural Networks: The Brain

---

# From Counting to Learning

```
BIGRAM (Counting):                NEURAL NETWORK (Learning):
┌─────────────────────┐           ┌─────────────────────┐
│                     │           │                     │
│   Count(b|a)        │           │   f(embed(a); θ)    │
│   ──────────        │           │                     │
│   Count(a)          │           │   θ = learned       │
│                     │           │       weights       │
│   (Fixed table)     │           │   (Flexible function)
└─────────────────────┘           └─────────────────────┘
        │                                   │
        ▼                                   ▼
  Only memorizes              Can GENERALIZE to
  exact patterns              unseen patterns!
```

---

# The Neural Network Architecture

```
Input: Last 3 characters → "a" "a" "b"
                            │   │   │
                            ▼   ▼   ▼
┌─────────────────────────────────────────────────────────────┐
│ EMBEDDING LAYER                                             │
│   'a' → [0.2, 0.8]     'a' → [0.2, 0.8]     'b' → [-0.5, 0.1]│
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼ Concatenate
                    [0.2, 0.8, 0.2, 0.8, -0.5, 0.1]
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│ HIDDEN LAYER: 6 inputs → 100 neurons → ReLU                │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT LAYER: 100 → 27 (one per character + end)           │
│ + SOFTMAX → Probabilities that sum to 1                     │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
            P(next='a')=0.05, P(next='i')=0.45, ...
```

---

# Creating Training Data: The Sliding Window

```
Text: "aabid"

Create (context → target) pairs by sliding a window:

    Position 0:  [., ., .] → 'a'     "What comes first?"
    Position 1:  [., ., a] → 'a'     "After nothing+a?"
    Position 2:  [., a, a] → 'b'     "After a, a?"
    Position 3:  [a, a, b] → 'i'     "After a, a, b?"
    Position 4:  [a, b, i] → 'd'     "After a, b, i?"
    Position 5:  [b, i, d] → '.'     "After b, i, d?"

    ┌───┬───┬───┐
    │ . │ a │ a │ ──► Context (input)
    └───┴───┴───┘
                └──► Target: 'b' (what we want to predict)
```

---

# Training: Learning from Mistakes

```
Step 1: Forward Pass
        ┌───────────────────────┐
        │ Input: [a, a, b]      │
        │ ↓                     │
        │ Network predicts:     │
        │ P(i)=0.10 ← Wrong!    │
        │ P(z)=0.30 ← Very wrong│
        │ P(a)=0.20             │
        └───────────────────────┘
        Actual answer: 'i'

Step 2: Compute Loss
        Loss = -log(0.10) = 2.3   ← High loss = bad prediction

Step 3: Backpropagation
        Adjust weights to make P(i) higher next time

Step 4: Repeat millions of times
        → Network learns to predict well!
```

---

# Gradient Descent: Finding the Best Weights

```
Loss
  ▲
  │
  │   ╲
  │    ╲
  │     ╲
  │      ╲
  │       ●  Start (random weights, high loss)
  │        ╲
  │         ╲   ← Follow the slope downhill
  │          ╲
  │           ╲
  │            ●  Better
  │             ╲
  │              ●  Even better
  │               ╲
  │                ●  Minimum! (best weights)
  └──────────────────────────────────────────► Weights

Each step: weights = weights - learning_rate × gradient
```

---

# Level 5: The Context Problem
## Why Fixed Windows Aren't Enough

---

# The Fatal Flaw

Our neural network has a **fixed context window** (e.g., 3 characters).

```
"Alice picked up the golden key. She walked to the door
 and tried to open it with the ___"

What the human sees:  "golden key" (earlier in story)
What the model sees:  "with the"  (only last 3 words!)

                    ┌─────────────────────────────────┐
Model's window:     │ ... with the ___                │
                    └─────────────────────────────────┘
                          ▲
                          │
    "key" is outside the window! The model forgot it!
```

---

# Attempted Solution: RNNs

**Recurrent Neural Networks** maintain a "memory" that carries forward.

```
       h₀        h₁        h₂        h₃        h₄
       │         │         │         │         │
       ▼         ▼         ▼         ▼         ▼
    ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
    │ RNN │──►│ RNN │──►│ RNN │──►│ RNN │──►│ RNN │──► ...
    └─────┘   └─────┘   └─────┘   └─────┘   └─────┘
       ▲         ▲         ▲         ▲         ▲
       │         │         │         │         │
      "The"    "cat"     "sat"     "on"      "the"

    The hidden state h carries information forward!
```

---

# Why RNNs Still Fail

**The Telephone Game Problem:**

```
Start: "Alice has a key"
  │
  ▼ (pass through 50 words)
  │
  ▼ (memory gets compressed, some info lost)
  │
  ▼ (pass through 50 more words)
  │
  ▼ (even more degraded)
  │
End: "She opened the door with the ___"

By now, "key" has been corrupted or forgotten!

This is the "Vanishing Gradient" problem:
Gradients become tiny → Old info can't influence predictions
```

---

# Level 6: The Revolution
## Attention: "Just Look Back!"

---

# The Brilliant Idea

What if, instead of compressing everything into a hidden state...
We could just **look back** at everything directly?

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  "Alice picked up the golden key. She walked to the door       │
│   and tried to open it with the ___"                           │
│                                                                 │
│   Fixed Window: Can only see "with the"                        │
│   RNN: Remembers a blurry summary                              │
│   ATTENTION: Can look at ANY word!                             │
│              ↑                                                  │
│              └── "Let me check... 'key' was mentioned!"        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# Attention: The Searchlight Analogy

```
Reading Methods Compared:

FIXED WINDOW (MLP):
    Reading with tunnel vision
    ┌───┐
    │░░░│ ← Can only see this tiny part
    └───┘

RNN:
    Reading while trying to remember everything
    "I think there was a key... or was it a lock?"

ATTENTION:
    Reading with a highlighter and search engine!
    "Let me search for 'object that opens doors'..."
    Found: "key" at position 7!
```

---

# How Attention Works: Q, K, V

Think of it like a **database lookup**:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Current word: "___" (need to fill in)                         │
│                                                                 │
│  QUERY (Q): "What object was mentioned that opens things?"     │
│                                                                 │
│  For each past word, we have:                                   │
│    KEY (K): What this word is about                            │
│    VALUE (V): The actual content/meaning                        │
│                                                                 │
│  Past words:                                                    │
│    "Alice"  → K: "person name"      V: [embedding of Alice]    │
│    "key"    → K: "object, opens"    V: [embedding of key]  ← Match!│
│    "door"   → K: "object, building" V: [embedding of door]     │
│                                                                 │
│  Score = similarity(Q, K)                                       │
│  Output = weighted sum of Values                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# Attention Scores Visualized

```
Predicting: "The animal didn't cross the street because it was too ___"
                                                              │
                            ┌─────────────────────────────────┘
                            ▼
            "it" refers to what?

Attention Scores (what "it" is looking at):

    The     animal  didn't  cross   the    street  because  it
    0.02    0.75    0.01    0.02    0.01   0.15    0.02     ---
            ^^^^                           ^^^^
            High attention!                Some attention

The model figures out "it" = "animal" (not "street")!
This is learned automatically from data.
```

---

# Self-Attention: Every Word Looks at Every Word

```
Sentence: "The cat sat on the mat"

             The   cat   sat   on   the   mat
           ┌─────┬─────┬─────┬────┬─────┬─────┐
    The    │ 0.5 │ 0.2 │ 0.1 │0.05│ 0.1 │0.05 │
    cat    │ 0.1 │ 0.5 │ 0.2 │0.05│ 0.1 │0.05 │
    sat    │ 0.1 │ 0.3 │ 0.3 │ 0.1│ 0.1 │ 0.1 │
    on     │ 0.1 │ 0.1 │ 0.2 │ 0.3│ 0.2 │ 0.1 │
    the    │ 0.2 │ 0.1 │ 0.1 │ 0.1│ 0.3 │ 0.2 │
    mat    │ 0.1 │ 0.2 │ 0.1 │0.05│ 0.2 │ 0.35│
           └─────┴─────┴─────┴────┴─────┴─────┘

Each row: Where does this word look for context?
Computed in parallel (not sequential like RNN)!
```

---

# The Transformer Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     TRANSFORMER BLOCK                            │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Input Embeddings: [The] [cat] [sat] [on] [the] [mat]      │  │
│  └────────────────────────────┬───────────────────────────────┘  │
│                               ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  + Positional Encoding (so model knows word order)          │  │
│  └────────────────────────────┬───────────────────────────────┘  │
│                               ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  SELF-ATTENTION (every word attends to every other)        │  │
│  └────────────────────────────┬───────────────────────────────┘  │
│                               ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  FEED-FORWARD NETWORK (process each position)               │  │
│  └────────────────────────────┬───────────────────────────────┘  │
│                               ▼                                   │
│                    Repeat 12-96 times!                           │
└──────────────────────────────────────────────────────────────────┘
```

---

# Level 7: From Theory to ChatGPT
## Scaling Up

---

# Our Toy Model vs ChatGPT

```
┌────────────────────────┬──────────────────┬──────────────────┐
│       Feature          │   Our Toy Model  │    ChatGPT       │
├────────────────────────┼──────────────────┼──────────────────┤
│ Vocabulary             │   27 (letters)   │ 100,000 (tokens) │
│ Embedding Size         │   2 dimensions   │ 12,288 dims      │
│ Layers                 │   1 layer        │ 96 layers        │
│ Attention Heads        │   1 head         │ 96 heads         │
│ Parameters             │   ~1,000         │ 175 BILLION      │
│ Training Data          │   1,000 names    │ 500B+ tokens     │
│ Training Time          │   1 minute       │ Months on 1000s  │
│                        │                  │ of GPUs          │
└────────────────────────┴──────────────────┴──────────────────┘
```

Same core algorithm. Just **much, much bigger**.

---

# Tokenization: Not Characters, Not Words

```
LLMs use "TOKENS" — subword units (BPE algorithm):

"unhappiness" → ["un", "happiness"]  or  ["un", "hap", "piness"]

Why?
- Characters: Too slow (many steps to generate a word)
- Words: Too many unique words (millions!)
- Tokens: Best of both worlds (~50,000 tokens)

Example:
┌─────────────────────────────────────────────────────────────────┐
│ Text: "ChatGPT is amazing!"                                      │
│                                                                 │
│ Tokens: ["Chat", "G", "PT", " is", " amazing", "!"]             │
│         [15496, 38,  2898,  318,    4998,      0]  ← Token IDs  │
└─────────────────────────────────────────────────────────────────┘
```

---

# Temperature: The Creativity Knob

When sampling the next token, we can adjust **temperature**:

```
Probabilities for next word after "I love to eat":

                Low Temp (0.1)      High Temp (2.0)
                (Conservative)      (Creative)
┌─────────────┬─────────────────┬─────────────────┐
│   pizza     │     0.80        │     0.25        │
│   pasta     │     0.15        │     0.20        │
│   shoes     │     0.01        │     0.15        │
│   clouds    │     0.001       │     0.12        │
│   dreams    │     0.0001      │     0.10        │
└─────────────┴─────────────────┴─────────────────┘

Low temp → Always picks "pizza" (boring but safe)
High temp → Might pick "clouds" (creative but weird)
```

---

# The Sampling Tree

Because we sample probabilistically, each generation is different!

```
                        "Once upon a"
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
           "time"          "day"         "midnight"
              │               │               │
        ┌─────┼─────┐    ┌────┼────┐    ┌────┼────┐
        ▼     ▼     ▼    ▼    ▼    ▼    ▼    ▼    ▼
     "there" "in"  ","  "a"  "in" ","  "a"  ","  "when"
        │     │     │    │    │    │    │    │    │
        ▼     ▼     ▼    ▼    ▼    ▼    ▼    ▼    ▼
      ...   ...   ... ...  ...  ... ...  ...  ...

Each path = a different story!
```

---

# The Complete Recipe

```
┌─────────────────────────────────────────────────────────────────┐
│                    How to Build an LLM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. COLLECT DATA                                                 │
│    → Scrape the internet, books, code (terabytes of text)      │
│                                                                 │
│ 2. TOKENIZE                                                     │
│    → Convert text to token sequences                            │
│                                                                 │
│ 3. BUILD ARCHITECTURE                                           │
│    → Transformer with embeddings + attention + FFN              │
│                                                                 │
│ 4. TRAIN (Pre-training)                                         │
│    → Predict next token, minimize loss                          │
│    → Weeks on thousands of GPUs                                 │
│                                                                 │
│ 5. FINE-TUNE (Instruction tuning)                               │
│    → Train on (instruction, response) pairs                     │
│                                                                 │
│ 6. RLHF (Optional but important!)                               │
│    → Human feedback to make it helpful & safe                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# Summary: The Full Stack

```
Layer 0: THE TASK
         Predict P(next_token | all_previous_tokens)

Layer 1: REPRESENTATION
         Tokens → Embeddings (meaning as vectors)

Layer 2: CONTEXT
         Self-Attention (look at relevant past tokens)

Layer 3: COMPUTATION
         Feed-Forward layers (process information)

Layer 4: STACKING
         Repeat attention+FFN 96 times for depth

Layer 5: TRAINING
         Next token prediction on internet-scale data

Layer 6: ALIGNMENT
         Instruction tuning + RLHF for helpfulness
```

---

# Resources to Learn More

1. **Andrej Karpathy** - "Neural Networks: Zero to Hero" (YouTube)
   - Builds GPT from scratch in Python

2. **Jay Alammar** - "The Illustrated Transformer" (Blog)
   - Beautiful visualizations of attention

3. **NanoGPT** - Karpathy's GitHub repo
   - Full GPT in ~300 lines of Python

4. **3Blue1Brown** - "Attention in Transformers" (YouTube)
   - Intuitive animations

---

# Thank You!

**"The best way to predict the future is to create it."**

The same simple idea — predicting the next token — powers everything
from autocomplete to ChatGPT to Claude.

## Questions?
