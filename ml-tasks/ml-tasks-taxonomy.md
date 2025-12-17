---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 22px;
    padding: 30px;
    color: #333;
  }
  h1 { color: #2E86AB; font-size: 1.8em; margin-bottom: 0.1em; }
  h2 { color: #06A77D; font-size: 1.2em; margin-top: 0; }
  strong { color: #D62828; }
  .box {
    background: #f8f9fa;
    border-left: 6px solid #2E86AB;
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 4px;
  }
  .task-meta { font-family: monospace; font-size: 0.85em; color: #666; display: block; margin-bottom: 4px; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  img { max-width: 100%; height: auto; display: block; margin: 0 auto; box-shadow: 2px 2px 8px rgba(0,0,0,0.1); }
---

# The Machine Learning Taxonomy
## Organizing 40+ Tasks by their Mathematical Roots

**Nipun Batra**
*IIT Gandhinagar*

---

# The Grand Map

Machine Learning isn't just a list of random tricks. It's a family tree.
Almost every task is a variation of **Classification** or **Regression**.

*   **Predicting a Label?** Classification.
*   **Predicting a Number?** Regression.
*   **Predicting a Sequence?** Repeated Classification.
*   **Predicting a Structure?** Classification + Regression combined.

---

# Part 1: The Supervised Family

![w:600](diagrams/taxonomy_supervised.png)

---

# Branch A: "Is it X or Y?" (Classification)

The simplest form of ML. $f(x) \rightarrow \{0, 1, \dots, K\}$

<div class="columns">
<div>

<div class="box">
<strong>1. Image Classification</strong>
<span class="task-meta">Input: Pixels | Output: Label</span>
Is this image a Cat or Dog?
</div>

<div class="box">
<strong>2. Sentiment Analysis</strong>
<span class="task-meta">Input: Text | Output: Label</span>
Is this review Positive or Negative?
</div>

</div>
<div>

<div class="box">
<strong>3. Spam Detection</strong>
<span class="task-meta">Input: Email | Output: Binary</span>
Is this junk?
</div>

<div class="box">
<strong>4. Topic Classification</strong>
<span class="task-meta">Input: Document | Output: Category</span>
Is this news about Sports, Politics, or Tech?
</div>

</div>
</div>

---

# Branch B: "How Much?" (Regression)

Predicting continuous values. $f(x) \rightarrow \mathbb{R}$

<div class="columns">
<div>

<div class="box">
<strong>5. House Price Prediction</strong>
<span class="task-meta">Input: Features | Output: $ Price</span>
Classic regression.
</div>

<div class="box">
<strong>6. Age Estimation</strong>
<span class="task-meta">Input: Face Image | Output: Years</span>
Predicting 25.4 years vs 25 years.
</div>

</div>
<div>

<div class="box">
<strong>7. Time Series Forecasting</strong>
<span class="task-meta">Input: History | Output: Future Value</span>
Predicting tomorrow's temperature.
</div>

<div class="box">
<strong>8. Bounding Box Regression</strong>
<span class="task-meta">Input: Image | Output: (x, y, w, h)</span>
Predicting the coordinates of an object (Part of Detection).
</div>

</div>
</div>

---

# Part 2: The Vision Hierarchy
*Combining Classification + Regression*

![w:700](diagrams/taxonomy_vision.png)

---

# Level 2: "Where is it?" (Detection)

We combine **Classification** (What) + **Regression** (Where).

<div class="columns">
<div>

<div class="box">
<strong>9. Object Detection</strong>
<span class="task-meta">Output: Box + Class</span>
"There is a Car at [10, 50, 200, 300]"
</div>

<div class="box">
<strong>10. Face Detection</strong>
<span class="task-meta">Output: Box</span>
Finding faces for auto-focus.
</div>

</div>
<div>

<div class="box">
<strong>11. Keypoint Detection (Pose)</strong>
<span class="task-meta">Output: (x,y) points</span>
Finding Elbows, Knees, Eyes. (Regression of 17 points).
</div>

<div class="box">
<strong>12. Text Detection (OCR)</strong>
<span class="task-meta">Output: Box around text</span>
Finding words in street signs.
</div>

</div>
</div>

---

# Level 3: "Which Pixels?" (Segmentation)

Now we classify **every single pixel**.

<div class="columns">
<div>

<div class="box">
<strong>13. Semantic Segmentation</strong>
<span class="task-meta">Output: Class per pixel</span>
Road, Sky, Tree (No distinction between two trees).
</div>

</div>
<div>

<div class="box">
<strong>14. Instance Segmentation</strong>
<span class="task-meta">Output: Class + ID per pixel</span>
Car #1 vs Car #2.
</div>

</div>
</div>

<div class="box">
<strong>15. Image Matting</strong>
<span class="task-meta">Output: Alpha Matte (Transparency)</span>
Zoom background blur / Green screen removal.
</div>

---

# Part 3: The Sequence Family
*Predicting Lists of things*

![w:800](diagrams/taxonomy_sequence.png)

---

# Many-to-Many (Seq2Seq)

Standard Classification is Many-to-One.
Seq2Seq is **Text In, Text Out**.

<div class="columns">
<div>

<div class="box">
<strong>16. Machine Translation</strong>
<span class="task-meta">English $\rightarrow$ Hindi</span>
Mapping sequence to sequence.
</div>

<div class="box">
<strong>17. Text Summarization</strong>
<span class="task-meta">Long Text $\rightarrow$ Short Text</span>
Extracting key information.
</div>

</div>
<div>

<div class="box">
<strong>18. Speech Recognition (ASR)</strong>
<span class="task-meta">Audio Wave $\rightarrow$ Text</span>
Mapping sound frames to phonemes/words.
</div>

<div class="box">
<strong>19. Text-to-Speech (TTS)</strong>
<span class="task-meta">Text $\rightarrow$ Audio Wave</span>
The reverse of ASR.
</div>

</div>
</div>

---

# Token-Level Tasks (Tagging)

Classifying each token in a sequence (like Semantic Seg for text).

<div class="columns">
<div>

<div class="box">
<strong>20. Named Entity Recognition (NER)</strong>
<span class="task-meta">Output: [PER, LOC, ORG] per word</span>
Identifying "Sundar Pichai" as PER.
</div>

</div>
<div>

<div class="box">
<strong>21. Part-of-Speech Tagging</strong>
<span class="task-meta">Output: [Noun, Verb] per word</span>
Grammatical analysis.
</div>

</div>
</div>

---

# Part 4: Unsupervised & Generative
*Learning without Labels*

![w:800](diagrams/taxonomy_unsupervised.png)

---

# Grouping & Representation

<div class="columns">
<div>

<div class="box">
<strong>22. Clustering</strong>
<span class="task-meta">Task: Find Groups</span>
Customer segmentation.
</div>

<div class="box">
<strong>23. Topic Modeling</strong>
<span class="task-meta">Task: Find Themes</span>
Discovering "Sports" cluster in news without labels.
</div>

</div>
<div>

<div class="box">
<strong>24. Dimensionality Reduction</strong>
<span class="task-meta">Task: Compression</span>
PCA / t-SNE. Visualizing high-dim data.
</div>

<div class="box">
<strong>25. Anomaly Detection</strong>
<span class="task-meta">Task: Find Outliers</span>
Credit fraud, manufacturing defects.
</div>

</div>
</div>

---

# Generative Tasks (The New Wave)

Modeling the distribution $P(X)$.

<div class="columns">
<div>

<div class="box">
<strong>26. Image Generation</strong>
<span class="task-meta">Noise $\rightarrow$ Image</span>
GANs, Diffusion (Midjourney).
</div>

<div class="box">
<strong>27. Text Generation</strong>
<span class="task-meta">Prefix $\rightarrow$ Continuation</span>
LLMs (GPT).
</div>

</div>
<div>

<div class="box">
<strong>28. Inpainting</strong>
<span class="task-meta">Masked Image $\rightarrow$ Full Image</span>
Filling holes.
</div>

<div class="box">
<strong>29. Style Transfer</strong>
<span class="task-meta">Content + Style $\rightarrow$ Image</span>
Artistic filters.
</div>

</div>
</div>

---

# Part 5: The Complex Ones (Multimodal + RL)

<div class="columns">
<div>

<div class="box">
<strong>30. Visual QA (VQA)</strong>
<span class="task-meta">Image + Text $\rightarrow$ Text</span>
"What color is the car?"
</div>

<div class="box">
<strong>31. Image Captioning</strong>
<span class="task-meta">Image $\rightarrow$ Text</span>
"A dog running on grass."
</div>

</div>
<div>

<div class="box">
<strong>32. Reinforcement Learning</strong>
<span class="task-meta">State $\rightarrow$ Action</span>
Playing Chess, Robot Control.
</div>

<div class="box">
<strong>33. Recommendation</strong>
<span class="task-meta">User History $\rightarrow$ Item Rank</span>
Netflix/Amazon.
</div>

</div>
</div>

---

# Summary

We didn't just list tasks. We grouped them by their **Mathematical Nature**.

1.  **Classification:** The parent of Vision/NLP classification.
2.  **Regression:** The parent of Prediction/Bounding Boxes.
3.  **Seq2Seq:** The parent of Translation/Speech.
4.  **Generative:** The parent of GPT/DALL-E.

**Understanding the root helps you solve the leaf.**

## Questions?
