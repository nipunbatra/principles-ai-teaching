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
  .task-card {
    background: #f8f9fa;
    border-left: 6px solid #2E86AB;
    padding: 15px;
    margin-bottom: 15px;
    border-radius: 4px;
    box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
  }
  .task-title { font-weight: bold; font-size: 1.1em; color: #2E86AB; }
  .io-line { font-family: monospace; font-size: 0.9em; color: #555; background: #e9ecef; padding: 2px 6px; border-radius: 3px; }
  .example { font-style: italic; color: #6c757d; font-size: 0.9em; margin-top: 5px; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
---

# The Machine Learning Task Zoo
## A Tour of 40+ Supervised & Unsupervised Problems

**Nipun Batra**
*IIT Gandhinagar*

---

# What is a "Task"?

In ML, a task is defined by its **Input ($X$)** and **Output ($Y$)**.

*   **Supervised:** We have pairs of $(X, Y)$. (e.g., Image -> "Cat")
*   **Unsupervised:** We only have $X$. (e.g., Image -> ???)
*   **Reinforcement:** We have State -> Action -> Reward.

Let's explore the ecosystem!

---

# Domain 1: Computer Vision (Seeing)
*From simple labels to pixel-perfect understanding.*

---

# CV 1: The Basics

![w:1000 center](diagrams/task_cv_types.png)

1.  **Image Classification**: Is there a cat?
2.  **Object Detection**: Where is the cat?
3.  **Semantic Segmentation**: Which pixels are "cat"?

---

# CV 2: Advanced Segmentation

<div class="columns">
<div>

<div class="task-card">
<div class="task-title">4. Instance Segmentation</div>
<div class="io-line">Input: Image → Output: Mask per Object</div>
<div class="example">Distinguishing separate people in a crowd, not just "person" pixels.</div>
</div>

<div class="task-card">
<div class="task-title">5. Panoptic Segmentation</div>
<div class="io-line">Input: Image → Output: Stuff + Things</div>
<div class="example">Combines Semantic (Sky, Road) + Instance (Car #1, Car #2). Crucial for Self-Driving.</div>
</div>

</div>
<div>

<div class="task-card">
<div class="task-title">6. Pose Estimation</div>
<div class="io-line">Input: Image → Output: Keypoints (Skeleton)</div>
<div class="example">Yoga apps, Kinect games, Sports analysis.</div>
</div>

</div>
</div>

---

# CV 3: Beyond 2D Labels

<div class="columns">
<div>

<div class="task-card">
<div class="task-title">7. Depth Estimation</div>
<div class="io-line">Input: RGB Image → Output: Depth Map</div>
<div class="example">Estimating distance from a single camera (Monocular Depth).</div>
</div>

<div class="task-card">
<div class="task-title">8. Optical Flow</div>
<div class="io-line">Input: Video Frames → Output: Motion Vectors</div>
<div class="example">Tracking how pixels move between frames.</div>
</div>

</div>
<div>

<div class="task-card">
<div class="task-title">9. Face Recognition</div>
<div class="io-line">Input: Face Crop → Output: Person ID</div>
<div class="example">Unlocking your iPhone, Airport security.</div>
</div>

<div class="task-card">
<div class="task-title">10. Visual QA (VQA)</div>
<div class="io-line">Input: Image + Question → Output: Answer</div>
<div class="example">Q: "What color is the shirt?" A: "Red".</div>
</div>

</div>
</div>

---

# Domain 2: Natural Language Processing (Reading)
*Understanding, Translating, and Generating Text.*

---

# NLP 1: Classification & Tagging

<div class="columns">
<div>

<div class="task-card">
<div class="task-title">11. Sentiment Analysis</div>
<div class="io-line">Input: Text → Output: Positive/Negative</div>
<div class="example">"This movie was terrible" → Negative.</div>
</div>

<div class="task-card">
<div class="task-title">12. Topic Classification</div>
<div class="io-line">Input: Document → Output: Category</div>
<div class="example">Gmail sorting emails into "Promotions", "Social", "Primary".</div>
</div>

</div>
<div>

<div class="task-card">
<div class="task-title">13. Part-of-Speech (POS) Tagging</div>
<div class="io-line">Input: Sentence → Output: Verb/Noun tags</div>
<div class="example">"Time(N) flies(V) like(P) an arrow(N)."</div>
</div>

</div>
</div>

---

# NLP 2: Information Extraction

![w:1000 center](diagrams/task_nlp_ner.png)

<div class="task-card">
<div class="task-title">14. Named Entity Recognition (NER)</div>
<div class="io-line">Input: Text → Output: Spans with types</div>
<div class="example">Extracting Dates, Prices, People from contracts.</div>
</div>

---

# NLP 3: Sequence to Sequence (Seq2Seq)

<div class="columns">
<div>

<div class="task-card">
<div class="task-title">15. Machine Translation</div>
<div class="io-line">Input: English → Output: Hindi</div>
<div class="example">Google Translate. "Hello" → "Namaste".</div>
</div>

<div class="task-card">
<div class="task-title">16. Text Summarization</div>
<div class="io-line">Input: Long Article → Output: Short Summary</div>
<div class="example">TL;DR bots, News aggregators.</div>
</div>

</div>
<div>

<div class="task-card">
<div class="task-title">17. Question Answering (QA)</div>
<div class="io-line">Input: Context + Query → Output: Answer Span</div>
<div class="example">Google Search Snippets.</div>
</div>

</div>
</div>

---

# NLP 4: Deep Understanding

<div class="columns">
<div>

<div class="task-card">
<div class="task-title">18. Natural Language Inference (NLI)</div>
<div class="io-line">Input: Premise + Hypothesis → Output: Entailment?</div>
<div class="example">P: "Man playing soccer." H: "Man is outside." → True.</div>
</div>

</div>
<div>

<div class="task-card">
<div class="task-title">19. Coreference Resolution</div>
<div class="io-line">Input: Text → Output: Cluster mentions</div>
<div class="example">"<strong>Elon</strong> bought Twitter. <strong>He</strong> changed the logo." (He = Elon).</div>
</div>

</div>
</div>

---

# Domain 3: Audio & Speech (Hearing)

<div class="columns">
<div>

<div class="task-card">
<div class="task-title">20. Speech-to-Text (ASR)</div>
<div class="io-line">Input: Waveform → Output: Text</div>
<div class="example">Siri, Alexa, YouTube Captions.</div>
</div>

<div class="task-card">
<div class="task-title">21. Text-to-Speech (TTS)</div>
<div class="io-line">Input: Text → Output: Waveform</div>
<div class="example">GPS Navigation voices, Screen readers.</div>
</div>

</div>
<div>

<div class="task-card">
<div class="task-title">22. Speaker Identification</div>
<div class="io-line">Input: Voice Clip → Output: Person ID</div>
<div class="example">"Voice Match" to unlock phones.</div>
</div>

<div class="task-card">
<div class="task-title">23. Music Generation</div>
<div class="io-line">Input: Genre/Lyrics → Output: Song</div>
<div class="example">Suno AI, Udio.</div>
</div>

</div>
</div>

---

# Domain 4: Unsupervised Learning
*Finding patterns without labels.*

![w:900 center](diagrams/task_unsupervised.png)

---

# Unsupervised Tasks

<div class="columns">
<div>

<div class="task-card">
<div class="task-title">24. Clustering</div>
<div class="io-line">Input: Data points → Output: Groups</div>
<div class="example">Customer Segmentation (High spenders vs Browsers).</div>
</div>

<div class="task-card">
<div class="task-title">25. Anomaly Detection</div>
<div class="io-line">Input: Data → Output: Outlier Score</div>
<div class="example">Credit Card Fraud detection, Factory defect detection.</div>
</div>

</div>
<div>

<div class="task-card">
<div class="task-title">26. Dimensionality Reduction</div>
<div class="io-line">Input: High Dim (1000) → Output: Low Dim (2)</div>
<div class="example">PCA, t-SNE. Visualizing complex data in 2D.</div>
</div>

<div class="task-card">
<div class="task-title">27. Association Rule Mining</div>
<div class="io-line">Input: Transactions → Output: Rules</div>
<div class="example">"People who buy Diapers also buy Beer."</div>
</div>

</div>
</div>

---

# Domain 5: Generative & Self-Supervised
*Creating new data & Learning from itself.*

---

# Generative Tasks

<div class="columns">
<div>

<div class="task-card">
<div class="task-title">28. Image Generation</div>
<div class="io-line">Input: Noise/Text → Output: Image</div>
<div class="example">Midjourney, DALL-E (Diffusion Models).</div>
</div>

<div class="task-card">
<div class="task-title">29. Image Inpainting</div>
<div class="io-line">Input: Masked Image → Output: Full Image</div>
<div class="example">Removing tourists from vacation photos.</div>
</div>

</div>
<div>

<div class="task-card">
<div class="task-title">30. Style Transfer</div>
<div class="io-line">Input: Photo + Painting → Output: Stylized Photo</div>
<div class="example">Prisma app (Make my photo look like Van Gogh).</div>
</div>

<div class="task-card">
<div class="task-title">31. Super Resolution</div>
<div class="io-line">Input: Low Res → Output: High Res</div>
<div class="example">4K Upscaling on TVs, Restoring old photos.</div>
</div>

</div>
</div>

---

# Inpainting Visualized

![w:700 center](diagrams/task_inpainting.png)

---

# Self-Supervised Learning (The "Secret Sauce")

These tasks create labels from the data itself!

<div class="columns">
<div>

<div class="task-card">
<div class="task-title">32. Masked Language Modeling</div>
<div class="io-line">Input: "Hello [MASK] world" → Output: "my"</div>
<div class="example">How **BERT** is trained. Fill in the blanks.</div>
</div>

</div>
<div>

<div class="task-card">
<div class="task-title">33. Next Token Prediction</div>
<div class="io-line">Input: "Hello my" → Output: "world"</div>
<div class="example">How **GPT** is trained. Predict the future.</div>
</div>

</div>
</div>

<div class="task-card">
<div class="task-title">34. Contrastive Learning</div>
<div class="io-line">Input: Two augmented images → Output: Same/Different</div>
<div class="example">SimCLR. Learning visual features without labels.</div>
</div>

---

# Domain 6: Tabular & RL
*Numbers and Agents.*

<div class="columns">
<div>

<div class="task-card">
<div class="task-title">35. Regression</div>
<div class="io-line">Input: Features → Output: Number</div>
<div class="example">Predicting House Prices, Stock prices.</div>
</div>

<div class="task-card">
<div class="task-title">36. Time-Series Forecasting</div>
<div class="io-line">Input: History → Output: Future</div>
<div class="example">Weather prediction, Sales forecasting.</div>
</div>

</div>
<div>

<div class="task-card">
<div class="task-title">37. Recommendation</div>
<div class="io-line">Input: User History → Output: Item Ranking</div>
<div class="example">Netflix "Top picks for you", Amazon products.</div>
</div>

<div class="task-card">
<div class="task-title">38. Reinforcement Learning</div>
<div class="io-line">Input: State → Output: Action</div>
<div class="example">AlphaGo, Robots learning to walk.</div>
</div>

</div>
</div>

---

# RL Loop Visualized

![w:700 center](diagrams/task_rl.png)

---

# Summary

We covered **38 different tasks**!

*   **Supervised:** You have the answer key (Labels). Most common in industry.
*   **Unsupervised:** You explore the data structure. Good for analytics.
*   **Self-Supervised:** The data is its own label. The engine behind LLMs.
*   **Reinforcement:** Learning by trial and error.

**Pick a task, find a dataset, and start building!**

## Questions?
