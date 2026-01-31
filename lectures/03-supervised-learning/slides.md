---
marp: true
theme: iitgn-modern
paginate: true
math: mathjax
---

<!-- _class: title-slide -->
<!-- _paginate: false -->

# Supervised Learning
# Deep Dive

## Linear & Logistic Regression

**Nipun Batra** | IIT Gandhinagar

---

# Learning Goals

By the end of this lecture, you will:

1. **Understand** how linear regression finds the best line
2. **Interpret** weights and what they mean
3. **Apply** logistic regression for classification
4. **Use** the sigmoid function to get probabilities
5. **Evaluate** models with appropriate metrics

---

# Recap: Supervised Learning

**We have:**
- Features (X): What we know about each example
- Labels (y): What we want to predict

**Goal:** Learn a function f where f(X) ≈ y

| If y is... | Task | Example |
|------------|------|---------|
| A number | Regression | Predict house price |
| A category | Classification | Spam or not spam |

---

<!-- _class: section-divider -->

# Part 1: Linear Regression

## Fitting a Line to Data

---

# The Simplest Prediction Problem

**Scenario:** You're a real estate agent. A client asks:

> "I'm looking at a 1750 sqft house. What should I expect to pay?"

You have data from recent sales:

| Size (sqft) | Price (₹ lakhs) |
|-------------|-----------------|
| 1000 | 40 |
| 1500 | 60 |
| 2000 | 80 |
| 2500 | 100 |

**Can you see the pattern?**

---

# The Pattern is Clear!

Every 500 sqft adds ₹20 lakhs.

| Size | Price | Pattern |
|------|-------|---------|
| 1000 | 40 | |
| 1500 | 60 | +500 sqft → +₹20 lakhs |
| 2000 | 80 | +500 sqft → +₹20 lakhs |
| 2500 | 100 | +500 sqft → +₹20 lakhs |

**So 1750 sqft should cost... ₹70 lakhs!**

You just did linear regression in your head.

---

# The Equation of a Line

$$\text{Price} = 0.04 \times \text{Size} + 0$$

Or more generally:

$$\hat{y} = w \cdot x + b$$

| Symbol | Name | Meaning | Our Example |
|--------|------|---------|-------------|
| $x$ | Input | Feature value | Size (sqft) |
| $\hat{y}$ | Output | Predicted value | Price |
| $w$ | Weight | Slope | 0.04 |
| $b$ | Bias | Intercept | 0 |

---

# What Does the Weight Mean?

**Weight w = 0.04 means:**

> "For every 1 sqft increase, price increases by ₹0.04 lakhs"

Or equivalently:
> "For every 100 sqft increase, price increases by ₹4 lakhs"

<div class="insight">

The weight tells you the **sensitivity** — how much does output change when input changes?

</div>

---

# What Does the Bias Mean?

**Bias b = 0 means:**

> "A 0 sqft house would cost ₹0"

In reality, bias captures the **baseline** cost:
- Land value
- Permits and fees
- Minimum construction cost

If b = 10, then even a tiny house costs at least ₹10 lakhs.

---

# Linear Regression in Python

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Our data
X = np.array([[1000], [1500], [2000], [2500]])  # Size
y = np.array([40, 60, 80, 100])                  # Price

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Now predict!
model.predict([[1750]])  # → 70.0 (₹70 lakhs)
```

---

# Understanding What the Model Learned

```python
print(f"Weight (w): {model.coef_[0]}")      # 0.04
print(f"Intercept (b): {model.intercept_}")  # 0.0
```

**The equation it learned:**
$$\text{Price} = 0.04 \times \text{Size} + 0$$

**Verify:**
- 1750 sqft → 0.04 × 1750 + 0 = **₹70 lakhs** ✓

---

# But What if Data Isn't Perfect?

Real data has **noise** — points don't fall exactly on a line.

| Size | Actual Price | On the Line? |
|------|--------------|--------------|
| 1000 | 42 | No (line says 40) |
| 1500 | 58 | No (line says 60) |
| 2000 | 83 | No (line says 80) |
| 2500 | 97 | No (line says 100) |

**Which line is "best"?**

---

# The Goal: Minimize Errors

**Residual** = Actual - Predicted = $y - \hat{y}$

| Size | Actual | Predicted | Residual |
|------|--------|-----------|----------|
| 1000 | 42 | 40 | +2 |
| 1500 | 58 | 60 | -2 |
| 2000 | 83 | 80 | +3 |
| 2500 | 97 | 100 | -3 |

**Goal:** Find w and b that make residuals as small as possible!

---

# Why Squared Errors?

We minimize **Sum of Squared Errors**:

$$\text{SSE} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

| Why Square? | Reason |
|-------------|--------|
| Errors don't cancel | +3 and -3 both contribute |
| Penalizes big errors more | Error of 10 costs 100, not 10 |
| Has nice math properties | Can solve with calculus |

---

# Multiple Features

What if price depends on more than just size?

| Size | Bedrooms | Bathrooms | Price |
|------|----------|-----------|-------|
| 1500 | 3 | 2 | 60 |
| 2000 | 4 | 3 | 90 |
| 1200 | 2 | 1 | 45 |

**The equation becomes:**

$$\text{Price} = w_1 \times \text{Size} + w_2 \times \text{Beds} + w_3 \times \text{Baths} + b$$

---

# Interpreting Multiple Weights

```python
# After training on multiple features:
# coef_ = [0.03, 5.0, 8.0]
# intercept_ = -10
```

| Feature | Weight | Interpretation |
|---------|--------|----------------|
| Size (sqft) | 0.03 | +100 sqft → +₹3 lakhs |
| Bedrooms | 5.0 | +1 bedroom → +₹5 lakhs |
| Bathrooms | 8.0 | +1 bathroom → +₹8 lakhs |

<div class="insight">

Each weight shows that feature's **independent contribution** to price!

</div>

---

# When Does Linear Regression Work?

| Scenario | Works? | Why |
|----------|--------|-----|
| Price vs Size (roughly linear) | ✅ Yes | Points follow a line |
| Height vs Weight | ✅ Yes | Roughly linear |
| Study hours vs Exam score | ✅ Mostly | Roughly linear |
| Age vs Happiness | ❌ No | U-shaped relationship |

<div class="warning">

Linear regression assumes the relationship is a **straight line**!

</div>

---

<!-- _class: section-divider -->

# Part 2: Logistic Regression

## From Numbers to Categories

---

# A Different Problem

**Scenario:** You're building a spam filter.

| Email | Exclamation marks | Has "FREE" | Is Spam? |
|-------|-------------------|------------|----------|
| 1 | 5 | Yes | ✅ Spam |
| 2 | 0 | No | ❌ Not Spam |
| 3 | 3 | Yes | ✅ Spam |
| 4 | 1 | No | ❌ Not Spam |

**The output is a category, not a number!**

---

# Why Can't We Use Linear Regression?

If we use linear regression:

$$\text{Spam Score} = w_1 \times \text{Exclamations} + w_2 \times \text{HasFREE} + b$$

**Problem:** This gives any number (-∞ to +∞)

| Score | What does it mean? |
|-------|-------------------|
| -2.5 | ??? |
| 0.3 | ??? |
| 1.5 | ??? |
| 147 | ??? |

We need something between 0 and 1 (a probability)!

---

# The Sigmoid Function

**Solution:** Squash any number to range (0, 1)

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

| Input (z) | Output σ(z) | Interpretation |
|-----------|-------------|----------------|
| -10 | 0.00005 | Almost certainly NOT spam |
| -2 | 0.12 | Probably not spam |
| 0 | 0.50 | 50-50 |
| +2 | 0.88 | Probably spam |
| +10 | 0.99995 | Almost certainly spam |

---

# The Sigmoid Shape

The sigmoid is an **S-curve**:

| Region | Behavior |
|--------|----------|
| Very negative z | Output ≈ 0 |
| z near 0 | Output changes rapidly |
| Very positive z | Output ≈ 1 |

**Key insight:** It converts any number to a probability!

---

# Logistic Regression Model

**Two steps:**

1. **Linear:** Compute a score
   $$z = w_1 x_1 + w_2 x_2 + b$$

2. **Sigmoid:** Convert to probability
   $$P(\text{spam}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

---

# A Concrete Example

**Email features:** 5 exclamation marks, has "FREE"

**Learned weights:** $w_1 = 0.5$, $w_2 = 2.0$, $b = -1.0$

**Step 1: Linear score**
$$z = 0.5 \times 5 + 2.0 \times 1 + (-1.0) = 3.5$$

**Step 2: Sigmoid**
$$P(\text{spam}) = \sigma(3.5) = 0.97$$

**Decision:** 97% → This is spam!

---

# The Decision Rule

| If P(spam) | Decision |
|------------|----------|
| > 0.5 | Predict SPAM |
| ≤ 0.5 | Predict NOT SPAM |

**The threshold 0.5 is a hyperparameter** — you can adjust it!

| Threshold | Effect |
|-----------|--------|
| 0.5 | Balanced |
| 0.3 | Catch more spam (but more false alarms) |
| 0.7 | Fewer false alarms (but miss some spam) |

---

# Logistic Regression in Python

```python
from sklearn.linear_model import LogisticRegression

# Data: [exclamations, has_FREE]
X = [[5, 1], [0, 0], [3, 1], [1, 0], [4, 1], [0, 0]]
y = [1, 0, 1, 0, 1, 0]  # 1=spam, 0=not spam

# Train
model = LogisticRegression()
model.fit(X, y)

# Predict class
model.predict([[4, 1]])  # → [1] (spam)

# Predict probability
model.predict_proba([[4, 1]])  # → [[0.12, 0.88]]
#                                   [P(not spam), P(spam)]
```

---

# Understanding predict_proba

```python
probs = model.predict_proba([[4, 1]])
# → [[0.12, 0.88]]

print(f"P(not spam) = {probs[0][0]:.2f}")  # 0.12
print(f"P(spam) = {probs[0][1]:.2f}")      # 0.88
```

| Class | Probability |
|-------|-------------|
| Not Spam (class 0) | 12% |
| Spam (class 1) | 88% |

**Always sums to 1.0!**

---

# Interpreting the Weights

```python
print(f"Weights: {model.coef_}")      # [[0.8, 2.1]]
print(f"Intercept: {model.intercept_}") # [-1.5]
```

| Feature | Weight | Effect |
|---------|--------|--------|
| Exclamations | +0.8 | More ! → Higher spam probability |
| Has FREE | +2.1 | "FREE" present → Much higher spam probability |

**Positive weight** → Increases P(spam)
**Negative weight** → Decreases P(spam)

---

# The Intuition

**Why does this work?**

The linear part finds a **decision boundary**:

$$w_1 x_1 + w_2 x_2 + b = 0$$

Points on one side → Class 0
Points on other side → Class 1

The sigmoid tells us **how confident** we are.

---

<!-- _class: section-divider -->

# Part 3: Model Evaluation

## How Good is Our Model?

---

# The Golden Rule

<div class="insight">

**Always evaluate on data the model has never seen!**

</div>

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train ONLY on training data
model.fit(X_train, y_train)

# Evaluate ONLY on test data
predictions = model.predict(X_test)
```

---

# Regression Metrics

For regression (predicting numbers):

| Metric | Formula | Meaning |
|--------|---------|---------|
| **MSE** | $\frac{1}{n}\sum(y - \hat{y})^2$ | Average squared error |
| **RMSE** | $\sqrt{MSE}$ | Error in same units as y |
| **MAE** | $\frac{1}{n}\sum|y - \hat{y}|$ | Average absolute error |

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
```

---

# Classification Metrics

For classification (predicting categories):

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.1%}")  # e.g., 92.5%
```

**But accuracy can be misleading...**

---

# The Accuracy Trap

**Scenario:** Disease detection (1% have disease, 99% healthy)

| Model | Strategy | Accuracy |
|-------|----------|----------|
| Dumb | Always predict "healthy" | **99%** |
| Smart | Tries to detect disease | 95% |

<div class="warning">

The "dumb" model has 99% accuracy but **misses ALL sick patients!**

</div>

---

# The Confusion Matrix

|  | Predicted: No | Predicted: Yes |
|--|---------------|----------------|
| **Actual: No** | TN (Correct!) | FP (False Alarm) |
| **Actual: Yes** | FN (Missed!) | TP (Correct!) |

| Term | Meaning |
|------|---------|
| TP | True Positive — Correctly detected |
| TN | True Negative — Correctly ruled out |
| FP | False Positive — False alarm |
| FN | False Negative — Missed case |

---

# Precision and Recall

**Precision:** Of those I flagged, how many were correct?
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall:** Of all positives, how many did I find?
$$\text{Recall} = \frac{TP}{TP + FN}$$

| Scenario | Priority |
|----------|----------|
| Spam filter | High precision (don't lose important emails!) |
| Cancer screening | High recall (don't miss any cancer!) |

---

# F1 Score

**Balances precision and recall:**

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

| Precision | Recall | F1 |
|-----------|--------|-----|
| 0.90 | 0.90 | 0.90 |
| 0.95 | 0.50 | 0.65 |
| 0.50 | 0.95 | 0.65 |

**F1 punishes imbalance!**

---

# Complete Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 2. Train
model = LogisticRegression()
model.fit(X_train, y_train)

# 3. Predict and evaluate
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.1%}")
print(classification_report(y_test, predictions))
```

---

# Linear vs Logistic: Summary

| Aspect | Linear Regression | Logistic Regression |
|--------|-------------------|---------------------|
| **Output** | Any number | Probability (0 to 1) |
| **Task** | Regression | Classification |
| **Equation** | $\hat{y} = wx + b$ | $P = \sigma(wx + b)$ |
| **Example** | Predict price | Predict spam/not spam |
| **Metric** | MSE, RMSE, MAE | Accuracy, Precision, Recall |

---

# Key Takeaways

1. **Linear Regression** fits a line through data
   - Weight = sensitivity (how much output changes per unit input)
   - Minimize squared errors

2. **Logistic Regression** classifies using the sigmoid
   - Converts any score to probability (0-1)
   - Decision threshold (usually 0.5)

3. **Evaluation matters**
   - Always use test data
   - Accuracy isn't everything — consider precision/recall

---

<!-- _class: title-slide -->

# You Now Understand the Basics!

## Next: Neural Networks

**Lab:** Implement linear and logistic regression on real data

**Interactive Notebook:** [L03_supervised_learning.ipynb](../../lecture_demo/L03_supervised_learning.ipynb)

*"All models are wrong, but some are useful."*
— George Box

**Questions?**
