# Lecture Demo Notebooks

Interactive Jupyter notebooks for the "Principles of AI" course. These notebooks are designed for:
- Live demonstrations during lectures
- Student follow-along and practice
- Self-paced learning and exploration

## Notebooks

| Lecture | Notebook | Topics Covered |
|---------|----------|----------------|
| L02 | `L02_data_foundation.ipynb` | Data types, features/labels, train/test split, sklearn API, clustering intro |
| L03 | `L03_supervised_learning.ipynb` | Linear regression, logistic regression, decision trees, K-NN, metrics |
| L04 | `L04_model_selection.ipynb` | Overfitting, bias-variance, cross-validation, grid search, ensembles |
| L05 | `L05_neural_networks.ipynb` | XOR problem, perceptrons, activations, MLPs, PyTorch basics |
| L06 | `L06_computer_vision.ipynb` | Images as pixels, convolution, CNNs, object detection, YOLO |
| L07 | `L07_language_models.ipynb` | Tokenization, next-token prediction, n-grams, embeddings, attention |
| L08 | `L08_generative_ai.ipynb` | Base model problem, SFT, RLHF, complete pipeline, ethics |

## Requirements

```bash
# Core requirements
pip install numpy matplotlib pandas scikit-learn

# For neural network notebooks (L05, L06)
pip install torch torchvision

# For computer vision demos (L06)
pip install ultralytics  # For YOLO

# For LLM demos (L08, optional)
pip install transformers
```

## Usage

1. **During Lecture**: Run cells incrementally to demonstrate concepts
2. **Student Practice**: Complete the exercise cells at the end of each notebook
3. **Self-Study**: Run through the entire notebook and experiment

## Features

Each notebook includes:
- **Clear explanations** with visual outputs
- **Real datasets** (Iris, MNIST, etc.)
- **Interactive visualizations** using matplotlib
- **Exercises** for hands-on practice
- **Summary tables** for quick reference

## Tips for Instructors

- Run notebooks fresh before each lecture (`Kernel > Restart & Clear Output`)
- Pre-run cells that download data or models
- Use exercises as in-class activities or homework
