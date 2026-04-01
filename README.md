# Fashion-vision-model
# Fashion MNIST — Logistic Regression Classifier

A baseline image classification pipeline trained on the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset using logistic regression. Covers end-to-end steps: data inspection, visualisation, normalisation, model training, and evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Pipeline](#pipeline)
- [Results](#results)
- [Usage](#usage)

---

## Overview

This script establishes a simple supervised learning baseline for classifying 28×28 grayscale fashion images into one of 10 categories. It is intended as a starting point before graduating to more complex architectures (CNNs, etc.).

---

## Dataset

| Property | Detail |
|---|---|
| Source | Fashion MNIST (Zalando) |
| Image size | 28 × 28 pixels (flattened to 784 features) |
| Classes | 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot) |
| Input variable | `fashion_data` — a pandas DataFrame with a `label` column and 784 pixel columns |

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualisation |
| `scikit-learn` | Model training and evaluation |

---

## Pipeline

### 1. Preprocessing (`preprocessing()`)
- Reports missing values, column names, shape, dtypes, and summary statistics.

### 2. Label Distribution
- Counts and bar-plots the frequency of each class label to check for class imbalance.

### 3. Image Visualisation
- Renders a 2×5 grid of the first 10 images using a green colourmap alongside their labels.
- Prints the raw pixel matrix of the first image for manual inspection.

### 4. Normalisation
```python
X = X / 255.0  # Scale pixel values from [0, 255] to [0, 1]
```

### 5. Train / Validation Split
```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 6. Model Training
```python
FS_model = LogisticRegression()
FS_model.fit(X_train, y_train)
```

### 7. Evaluation
- **Accuracy score** — overall correct predictions
- **Classification report** — per-class precision, recall, and F1-score

---

## Results

Metrics are printed to stdout after training. Example output shape:

```
Model accuracy:
0.84

Classification report:
              precision    recall  f1-score   support
           0       0.82      0.85      0.83      1000
           ...
```

> Logistic regression on raw normalised pixels serves as a performance floor. Results will vary with solver settings and convergence tolerance.

---

## Usage

```python
import pandas as pd

# Load your dataset
fashion_data = pd.read_csv('fashion-mnist_train.csv')

# Run the script
# All steps execute sequentially once fashion_data is defined
```

> **Note:** The script expects `fashion_data` to be defined in the global scope before execution. Wrap the pipeline in a `main()` function or notebook cells for cleaner reuse.

---

## Potential Improvements

- Tune `LogisticRegression` hyperparameters (`C`, `solver`, `max_iter`)
- Add a confusion matrix heatmap for deeper error analysis
- Benchmark against a shallow neural network or SVM baseline
- Export the trained model with `joblib` for reuse
