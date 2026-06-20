# Unified Defect Detection: An Enhanced Approach for Mixed-Type Wafer Defects

An intelligent, scalable, and modular Deep Learning framework built to automatically identify and categorize single-type and complex mixed-type wafer map defects in semiconductor manufacturing. This framework addresses real-world fabrication challenges, including visual noise robustness, multi-class compound signatures, and severe class imbalance.

## Overview

As semiconductor nodes shrink, manual and classical rule-based inspections fall short in handling mixed-type anomalies across wafer maps. This repository contains a modular pipeline that leverages **Hybrid Architecture (CNN + Vision Transformer)** alongside standard baselines to prioritize localized structural details while retaining a holistic global context.

### Key Framework Highlights

* **Hybrid Intelligence Core:** Bridges localized texture extraction (CNN) with long-range dependency modeling (Vision Transformers).


* **Noise Resiliency:** Implements advanced preprocessing filters and multi-type data augmentation schemas designed for industrial environments.


* **Rigorous Benchmarking:** Validated against 7 state-of-the-art computer vision models (ResNet, DenseNet, VGGNet, EfficientNet, etc.) using standard metrics.

## Dataset Profile

The architecture uses the public **MixedWM38** dataset, consisting of over 38,000 wafer map images categorized into 38 distinct defect configurations (1 normal pattern, 8 single patterns, and 29 compound/mixed defect patterns).

### Dataset Attributes

| Attribute | Configuration Details |
| --- | --- |
| **Total Samples** | 38,015 wafer maps

 |
| **Image Dimensions** | $52 \times 52$ pixels (Grayscale representations)

 |
| **`arr_0` Key Matrix** | `0`: Blank spot | `1`: Passed die | `2`: Broken die

 |
| **`arr_1` Key Labels** | 8-dimensional One-Hot encoded foundational classes

 |

## Installation & Environment Setup

Ensure you have python 3.8+ installed. Clone the repository and install the relevant dependencies.

```bash
# Clone the repository
git clone https://github.com/your-username/Unified-Wafer-Defect-Detection.git
cd Unified-Wafer-Defect-Detection

# Install essential libraries
pip install numpy pandas matplotlib scikit-learn tensorflow keras torch torchvision

```

## Implementation Walkthrough

### 1. Data Loading and Inspection

Load the public `.npz` wafer matrix structures:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = np.load("Wafer_Map_Datasets.npz")
images, labels = data['arr_0'], data['arr_1']

print(f"Loaded {len(images)} wafer map matrices.") # Expected: 38015

# Visualize a sample defect signature
plt.imshow(images[2], cmap='gray')
plt.title(f"Defect Matrix Class Vector: {labels[2]}")
plt.show()

```

### 2. Standardized Deep Training Splits

To protect metrics from data leakage, the pipeline enforces a stratified allocation scheme:

```python
from sklearn.model_selection import train_test_split

# Split Schema: 70% Train, 15% Validation, 15% Evaluation
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.30, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

```

## Architectural Design Performance Summary

Through rigorous exploration, the **Hybrid CNN-ViT Architecture** outperformed classical baseline frameworks:

### Evaluation Metrics Across Models (Test Set Validation)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Performance Notes |
| --- | --- | --- | --- | --- | --- |
| **Proposed Hybrid (CNN-ViT)** | **0.86** | **0.92** | **0.86** | **0.87** | **Optimal balance of global and local features**.

 |
| ResNet | 0.82 | 0.90 | 0.82 | 0.84 | High precision but lower defect sensitivity.

 |
| VGGNet | 0.80 | 0.90 | 0.80 | 0.83 | Highly symmetrical, lacks long-range attention.

 |
| Baseline CNN | 0.62 | 0.87 | 0.62 | 0.69 | Struggles with rare compound patterns.

 |
| EfficientNet | 0.34 | 0.41 | 0.34 | 0.31 | Severe underfitting on structural representations.

 |
| DenseNet | 0.16 | 0.14 | 0.16 | 0.10 | Vulnerable to structural parameter mismatches.

 |
