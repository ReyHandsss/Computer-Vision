# EMNIST Handwritten Character Classification using HOG and SVM with LOOCV

## Description

This project implements handwritten character classification on the EMNIST (Extended MNIST) Letters dataset using:

* HOG (Histogram of Oriented Gradients) Feature Extraction
* SVM (Support Vector Machine) Classifier
* LOOCV (Leave-One-Out Cross Validation)

The program performs:

* Data preprocessing
* Feature extraction using HOG
* Character classification using SVM
* Hyperparameter tuning
* Performance evaluation using:

  * Accuracy
  * Precision
  * F1-Score
  * Confusion Matrix

---

# Dataset

Dataset used:

* EMNIST Letters Dataset

Dataset source:

* https://www.kaggle.com/datasets/crawford/emnist/data

The dataset contains:

* Handwritten English letters A-Z
* 28x28 grayscale images
* 26 classes

---

# Technologies Used

* Python
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Scikit-image
* MLXtend
* Joblib

---

# Methodology

## 1. Data Preprocessing

The EMNIST dataset is loaded from local IDX files.

Original labels:

* 1–26

Converted labels:

* 0–25

The dataset is reduced to:

* 20 samples per class

This is done to reduce computation time during LOOCV.

---

## 2. HOG Feature Extraction

HOG (Histogram of Oriented Gradients) is used to extract image features.

HOG parameters:

* orientations = 9
* pixels_per_cell = (4,4)
* cells_per_block = (2,2)
* block_norm = 'L2-Hys'

The extracted HOG features are used as input for the SVM classifier.

---

## 3. SVM Classification

Support Vector Machine (SVM) is used as the classifier.

Hyperparameter tuning:

* C = [1, 10]
* kernel = ['linear', 'rbf']
* gamma = ['scale']

---

## 4. LOOCV Evaluation

The evaluation method used is:

* Leave-One-Out Cross Validation (LOOCV)

LOOCV process:

* One sample becomes testing data
* Remaining samples become training data
* Repeated until all data have been tested

---

# Evaluation Metrics

The program evaluates:

* Accuracy
* Precision
* F1-Score
* Confusion Matrix

---

# Results

The model successfully classifies handwritten letters A-Z using HOG and SVM.

The confusion matrix shows:

* Correct predictions on the diagonal
* Misclassifications outside the diagonal

The model performance depends on:

* HOG parameters
* SVM parameters
* Number of samples used

---

# Project Structure

```bash
project/
│
├── dataset/
│   ├── emnist-letters-train-images-idx3-ubyte
│   ├── emnist-letters-train-labels-idx1-ubyte
│
├── emnist_hog_svm_loocv.ipynb
│
├── emnist_hog_svm_loocv.pkl
│
└── README.md
```

---

# Installation

Install required libraries:

```bash
pip install mlxtend scikit-image seaborn joblib
```

---

# Run the Program

Open Jupyter Notebook:

```bash
jupyter notebook
```

Run:

* `emnist_hog_svm_loocv.ipynb`

---

# Output

The program generates:

* HOG visualization
* Classification report
* Accuracy
* Precision
* F1-score
* Confusion matrix
* Predicted handwritten letters

---

# Author

Rayhan Ahmad
Robotics Engineering 

---

# Course

Computer Vision
Robotics Engineering
2026

---
