# Breast Cancer Diagnosis Using MLP Neural Network

## Project Description
This project implements a Multilayer Perceptron (MLP) Neural Network using PyTorch for breast cancer diagnosis.

The model classifies tumors into:
- Malignant
- Benign

The project demonstrates:
- Data preprocessing
- Feature scaling
- Data augmentation
- Batch normalization
- Dropout regularization
- Early stopping
- Learning rate scheduling
- Multiple experiments
- Performance visualization

---

# Dataset
Dataset Used:
Breast Cancer Wisconsin Dataset

Dataset Source:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

---

# Technologies Used
- Python
- PyTorch
- Scikit-learn
- NumPy
- Matplotlib

---

# Experiments

| Experiment | Activation | Learning Rate | Hidden Layers |
|---|---|---|---|
| Experiment 1 | ReLU | 0.001 | 32 → 16 |
| Experiment 2 | Tanh | 0.001 | 64 → 32 |

---

# Performance Metrics
- Accuracy
- Binary Cross Entropy Loss

---

# How to Run

## Install Libraries

```bash
pip install -r requirements.txt
```

## Run Project

```bash
python main.py
```

---

# Results
The model achieved high classification accuracy on the test dataset.

Graphs are saved inside the `results` folder.
