# Breast Cancer Diagnosis Using MLP Neural Network

## Project Description
This project implements a Multilayer Perceptron (MLP) Neural Network using PyTorch for breast cancer diagnosis.

The model classifies tumors into:
- Malignant
- Benign

The project includes:
- Data preprocessing
- Feature scaling
- Data augmentation
- Batch normalization
- Dropout regularization
- Early stopping
- Multiple experiments
- Performance visualization

---

# Dataset
Breast Cancer Wisconsin Dataset

Dataset Link:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

---

# Technologies Used
- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn

---

# Experiments

| Experiment | Activation Function | Hidden Layers | Test Accuracy | Test Loss |
|---|---|---|---|---|
| Experiment 1 | ReLU | 32 → 16 | 91.23% | 0.4553 |
| Experiment 2 | Tanh | 64 → 32 | 92.98% | 0.1995 |

---

# Best Model

Experiment 2 achieved the best performance because it produced:
- Higher accuracy
- Lower loss

---

# Regularization Techniques
The following techniques were used:
- Dropout
- Batch Normalization
- Data Augmentation
- Early Stopping

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
Training and validation curves are saved inside the `results` folder.

---

# Author
Ahmed
