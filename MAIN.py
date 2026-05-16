
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# =====================================================
# Create results directory
# =====================================================
os.makedirs("results", exist_ok=True)

# =====================================================
# Load Dataset
# =====================================================
data = load_breast_cancer()
X = data.data
y = data.target

print("Dataset Shape:", X.shape)
print("Classes:", np.unique(y))

# =====================================================
# Split Dataset
# =====================================================
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print("Training Samples:", X_train.shape[0])
print("Validation Samples:", X_val.shape[0])
print("Testing Samples:", X_test.shape[0])

# =====================================================
# Feature Scaling
# =====================================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# =====================================================
# Data Augmentation
# Add Gaussian Noise to training data only
# =====================================================
noise_factor = 0.05
num_augments = 1

augmented_samples = [X_train]
augmented_labels = [y_train]

for _ in range(num_augments):
    noise = np.random.normal(
        loc=0.0,
        scale=noise_factor,
        size=X_train.shape
    )

    X_train_augmented = X_train + noise

    augmented_samples.append(X_train_augmented)
    augmented_labels.append(y_train)

X_train_final = np.vstack(augmented_samples)
y_train_final = np.hstack(augmented_labels)

print("Augmented Training Shape:", X_train_final.shape)

# =====================================================
# Convert Data to PyTorch Tensors
# =====================================================
X_train = torch.FloatTensor(X_train_final)
y_train = torch.FloatTensor(y_train_final).view(-1, 1)

X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val).view(-1, 1)

X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# =====================================================
# Experiment Configurations
# =====================================================
experiments = {
    "Experiment_1": {
        "activation": nn.ReLU(),
        "learning_rate": 0.001,
        "hidden1": 32,
        "hidden2": 16
    },

    "Experiment_2": {
        "activation": nn.Tanh(),
        "learning_rate": 0.001,
        "hidden1": 64,
        "hidden2": 32
    }
}

# =====================================================
# Store Final Results
# =====================================================
results_summary = []

# =====================================================
# Run Experiments
# =====================================================
for exp_name, config in experiments.items():

    print("\n" + "=" * 50)
    print(f"Running {exp_name}")
    print("=" * 50)

    # =================================================
    # Build MLP Model
    # =================================================
    model = nn.Sequential(
        nn.Linear(30, config["hidden1"]),
        nn.BatchNorm1d(config["hidden1"]),
        config["activation"],
        nn.Dropout(0.3),

        nn.Linear(config["hidden1"], config["hidden2"]),
        nn.BatchNorm1d(config["hidden2"]),
        config["activation"],
        nn.Dropout(0.3),

        nn.Linear(config["hidden2"], 1),
        nn.Sigmoid()
    )

    # =================================================
    # Loss Function and Optimizer
    # =================================================
    criterion = nn.BCELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.5
    )

    # =================================================
    # Training Variables
    # =================================================
    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')
    patience = 10
    counter = 0

    best_model = copy.deepcopy(model.state_dict())

    # =================================================
    # Training Loop
    # =================================================
    for epoch in range(100):

        # =============================================
        # TRAINING
        # =============================================
        model.train()

        train_outputs = model(X_train)

        train_loss = criterion(train_outputs, y_train)

        optimizer.zero_grad()

        train_loss.backward()

        optimizer.step()

        # =============================================
        # TRAIN ACCURACY
        # =============================================
        train_preds = (train_outputs > 0.5).float()

        train_acc = (
            (train_preds == y_train).sum() / y_train.shape[0]
        ).item()

        # =============================================
        # VALIDATION
        # =============================================
        model.eval()

        with torch.no_grad():

            val_outputs = model(X_val)

            val_loss = criterion(val_outputs, y_val)

            val_preds = (val_outputs > 0.5).float()

            val_acc = (
                (val_preds == y_val).sum() / y_val.shape[0]
            ).item()

        # =============================================
        # Save Metrics
        # =============================================
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(
            f"Epoch [{epoch+1}/100] | "
            f"Train Loss: {train_loss.item():.4f} | "
            f"Val Loss: {val_loss.item():.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        scheduler.step()

        # =============================================
        # Early Stopping
        # =============================================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model = copy.deepcopy(model.state_dict())

        else:
            counter += 1

        if counter >= patience:
            print("Early Stopping Triggered")
            break

    # =================================================
    # Load Best Model
    # =================================================
    model.load_state_dict(best_model)

    # =================================================
    # Testing Phase
    # =================================================
    model.eval()

    with torch.no_grad():

        test_outputs = model(X_test)

        test_loss = criterion(test_outputs, y_test)

        test_preds = (test_outputs > 0.5).float()

        test_acc = (
            (test_preds == y_test).sum() / y_test.shape[0]
        ).item()

    # =================================================
    # Print Final Results
    # =================================================
    print("\nFinal Test Results")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Final Test Loss: {test_loss.item():.4f}")

    # =================================================
    # Save Results Summary
    # =================================================
    results_summary.append({
        "Experiment": exp_name,
        "Activation": config["activation"].__class__.__name__,
        "Learning Rate": config["learning_rate"],
        "Hidden Layers": f'{config["hidden1"]} -> {config["hidden2"]}',
        "Test Accuracy": round(test_acc, 4),
        "Test Loss": round(test_loss.item(), 4)
    })

    # =================================================
    # Plot Loss Curves
    # =================================================
    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.title(f"Loss Curves - {exp_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()

    plt.savefig(f"results/loss_curve_{exp_name}.png")

    plt.close()

    # =================================================
    # Plot Accuracy Curves
    # =================================================
    plt.figure(figsize=(8, 5))

    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")

    plt.title(f"Accuracy Curves - {exp_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()

    plt.savefig(f"results/accuracy_curve_{exp_name}.png")

    plt.close()

# =====================================================
# Final Comparison Table
# =====================================================
print("\n" + "=" * 60)
print("FINAL EXPERIMENT COMPARISON")
print("=" * 60)

for result in results_summary:

    print(f"Experiment: {result['Experiment']}")
    print(f"Activation Function: {result['Activation']}")
    print(f"Learning Rate: {result['Learning Rate']}")
    print(f"Hidden Layers: {result['Hidden Layers']}")
    print(f"Test Accuracy: {result['Test Accuracy']}")
    print(f"Test Loss: {result['Test Loss']}")
    print("-" * 50)
