"""
utils.py — Shared helpers for MNIST exercises
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_mnist():
    """Fetch MNIST and return train/test splits with binary (5 vs not-5) targets."""
    print("Fetching MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"].astype(np.uint8)

    # Standard MNIST split: first 60k train, last 10k test
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Binary targets for "5 vs not-5"
    y_train_5 = (y_train == 5)
    y_test_5  = (y_test  == 5)

    print(f"  Training set : {X_train.shape[0]:,} samples")
    print(f"  Test set     : {X_test.shape[0]:,} samples")
    return X_train, X_test, y_train, y_test, y_train_5, y_test_5


# ─────────────────────────────────────────────
# Plot: Precision & Recall vs Threshold (Fig 3-4)
# ─────────────────────────────────────────────

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, highlight_threshold=None, filepath="precision_recall_vs_threshold.png"):
    """
    Plot Precision and Recall as functions of the decision threshold.

    Parameters
    ----------
    precisions          : array from precision_recall_curve()
    recalls             : array from precision_recall_curve()
    thresholds          : array from precision_recall_curve()
    highlight_threshold : optional float — draws a vertical line at this value
    filepath            : string optional — where to save the figure
    """
    plt.figure(figsize=(9, 4))
    plt.plot(thresholds, precisions[:-1], "b--", linewidth=2, label="Precision")
    plt.plot(thresholds, recalls[:-1],    "g-",  linewidth=2, label="Recall")

    if highlight_threshold is not None:
        # Find closest index
        idx = np.argmin(np.abs(thresholds - highlight_threshold))
        plt.plot(thresholds[idx], precisions[idx], "bo", markersize=8)
        plt.plot(thresholds[idx], recalls[idx],    "go", markersize=8)
        plt.axvline(x=highlight_threshold, color="gray", linestyle=":", linewidth=1.5,
                    label=f"Threshold = {highlight_threshold}")

    plt.xlabel("Threshold", fontsize=13)
    plt.ylabel("Score", fontsize=13)
    plt.title("Precision and Recall vs Decision Threshold", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.show()
    print(f"  Saved → {filepath}")


# ─────────────────────────────────────────────
# Plot: ROC Curve (Fig 3-6)
# ─────────────────────────────────────────────

def plot_roc_curve(fpr, tpr, label=None, compare_fpr=None, compare_tpr=None, compare_label=None, filepath="roc_curve.png"):
    """
    Plot the ROC curve (TPR vs FPR).

    Optionally overlay a second curve for comparison (e.g. Random Forest vs SGD).

    Parameters
    ----------
    fpr, tpr             : arrays from roc_curve()
    label                : legend label for the primary curve
    compare_fpr/tpr      : optional second curve arrays
    compare_label        : legend label for the second curve
    filepath             : string optional — where to save the figure
    """
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, "b-", linewidth=2, label=label or "Classifier")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")  # Dashed diagonal

    if compare_fpr is not None and compare_tpr is not None:
        plt.plot(compare_fpr, compare_tpr, "r-", linewidth=2,
                 label=compare_label or "Comparison Classifier")

    plt.xlabel("False Positive Rate (1 – Specificity)", fontsize=13)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=13)
    plt.title("ROC Curve", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.4)
    plt.axis([0, 1, 0, 1])
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.show()
    print(f"  Saved → {filepath}")


# ─────────────────────────────────────────────
# Plot: Confusion Matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(matrix, title="Confusion Matrix", filepath="confusion_matrix.png"):
    """Display a confusion matrix as a heatmap."""
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=14)
    plt.colorbar()
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.show()
    print(f"  Saved → {filepath}")


# ─────────────────────────────────────────────
# Plot: Digits Grid
# ─────────────────────────────────────────────

def plot_digits(instances, images_per_row=10, options=None):
    """Plot a grid of digit images on the current axes."""
    if not instances.size:
        return
    if options is None:
        options = {}
    
    # Each image in MNIST is 28x28
    size = 28
    images_per_row = min(len(instances), images_per_row)
    n_rows = (len(instances) - 1) // images_per_row + 1
    
    # Fill remaining spots with blank images
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)
    
    # Create the grids
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size, images_per_row * size)
    
    plt.imshow(big_image, cmap="binary", **options)
    plt.axis("off")