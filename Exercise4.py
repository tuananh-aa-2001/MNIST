"""
exercise4_multioutput.py — Multioutput Classification
=====================================================
Goal: Train a classifier that takes a noisy image as input and outputs a clean digit image.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from utils import load_mnist, plot_digits

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
X_train, X_test, y_train_labels, y_test_labels, _, _ = load_mnist()

# ─────────────────────────────────────────────
# 2. Add Noise to create Multioutput Targets
# ─────────────────────────────────────────────
np.random.seed(42)
noise_train = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise_train
noise_test = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise_test

# Targets are the original images
y_train_mod = X_train
y_test_mod = X_test

# ─────────────────────────────────────────────
# 3. Train KNN Classifier
# ─────────────────────────────────────────────
print("\n── Training KNN Classifier for Multioutput ──")
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)

# ─────────────────────────────────────────────
# 4. Make Predictions and Clean Image
# ─────────────────────────────────────────────
print("\n── Cleaning a Noisy Image ──")
some_index = 0

# The predict function expects a 2D array, so we slice exactly one element
clean_digit = knn_clf.predict(X_test_mod[some_index:some_index+1])

# Plot the noisy image and the cleaned image side by side
plt.figure(figsize=(8, 4))

plt.subplot(121)
plot_digits(X_test_mod[some_index:some_index+1])
plt.title("Noisy Image")

plt.subplot(122)
plot_digits(clean_digit)
plt.title("Cleaned Image (Prediction)")

plt.tight_layout()
os.makedirs("images/exercise4", exist_ok=True)
plt.savefig("images/exercise4/cleaned_digit.png", dpi=150)
plt.show()
print("  Saved → images/exercise4/cleaned_digit.png")
