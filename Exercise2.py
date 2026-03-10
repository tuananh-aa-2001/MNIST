"""
exercise2_augment.py — Data Augmentation via Image Shifting
============================================================
Goals:
  1. Write a shift() function that moves a 28×28 MNIST image
     1 pixel in any direction (up, down, left, right)
  2. Augment the training set: each image → 4 shifted copies
     (60,000 → 300,000 training samples)
  3. Retrain the best KNN model on the expanded dataset
  4. Compare accuracy to Exercise 1's baseline
"""

import numpy as np
from scipy.ndimage import shift as scipy_shift
from sklearn.neighbors import KNeighborsClassifier
from utils import load_mnist


# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test, _, _ = load_mnist()


# ─────────────────────────────────────────────
# 2. Image Shift Function
# ─────────────────────────────────────────────

def shift_image(image, dx, dy):
    """
    Shift a flat 784-pixel MNIST image by (dx, dy) pixels.

    Parameters
    ----------
    image : 1-D array of length 784
    dx    : horizontal shift (+1 = right, -1 = left)
    dy    : vertical shift   (+1 = down,  -1 = up)

    Returns
    -------
    Shifted image as a flat 1-D array of length 784
    """
    image_2d = image.reshape(28, 28)                      # Reshape to 2D
    shifted  = scipy_shift(image_2d, shift=[dy, dx], cval=0)  # cval=0 fills with black
    return shifted.reshape([-1])                          # Flatten back to 1D


# ─────────────────────────────────────────────
# 3. Augment the Training Set
# ─────────────────────────────────────────────

print("\n── Augmenting Training Set ──")
print(f"Original size: {X_train.shape[0]:,} samples")

directions = [
    ( 1,  0),   # right
    (-1,  0),   # left
    ( 0,  1),   # down
    ( 0, -1),   # up
]

X_augmented = [X_train]
y_augmented = [y_train]

for dx, dy in directions:
    shifted = np.apply_along_axis(shift_image, axis=1, arr=X_train, dx=dx, dy=dy)
    X_augmented.append(shifted)
    y_augmented.append(y_train)

X_train_aug = np.concatenate(X_augmented)
y_train_aug = np.concatenate(y_augmented)

# Shuffle so classes are evenly distributed during training
shuffle_idx   = np.random.permutation(len(X_train_aug))
X_train_aug   = X_train_aug[shuffle_idx]
y_train_aug   = y_train_aug[shuffle_idx]

print(f"Augmented size: {X_train_aug.shape[0]:,} samples (5× original)")


# ─────────────────────────────────────────────
# 4. Train Best KNN on Augmented Data
#    (use best params from Exercise 1 grid search)
# ─────────────────────────────────────────────

print("\n── Training KNN on Augmented Dataset ──")

# Best params found in Exercise 1 — update if your grid search found different values
BEST_PARAMS = {"n_neighbors": 4, "weights": "distance"}
print(f"Using params: {BEST_PARAMS}")

knn_aug = KNeighborsClassifier(**BEST_PARAMS, n_jobs=-1)
knn_aug.fit(X_train_aug, y_train_aug)


# ─────────────────────────────────────────────
# 5. Evaluate and Compare
# ─────────────────────────────────────────────

print("\n── Results ──")
aug_accuracy = knn_aug.score(X_test, y_test)
print(f"Augmented model test accuracy : {aug_accuracy:.4f}")

# Baseline from Exercise 1 (hardcoded for comparison — replace with actual result)
BASELINE_ACCURACY = 0.9714
print(f"Baseline (Exercise 1) accuracy: {BASELINE_ACCURACY:.4f}")
delta = aug_accuracy - BASELINE_ACCURACY
print(f"Improvement                   : {delta:+.4f} ({delta * 100:+.2f}%)")

if aug_accuracy > BASELINE_ACCURACY:
    print("✓ Data augmentation improved accuracy!")
else:
    print("✗ No improvement — check best params or augmentation logic.")