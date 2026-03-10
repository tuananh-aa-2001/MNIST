"""
exercise3_multiclass.py — Multiclass Classification and Error Analysis
======================================================================
Goals:
  1. Train a Multiclass Classifier using SGD (One-vs-Rest)
  2. Implement Error Analysis by visualizing confusion matrices
  3. Visualize specific misclassifications (e.g. 3s as 5s vs actual 5s)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from utils import load_mnist, plot_confusion_matrix, plot_digits

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test, _, _ = load_mnist()

# ─────────────────────────────────────────────
# 2. Multiclass Classification (One-vs-Rest)
# ─────────────────────────────────────────────

print("\n── Multiclass SGD Classifier ──")
sgd_clf = SGDClassifier(random_state=42)
# Under the hood, sklearn detects multiclass `y_train` 
# and automatically trains 10 binary SGD classifiers (One-vs-Rest).
sgd_clf.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 3. Error Analysis: Confusion Matrix
# ─────────────────────────────────────────────

print("\n── Error Analysis ──")
# We use cross_val_predict for out-of-core evaluation directly on the train set
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)

conf_mx = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix:\n", conf_mx)

# Plot raw confusion matrix
os.makedirs("images/exercise3", exist_ok=True)
plot_confusion_matrix(conf_mx, title="Confusion Matrix (Raw Counts)", filepath="images/exercise3/confusion_matrix.png")

# Now let's focus on the errors. We divide each value by the number of images 
# in the corresponding class to compare error rates instead of absolute counts.
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

# Fill the diagonal with zeros to keep only the errors
np.fill_diagonal(norm_conf_mx, 0)

plt.figure(figsize=(8, 8))
plt.imshow(norm_conf_mx, interpolation="nearest", cmap=plt.cm.gray)
plt.title("Confusion Matrix (Errors Normalized)")
plt.colorbar()
plt.ylabel("True Label", fontsize=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.tight_layout()
plt.savefig("images/exercise3/normalized_confusion_matrix.png", dpi=150)
plt.show()
print("  Saved → images/exercise3/normalized_confusion_matrix.png")


# ─────────────────────────────────────────────
# 4. Error Analysis: Visualizing Misclassifications
# ─────────────────────────────────────────────

print("\n── Visualizing Missing Assignments (3s and 5s) ──")

# Example: The matrix usually shows a high error between 3s and 5s
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8, 8))

plt.subplot(221)
plot_digits(X_aa[:25], images_per_row=5)
plt.title(f"Actual {cl_a}, Pred {cl_a}")

plt.subplot(222)
plot_digits(X_ab[:25], images_per_row=5)
plt.title(f"Actual {cl_a}, Pred {cl_b}")

plt.subplot(223)
plot_digits(X_ba[:25], images_per_row=5)
plt.title(f"Actual {cl_b}, Pred {cl_a}")

plt.subplot(224)
plot_digits(X_bb[:25], images_per_row=5)
plt.title(f"Actual {cl_b}, Pred {cl_b}")

plt.tight_layout()
plt.savefig("images/exercise3/error_analysis_3_and_5.png", dpi=150)
plt.show()
print("  Saved → images/exercise3/error_analysis_3_and_5.png")
