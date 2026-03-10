"""
exercise1_knn.py — Binary Classification, Evaluation & KNN Grid Search
=======================================================================
Goals:
  1. Train an SGD binary classifier ("5 vs not-5")
  2. Evaluate with cross-validation, confusion matrix,
     Precision/Recall vs Threshold plot, and ROC curve
  3. Run GridSearchCV on KNeighborsClassifier to beat 97% accuracy
"""

import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)
from utils import (
    load_mnist,
    plot_precision_recall_vs_threshold,
    plot_roc_curve,
    plot_confusion_matrix,
)


# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test, y_train_5, y_test_5 = load_mnist()


# ─────────────────────────────────────────────
# 2. Train SGD Binary Classifier (5 vs not-5)
# ─────────────────────────────────────────────

print("\n── SGD Binary Classifier ──")
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Cross-validation accuracy
cv_scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print(f"Cross-val accuracy scores : {cv_scores}")
print(f"Mean accuracy             : {cv_scores.mean():.4f}")


# ─────────────────────────────────────────────
# 3. Confusion Matrix
# ─────────────────────────────────────────────

print("\n── Confusion Matrix ──")
os.makedirs("images/exercise1", exist_ok=True)
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)
plot_confusion_matrix(cm, title="SGD — Confusion Matrix (5 vs not-5)", filepath="images/exercise1/confusion_matrix.png")


# ─────────────────────────────────────────────
# 4. Precision / Recall vs Threshold (Fig 3-4)
# ─────────────────────────────────────────────

print("\n── Precision & Recall vs Threshold ──")
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

plot_precision_recall_vs_threshold(
    precisions, recalls, thresholds,
    highlight_threshold=0.0,   # Default SGD threshold is 0
    filepath="images/exercise1/precision_recall_vs_threshold.png"
)


# ─────────────────────────────────────────────
# 5. ROC Curve (Fig 3-6)
# ─────────────────────────────────────────────

print("\n── ROC Curve ──")
fpr, tpr, roc_thresholds = roc_curve(y_train_5, y_scores)
auc_score = roc_auc_score(y_train_5, y_scores)
print(f"SGD ROC AUC: {auc_score:.4f}")

plot_roc_curve(fpr, tpr, label=f"SGD (AUC = {auc_score:.3f})", filepath="images/exercise1/roc_curve.png")


# ─────────────────────────────────────────────
# 6. KNeighborsClassifier — Grid Search for >97%
# ─────────────────────────────────────────────

print("\n── KNN Grid Search ──")
knn_clf = KNeighborsClassifier()

param_grid = {
    "n_neighbors": [3, 4, 5],
    "weights"    : ["uniform", "distance"],
}

grid_search = GridSearchCV(
    knn_clf,
    param_grid,
    cv=3,
    scoring="accuracy",
    verbose=2,
    n_jobs=2,       # Use all CPU cores
)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters : {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

# Evaluate on the test set
best_knn = grid_search.best_estimator_
test_accuracy = best_knn.score(X_test, y_test)
print(f"Test set accuracy: {test_accuracy:.4f}")

if test_accuracy >= 0.97:
    print("✓ Target achieved: >97% accuracy on the test set!")
else:
    print(f"✗ Below target — got {test_accuracy:.4f}. Consider expanding the grid.")