MNIST Machine Learning Exercises
================================

This project guides you through classic machine learning concepts on the MNIST dataset, which is essentially the "Hello World" of machine learning. It provides a benchmark to quickly test how well new classification algorithms work.

The MNIST Dataset
-----------------
MNIST is a classic dataset of 70,000 handwritten digit images from 0 to 9, each sized at 28x28 pixels. The exercises in this repository deal with classifying these digits effectively while explaining different ML principles.

1. Binary Classification
------------------------
Instead of recognizing all 10 digits at once, a binary classifier simplifies the problem to just two outcomes. For example, "Is this a 5?" vs. "Is this not a 5?". The SGD (Stochastic Gradient Descent) classifier is an excellent starting point because it is fast and handles large datasets efficiently.
*See `Exercise1.py` for reading about Binary Classification models.*

2. Evaluating a Classifier
--------------------------
Properly measuring how well your model learned is critical.
* Cross-validation: Splits your data into folds, trains on some, tests on others, and averages the results. This offers a more reliable and robust accuracy estimate than a single naive train/test split.
* Confusion Matrix: Far more informative than raw accuracy. It shows exactly how a model is making mistakes — for instance, how often it mistakenly identifies 5s as 3s.
* Precision & Recall tradeoffs: Precision asks "Of what I predicted positive, how many were right?" whilst Recall asks "Of all actual positives, how many did I truly catch?". Typically, increasing precision reduces recall, and vice-versa.
* ROC Curve: Plots the True Positive Rate (Recall) vs. the False Positive Rate at various decision thresholds, effectively giving a visual sense of the overall capability of a binary classifier.

3. Multiclass Classification
----------------------------
When you need to distinguish between more than two classes (e.g. all 10 digits), you use a multiclass classifier. 
Some algorithms, like Random Forests or Naive Bayes, natively support multiclass tasks. Others, like Support Vector Machines (SVMs) or logistic regressors, are strongly suited as binary-only by default. In these algorithms, strategies like One-vs-Rest (OvR) (train one binary classifier for each separate class, 10 for MNIST) or One-vs-One (OvO) are used to extend their capabilities to multiple labels.
*See `Exercise3.py` for the code implementation on Multiclass variants.*

4. Error Analysis
-----------------
Rather than arbitrarily tuning hyperparameters hoping for the score to go up, error analysis instructs checking exactly what your model is getting wrong.
By examining confusion matrices, observing patterns in misclassified elements, understanding specific failing logic, and using those insights you can easily guide data cleaning or feature engineering strategies. Visualizing precisely where the classifier fails is a powerful way to troubleshoot machine learning models.
*See `Exercise3.py` for evaluating and visualising confused digits.*

5. Data Augmentation
--------------------
Often, a powerful trick to improve the model without changing code is to artificially expand your training set. By creating shifted versions of each particular image (shifted left, right, up, down by one pixel) and injecting them into the training set, you provide the model with a more expansive array of examples. This usually forces the model to learn translation invariance, ultimately boosting accuracy.
*See `Exercise2.py` for how this acts as a force multiplier for predicting correct digits.*

6. Multioutput Classification
-----------------------------
Multioutput-multiclass classification (multioutput classification) is a generalization of multilabel classification where each label can be multiclass. An example is a system that removes noise from images. It takes a noisy digit image as input and outputs a clean digit image. The classifier's output is multilabel (one label per pixel) and each label can have multiple values (pixel intensity ranges from 0 to 255).
*See `Exercise4.py` for implementing image denoising using a K-Nearest Neighbors classifier.*

Output Images
-------------
All generated plots, charts, and visualizations are saved into their respective exercise subdirectories (e.g., `images/exercise1/`, `images/exercise3/`, etc.) to keep the project root clean and organized.
