import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

import time
start = time.time()

# Generate synthetic 2D data with two areas - test 118 and 121
X, y = make_classification(n_features=2, n_redundant=0, random_state=121,
                           n_informative=2, n_clusters_per_class=1, n_classes=2)

# Plot the data
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Define a function to calculate accuracy of the cuts
def evaluate_cuts(threshold_x, threshold_y, X, y):
    # Here we have four options, how to apply the cuts and we choose
    # the best one
    acc = np.zeros(4)

    cuts_x = X[:, 0] > threshold_x  #slicing w/ indexing
    cuts_y = X[:, 1] > threshold_y
    labels = np.where(cuts_x & cuts_y, 0, 1) # Return elements chosen from x or y depending on condition
    acc[0] = accuracy_score(y, labels) # In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.

    cuts_x = X[:, 0] < threshold_x
    cuts_y = X[:, 1] < threshold_y
    labels = np.where(cuts_x & cuts_y, 0, 1)
    acc[1] = accuracy_score(y, labels)

    cuts_x = X[:, 0] > threshold_x
    cuts_y = X[:, 1] < threshold_y
    labels = np.where(cuts_x & cuts_y, 0, 1)
    acc[2] = accuracy_score(y, labels)

    cuts_x = X[:, 0] < threshold_x
    cuts_y = X[:, 1] > threshold_y
    labels = np.where(cuts_x & cuts_y, 0, 1)
    acc[3] = accuracy_score(y, labels)

    return np.max(acc)

# Perform grid search to find the best thresholds
best_score = 0
best_thresholds = (0, 0)
for tx in np.linspace(X[:, 0].min(), X[:, 0].max(), 100):
    for ty in np.linspace(X[:, 1].min(), X[:, 1].max(), 100):
        score = evaluate_cuts(tx, ty, X, y)
        if score > best_score:
            best_score = score
            print("Best score (accuracy): ",score)
            best_thresholds = (tx, ty)

end = time.time()
print(f"Time taken to find the best thresholds: {end - start}")

# Get the best thresholds
threshold_x, threshold_y = best_thresholds

# Plot the optimized cuts
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axvline(threshold_x, color='red', linestyle='--', label='Cut X')
plt.axhline(threshold_y, color='blue', linestyle='--', label='Cut Y')
plt.title('Optimized Cuts in 2D Training Data')
plt.legend()
plt.text(X[:, 0].min(), X[:, 1].max(), 'accuracy: '+str(best_score), fontsize=12, color='black', ha='left')
plt.show()
