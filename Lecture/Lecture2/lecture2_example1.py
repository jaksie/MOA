import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # linear discriminant

from sklearn.datasets import make_classification

# Generate synthetic 2D data with two areas
X, y = make_classification(n_samples=500,n_features=2, n_redundant=0, random_state=121,
                           n_informative=2, n_clusters_per_class=1, n_classes=2)

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

# ask how to classify a given point
#print("Predict for [-0.8, -1]: ", clf.predict([[-0.8, -1]]))
#print("Predict for [0.8, 1]: ", clf.predict([[0.8, 1]]))


from sklearn.inspection import DecisionBoundaryDisplay

'''
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="predict_proba",
    plot_method="pcolormesh",
    ax=None,
    cmap="RdBu",
    alpha=0.5,
)
'''

DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="predict_proba",
    plot_method="contour",
    ax=None,
    alpha=1.0,
    levels=[0.5],
    )
    
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.text(X[:, 0].min(), X[:, 1].max(), 'accuracy: '+str(clf.score(X,y)), fontsize=12, color='black', ha='left')
plt.show()

