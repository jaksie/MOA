import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # linear discriminant
from sklearn.naive_bayes import GaussianNB

# Generate synthetic 2D data with two areas
X, y = make_circles(n_samples=500,noise=0.2, factor=0.5, random_state=1)

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#clf = LinearDiscriminantAnalysis()
gnb = GaussianNB(var_smoothing=0)
gnb.fit(X, y)

# ask how to classify a given point
#print("Predict for [-0.8, -1]: ", clf.predict([[-0.8, -1]]))
#print("Predict for [0.8, 1]: ", clf.predict([[0.8, 1]]))


from sklearn.inspection import DecisionBoundaryDisplay


DecisionBoundaryDisplay.from_estimator(
    gnb,
    X,
    response_method="predict_proba",
    plot_method="pcolormesh",
    ax=None,
    cmap="RdBu",
    alpha=0.5,
)


DecisionBoundaryDisplay.from_estimator(
    gnb,
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
plt.text(X[:, 0].min(), X[:, 1].max(), 'accuracy: '+str(gnb.score(X,y)), fontsize=12, color='black', ha='left')
plt.show()
