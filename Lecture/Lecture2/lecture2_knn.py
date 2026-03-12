import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# number of neighbors in KNN
n_neighbors = 10


# Generate sample data
n_samples = 300
centers = [[1, 1], [-1, -1], [1, -1]]

X, y = datasets.make_blobs(
    n_samples=n_samples, centers=centers, cluster_std=0.80, random_state=0
)
print("Target is 0,1 or 2: ",y)



# Train - test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Plot the training points
color = ['red','green','blue']
for i in range(3):
    plt.scatter(X_train[np.where(y_train==i), 0], X_train[np.where(y_train==i), 1], s=10, marker="v",color=color[i])
plt.show()



# we create an instance of Neighbours Classifier and fit the data.
# K nearest neighbors
clf = neighbors.KNeighborsClassifier(n_neighbors, weights="uniform") # "distance"
# Decision tree
#clf = DecisionTreeClassifier(max_depth=3)

clf.fit(X_train, y_train)

print("Training accuracy: ",accuracy_score(y_train, clf.predict(X_train)))
print("Testing accuracy:  ",accuracy_score(y_test, clf.predict(X_test)))



h = 0.02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])


plt.figure(figsize=(6, 6))
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
color = ['red','green','blue']
for i in range(3):
  #  plt.scatter(X_test[np.where(y_test==i), 0], X_test[np.where(y_test==i), 1], s=10, marker="o",color=color[i])
    plt.scatter(X_train[np.where(y_train==i), 0], X_train[np.where(y_train==i), 1], s=10, marker="v",color=color[i])

plt.title(
        "3-Class classification (k = %i, weights = uniform)" % (n_neighbors)
)

plt.show()
