import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


# Load MNIST data (28x28 images)
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Convert to NumPy and normalize
X = X.to_numpy() / 255.0
y = y.astype(int)

# Plot first 100 images
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1,
    hspace=0.05, wspace=0.05
)

for i in range(100):
    ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(X[i].reshape(28, 28),
              cmap=plt.cm.binary,
              interpolation='nearest')
    ax.text(0, 7, str(y[i]), color='red')

plt.show()


