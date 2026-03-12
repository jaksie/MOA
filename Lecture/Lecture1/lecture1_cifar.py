import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# time it
import time
start = time.time()

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()

end = time.time()
print(f"Time taken to load data: {end - start}")

# Class names
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

num_classes = 10

# Plot one random image per class
fig = plt.figure(figsize=(10, 4))

for i in range(num_classes):
    ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])

    # indices of images belonging to class i
    idx = np.where(y_train == i)[0]

    # randomly choose one image
    img_num = np.random.choice(idx)

    im = x_train[img_num]  # (32, 32, 3)
    ax.imshow(im)
    ax.set_title(class_names[i], fontsize=12)

plt.tight_layout()
plt.show()

