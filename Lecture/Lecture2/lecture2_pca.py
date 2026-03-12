from sklearn.datasets import load_digits
digits = load_digits()


from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 10))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(100):
    ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

plt.show()


# Perform PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
proj = pca.fit_transform(digits.data)
print(proj.shape)

#Plot the projection
plt.figure()
plt.scatter(proj[:, 0], proj[:, 1], c=digits.target.astype(int), cmap=plt.get_cmap('Paired', 10), vmin=-0.5, vmax=9.5)
plt.colorbar()
plt.show()


# We use now 64 principal components
pca = PCA(n_components=64)
proj = pca.fit_transform(digits.data)

principal = pca.explained_variance_
print(pca.explained_variance_)

fig, ax = plt.subplots(tight_layout=True)
ax.bar( [x for x in range(principal.size)], principal)
plt.show()


# Number of components after PCA
n_components=16
pca = PCA(n_components)
proj = pca.fit_transform(digits.data)

principal = pca.explained_variance_
print(pca.explained_variance_)

fig, ax = plt.subplots(tight_layout=True)
ax.bar( [x for x in range(principal.size)], principal)
plt.show()

import numpy as np
from sklearn.model_selection import train_test_split

# Instead of 8x8 digits we have PCA transformed vectors of the length n_components
# split the data into training and validation sets
seed = np.random.randint(0,1000)
X_train, X_test, y_train, y_test = train_test_split(pca.fit_transform(digits.data), digits.target, random_state=seed)


from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# train the model
# comment out the model you don't use
clf = GaussianNB()
#clf = LinearDiscriminantAnalysis()


print(clf.fit(X_train, y_train))
#print(clf.feature_importances_)

# use the model to predict the labels of the test data
# test data were not used for training!
predicted = clf.predict(X_test)
expected = y_test

print("Score = ",clf.score(X_test, y_test))


# Plot the prediction
fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# split the original data (i.e. images!) into training and validation sets
XP_train, XP_test, yP_train, yP_test = train_test_split(digits.data, digits.target, random_state=seed)

# plot the digits: each image is 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(XP_test.reshape(-1, 8, 8)[i], cmap=plt.cm.binary,
              interpolation='nearest')

    # label the image with the target value
    if predicted[i] == expected[i]:
        ax.text(0, 7, str(expected[i])+" "+str(predicted[i]), color='green')
    else:
        ax.text(0, 7, str(expected[i])+" "+str(predicted[i]), color='red')

plt.show()

matches = (predicted == expected)
print(matches.sum())
print(len(matches))
matches.sum() / float(len(matches))
from sklearn import metrics
print(metrics.classification_report(expected, predicted))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(expected, predicted)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


# plot the 16 eigendigits: each image is 8x8 pixels
eigendigits = pca.components_.reshape((n_components, 8, 8))

fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(len(eigendigits)):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(eigendigits[i], cmap=plt.cm.binary,
              interpolation='nearest')
plt.show()



# plot PCA reconstructed digits
X_test_inv = pca.inverse_transform(X_test)

# Plot the prediction
fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the digits: each image is 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(X_test_inv.reshape(-1, 8, 8)[i], cmap=plt.cm.binary,
              interpolation='nearest')

    # label the image with the target value
    if predicted[i] == expected[i]:
        ax.text(0, 7, str(expected[i])+" "+str(predicted[i]), color='green')
    else:
        ax.text(0, 7, str(expected[i])+" "+str(predicted[i]), color='red')

plt.show()
