import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.inspection import DecisionBoundaryDisplay


# Import the IRIS dataset
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data#[:, 2:4]
y = iris.target

# Let's group together labels 1 and 2 so we have a binary problem:
y[y == 1] = 2

# Create some SVM instances and fit to our data.
C = 1.0  # SVM regularization parameter
models = (
    svm.LinearSVC(C=C, max_iter=10000),
    svm.SVC(kernel="linear", C=C),
    svm.SVC(kernel="rbf", gamma=0.7, C=C),
    svm.SVC(kernel="rbf", gamma=4.0, C=C),
    svm.SVC(kernel="poly", degree=2, gamma="auto", C=C),
    svm.SVC(kernel="poly", degree=7, gamma="auto", C=C),
)
models = (clf.fit(X, y) for clf in models)

# Titles for the plots
titles = (
    "SVC with linear kernel",
    "LinearSVC (linear kernel)",
    "SVC with RBF kernel, $\gamma$=0.7",
    "SVC with RBF kernel, $\gamma$=4.0",
    "SVC with polynomial (degree 2) kernel",
    "SVC with polynomial (degree 7) kernel",
)

# Set-up 3x2 grid for plotting.
fig, sub = plt.subplots(3, 2, figsize=(8, 8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)




# We train using all features but to visualize the boundaries we need to look
# at some 2-D projection of our features, let's just take the first two components (sepal width and length)
X = X[:, :2]
X0, X1 = X[:, 0], X[:, 1]

# Now let's plot the decision boundaries and labeled 2-D features
for clf, title, ax in zip(models, titles, sub.flatten()):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=plt.cm.jet,
        alpha=0.8,
        ax=ax,
        xlabel=iris.feature_names[0],
        ylabel=iris.feature_names[1],
    )
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
