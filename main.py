import matplotlib.pyplot as plt
from sklearn import svm, datasets, model_selection
from sklearn.inspection import DecisionBoundaryDisplay


# Import a dataset
#my_data = datasets.load_breast_cancer()
#my_data = datasets.load_wine()
my_data = datasets.load_iris()

X = my_data.data
y = my_data.target
# Let's group together labels 1 and 2 so we have a binary problem:
y[y == 2] = 1
# To visualize the boundaries we need to look at some 2-D projection of our features,
# let's just take the first two components (sepal width and length)
X = X[:, :2]

# Split into a train and test set:
X, Xte, y, yte = model_selection.train_test_split(X,y,test_size=0.33, random_state=1234)

# Create some SVM instances and fit to our data.
C = 1.0  # SVM regularization parameter
models = (
    svm.LinearSVC(C=C, max_iter=10000),
    svm.SVC(kernel="linear", C=C),
    svm.SVC(kernel="rbf", gamma=0.7, C=C),
    svm.SVC(kernel="rbf", gamma=4.0, C=C),
    svm.SVC(kernel="poly", degree=2, gamma="auto", C=C),
    svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
)
models = (clf.fit(X, y) for clf in models)

# Titles for the plots
titles = (
    "SVC with linear kernel",
    "LinearSVC (linear kernel)",
    "SVC with RBF kernel, $\gamma$=0.7",
    "SVC with RBF kernel, $\gamma$=4.0",
    "SVC with polynomial (degree 2) kernel",
    "SVC with polynomial (degree 3) kernel",
)

# Set-up 3x2 grid for plotting.
fig, sub = plt.subplots(3, 2, figsize=(8, 8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# Break the train points into 1st & 2nd dims
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
        xlabel=my_data.feature_names[0],
        ylabel=my_data.feature_names[1],
    )
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    # Display Accuracy
    yhat = clf.predict(Xte)
    acc = 100*sum(yte == yhat)/len(yte)
    print(title+f' Accuracy: {acc}')

plt.show()
