import matplotlib.pyplot as plt
from sklearn import svm, datasets, model_selection, preprocessing
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils import Bunch
import numpy as np

# Import a dataset
d_set_name = 'xor' #'wine'  # 'iris'  #  'wine' # 'xor'

# Import a dataset
match d_set_name:
    case 'iris':
        my_data = datasets.load_iris()
    case 'breast_cancer':
        my_data = datasets.load_breast_cancer()
    case 'wine':
        my_data = datasets.load_wine()
    case 'xor':
        Nsamps = 1000
        A = 2*np.random.rand(1, Nsamps) - 1.0
        B = 2*np.random.rand(1, Nsamps) - 1.0
        target = (((A >= 0) | (B >= 0)) & ~((A >= 0) & (B >= 0))).astype(int)
        target = target.reshape(target.shape[1:])
        data = np.concatenate([A, B]).transpose()
        my_data = Bunch(data=data, target=target, feature_names=['A', 'B'])


X = my_data.data
y = my_data.target

# Let's group together labels 1 and 2 so we have a binary problem:
y[y == 2] = 1
# To visualize the boundaries we need to look at some 2-D projection of our features,
# let's just take the first two components (sepal width and length)
X = X[:, :2]

# Split into a train and test set:
X, Xte, y, yte = model_selection.train_test_split(X, y, test_size=0.33, random_state=1234)


# Use the training data to determine normalization parameters:
scaler = preprocessing.StandardScaler().fit(X)

# Then transform both train and test sets to use this normalization:
X = scaler.transform(X)
Xte = scaler.transform(Xte)

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
    "SVM with linear kernel",
    "LinearSVM (linear kernel)",
    "SVM with RBF kernel, $\gamma$=0.7",
    "SVM with RBF kernel, $\gamma$=4.0",
    "SVM with polynomial (degree 2) kernel",
    "SVM with polynomial (degree 3) kernel",
)

# Set-up 3x2 grid for plotting.
fig, sub = plt.subplots(3, 2, figsize=(8, 8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
fig.suptitle(f'{d_set_name} Dataset')
# Break the train points into 1st & 2nd dims
X0, X1 = X[:, 0], X[:, 1]

# Now let's plot the decision boundaries and labeled 2-D features
print(f'Decision Boundaries and Model Accuracies for {d_set_name} Dataset:')
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

