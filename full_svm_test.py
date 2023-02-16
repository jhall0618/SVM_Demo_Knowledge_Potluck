from sklearn import svm, datasets, model_selection
from sklearn import preprocessing
import jacks_datasets


d_set_name = 'xor' #'rings' #'wine'  # 'iris'  #  'wine' # 'xor' # 'breast_cancer' #

# Import a dataset
match d_set_name:
    case 'iris':
        my_data = datasets.load_iris()
    case 'breast_cancer':
        my_data = datasets.load_breast_cancer()
    case 'wine':
        my_data = datasets.load_wine()
    case 'xor':
        my_data = jacks_datasets.retrieve_jacks_data(d_set_name)
    case 'rings':
        my_data = jacks_datasets.retrieve_jacks_data(d_set_name)

X = my_data.data
y = my_data.target

# Let's group together labels 1 and 2 so we have a binary problem:
y[y == 2] = 1

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

# Now let's plot the decision boundaries and labeled 2-D features
print(f'Model Accuracies for {d_set_name} Dataset:')
for clf, title in zip(models, titles):
    # Display Accuracy
    yhat = clf.predict(Xte)
    acc = 100*sum(yte == yhat)/len(yte)
    print(title+f' Accuracy: {acc}')

