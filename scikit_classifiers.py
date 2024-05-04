from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from load_datasets import load_abalone_dataset, load_iris_dataset, load_wine_dataset

ABALONE = 'abalone'
IRIS = 'iris'
WINE = 'wine'
TRAIN_RATIO = 0.7

iris_train, iris_train_labels, iris_test, iris_test_labels = load_iris_dataset(TRAIN_RATIO)
wine_train, wine_train_labels, wine_test, wine_test_labels = load_wine_dataset(TRAIN_RATIO)
abalone_train, abalone_train_labels, abalone_test, abalone_test_labels = load_abalone_dataset(TRAIN_RATIO)

def print_results(cm, accuracy, precision, recall, f1):
    print(f'Confusion Matrix:\n{cm}')
    print(f'Accuracy: {accuracy * 100:.5f}%')
    print(f'Precision: {precision * 100:.5f}%')
    print(f'Recall: {recall * 100:.5f}%')
    print(f'F1 Score: {f1 * 100:.5f}%')

def show_scikit_results(dataset, y_pred):
    # Get test labels
    if dataset == IRIS:
        test_labels = iris_test_labels
    elif dataset == WINE:
        test_labels = wine_test_labels
    elif dataset == ABALONE:
        test_labels = abalone_test_labels
    else:
        raise ValueError('Error: invalid dataset')
    
    # Show results
    cm = confusion_matrix(test_labels, y_pred)
    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred, average='weighted')
    recall = recall_score(test_labels, y_pred, average='weighted')
    f1 = f1_score(test_labels, y_pred, average='weighted')
    print_results(cm, accuracy, precision, recall, f1)

def test_scikit_dt_iris():
    print('\nScikit-learn Decision Tree - Iris:')
    # Train a decision tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(iris_train, iris_train_labels)

    # Make predictions on the test set and show results
    y_pred = clf.predict(iris_test)
    show_scikit_results(IRIS, y_pred)

def test_scikit_nn_iris():
    print('\nScikit-learn Neural Network - Iris:')
    # Train a neural network classifier
    clf = MLPClassifier()
    clf.fit(iris_train, iris_train_labels)

    # Make predictions on the test set and show results
    y_pred = clf.predict(iris_test)
    show_scikit_results(IRIS, y_pred)

def test_scikit_dt_wine():
    print('\nScikit-learn Decision Tree - Wine:')
    # Train a decision tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(wine_train, wine_train_labels)

    # Make predictions on the test set and show results
    y_pred = clf.predict(wine_test)
    show_scikit_results(WINE, y_pred)

def test_scikit_nn_wine():
    print('\nScikit-learn Neural Network - Wine:')
    # Train a neural network classifier
    clf = MLPClassifier()
    clf.fit(wine_train, wine_train_labels)

    # Make predictions on the test set and show results
    y_pred = clf.predict(wine_test)
    show_scikit_results(WINE, y_pred)

def test_scikit_dt_abalone():
    print('\nScikit-learn Decision Tree - Abalone:')
    # Train a decision tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(abalone_train, abalone_train_labels)

    # Make predictions on the test set and show results
    y_pred = clf.predict(abalone_test)
    show_scikit_results(ABALONE, y_pred)

def test_scikit_nn_abalone():
    print('\nScikit-learn Neural Network - Abalone:')
    # Train a neural network classifier
    clf = MLPClassifier()
    clf.fit(abalone_train, abalone_train_labels)

    # Make predictions on the test set and show results
    y_pred = clf.predict(abalone_test)
    show_scikit_results(ABALONE, y_pred)