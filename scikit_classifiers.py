from time import time
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from load_datasets import load_abalone_dataset, load_iris_dataset, load_wine_dataset

ABALONE = 'abalone'
ABALONE_ONE_HOT = 'abalone one hot'
IRIS = 'iris'
WINE = 'wine'
TRAIN_RATIO = 0.7

iris_train, iris_train_labels, iris_test, iris_test_labels = load_iris_dataset(TRAIN_RATIO)
wine_train, wine_train_labels, wine_test, wine_test_labels = load_wine_dataset(TRAIN_RATIO)
abalone_train, abalone_train_labels, abalone_test, abalone_test_labels = load_abalone_dataset(TRAIN_RATIO)
abalone_train_one_hot, abalone_train_labels_one_hot, abalone_test_one_hot, abalone_test_labels_one_hot = load_abalone_dataset(TRAIN_RATIO, one_hot_encoding=True)

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
    elif dataset == ABALONE_ONE_HOT:
        test_labels = abalone_test_labels_one_hot
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

    start = time()
    clf.fit(iris_train, iris_train_labels)
    end = time()
    print(f'Training took {(end - start) * 1000:.2f} ms')

    start = time()
    clf.predict([iris_train[0]])
    end = time()
    print(f'Prediction took {(end - start) * 1000:.2f} ms')

    # Make predictions on the test set and show results
    y_pred = clf.predict(iris_test)
    show_scikit_results(IRIS, y_pred)

def test_scikit_nn_iris(num_layers, hidden_layer_size):
    print('\nScikit-learn Neural Network - Iris:')
    # Train a neural network classifier
    clf = MLPClassifier(hidden_layer_sizes=tuple([hidden_layer_size for _ in range(num_layers)]),
                        activation='logistic',
                        solver='sgd',
                        batch_size='auto',
                        max_iter=2000)

    start = time()
    clf.fit(iris_train, iris_train_labels)
    end = time()
    print(f'Training took {(end - start) * 1000:.2f} ms')

    start = time()
    clf.predict([iris_train[0]])
    end = time()
    print(f'Prediction took {(end - start) * 1000:.2f} ms')

    # Make predictions on the test set and show results
    y_pred = clf.predict(iris_test)
    show_scikit_results(IRIS, y_pred)

def test_scikit_knn_iris(num_neighbors):
    print('\nScikit-learn K-Nearest Neighbors - Iris:')
    clf = KNeighborsClassifier(n_neighbors=num_neighbors)

    start = time()
    clf.fit(iris_train, iris_train_labels)
    end = time()
    print(f'Training took {(end - start) * 1000:.2f} ms')

    start = time()
    clf.predict([iris_train[0]])
    end = time()
    print(f'Prediction took {(end - start) * 1000:.2f} ms')

    # Make predictions on the test set and show results
    y_pred = clf.predict(iris_test)
    show_scikit_results(IRIS, y_pred)

def test_scikit_naive_bayes_iris():
    print('\nScikit-learn Gaussian Naive Bayes - Iris:')
    clf = GaussianNB()

    start = time()
    clf.fit(iris_train, iris_train_labels)
    end = time()
    print(f'Training took {(end - start) * 1000:.2f} ms')

    start = time()
    clf.predict([iris_train[0]])
    end = time()
    print(f'Prediction took {(end - start) * 1000:.2f} ms')

    # Make predictions on the test set and show results
    y_pred = clf.predict(iris_test)
    show_scikit_results(IRIS, y_pred)

def test_scikit_dt_wine():
    print('\nScikit-learn Decision Tree - Wine:')
    # Train a decision tree classifier
    clf = DecisionTreeClassifier()

    start = time()
    clf.fit(wine_train, wine_train_labels)
    end = time()
    print(f'Training took {(end - start) * 1000:.2f} ms')

    start = time()
    clf.predict([wine_train[0]])
    end = time()
    print(f'Prediction took {(end - start) * 1000:.2f} ms')

    # Make predictions on the test set and show results
    y_pred = clf.predict(wine_test)
    show_scikit_results(WINE, y_pred)

def test_scikit_nn_wine(num_layers, hidden_layer_size):
    print('\nScikit-learn Neural Network - Wine:')
    # Train a neural network classifier
    clf = MLPClassifier(hidden_layer_sizes=tuple([hidden_layer_size for _ in range(num_layers)]),
                        activation='logistic',
                        solver='sgd',
                        batch_size='auto',
                        max_iter=2000)

    start = time()
    clf.fit(wine_train, wine_train_labels)
    end = time()
    print(f'Training took {(end - start) * 1000:.2f} ms')

    start = time()
    clf.predict([wine_train[0]])
    end = time()
    print(f'Prediction took {(end - start) * 1000:.2f} ms')

    # Make predictions on the test set and show results
    y_pred = clf.predict(wine_test)
    show_scikit_results(WINE, y_pred)

def test_scikit_knn_wine(num_neighbors):
    print('\nScikit-learn K-Nearest Neighbors - Wine:')
    clf = KNeighborsClassifier(n_neighbors=num_neighbors)

    start = time()
    clf.fit(wine_train, wine_train_labels)
    end = time()
    print(f'Training took {(end - start) * 1000:.2f} ms')

    start = time()
    clf.predict([wine_train[0]])
    end = time()
    print(f'Prediction took {(end - start) * 1000:.2f} ms')

    # Make predictions on the test set and show results
    y_pred = clf.predict(wine_test)
    show_scikit_results(WINE, y_pred)

def test_scikit_naive_bayes_wine():
    print('\nScikit-learn Gaussian Naive Bayes - Wine:')
    clf = GaussianNB()

    start = time()
    clf.fit(wine_train, wine_train_labels)
    end = time()
    print(f'Training took {(end - start) * 1000:.2f} ms')

    start = time()
    clf.predict([wine_train[0]])
    end = time()
    print(f'Prediction took {(end - start) * 1000:.2f} ms')

    # Make predictions on the test set and show results
    y_pred = clf.predict(wine_test)
    show_scikit_results(WINE, y_pred)

def test_scikit_dt_abalone():
    print('\nScikit-learn Decision Tree - Abalone:')
    # Train a decision tree classifier
    clf = DecisionTreeClassifier()

    start = time()
    clf.fit(abalone_train, abalone_train_labels)
    end = time()
    print(f'Training took {(end - start) * 1000:.2f} ms')

    start = time()
    clf.predict([abalone_train[0]])
    end = time()
    print(f'Prediction took {(end - start) * 1000:.2f} ms')

    # Make predictions on the test set and show results
    y_pred = clf.predict(abalone_test)
    show_scikit_results(ABALONE, y_pred)

def test_scikit_nn_abalone(num_layers, hidden_layer_size):
    print('\nScikit-learn Neural Network - Abalone:')
    # Train a neural network classifier
    clf = MLPClassifier(hidden_layer_sizes=tuple([hidden_layer_size for _ in range(num_layers)]),
                        activation='logistic',
                        solver='sgd',
                        batch_size='auto',
                        max_iter=2000)

    start = time()
    clf.fit(abalone_train_one_hot, abalone_train_labels_one_hot)
    end = time()
    print(f'Training took {(end - start) * 1000:.2f} ms')

    start = time()
    clf.predict([abalone_train_one_hot[0]])
    end = time()
    print(f'Prediction took {(end - start) * 1000:.2f} ms')

    # Make predictions on the test set and show results
    y_pred = clf.predict(abalone_test_one_hot)
    show_scikit_results(ABALONE_ONE_HOT, y_pred)

def test_scikit_knn_abalone(num_neighbors):
    print('\nScikit-learn K-Nearest Neighbors - Abalone:')
    clf = KNeighborsClassifier(n_neighbors=num_neighbors)

    start = time()
    clf.fit(abalone_train_one_hot, abalone_train_labels_one_hot)
    end = time()
    print(f'Training took {(end - start) * 1000:.2f} ms')

    start = time()
    clf.predict([abalone_train_one_hot[0]])
    end = time()
    print(f'Prediction took {(end - start) * 1000:.2f} ms')

    # Make predictions on the test set and show results
    y_pred = clf.predict(abalone_test_one_hot)
    show_scikit_results(ABALONE_ONE_HOT, y_pred)

def test_scikit_naive_bayes_abalone():
    print('\nScikit-learn Gaussian Naive Bayes - Abalone:')
    clf = GaussianNB()

    start = time()
    clf.fit(abalone_train_one_hot, abalone_train_labels_one_hot)
    end = time()
    print(f'Training took {(end - start) * 1000:.2f} ms')

    start = time()
    clf.predict([abalone_train_one_hot[0]])
    end = time()
    print(f'Prediction took {(end - start) * 1000:.2f} ms')

    # Make predictions on the test set and show results
    y_pred = clf.predict(abalone_test_one_hot)
    show_scikit_results(ABALONE_ONE_HOT, y_pred)