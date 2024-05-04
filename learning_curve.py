import matplotlib.pyplot as plt


from DecisionTree import DecisionTree
from load_datasets import load_abalone_dataset, load_iris_dataset, load_wine_dataset


# Datasets
ABALONE = 'abalone'
IRIS = 'iris'
WINE = 'wine'
TRAIN_RATIO = 0.7

# Hyperparameters - Decision Trees
DECISION_TREE_MAX_DEPTHS = {
    ABALONE: 8,
    IRIS: 2,
    WINE: 10,
}

# Load datasets
iris_train, iris_train_labels, iris_test, iris_test_labels = load_iris_dataset(TRAIN_RATIO)
wine_train, wine_train_labels, wine_test, wine_test_labels = load_wine_dataset(TRAIN_RATIO)
abalone_train, abalone_train_labels, abalone_test, abalone_test_labels = load_abalone_dataset(TRAIN_RATIO)

# Classifiers - Decision Trees
dt_abalone = DecisionTree(max_depth=DECISION_TREE_MAX_DEPTHS[ABALONE])
dt_iris = DecisionTree(max_depth=DECISION_TREE_MAX_DEPTHS[IRIS])
dt_wine = DecisionTree(max_depth=DECISION_TREE_MAX_DEPTHS[WINE])

# Percentages to use for learing curve
percentages = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

# Get the classifer
def get_classifier(dataset):
    if dataset == 'abalone':
        return dt_abalone
    elif dataset == 'iris':
        return dt_iris
    elif dataset == 'wine':
        return dt_wine
    else:
        raise ValueError('Error: invalid dataset')

# Get the data
def get_data(dataset):
    if dataset == 'abalone':
        return abalone_train, abalone_train_labels, abalone_test, abalone_test_labels
    elif dataset == 'iris':
        return iris_train, iris_train_labels, iris_test, iris_test_labels
    elif dataset == 'wine':
        return wine_train, wine_train_labels, wine_test, wine_test_labels
    else:
        raise ValueError('Error: invalid dataset')

# Compute the learning curve
def show_learning_curve(dataset):
    accuracies = []

    classifier = get_classifier(dataset)
    train, train_labels, test, test_labels = get_data(dataset)
    for percentage in percentages:
        num_elements = int(len(train) * percentage / 100)
        train_sample = train[:num_elements]
        train_labels_sample = train_labels[:num_elements]

        classifier.train(train_sample, train_labels_sample)

        _, accuracy, _, _, _ = classifier.evaluate(test, test_labels)
        accuracies.append((percentage, accuracy))

        print(f'Got {accuracy:.2f} accuracy for {percentage}% ({num_elements} samples).')

    # Extract percentage and accuracy from the list of tuples
    x = [item[0] for item in accuracies]
    y = [item[1] for item in accuracies]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='blue', label='Accuracy')
    plt.title(f'Learning curve of decision tree on {dataset}')
    plt.xlabel('Percentage of the training set')
    plt.ylabel('Accuracy of the predictions')
    plt.grid(True)
    plt.legend()
    plt.show()

show_learning_curve('abalone')