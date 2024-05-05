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

DECISION_TREE_P_VALUES = {
    ABALONE: 0.05,
    IRIS: 0.05,
    WINE: 0.05,
}

# Load datasets
iris_train, iris_train_labels, iris_test, iris_test_labels = load_iris_dataset(TRAIN_RATIO)
wine_train, wine_train_labels, wine_test, wine_test_labels = load_wine_dataset(TRAIN_RATIO)
abalone_train, abalone_train_labels, abalone_test, abalone_test_labels = load_abalone_dataset(TRAIN_RATIO)

# Classifiers - Decision Trees
dt_abalone_no_pruning = DecisionTree(max_depth=DECISION_TREE_MAX_DEPTHS[ABALONE], pruning=False)
dt_iris_no_pruning = DecisionTree(max_depth=DECISION_TREE_MAX_DEPTHS[IRIS], pruning=False)
dt_wine_no_pruning = DecisionTree(max_depth=DECISION_TREE_MAX_DEPTHS[WINE], pruning=False)
dt_abalone_pruning = DecisionTree(max_depth=DECISION_TREE_MAX_DEPTHS[ABALONE], pruning=True, p_value_threshold=DECISION_TREE_P_VALUES[ABALONE])
dt_iris_pruning = DecisionTree(max_depth=DECISION_TREE_MAX_DEPTHS[IRIS], pruning=True, p_value_threshold=DECISION_TREE_P_VALUES[IRIS])
dt_wine_pruning = DecisionTree(max_depth=DECISION_TREE_MAX_DEPTHS[WINE], pruning=True, p_value_threshold=DECISION_TREE_P_VALUES[WINE])

# Percentages to use for learing curve
percentages = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

# Get the classifer
def get_classifier(dataset, pruning):
    if pruning:
        if dataset == 'abalone':
            return dt_abalone_pruning
        elif dataset == 'iris':
            return dt_iris_pruning
        elif dataset == 'wine':
            return dt_wine_pruning
        else:
            raise ValueError('Error: invalid dataset')
    else:
        if dataset == 'abalone':
            return dt_abalone_no_pruning
        elif dataset == 'iris':
            return dt_iris_no_pruning
        elif dataset == 'wine':
            return dt_wine_no_pruning
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
def show_learning_curve(dataset, pruning, print_results=False):
    # Store accuracies
    accuracies = []

    # Get the classifier & the data
    classifier = get_classifier(dataset, pruning)
    train, train_labels, test, test_labels = get_data(dataset)

    # Compute the accuracy for each percentage
    for percentage in percentages:
        # Sample the training set
        num_elements = int(len(train) * percentage / 100)
        train_sample = train[:num_elements]
        train_labels_sample = train_labels[:num_elements]

        # Train
        classifier.train(train_sample, train_labels_sample)

        # Evaluate and store the accuracy
        _, accuracy, _, _, _ = classifier.evaluate(test, test_labels)
        accuracies.append((percentage, accuracy))

        # Print results if needed
        if print_results:
            print(f'Got {accuracy * 100:.2f}% accuracy for {percentage}% ({num_elements} samples).')

    # Extract percentage and accuracy from the list of tuples
    x = [item[0] for item in accuracies]
    y = [item[1] for item in accuracies]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='blue', label='Accuracy')
    title = f'Learning curve of decision tree on {dataset} dataset'
    if pruning:
        title += f' with pruning (p-value threshold: {DECISION_TREE_P_VALUES[dataset]})'
    else:
        title += ' without pruning'
    plt.title(title)
    plt.xlabel('Percentage of the training set')
    plt.ylabel('Accuracy of the predictions')
    plt.grid(True)
    plt.legend()
    plt.show()

def test_learning_curve():
    datasets = [ABALONE, IRIS, WINE]
    pruning = [False, True]
    print_results = True
    for dataset in datasets:
        for p in pruning:
            show_learning_curve(dataset, p, print_results=print_results)

if __name__ == '__main__':
    test_learning_curve()
