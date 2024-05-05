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

def print_results(cm, accuracy, precision, recall, f1):
    print(f'Confusion Matrix:\n{cm}')
    print(f'Accuracy: {accuracy * 100:.5f}%')
    print(f'Precision: {precision * 100:.5f}%')
    print(f'Recall: {recall * 100:.5f}%')
    print(f'F1 Score: {f1 * 100:.5f}%')

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

# Test the classifier
def test_classifier(dataset, pruning):
    # Get the classifier & the data
    classifier = get_classifier(dataset, pruning)
    train, train_labels, test, test_labels = get_data(dataset)
    print(f'\nDecision Tree - {dataset}:')
    print(f'Max Depth: {classifier.max_depth}')
    print(f'Pruning: {"enabled" if classifier.pruning else "disabled"}{"" if not classifier.pruning else f" (p-value threshold: {classifier.p_value_threshold})"}')

    # Train
    classifier.train(train, train_labels)

    # Evaluate and store the accuracy
    cm, accuracy, precision, recall, f1 = classifier.evaluate(test, test_labels)
    print_results(cm, accuracy, precision, recall, f1)

def main():
    datasets = [ABALONE, IRIS, WINE]
    pruning = [False, True]
    for dataset in datasets:
        for p in pruning:
            test_classifier(dataset, p)

if __name__ == '__main__':
    main()
