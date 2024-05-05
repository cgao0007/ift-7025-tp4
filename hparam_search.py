import random

import numpy as np

from sklearn.neural_network import MLPClassifier

TRAIN_RATIO = 1.
N_FOLDS = 10

random.seed(42)


def k_fold_split(dataset, labels, n_folds):
    dataset_splits = np.array_split(dataset, n_folds)
    label_splits = np.array_split(labels, n_folds)

    return dataset_splits, label_splits


def train_test_split(dataset_splits, labels_splits, index):
    train_X = np.concatenate([dataset_splits[j] for j in range(len(dataset_splits)) if j != index])
    train_y = np.concatenate([labels_splits[j] for j in range(len(labels_splits)) if j != index])

    test_X = dataset_splits[index]
    test_y = labels_splits[index]

    return train_X, train_y, test_X, test_y


def compute_error_for_hidden_size(hidden_size, fold, dataset_split, labels_split):
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(hidden_size,),
                                   activation='logistic',
                                   solver='sgd',
                                   batch_size='auto')

    train_predictors, train_labels, test_predictors, test_labels = train_test_split(dataset_split, labels_split, fold)

    mlp_classifier.fit(train_predictors, train_labels)

    accuracy = mlp_classifier.score(test_predictors, test_labels)

    return 1. - accuracy


def compute_error_for_num_layers(num_layers, hidden_size, fold, dataset_split, labels_split):
    mlp_classifier = MLPClassifier(hidden_layer_sizes=tuple([hidden_size for _ in range(num_layers)]),
                                   activation='logistic',
                                   solver='sgd',
                                   batch_size='auto')

    train_predictors, train_labels, test_predictors, test_labels = train_test_split(dataset_split, labels_split, fold)

    mlp_classifier.fit(train_predictors, train_labels)

    accuracy = mlp_classifier.score(test_predictors, test_labels)

    return 1. - accuracy


def compute_mean_errors_for_hidden_layer_size(predictors, labels, sizes):
    predictor_splits, label_splits = k_fold_split(predictors, labels, N_FOLDS)

    mean_errors = np.ones(len(sizes))

    for i, n in enumerate(sizes):
        print(f"Computing mean error for hidden size {n}")
        errors_for_n_folds = np.ones(N_FOLDS)

        for f in range(N_FOLDS):
            error = compute_error_for_hidden_size(n, f, predictor_splits, label_splits)
            errors_for_n_folds[f] = error

        mean_errors[i] = np.mean(errors_for_n_folds)

    print(f"Hidden sizes: {sizes}")
    print(f"Mean errors: {mean_errors}")
    print(f"Min error: {np.min(mean_errors)}")
    print(f"Best hidden_size value: {sizes[np.argmin(mean_errors)]}\n")

    return mean_errors


def compute_mean_errors_for_nums_layers(predictors, labels, nums_layers, hidden_size):
    predictor_splits, label_splits = k_fold_split(predictors, labels, N_FOLDS)

    mean_errors = np.ones(len(nums_layers))

    for i, n in enumerate(nums_layers):
        print(f"Computing mean error for {n} layers")
        errors_for_n_folds = np.ones(N_FOLDS)

        for f in range(N_FOLDS):
            error = compute_error_for_num_layers(n, hidden_size, f, predictor_splits, label_splits)
            errors_for_n_folds[f] = error

        mean_errors[i] = np.mean(errors_for_n_folds)

    print(f"Numbers of layers: {nums_layers}")
    print(f"Mean errors: {mean_errors}")
    print(f"Min error: {np.min(mean_errors)}")
    print(f"Best hidden_size value: {nums_layers[np.argmin(mean_errors)]}\n")

    return mean_errors