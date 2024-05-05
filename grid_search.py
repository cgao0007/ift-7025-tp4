import random

import numpy as np

from Knn import Knn
from load_datasets import load_iris_dataset, load_wine_dataset, load_abalone_dataset

TRAIN_RATIO = 1.
N_NEIGHBORS = [5, 10, 20, 50]
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


def compute_error_for_fold(n_neighbors, fold, dataset_split, labels_split):
    knn = Knn(n_neighbors=n_neighbors)

    train_predictors, train_labels, test_predictors, test_labels = train_test_split(dataset_split, labels_split, fold)

    knn.train(train_predictors, train_labels)

    _, accuracy, _, _, _ = knn.evaluate(test_predictors, test_labels)

    return 1. - accuracy


iris, iris_labels, _, _ = load_iris_dataset(TRAIN_RATIO)
wine, wine_labels, _, _ = load_wine_dataset(TRAIN_RATIO)
abalone, abalone_labels, _, _ = load_abalone_dataset(TRAIN_RATIO)

iris_split, iris_labels_split = k_fold_split(iris, iris_labels, N_FOLDS)
wine_split, wine_labels_split = k_fold_split(wine, wine_labels, N_FOLDS)
abalone_split, abalone_labels_split = k_fold_split(abalone, abalone_labels, N_FOLDS)

iris_errors = np.ones(len(N_NEIGHBORS))
wine_errors = np.ones(len(N_NEIGHBORS))
abalone_errors = np.ones(len(N_NEIGHBORS))

for i, n in enumerate(N_NEIGHBORS):
    iris_errors_for_n_folds = np.ones(N_FOLDS)
    wine_errors_for_n_folds = np.ones(N_FOLDS)
    abalone_errors_for_n_folds = np.ones(N_FOLDS)

    for f in range(N_FOLDS):
        print(f"Computing error for {n} neighbors, split #{f}...")

        print("Computing error for iris...")
        error = compute_error_for_fold(n, f, iris_split, iris_labels_split)
        print(f"error: {error}")
        iris_errors_for_n_folds[f] = error

        print("Computing error for wine...")
        error = compute_error_for_fold(n, f, wine_split, wine_labels_split)
        print(f"error: {error}")
        wine_errors_for_n_folds[f] = error

        print("Computing error for abalone...")
        error = compute_error_for_fold(n, f, abalone_split, abalone_labels_split)
        print(f"error: {error}")
        abalone_errors_for_n_folds[f] = error

    iris_errors[i] = np.mean(iris_errors_for_n_folds)
    wine_errors[i] = np.mean(wine_errors_for_n_folds)
    abalone_errors[i] = np.mean(abalone_errors_for_n_folds)

# On fait la moyenne des les trois jeux de données pour l'erreur donnée par chaque valeur de n_neighbors
# Il s'agit d'une métrique additionnelle qui n'a finalement pas été utilisée directement
n_neighbors_error_means = np.array([np.mean(np.array([iris_errors[i], wine_errors[i], abalone_errors[i]])) for i in
                                    range(len(N_NEIGHBORS))])

print(f"n_neighbors values: {N_NEIGHBORS}")

print(f"Iris errors: {iris_errors}")
print(f"Min error: {np.min(iris_errors)}")
print(f"Best n_neighbors value: {N_NEIGHBORS[np.argmin(iris_errors)]}\n")

print(f"Wine errors: {wine_errors}")
print(f"Min error: {np.min(wine_errors)}")
print(f"Best n_neighbors value: {N_NEIGHBORS[np.argmin(wine_errors)]}\n")

print(f"Abalone errors: {abalone_errors}")
print(f"Min error: {np.min(abalone_errors)}")
print(f"Best n_neighbors value: {N_NEIGHBORS[np.argmin(abalone_errors)]}\n")

print(f"Error means: {n_neighbors_error_means}")
print(f"Min error: {np.min(n_neighbors_error_means)}")
print(f"Best n_neighbors value: {N_NEIGHBORS[np.argmin(n_neighbors_error_means)]}")
