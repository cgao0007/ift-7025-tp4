import numpy as np
import sys

from DecisionTree import DecisionTree
from NeuralNet import NeuralNet
from hparam_search import compute_mean_errors_for_hidden_layer_size, compute_mean_errors_for_nums_layers
from learning_curve import test_learning_curve
from load_datasets import load_abalone_dataset, load_iris_dataset, load_wine_dataset
from pruning import test_pruning
from matplotlib import pyplot as plt
from scikit_classifiers import test_scikit_dt_abalone, test_scikit_dt_iris, test_scikit_dt_wine, test_scikit_nn_abalone, test_scikit_nn_iris, test_scikit_nn_wine


"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entraîner votre classifieur
4- Le tester
"""

########################
# 0 - Global Variables #
########################
# Datasets
ABALONE = 'abalone'
IRIS = 'iris'
WINE = 'wine'
ABALONE_FEATURES_WITHOUT_ONE_HOT = 8
ABALONE_FEATURES_WITH_ONE_HOT = 10
IRIS_FEATURES = 4
WINE_FEATURES = 11

# Train Ratio
TRAIN_RATIO = 0.7

# Values for hparam search
HIDDEN_LAYER_SIZES = [5, 50, 100, 250, 500]
NUMBERS_OF_HIDDEN_LAYERS = range(1, 6)

# Tests to run
TEST_DECISION_TREE = False
TEST_NEURAL_NET = False
LEARNING_CURVE = False
PRUNING = False
NEURAL_NET_HIDDEN_SIZE_SEARCH = False
NEURAL_NET_NUM_LAYERS_SEARCH = False
TEST_OPTIMAL_NEURAL_NET = True
TEST_ALL_CLASSIFIERS = False

#########################################################
# 1 - Initialiser votre classifieur avec ses paramètres #
#########################################################
# Hyperparameters - Decision Trees
DECISION_TREE_MAX_DEPTHS = {
    ABALONE: 8,
    IRIS: 2,
    WINE: 10,
}

# Hyperparameters - Neural Networks
NEURAL_NET_LEARNING_RATES = {
    ABALONE: 0.1,
    IRIS: 0.1,
    WINE: 0.2,
}
NEURAL_NET_BATCH_SIZES = {
    ABALONE: 200,
    IRIS: 20,
    WINE: 50,
}
NEURAL_NET_EPOCHS = {
    ABALONE: 1000,
    IRIS: 1000,
    WINE: 200,
}
NEURAL_NET_HIDDEN_SIZES = {
    ABALONE: 7,
    IRIS: 4,
    WINE: 8,
}
NEURAL_NET_OUTPUT_SIZES = {
    ABALONE: 3,
    IRIS: 3,
    WINE: 2,
}
LAYER_SIZES_FOUND_BY_SEARCH = {
    ABALONE: 50,
    IRIS: 50,
    WINE: 100
}
NUM_LAYERS_FOUND_BY_SEARCH = {
    ABALONE: 1,
    IRIS: 1,
    WINE: 1
}

# Classifiers - Decision Trees
dt_abalone = DecisionTree(max_depth=DECISION_TREE_MAX_DEPTHS[ABALONE])
dt_iris = DecisionTree(max_depth=DECISION_TREE_MAX_DEPTHS[IRIS])
dt_wine = DecisionTree(max_depth=DECISION_TREE_MAX_DEPTHS[WINE])

# Classifiers - Neural Networks
nn_abalone = NeuralNet(input_size=ABALONE_FEATURES_WITH_ONE_HOT,
                       hidden_layer_size=NEURAL_NET_HIDDEN_SIZES[ABALONE],
                       output_size=NEURAL_NET_OUTPUT_SIZES[ABALONE],
                       learning_rate=NEURAL_NET_LEARNING_RATES[ABALONE],
                       batch_size=NEURAL_NET_BATCH_SIZES[ABALONE])
nn_iris = NeuralNet(input_size=IRIS_FEATURES,
                    hidden_layer_size=NEURAL_NET_HIDDEN_SIZES[IRIS],
                    output_size=NEURAL_NET_OUTPUT_SIZES[IRIS],
                    learning_rate=NEURAL_NET_LEARNING_RATES[IRIS],
                    batch_size=NEURAL_NET_BATCH_SIZES[IRIS])
nn_wine = NeuralNet(input_size=WINE_FEATURES,
                    hidden_layer_size=NEURAL_NET_HIDDEN_SIZES[WINE],
                    output_size=NEURAL_NET_OUTPUT_SIZES[WINE],
                    learning_rate=NEURAL_NET_LEARNING_RATES[WINE],
                    batch_size=NEURAL_NET_BATCH_SIZES[WINE])


############################
# 2 - Charger les datasets #
############################
iris_train, iris_train_labels, iris_test, iris_test_labels = load_iris_dataset(TRAIN_RATIO)
wine_train, wine_train_labels, wine_test, wine_test_labels = load_wine_dataset(TRAIN_RATIO)
abalone_train, abalone_train_labels, abalone_test, abalone_test_labels = load_abalone_dataset(TRAIN_RATIO)
abalone_train_one_hot, abalone_train_labels_one_hot, abalone_test_one_hot, abalone_test_labels_one_hot = load_abalone_dataset(TRAIN_RATIO, one_hot_encoding=True)


###################################
# 3 - Entrainer votre classifieur #
###################################

if TEST_DECISION_TREE:
    dt_abalone.train(abalone_train, abalone_train_labels)
    dt_iris.train(iris_train, iris_train_labels)
    dt_wine.train(wine_train, wine_train_labels)

if TEST_NEURAL_NET:
    nn_abalone.train(abalone_train_one_hot, abalone_train_labels_one_hot, epochs=NEURAL_NET_EPOCHS[ABALONE])
    nn_iris.train(iris_train, iris_train_labels, epochs=NEURAL_NET_EPOCHS[IRIS])
    nn_wine.train(wine_train, wine_train_labels, epochs=NEURAL_NET_EPOCHS[WINE])


################################
# 4 - Tester votre classifieur #
################################
def print_results(cm, accuracy, precision, recall, f1):
    print(f'Confusion Matrix:\n{cm}')
    print(f'Accuracy: {accuracy * 100:.5f}%')
    print(f'Precision: {precision * 100:.5f}%')
    print(f'Recall: {recall * 100:.5f}%')
    print(f'F1 Score: {f1 * 100:.5f}%')

def test_dt_iris():
    # Evaluate Decision Tree on Iris test dataset
    print('\nDecision Tree - Iris:')
    print(f'Decision Tree - Max Depth: {dt_iris.max_depth}')
    cm, accuracy, precision, recall, f1 = dt_iris.evaluate(iris_test, iris_test_labels)
    print_results(cm, accuracy, precision, recall, f1)

def test_nn_iris():
    # Evaluate Neural Network on Iris test dataset
    print('\nNeural Network - Iris:')
    cm, accuracy, precision, recall, f1 = nn_iris.evaluate(iris_test, iris_test_labels)
    print_results(cm, accuracy, precision, recall, f1)

def test_dt_wine():
    # Evaluate Decision Tree on Wine test dataset
    print('\nDecision Tree - Wine:')
    print(f'Decision Tree - Max Depth: {dt_wine.max_depth}')
    cm, accuracy, precision, recall, f1 = dt_wine.evaluate(wine_test, wine_test_labels)
    print_results(cm, accuracy, precision, recall, f1)

def test_nn_wine():
    # Evaluate Neural Network on Wine test dataset
    print('\nNeural Network - Wine:')
    cm, accuracy, precision, recall, f1 = nn_wine.evaluate(wine_test, wine_test_labels)
    print_results(cm, accuracy, precision, recall, f1)

def test_dt_abalone():
    # Evaluate Decision Tree on Abalone test dataset
    print('\nDecision Tree - Abalone:')
    print(f'Decision Tree - Max Depth: {dt_abalone.max_depth}')
    cm, accuracy, precision, recall, f1 = dt_abalone.evaluate(abalone_test, abalone_test_labels)
    print_results(cm, accuracy, precision, recall, f1)

def test_nn_abalone():
    # Evaluate Neural Network on Abalone test dataset
    print('\nNeural Network - Abalone:')
    cm, accuracy, precision, recall, f1 = nn_abalone.evaluate(abalone_test_one_hot, abalone_test_labels_one_hot)
    print_results(cm, accuracy, precision, recall, f1)


############
# 5 - Main #
############
def main():
    print('Evaluation on Test Datasets:')

    # Test on each classifier
    if TEST_DECISION_TREE:
        test_dt_iris()
        test_dt_wine()
        test_dt_abalone()
        test_scikit_dt_iris()
        test_scikit_dt_wine()
        test_scikit_dt_abalone()

    if TEST_NEURAL_NET:
        test_nn_wine()
        test_nn_iris()
        test_nn_abalone()

    # Pruning
    if PRUNING:
        test_pruning()

    # Learning Curve
    if LEARNING_CURVE:
        test_learning_curve()

    # Neural Net hidden size hparam search
    if NEURAL_NET_HIDDEN_SIZE_SEARCH:
        print("Iris hidden size search:")
        iris_mean_errors = compute_mean_errors_for_hidden_layer_size(iris_train, iris_train_labels, HIDDEN_LAYER_SIZES)
        print("Wine hidden size search:")
        wine_mean_errors = compute_mean_errors_for_hidden_layer_size(wine_train, wine_train_labels, HIDDEN_LAYER_SIZES)
        print("Abalone hidden size search:")
        abalone_mean_errors = compute_mean_errors_for_hidden_layer_size(abalone_train_one_hot, abalone_train_labels_one_hot, HIDDEN_LAYER_SIZES)

        plt.xticks(HIDDEN_LAYER_SIZES)
        plt.plot(HIDDEN_LAYER_SIZES, iris_mean_errors, label="Iris")
        plt.plot(HIDDEN_LAYER_SIZES, wine_mean_errors, label="Wine")
        plt.plot(HIDDEN_LAYER_SIZES, abalone_mean_errors, label="Abalone")
        plt.xlabel("Neurones de la couche cachée")
        plt.ylabel("Erreur en validation croisée à 10 plis")
        plt.legend()
        plt.show()

    # Neural Net number of layers hparam search
    if NEURAL_NET_NUM_LAYERS_SEARCH:
        print("Iris num layers search:")
        iris_mean_errors = compute_mean_errors_for_nums_layers(iris_train, iris_train_labels, NUMBERS_OF_HIDDEN_LAYERS, LAYER_SIZES_FOUND_BY_SEARCH[IRIS])
        print("Wine num layers search:")
        wine_mean_errors = compute_mean_errors_for_nums_layers(wine_train, wine_train_labels, NUMBERS_OF_HIDDEN_LAYERS, LAYER_SIZES_FOUND_BY_SEARCH[WINE])
        print("Abalone num layers search:")
        abalone_mean_errors = compute_mean_errors_for_nums_layers(abalone_train_one_hot, abalone_train_labels_one_hot, NUMBERS_OF_HIDDEN_LAYERS, LAYER_SIZES_FOUND_BY_SEARCH[ABALONE])

        plt.xticks(NUMBERS_OF_HIDDEN_LAYERS)
        plt.plot(NUMBERS_OF_HIDDEN_LAYERS, iris_mean_errors, label="Iris")
        plt.plot(NUMBERS_OF_HIDDEN_LAYERS, wine_mean_errors, label="Wine")
        plt.plot(NUMBERS_OF_HIDDEN_LAYERS, abalone_mean_errors, label="Abalone")
        plt.xlabel("Nombre de couches")
        plt.ylabel("Erreur en validation croisée à 10 plis")
        plt.legend()
        plt.show()

    if TEST_OPTIMAL_NEURAL_NET:
        test_scikit_nn_iris(NUM_LAYERS_FOUND_BY_SEARCH[IRIS], LAYER_SIZES_FOUND_BY_SEARCH[IRIS])
        test_scikit_nn_wine(NUM_LAYERS_FOUND_BY_SEARCH[WINE], LAYER_SIZES_FOUND_BY_SEARCH[WINE])
        test_scikit_nn_abalone(NUM_LAYERS_FOUND_BY_SEARCH[ABALONE], LAYER_SIZES_FOUND_BY_SEARCH[ABALONE])

    # Scikit Classifiers
    # test_scikit_dt_iris()
    # test_scikit_dt_wine()
    # test_scikit_dt_abalone()
    # test_scikit_nn_iris()
    # test_scikit_nn_wine()
    # test_scikit_nn_abalone()


if __name__ == '__main__':
    main()
