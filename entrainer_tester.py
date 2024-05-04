import numpy as np
import sys

from DecisionTree import DecisionTree
from load_datasets import load_abalone_dataset, load_iris_dataset, load_wine_dataset
from NeuralNet import NeuralNet
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
ABALONE_FEATURES = 8
IRIS_FEATURES = 4
WINE_FEATURES = 11

TRAIN_RATIO = 0.7


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
    ABALONE: 0.01,
    IRIS: 0.09,
    WINE: 0.01,
}
NEURAL_NET_EPOCHS = {
    ABALONE: 100,
    IRIS: 100,
    WINE: 100,
}
NEURAL_NET_HIDDEN_SIZES = {
    ABALONE: 3,
    IRIS: 3,
    WINE: 3,
}
NEURAL_NET_OUTPUT_SIZES = {
    ABALONE: 1,
    IRIS: 1,
    WINE: 1,
}

# Classifiers - Decision Trees
dt_abalone = DecisionTree(max_depth=DECISION_TREE_MAX_DEPTHS[ABALONE])
dt_iris = DecisionTree(max_depth=DECISION_TREE_MAX_DEPTHS[IRIS])
dt_wine = DecisionTree(max_depth=DECISION_TREE_MAX_DEPTHS[WINE])

# Classifiers - Neural Networks
nn_abalone = NeuralNet(input_size=ABALONE_FEATURES,
                       hidden_size=NEURAL_NET_HIDDEN_SIZES[ABALONE],
                       output_size=NEURAL_NET_OUTPUT_SIZES[ABALONE],
                       learning_rate=NEURAL_NET_LEARNING_RATES[ABALONE],
                       epochs=NEURAL_NET_EPOCHS[ABALONE])
nn_iris = NeuralNet(input_size=IRIS_FEATURES,
                    hidden_size=NEURAL_NET_HIDDEN_SIZES[IRIS],
                    output_size=NEURAL_NET_OUTPUT_SIZES[IRIS],
                    learning_rate=NEURAL_NET_LEARNING_RATES[IRIS],
                    epochs=NEURAL_NET_EPOCHS[IRIS])
nn_wine = NeuralNet(input_size=WINE_FEATURES,
                    hidden_size=NEURAL_NET_HIDDEN_SIZES[WINE],
                    output_size=NEURAL_NET_OUTPUT_SIZES[WINE],
                    learning_rate=NEURAL_NET_LEARNING_RATES[WINE],
                    epochs=NEURAL_NET_EPOCHS[WINE])


############################
# 2 - Charger les datasets #
############################
iris_train, iris_train_labels, iris_test, iris_test_labels = load_iris_dataset(TRAIN_RATIO)
wine_train, wine_train_labels, wine_test, wine_test_labels = load_wine_dataset(TRAIN_RATIO)
abalone_train, abalone_train_labels, abalone_test, abalone_test_labels = load_abalone_dataset(TRAIN_RATIO)


###################################
# 3 - Entrainer votre classifieur #
###################################
# Abalone
dt_abalone.train(abalone_train, abalone_train_labels)
nn_abalone.train(abalone_train, abalone_train_labels)

# Iris
dt_iris.train(iris_train, iris_train_labels)
nn_iris.train(iris_train, iris_train_labels)

# Wine
dt_wine.train(wine_train, wine_train_labels)
nn_wine.train(wine_train, wine_train_labels)


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
    cm, accuracy, precision, recall, f1 = nn_abalone.evaluate(abalone_test, abalone_test_labels)
    print_results(cm, accuracy, precision, recall, f1)


############
# 5 - Main #
############
def main():
    print('Evaluation on Test Datasets:')
    
    # Test on each classifier
    test_dt_iris()
    test_nn_iris()
    test_dt_wine()
    test_nn_wine()
    test_dt_abalone()
    test_nn_abalone()
    
    # Scikit Classifiers
    test_scikit_dt_iris()
    test_scikit_dt_wine()
    test_scikit_dt_abalone()
    test_scikit_nn_iris()
    test_scikit_nn_wine()
    test_scikit_nn_abalone()


if __name__ == '__main__':
    main()
