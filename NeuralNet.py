"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenir au moins les 3 méthodes definies ici bas, 
    * train 	: pour entraîner le modèle sur l'ensemble d'entrainement.
    * predict 	: pour prédire la classe d'un exemple donné.
    * evaluate 		: pour evaluer le classifieur avec les métriques demandées.
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""
import numpy as np


# le nom de votre classe
# DecisionTree pour l'arbre de décision
# NeuralNet pour le réseau de neurones

class NeuralNet:  # nom de la class à changer

    def __init__(self, input_size, output_size, hidden_size):
        """
        C'est un Initializer.
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        self.n_inputs = input_size
        self.n_outputs = output_size
        self.hidden_layer_size = hidden_size

        # Le biais est le poids à l'index 0 pour chaque neurone
        self.input_weights = np.empty((hidden_size, input_size + 1))
        self.output_weights = np.empty((output_size, hidden_size + 1))

    def train(self, train, train_labels):  # vous pouvez rajouter d'autres attributs au besoin
        """
        C'est la méthode qui va entrainer votre modèle,
        train est une matrice de type Numpy et de taille nxm, avec
        n : le nombre d'exemple d'entrainement dans le dataset
        m : le nombre d'attributs (le nombre de caractéristiques)

        train_labels : est une matrice numpy de taille nx1

        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire

        """
        # En supposant qu'on a un batch size équivalent à la taille du jeu de données
        outputs = [self.__forward(x) for x in train]
        losses = [self.__compute_loss(outputs[i], train_labels[i]) for i in range(len(train_labels))]


        pass

    def predict(self, x):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        out = self.__forward(x)
        return np.argmax(out)

    def evaluate(self, X, y):
        """
        c'est la méthode qui va évaluer votre modèle sur les données X
        l'argument X est une matrice de type Numpy et de taille nxm, avec
        n : le nombre d'exemple de test dans le dataset
        m : le nombre d'attributs (le nombre de caractéristiques)

        y : est une matrice numpy de taille nx1

        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire
        """
    pass

    def __forward(self, x):
        out = np.concatenate([np.array([1]), x])
        out = np.matmul(self.input_weights, out)
        out = self.__sigmoid(out)

        out = np.concatenate([np.array([1]), out])
        out = np.matmul(self.output_weights, out)
        out = self.__sigmoid(out)

        return out

    # MSE utilisée pour simplifier le calcul du gradient
    def __compute_loss(self, output, label):
        pass

    def __convert_label_to_one_hot(self, label):
        one_hot = np.zeros(self.output_weights)
        one_hot[label] = 1

        return one_hot

    @staticmethod
    def __sigmoid(x):
        return 1. / (1. + np.exp(-x))
