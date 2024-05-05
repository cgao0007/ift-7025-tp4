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

    def __init__(self, input_size, output_size, hidden_layer_size, learning_rate):
        """
        C'est un Initializer.
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate

        # Le biais est le poids à l'index 0 pour chaque neurone
        self.hidden_layer_weights = np.empty((hidden_layer_size, input_size + 1))
        self.output_layer_weights = np.empty((output_size, hidden_layer_size + 1))

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
        x = train[0]
        y = self.__convert_label_to_one_hot(train_labels[0])

        # --- Calcul du gradient ---

        # Ajout d'une entrée à 1 pour le biais
        in_h = np.concatenate([np.array([1]), x])
        out = np.matmul(self.hidden_layer_weights, in_h)
        o_h = self.__sigmoid(out)

        in_k = np.concatenate([np.array([1]), o_h])
        out = np.matmul(self.output_layer_weights, in_k)
        o_k = self.__sigmoid(out)

        d_k = o_k * (1 - o_k) * (y - o_k)

        # On exclut le biais de la couche de sortie puisqu'il n'affecte pas la couche cachée
        d_h = o_h * (1 - o_h) * np.matmul(d_k, self.output_layer_weights[:, 1:])

        # --- Mise à jour des poids et biais ---

        self.output_layer_weights += self.learning_rate * np.matmul(np.expand_dims(d_k, axis=1), np.expand_dims(in_k, axis=0))
        self.hidden_layer_weights += self.learning_rate * np.matmul(np.expand_dims(d_h, axis=1), np.expand_dims(in_h, axis=0))

        print("allo")

    def predict(self, x):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        out = np.concatenate([np.array([1]), x])
        out = np.matmul(self.hidden_layer_weights, out)
        out = self.__sigmoid(out)

        out = np.concatenate([np.array([1]), out])
        out = np.matmul(self.output_layer_weights, out)
        out = self.__sigmoid(out)

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

    def __convert_label_to_one_hot(self, label):
        one_hot = np.zeros(self.output_size)
        one_hot[label] = 1

        return one_hot

    @staticmethod
    def __sigmoid(x):
        return 1. / (1. + np.exp(-x))
