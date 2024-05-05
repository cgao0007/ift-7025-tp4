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

        # Initialisation de Xavier (le biais est le poids à l'index 0 pour chaque neurone)
        self.hidden_layer_weights = np.random.standard_normal((hidden_layer_size, input_size + 1)) / np.sqrt(input_size + 1)
        self.output_layer_weights = np.random.standard_normal((output_size, hidden_layer_size + 1)) / np.sqrt(hidden_layer_size + 1)

    def train(self, train, train_labels, epochs):  # vous pouvez rajouter d'autres attributs au besoin
        """
        C'est la méthode qui va entrainer votre modèle,
        train est une matrice de type Numpy et de taille nxm, avec
        n : le nombre d'exemple d'entrainement dans le dataset
        m : le nombre d'attributs (le nombre de caractéristiques)

        train_labels : est une matrice numpy de taille nx1

        epochs: le nombre d'époques d'entraînement (itérations sur tout le jeu d'entraînement)

        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire

        """
        for i in range(epochs):
            self.__train_for_epoch(train, train_labels)

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
        # Initialize variables
        epsilon = 1e-10  # Small constant to prevent division by zero
        num_classes = len(np.unique(y))
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        correct_predictions = 0

        # Iterate over all tests
        for i in range(len(X)):
            # Get prediction
            prediction = self.predict(X[i])

            # Update confusion matrix
            confusion_matrix[y[i], prediction] += 1

            # Check if prediction is correct
            if prediction == y[i]:
                correct_predictions += 1

        # Compute metrics
        accuracy = correct_predictions / len(X)
        precision = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0) + epsilon)
        recall = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)

        # Handle nan values in F1 if precision or recall is zero
        f1 = np.nan_to_num(f1)

        # Compute average precision, recall, and f1-score
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)

        return confusion_matrix, accuracy, avg_precision, avg_recall, avg_f1

    def __train_for_epoch(self, train, train_labels):
        total_output_layer_gradient = np.zeros(self.output_layer_weights.shape)
        total_hidden_layer_gradient = np.zeros(self.hidden_layer_weights.shape)
        batch_size = len(train)

        for i in range(batch_size):
            x = train[i]
            y = self.__convert_label_to_one_hot(train_labels[i])

            # --- Calcul du gradient ---

            # Ajout d'une entrée à 1 pour le biais
            in_h = np.concatenate([np.array([1]), x])
            out = np.matmul(self.hidden_layer_weights, in_h)
            o_h = self.__sigmoid(out)

            # Ajout d'une entrée à 1 pour le biais
            in_k = np.concatenate([np.array([1]), o_h])
            out = np.matmul(self.output_layer_weights, in_k)
            o_k = self.__sigmoid(out)

            d_k = o_k * (1 - o_k) * (y - o_k)

            # On exclut le biais de la couche de sortie puisqu'il n'affecte pas la couche cachée
            d_h = o_h * (1 - o_h) * np.matmul(d_k, self.output_layer_weights[:, 1:])

            total_output_layer_gradient += np.matmul(np.expand_dims(d_k, axis=1), np.expand_dims(in_k, axis=0))
            total_hidden_layer_gradient += np.matmul(np.expand_dims(d_h, axis=1), np.expand_dims(in_h, axis=0))

        # --- Mise à jour des poids et biais ---
        self.output_layer_weights += self.learning_rate * (total_output_layer_gradient / batch_size)
        self.hidden_layer_weights += self.learning_rate * (total_hidden_layer_gradient / batch_size)

    def __convert_label_to_one_hot(self, label):
        one_hot = np.zeros(self.output_size)
        one_hot[label] = 1

        return one_hot

    @staticmethod
    def __sigmoid(x):
        return 1. / (1. + np.exp(-x))
