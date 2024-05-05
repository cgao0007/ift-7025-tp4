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

class NeuralNet:

    def __init__(self, input_size, output_size, hidden_layer_size, learning_rate, batch_size=None):
        """
        C'est un Initializer.
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Initialisation de Xavier (le biais est le poids à l'index 0 pour chaque neurone)
        self.hidden_layer_weights = np.random.normal(loc=0.,
                                                     scale=np.sqrt(2 / (hidden_layer_size + input_size + 1)),
                                                     size=(hidden_layer_size, input_size + 1))
        self.output_layer_weights = np.random.normal(loc=0.,
                                                     scale=np.sqrt(2 / (output_size + hidden_layer_size + 1)),
                                                     size=(output_size, hidden_layer_size + 1))

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
        concatenated_train_data = np.concatenate([train, np.expand_dims(train_labels, axis=1)], axis=1)

        # On met les données d'entraînement dans un ordre aléatoire pour éviter de ne voir qu'une classe à la fois
        np.random.shuffle(concatenated_train_data)

        if self.batch_size is None:
            train_batches = [concatenated_train_data]
        else:
            approximate_number_of_batches = len(train) // self.batch_size
            train_batches = np.array_split(concatenated_train_data, approximate_number_of_batches)

        for i in range(epochs):
            # On mélange les batches pour avoir un ordre différent à chaque itération
            np.random.shuffle(train_batches)

            for j in range(len(train_batches)):
                batch = train_batches[j]
                self.__train_with_batch(batch[:, :-1], batch[:, -1].astype(int))

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

    def __train_with_batch(self, train, train_labels):
        total_output_layer_gradient = np.zeros(self.output_layer_weights.shape)
        total_hidden_layer_gradient = np.zeros(self.hidden_layer_weights.shape)
        size_of_current_batch = len(train)

        for i in range(size_of_current_batch):
            x = train[i]
            y = self.__convert_label_to_one_hot(train_labels[i])

            # --- Calcul du gradient ---

            # Ajout d'une entrée à 1 pour multiplier le biais
            in_h = np.concatenate([np.array([1]), x])
            out = np.matmul(self.hidden_layer_weights, in_h)
            o_h = self.__sigmoid(out)

            # Ajout d'une entrée à 1 pour multiplier le biais
            in_k = np.concatenate([np.array([1]), o_h])
            out = np.matmul(self.output_layer_weights, in_k)
            o_k = self.__sigmoid(out)

            d_k = o_k * (1 - o_k) * (y - o_k)
            d_h = o_h * (1 - o_h) * np.matmul(d_k, self.output_layer_weights[:, 1:])

            # Ajout des gradients de chaque poids
            total_output_layer_gradient += np.matmul(np.expand_dims(d_k, axis=1), np.expand_dims(in_k, axis=0))
            total_hidden_layer_gradient += np.matmul(np.expand_dims(d_h, axis=1), np.expand_dims(in_h, axis=0))

        # --- Mise à jour des poids et biais ---

        # Pour chaque poids, on utilise le gradient moyen sur toute la batch
        self.output_layer_weights += self.learning_rate * (total_output_layer_gradient / size_of_current_batch)
        self.hidden_layer_weights += self.learning_rate * (total_hidden_layer_gradient / size_of_current_batch)

    def __convert_label_to_one_hot(self, label):
        one_hot = np.zeros(self.output_size)
        one_hot[label] = 1

        return one_hot

    @staticmethod
    def __sigmoid(x):
        return 1. / (1. + np.exp(-x))
