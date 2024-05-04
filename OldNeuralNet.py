"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenir au moins les 3 méthodes definies ici bas, 
    * train     : pour entraîner le modèle sur l'ensemble d'entrainement.
    * predict     : pour prédire la classe d'un exemple donné.
    * evaluate         : pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

import numpy as np


#TODO enlever

# le nom de votre classe
# DecisionTree pour l'arbre de décision
# NeuralNet pour le réseau de neurones

class OldNeuralNet: #nom de la class à changer

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=100):
        """
        C'est un Initializer. 
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        # Initialize weights and biases
        self.weights_input_to_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        
        self.weights_hidden_to_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))
        
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, x):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid function.
        """
        return x * (1 - x)
        
        
    def train(self, train, train_labels):
        """
        C'est la méthode qui va entrainer votre modèle,
        train est une matrice de type Numpy et de taille nxm, avec 
        n : le nombre d'exemple d'entrainement dans le dataset
        m : le nombre d'attributs (le nombre de caractéristiques)
        
        train_labels : est une matrice numpy de taille nx1
        
        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire
        
        """
        for epoch in range(self.epochs):
            for x, y in zip(train, train_labels):
                x = x.reshape(1, -1)  # Reshape x to be a row vector
                y = y.reshape(1, -1)  # Reshape y to be a row vector
                
                # Forward pass
                hidden_layer_input = np.dot(x, self.weights_input_to_hidden) + self.bias_hidden
                hidden_layer_output = self.sigmoid(hidden_layer_input)
                
                output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_to_output) + self.bias_output
                predicted_output = self.sigmoid(output_layer_input)
                
                # Calculate error
                error = y - predicted_output
                
                # Backpropagation
                d_predicted_output = error * self.sigmoid_derivative(predicted_output)
                
                error_hidden_layer = d_predicted_output.dot(self.weights_hidden_to_output.T)
                d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_layer_output)
                
                # Update weights and biases
                self.weights_hidden_to_output += hidden_layer_output.T.dot(d_predicted_output) * self.learning_rate
                self.bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * self.learning_rate
                
                self.weights_input_to_hidden += x.T.dot(d_hidden_layer) * self.learning_rate
                self.bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate

            if epoch % 10 == 0:
                loss = np.mean(np.square(y - predicted_output))
                print(f"Epoch {epoch}, Loss: {loss}")
        
    def predict(self, x):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        # Forward pass to compute the output
        hidden_layer_input = np.dot(x, self.weights_input_to_hidden) + self.bias_hidden
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_to_output) + self.bias_output
        predicted_output = self.sigmoid(output_layer_input)
        
        # Convert probabilities to class labels (0 or 1)
        return (predicted_output > 0.5).astype(int)
        
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
        num_classes = len(np.unique(y))
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        correct_predictions = 0

        # Iterate over all tests
        for i in range(len(X)):
            # Get prediction
            prediction = self.predict(X[i])
            
            # Update confusion matrix
            confusion_matrix[int(y[i]), int(prediction)] += 1

            # Check if prediction is correct
            if prediction == y[i]:
                correct_predictions += 1

        # Compute metrics
        accuracy = correct_predictions / len(X)
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
            recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
            f1 = 2 * (precision * recall) / (precision + recall)

        # Replace NaN values with zero for precision and F1 score
        precision = np.nan_to_num(precision)
        f1 = np.nan_to_num(f1)

        # Compute average precision, recall, and F1 score
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)
        
        return confusion_matrix, accuracy, avg_precision, avg_recall, avg_f1
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
        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        f1 = 2 * (precision * recall) / (precision + recall)

        # Compute average precision, recall, and f1-score
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)
        
        return confusion_matrix, accuracy, avg_precision, avg_recall, avg_f1
        
    
    # Vous pouvez rajouter d'autres méthodes et fonctions,
    # il suffit juste de les commenter.