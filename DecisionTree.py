"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenir au moins les 3 méthodes definies ici bas, 
    * train     : pour entraîner le modèle sur l'ensemble d'entrainement.
    * predict     : pour prédire la classe d'un exemple donné.
    * evaluate         : pour evaluer le classifieur avec les métriques demandées. 
"""

import numpy as np
from scipy.stats import chi2


class DecisionTree:

    def __init__(self, max_depth=None, pruning=False, p_value_threshold=0.05):
        """
        Arguments:
        - max_depth: int, maximum depth of the tree
        - pruning: bool, whether to enable pruning or not
        """
        self.tree = None
        self.max_depth = max_depth
        self.pruning = pruning
        self.p_value_threshold = p_value_threshold


    # Class Node: represents a node in the decision tree
    class Node:
        '''
        Class Node: represents a node in the decision tree
  
        Attributes:
        - feature_index: index of the feature used to split the node
        - threshold: threshold value used to split the node
        - left: left child node
        - right: right child node
        - value: value of the node if it is a leaf node
          '''
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
        
        
    def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
        """
        C'est la méthode qui va entrainer votre modèle,
        train est une matrice de type Numpy et de taille nxm, avec 
        n : le nombre d'exemple d'entrainement dans le dataset
        m : le nombre d'attributs (le nombre de caractéristiques)
        
        train_labels : est une matrice numpy de taille nx1
        """
        self.tree = self._build_tree(train, train_labels, depth=0)
    

    def _build_tree(self, data, labels, depth):
        # Base case checks for pure node or maximum depth
        if self._should_stop_splitting(labels, depth):
            return self.Node(value=self._calculate_leaf_value(labels))

        # Find the best feature and threshold to split on
        best_feature, best_threshold, best_left_idx, best_right_idx = self._find_best_split(data, labels)

        # Check if split found
        if best_feature is None:
            return self.Node(value=self._calculate_leaf_value(labels))

        # Build the subtrees recursively
        left_subtree = self._build_tree(data[best_left_idx], labels[best_left_idx], depth + 1)
        right_subtree = self._build_tree(data[best_right_idx], labels[best_right_idx], depth + 1)
        
        return self.Node(feature_index=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _should_stop_splitting(self, labels, depth):
        # Check if all labels are the same or if maximum depth has been reached
        return len(np.unique(labels)) == 1 or (self.max_depth is not None and depth >= self.max_depth)

    def _find_best_split(self, data, labels):
        _, num_features = data.shape
        
        # Initialize Gini score and best feature and threshold
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None

		# Iterate over all features
        for feature_index in range(num_features):
            feature_values = data[:, feature_index]
            possible_thresholds = np.unique(feature_values)

			# Iterate over all possible thresholds
            for threshold in possible_thresholds:
                left_idx, right_idx = self._split(data, feature_index, threshold)

				# Skip if either left or right indices are empty
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue

				# Calculate Gini impurity
                gini = self._calculate_gini(labels, left_idx, right_idx)

				# Update if better Gini score
                if gini < best_gini:
                    # Check if pruning is enabled
                    if self.pruning:
                        if self._chi_squared_test(labels, left_idx, right_idx):
                            best_gini = gini
                            best_feature = feature_index
                            best_threshold = threshold
                            best_left_idx = left_idx
                            best_right_idx = right_idx
                    
                    # No pruning
                    else:
                        best_gini = gini
                        best_feature = feature_index
                        best_threshold = threshold
                        best_left_idx = left_idx
                        best_right_idx = right_idx

		# Return the best feature and threshold
        return best_feature, best_threshold, best_left_idx, best_right_idx


    def _split(self, data, feature_index, threshold):
        # Get the indices of the samples that have the feature value less than or equal to the threshold
        left_idx = np.where(data[:, feature_index] <= threshold)[0]
        
        # Get the indices of the samples that have the feature value greater than the threshold
        right_idx = np.where(data[:, feature_index] > threshold)[0]
        
        # Return the left and right indices
        return left_idx, right_idx

    def _calculate_gini(self, labels, left_idx, right_idx):
        # Calculate Gini impurity for a split
        left_labels = labels[left_idx]
        right_labels = labels[right_idx]
        left_gini = 1.0 - sum([pow((np.sum(left_labels == label) / len(left_labels)), 2) for label in np.unique(left_labels)])
        right_gini = 1.0 - sum([pow((np.sum(right_labels == label) / len(right_labels)), 2) for label in np.unique(right_labels)])
        
        # Weighted average of the Gini impurity of left and right nodes
        total = len(left_idx) + len(right_idx)
        weighted_gini = (len(left_idx) / total) * left_gini + (len(right_idx) / total) * right_gini
        return weighted_gini
    
    def _chi_squared_test(self, labels, left_idx, right_idx):
        epsilon = 1e-10  # Small constant to prevent division by zero
        total_counts = np.bincount(labels)
        left_counts = np.bincount(labels[left_idx], minlength=total_counts.size)
        right_counts = np.bincount(labels[right_idx], minlength=total_counts.size)
        expected_left_counts = total_counts * (np.sum(left_idx) / len(labels)) + epsilon
        expected_right_counts = total_counts * (np.sum(right_idx) / len(labels)) + epsilon
        chi_stat = np.sum((left_counts - expected_left_counts)**2 / expected_left_counts) + \
                np.sum((right_counts - expected_right_counts)**2 / expected_right_counts)
        p_value = chi2.sf(chi_stat, df=len(total_counts) - 1)
        return p_value < self.p_value_threshold

    def _calculate_leaf_value(self, labels):
        # Majority class as the value of the leaf
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]
        
    def predict(self, x):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        # Start from the root
        node = self.tree
        
        # Iterate over all nodes in the tree
        while node.value is None:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        # Return the value of the leaf
        return node.value
        
    def evaluate(self, X, y):
        """
        c'est la méthode qui va évaluer votre modèle sur les données X
        l'argument X est une matrice de type Numpy et de taille nxm, avec 
        n : le nombre d'exemple de test dans le dataset
        m : le nombre d'attributs (le nombre de caractéristiques)
        
        y : est une matrice numpy de taille nx1
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
        # Initialize variables
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
