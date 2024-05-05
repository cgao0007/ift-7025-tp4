"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenir au moins les 3 méthodes definies ici bas, 
    * train     : pour entraîner le modèle sur l'ensemble d'entrainement.
    * predict     : pour prédire la classe d'un exemple donné.
    * evaluate         : pour evaluer le classifieur avec les métriques demandées. 
"""

from __future__ import annotations
import numpy as np
from scipy.stats import chi2


class DecisionTree:

    def __init__(self, max_depth: int=None, pruning: bool=False, p_value_threshold: float=0.05):
        """
        Arguments:
        - max_depth: int, maximum depth of the tree
        - pruning: bool, whether to enable pruning or not
        - p_value_threshold: float, threshold for p-value for chi-squared test
        """
        self.tree: DecisionTree.Node = None
        self.max_depth: int = max_depth
        self.pruning: bool = pruning
        self.p_value_threshold: float = p_value_threshold


    # Class Node: represents a node in the decision tree
    class Node:
        '''
        Class Node: represents a node in the decision tree

        A node can be either a leaf or a non-leaf node:
        - If it is a leaf node, it will have a value
        - If it is a non-leaf node, it will have a feature_index, threshold, left and right child nodes
  
        Attributes:
        - feature_index: int, index of the feature used to split the node
        - threshold: float, threshold value used to split the node
        - left: DecisionTree.Node, left child node
        - right: DecisionTree.Node, right child node
        - value: float, value of the node if it is a leaf node
        '''
        def __init__(self, feature_index: int=None, threshold: float=None, left: DecisionTree.Node=None, right: DecisionTree.Node=None, value: float=None):
            self.feature_index: int = feature_index
            self.threshold: float = threshold
            self.left: DecisionTree.Node = left
            self.right: DecisionTree.Node = right
            self.value: float = value
        
        
    def train(self, train, train_labels):
        """
        C'est la méthode qui va entrainer votre modèle,
        train est une matrice de type Numpy et de taille nxm, avec 
        n : le nombre d'exemple d'entrainement dans le dataset
        m : le nombre d'attributs (le nombre de caractéristiques)
        
        train_labels : est une matrice numpy de taille nx1
        """
        # Build the decision tree
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
        
        # Return the non-leaf node
        return self.Node(feature_index=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _should_stop_splitting(self, labels, depth):
        # Check if maximum depth has been reached
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        
        # Check if all labels are the same
        if len(np.unique(labels)) == 1:
            return True
        
        # Otherwise, continue
        return False

    def _find_best_split(self, data, labels):
        # Get number of features
        _, num_features = data.shape
        
        # Initialize Gini score and best feature and threshold
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None

        # Helper to update Gini
        def update_gini(gini, feature, threshold, left_idx, right_idx):
            # Check if gini is less than best gini
            if gini < best_gini:
                # Update best gini
                return gini, feature, threshold, left_idx, right_idx
            else:
                # Otherwise, return previous best
                return best_gini, best_feature, best_threshold, best_left_idx, best_right_idx

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

                # Check if pruning is enabled
                if self.pruning:
                    if self._chi_squared_test(labels, left_idx, right_idx):
                        best_gini, best_feature, best_threshold, best_left_idx, best_right_idx = update_gini(gini, feature_index, threshold, left_idx, right_idx)
                else:
                    best_gini, best_feature, best_threshold, best_left_idx, best_right_idx = update_gini(gini, feature_index, threshold, left_idx, right_idx)

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

        # Compute left Gini
        left_gini = 1.0
        for label in np.unique(left_labels):
            p = np.sum(left_labels == label) / len(left_labels)
            left_gini -= pow(p, 2)
        
        # Compute right Gini
        right_gini = 1.0
        for label in np.unique(right_labels):
            p = np.sum(right_labels == label) / len(right_labels)
            right_gini -= pow(p, 2)
        
        # Compute weighted average of the Gini impurity of left and right nodes
        total = len(left_idx) + len(right_idx)
        weighted_gini = (len(left_idx) / total) * left_gini + (len(right_idx) / total) * right_gini

        # Return the weighted Gini impurity
        return weighted_gini
    
    def _chi_squared_test(self, labels, left_idx, right_idx):
        epsilon = 1e-10  # To prevent division by zero

        # Count the number of samples in each class
        total_counts = np.bincount(labels)

        # Count the number of samples in each class per split
        left_counts = np.bincount(labels[left_idx], minlength=total_counts.size) # Add some padding
        right_counts = np.bincount(labels[right_idx], minlength=total_counts.size) # Add some padding

        # Compute the expected counts per class per split (proportionnally to the number of samples in each class)
        expected_left_counts = total_counts * (np.sum(left_idx) / len(labels)) + epsilon
        expected_right_counts = total_counts * (np.sum(right_idx) / len(labels)) + epsilon

        # Compute left chi-squared statistics
        left_squared_difference = np.sum(pow((left_counts - expected_left_counts), 2))
        left_normalized_difference = left_squared_difference / np.sum(expected_left_counts)

        # Compute right chi-squared statistics
        right_squared_difference = np.sum(pow((right_counts - expected_right_counts), 2))
        right_normalized_difference = right_squared_difference / np.sum(expected_right_counts)

        # Sum up the chi-squared statistics
        chi2_stat = left_normalized_difference + right_normalized_difference
        
        # Compute the p-value with the survival function of the chi-squared distribution
        p_value = chi2.sf(chi2_stat, df=len(total_counts) - 1)

        # Return True if the p-value is less than the threshold, False otherwise
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
        
        # Traverse the tree
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
        epsilon = 1e-10  # To prevent division by zero
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
