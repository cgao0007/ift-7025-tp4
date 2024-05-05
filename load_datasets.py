import numpy as np
import random

IRIS_DATASET = 'datasets/bezdekIris.data'
WINE_DATASET = 'datasets/binary-winequality-white.csv'
ABALONE_DATASET = 'datasets/abalone-intervalles.csv'


def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples qui vont etre attribués à l'entrainement,
        le reste des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisés
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
        
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
        
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """

    random.seed(1)  # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

    # Le fichier du dataset est dans le dossier datasets en attaché
    data = None
    with open(IRIS_DATASET, 'r') as f:
        data = f.readlines()

    if data is None:
        raise IOError(f'Failed to read the iris dataset file: {IRIS_DATASET}')

    # TODO : le code ici pour lire le dataset

    # REMARQUE très importante : 
    # remarquez bien comment les exemples sont ordonnés dans
    # le fichier du dataset, ils sont ordonnés par type de fleur, cela veut dire que 
    # si vous lisez les exemples dans cet ordre et que si par exemple votre ration est de 60%,
    # vous n'allez avoir aucun exemple du type Iris-virginica pour l'entrainement, pensez
    # donc à utiliser la fonction random.shuffle pour melanger les exemples du dataset avant de séparer
    # en train et test.

    # Donc: shuffle data first
    random.shuffle(data)

    # Split into train and test
    split = int(len(data) * train_ratio)
    train_data = data[:split]
    test_data = data[split:]

    # Helper functions
    def row_to_features(row):
        features = row.split(',')[:-1]
        for i, feature in enumerate(features):
            features[i] = float(feature)
        return features

    def row_to_label(row):
        label = row.split(',')[-1].strip()
        return conversion_labels[label]

    # Train
    train_list = []

    # Extract features (omit last column, which is the label)
    for row in train_data:
        train_list.append(row_to_features(row))

    # Store in a numpy array
    train = np.array(train_list)

    # Train label
    train_label_list = []

    # Extract labels
    for row in train_data:
        train_label_list.append(row_to_label(row))

    # Store in a numpy array
    train_labels = np.array(train_label_list)

    # Test
    test_list = []

    # Extract features (omit last column, which is the label)
    for row in test_data:
        test_list.append(row_to_features(row))

    # Store in a numpy array
    test = np.array(test_list)

    # Test label
    test_label_list = []

    # Extract labels
    for row in test_data:
        test_label_list.append(row_to_label(row))

    # Store in a numpy array
    test_labels = np.array(test_label_list)

    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy. 
    return train, train_labels, test, test_labels


def load_wine_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Binary Wine quality

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
        
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """

    random.seed(1)  # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Le fichier du dataset est dans le dossier datasets en attaché
    data = None
    with open(WINE_DATASET, 'r') as f:
        data = f.readlines()

    if data is None:
        raise IOError(f'Failed to read the wine dataset file: {WINE_DATASET}')

    # TODO : le code ici pour lire le dataset
    random.shuffle(data)

    split = int(len(data) * train_ratio)
    train_data = data[:split]
    test_data = data[split:]

    # Helper functions
    def row_to_features(row):
        features = row.split(',')[:-1]
        for i, feature in enumerate(features):
            features[i] = float(feature)
        return features

    def row_to_label(row):
        label = row.split(',')[-1]
        return int(label)

    # Train
    train_list = []

    # Extract features (omit last column, which is the label)
    for row in train_data:
        train_list.append(row_to_features(row))

    # Store in a numpy array
    train = np.array(train_list)

    # Train label
    train_label_list = []

    # Extract labels
    for row in train_data:
        train_label_list.append(row_to_label(row))

    # Store in a numpy array
    train_labels = np.array(train_label_list)

    # Test
    test_list = []

    # Extract features (omit last column, which is the label)
    for row in test_data:
        test_list.append(row_to_features(row))

    # Store in a numpy array
    test = np.array(test_list)

    # Test label
    test_label_list = []

    # Extract labels
    for row in test_data:
        test_label_list.append(row_to_label(row))

    # Store in a numpy array
    test_labels = np.array(test_label_list)

    return train, train_labels, test, test_labels


def load_abalone_dataset(train_ratio, one_hot_encoding=False):
    """
    Cette fonction a pour but de lire le dataset Abalone-intervalles

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

        one_hot_encoding: si True, la variable 'sex' est encodée comme un vecteur one-hot

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """

    random.seed(1)  # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Le fichier du dataset est dans le dossier datasets en attaché
    data = None
    with open(ABALONE_DATASET, 'r') as f:
        data = f.readlines()

    if data is None:
        raise IOError(f'Failed to read the abalone dataset file: {ABALONE_DATASET}')

    sex_to_number = {'M': 0, 'F': 1, 'I': 2}
    sex_to_one_hot = {'M': [1, 0, 0], 'F': [0, 1, 0], 'I': [0, 0, 1]}

    # TODO : le code ici pour lire le dataset
    random.shuffle(data)

    split = int(len(data) * train_ratio)
    train_data = data[:split]
    test_data = data[split:]

    # Helper functions
    def row_to_features(row, one_hot_encoding):
        features = row.split(',')[:-1]

        if one_hot_encoding:
            encoded_sex = sex_to_one_hot[features[0]]
            offset = 3
        else:
            encoded_sex = [sex_to_number[features[0]]]
            offset = 1
        features = encoded_sex + features[1:]

        for i, feature in enumerate(features[offset:]):
            features[i + offset] = float(feature)

        return features

    def row_to_label(row):
        label = row.split(',')[-1]
        return int(float(label))

    # Train
    train_list = []

    # Extract features (omit last column, which is the label)
    for row in train_data:
        train_list.append(row_to_features(row, one_hot_encoding))

    # Store in a numpy array
    train = np.array(train_list)

    # Train label
    train_label_list = []

    # Extract labels
    for row in train_data:
        train_label_list.append(row_to_label(row))

    # Store in a numpy array
    train_labels = np.array(train_label_list)

    # Test
    test_list = []

    # Extract features (omit last column, which is the label)
    for row in test_data:
        test_list.append(row_to_features(row, one_hot_encoding))

    # Store in a numpy array
    test = np.array(test_list)

    # Test label
    test_label_list = []

    # Extract labels
    for row in test_data:
        test_label_list.append(row_to_label(row))

    # Store in a numpy array
    test_labels = np.array(test_label_list)

    # La fonction doit retourner 4 structures de données de type Numpy.
    return train, train_labels, test, test_labels
