# TP4

## Fichiers
### Fichiers de base
- `DecisionTree.py`: Decision Trees Classifier
- `NeuralNet.py`: Neural Network Classifier
- `entrainer_tester.py`: Entrainer et tester nos classifiers
- `load_datasets.py`: Load les datasets

### Fichiers ajoutés
- `learning_curve.py`: Visualisation de la courbe d'apprentissage pour le modèle d'Arbres de décision (avec et sans élagage) -> Not run in `entrainer_tester.py`
- `pruning.py`: Implémentation de l'élagage dans le modèle d'Arbres de décision ; Comparaison de performance avec et sans élagage -> Not run in `entrainer_tester.py`
- `scikit_classifiers.py`: Classifiers de scikit-learn (train and test functions provided -> called in `entrainer_tester.py`)

## Répartition des tâches de travail entre les membres d'équipe
- Arbres de décision: Christine Gao
- Réseaux de neurones : Félix Dupré-Ouellet
- Fichiers load_datasets.py et entrainer_tester.py : contribution par les 2 membres
- Rapport : contribution par les 2 membres

## Résultats
Le fichier `entrainer_tester.py` compare les résultats de nos classifiers (sans pruning) avec ceux de scikit-learn. Par défaut, il ne montre pas les courbes d'apprentissage ni la comparaison entre pruning et sans pruning.

Pour voir les courbes d'apprentissage, run le fichier `learning_curve.py`, ou changer le bool `LEARNING_CURVE` à `True` dans le fichier `entrainer_tester.py` (section `0 - Global Variables`).

Pour voir l'élagage, run le fichier `pruning.py`, ou changer le bool `PRUNING` à `True` dans le fichier `entrainer_tester.py` (section `0 - Global Variables`).
