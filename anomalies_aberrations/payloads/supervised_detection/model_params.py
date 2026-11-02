"""
Hyperparamètres des modèles de détection de rupture
"""
# --- LighGBM (MODEL_TYPE = 'lightgbm', librairie lightgbm.py)
LIGHTGBM_PARAMS = {
    # Structure des arbres
    'boosting_type': 'gbdt',  # 'gbdt', 'dart' ou 'rf'
    'num_leaves': 31,  # nombre maximal de feuilles par arbre
    'max_depth': -1,  # profondeur maximale (-1 : illimitée)
    'min_child_samples': 20,  # nombre minimal d'exemples par feuille
    'min_child_weight': 1e-3,  # somme minimale des hessiens dans une feuille
    'min_split_gain': 0.0,  # gain minimal pour couper un nœud
    'subsample_for_bin': 200000,  # nb d'exemples utilisés pour construire les bins des features
    # Boosting
    'n_estimators': 100,  # nombre d'arbres
    'learning_rate': 0.1,  # pas d'apprentissage du boosting
    # Échantillonnage
    'subsample': 1.0,  # part des lignes tirée pour chaque arbre
    'subsample_freq': 0,  # fréquence du tirage de lignes (0 : désactivé)
    'colsample_bytree': 1.0,  # part des colonnes tirée pour chaque arbre
    # Régularisation
    'reg_alpha': 0.0,  # régularisation L1
    'reg_lambda': 0.0,  # régularisation L2, le « lambda »
    # Objectif et déséquilibre de classes
    'objective': None,  # None : 'binary' pour un classifieur à deux classes
    'class_weight': None,  # None, 'balanced' ou {classe: poids}
    'scale_pos_weight': 1.0,  # poids des positifs (1.0 : pas de repondération)
    # Reproductibilité (avec random_state, fixé dans training.py)
    'deterministic': True,  # résultats identiques à données et paramètres égaux, quel que soit le nb de threads
    'force_row_wise': True,  # recommandé par LightGBM avec deterministic (évite le choix auto row/col-wise)
    # Divers
    'importance_type': 'split',  # 'split' ou 'gain'
    'n_jobs': None,  # None : tous les cœurs
}

# --- Réseau de neurones (MODEL_TYPE = 'mlp', module neural_network.py) 
# --- Défini les hyperparamètres de l'entraînement et les paramètres architecturels du modèle 
# ---------------------------------
MLP_PARAMS = {
    # Prétraitement, ajusté sur le train uniquement
    'imputer_strategy': 'median',  # imputation des NaN : 'median', 'mean', 'most_frequent' ou 'constant'
    'add_missing_indicator': True,  # ajoute des colonnes indiquant les valeurs manquantes
    # Architecture
    'n_hidden_layers': 4,  # nombre de couches cachées
    'hidden_units': 200,  # nombre de neurones par couche cachée
    'optimizer': 'adadelta',  # 'adadelta', 'adam', 'adamw', 'sgd' ou 'rmsprop'
    # Le learning_rate dépend de l'optimiseur : 1.0 pour Adadelta (défaut PyTorch, il adapte lui-même
    # le pas), mais plutôt 1e-3 pour Adam/AdamW/RMSprop et 1e-2 pour SGD. À changer avec l'optimiseur.
    'learning_rate': 1.0,
    'batch_size': 256,
    'max_epochs': 500,  # nombre maximal d'époques
    'patience': 10,  # arrêt si la loss de validation ne progresse pas pendant ce nombre d'époques
    'val_fraction': 0.1,  # part du train réservée à la validation (early stopping)
    'weight_decay': 0.0,  # régularisation L2 (le « lambda » des cours)
    'pos_weight': None,  # multiplie la loss des positifs (comme scale_pos_weight pour LightGBM)
}
MLP_THRESHOLD = 0.5  # seuil de probabilité au-dessus duquel predict renvoie 1

MODEL_PARAMS = {'lightgbm': LIGHTGBM_PARAMS, 'mlp': MLP_PARAMS}
