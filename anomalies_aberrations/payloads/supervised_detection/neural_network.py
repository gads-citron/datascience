"""
Réseau de neurones PyTorch de détection de rupture : mêmes features en entrée que le modèle
LightGBM (FeatureBuilder, data_preparation.py), 4 couches cachées de 200 neurones (ReLU), une
sortie logit. TorchMLPClassifier suit l'interface sklearn (fit / predict / predict_proba), comme
LGBMClassifier : training.py l'utilise via SupervisedModel(model_type='mlp') sans rien changer
d'autre. Le réseau ne gère pas les NaN et est sensible à l'échelle des features : le classifieur
applique donc en interne une imputation par la médiane du train (avec indicateurs de valeur
manquante en colonnes supplémentaires, pour que le réseau voie qu'une valeur manquait au lieu de
la confondre avec une vraie valeur) puis une standardisation, ajustées sur le train uniquement.
"""
import copy
import os

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from model_params import MLP_PARAMS, MLP_THRESHOLD  # hyperparamètres par défaut : model_params.py

# Requis par torch.use_deterministic_algorithms sur GPU (cuBLAS) ; à fixer avant toute init CUDA.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

OPTIMIZERS = {
    'adadelta': torch.optim.Adadelta,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD,
    'rmsprop': torch.optim.RMSprop,
}


def build_mlp(n_inputs, n_hidden_layers=MLP_PARAMS['n_hidden_layers'], hidden_units=MLP_PARAMS['hidden_units']):
    """Perceptron multicouche : n_hidden_layers couches Linear(hidden_units) + ReLU, puis une
    couche de sortie à un logit (la sigmoïde est dans la loss à l'entraînement et dans
    predict_proba)."""
    layers, size = [], n_inputs
    for _ in range(n_hidden_layers):
        layers += [nn.Linear(size, hidden_units), nn.ReLU()]
        size = hidden_units
    layers.append(nn.Linear(size, 1))
    return nn.Sequential(*layers)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Classifieur binaire PyTorch à l'interface sklearn. pos_weight (None par défaut, comme
    scale_pos_weight pour LightGBM) multiplie la loss des positifs pour compenser le déséquilibre
    entre classes."""

    def __init__(self, n_hidden_layers=MLP_PARAMS['n_hidden_layers'], hidden_units=MLP_PARAMS['hidden_units'],
                 optimizer=MLP_PARAMS['optimizer'], learning_rate=MLP_PARAMS['learning_rate'], batch_size=MLP_PARAMS['batch_size'],
                 max_epochs=MLP_PARAMS['max_epochs'], patience=MLP_PARAMS['patience'],
                 val_fraction=MLP_PARAMS['val_fraction'], weight_decay=MLP_PARAMS['weight_decay'],
                 pos_weight=MLP_PARAMS['pos_weight'], imputer_strategy=MLP_PARAMS['imputer_strategy'],
                 add_missing_indicator=MLP_PARAMS['add_missing_indicator'], random_state=None):
        self.n_hidden_layers = n_hidden_layers
        self.hidden_units = hidden_units
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight
        self.imputer_strategy = imputer_strategy
        self.add_missing_indicator = add_missing_indicator
        self.random_state = random_state

    @staticmethod
    def _device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)  # seed aussi les GPU
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.preprocessor_ = make_pipeline(SimpleImputer(strategy=self.imputer_strategy,
                                                         add_indicator=self.add_missing_indicator),
                                           StandardScaler())
        X_all = self.preprocessor_.fit_transform(np.asarray(X, dtype=np.float32)).astype(np.float32)

        # Jeu de validation pour l'early stopping, stratifié quand les deux classes sont assez
        # nombreuses pour l'être.
        stratify = y if min(y.sum(), (y == 0).sum()) >= 2 else None
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y, test_size=self.val_fraction, stratify=stratify, random_state=self.random_state)

        device = self._device()
        self.network_ = build_mlp(X_all.shape[1], self.n_hidden_layers, self.hidden_units).to(device)
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer inconnu: {self.optimizer} (choix : {sorted(OPTIMIZERS)})")
        optimizer = OPTIMIZERS[self.optimizer](self.network_.parameters(), lr=self.learning_rate,
                                               weight_decay=self.weight_decay)
        pos_weight = None if self.pos_weight is None else torch.tensor([self.pos_weight], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_tr_t, y_tr_t = torch.from_numpy(X_tr).to(device), torch.from_numpy(y_tr).to(device)
        X_val_t, y_val_t = torch.from_numpy(X_val).to(device), torch.from_numpy(y_val).to(device)
        best_loss, best_state, stale = float('inf'), None, 0
        self.train_loss_history_, self.val_loss_history_ = [], []
        for _ in range(self.max_epochs):
            self.network_.train()
            order = torch.randperm(len(X_tr_t), device=device)
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.network_(X_tr_t[batch]).squeeze(1), y_tr_t[batch])
                loss.backward()
                optimizer.step()
            self.network_.eval()
            with torch.no_grad():
                # Loss train recalculée en mode eval sur tout le train, pour être comparable à la val.
                train_loss = loss_fn(self.network_(X_tr_t).squeeze(1), y_tr_t).item()
                val_loss = loss_fn(self.network_(X_val_t).squeeze(1), y_val_t).item()
            self.train_loss_history_.append(train_loss)
            self.val_loss_history_.append(val_loss)
            if val_loss < best_loss:
                best_loss, best_state, stale = val_loss, copy.deepcopy(self.network_.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.network_.load_state_dict(best_state)
        self.network_.to('cpu')  # le modèle sauvegardé se recharge sans GPU
        return self

    def predict_proba(self, X):
        X_t = torch.from_numpy(
            self.preprocessor_.transform(np.asarray(X, dtype=np.float32)).astype(np.float32))
        self.network_.eval()
        with torch.no_grad():
            p1 = torch.sigmoid(self.network_(X_t).squeeze(1)).numpy()
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= MLP_THRESHOLD).astype(int)


def build_neural_network(**params):
    """Réseau de neurones de détection de rupture ; params (n_hidden_layers, hidden_units,
    learning_rate, pos_weight, random_state, ...) sont transmis à TorchMLPClassifier."""
    return TorchMLPClassifier(**params)
