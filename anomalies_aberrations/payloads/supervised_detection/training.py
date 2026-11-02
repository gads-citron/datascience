"""
Script pour apprendre à détecter les anomalies en mode supervisé à partir d'un dataset synthétique.
La préparation des données (séries, features, split train/test) est dans data_preparation.py.
"""
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix

import paths
from model_params import MODEL_PARAMS
from neural_network import build_neural_network

# W, T, M et NORM_MAX ne servent pas ici : evaluation.py et predict.py les importent depuis training.
from data_preparation import load_train_test, dataset_hash, RANDOM_STATE, TRAIN_RATIO, W, T, M, NORM_MAX  # noqa: F401

MLFLOW_TRACKING_URI = 'arn:aws:sagemaker:eu-west-3:585762237892:mlflow-app/app-CZPP4GNRF7VT'
MLFLOW_EXPERIMENT = 'rupture_detection'  # surchargeable par la variable d'env MLFLOW_EXPERIMENT_NAME
MODEL_TYPE = 'lightgbm'  # 'lightgbm' ou 'mlp' 
REGISTERED_MODEL_NAME = 'rupture-detection-model'
MODEL_URI = f'models:/{REGISTERED_MODEL_NAME}/latest'

class SupervisedModel:
    """Enveloppe un modèle supervisé (LightGBM pour commencer) pour l'entraînement/l'évaluation."""

    def __init__(self, model_type='lightgbm', **model_params):
        self.model_type = model_type
        self.model_params = model_params
        self.model = None

    def _build_model(self):
        if self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(**self.model_params)
        if self.model_type == 'mlp':
            return build_neural_network(**self.model_params)
        raise ValueError(f"model_type inconnu: {self.model_type}")

    def log_to_mlflow(self, X_example, artifact_path='model'):
        """Enregistre le modèle entraîné dans le run MLflow courant, avec le flavor adapté au type
        de modèle (LightGBM ou Pipeline sklearn pour le réseau de neurones)."""
        signature = mlflow.models.infer_signature(X_example, self.predict(X_example))
        registered = REGISTERED_MODEL_NAME
        if self.model_type == 'lightgbm':
            mlflow.lightgbm.log_model(self.model, artifact_path=artifact_path, signature=signature,
                                      registered_model_name=registered)
        else:
            # code_paths : le classifieur PyTorch est une classe de neural_network.py (qui importe
            # model_params.py), nécessaires pour recharger le modèle depuis MLflow.
            here = os.path.dirname(__file__)
            mlflow.sklearn.log_model(self.model, artifact_path=artifact_path, signature=signature,
                                     registered_model_name=registered,
                                     # skops (format par défaut) ne sait pas sérialiser les tenseurs PyTorch.
                                     serialization_format='cloudpickle',
                                     code_paths=[os.path.join(here, 'neural_network.py'),
                                                 os.path.join(here, 'model_params.py')])

    def fit(self, X, y):
        self.model = self._build_model()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def metrics(self, X, y):
        """Métriques numériques de evaluate (TN/FP/FN/TP, FPR, précision, rappel et F1 sur les
        positifs), sous forme de dict : c'est ce qui est loggé dans MLflow."""
        y_pred = self.predict(X)
        tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, y_pred, labels=[0, 1]).ravel())
        n_neg = tn + fp
        n_pos = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else float('nan')
        recall = tp / n_pos if n_pos else float('nan')
        return {
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'fpr': fp / n_neg if n_neg else float('nan'),
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if (precision + recall) else float('nan'),
        }

    def evaluate(self, X, y):
        """Rapport de classification, complété par ce qui se passe du côté des négatifs. Le rappel
        seul est trompeur ici : avec beaucoup plus de négatifs que de positifs, un taux de faux
        positifs même faible représente beaucoup plus de points que le nombre de vraies ruptures,
        et c'est ce qui rendrait le détecteur inexploitable en production. On sort donc :
        - FPR (FP / négatifs) : la proportion de points normaux signalés à tort comme rupture
        - la précision sur les positifs : sur toutes les alertes levées, la part de vraies ruptures
        """
        y_pred = self.predict(X)
        m = self.metrics(X, y)
        tn, fp, fn, tp = m['tn'], m['fp'], m['fn'], m['tp']
        n_neg = tn + fp
        n_pos = tp + fn
        fpr = m['fpr']
        precision = m['precision']
        return "\n".join([
            classification_report(y, y_pred, digits=3, zero_division=0),
            f"Matrice de confusion : TN={tn}  FP={fp}  FN={fn}  TP={tp}",
            f"Négatifs : {n_neg} — {fp} faux positifs, soit un FPR de {100 * fpr:.3f}%",
            f"Positifs : {n_pos} — {fn} rupture(s) manquée(s)",
            f"Précision des alertes : {100 * precision:.1f}% ({tp} vraie(s) sur {tp + fp} levée(s))",
        ])

    @classmethod
    def load(cls, model_uri=MODEL_URI, model_type=MODEL_TYPE):
        """Recharge depuis MLflow un modèle enregistré par log_to_mlflow (model_uri : 'models:/nom/
        version' ou 'runs:/<run_id>/model') dans une nouvelle instance de SupervisedModel."""
        init_mlflow()
        instance = cls(model_type=model_type)
        loader = mlflow.lightgbm if model_type == 'lightgbm' else mlflow.sklearn
        instance.model = loader.load_model(model_uri)
        return instance


def init_mlflow():
    """Pointe MLflow vers le serveur de tracking SageMaker et l'expérience de la détection."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(os.environ.get('MLFLOW_EXPERIMENT_NAME', MLFLOW_EXPERIMENT))


def log_loss_curves(network):
    """Trace les courbes de loss train/validation du réseau de neurones (une valeur par époque) et
    les enregistre dans le run MLflow, en figure et en métriques par époque."""
    train_loss, val_loss = network.train_loss_history_, network.val_loss_history_
    for epoch, (tr, va) in enumerate(zip(train_loss, val_loss), start=1):
        mlflow.log_metrics({'train_loss': tr, 'val_loss': va}, step=epoch)
    best_epoch = int(np.argmin(val_loss)) + 1
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label='train')
    ax.plot(epochs, val_loss, label='validation')
    ax.axvline(best_epoch, color='grey', linestyle='--', label=f'meilleure époque ({best_epoch})')
    ax.set_xlabel('Époque')
    ax.set_ylabel('Loss (BCE)')
    ax.set_title('Loss du réseau de neurones')
    ax.legend()
    mlflow.log_figure(fig, 'loss_curves.png')
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_train_test()
    print(f"Train : {int(y_train.sum())} positif(s) / {int((y_train == 0).sum())} négatif(s)")
    print(f"Test  : {int(y_test.sum())} positif(s) / {int((y_test == 0).sum())} négatif(s)")

    model = SupervisedModel(model_type=MODEL_TYPE, random_state=RANDOM_STATE, **MODEL_PARAMS[MODEL_TYPE])
    init_mlflow()
    now = datetime.now()
    run_name = f"{MODEL_TYPE}_{now:%Y%m%d}_{now:%H%M%S}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            'model_type': model.model_type, **model.model_params,
            'W': W, 'T': T, 'M': M, 'NORM_MAX': NORM_MAX, 'TRAIN_RATIO': TRAIN_RATIO,
            'n_train_pos': int(y_train.sum()), 'n_train_neg': int((y_train == 0).sum()),
            'n_test_pos': int(y_test.sum()), 'n_test_neg': int((y_test == 0).sum()),
            'data_root': paths.DATA_ROOT,
            'test_set_hash': dataset_hash(X_test, y_test),  # evaluation.py vérifie que son test set est identique
        })
        model.fit(X_train, y_train)
        report = model.evaluate(X_test, y_test)
        print(report)
        mlflow.log_metrics({k: float(v) for k, v in model.metrics(X_test, y_test).items()})
        mlflow.log_text(report, 'evaluation_report.txt')
        if MODEL_TYPE == 'mlp':
            log_loss_curves(model.model)
        model.log_to_mlflow(X_test)
        print(f"Modèle enregistré dans MLflow sous '{REGISTERED_MODEL_NAME}' (run {mlflow.active_run().info.run_id})")
