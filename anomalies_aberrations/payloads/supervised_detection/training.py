"""
Script pour apprendre à détecter les anomalies en mode supervisé à partir d'un dataset synthétique.
La préparation des données (séries, features, split train/test) est dans data_preparation.py.
"""
import os
from datetime import date
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.metrics import classification_report, confusion_matrix

import paths

# W, T, M et NORM_MAX ne servent pas ici : evaluation.py et predict.py les importent depuis training.
from data_preparation import FeatureBuilder, split_train_test, SOURCE_VT, RANDOM_STATE, TRAIN_RATIO, W, T, M, NORM_MAX  # noqa: F401

MLFLOW_TRACKING_URI = 'arn:aws:sagemaker:eu-west-3:585762237892:mlflow-app/app-CZPP4GNRF7VT'
MLFLOW_EXPERIMENT = 'rupture_detection'  # surchargeable par la variable d'env MLFLOW_EXPERIMENT_NAME
MODEL_DIR = paths.data_path('models')
# Modèle chargé par défaut par SupervisedModel.load ; save() écrit sous un nom daté (dated_model_path).
MODEL_PATH = paths.join(MODEL_DIR,'rupture_model_20_10_2026.joblib')
MIN_CHILD_SAMPLES = 20  # nombre minimal d'exemples par feuille LightGBM (défaut LightGBM : 20)

class SupervisedModel:
    """Enveloppe un modèle supervisé (LightGBM pour commencer) pour l'entraînement/l'évaluation."""

    def __init__(self, model_type='lightgbm', **model_params):
        self.model_type = model_type
        self.model_params = model_params
        self.model = None

    def _build_model(self):
        if self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(**self.model_params)
        raise ValueError(f"model_type inconnu: {self.model_type}")

    def fit(self, X, y):
        self.model = self._build_model()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def metrics(self, X, y):
        """Métriques numériques de evaluate (TN/FP/FN/TP, FPR, précision et rappel sur les
        positifs), sous forme de dict : c'est ce qui est loggé dans MLflow."""
        y_pred = self.predict(X)
        tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, y_pred, labels=[0, 1]).ravel())
        n_neg = tn + fp
        n_pos = tp + fn
        return {
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'fpr': fp / n_neg if n_neg else float('nan'),
            'precision': tp / (tp + fp) if (tp + fp) else float('nan'),
            'recall': tp / n_pos if n_pos else float('nan'),
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

    def save(self, path=None):
        """Sauvegarde le modèle entraîné (self.model) pour le recharger depuis un autre fichier.
        Sans path, le nom contient la date du jour (dated_model_path) pour ne pas écraser un modèle
        précédent."""
        path = path or dated_model_path()
        paths.makedirs(paths.dirname(path))
        paths.dump_joblib(self.model, path)
        return path

    @classmethod
    def load(cls, path=MODEL_PATH, model_type='lightgbm'):
        """Recharge un modèle sauvegardé par save() dans une nouvelle instance de SupervisedModel."""
        instance = cls(model_type=model_type)
        instance.model = paths.load_joblib(path)
        return instance

def dated_model_path(model_dir=MODEL_DIR, today=None):
    """Chemin models/rupture_model_jj_mm_aaaa.joblib avec la date du jour. Si ce fichier existe
    déjà (plusieurs entraînements le même jour), ajoute un suffixe _2, _3, etc. plutôt que de
    l'écraser."""
    stamp = (today or date.today()).strftime('%d_%m_%Y')
    path = paths.join(model_dir, f'rupture_model_{stamp}.joblib')
    n = 2
    while paths.exists(path):
        path = paths.join(model_dir,f'rupture_model_{stamp}_{n}.joblib')
        n += 1
    return path


if __name__ == "__main__":
    vt_df = paths.read_csv(SOURCE_VT, dtype={'device_id': str, 'sensor_id': str}, parse_dates=['payload_timestamp'])
    train_idx, test_idx = split_train_test(vt_df)

    tr_builder = FeatureBuilder(train_idx)
    X_train, y_train = tr_builder.generer_sample()
    ts_builder = FeatureBuilder(test_idx)
    X_test, y_test = ts_builder.generer_sample()
    print(f"Train : {int(y_train.sum())} positif(s) / {int((y_train == 0).sum())} négatif(s)")
    print(f"Test  : {int(y_test.sum())} positif(s) / {int((y_test == 0).sum())} négatif(s)")

    # Pas de scale_pos_weight : les positifs ne sont pas repondérés malgré le déséquilibre.
    # min_child_samples (taille minimale d'une feuille) : le défaut (20) donne 64 faux positifs et
    # 11 ruptures manquées sur le test. Testé plus bas (5, 15), le FPR explose sans gain de rappel ;
    # testé plus haut (50), le rappel s'effondre (48 ruptures manquées sur 59) : on reste à 20.
    model = SupervisedModel(model_type='lightgbm', random_state=RANDOM_STATE,
                            min_child_samples=MIN_CHILD_SAMPLES)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(os.environ.get('MLFLOW_EXPERIMENT_NAME', MLFLOW_EXPERIMENT))
    with mlflow.start_run():
        mlflow.log_params({
            'model_type': model.model_type, **model.model_params,
            'W': W, 'T': T, 'M': M, 'NORM_MAX': NORM_MAX, 'TRAIN_RATIO': TRAIN_RATIO,
            'n_train_pos': int(y_train.sum()), 'n_train_neg': int((y_train == 0).sum()),
            'n_test_pos': int(y_test.sum()), 'n_test_neg': int((y_test == 0).sum()),
            'data_root': paths.DATA_ROOT,
        })
        model.fit(X_train, y_train)
        report = model.evaluate(X_test, y_test)
        print(report)
        mlflow.log_metrics({k: float(v) for k, v in model.metrics(X_test, y_test).items()})
        mlflow.log_text(report, 'evaluation_report.txt')
        mlflow.lightgbm.log_model(model.model, artifact_path='model',
                                  signature=mlflow.models.infer_signature(X_test, model.predict(X_test)))
        model_path = model.save()
        mlflow.set_tag('joblib_path', model_path)
        print(f"Modèle sauvegardé : {model_path}")
