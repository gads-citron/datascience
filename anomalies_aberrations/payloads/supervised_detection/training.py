"""
Script pour apprendre à détecter les anomalies en mode supervisé à partir d'un dataset synthétique.
"""
import glob
import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import classification_report

SYNTHETIC_DATA = os.path.join(os.path.dirname(__file__), 'synthetic_data', 'ruptures')
SOURCE_VT = os.path.join(SYNTHETIC_DATA, 'verite_terrain.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'rupture_model.joblib')
NEGATIVE_RATIO = 3.0  # nombre de lignes négatives conservées par ligne positive (0 à 1+)
TRAIN_RATIO = 0.7  # proportion de l'ensemble d'entraînement
W = 7  # nombre de points avant pos utilisés pour les features de lag (valeur, delta_t, delta_v)
T = 50  # nombre de points avant pos utilisés pour la médiane de delta_t
M = 5  # taille des fenêtres utilisées pour les moyennes quadratiques de la valeur
RANDOM_STATE = 142

class SeriesStore:
    """Charge et met en cache, par couple device_id/sensor_id, les séries brutes issues des
    fichiers parquet générés par creation_rupture.py (il peut y en avoir plusieurs par couple,
    un par période/run), avec value_diff et time_delta_seconds recalculés."""

    def __init__(self, inpath=SYNTHETIC_DATA):
        self.inpath = inpath
        self._cache = {}

    def fetch_series(self, device_id, sensor_id):
        key = (device_id, sensor_id)
        if key not in self._cache:
            self._cache[key] = self._load_series(device_id, sensor_id)
        return self._cache[key]

    def _load_series(self, device_id, sensor_id):
        pattern = os.path.join(self.inpath, f"{device_id}_{sensor_id}_*.parquet")
        paths = sorted(glob.glob(pattern))
        if not paths:
            return pd.DataFrame(columns=['payload_timestamp', 'raw_value', 'value_diff', 'time_delta_seconds'])
        df = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        df = df.sort_values('payload_timestamp').drop_duplicates('payload_timestamp').reset_index(drop=True)
        df['value_diff'] = df['raw_value'].diff()
        df['time_delta_seconds'] = df['payload_timestamp'].diff().dt.total_seconds()
        return df


class FeatureBuilder:
    """Transforme les lignes de vérité terrain (device_id/sensor_id/payload_timestamp/label) en
    features exploitables par le modèle, à partir des séries brutes correspondantes : pour chaque
    timestamp candidat, on compare une fenêtre de valeurs avant/après ce timestamp."""

    def __init__(self, sample_idx, w=W, t=T, m=M, series_store=None, inpath=SYNTHETIC_DATA):
        self.idx = sample_idx
        self.w = w
        self.t = t
        self.m = m
        self.series_store = series_store or SeriesStore(inpath=inpath)

    def generer_sample(self):
        """Construit la matrice de features (une ligne par exemple device/sensor/timestamp) et le
        vecteur de labels correspondant, à partir des lignes de vérité terrain de self.idx."""
        feature_rows = []
        labels = []
        for (device_id, sensor_id), group in self.idx.groupby(['device_id', 'sensor_id']):
            series = self.series_store.fetch_series(device_id, sensor_id)
            if series.empty:
                continue
            positions = series['payload_timestamp'].searchsorted(group['payload_timestamp'])
            for pos, label in zip(positions, group['label']):
                feature_rows.append(self.build_features(series, pos))
                labels.append(label)
        X = pd.DataFrame(feature_rows)
        y = pd.Series(labels, name='label').reset_index(drop=True)
        return X, y

    def build_features(self, series, pos):
        """Calcule les features autour de la position pos (index de ligne dans series), toutes
        invariantes à l'échelle du compteur (aucune valeur absolue de payload_value, seulement des
        écarts et des ratios) :
        - delta_t (écart en secondes avec le point précédent) en t, t-1, ..., t-W
        - delta_v (écart de valeur avec le point précédent) en t, t-1, ..., t-W
        - médiane de delta_t sur les T points avant pos
        - jump_ratio : variation relative de la moyenne quadratique de la valeur entre la fenêtre
          [t-M, t-1] et [t-M, t] (remplace les moyennes quadratiques brutes, sensibles à l'échelle)
        """
        features = {}
        for lag in range(self.w + 1):
            idx = pos - lag
            valid = idx >= 0
            suffix = 't' if lag == 0 else f't-{lag}'
            features[f'delta_t_{suffix}'] = series['time_delta_seconds'].iloc[idx] if valid else np.nan
            features[f'delta_v_{suffix}'] = series['value_diff'].iloc[idx] if valid else np.nan

        delta_t_window = series['time_delta_seconds'].iloc[max(0, pos - self.t):pos]
        features['delta_t_median'] = delta_t_window.median() if len(delta_t_window) else np.nan

        value_prev = series['raw_value'].iloc[max(0, pos - self.m):pos]
        value_curr = series['raw_value'].iloc[max(0, pos - self.m):pos + 1]
        rms_prev = np.sqrt((value_prev ** 2).mean()) if len(value_prev) else np.nan
        rms_curr = np.sqrt((value_curr ** 2).mean()) if len(value_curr) else np.nan
        features['jump_ratio'] = (rms_curr - rms_prev) / (abs(rms_prev) + 1e-6) if pd.notna(rms_prev) else np.nan

        return features

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

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return classification_report(y, y_pred)

    def save(self, path=MODEL_PATH):
        """Sauvegarde le modèle entraîné (self.model) pour le recharger depuis un autre fichier."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        return path

    @classmethod
    def load(cls, path=MODEL_PATH, model_type='lightgbm'):
        """Recharge un modèle sauvegardé par save() dans une nouvelle instance de SupervisedModel."""
        instance = cls(model_type=model_type)
        instance.model = joblib.load(path)
        return instance


def _limit_negatives(df, negative_ratio, random_state):
    """Sous-échantillonne les négatifs d'un split pour respecter negative_ratio (nombre de
    négatifs par positif)."""
    positives = df[df['label'] == 1]
    negatives = df[df['label'] == 0]
    n_negatives = min(len(negatives), round(len(positives) * negative_ratio))
    negatives = negatives.sample(n=n_negatives, random_state=random_state)
    return pd.concat([positives, negatives], ignore_index=True)


def split_train_test(vt_df, negative_ratio=NEGATIVE_RATIO, train_ratio=TRAIN_RATIO, random_state=RANDOM_STATE):
    """Découpe la vérité terrain en train/test au niveau des couples device_id/sensor_id (jamais
    une série à cheval sur les deux splits, pour éviter toute fuite : le modèle ne doit pas être
    évalué sur des séries qu'il a vues en train), puis sous-échantillonne les négatifs dans chaque
    split pour respecter negative_ratio."""
    pairs = vt_df[['device_id', 'sensor_id']].drop_duplicates().sample(frac=1, random_state=random_state)
    n_train_pairs = round(len(pairs) * train_ratio)
    train_pairs = pairs.iloc[:n_train_pairs]
    test_pairs = pairs.iloc[n_train_pairs:]

    train_df = vt_df.merge(train_pairs, on=['device_id', 'sensor_id'])
    test_df = vt_df.merge(test_pairs, on=['device_id', 'sensor_id'])

    train_idx = _limit_negatives(train_df, negative_ratio, random_state)
    test_idx = _limit_negatives(test_df, negative_ratio, random_state)
    return train_idx, test_idx

if __name__ == "__main__":
    vt_df = pd.read_csv(SOURCE_VT, dtype={'device_id': str, 'sensor_id': str}, parse_dates=['payload_timestamp'])
    train_idx, test_idx = split_train_test(vt_df)

    tr_builder = FeatureBuilder(train_idx)
    X_train, y_train = tr_builder.generer_sample()
    ts_builder = FeatureBuilder(test_idx)
    X_test, y_test = ts_builder.generer_sample()

    model = SupervisedModel(model_type='lightgbm')
    model.fit(X_train, y_train)
    print(model.evaluate(X_test, y_test))
    model_path = model.save()
    print(f"Modèle sauvegardé : {model_path}")
