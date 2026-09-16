"""
Script pour apprendre à détecter les anomalies en mode supervisé à partir d'un dataset synthétique.
"""
import glob
import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix

SYNTHETIC_DATA = os.path.join(os.path.dirname(__file__), 'synthetic_data', 'ruptures')
SOURCE_VT = os.path.join(SYNTHETIC_DATA, 'verite_terrain.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'rupture_model.joblib')
NEGATIVE_RATIO = 100  # nombre de négatifs tirés dans la série par rupture injectée
TRAIN_RATIO = 0.7  # proportion de l'ensemble d'entraînement
W = 7  # nombre de points avant pos utilisés pour les features de lag (valeur, delta_t, delta_v)
T = 50  # nombre de points avant pos utilisés pour la médiane de delta_t
M = 5  # taille des fenêtres utilisées pour les moyennes quadratiques de la valeur
NORM_MAX = 100  # borne haute de l'échelle sur laquelle on ramène la fenêtre de T points
RANDOM_STATE = 142


def normalize_window(values, out_max=NORM_MAX):
    """Ramène une fenêtre de valeurs sur l'échelle [0, out_max] par transformation affine : le
    minimum de la fenêtre devient 0, son maximum out_max.
"""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    amplitude = vmax - vmin
    if not np.isfinite(amplitude) or amplitude == 0:
        return np.zeros_like(values)
    return out_max * (values - vmin) / amplitude

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
    """Transforme les lignes de vérité terrain d'un split en features exploitables par le modèle.
    Les positifs sont les timestamps fournis en entrée (les lignes label=1 de verite_terrain.csv) ;
    tous les autres points de la série du couple sont des négatifs, sous-échantillonnés à
    negative_ratio négatifs par positif."""

    def __init__(self, sample_idx, w=W, t=T, m=M, negative_ratio=NEGATIVE_RATIO,
                 random_state=RANDOM_STATE, series_store=None, inpath=SYNTHETIC_DATA):
        self.idx = sample_idx
        self.w = w
        self.t = t
        self.m = m
        self.negative_ratio = negative_ratio
        self.random_state = random_state
        self.series_store = series_store or SeriesStore(inpath=inpath)

    def generer_sample(self):
        """Construit la matrice de features (une ligne par exemple device/sensor/timestamp) et le
        vecteur de labels correspondant. Pour chaque couple device/sensor de self.idx, on charge la
        série complète : les timestamps de vérité terrain donnent les positifs, un tirage parmi tous
        les autres points de la série donne les négatifs."""
        rng = np.random.default_rng(self.random_state)
        feature_rows = []
        labels = []
        for (device_id, sensor_id), group in self.idx.groupby(['device_id', 'sensor_id']):
            series = self.series_store.fetch_series(device_id, sensor_id)
            if series.empty:
                continue
            is_positive = series['payload_timestamp'].isin(set(group['payload_timestamp'])).to_numpy()
            positive_positions = np.flatnonzero(is_positive)
            if len(positive_positions) != len(group):
                print(f"{len(group) - len(positive_positions)} timestamp(s) de vérité terrain "
                      f"introuvable(s) dans la série {device_id} / {sensor_id}")
            # Avant la position w, les features de lag n'ont pas assez d'historique et valent NaN :
            # on exclut cette zone de chauffe des candidats négatifs.
            candidates = np.flatnonzero(~is_positive)
            candidates = candidates[candidates >= self.w]
            negative_positions = self.sample_negative_positions(candidates, len(positive_positions), rng)
            for pos in positive_positions:
                feature_rows.append(self.build_features(series, int(pos)))
                labels.append(1)
            for pos in negative_positions:
                feature_rows.append(self.build_features(series, int(pos)))
                labels.append(0)
        X = pd.DataFrame(feature_rows)
        y = pd.Series(labels, name='label').reset_index(drop=True)
        return X, y

    def sample_negative_positions(self, candidates, n_positives, rng):
        """Tire au hasard negative_ratio négatifs par positif parmi candidates (les positions de la
        série qui ne portent pas de rupture injectée et qui ont assez d'historique)."""
        n = min(len(candidates), round(n_positives * self.negative_ratio))
        if n <= 0:
            return np.array([], dtype=int)
        return rng.choice(candidates, size=n, replace=False)

    def build_features(self, series, pos):
        """Calcule les features autour de la position pos (index de ligne dans series), toutes
        invariantes à l'échelle du compteur (aucune valeur absolue de payload_value, seulement des
        écarts et des ratios) :
        - delta_t (écart en secondes avec le point précédent) en t, t-1, ..., t-W
        - delta_v (écart de valeur avec le point précédent) en t, t-1, ..., t-W, mesuré sur la
          fenêtre des T points précédant pos (pos inclus) ramenée sur l'échelle 0-NORM_MAX par
          normalize_window : un même écart relatif donne la même feature quel que soit le compteur
        - médiane de delta_t sur les T points avant pos
        - jump_ratio : variation relative de la moyenne quadratique entre la fenêtre [t-M, t-1] et
          [t-M, t], calculée elle aussi sur la fenêtre normalisée. Sur les valeurs brutes d'un
          compteur cumulatif (qui vaut par exemple 1 000 000), les deux moyennes quadratiques sont
          quasi égales et le ratio est écrasé vers 0 quelle que soit la rupture : c'est le décalage
          du compteur, et non son échelle, qui tuait la feature.
        """
        features = {}
        value_window = series['raw_value'].iloc[max(0, pos - self.t):pos + 1].to_numpy(dtype=float)
        normalized = normalize_window(value_window)
        # norm_diff[-1] est l'écart en pos, norm_diff[-1-lag] celui en pos-lag
        norm_diff = np.diff(normalized) if normalized.size > 1 else np.array([])

        for lag in range(self.w + 1):
            idx = pos - lag
            valid = idx >= 0
            suffix = 't' if lag == 0 else f't-{lag}'
            features[f'delta_t_{suffix}'] = series['time_delta_seconds'].iloc[idx] if valid else np.nan
            diff_idx = len(norm_diff) - 1 - lag
            features[f'delta_v_{suffix}'] = norm_diff[diff_idx] if diff_idx >= 0 else np.nan

        delta_t_window = series['time_delta_seconds'].iloc[max(0, pos - self.t):pos]
        features['delta_t_median'] = delta_t_window.median() if len(delta_t_window) else np.nan

        # Sous-fenêtres de M points prises dans la fenêtre déjà normalisée : normalized[-1] est pos
        value_curr = normalized[-(self.m + 1):]
        value_prev = normalized[-(self.m + 1):-1]
        rms_prev = np.sqrt((value_prev ** 2).mean()) if value_prev.size else np.nan
        rms_curr = np.sqrt((value_curr ** 2).mean()) if value_curr.size else np.nan
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
        """Rapport de classification, complété par ce qui se passe du côté des négatifs. Le rappel
        seul est trompeur ici : avec NEGATIVE_RATIO négatifs pour un positif, un taux de faux
        positifs même faible représente beaucoup plus de points que le nombre de vraies ruptures,
        et c'est ce qui rendrait le détecteur inexploitable en production. On sort donc :
        - FPR (FP / négatifs) : la proportion de points normaux signalés à tort comme rupture
        - la précision sur les positifs : sur toutes les alertes levées, la part de vraies ruptures
        """
        y_pred = self.predict(X)
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        n_neg = tn + fp
        n_pos = tp + fn
        fpr = fp / n_neg if n_neg else float('nan')
        precision = tp / (tp + fp) if (tp + fp) else float('nan')
        return "\n".join([
            classification_report(y, y_pred, digits=3, zero_division=0),
            f"Matrice de confusion : TN={tn}  FP={fp}  FN={fn}  TP={tp}",
            f"Négatifs : {n_neg} — {fp} faux positifs, soit un FPR de {100 * fpr:.3f}%",
            f"Positifs : {n_pos} — {fn} rupture(s) manquée(s)",
            f"Précision des alertes : {100 * precision:.1f}% ({tp} vraie(s) sur {tp + fp} levée(s))",
        ])

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

def split_train_test(vt_df, train_ratio=TRAIN_RATIO, random_state=RANDOM_STATE):
    """Découpe les seuls positifs de la vérité terrain (label=1, c'est-à-dire les ruptures
    injectées) en train/test, au niveau des couples device_id/sensor_id : jamais une série à cheval
    sur les deux splits, pour éviter toute fuite (le modèle ne doit pas être évalué sur des séries
    qu'il a vues en train). Les négatifs ne sont pas découpés ici — ils sont tirés par FeatureBuilder
    parmi tous les points non marqués rupture de la série de chaque couple, donc du bon côté du
    split par construction."""
    positives = vt_df[vt_df['label'] == 1]
    pairs = positives[['device_id', 'sensor_id']].drop_duplicates().sample(frac=1, random_state=random_state)
    n_train_pairs = round(len(pairs) * train_ratio)
    train_pairs = pairs.iloc[:n_train_pairs]
    test_pairs = pairs.iloc[n_train_pairs:]

    train_idx = positives.merge(train_pairs, on=['device_id', 'sensor_id'])
    test_idx = positives.merge(test_pairs, on=['device_id', 'sensor_id'])
    print(f"{len(train_pairs)} couple(s) en train / {len(test_pairs)} en test "
          f"({len(train_idx)} positif(s) / {len(test_idx)})")
    return train_idx, test_idx

if __name__ == "__main__":
    vt_df = pd.read_csv(SOURCE_VT, dtype={'device_id': str, 'sensor_id': str}, parse_dates=['payload_timestamp'])
    train_idx, test_idx = split_train_test(vt_df)

    tr_builder = FeatureBuilder(train_idx)
    X_train, y_train = tr_builder.generer_sample()
    ts_builder = FeatureBuilder(test_idx)
    X_test, y_test = ts_builder.generer_sample()
    print(f"Train : {int(y_train.sum())} positif(s) / {int((y_train == 0).sum())} négatif(s)")
    print(f"Test  : {int(y_test.sum())} positif(s) / {int((y_test == 0).sum())} négatif(s)")

    model = SupervisedModel(model_type='lightgbm')
    model.fit(X_train, y_train)
    print(model.evaluate(X_test, y_test))
    model_path = model.save()
    print(f"Modèle sauvegardé : {model_path}")
