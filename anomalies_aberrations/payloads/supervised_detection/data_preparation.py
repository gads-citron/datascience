"""
Préparation des données d'entraînement du modèle de détection de rupture : chargement des séries
parquet (SeriesStore), construction des features et des labels (FeatureBuilder, normalize_window)
et découpage train/test par couple device_id/sensor_id (list_series_pairs, split_train_test).
"""
import numpy as np
import pandas as pd

import paths

TRAINING_DATA = paths.data_path('training_data')
SYNTHETIC_DATA = paths.join(TRAINING_DATA, 'ruptures')
CONSTANTS_DATA = paths.join(TRAINING_DATA, 'constants')
REAL_NEG_DATA = paths.join(TRAINING_DATA, 'real_neg_samples')
SOURCE_VT = paths.join(TRAINING_DATA, 'positive_cases.csv')
# Dossiers de parquet parcourus pour les séries : ruptures (avec vérité terrain), constants
# (séries synthétiques en paliers de creation_constants.py) et real_neg_samples (périodes réelles
# saines collectées par real_neg_samples.py). constants n'a aucun positif : il sert uniquement de
# négatifs supplémentaires via list_series_pairs.
DATA_PATHS = [SYNTHETIC_DATA, CONSTANTS_DATA, REAL_NEG_DATA]
NEGATIVES_WITHOUT_POSITIVE = 500  # nombre fixe de négatifs tirés pour un couple sans aucun positif
TRAIN_RATIO = 0.7 # proportion de l'ensemble d'entraînement
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
    if np.isnan(values).all():
        return values  # fenêtre sans aucune valeur : on garde les NaN (pas de remplissage par 0)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    amplitude = vmax - vmin
    if not np.isfinite(amplitude) or amplitude == 0:
        return np.zeros_like(values)
    return out_max * (values - vmin) / amplitude

class SeriesStore:
    """Charge et met en cache, par couple device_id/sensor_id, les séries brutes issues des
    fichiers parquet générés par creation_rupture.py et creation_constants.py (il peut y en avoir
    plusieurs par couple, un par période/run), avec value_diff et time_delta_seconds recalculés."""

    def __init__(self, inpaths=DATA_PATHS):
        self.inpaths = [inpaths] if isinstance(inpaths, str) else inpaths
        self._cache = {}

    def fetch_series(self, device_id, sensor_id):
        key = (device_id, sensor_id)
        if key not in self._cache:
            self._cache[key] = self._load_series(device_id, sensor_id)
        return self._cache[key]

    def _load_series(self, device_id, sensor_id):
        files = sorted(path for inpath in self.inpaths
                        for path in paths.glob(paths.join(inpath, f"{device_id}_{sensor_id}_*.parquet")))
        if not files:
            return pd.DataFrame(columns=['payload_timestamp', 'raw_value', 'value_diff',
                                         'time_delta_seconds', 'is_real_neg'])
        frames = []
        for path in files:
            frame = paths.read_parquet(path)
            # is_real_neg : le point vient d'une période saine de real_neg_samples/ (un timestamp
            # présent dans un de ces fichiers reste marqué même s'il figure aussi ailleurs).
            frame['is_real_neg'] = paths.same_dir(paths.dirname(path), REAL_NEG_DATA)
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
        df['is_real_neg'] = df.groupby('payload_timestamp')['is_real_neg'].transform('max')
        df = df.sort_values('payload_timestamp').drop_duplicates('payload_timestamp').reset_index(drop=True)
        df['value_diff'] = df['raw_value'].diff()
        df['time_delta_seconds'] = df['payload_timestamp'].diff().dt.total_seconds()
        return df


class FeatureBuilder:
    """Transforme les lignes de vérité terrain d'un split en features exploitables par le modèle.
    Les positifs sont les timestamps fournis en entrée (toutes les lignes de verite_terrain.csv) ;
    les négatifs sont tirés, à raison de negatives_without_positive points par couple, parmi les
    points de la série dont le timestamp n'est pas dans la vérité terrain ; pour un couple présent
    dans la vérité terrain, parmi ceux des périodes saines de real_neg_samples/ uniquement."""

    def __init__(self, sample_idx, w=W, t=T, m=M,
                 random_state=RANDOM_STATE, series_store=None, inpaths=DATA_PATHS,
                 negatives_without_positive=NEGATIVES_WITHOUT_POSITIVE):
        self.idx = sample_idx
        self.w = w
        self.t = t
        self.m = m
        self.random_state = random_state
        self.series_store = series_store or SeriesStore(inpaths=inpaths)
        self.negatives_without_positive = negatives_without_positive

    def generer_sample(self):
        """Construit la matrice de features (une ligne par exemple device/sensor/timestamp) et le
        vecteur de labels correspondant. Pour chaque couple device/sensor de self.idx, on charge la
        série complète : les timestamps de vérité terrain donnent les positifs ; un couple sans
        vérité terrain donne des négatifs, tirés parmi les points de sa série."""
        rng = np.random.default_rng(self.random_state)
        feature_rows = []
        labels = []
        for (device_id, sensor_id), group in self.idx.groupby(['device_id', 'sensor_id']):
            series = self.series_store.fetch_series(device_id, sensor_id)
            if series.empty:
                continue
            # Chaque ligne de vérité terrain est un positif ; les lignes factices des couples sans
            # positif (split_train_test) ont un payload_timestamp nul et sont écartées ici.
            positives = group.dropna(subset=['payload_timestamp'])
            is_positive = series['payload_timestamp'].isin(set(positives['payload_timestamp'])).to_numpy()
            positive_positions = np.flatnonzero(is_positive)
            if len(positive_positions) != len(positives):
                print(f"{len(positives) - len(positive_positions)} timestamp(s) de vérité terrain "
                      f"introuvable(s) dans la série {device_id} / {sensor_id}")
            # Avant la position w, les features de lag n'ont pas assez d'historique et valent NaN :
            # on exclut cette zone de chauffe des candidats négatifs.
            # Un point est négatif si son timestamp n'est pas dans la vérité terrain. Pour un couple
            # présent dans la vérité terrain, les autres points peuvent contenir des ruptures non
            # étiquetées : seuls ceux des périodes saines de real_neg_samples/ sont éligibles.
            eligible = ~is_positive
            if len(positives):
                eligible &= series['is_real_neg'].to_numpy(dtype=bool)
            candidates = np.flatnonzero(eligible)
            candidates = candidates[candidates >= self.w]
            n_negatives = self.negatives_without_positive
            negative_positions = self.sample_negative_positions(candidates, n_negatives, rng)
            for pos in positive_positions:
                feature_rows.append(self.build_features(series, int(pos)))
                labels.append(1)
            for pos in negative_positions:
                feature_rows.append(self.build_features(series, int(pos)))
                labels.append(0)
        X = pd.DataFrame(feature_rows)
        y = pd.Series(labels, name='label').reset_index(drop=True)
        return X, y

    def sample_negative_positions(self, candidates, n_negatives, rng):
        """Tire au hasard n_negatifs positions parmi candidates (les positions de la série qui ne
        portent pas de rupture injectée et qui ont assez d'historique)."""
        n = min(len(candidates), n_negatives)
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


def list_series_pairs(inpaths=DATA_PATHS):
    """Couples device_id/sensor_id pour lesquels il existe au moins un parquet
    {device_id}_{sensor_id}_*.parquet dans l'un des inpaths (suppose qu'aucun des deux ids ne
    contient de '_')."""
    inpaths = [inpaths] if isinstance(inpaths, str) else inpaths
    rows = []
    for inpath in inpaths:
        for path in paths.glob(paths.join(inpath, '*.parquet')):
            parts = paths.basename(path).split('_', 2)
            if len(parts) == 3:
                rows.append(parts[:2])
    return pd.DataFrame(rows, columns=['device_id', 'sensor_id']).drop_duplicates()


def split_train_test(vt_df, train_ratio=TRAIN_RATIO, random_state=RANDOM_STATE, inpaths=DATA_PATHS):
    """Découpe en train/test, au niveau des couples device_id/sensor_id : jamais une série à cheval
    sur les deux splits, pour éviter toute fuite (le modèle ne doit pas être évalué sur des séries
    qu'il a vues en train). Les couples découpés sont ceux qui ont des positifs dans la vérité
    terrain (chaque ligne est un positif, la colonne label éventuelle est ignorée) ainsi que ceux
    qui ont une série parquet mais aucune ligne de vérité terrain : pour ces derniers, on ajoute
    une ligne factice (payload_timestamp=NaT) afin que FeatureBuilder les parcoure et en tire des
    négatifs. Les négatifs ne sont pas découpés ici — ils sont tirés par FeatureBuilder parmi tous
    les points non marqués rupture de la série de chaque couple, donc du bon côté du split par
    construction."""
    positives = vt_df.drop(columns='label', errors='ignore')
    positive_pairs = positives[['device_id', 'sensor_id']].drop_duplicates()
    pairs = (pd.concat([positive_pairs, list_series_pairs(inpaths)]).drop_duplicates()
             .sample(frac=1, random_state=random_state).reset_index(drop=True))
    n_train = round(len(pairs) * train_ratio)
    train_pairs = pairs.iloc[:n_train]
    test_pairs = pairs.iloc[n_train:]

    def split_rows(split_pairs):
        pos = positives.merge(split_pairs, on=['device_id', 'sensor_id'])
        empty = split_pairs.merge(positive_pairs, on=['device_id', 'sensor_id'], how='left', indicator=True)
        empty = empty[empty['_merge'] == 'left_only'].drop(columns='_merge')
        empty = empty.assign(payload_timestamp=pd.NaT)
        return pd.concat([pos, empty], ignore_index=True)

    train_idx = split_rows(train_pairs)
    test_idx = split_rows(test_pairs)
    n_train_pos = int(train_idx['payload_timestamp'].notna().sum())
    n_test_pos = int(test_idx['payload_timestamp'].notna().sum())
    print(f"{len(train_pairs)} couple(s) en train / {len(test_pairs)} en test "
          f"({n_train_pos} positif(s) / {n_test_pos}), dont "
          f"{len(train_idx) - n_train_pos} / {len(test_idx) - n_test_pos} sans positif")
    return train_idx, test_idx