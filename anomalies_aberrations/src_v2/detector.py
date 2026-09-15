import os
import numpy as np
import pandas as pd
import joblib
import param as p

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'rupture_model.joblib')

def load():
    """Recharge le modèle sauvegardé."""
    model = joblib.load(MODEL_PATH)
    return model

def null_ranges(df, dev, sens, col):
    """Un compteur à 0 avant son premier payload non nul est un état normal (pas encore mis en
    service) ; après cette date, c'est une anomalie. Renvoie un masque booléen aligné sur df."""
    result = col.get_first_payload(dev, sens)
    date_first_payload = result.iloc[0, 0] if not result.empty else None
    if date_first_payload is None or pd.isna(date_first_payload):
        return pd.Series(False, index=df.index)
    is_null = df['raw_value'].eq(0)
    is_after = df['payload_timestamp'] > date_first_payload
    return is_null & is_after

def reset_compteurs(df):
    is_reset = df['raw_value'].diff() < 0
    return is_reset

def build_features_vectorized(series, w=p.W, t=p.T, m=p.M):
    """Version vectorisée de FeatureBuilder.build_features (training.py), appliquée à toute la
    série en une fois plutôt que ligne par ligne. Toutes les features sont invariantes à l'échelle
    du compteur (aucune valeur absolue de payload_value) : les features de lag ne sont que des
    shift() sur les colonnes déjà calculées (time_delta_seconds, value_diff), et le saut de valeur
    est exprimé en variation relative (jump_ratio) plutôt qu'en moyenne quadratique brute."""
    features = pd.DataFrame(index=series.index)
    for lag in range(w + 1):
        suffix = 't' if lag == 0 else f't-{lag}'
        features[f'delta_t_{suffix}'] = series['time_delta_seconds'].shift(lag)
        features[f'delta_v_{suffix}'] = series['value_diff'].shift(lag)

    # fenêtre des T points avant pos (donc décalée d'1, pos lui-même exclu)
    features['delta_t_median'] = series['time_delta_seconds'].rolling(window=t, min_periods=1).median().shift(1)

    sq_value = series['raw_value'] ** 2
    # moyenne quadratique entre t-1 et t-M (fenêtre décalée d'1, pos exclu)
    rms_prev = np.sqrt(sq_value.rolling(window=m, min_periods=1).mean()).shift(1)
    # moyenne quadratique entre t et t-M (pos inclus)
    rms_curr = np.sqrt(sq_value.rolling(window=m + 1, min_periods=1).mean())
    # variation relative entre les deux fenêtres, invariante à l'échelle du compteur
    features['jump_ratio'] = (rms_curr - rms_prev) / (rms_prev.abs() + 1e-6)

    return features

def rupture_detection_model(df, model):
    """Détecte les ruptures à l'aide du modèle supervisé entraîné, à la place de PELT : plus besoin
    de segmenter au préalable sur les resets, le modèle est appliqué directement sur toute la série
    (les features de training.py/FeatureBuilder intègrent déjà le contexte via des lags/fenêtres)."""
    is_rupture = pd.Series(False, index=df.index)
    valid_index = df.index[df['raw_value'].notna()]
    if len(valid_index) < 2:
        return is_rupture

    series = df.loc[valid_index, ['payload_timestamp', 'raw_value']].sort_values('payload_timestamp')
    series['value_diff'] = series['raw_value'].diff()
    series['time_delta_seconds'] = series['payload_timestamp'].diff().dt.total_seconds()

    features = build_features_vectorized(series)
    predictions = model.predict(features)
    is_rupture.loc[series.index[predictions == 1]] = True
    return is_rupture

def find_anomalies(sample, model, dev, sens, col):
    anom_df = []
    null_val = null_ranges(sample, dev, sens, col)
    anom_df.append(sample.loc[null_val, ['device_id', 'sensor_id', 'payload_timestamp', 'raw_value']]\
                                    .assign(anomaly_type='is_null'))
    sample = sample.loc[~null_val]
    reset_cpt = reset_compteurs(sample)
    anom_df.append(sample.loc[reset_cpt, ['device_id', 'sensor_id', 'payload_timestamp', 'raw_value']]\
                                    .assign(anomaly_type='reset'))
    is_rupture = rupture_detection_model(sample, model)
    anom_df.append(sample.loc[is_rupture, ['device_id', 'sensor_id', 'payload_timestamp', 'raw_value']]\
                                    .assign(anomaly_type='rupture'))
    return anom_df
