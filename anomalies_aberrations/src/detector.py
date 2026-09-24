import os
import warnings
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
    features = pd.DataFrame(index=series.index)
    raw = series['raw_value']
    value_window = raw.rolling(window=t + 1, min_periods=1)
    vmin = value_window.min()
    span = value_window.max() - vmin
    
    flat = span <= 0
    scale = p.NORM_MAX / span.where(~flat)

    for lag in range(w + 1):
        suffix = 't' if lag == 0 else f't-{lag}'
        features[f'delta_t_{suffix}'] = series['time_delta_seconds'].shift(lag)
        delta_v = series['value_diff'].shift(lag)
        features[f'delta_v_{suffix}'] = (scale * delta_v).where(delta_v != 0, 0.0)

    features['delta_t_median'] = series['time_delta_seconds'].rolling(window=t, min_periods=1).median().shift(1)

    padded = np.concatenate([np.full(m, np.nan), raw.to_numpy(dtype=float)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, m + 1)
    normalized = (windows - vmin.to_numpy()[:, None]) * scale.to_numpy()[:, None]
    with warnings.catch_warnings():  # au tout premier point, la fenêtre [pos-M, pos-1] est vide
        warnings.simplefilter('ignore', RuntimeWarning)
        rms_curr = np.sqrt(np.nanmean(normalized ** 2, axis=1))          # [t-M, t], pos inclus
        rms_prev = np.sqrt(np.nanmean(normalized[:, :-1] ** 2, axis=1))  # [t-M, t-1], pos exclu
    jump_ratio = pd.Series((rms_curr - rms_prev) / (np.abs(rms_prev) + 1e-6), index=series.index)
    empty_prev = np.isnan(windows[:, :-1]).all(axis=1)
    features['jump_ratio'] = jump_ratio.where(~flat, 0.0).mask(empty_prev)

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
