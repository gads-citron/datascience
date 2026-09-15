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

def null_ranges(df):
    return df["raw_value"].eq(0)

def reset_compteurs(df):
    is_reset = df['raw_value'].diff() < 0
    return is_reset

def build_features_vectorized(series, w=p.W, t=p.T, m=p.M):
    """Version vectorisée de FeatureBuilder.build_features (training.py), appliquée à toute la
    série en une fois plutôt que ligne par ligne : les features de lag ne sont que des shift()
    sur les colonnes déjà calculées (raw_value, time_delta_seconds, value_diff), et les agrégats
    (médiane, moyenne quadratique) des rolling — même sémantique de fenêtre, en vectorisé."""
    features = pd.DataFrame(index=series.index)
    for lag in range(w + 1):
        suffix = 't' if lag == 0 else f't-{lag}'
        features[f'value_{suffix}'] = series['raw_value'].shift(lag)
        features[f'delta_t_{suffix}'] = series['time_delta_seconds'].shift(lag)
        features[f'delta_v_{suffix}'] = series['value_diff'].shift(lag)

    # fenêtre des T points avant pos (donc décalée d'1, pos lui-même exclu)
    features['delta_t_median'] = series['time_delta_seconds'].rolling(window=t, min_periods=1).median().shift(1)

    sq_value = series['raw_value'] ** 2
    # moyenne quadratique entre t-1 et t-M (fenêtre décalée d'1, pos exclu)
    features['value_rms_prev'] = np.sqrt(sq_value.rolling(window=m, min_periods=1).mean()).shift(1)
    # moyenne quadratique entre t et t-M (pos inclus)
    features['value_rms_curr'] = np.sqrt(sq_value.rolling(window=m + 1, min_periods=1).mean())

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

def find_anomalies(sample, model):
    anom_df = []
    null_val = null_ranges(sample)
    anom_df.append(sample.loc[null_val, ['device_id', 'sensor_id', 'payload_timestamp', 'raw_value']]\
                                    .assign(anomaly_type='is_null',
                                            date_start=lambda d: d['payload_timestamp'],
                                            date_end=lambda d: d['payload_timestamp']))
    sample = sample.loc[~null_val]
    reset_cpt = reset_compteurs(sample)
    anom_df.append(sample.loc[reset_cpt, ['device_id', 'sensor_id', 'payload_timestamp', 'raw_value']]\
                                    .assign(anomaly_type='reset',
                                            date_start=lambda d: d['payload_timestamp'],
                                            date_end=lambda d: d['payload_timestamp']))
    is_rupture = rupture_detection_model(sample, model)
    anom_df.append(sample.loc[is_rupture, ['device_id', 'sensor_id', 'payload_timestamp', 'raw_value']]\
                                    .assign(anomaly_type='rupture',
                                            date_start=lambda d: d['payload_timestamp'],
                                            date_end=lambda d: d['payload_timestamp']))
    return anom_df

#-----# Deprecated for now
def commissioning(df, start):
    valid = df.dropna(subset=['raw_value'])
    valid = valid[valid['raw_value'] != 0]
    empty_result = pd.DataFrame(columns=['device_id', 'sensor_id', 'ts', 'raw_value', 'anomaly_type', 'date', 'segment_start', 'segment_end'])
    if valid.empty:
        return empty_result

    first = valid.iloc[0]
    date = pd.to_datetime(first['ts'], unit='s', utc=True).tz_convert('Europe/Paris')
    if date.date() == pd.Timestamp(start).date():
        return empty_result
    return pd.DataFrame([{
        'device_id': first['device_id'],
        'sensor_id': first['sensor_id'],
        'ts': first['ts'],
        'raw_value': first['raw_value'],
        'anomaly_type': 'commissioning',
        'date': date,
        'segment_start': date,
        'segment_end': date,
    }])

def multi_freq_detection(df, e=1):
    delta = df['ts'].diff()
    T = delta.median()
    r = delta % T
    proportions = delta[r > 0].value_counts(normalize=True)
    proportions = proportions[proportions >= 0.2]
    if len(proportions) > 1:
        if proportions.index.min() <= 5:
            return proportions
        if np.sum(proportions.index % proportions.index.min()) > 0:
            return proportions
    return []

def multi_frequence(df, start, end):
    periodes = multi_freq_detection(df)
    res = pd.DataFrame(columns=['device_id', 'sensor_id', 'ts',
                                'raw_value', 'anomaly_type', 'date',
                                'segment_start', 'segment_end'])
    if len(periodes) > 1 and periodes.index.max() - periodes.index.min() > 10:
        start_ts = pd.Timestamp(start).tz_localize('Europe/Paris')
        end_ts = pd.Timestamp(end).tz_localize('Europe/Paris')
        res = pd.DataFrame([{
            'device_id': df['device_id'].iloc[0],
            'sensor_id': df['sensor_id'].iloc[0],
            'ts': np.nan,
            'raw_value': np.nan,
            'anomaly_type': 'multi-freq',
            'date': start_ts,
            'segment_start': start_ts,
            'segment_end': end_ts,
        }])
    return res
