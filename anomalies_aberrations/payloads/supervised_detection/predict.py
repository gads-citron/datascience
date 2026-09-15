"""
Applique le modèle de détection de rupture entraîné (training.py) sur la série d'un couple
device_id/sensor_id, lue directement dans les fichiers parquet de synthetic_data/ruptures, et
sauvegarde pour chaque payload_timestamp la prédiction du modèle (0 ou 1).

Usage: python predict.py <device_id> <sensor_id> [--start ...] [--end ...]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from training import SeriesStore, SupervisedModel, W, T, M

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'predictions.csv')


def build_features_vectorized(series, w=W, t=T, m=M):
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


def predict_period(device_id, sensor_id, start=None, end=None, series_store=None, model=None):
    """Charge toute la série device_id/sensor_id (pour que les premiers timestamps de la période
    profitent quand même de l'historique disponible avant start), construit les features de
    chaque timestamp, applique le modèle, puis restreint le résultat à [start, end]."""
    series_store = series_store or SeriesStore()
    series = series_store.fetch_series(device_id, sensor_id)
    columns = ['device_id', 'sensor_id', 'payload_timestamp', 'prediction']
    if series.empty:
        return pd.DataFrame(columns=columns)

    features_start = time.perf_counter()
    features = build_features_vectorized(series)
    features_elapsed = time.perf_counter() - features_start
    print(f"Temps de création des features : {features_elapsed:.2f}s")

    model = model or SupervisedModel.load()
    predict_start = time.perf_counter()
    predictions = model.predict(features)
    predict_elapsed = time.perf_counter() - predict_start
    print(f"Temps de prédiction : {predict_elapsed:.2f}s")

    result = pd.DataFrame({
        'device_id': device_id,
        'sensor_id': sensor_id,
        'payload_timestamp': series['payload_timestamp'],
        'prediction': predictions,
    })
    if start is not None:
        result = result[result['payload_timestamp'] >= pd.Timestamp(start)]
    if end is not None:
        result = result[result['payload_timestamp'] <= pd.Timestamp(end)]
    return result.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Applique le modèle de détection de rupture sur un couple device_id/sensor_id "
                    "et sauvegarde la prédiction (0 ou 1) pour chaque payload_timestamp.")
    parser.add_argument('--device_id')
    parser.add_argument('--sensor_id')
    parser.add_argument('--start', help="Date de début, ex: '2026-01-01 00:00:00'")
    parser.add_argument('--end', help="Date de fin, ex: '2026-09-10 23:59:59'")
    parser.add_argument('--output', default=OUTPUT_PATH)
    args = parser.parse_args()

    start_time = time.perf_counter()
    result = predict_period(args.device_id, args.sensor_id, args.start, args.end)
    result.to_csv(args.output, index=False)
    elapsed = time.perf_counter() - start_time
    print(f"{len(result)} prédiction(s) sauvegardée(s) : {args.output}")
    print(f"Temps d'exécution : {elapsed:.2f}s")

if __name__ == "__main__":
    main()
