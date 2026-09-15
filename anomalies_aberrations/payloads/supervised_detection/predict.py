"""
Applique le modèle de détection de rupture entraîné (training.py) sur la série d'un couple
device_id/sensor_id, lue directement en base ClickHouse (datawarehouse_prod.fact_payload), et
sauvegarde pour chaque payload_timestamp la prédiction du modèle (0 ou 1).

Usage: python predict.py --device_id ... --sensor_id ... --start ... --end ...
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from training import SupervisedModel, W, T, M

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
from db_connector import ClickHouseClient  # noqa: E402

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'predictions.csv')

def query_serie(dev, sens, start, end):
    table = 'datawarehouse_prod.fact_payload'
    return f""" SELECT device_id, payload_metric_id AS sensor_id,
                payload_timestamp, toFloat32OrNull(payload_value) AS raw_value
                FROM {table}
                WHERE device_id = '{dev}'
                AND payload_metric_id = '{sens}'
                AND payload_timestamp BETWEEN '{start}' AND '{end}'
                ORDER BY payload_timestamp;"""


def build_features_vectorized(series, w=W, t=T, m=M):
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

def predict_period(device_id, sensor_id, start, end, client=None, model=None):
    """Charge la série device_id/sensor_id sur [start, end] directement depuis ClickHouse (comme
    src_v2/main.py), construit les features de chaque timestamp puis applique le modèle."""
    if start is None or end is None:
        raise ValueError("start et end sont requis pour interroger ClickHouse")

    client = client or ClickHouseClient('prod')
    series = client.query(query_serie(device_id, sensor_id, start, end))
    columns = ['device_id', 'sensor_id', 'payload_timestamp', 'prediction']
    if series.empty:
        return pd.DataFrame(columns=columns)

    series = series.sort_values('payload_timestamp').drop_duplicates('payload_timestamp').reset_index(drop=True)
    series['value_diff'] = series['raw_value'].diff()
    series['time_delta_seconds'] = series['payload_timestamp'].diff().dt.total_seconds()

    features_start = time.perf_counter()
    features = build_features_vectorized(series)
    features_elapsed = time.perf_counter() - features_start
    print(f"Temps de création des features : {features_elapsed:.2f}s")

    model = model or SupervisedModel.load()
    predict_start = time.perf_counter()
    predictions = model.predict(features)
    predict_elapsed = time.perf_counter() - predict_start
    print(f"Temps de prédiction : {predict_elapsed:.2f}s")

    return pd.DataFrame({
        'device_id': device_id,
        'sensor_id': sensor_id,
        'payload_timestamp': series['payload_timestamp'],
        'prediction': predictions,
    })


def main():
    parser = argparse.ArgumentParser(
        description="Applique le modèle de détection de rupture sur un couple device_id/sensor_id "
                    "et sauvegarde la prédiction (0 ou 1) pour chaque payload_timestamp.")
    parser.add_argument('--device_id', required=True)
    parser.add_argument('--sensor_id', required=True)
    parser.add_argument('--start', required=True, help="Date de début, ex: '2026-01-01 00:00:00'")
    parser.add_argument('--end', required=True, help="Date de fin, ex: '2026-09-10 23:59:59'")
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
