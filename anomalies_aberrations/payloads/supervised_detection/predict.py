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
import warnings

import numpy as np
import pandas as pd

from training import SupervisedModel, W, T, M, NORM_MAX

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
    est exprimé en variation relative (jump_ratio) plutôt qu'en moyenne quadratique brute.

    Les delta_v et jump_ratio sont calculés sur la fenêtre des t points précédents ramenée sur
    l'échelle 0-NORM_MAX, comme le fait normalize_window côté training. La transformation étant
    affine, on n'a besoin ici que de ses deux paramètres — le minimum et l'amplitude de la fenêtre —
    plutôt que de normaliser la fenêtre point par point. Toute évolution doit rester alignée sur
    training.build_features, sinon le modèle ne voit pas les mêmes features qu'à l'entraînement."""
    features = pd.DataFrame(index=series.index)
    raw = series['raw_value']
    value_window = raw.rolling(window=t + 1, min_periods=1)
    vmin = value_window.min()
    span = value_window.max() - vmin
    # Une fenêtre plate n'a pas d'échelle : on neutralise la division ici et on retombe sur 0 plus
    # bas, comme le fait normalize_window en renvoyant des zéros.
    flat = span <= 0
    scale = NORM_MAX / span.where(~flat)

    for lag in range(w + 1):
        suffix = 't' if lag == 0 else f't-{lag}'
        features[f'delta_t_{suffix}'] = series['time_delta_seconds'].shift(lag)
        delta_v = series['value_diff'].shift(lag)
        features[f'delta_v_{suffix}'] = (scale * delta_v).where(delta_v != 0, 0.0)

    # fenêtre des T points avant pos (donc décalée d'1, pos lui-même exclu)
    features['delta_t_median'] = series['time_delta_seconds'].rolling(window=t, min_periods=1).median().shift(1)

    # Moyennes quadratiques sur la fenêtre normalisée. On reconstruit explicitement les M+1 points
    # de la fenêtre pour retrancher vmin AVANT d'élever au carré. Développer la moyenne de (x-vmin)²
    # en E[x²] - 2·vmin·E[x] + vmin² tiendrait en trois rolling means, mais serait inutilisable sur
    # un compteur cumulatif : avec x ≈ 1e6, x² ≈ 1e12 pour une variance locale de quelques unités,
    # la soustraction perd toute précision et le ratio part à plusieurs milliers.
    padded = np.concatenate([np.full(m, np.nan), raw.to_numpy(dtype=float)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, m + 1)
    normalized = (windows - vmin.to_numpy()[:, None]) * scale.to_numpy()[:, None]
    with warnings.catch_warnings():  # au tout premier point, la fenêtre [pos-M, pos-1] est vide
        warnings.simplefilter('ignore', RuntimeWarning)
        rms_curr = np.sqrt(np.nanmean(normalized ** 2, axis=1))          # [t-M, t], pos inclus
        rms_prev = np.sqrt(np.nanmean(normalized[:, :-1] ** 2, axis=1))  # [t-M, t-1], pos exclu
    jump_ratio = pd.Series((rms_curr - rms_prev) / (np.abs(rms_prev) + 1e-6), index=series.index)
    # fenêtre plate : pas d'échelle, donc 0. Premier point de la série : la fenêtre [t-M, t-1] est
    # entièrement hors série, il n'y a rien à comparer, donc NaN (et non 0).
    empty_prev = np.isnan(windows[:, :-1]).all(axis=1)
    features['jump_ratio'] = jump_ratio.where(~flat, 0.0).mask(empty_prev)

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
    parser.add_argument('--start', default = '2026-01-01 00:00:00', help="Date de début, ex: '2026-01-01 00:00:00'")
    parser.add_argument('--end', default = '2026-09-10 00:00:00', help="Date de fin, ex: '2026-09-10 23:59:59'")
    parser.add_argument('--output', default=OUTPUT_PATH)
    args = parser.parse_args()

    start_time = time.perf_counter()
    result = predict_period(args.device_id, args.sensor_id, args.start, args.end)
    result.to_csv(args.output, index=False)
    elapsed = time.perf_counter() - start_time
    n_positives = int((result['prediction'] == 1).sum())
    pct = 100 * n_positives / len(result) if len(result) else 0
    print(f"{len(result)} prédiction(s) sauvegardée(s) : {args.output}")
    print(f"{n_positives} rupture(s) détectée(s) ({pct:.2f}% des points)")
    print(f"Temps d'exécution : {elapsed:.2f}s")

if __name__ == "__main__":
    main()
