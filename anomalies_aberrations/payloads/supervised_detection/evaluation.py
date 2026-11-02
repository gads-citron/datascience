"""
Évalue un modèle de détection de rupture (training.py), au choix selon le mode :
- 'test'       : évaluation classique sur EXACTEMENT le jeu de test de l'entraînement (même code
                 que training.py : data_preparation.load_train_test, tout est seedé) ;
- 'production' : applique le modèle aux couples device_id/sensor_id d'un CSV (colonnes device_id,
                 sensor_id ; par défaut training_data/evaluation_stage_env.csv) sur une période
                 [start, end] de ClickHouse, et compte les ruptures
                 détectées, comparées aux anomalies déjà connues dans ANOM_TABLE pour un premier
                 contrôle terrain.
Le modèle est désigné par model_uri ('models:/nom/version', 'runs:/<run_id>/model' ou chemin local).

    python evaluation.py test --model-uri models:/rupture-detection-model/3
    python evaluation.py production --model-uri runs:/<run_id>/model \\
        --start 2026-03-01 --end 2026-04-01   # --pairs-csv pour un autre fichier de couples
"""
import argparse
import os
import re
import sys
import time
import warnings
from datetime import timedelta

import mlflow
import numpy as np
import pandas as pd

import paths
from training import SupervisedModel, init_mlflow, MODEL_URI, W, T, M, NORM_MAX
from data_preparation import load_train_test, dataset_hash, TRAINING_DATA

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))

SQL_DIR = os.path.join(os.path.dirname(__file__), 'sql')
ANOM_TABLE = 'Myriam_test.sous_comptage_anomalies'
FACT_TABLE = 'datawarehouse_stage.fact_payload'
DATE_COLUMN = 'payload_timestamp'  # pas de colonne updated_at distincte dans fact_payload ici
# Historique chargé avant START pour que les features de lag (W=7, T=50 points) du premier point
# de la période à évaluer ne soient pas NaN faute de contexte. Marge large car la cadence réelle
# des capteurs varie beaucoup (10 min à plusieurs heures) : ajuster si trop lent/juste.
CONTEXT_MARGIN = timedelta(days=60)

# Couples device_id/sensor_id évalués en mode 'production' par défaut (colonnes site_id, device_id, sensor_id).
PAIRS_CSV = paths.join(TRAINING_DATA, 'evaluation_stage_env.csv')
START = '2026-01-01 00:00:00'
END = '2026-09-01 00:00:00'


def read_sql(name):
    with open(os.path.join(SQL_DIR, name), encoding='utf-8') as f:
        return f.read()


def sql_literal(value):
    """Formate une valeur Python en littéral SQL ClickHouse : chaînes échappées/quotées, dates en
    'YYYY-MM-DD HH:MM:SS', listes de tuples device_id/sensor_id en Array(Tuple(String, String))."""
    if isinstance(value, str):
        return "'" + value.replace("'", "\\'") + "'"
    if isinstance(value, pd.Timestamp):
        return "'" + value.strftime('%Y-%m-%d %H:%M:%S') + "'"
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if items and isinstance(items[0], (list, tuple)):
            return '[' + ', '.join('(' + ', '.join(sql_literal(v) for v in item) + ')' for item in items) + ']'
        return '[' + ', '.join(sql_literal(v) for v in items) + ']'
    return str(value)


def render_sql(name, raw=None, literals=None):
    """Charge le template sql/{name} et substitue ses placeholders {nom} ou {nom:Type} : raw
    substitue tel quel (identifiants de table/colonne), literals formate en littéral SQL."""
    sql = read_sql(name)
    for key, value in (raw or {}).items():
        sql = re.sub(r'\{' + re.escape(key) + r'(:[^}]+)?\}', lambda m, v=value: v, sql)
    for key, value in (literals or {}).items():
        sql = re.sub(r'\{' + re.escape(key) + r'(:[^}]+)?\}', lambda m, v=value: sql_literal(v), sql)
    return sql


def read_pairs_csv(pairs_csv):
    """Lit le CSV (colonnes device_id et sensor_id) des couples à évaluer en mode 'production' et
    renvoie la liste de tuples (device_id, sensor_id) sans doublon."""
    df = paths.read_csv(pairs_csv, dtype={'device_id': str, 'sensor_id': str})
    missing = {'device_id', 'sensor_id'} - set(df.columns)
    if missing:
        raise ValueError(f"{pairs_csv} : colonne(s) manquante(s) {sorted(missing)} (attendu : device_id, sensor_id)")
    return list(df[['device_id', 'sensor_id']].dropna().drop_duplicates().itertuples(index=False, name=None))


def query_payloads(pairs, start, end, client, context_margin=CONTEXT_MARGIN):
    """Exécute sql/payloads.sql pour les couples device_id/sensor_id de pairs : interroge
    [ctx_start, end] (ctx_start = start - context_margin) pour avoir assez d'historique avant
    start et calculer les features de lag sans trou. Chaque ligne renvoyée porte un flag in_window
    (True si dans [start, end], False si c'est du pur contexte, jamais reporté dans les résultats)."""
    ctx_start = pd.Timestamp(start) - context_margin
    sql = render_sql(
        'payloads.sql',
        raw={'fact_table': FACT_TABLE, 'date_column': DATE_COLUMN},
        literals={
            'start': start, 'end': end,
            'floor': ctx_start,  # pas de colonne date_column distincte ici : le garde-fou floor
                                  # n'a rien à filtrer en plus de [ctx_start, newest_ts]
            'ctx_start': ctx_start,
            'newest_ts': end,
            'pairs': pairs,
        },
    )
    return client.query(sql)


def query_existing_anomalies(pairs, min_ts, max_ts, client):
    """Exécute sql/existing_anomalies.sql : anomalies déjà enregistrées dans OUT_TABLE pour les
    couples pairs sur [min_ts, max_ts]."""
    sql = render_sql(
        'existing_anomalies.sql',
        raw={'out_table': ANOM_TABLE},
        literals={'pairs': pairs, 'min_ts': min_ts, 'max_ts': max_ts},
    )
    return client.query(sql)


def build_features_vectorized(series, w=W, t=T, m=M):
    """Reprise vectorisée de FeatureBuilder.build_features (training.py) pour calculer les
    features de toute une série en une fois plutôt que point par point."""
    features = pd.DataFrame(index=series.index)
    raw = series['raw_value']
    value_window = raw.rolling(window=t + 1, min_periods=1)
    vmin = value_window.min()
    span = value_window.max() - vmin
    flat = span <= 0
    scale = NORM_MAX / span.where(~flat)

    for lag in range(w + 1):
        suffix = 't' if lag == 0 else f't-{lag}'
        features[f'delta_t_{suffix}'] = series['time_delta_seconds'].shift(lag)
        delta_v = series['value_diff'].shift(lag)
        features[f'delta_v_{suffix}'] = (scale * delta_v).where(delta_v != 0, 0.0)

    # fenêtre des T points avant pos (donc décalée d'1, pos lui-même exclu)
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


def predict_batch(payloads, model):
    """Calcule les features et applique le modèle groupe par groupe (device_id, sensor_id) sur
    l'ensemble contexte+fenêtre renvoyé par query_payloads, puis renvoie une ligne par point avec
    sa prédiction et son flag in_window (le contexte avant START ne sert qu'à amorcer les features
    de lag, il n'est jamais reporté dans les résultats)."""
    results = []
    for (device_id, sensor_id), group in payloads.groupby(['device_id', 'sensor_id']):
        series = group.sort_values('payload_timestamp').drop_duplicates('payload_timestamp').reset_index(drop=True)
        series['value_diff'] = series['raw_value'].diff()
        series['time_delta_seconds'] = series['payload_timestamp'].diff().dt.total_seconds()
        features = build_features_vectorized(series)
        predictions = model.predict(features)
        results.append(pd.DataFrame({
            'device_id': device_id,
            'sensor_id': sensor_id,
            'payload_timestamp': series['payload_timestamp'],
            'prediction': predictions,
            'in_window': series['in_window'].astype(bool).to_numpy(),  # ClickHouse renvoie 0/1
        }))
    columns = ['device_id', 'sensor_id', 'payload_timestamp', 'prediction', 'in_window']
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=columns)


def cnt_anom(predictions, existing_anomalies):
    """Compare, par couple device_id/sensor_id, le nombre de ruptures prédites par le modèle (sur
    les points in_window uniquement) au nombre d'anomalies déjà connues dans OUT_TABLE -- un
    premier contrôle terrain, pas un vrai rapprochement temporel point à point."""
    reported = predictions[predictions['in_window']]
    predicted_counts = (reported[reported['prediction'] == 1]
                         .groupby(['device_id', 'sensor_id']).size().rename('n_predicted'))
    existing_counts = (existing_anomalies.groupby(['device_id', 'sensor_id']).size().rename('n_existing')
                        if not existing_anomalies.empty
                        else pd.Series(name='n_existing', dtype=int))
    total_points = reported.groupby(['device_id', 'sensor_id']).size().rename('total_points')
    summary = pd.concat([total_points, predicted_counts, existing_counts], axis=1).fillna(0).astype(int)
    return summary.reset_index()


def summarize_by_device(predictions):
    """Anomalies prédites par device_id (tous capteurs confondus, points in_window uniquement) :
    total, nombre de jours distincts ayant au moins une anomalie, et moyenne par jour sur ces jours
    (même définition que la moyenne par jour calculée sur electric_anomalies)."""
    anomalies = predictions[predictions['in_window'] & (predictions['prediction'] == 1)]
    anomalies = anomalies.assign(day=anomalies['payload_timestamp'].dt.date)
    by_device = anomalies.groupby('device_id').agg(n_anomalies=('day', 'size'), n_days=('day', 'nunique'))
    by_device['avg_per_day'] = by_device['n_anomalies'] / by_device['n_days']
    return by_device.reset_index()


def query_device_info(device_ids, client):
    """site_name et customer_account_name de chaque device_id (datamart_sous_comptage.device_properties)."""
    sql = f"""SELECT device_id, any(site_name) AS site_name, any(customer_account_name) AS customer_account_name
        FROM datamart_sous_comptage.device_properties
        WHERE device_id IN {sql_literal(list(device_ids))}
        GROUP BY device_id"""
    return client.query(sql)


def print_device_maxima(by_device, client):
    """Affiche le device_id avec le plus d'anomalies au total et celui avec la plus forte moyenne
    par jour, avec leur site_name et customer_account_name. Si device_properties n'a pas ces
    colonnes ou ne répond pas, les noms sont omis plutôt que de perdre le résultat."""
    if by_device.empty:
        print("\nAucune anomalie détectée.")
        return
    try:
        info = query_device_info(by_device['device_id'], client)
        by_device = by_device.merge(info, on='device_id', how='left')
    except Exception as exc:
        print(f"\n(site_name / customer_account_name indisponibles : {exc})")
        by_device = by_device.assign(site_name=None, customer_account_name=None)

    def describe(row, value):
        return (f"device_id {row['device_id']} — site « {row['site_name']} », "
                f"client « {row['customer_account_name']} » : {value}")

    top_total = by_device.loc[by_device['n_anomalies'].idxmax()]
    top_avg = by_device.loc[by_device['avg_per_day'].idxmax()]
    print("\nMax d'anomalies au total :")
    print(describe(top_total, f"{int(top_total['n_anomalies'])} anomalie(s) sur {int(top_total['n_days'])} jour(s)"))
    print("Max d'anomalies par jour en moyenne :")
    print(describe(top_avg, f"{top_avg['avg_per_day']:.2f} par jour ({int(top_avg['n_anomalies'])} anomalie(s) sur {int(top_avg['n_days'])} jour(s))"))


def load_model(model_uri=MODEL_URI):
    """Charge le modèle désigné par model_uri ('models:/nom/version', 'runs:/<run_id>/model' ou
    chemin local). Le type (LightGBM ou réseau de neurones) est lu dans les flavors MLflow du
    modèle, pas dans training.MODEL_TYPE, qui peut avoir changé depuis l'entraînement. Renvoie
    (SupervisedModel, run_id du run d'entraînement ou None)."""
    init_mlflow()
    info = mlflow.models.get_model_info(model_uri)
    model_type = 'lightgbm' if 'lightgbm' in info.flavors else 'mlp'
    print(f"Modèle {model_uri} : type {model_type}, run {info.run_id}")
    return SupervisedModel.load(model_uri, model_type=model_type), info.run_id


def evaluate_test_set(model, run_id=None):
    """Mode 'test' : évalue le modèle sur le jeu de test de training.py, reconstruit par le même
    code (load_train_test) et les mêmes seeds. Si le run d'entraînement a loggé l'empreinte de
    son test set (test_set_hash), vérifie qu'elle est identique."""
    _, _, X_test, y_test = load_train_test()
    test_hash = dataset_hash(X_test, y_test)
    logged_hash = mlflow.get_run(run_id).data.params.get('test_set_hash') if run_id else None
    if logged_hash is None:
        print(f"Empreinte du test set : {test_hash} (pas d'empreinte loggée à l'entraînement, "
              f"identité non vérifiable)")
    elif logged_hash == test_hash:
        print(f"Test set identique à celui de l'entraînement (empreinte {test_hash})")
    else:
        print(f"ATTENTION : test set différent de celui de l'entraînement "
              f"({test_hash} ici, {logged_hash} à l'entraînement) : données, code ou seeds ont changé")
    print(f"Test : {int(y_test.sum())} positif(s) / {int((y_test == 0).sum())} négatif(s)\n")
    report = model.evaluate(X_test, y_test)
    print(report)
    return model.metrics(X_test, y_test)


def evaluate_production(model, pairs_csv, start, end):
    """Mode 'production' : applique model aux couples device_id/sensor_id listés dans pairs_csv
    sur [start, end] (ClickHouse) et compte les ruptures détectées."""
    from db_connector import ClickHouseClient  # import tardif : le mode 'test' n'en a pas besoin
    client = ClickHouseClient('stage')

    pairs = read_pairs_csv(pairs_csv)
    print(f"{len(pairs)} couple(s) device/sensor lus dans {pairs_csv}")
    if not pairs:
        print("Aucun couple à évaluer, arrêt.")
        return None

    query_start = time.perf_counter()
    payloads = query_payloads(pairs, start, end, client)
    print(f"{len(payloads)} ligne(s) chargée(s) (contexte inclus) en {time.perf_counter() - query_start:.1f}s")

    predict_start = time.perf_counter()
    predictions = predict_batch(payloads, model)
    print(f"Prédictions calculées en {time.perf_counter() - predict_start:.1f}s")

    existing = query_existing_anomalies(pairs, start, end, client)
    summary = cnt_anom(predictions, existing)

    reported = predictions[predictions['in_window']]
    n_predicted = int((reported['prediction'] == 1).sum())
    pct = 100 * n_predicted / len(reported) if len(reported) else 0
    print(f"\n{n_predicted} rupture(s) prédite(s) sur {len(reported)} point(s) évalué(s) ({pct:.3f}%)")
    print(f"{len(existing)} anomalie(s) déjà connue(s) dans {ANOM_TABLE} sur la période")
    print("\nRésumé par couple device/sensor :")
    print(summary.to_string(index=False))

    print_device_maxima(summarize_by_device(predictions), client)
    return summary


def main(model_uri=MODEL_URI, mode='test', pairs_csv=PAIRS_CSV, start=START, end=END):
    """Point d'entrée : mode 'test' (jeu de test de l'entraînement) ou 'production' (couples du
    CSV pairs_csv, par défaut training_data/evaluation_stage_env.csv, sur [start, end]) avec le
    modèle désigné par model_uri."""
    model, run_id = load_model(model_uri)
    if mode == 'test':
        return evaluate_test_set(model, run_id)
    if mode == 'production':
        return evaluate_production(model, pairs_csv, start, end)
    raise ValueError(f"mode inconnu: {mode} ('test' ou 'production')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['test', 'production'])
    parser.add_argument('--model-uri', default=MODEL_URI, help=f"défaut : {MODEL_URI}")
    parser.add_argument('--pairs-csv', default=PAIRS_CSV,
                        help=f"mode production : CSV des couples à évaluer (colonnes device_id, sensor_id) ; défaut : {PAIRS_CSV}")
    parser.add_argument('--start', default=START, help="mode production : début de période")
    parser.add_argument('--end', default=END, help="mode production : fin de période")
    args = parser.parse_args()
    main(args.model_uri, args.mode, args.pairs_csv, args.start, args.end)
