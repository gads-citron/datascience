import os
import sys

import numpy as np
import pandas as pd
import ruptures as rpt
import plotly.graph_objects as go
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from db_connector import ClickHouseClient

client = ClickHouseClient()
WINDOW = 4
SOURCE_VT = os.path.join(os.path.dirname(__file__), 'data', 'sample_anomalies_IQ_2026', 'verite_terrain_IQ_2026.csv')
# SOURCE_VT = os.path.join(os.path.dirname(__file__), 'data', 'sample_anomalies_2026_trim_1', 'verite_terrain.csv')
SOURCE_ANOMALIES_ALL = os.path.join(os.path.dirname(__file__), 'anomalies_all.csv')
PEN = 0.01  # pénalité PELT (modèle "l2" sur le signal log10) : plus elle est haute, moins il y a de ruptures détectées
MIN_RATIO = 5.  # si défini (float) : une rupture n'est gardée que si elle passe le test défini par RATIO_BASIS ; si None, aucun filtre, toutes les ruptures trouvées par PELT sont gardées
RATIO_BASIS = 'level'  # 'level' : after_mean >= MIN_RATIO x before_mean (comportement historique, adapté à un baseline proche de 0) ; 'step' : (after_mean - before_mean) >= MIN_RATIO x incrément typique (médiane des |diff|) du segment précédent (adapté à un saut sur un signal déjà loin de 0, mais confond pente et bruit) ; 'trend' : compare le saut réel à ce qu'on obtiendrait en extrapolant la pente du segment précédent, normalisé par le bruit résiduel autour de cette pente (distingue une vraie marche d'une simple continuation de tendance)
SMOOTHING_HOURS = 0  # nombre d'heures précédentes utilisées pour lisser le signal avant détection ; mettre à 0 pour désactiver le lissage
CLIP_NEGATIVE = True  # True : ramène les valeurs négatives à 0 avant transformation (comportement historique) ; False : conserve les valeurs négatives telles quelles
APPLY_LOG = True  # True : applique une compression log10 (signée si CLIP_NEGATIVE=False, pour rester définie sur les négatifs) ; False : pas de transformation logarithmique
MIN_SIZE = 2  # nombre minimum de points entre deux ruptures PELT (168 = 1 semaine en données horaires) ; empêche PELT de découper une pente douce en plein de petits segments
INTERPOLATE_MISSING = False  # True : comble les trous internes (valeurs manquantes) par interpolation linéaire avant détection ; False : les lignes manquantes sont simplement ignorées (comportement historique)
LINEAR_OUTLIER_THRESHOLD = 10  # un point est marqué aberrant si son résidu au modèle linéaire dépasse LINEAR_OUTLIER_THRESHOLD x la MAD des résidus
SAMPLE = 1  # anomalies IQ 2025 ; 2 = anomalies ss_comptage 2025

def query_serie_subcounting(dev_id, sens_id, start, end, smoothing_hours=SMOOTHING_HOURS):
    dataset = 'datamart_sous_comptage.distributed_dataset_hourly_payloads_merged'
    if smoothing_hours > 0:
        smoothed_expr = f"""avg(sum_payload_value_hour_curr) OVER (
        ORDER BY payload_expected_time
        ROWS BETWEEN {smoothing_hours} PRECEDING AND CURRENT ROW
    ) AS smoothed_value"""
    else:
        smoothed_expr = "sum_payload_value_hour_curr AS smoothed_value"
    query = f"""SELECT
        device_id,
        sensor_id,
        site_id,
        payload_expected_time,
        eval_prev_hour,
        sum_payload_value_hour_curr,
        {smoothed_expr}
    FROM {dataset}
    WHERE device_id = '{dev_id}'
    AND sensor_id = '{sens_id}'
    AND payload_expected_time BETWEEN '{start}' AND '{end}'
    ORDER BY payload_expected_time;"""
    return query

def detect_changepoints(df, pen=PEN, min_ratio=MIN_RATIO, ratio_basis=RATIO_BASIS, clip_negative=CLIP_NEGATIVE, apply_log=APPLY_LOG, min_size=MIN_SIZE, interpolate_missing=INTERPOLATE_MISSING, reverse=False):
    is_changepoint = pd.Series(False, index=df.index)
    smoothed = df['smoothed_value']
    if interpolate_missing:
        # ne comble que les trous internes (entre deux valeurs connues) ; les NaN en bord de série restent NaN
        smoothed = smoothed.interpolate(method='linear')
    valid_index = df.index[smoothed.notna()]
    raw_signal = smoothed.loc[valid_index].to_numpy()

    if reverse:
        raw_signal = raw_signal[::-1]
        valid_index = valid_index[::-1]

    signal = np.clip(raw_signal, 0, None) if clip_negative else raw_signal
    if apply_log:
        signal = np.sign(signal) * np.log10(np.abs(signal) + 1)
    signal = signal.reshape(-1, 1)
    if len(signal) < 2:
        return is_changepoint

    algo = rpt.Pelt(model='l2', min_size=min_size).fit(signal)
    bkps = algo.predict(pen=pen)
    bounds = [0] + bkps

    if min_ratio is None:
        # pas de filtre : on garde toutes les ruptures trouvées par PELT
        kept = [mid - 1 for start, mid, end in zip(bounds, bounds[1:], bounds[2:])]
    else:
        # ne garde que les ruptures où le segment suivant est nettement PLUS HAUT que le précédent
        kept = []
        for start, mid, end in zip(bounds, bounds[1:], bounds[2:]):
            before_mean = raw_signal[start:mid].mean()
            after_mean = raw_signal[mid:end].mean()
            if ratio_basis == 'level':
                if after_mean >= min_ratio * max(before_mean, 1e-9):
                    kept.append(mid - 1)
            elif ratio_basis == 'step':
                before_segment = raw_signal[start:mid]
                step = np.median(np.abs(np.diff(before_segment))) if len(before_segment) > 1 else 0.0
                if (after_mean - before_mean) >= min_ratio * max(step, 1e-9):
                    kept.append(mid - 1)
            elif ratio_basis == 'trend':
                before_segment = raw_signal[start:mid]
                if len(before_segment) >= 2:
                    x = np.arange(len(before_segment))
                    slope, intercept = np.polyfit(x, before_segment, 1)
                    expected_at_boundary = intercept + slope * len(before_segment)
                    noise = np.median(np.abs(before_segment - (intercept + slope * x)))
                else:
                    expected_at_boundary, noise = before_mean, 0.0
                if (after_mean - expected_at_boundary) >= min_ratio * max(noise, 1e-9):
                    kept.append(mid - 1)
            else:
                raise ValueError(f"ratio_basis inconnu: {ratio_basis}")

    if kept:
        is_changepoint.loc[valid_index[kept]] = True
    return is_changepoint

def detect_linear_outliers(df, threshold=LINEAR_OUTLIER_THRESHOLD, interpolate_missing=INTERPOLATE_MISSING):
    # ajuste une regression lineaire sur toute la serie et flag les points dont le residu s'ecarte trop du modele
    is_outlier = pd.Series(False, index=df.index)
    smoothed = df['smoothed_value']
    if interpolate_missing:
        smoothed = smoothed.interpolate(method='linear')
    valid_index = df.index[smoothed.notna()]
    raw_signal = smoothed.loc[valid_index].to_numpy()
    if len(raw_signal) < 2:
        return is_outlier

    x = np.arange(len(raw_signal))
    slope, intercept = np.polyfit(x, raw_signal, 1)
    residuals = raw_signal - (intercept + slope * x)

    mad = np.median(np.abs(residuals - np.median(residuals)))
    if mad == 0:
        return is_outlier

    outlier_mask = np.abs(residuals) >= threshold * mad
    is_outlier.loc[valid_index[outlier_mask]] = True
    return is_outlier

def plot_changepoints(df, title):
    start_times = df.loc[df['is_rupture_start'], 'payload_expected_time'].tolist()
    # end_times = df.loc[df['is_rupture_end'], 'payload_expected_time'].tolist()
    # bounds = [df['payload_expected_time'].min()] + sorted(start_times + end_times) + [df['payload_expected_time'].max()]
    bounds = [df['payload_expected_time'].min()] + sorted(start_times) + [df['payload_expected_time'].max()]

    fig = go.Figure()
    for i in range(len(bounds) - 1):
        fig.add_vrect(
            x0=bounds[i], x1=bounds[i + 1],
            fillcolor='#6E7BF2' if i % 2 == 0 else '#B7BEF7',
            opacity=0.15, line_width=0,
        )
    fig.add_trace(go.Scatter(
        x=df['payload_expected_time'], y=df['sum_payload_value_hour_curr'],
        mode='lines', name='sum_payload_value_hour_curr', line=dict(color='#2B2F77', width=2)
    ))
    outliers = df.loc[df['is_linear_outlier']]
    fig.add_trace(go.Scatter(
        x=outliers['payload_expected_time'], y=outliers['sum_payload_value_hour_curr'],
        mode='markers', name='anomalie modèle linéaire',
        marker=dict(color='#F2A93B', size=9, symbol='circle-open', line=dict(width=2)),
    ))
    for t in start_times:
        fig.add_vline(x=t, line=dict(color='#D64550', width=1, dash='dash'))
        fig.add_annotation(
            x=t, y=1, yref='paper', yanchor='bottom', showarrow=False,
            text=t.strftime('%Y-%m-%d %H:%M'), textangle=-90,
            font=dict(color='#D64550', size=10),
        )
    # for t in end_times:
    #     fig.add_vline(x=t, line=dict(color='#2E9E5B', width=1, dash='dash'))
    fig.update_layout(title=title, xaxis_title='payload_expected_time', yaxis_title='sum_payload_value_hour_curr')
    fig.show()

if __name__ == "__main__":
    if SAMPLE == 1:
        vt_df = pd.read_csv(SOURCE_VT)
        vt_df['payload_expected_time'] = pd.to_datetime(vt_df['payload_expected_time'], format='ISO8601', utc=True).dt.tz_convert('Europe/Paris')
        pairs = vt_df[['device_id', 'sensor_id']].drop_duplicates()[:13].itertuples(index=False)
        start = vt_df['payload_expected_time'].min().strftime('%Y-%m-%d %H:%M:%S')
        end = vt_df['payload_expected_time'].max().strftime('%Y-%m-%d %H:%M:%S')
    elif SAMPLE == 2:
        anomalies_all_df = pd.read_csv(SOURCE_ANOMALIES_ALL)
        anomalies_all_df['payload_expected_time'] = pd.to_datetime(anomalies_all_df['payload_expected_time'], format='ISO8601', utc=True)
        anomalies_2025_df = anomalies_all_df[
            (anomalies_all_df['payload_expected_time'] >= pd.Timestamp('2025-01-01', tz='UTC')) &
            (anomalies_all_df['payload_expected_time'] < pd.Timestamp('2025-04-01', tz='UTC'))
        ]
        pairs = anomalies_2025_df[['device_id', 'sensor_id']].drop_duplicates().itertuples(index=False)
        start = anomalies_2025_df['payload_expected_time'].min().strftime('%Y-%m-%d %H:%M:%S')
        end = anomalies_2025_df['payload_expected_time'].max().strftime('%Y-%m-%d %H:%M:%S')
    else:
        raise ValueError(f"SAMPLE inconnu: {SAMPLE}")

    OUTPUT_PATH = './ruptures_detectees.csv'
    SAVE_EVERY = 100

    def build_ruptures_df(frames):
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['device_id', 'sensor_id', 'anomaly_date', 'type'])

    ruptures_found = []
    for i, (dev_id, sens_id) in tqdm(enumerate(pairs, start=1)):
        print(dev_id, sens_id)
        continue
        df = client.query(query_serie_subcounting(dev_id, sens_id, start, end))
        if df.empty:
            continue
        df['is_linear_outlier'] = detect_linear_outliers(df)
        # PELT tourne sur les données nettoyées : les points marqués aberrants par le modèle linéaire sont mis à NaN
        # (donc exclus par detect_changepoints, comme des valeurs manquantes) avant la détection de rupture
        df_cleaned = df.copy()
        df_cleaned.loc[df_cleaned['is_linear_outlier'], 'smoothed_value'] = np.nan
        df['is_rupture_start'] = detect_changepoints(df_cleaned)
        # if df.loc[df['is_rupture_start']].empty:
        #     continue
        plot_changepoints(df, f"{dev_id} / {sens_id}")
        ruptures_found.append(
            df.loc[df['is_rupture_start'], ['device_id', 'sensor_id', 'payload_expected_time']]
            .rename(columns={'payload_expected_time': 'anomaly_date'})
            .assign(type='rupture')
        )
        ruptures_found.append(
            df.loc[df['is_linear_outlier'], ['device_id', 'sensor_id', 'payload_expected_time']]
            .rename(columns={'payload_expected_time': 'anomaly_date'})
            .assign(type='linear_outlier')
        )
        if i % SAVE_EVERY == 0:
            checkpoint_df = build_ruptures_df(ruptures_found)
            if not checkpoint_df.empty:
                checkpoint_df.to_csv(OUTPUT_PATH, index=False)
                print(f"[checkpoint] {i} devices traités, dataset sauvegardé dans {OUTPUT_PATH}")

    ruptures_df = build_ruptures_df(ruptures_found)
    if not ruptures_df.empty:
        ruptures_df.to_csv(OUTPUT_PATH, index=False)
    # print(ruptures_df)
