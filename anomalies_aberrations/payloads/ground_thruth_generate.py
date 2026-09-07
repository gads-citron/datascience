import os
import sys
import numpy as np
import pandas as pd
import ruptures as rpt
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from db_connector import ClickHouseClient
from detection_multi_frequence import detection

PEN = 0.01  # pénalité PELT (modèle "l2" sur le signal log10) : plus elle est haute, moins il y a de ruptures détectées
MIN_RATIO = 5.  # si défini (float) : une rupture n'est gardée que si elle passe le test défini par RATIO_BASIS ; si None, aucun filtre, toutes les ruptures trouvées par PELT sont gardées
SMOOTHING_HOURS = 0  # nombre d'heures précédentes utilisées pour lisser le signal avant détection ; mettre à 0 pour désactiver le lissage
CLIP_NEGATIVE = True  # True : ramène les valeurs négatives à 0 avant transformation (comportement historique) ; False : conserve les valeurs négatives telles quelles
APPLY_LOG = False  # True : applique une compression log10 (signée si CLIP_NEGATIVE=False, pour rester définie sur les négatifs) ; False : pas de transformation logarithmique
MIN_SIZE = 2  # nombre minimum de points entre deux ruptures PELT
LINEAR_OUTLIER_THRESHOLD = 100  # un point est marqué aberrant si son résidu au modèle linéaire dépasse LINEAR_OUTLIER_THRESHOLD x la MAD des résidus
MIN_SEGMENT_DURATION_HOURS = 24  # durée minimale (en heures) d'un segment pour être conservé par segmentize
PAIRS_SOURCE = 'manual_pairs'  # fonction utilisée par main() pour récupérer les paires (device_id, sensor_id) : 'sample_1', 'sample_1bis', 'sample_2' ou 'manual_pairs'

# OUTPUT_FILE = 'payload_anomaly_client_IQ_2026.csv'
OUTPUT_FILE = 'sample_1bis_anomalies_sup_1000_2025.csv'
START = '2025-01-01 00:00:00'
END = '2025-12-31 00:00:00'

def sample_2():
    """ Tous les équipements/PDM de la famille electrique. appartenant aux clients IQ."""
    anomalies_path = os.path.join(os.path.dirname(__file__), '..', 'sous_comptage/data', 'clients_iQspot_sous_comptage.csv')
    anomalies_df = pd.read_csv(anomalies_path)
    entities = anomalies_df['entity'].unique()
    client = ClickHouseClient()
    pairs = []
    for entity in entities:
        query = f"""SELECT distinct device_id, sensor_id
            FROM datamart_sous_comptage.distributed_dataset_hourly_payloads_merged
            WHERE entity = '{entity}'
            and device_type like '%lectr%';"""
        result_df = client.query(query)
        if result_df.empty:
            continue
        pairs.extend(result_df[['device_id', 'sensor_id']].itertuples(index=False, name=None))
    return pairs

def sample_1():
    """Les anomalies détectées dans le sous comptage à l'aide de la requête SQL qui teste l'IQR des valeurs. 
            Uniquement les anomalies INFERIEURES à 1000. Normalement celles où il y a le PLUS de faux positifs"""
    anomalies_path = os.path.join(os.path.dirname(__file__), '..', 'sous_comptage', 'anomalies_all.csv')
    anomalies_df = pd.read_csv(anomalies_path)
    anomalies_df = anomalies_df[anomalies_df['eval_prev_hour'] < 1000]
    anomalies_df = anomalies_df.drop_duplicates(subset='site_id')
    return list(anomalies_df[['device_id', 'sensor_id']]\
                 .itertuples(index=False, name=None))

def sample_1bis(n_sites=40):
    """Les anomalies détectées dans le sous comptage à l'aide de la requête SQL qui teste l'IQR des valeurs.
        Uniquement les anomalies SUPERIEURES à 1000. Normalement celles où il y a le MOINS de faux positifs.
        Tire au hasard n_sites sites et renvoie les device_id/sensor_id de ces sites."""
    anomalies_path = os.path.join(os.path.dirname(__file__), '..', 'sous_comptage', 'anomalies_all.csv')
    anomalies_df = pd.read_csv(anomalies_path)
    anomalies_df = anomalies_df[anomalies_df['eval_prev_hour'] > 1000]
    site_ids = anomalies_df['site_id'].unique()
    selected_sites = pd.Series(site_ids).sample(n=min(n_sites, len(site_ids))).tolist()
    anomalies_df = anomalies_df[anomalies_df['site_id'].isin(selected_sites)]
    return list(anomalies_df[['device_id', 'sensor_id']].drop_duplicates()\
                 .itertuples(index=False, name=None))

def manual_pairs():
    return [('604270ca83c2f34ea4220113', '59c3be73bee9f849f51461d3'),
            ('604270ca83c2f34ea4220111', '59c3be73bee9f849f51461d3'), 
            ('69c511321a57edebda8c1b6e', '662bc1c93f0afb78ad05983d'), 
            ('69c5589f2850867710cf3db6', '662bc1c93f0afb78ad05983d'), 
            ('69c5589f2850867710cf3dda', '662bc1c93f0afb78ad05983d'), 
            ('69c5589f2850867710cf3dfe', '662bc1c93f0afb78ad05983d'), 
            ('69c558e92850867710cf6b3a', '662bc1c93f0afb78ad05983d'), 
            ('69c55d0f2850867710d02716', '662bc1c93f0afb78ad05983d'), 
            ('69c55d112850867710d02858', '662bc1c93f0afb78ad05983d'), 
            ('69ca9c980630341bd0ecb2f6', '662bc1c93f0afb78ad05983d'), 
            ('69ca9c980630341bd0ecb303', '662bc1c93f0afb78ad05983d'), 
            ('693aa02378252e8bfd4f872d', '662bc1c93f0afb78ad05983d'), 
            ('693aa03578252e8bfd4f8caa', '662bc1c93f0afb78ad05983d'), 
            ('693aa03d78252e8bfd4f8f52', '662bc1c93f0afb78ad05983d')]

PAIRS_SOURCES = {
    'sample_1': sample_1,
    'sample_1bis': sample_1bis,
    'sample_2': sample_2,
    'manual_pairs': manual_pairs,
}
def null_ranges(df, start_list):
    """
    Identify null ranges and valid segments between them.

    Args:
        df: DataFrame with 'ts' (timestamp in seconds)
        start_list: List of timestamps marking start of null periods

    Returns:
        null_ranges: List of tuples (start_date, end_date) for null periods
        valid_segments: List of tuples (preceding_ts, following_ts) for valid segments
    """
    null_ranges_list = []
    valid_segments_list = []
    start_list_sorted = sorted(start_list)

    if start_list_sorted[0] > df.ts.min():
        # Add segment before first null_range: from first reading to last reading before first null_start
        first_before_null = df[df['ts'] < start_list_sorted[0]]['ts'].max()
        valid_segments_list.append((df['ts'].min(), first_before_null))

    # Process each null_range
    i = 0
    while i+1 <= len(start_list_sorted):
        null_start = start_list_sorted[i]
        null_end = df[(df['ts'] > null_start) & (df['raw_value'] == 0)]['ts'].max()
        null_ranges_list.append((null_start, null_end))
        i+=1

    if start_list_sorted[-1] < df.ts.max():
        # Add segment after last null_range
        first_after_null = df[df['ts'] > null_end]['ts'].min()
        valid_segments_list.append((first_after_null, df['ts'].max()))
    return null_ranges_list, valid_segments_list
        

def query_serie(dev_id, sens_id, start, end):
    dataset = 'datawarehouse_prod.fact_payload'
    query = f"""SELECT
        device_id,
        payload_metric_id AS sensor_id,
        toUnixTimestamp(fp.payload_timestamp) AS ts,
        payload_timestamp,
        toFloat32OrNull(fp.payload_value) AS raw_value
    FROM {dataset} AS fp
    WHERE fp.device_id = '{dev_id}'
    AND fp.payload_metric_id = '{sens_id}'
    AND fp.payload_timestamp BETWEEN '{start}' AND '{end}'
    ORDER BY fp.payload_timestamp;"""
    return query

def segmentize(df, min_duration_hours=MIN_SEGMENT_DURATION_HOURS):
    segment = df.copy()
    reset_ts_list, null_range_list = reset_compteurs(segment)
    is_reset = segment['ts'].isin(reset_ts_list)
    segment['time_delta_seconds'] = segment['ts'].diff()
    T = segment['time_delta_seconds'].median()
    is_cut = (segment['time_delta_seconds'] > T+600) & (segment['time_delta_seconds'] % T == 0)    
    segment['segment_id'] = (is_cut | is_reset).cumsum()

    segment_duration = segment.groupby('segment_id')['ts'].transform(lambda s: s.max() - s.min())
    segment = segment[segment_duration >= min_duration_hours * 3600]
    return segment, null_range_list

def rupture_detection(segment, pen=PEN, min_ratio=MIN_RATIO, clip_negative=CLIP_NEGATIVE, 
                      apply_log=APPLY_LOG, min_size=MIN_SIZE):
    is_changepoint = pd.Series(False, index=segment.index)
    smoothed = segment['raw_value']    
    valid_index = segment.index[smoothed.notna()]
    raw_signal = smoothed.loc[valid_index].to_numpy()   

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
            if after_mean >= min_ratio * max(before_mean, 1e-9):
                kept.append(mid - 1)
    if kept:
        is_changepoint.loc[valid_index[kept]] = True
    return is_changepoint

def reset_compteurs(df):
    is_reset = df['raw_value'].diff() < 0
    reset_with_zero = is_reset & (df['raw_value'] == 0)

    null_range_list = []
    flat = []

    if reset_with_zero.any():
        start_list = df.loc[reset_with_zero, 'ts'].tolist()
        null_range_list, valid_segment = null_ranges(df, start_list)
        flat = sorted([t for pair in null_range_list for t in pair])
    return flat, null_range_list

def commissioning(df):
    valid = df.dropna(subset=['raw_value'])
    valid = valid[valid['raw_value'] != 0]
    if valid.empty:
        return pd.DataFrame(columns=['device_id', 'sensor_id', 'ts', 'raw_value', 'type', 'date', 'segment_start', 'segment_end'])

    first = valid.iloc[0]
    date = pd.to_datetime(first['ts'], unit='s', utc=True).tz_convert('Europe/Paris')
    return pd.DataFrame([{
        'device_id': first['device_id'],
        'sensor_id': first['sensor_id'],
        'ts': first['ts'],
        'raw_value': first['raw_value'],
        'type': 'commissioning',
        'date': date,
        'segment_start': date,
        'segment_end': date,
    }])

def multi_frequence(df, start, end):
    periodes = detection(df)
    if len(periodes) > 1 and periodes.index.max() - periodes.index.min() > 10:
        start_ts = pd.Timestamp(start).tz_localize('Europe/Paris')
        end_ts = pd.Timestamp(end).tz_localize('Europe/Paris')
        return pd.DataFrame([{
            'device_id': df['device_id'].iloc[0],
            'sensor_id': df['sensor_id'].iloc[0],
            'ts': np.nan,
            'raw_value': np.nan,
            'type': 'multi-freq',
            'date': start_ts,
            'segment_start': start_ts,
            'segment_end': end_ts,
        }])
    return pd.DataFrame(columns=['device_id', 'sensor_id', 'ts', 'raw_value', 'type', 'date', 'segment_start', 'segment_end'])

def detect_linear_outliers(df, threshold=LINEAR_OUTLIER_THRESHOLD):
    # ajuste une regression lineaire sur toute la serie et flag les points dont le residu s'ecarte trop du modele
    is_outlier = pd.Series(False, index=df.index)
    smoothed = df['raw_value']
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


def main():
    pairs = PAIRS_SOURCES[PAIRS_SOURCE]()

    columns = ['device_id', 'sensor_id', 'ts', 'raw_value', 'type', 'date', 'segment_start', 'segment_end']
    SAVE_EVERY = 10

    def build_verite_terrain(frames, null_range_list):
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        df = df.reindex(columns=columns)
        df['date'] = df['date'].astype(object)
        missing_date = df['date'].isna()
        df.loc[missing_date, 'date'] = pd.to_datetime(df.loc[missing_date, 'ts'], \
                                                       unit='s', utc=True).dt.tz_convert('Europe/Paris')
        null_rows = []
        for dev_id, sens_id, start, end in null_range_list:
            null_rows.append({
                'device_id': dev_id,
                'sensor_id': sens_id,
                'ts': np.nan,
                'raw_value': np.nan,
                'type': 'null_index',
                'date': None,
                'segment_start': pd.to_datetime(start, unit='s', utc=True).tz_convert('Europe/Paris'),
                'segment_end': pd.to_datetime(end, unit='s', utc=True).tz_convert('Europe/Paris'),
            })
        df = pd.concat([df, pd.DataFrame(null_rows)], ignore_index=True)
        
        return df

    client = ClickHouseClient()
    anomalies_found = []
    all_null_ranges = []

    for _, (dev_id, sens_id) in enumerate(tqdm(pairs), start=1):
        df = client.query(query_serie(dev_id, sens_id, START, END))
        if df.empty:
            continue
        df['payload_timestamp'] = pd.to_datetime(df['ts'], unit='s', utc=True).dt.tz_convert('Europe/Paris')
        commissioning_result = commissioning(df)
        anomalies_found.append(commissioning_result)
        if not commissioning_result.empty:
            df = df[df['ts'] >= commissioning_result['ts'].iloc[0]]
        if df.empty:
            continue
        multi_freq_result = multi_frequence(df, START, END)
        if not multi_freq_result.empty:
            anomalies_found.append(multi_freq_result)
            continue
        segment, null_range_list = segmentize(df)
        all_null_ranges.extend((dev_id, sens_id, start, end) for start, end in null_range_list)
        for _, segment_df in segment.groupby('segment_id'):
            # is_outlier = detect_linear_outliers(segment_df) # deprecated. Overkill avec la fonction de detection de rupture
            is_rupture = rupture_detection(segment_df)
            segment_start = pd.to_datetime(segment_df['ts'].min(), unit='s', utc=True).tz_convert('Europe/Paris')
            segment_end = pd.to_datetime(segment_df['ts'].max(), unit='s', utc=True).tz_convert('Europe/Paris')
            # anomalies_found.append(segment_df.loc[is_outlier, ['device_id', 'sensor_id', 'ts', 'raw_value']]\
            #                        .assign(type='outlier', segment_start=segment_start, segment_end=segment_end))
            anomalies_found.append(segment_df.loc[is_rupture, ['device_id', 'sensor_id', 'ts', 'raw_value']]\
                                   .assign(type='rupture', segment_start=segment_start, segment_end=segment_end))

        # if i % SAVE_EVERY == 0:
        #     build_verite_terrain(anomalies_found).to_csv(OUTPUT_FILE, index=False)

    verite_terrain_df = build_verite_terrain(anomalies_found, all_null_ranges)
    # verite_terrain_df.to_csv(OUTPUT_FILE, index=False)
    print(verite_terrain_df)

if __name__ == "__main__":
    main()
