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
CLIP_NEGATIVE = False  # True : ramène les valeurs négatives à 0 avant transformation (comportement historique) ; False : conserve les valeurs négatives telles quelles
APPLY_LOG = False  # True : applique une compression log10 (signée si CLIP_NEGATIVE=False, pour rester définie sur les négatifs) ; False : pas de transformation logarithmique
MIN_SIZE = 2  # nombre minimum de points entre deux ruptures PELT
LINEAR_OUTLIER_THRESHOLD = 100  # un point est marqué aberrant si son résidu au modèle linéaire dépasse LINEAR_OUTLIER_THRESHOLD x la MAD des résidus
MIN_SEGMENT_DURATION_HOURS = 2  # durée minimale (en heures) d'un segment pour être conservé par segmentize
PAIRS_SOURCE = 'manual_sites'  # fonction utilisée par main() pour récupérer les paires (device_id, sensor_id) : 'sample_1', 'sample_1bis', 'sample_2' ou 'manual_pairs'

# OUTPUT_FILE = 'payload_anomaly_client_IQ_2026.csv'
OUTPUT_FILE = 'sample_1bis_anomalies_sup_1000_2025.csv'
START = '2026-01-27 00:00:00'
END = '2026-09-15 23:59:59'

def sample_3():
    """ Tous les équipements/PDM de la famille electrique et capteur de type index sauf enedis (4700 samples)."""
    pairs_path = os.path.join(os.path.dirname(__file__), '.', 'payloads', 'device_sensor_index_pairs.csv')
    pairs_df = pd.read_csv(pairs_path)
    return pairs_df

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
    return [('69c558e92850867710cf6b3a', '662bc1c93f0afb78ad05983d') ,
            ('6042714d83c2f34ea422022f', '59c3be73bee9f849f51461d3'),
            ('604270ca83c2f34ea4220113', '59c3be73bee9f849f51461d3'),
            ('604270ca83c2f34ea4220111', '59c3be73bee9f849f51461d3'),
            ('69c511321a57edebda8c1b6e', '662bc1c93f0afb78ad05983d'), 
            ('69c5589f2850867710cf3db6', '662bc1c93f0afb78ad05983d'), 
            ('69c5589f2850867710cf3dda', '662bc1c93f0afb78ad05983d'), 
            ('69c5589f2850867710cf3dfe', '662bc1c93f0afb78ad05983d'), 
            ('69c55d0f2850867710d02716', '662bc1c93f0afb78ad05983d'), 
            ('69c55d112850867710d02858', '662bc1c93f0afb78ad05983d'), 
            ('69ca9c980630341bd0ecb2f6', '662bc1c93f0afb78ad05983d'), 
            ('69ca9c980630341bd0ecb303', '662bc1c93f0afb78ad05983d'), 
            ('693aa02378252e8bfd4f872d', '662bc1c93f0afb78ad05983d'), 
            ('693aa03578252e8bfd4f8caa', '662bc1c93f0afb78ad05983d'), 
            ('693aa03d78252e8bfd4f8f52', '662bc1c93f0afb78ad05983d')]

def manual_sites():
    return ['69c545822850867710ce9808', '6037dce8f435564fd2458b91',
            '69c505b51a57edebda8bff3c', '69c545802850867710ce9673',   
            '69c5426c1a57edebda8c6a7f', '69ca8e560630341bd0ec52c0']

PAIRS_SOURCES = {
    'sample_1': sample_1,
    'sample_1bis': sample_1bis,
    'sample_2': sample_2,
    'manual_pairs': manual_pairs,
    'sample_3' : sample_3}

def null_ranges(df):
    return df["raw_value"].eq(0)

def get_site_pairs(site_id):
    query = f"""SELECT DISTINCT device_id, sensor_id
    FROM datamart_sous_comptage.device_properties
    WHERE site_id = '{site_id}'
    AND lower(device_type) LIKE '%lectr%'
    AND lower(sensor_name) LIKE '%index%'
    AND lower(sensor_name) NOT LIKE '%(cons)%'"""
    return query

def query_by_site(pairs_df, start, end):
    device_ids = "','".join(pairs_df['device_id'].unique())
    sensor_ids = "','".join(pairs_df['sensor_id'].unique())
    query = f"""SELECT device_id, payload_metric_id AS sensor_id, payload_timestamp,
        toFloat32OrNull(payload_value) AS raw_value
    FROM datawarehouse_prod.fact_payload
    WHERE payload_metric_id IN ('{sensor_ids}')
    AND device_id IN ('{device_ids}')
    AND payload_timestamp BETWEEN '{start}' AND '{end}'
    ORDER BY payload_timestamp;"""
    return query

def query_by_dev(dev_id, sens_id, start, end):
    dataset = 'datawarehouse_prod.fact_payload'
    query = f"""SELECT
        device_id,
        payload_metric_id AS sensor_id,
        payload_timestamp,
        toFloat32OrNull(fp.payload_value) AS raw_value
    FROM {dataset} AS fp
    WHERE fp.device_id = '{dev_id}'
    AND fp.payload_metric_id = '{sens_id}'
    AND fp.payload_timestamp BETWEEN '{start}' AND '{end}'
    ORDER BY fp.payload_timestamp;"""    
    return query

def segmentize(df, reset_ts_list):
    segment = df.copy()
    is_reset = segment['payload_timestamp'].isin(reset_ts_list)
    segment['segment_id'] = (is_reset).cumsum()
    return segment

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

    algo = rpt.Pelt(model='l2', min_size=min_size, jump=3).fit(signal)
    bkps = algo.predict(pen=pen)
    bounds = [0] + bkps

    if min_ratio is None:
        # pas de filtre : on garde toutes les ruptures trouvées par PELT
        kept = [mid for start, mid, end in zip(bounds, bounds[1:], bounds[2:])]
    else:
        # ne garde que les ruptures où le segment suivant est nettement PLUS HAUT ou PLUS BAS que le précédent
        kept = []
        for start, mid, end in zip(bounds, bounds[1:], bounds[2:]):
            before_mean = raw_signal[start:mid].mean()
            after_mean = raw_signal[mid:end].mean()
            is_up = after_mean >= min_ratio * max(before_mean, 1e-9)
            if is_up :
                kept.append(mid)
    if kept:
        is_changepoint.loc[valid_index[kept]] = True
    return is_changepoint

def reset_compteurs(df):
    is_reset = df['raw_value'].diff() < 0
    return is_reset

def commissioning(df):
    valid = df.dropna(subset=['raw_value'])
    valid = valid[valid['raw_value'] != 0]
    if valid.empty:
        return pd.DataFrame(columns=['device_id', 'sensor_id', 'ts', 'raw_value', 'type', 'date', 'segment_start', 'segment_end'])

    first = valid.iloc[0]
    date = pd.to_datetime(first['ts'], unit='s')
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

def find_anomalies(sample):
    anom_df = []
    null_val = null_ranges(sample)
    anom_df.append(sample.loc[null_val, ['device_id', 'sensor_id', 'payload_timestamp', 'raw_value']]\
                                    .assign(type='is_null', segment_start=START, segment_end=END))
    sample = sample.loc[~null_val]
    reset_cpt = reset_compteurs(sample)
    anom_df.append(sample.loc[reset_cpt, ['device_id', 'sensor_id', 'payload_timestamp', 'raw_value']]\
                                    .assign(type='reset', segment_start=START, segment_end=END))
    segment = segmentize(sample, sample.loc[reset_cpt, 'payload_timestamp'].tolist())
    for _, segment_df in segment.groupby('segment_id'):
        is_rupture = rupture_detection(segment_df)
        segment_start = pd.to_datetime(segment_df['payload_timestamp'].min(), unit='s')
        segment_end = pd.to_datetime(segment_df['payload_timestamp'].max(), unit='s')
        anom_df.append(segment_df.loc[is_rupture, ['device_id', 'sensor_id', 'payload_timestamp', 'raw_value']]\
                            .assign(type='rupture', segment_start=segment_start, segment_end=segment_end))
    return anom_df     

def main():
    site_list = manual_sites()
    columns = ['device_id', 'sensor_id', 'payload_timestamp', 'raw_value', 'type', 'segment_start', 'segment_end']
    client = ClickHouseClient()
    for site_id in tqdm(site_list):
        anom_res = []
        pairs = client.query(get_site_pairs(site_id))   
        if pairs.empty : 
            continue
        sample = client.query(query_by_site(pairs, START, END))
        if sample.empty:
            continue
        for (device_id, sensor_id), df in sample.groupby(['device_id', 'sensor_id']):        
            anom_res.extend(find_anomalies(df))
        results = pd.concat(anom_res, ignore_index=True) if anom_res else pd.DataFrame(columns=columns)
        results = results.reindex(columns=columns)
        # results.to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE), index=False)
        print(results[results.type!='is_null'])

if __name__ == "__main__":
    main()
