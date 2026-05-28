from anomaly_detector import Detector #, DataProcessor
from db_connector import ClickHouseClient
from tqdm import tqdm
import pandas as pd

#%%# variables
sous_comptage = 'datamart_sous_comptage.distributed_dataset_hourly_payloads_merged'
start = "2026-01-01 00:00:00"
end = "2026-06-01 00:00:00"
sensor_idx = '662bc1c93f0afb78ad05983d'# Index consommation électricité (kWh)
client_iq = pd.read_csv(f'./data/clients_iQspot_sous_comptage.csv')
client = ClickHouseClient()
x_time = 'payload_expected_time'
y_column = 'eval_prev_hour' # iq spot data : only index
window_weeks = 8

#liste des devices et sensors pour les clients candidats:
q = f"""SELECT distinct device_id, entity, site_id
            FROM {sous_comptage}
            WHERE 1=1
                AND payload_expected_time between '{start}' and '{end}'
                AND entity IN {client_iq.entity.to_list()}
                AND lower(device_type) like '%lect%'
                """
candidates = client.query(q)
anomalies = []
for i, row in tqdm(candidates.iterrows(), total=candidates.shape[0]):
    q = f"""SELECT device_id, sensor_id, entity, {x_time}, {y_column},
                    device_name, sensor_name, customer_account_name, site_id
            FROM {sous_comptage}
            WHERE 1=1
                AND payload_expected_time between '{start}' and '{end}'
                AND entity = '{row.entity}'
                AND device_id = '{row.device_id}'
                AND sensor_id  = '{sensor_idx}'
                ORDER BY payload_expected_time ASC"""
    smp = client.query(q)
    if smp.empty:
        continue
    if smp[y_column].max() < 1000:
        continue
    smp = smp.copy()
    smp[x_time] = pd.to_datetime(smp[x_time])
    smp[y_column] = smp[y_column].interpolate()
    delta   = pd.Timedelta(weeks=window_weeks)
    current = smp[x_time].min()
    end_ts  = smp[x_time].max()

    while current < end_ts:
        mask_w = (smp[x_time] >= current) & (smp[x_time] < current + delta)
        w = smp.loc[mask_w].copy()
        current += delta
        if w.empty:
            continue
        try:
            w['isof'] = Detector('iso_forest', {'th': -0.7}).predict(w.copy(), y_column)['anomaly'].values
            w['stl']  = Detector('stl',        {'k': 4}  ).predict(w.copy(), y_column)['anomaly'].values
            w['iqr']  = Detector('iqr',        {'k': 1.5}  ).predict(w.copy(), y_column)['anomaly'].values
            w['value'] = w[y_column]
        except Exception as e:
            print(f"  ⚠ {row.device_id} [{current - delta} → {current}] : {e}")
            continue
        mask = w[['isof', 'stl', 'iqr']].any(axis=1)
        if mask.any():
            anomalies.append(
                w.loc[mask, ['customer_account_name', 'site_id', 'device_id', 'sensor_id',
                             'device_name', 'sensor_name', 'payload_expected_time', 'value', 'isof', 'stl']])
if anomalies:
    pd.concat(anomalies).to_csv(f'./data/iq_spot_detection_stl_isof_2026.csv')
