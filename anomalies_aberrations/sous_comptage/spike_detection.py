from anomaly_detector import Detector, DataProcessor
from db_connector import ClickHouseClient
from tqdm import tqdm
import pandas as pd

#%%# variables
input_folder = './data/sample_anomalies_2025_sem_1'
sous_comptage = 'datamart_sous_comptage.distributed_dataset_hourly_payloads_merged'
start = "2025-01-01 00:00:00"
end = "2025-12-31 00:00:00"
list_anom = pd.read_csv(f'{input_folder}/candidates_anom.csv')
# list_anom = list_anom[list_anom.spikes>0]
list_anom['month'] = pd.to_datetime(list_anom.payload_expected_time).dt.month
list_anom = list_anom[['customer_account_name', 'device_name', 'sensor_name', 'device_id', 'sensor_id', 'month']].drop_duplicates()
client = ClickHouseClient()
x_time = 'payload_expected_time'

experiments = [
    {"method": "iqr",  "k":  50},
    {"method": "iqr",  "k": 100},
    {"method": "iqr",  "k": 150},
    {"method": "isof", "th": -0.5},
    {"method": "isof", "th": -0.7},
    {"method": "isof", "th": -0.9},
    {"method": "stl",  "k": 3},
    {"method": "stl",  "k": 4},
    {"method": "stl",  "k": 5},
]

anomalies = []
for i, row in tqdm(list_anom.iterrows(), total=list_anom.shape[0]):
    q = f"""SELECT device_id, sensor_id, entity, {x_time}, eval_prev_hour, sum_payload_value_hour_curr,
                    device_name, sensor_name, customer_account_name, site_id
            FROM {sous_comptage}
            WHERE 1=1
                AND payload_expected_time between '{start}' and '{end}'
                AND customer_account_name = '{row.customer_account_name.replace("'", "''")}'
                AND device_name = '{row.device_name.replace("'", "''")}'
                AND sensor_name  = '{row.sensor_name.replace("'", "''")}'
                AND month = '{row.month}'
                ORDER BY payload_expected_time DESC"""
    smp = client.query(q)
    if smp.empty:
        continue
    else:
        smp = smp[smp.entity == smp.entity.value_counts().index[0]]  # dédupliquer la données par client
        smp = smp[~smp['sum_payload_value_hour_curr'].isna()]
        record = {
            "device_name": row.device_name,
            "sensor_name": row.sensor_name,
            "customer_account_name": row.customer_account_name,
        }
        checking = DataProcessor(record, smp)
        if 'index' in row.sensor_name.lower():
            y_column = 'eval_prev_hour'
        elif not checking.serie_typology('sum_payload_value_hour_curr', tol=0.3):
            y_column = 'sum_payload_value_hour_curr'
        else:
            y_column = 'eval_prev_hour'
        smp['med_max20'] = smp[y_column].sort_values().tail(20).mean()
        smp['value'] = smp[y_column]
        smp['feature'] = y_column
        detector = Detector(smp, y_column)
        for exp in experiments:
            try:
                if exp["method"] == "iqr":
                    col = f'iqr_k{exp["k"]}'
                    smp[col] = detector.iqr(k=exp["k"])
                elif exp["method"] == "isof":
                    col = f'isof_th{exp["th"]}'
                    smp[col] = detector.iso_forest(th=exp["th"])
                elif exp["method"] == "stl":
                    col = f'stl_k{exp["k"]}'
                    smp[col] = detector.stl(k=exp["k"]) 
            except Exception as e:
                record["error"] = e
        method_cols = [c for c in smp.columns if c.startswith(('iqr_', 'isof_', 'stl_'))]
        mask = smp[method_cols].any(axis=1)
        if not mask.any():
            continue
        anomalies.append(
            smp.loc[mask, ['customer_account_name', 'site_id', 'device_id', 'sensor_id', 'device_name', 'sensor_name',
                           'payload_expected_time', 'feature', 'value', 'med_max20'] + method_cols])

if anomalies:
    pd.concat(anomalies).to_csv(f'{input_folder}/method_comparison.csv')
