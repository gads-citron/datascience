from anomaly_detector import ClickHouseClient, Detector, DataProcessor
from tqdm import tqdm

#%%# variables
sous_comptage = 'datamart_sous_comptage.distributed_dataset_hourly_payloads_merged'
start = "2025-01-01 00:00:00"
end = "2025-06-31 00:00:00"

client = ClickHouseClient()

#%% liste des clients
q =f""" SELECT distinct customer_account_name
        FROM {sous_comptage}
        WHERE 1=1
            AND payload_expected_time between '{start}' and '{end}'
            AND eval_prev_hour IS NOT NULL
            AND device_type like '%lectr%'
            AND sensor_name not like '%(Cons)%'
            AND sensor_name not like '%gaz%'
            AND sensor_name not like '%\''eau%'
        LIMIT 100"""
customers = client.query(q)
list_cust = customers.customer_account_name.to_list()
#%%
algo_run = []
anomalies = []
for custom in list_cust[:10]:
    q = f"""
            SELECT distinct device_id, device_name, sensor_id, sensor_name, month
            FROM {sous_comptage}
            WHERE 1=1
                AND sum_payload_value_hour_curr IS NOT NULL
                AND customer_account_name = '{custom.replace("'", "''")}'
                AND device_type like '%lectr%'
                AND sensor_name not like '%(Cons)%'
                AND sensor_name not like '%gaz%'
                AND sensor_name not like '%\''eau%'
                AND payload_expected_time between '{start}' and '{end}'"""
    equip_list = client.query(q)
    for row in tqdm(equip_list[:100].itertuples(index=False)):
        q = f"""SELECT device_id, sensor_id, payload_expected_time, entity, eval_prev_hour, sum_payload_value_hour_curr
                FROM {sous_comptage}
                WHERE 1=1
                    AND payload_expected_time between '{start}' and '{end}'
                    AND month = '{row.month}'
                    AND sensor_id = '{row.sensor_id}'
                    AND device_id = '{row.device_id}'
                    AND customer_account_name = '{custom.replace("'", "''")}'
                    AND eval_prev_hour > 0
                ORDER BY payload_expected_time
                LIMIT 100000"""
        smp = client.query(q)
        if smp.empty:
            continue
        else:
            smp = smp[smp.entity == smp.entity.drop_duplicates()[0]]  # dédupliquer la données par client
            record = {
                "device_id": row.device_id,
                "device_name": row.device_name,
                "sensor_id": row.sensor_id,
                "sensor_name": row.sensor_name,
                "customer_account_name": custom,
            }
            checking = DataProcessor(record, smp)
            y_column = 'sum_payload_value_hour_curr'
            incremental = checking.serie_typology(y_column)
            if incremental:
                y_column = 'eval_prev_hour'
            try:
                detector = Detector(smp, y_column)
                smp['iqr'] = detector.iqr()
                smp['isof'] = detector.iso_forest()
            except Exception as e:
                record["error"] = e
            record['incremental_data'] = incremental
            if smp.iqr.sum() == 0:
                record['IQR_anomaly'] = False
            else:
                record['IQR_anomaly'] = True
                anomalies.append(
                    smp.loc[(smp.iqr==True)|(smp.isof==True),
                    ['device_id', 'payload_expected_time', 'sensor_id', y_column, 'iqr', 'isof']])
            algo_run.append(record)