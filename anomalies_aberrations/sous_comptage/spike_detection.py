from anomaly_detector import ClickHouseClient, Detector, DataProcessor
from tqdm import tqdm
import pandas as pd

#%%# variables
sous_comptage = 'datamart_sous_comptage.distributed_dataset_hourly_payloads_merged'
start = "2025-01-01 00:00:00"
end = "2025-06-30 00:00:00"
list_anom = pd.read_csv('./data/anomalies_2025_sem_1.csv')
list_anom = pd.read_csv('./data/spike_list_query')
list_anom = list_anom[list_anom.spikes>0]
client = ClickHouseClient()
x_time = 'payload_expected_time'

algo_run = []
anomalies = []
for i, row in list_anom.iterrows():
    q = f"""SELECT device_id, sensor_id, entity, {x_time}, eval_prev_hour, sum_payload_value_hour_curr,
                    device_name, sensor_name, customer_account_name, site_id
            FROM {sous_comptage}
            WHERE 1=1
                AND payload_expected_time between '{start}' and '{end}'
                AND customer_account_name = '{row.customer_account_name.replace("'", "''")}'
                AND device_name = '{row.device_name}'                
                AND sensor_name  = '{row.sensor_name}'
                AND month = '{row.month}'
                ORDER BY payload_expected_time DESC"""
    smp = client.query(q)
    if smp.empty:
        continue
    else:
        smp = smp[smp.entity == smp.entity.value_counts().index[0]]  # dédupliquer la données par client
        smp = smp[~smp[y_column].isna()]
        record = {
            "device_name": row.device_name,
            "sensor_name": row.sensor_name,
            "customer_account_name": row.customer_account_name,
        }
        # check incremental data
        checking = DataProcessor(record, smp)
        if 'index' in row.sensor_name.lower():        
            y_column = 'eval_prev_hour'
        elif not checking.serie_typology('sum_payload_value_hour_curr', tol=0.3):
            y_column = 'sum_payload_value_hour_curr'
        else : 
            y_column = 'eval_prev_hour'        
        try:
            detector = Detector(smp, y_column)
            smp['iqr'] = detector.iqr()
            smp['isof'] = detector.iso_forest(th=1.5)
            smp['stl'] = detector.stl(k=4)
            smp['med_max20'] = smp[y_column].sort_values().tail(20).mean()
            smp['value'] = smp[y_column]
            smp['feature'] = y_column
        except Exception as e:
            record["error"] = e
        record['IQR_anomaly'] = smp.iqr.sum() == 0
        record['Isol_forest'] = smp.isof.sum() == 0
        record['feature'] = y_column
        anomalies.append(
            smp.loc[(smp.iqr==True)|(smp.isof==True),
            ['customer_account_name', 'site_id', 'device_id', 'sensor_id', 'device_name', 'sensor_name',
             'payload_expected_time', 'feature', 'value', 'iqr', 'isof']])
        algo_run.append(record)

algo_run = pd.DataFrame(algo_run)
anomalies = pd.concat(anomalies)
algo_run.to_csv('./data/synth_comparison_2025_sem_1.csv')
anomalies.to_csv('./data/anomalies_comparison_2025_sem_1.csv')