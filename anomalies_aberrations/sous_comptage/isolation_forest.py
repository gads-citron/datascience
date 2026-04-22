# isolation forest
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import urllib.parse
import os
import clickhouse_connect
from sklearn.ensemble import IsolationForest
import seaborn as sns
from tqdm import tqdm

isof = IsolationForest(n_estimators=500)
#%%# variables
sous_comptage = 'datamart_sous_comptage.distributed_dataset_hourly_payloads_merged'
# %% clickhouse env load
load_dotenv()
host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
client = clickhouse_connect.get_client(host=host, username=username,
                                       port=443, password=password)

def plot_conso(df, dev, conso):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.payload_expected_time, y=df[conso],
                             mode="lines", line={'color': col.orange, 'width': 1}))
    fig.update_layout(title=f'device_id: {dev}',
                      xaxis_title="Time", yaxis_title=conso, template="simple_white")
    fig.update_xaxes(ticks='outside', dtick=str(24*60*60*1000),
                     tickangle=-45, tickformat='%a-%b-%e')
    fig.show()

def check_continuous(s):
    if s.value_counts().shape[0] <10:
        return 1
    else: return 0

def serie_typology(df, tol=0.1):
    score = np.mean((df.shift() - df) < 0) # combien de fois la valeur décrémente
    if score > tol:
        return 0 # volume
    else:
        return 1 # index

def isolation_forest(smp, y_col):
    isof = IsolationForest(n_estimators=500)
    smp['hour'] = smp.payload_expected_time.dt.hour
    smp['day_week'] = smp.payload_expected_time.dt.day_of_week
    smp['hour_week'] = (smp.day_week * 24 + smp.hour).astype('float32')

    isof.fit(smp[['hour_week', y_col]])
    scores = isof.score_samples(smp[['hour_week', y_col]])
    # sns.lineplot(x = np.arange(1, scores.shape[0]+1), y = np.sort(scores))

    isof.offset_ = np.percentile(scores, 2)
    isof.decision_function(smp[['hour_week', y_col]])
    smp['month'] = row.month
    smp['isolation_forest'] = isof.predict(smp[['hour_week', y_col]])

#%%
iqr_anom = df = pd.read_csv('./data/anomalies_2025_sem_1.csv')
iqr_anom = iqr_anom[iqr_anom.spikes >0]

start = "2025-01-01 00:00:00"
end = "2025-06-31 00:00:00"
iso_anom = []
for i, row in tqdm(iqr_anom.iterrows()):
    y_column = 'sum_payload_value_hour_curr'
    smp = client.query_df(f"""
                SELECT device_id, payload_expected_time, entity, eval_prev_hour, sum_payload_value_hour_curr
                FROM {sous_comptage}
                WHERE 1=1
                    AND payload_expected_time between '{start}' and '{end}'
                    AND month = '{row.month}'
                    AND sensor_name = '{row.sensor_name.replace("'", "''")}'
                    AND device_name = '{row.device_name.replace("'", "''")}'
                    AND customer_account_name = '{row.customer_account_name.replace("'", "''")}'
                    AND eval_prev_hour > 0
                ORDER BY payload_expected_time
                LIMIT 100000""")
    if row.incremental_data:
        y_column = 'eval_prev_hour'
    smp['hour'] = smp.payload_expected_time.dt.hour
    smp['day_week'] = smp.payload_expected_time.dt.day_of_week
    smp['hour_week'] = (smp.day_week*24+smp.hour).astype('float32')

    isof.fit(smp[['hour_week', y_column]])
    scores = isof.score_samples(smp[['hour_week', y_column]])
    # sns.lineplot(x = np.arange(1, scores.shape[0]+1), y = np.sort(scores))

    isof.offset_ = np.percentile(scores, 2)
    isof.decision_function(smp[['hour_week', y_column]])
    smp['month'] = row.month
    smp['iso_anomaly'] = isof.predict(smp[['hour_week', y_column]])
    smp['customer_account_name'] = row.customer_account_name
    smp['sensor_name'] = row.sensor_name
    smp['device_name'] = row.device_name
    iso_anom.append(smp[smp.iso_anomaly == -1][['device_id', 'payload_expected_time',
                                                y_column, 'customer_account_name', 'month',
                                                'sensor_name', 'device_name']])

iso_anom = pd.concat(iso_anom)
cnt_iso_anom = iso_anom.groupby(['customer_account_name', 'device_name', 'sensor_name', 'month'])['payload_expected_time'].count()
cnt_iso_anom = pd.DataFrame(cnt_iso_anom)
cnt_iso_anom.rename({'payload_expected_time':'isolation_forest'}, axis=1, inplace=True)
comparison = cnt_iso_anom.merge(iqr_anom[['customer_account_name', 'device_name', 'sensor_name', 'month', 'spikes']],
                                on = ['customer_account_name', 'device_name', 'sensor_name', 'month'])

# sns.scatterplot(smp[['hour_week', 'eval_prev_hour']], x = 'hour_week', y='eval_prev_hour', hue=y_iso, palette="tab10")



