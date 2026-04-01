from dotenv import load_dotenv
import urllib.parse
import os
import clickhouse_connect
from tqdm import tqdm
import plotly.graph_objects as go
import citron_colors as col
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#%%
# variables
sous_comptage = 'datamart_sous_comptage.distributed_dataset_hourly_payloads_merged'
history = "toDateTime('2025-01-01 01:00:00')"
# %% clickhouse env load
load_dotenv()
host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
client = clickhouse_connect.get_client(host=host, username=username,
                                       port=443, password=password)
#%%
def detect_outliers_iqr(data, k = 100):
    data.dropna(inplace=True)
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    if iqr <1:
        iqr = 10
    cutoff = iqr * k
    thres = q75 + cutoff
    return (data > thres, thres)

def plot_conso(df, dev, conso):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.payload_expected_time, y=df[conso],
                             mode="lines", line={'color': col.orange, 'width': 1}))
    fig.update_layout(title=f'device_id: {dev}',
                      xaxis_title="Time", yaxis_title=conso, template="simple_white")
    fig.update_xaxes(ticks='outside', dtick=str(24*60*60*1000),
                     tickangle=-45, tickformat='%a-%b-%e')
    fig.show()

def check_anomalies(s, col):
    if s[col].dropna().empty :
        return 2,2
    neg = 0
    if not s[s[col] <0].empty:
        neg = 1
    miss = s[col].isna().mean()
    anom = detect_outliers_iqr(s[col])[0].sum()
    card = s[col].value_counts().shape[0]
    return neg, anom, card, miss

def serie_typology(df, tol=0.1):
    score = np.mean((df.shift() - df) < 0)
    if score > tol:
        return 0 # volume
    else:
        return 1 # index
#%% liste des capteurs relevant
sensors = pd.read_csv('./data/liste_sensor_electric_devices.csv', sep=';')
smart_imp = sensors[sensors.smart_impulse==1].sensor_id.to_list()
# sensors_idx = sensors[(sensors.Index==1) & (sensors.enedis == 0)]
sensors_idx = sensors[(sensors.enedis == 0)]
#%% Anomalies où il y a changement d'index
start = "toDateTime('2025-01-01 01:00:00')"
end = "toDateTime('2025-12-31 01:00:00')"

list_anom = client.query_df(f"""select distinct device_name, device_id, payload_expected_time, sensor_name,
                customer_account_name, eval_prev_hour
    from {sous_comptage}
    where 1=1
--     and customer_account_name == 'ENAC'
    and sensor_id IN {sensors_idx.sensor_id.to_list()}
    AND payload_expected_time > {start}
    AND payload_expected_time < {end}
    and eval_prev_hour is null  --volume conso horaire
    and payload_value_start_of_hour is not null --premier relevé d'index reçu sur l'heure
    and device_type like '%lectr%'
    limit 2000;""")
#%%
check_column = 'eval_prev_hour'
volume = 'sum_payload_value_hour_curr'
anomaly = []
for i, row in tqdm(list_anom[:].iterrows()):
    r_dict = {}
    df = client.query_df(f"""
        SELECT device_id, payload_expected_time, customer_account_name,
        sensor_id, entity, {check_column}, {volume}
        FROM {sous_comptage}
        WHERE 1=1
            AND sensor_name = '{row.sensor_name}'
         AND device_id = '{row.device_id}'
         AND payload_expected_time > {start}
         AND payload_expected_time < {end}
        ORDER BY payload_expected_time DESC
        settings max_rows_to_read = 10000000000000;""")
    # filter over one patrimonial data only
    df = df[df.entity == df.entity.values[0]]
    if not serie_typology(df[volume]):
        continue
    if df[check_column].dropna().empty :
        r_dict['missing'] = df.shape[0]
        continue
    neg, anom, cardinality, missing = check_anomalies(df, check_column)
    r_dict['negative'] = neg
    r_dict['outlier'] = anom
    r_dict['cardinality'] = cardinality
    r_dict['missing'] = missing
    r_dict['device_id'] = row.device_id
    r_dict['client'] = df.customer_account_name.values[0]
    anomaly.append(r_dict)
anomaly = pd.DataFrame(anomaly)
anomaly = anomaly.merge(list_anom[['device_name', 'device_id', 'sensor_name',
       'customer_account_name']].drop_duplicates(), on='device_id')
#%% Liste des equipements où données négatives
neg_dev = ['6042734d83c2f34ea4220621', '6042734f83c2f34ea4220625', '6042735383c2f34ea422062d',
           '6042735683c2f34ea4220633', '61543499a665731c2f7a6158',
           '615434cda665731c2f7a61a4', '615434c5a665731c2f7a6198', '615434bba665731c2f7a6188']
# capteurs compteurs d'eau froide et compteur d'eau
scale_site = dataset_test[~dataset_test.device_id.isin(['6554a43520122b76f0570387', '6554f7e920122b2b22570439'])]\
    .set_index('payload_expected_time').eval_prev_hour_converted.resample("h").sum()
#%%
dev = '67bc832a05fcd3f6eae21508'
sens = 'Index Consommation Electrique'
df = client.query_df(f"""
    SELECT device_id, payload_expected_time, customer_account_name,
    sensor_id, entity, {check_column}, sum_payload_value_hour_curr
    FROM {sous_comptage}
    WHERE 1=1
        AND sensor_name = '{sens}'
     AND device_id = '{dev}'
     AND payload_expected_time > {history}
    ORDER BY payload_expected_time DESC
    settings max_rows_to_read = 10000000000000;""")