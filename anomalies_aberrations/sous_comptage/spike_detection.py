from dotenv import load_dotenv
import urllib.parse
import os
import clickhouse_connect
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#%%# variables
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
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    if iqr <1:
        iqr = 10
    cutoff = iqr * k
    thres = q75 + cutoff
    num_outliers = np.sum(data > thres)
    return num_outliers

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

def anom_loop_device(sens, dev, c, m, s):
    r_dict = {'device_name':dev, 'sensor_name':sens, 'customer_account_name':c, 'month':m}
    if s.sum() == 0:
        r_dict['error'] = "Null values"
        return r_dict
    if check_continuous(s):
        r_dict['error'] = 'continuous_values'
    if s.isna().mean() > 0.1:
        r_dict['missing'] = 'more_than_10%'
        if s.isna().mean() > 0.8:
            r_dict['error'] = 'more_than_80%'
            return r_dict
    try :
        r_dict['spikes'] = detect_outliers_iqr(s.dropna(), k=100)
    except:
        r_dict['error'] = "run_error"
    return r_dict
#%% liste des capteurs electricité
sensors = pd.read_csv('./data/liste_sensor_electric_devices.csv', sep=';')
sensors_idx = sensors[(sensors.enedis == 0) & (sensors.smart_impulse==0) & (sensors.Considere==1)]
#%% liste des clients
start = "2025-01-01 00:00:00"
# start = "2025-06-30 23:59:00"
end = "2025-06-30 00:00:00"
# list_anom = pd.read_csv('./data/spike_list_clk_query.csv')
customers = client.query_df(f"""
        SELECT distinct customer_account_name
        FROM {sous_comptage}
        WHERE 1=1
         AND payload_expected_time between '{start}' and '{end}'
         AND eval_prev_hour IS NOT NULL
         AND sensor_id IN {sensors_idx.sensor_id.to_list()}
         LIMIT 100""")
list_cust = customers.customer_account_name.to_list()
#%%
check_column = 'eval_prev_hour'
# check_column = 'sum_payload_value_hour_curr'
# custom = 'ISAE - SUPAERO'
custom = 'ENAC'
custom_anom = []
for custom in list_cust[:]:
    equip_list = client.query_df(f"""
            SELECT distinct device_id, device_name, sensor_id, sensor_name, month
            FROM {sous_comptage}
            WHERE 1=1
                AND eval_prev_hour IS NOT NULL
                AND customer_account_name = '{custom.replace("'", "''")}'
                AND sensor_id IN {sensors_idx.sensor_id.to_list()}
                AND payload_expected_time between '{start}' and '{end}'""")
    anomaly = []
    print(custom)
    for row in tqdm(equip_list[:100].itertuples(index=False)):
        smp = client.query_df(f"""
                    SELECT device_id, payload_expected_time, entity, eval_prev_hour
                    FROM {sous_comptage}
                    WHERE 1=1
                        AND payload_expected_time between '{start}' and '{end}'
                        AND month = '{row.month}'
                        AND sensor_id = '{row.sensor_id}'
                        AND device_id = '{row.device_id}'
                        AND eval_prev_hour > 0
                    ORDER BY payload_expected_time
                    LIMIT 100000""")
        if smp.empty:
            continue
        else:
            smp = smp[smp.entity == smp.entity.values[0]]
            records = anom_loop_device(row.sensor_name, row.device_name, custom, row.month, smp[check_column])
            anomaly.append(records)
    anomaly = pd.DataFrame(anomaly)
    custom_anom.append(anomaly)
custom_anom = pd.concat(custom_anom)