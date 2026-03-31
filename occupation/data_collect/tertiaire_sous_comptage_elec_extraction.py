# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 11:06:06 2025
@author: Myriam Lopez
"""
from dotenv import load_dotenv
import urllib.parse
import os
import clickhouse_connect
import pandas as pd
from tqdm import tqdm
# %% clickhouse env load
load_dotenv()
host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
client = clickhouse_connect.get_client(host=host, username=username,
                                       port=443, password=password)
#%% Liste des site/client/device par type de client (collectivite/fonciere/tertiaire)
devices_elec = pd.read_csv('../data/device_sous_comptage_elec_raw.csv', sep=';')
devices_elec = devices_elec[~devices_elec.Typologie_client.isna()]
devices_elec = devices_elec.drop_duplicates()
site_info = devices_elec[['site_typology', 'site_nature', 'site_sub_nature', 'meter_type',
       'site_id', 'customer_account_name', 'Typologie_client']].drop_duplicates()
#%% Liste des capteurs dédié au comptage electricité
sensors = pd.read_csv('../data/liste_sensor_electric_devices.csv', sep=';')
sensors = sensors[sensors.Considere==1]
sensors_vol = sensors[sensors.Index==0]
#%% filtrer les équipements qui envoient de la données sur les capteurs qui nous intéressent
devices_elec = client.query_df(f"""
    SELECT distinct site_id, device_id
    FROM {p.datasource}
    WHERE 1=1
     AND match (device_type, '{p.dev_type}')
     AND sensor_id IN {sensors_vol.sensor_id.drop_duplicates().to_list()}
    settings max_rows_to_read = 10000000000000;""")
#%% merge devices avec infos du site
devices_elec = devices_elec.merge(site_info[~site_info.meter_type.isna()])
devices_elec.to_csv('../data/sous_comptage_elec_tertiaire.csv', index=False)
#%% test sur un site du segment Tertiaire en sous-comptage :
history = "toDateTime('2026-01-01 01:00:00')"
site_id = '654ad02bd5e238fd772e39d4'
dataset_test = []
for dev in tqdm(devices_elec[devices_elec.site_id=='654ad02bd5e238fd772e39d4'].device_id.to_list()):
    df = client.query_df(f"""
        SELECT device_id, payload_expected_time, sum_payload_value_hour_curr, eval_prev_hour_converted
        FROM {p.datasource}
        WHERE 1=1
        AND payload_expected_time > {history} 
        AND device_id = '{dev}'
        ORDER BY payload_expected_time DESC
        settings max_rows_to_read = 10000000000000;""")
    dataset_test.append(df)

dataset_test = pd.concat(dataset_test)
dataset_test['hour'] = dataset_test.payload_expected_time.dt.hour
dataset_test['day'] = dataset_test.payload_expected_time.dt.day_of_week
dataset_test['week'] = dataset_test.payload_expected_time.dt.isocalendar().week
profiles = dataset_test.groupby(['device_id', 'day', 'hour']).eval_prev_hour_converted.mean().reset_index()

to_delete = ['6554a43520122b76f0570387', '6554f7e920122b2b22570439'] # dev aux valeurs négatives
#%% matcher les données au profile
profiles.columns = ['device_id', 'day', 'hour', 'baseline']
dataset_test = dataset_test.merge(profiles, on=["device_id", "day", "hour"], how="left")
#%%
for i, df in dataset_test[dataset_test.device_id == '6554adfd20122b1dc9570392'].groupby('week'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(name = 'real', x=df.payload_expected_time, y=df.eval_prev_hour_converted,
                             mode="lines", line={'color': 'orange', 'width': 1}))
    fig.add_trace(go.Scatter(name = 'baseline', x=df.payload_expected_time, y=df.baseline,
                             mode="lines", line={'color': 'blue', 'width': 0.5}))
    fig.update_layout(title=f'device_id: {dev}',
                      xaxis_title="Time", yaxis_title="sum_payload_value_hour_curr",
                      template="simple_white")
    fig.update_xaxes(ticks='outside',
                     dtick=str(24*60*60*1000),
                     tickangle=-45,
                     tickformat='%a-%b-%e')
    fig.show()


