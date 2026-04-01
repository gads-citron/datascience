# -*- coding: utf-8 -*-
"""
Created on Tue Mar  31 11:58:30 2026

@author: Myriam Lopez
"""
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
import urllib.parse
import os
import clickhouse_connect
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
import param as p
#%% functions
def db_connect(type='prod'):
    load_dotenv()
    username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
    host = ''
    password = ''
    if type=='prod':
        host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
        password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
    elif type == 'stage':
        host = urllib.parse.quote_plus(os.environ.get("clk_stg"))
        password = urllib.parse.quote_plus(os.environ.get('clk_stg_pass'))
    # si host = password = '' : lever une erreur
    conn = clickhouse_connect.get_client(host=host, username=username, port=443, password=password)
    return conn
#%%
client = db_connect(type='prod')
db_stage = db_connect(type='stage')
#%% Liste des sites sans bâtiment ou site parmis ses enfants
liste_site_sans_enfant = pd.read_csv('./data/neo4j_sites_sans_enfants.csv')
#%% les données obtenues pour l'occupation V0
sample = db_stage.query_df(f"""
            SELECT device_id, sensor_id, site_id, min(payload_timestamp) start, max(payload_timestamp) end
            FROM Myriam_test.dataset_occupancy
            WHERE 1=1
            group by device_id, sensor_id, site_id
            settings max_rows_to_read = 10000000000000;""")
#%% compute and write to database
y_column = 'eval_prev_hour' # colonne pour données de type index
x_time = 'payload_expected_time'
results = []
for i, row in tqdm(sample[:].iterrows()):
    df = client.query_df(f"""
        SELECT device_id, {x_time}, {y_column}, sensor_name
        FROM {p.datasource}
        WHERE 1=1
        AND {x_time} > '{row.start}'
        AND {x_time} <= '{row.end}'
        AND device_id = '{row.device_id}'
        AND sensor_id = '{row.sensor_id}'
        ORDER BY {x_time} DESC
        settings max_rows_to_read = 10000000000000;""")
    scaler = MinMaxScaler(feature_range=(0, 1))
    # df = df.resample('1h', on=x_time)[y_column].sum().astype(float)
    # smp = df.fillna(0)
    df['scaled'] = scaler.fit_transform(pd.DataFrame(df[y_column].fillna(0))).flatten()
    df['hour'] = df[x_time].dt.hour
    df['day'] = df[x_time].dt.day_of_week
    df['week'] = df[x_time].dt.isocalendar().week
    profiles = df.groupby(['day', 'hour'])['scaled'].mean().reset_index()
    profiles.columns = ['day', 'hour', 'baseline']
    # data seasonal
    s = db_stage.query_df(f"""
                SELECT device_id, payload_timestamp, occupancy_value
                FROM Myriam_test.dataset_occupancy
                WHERE 1=1
                AND device_id = '{row.device_id}'
                settings max_rows_to_read = 10000000000000;""")
    s.loc[s.occupancy_value < 0, "occupancy_value"] = 0
    s['hour'] = s.payload_timestamp.dt.hour
    s['day'] = s.payload_timestamp.dt.day_of_week
    val_set = s.merge(profiles, on=["day", "hour"], how="left")
    results.append({
        'device_id':row.device_id, \
        'start':row.start,
        'end':row.end,
        'score':root_mean_squared_error(val_set.baseline.fillna(0), \
                                        val_set.occupancy_value)})
results = pd.DataFrame(results)
#%%
baseline = pd.concat(baseline)
dataset_test = dataset_test.merge(profiles, on=["device_id", "day", "hour"], how="left")