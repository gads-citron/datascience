# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 17:36:30 2026

@author: Myriam Lopez
"""
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
import urllib.parse
import os
import clickhouse_connect
import pandas as pd
import numpy as np
from tqdm import tqdm
import param as p
import datetime as dt
from Seasonality import Seasonality
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

def saveParquet(df, folder):
    df['year'] = df.index.year
    df['month'] = df.index.month
    df.reset_index(inplace=True)
    df.to_parquet('./occupancy_citron/data/'+folder+'/', partition_cols=['year', 'month'])

def serie_typology(df, tol=0.1):
    score = np.mean((df.shift() - df) < 0)
    if score > tol:
        return 0 # volume
    else :
        return 1 # index

def write_to_db(s_occ, r, y_col, db):
    s_occ['device_id'] = r.device_id
    s_occ['sensor_name'] = r.sensor_name
    s_occ['sensor_id'] = r.sensor_id
    s_occ['site_id'] = r.site_id
    s_occ['data_source'] = [['.'.join([p.datasource, y_col])].copy() for _ in range(len(s_occ))]
    s_occ['model_version'] = model_version
    s_occ['ingested_at'] = dt.datetime.now()
    db.insert_df(p.dataout, occ)
#%% liste des équipements et sites
devices = pd.read_csv('./data/sous_comptage_elec_tertiaire.csv')
#%% liste des capteurs relevant
sensors = pd.read_csv('./data/liste_sensor_electric_devices.csv', sep=';')
# sensors = sensors[sensors.Considere==1]
sensors = sensors[sensors.Index==1]
#%%
df = devices[(devices.meter_type == 'Sous-compteur') \
                 & (devices.Typologie_client == 'Tertiaire')]
client = db_connect(type='prod')
# uniquement les capteurs volumétriques (sur le papier)
sample = client.query_df(f"""
            SELECT distinct device_id, sensor_id, sensor_name, site_id
            FROM {p.datasource}
            WHERE 1=1
            AND payload_expected_time >= '{dt.datetime(2026,1,1)}'
            AND device_id  IN {df.device_id.to_list()}
            AND sensor_id IN {sensors.sensor_id.to_list()}
            settings max_rows_to_read = 10000000000000;""")
#%% compute and write to database
# occupancy = []
model_version = 'V0'
y_column = 'eval_prev_hour' # colonne pour données de type index
# y_column = 'sum_payload_value_hour_curr' # colonne pour des données volumétriques
x_time = 'payload_expected_time'
db_stage = db_connect(type='stage')
run_errors = []
for i, row in tqdm(sample[:].iterrows()):
    end = dt.datetime(2026,3,27)
    start = end - dt.timedelta(hours=3*168)
    while end >= dt.datetime(2026, 1, 1):
        error_dict = {"device":row.device_id, "start":start, 'end':end,
                      "sensor_name":row.sensor_name, "feature_col":y_column}
        df = client.query_df(f"""
            SELECT device_id, {x_time}, {y_column}
            FROM {p.datasource}
            WHERE 1=1
            AND {x_time} > '{start}'
            AND {x_time} <= '{end}'
            AND device_id = '{row.device_id}'
            AND sensor_id = '{row.sensor_id}'
            ORDER BY {x_time} DESC
            settings max_rows_to_read = 10000000000000;""")
        if df.empty:
            error_dict['error']="no_data"
            run_errors.append(error_dict)
            end -= dt.timedelta(hours=168)
            start -= dt.timedelta(hours=168)
            continue
        if df[y_column].isna().mean() > 0.1 :
            error_dict['error'] = "too_many_NAN"
            run_errors.append(error_dict)
            end -= dt.timedelta(hours=168)
            start -= dt.timedelta(hours=168)
            continue
        if df[y_column].astype(float).sum() == 0:
            error_dict['error']="zero_values"
            run_errors.append(error_dict)
            end -= dt.timedelta(hours=168)
            start -= dt.timedelta(hours=168)
            continue
        if df[y_column].value_counts().shape[0] < 100:
            error_dict['error']="discrete"
            run_errors.append(error_dict)
            end -= dt.timedelta(hours=168)
            start -= dt.timedelta(hours=168)
            continue
        if serie_typology(df[y_column]):
            error_dict['error'] = "index_serie"
            run_errors.append(error_dict)
            end -= dt.timedelta(hours=168)
            start -= dt.timedelta(hours=168)
            continue
        if df.shape[0] < 168*3:
            error_dict['error']="history_not_enough"
            run_errors.append(error_dict)
            end -= dt.timedelta(hours=168)
            start -= dt.timedelta(hours=168)
            continue
        scaler = MinMaxScaler(feature_range=(0, 1))
        df = df.resample('1h', on=x_time)[y_column].sum().astype(float)
        smp = df.fillna(0)
        scaled = scaler.fit_transform(pd.DataFrame(smp)).flatten()[:168*3]
        try:
            idx = df.index
            s_extract = Seasonality()
            occ = s_extract.compute(scaled, idx)
            occ.reset_index(inplace=True)
            occ.rename({x_time:'payload_timestamp'}, inplace=True, axis=1)
            write_to_db(occ, row, y_column,db_stage)
            # occupancy.append(occ)
            end -= dt.timedelta(hours=168)
            start -= dt.timedelta(hours=168)
            run_errors.append(error_dict)
        except:
            error_dict['error']="frstl_pb"
            run_errors.append(error_dict)
            end -= dt.timedelta(hours=168)
            start -= dt.timedelta(hours=168)
            continue
run_errors = pd.DataFrame(run_errors)
#%% liste des zones en fonction des capteurs et noms d'équiepement
df = client.query_df(f"""
            SELECT distinct device_id, sensor_name
            FROM {p.datasource}
            WHERE 1=1
            AND payload_expected_time >= '{dt.datetime(2026,1,1)}'
            AND device_id  IN {sample.device_id.to_list()}
            AND sensor_id IN {sensors_vol.sensor_id.to_list()}
            settings max_rows_to_read = 10000000000000;""")
#%% CHECK
df = client.query_df(f"""
    SELECT device_id, {x_time}, {y_column}
    FROM {p.datasource}
    WHERE 1=1
    AND {x_time} > '{start}'
    AND {x_time} <= '{end}'
    AND device_id = '{d}'
    AND sensor_name like '{sens}'
    ORDER BY {x_time} DESC
    settings max_rows_to_read = 10000000000000;""")