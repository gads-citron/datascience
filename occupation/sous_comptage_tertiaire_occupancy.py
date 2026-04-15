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
    score = np.mean((df.shift() - df) < 0) # combien de fois la valeur décrémente
    if score > tol:
        return 0 # volume
    else:
        return 1 # index

def write_to_db(s_occ, r, y_col, dataType, db):
    s_occ['device_id'] = r.device_id
    s_occ['sensor_name'] = r.sensor_name
    s_occ['sensor_id'] = r.sensor_id
    s_occ['site_id'] = r.site_id
    s_occ['data_source'] = [['.'.join([p.datasource, y_col])].copy() for _ in range(len(s_occ))]
    s_occ['model_version'] = p.model_version
    s_occ['valueType'] = dataType
    s_occ['ingested_at'] = dt.datetime.now()
    db.insert_df(p.dataout, occ)

def preprocess(s, y_col):
    """
    Parameters
    ----------
    s : pandas.Series()
        3 weeks water flow consumption (without spraying/fire network)
    y_col : string column name used for data collect

    Returns
    -------
    s.flow : pandas.Series()
        3 weeks water flow consumption without potential peak anomaly.

    """
    q3 = s.quantile(0.75)
    q1 = s.quantile(0.25)
    u_fence = q3 + (1.5 * (q3 - q1))
    s = pd.DataFrame(s)
    id_error = s.loc[(s[y_col] > u_fence)].index
    s['h'] = s.index.weekday.astype(str) + s.index.hour.astype(str)
    idx = [str(x)+str(y) for x in range(7) for y in range(24)]
    g = s.loc[(s[y_col] <= u_fence)].groupby('h').mean()
    g = pd.DataFrame(index=idx).merge(g,
                                      right_index=True,
                                      left_index=True, how='outer')
    g = g.ffill()
    s['avg'] = s.replace(g[y_col]).h
    s.loc[id_error, y_col] = s.loc[id_error, "avg"].astype(float).values
    return s[y_col]

def check_problems(s, errors, y_col):
    """
        Parameters
        ----------
        s : sample data to be checked
        errors : log map to be filled
        y_col : col in "datamart_sous_comptage" used to read data
        Returns
        -------
        error_dict : map of error description
        flag_return : 0 if no problem; 1 else
        y_col : same as input if no change
        datat : datatype for tracking purpose (data index or volume)
    """
    datat = 'volume'
    flag_return = 0
    if s.empty:
        error_dict['error'] = "no_data"
        flag_return = 1
    if s[y_col].isna().mean() > 0.1:
        error_dict['error'] = "too_many_NAN"
        flag_return = 1
    if serie_typology(s[y_col]):
        y_col = 'eval_prev_hour'
        datat = 'index'
        if serie_typology(s[y_col]):
            errors['error'] = "index_serie_on_eval_prev"
            flag_return = 1
    if s[y_column].astype(float).sum() == 0:
        errors['error'] = "zero_values"
        flag_return = 1
    if df[y_column].value_counts().shape[0] < 100:
        errors['error'] = "continuous"
        flag_return = 1
    if df.shape[0] < 168 * 3:
        errors['error'] = "history_not_enough"
        flag_return = 1
    return errors, flag_return, y_col, datat

#%% DtB connect
client = db_connect(type='prod')
#%% sites & devices candidates
# df = devices[(devices.meter_type == 'Sous-compteur') \
#                  & (devices.Typologie_client == 'Tertiaire')]
# #%% sensors list related to elec devices
iq_entities = pd.read_csv('./data/clients_ajoutés_sous_comptage.csv')
# sensors_elec = pd.read_csv('./data/liste_sensor_electric_devices.csv', sep=';')
# sensors = sensors[sensors.Considere==1] # no gaz/water sensor

#%% liste des equipements et sites tertiaire provenant de iqspot
sample = client.query_df(f"""
            SELECT distinct site_id, device_id, sensor_id, sensor_name
            FROM {p.datasource}
            WHERE 1=1
            AND payload_expected_time between '{p.start}' and '{p.end}'
            and site_typology = 'Tertiaire'
            and lower(device_type) like '{p.dev_type_water}'
            and entity IN {iq_entities.entity.to_list()}
            settings max_rows_to_read = 10000000000000;""")
#%% compute and write to database
x_time = 'payload_expected_time'
db_stage = db_connect(type='stage')
run_errors = []
for i, row in tqdm(sample[:500].iterrows()):
    end = dt.datetime(2026,3,27)
    start = end - dt.timedelta(hours=3*168)
    while end >= dt.datetime(2026, 1, 1):
        y_column = 'sum_payload_value_hour_curr'
        error_dict = {"device":row.device_id, "start":start, 'end':end,
                      "sensor_name":row.sensor_name, "feature_col":y_column}
        df = client.query_df(f"""
            SELECT device_id, {x_time}, sum_payload_value_hour_curr, eval_prev_hour
            FROM {p.datasource}
            WHERE 1=1
            AND {x_time} between '{start}' and '{end}'
            AND device_id = '{row.device_id}'
            AND sensor_id = '{row.sensor_id}'
            ORDER BY {x_time} DESC
            settings max_rows_to_read = 10000000000000;""")
        error_dict, flag, y_column, datatype = check_problems(df, error_dict, y_column)
        if flag:
            run_errors.append(error_dict)
            end -= dt.timedelta(hours=168)
            start -= dt.timedelta(hours=168)
            continue
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df = df.resample('1h', on=x_time)[y_column].sum().astype(float)
            smp = preprocess(df, y_column)
            scaled = scaler.fit_transform(pd.DataFrame(smp)).flatten()
            try:
                idx = df.index
                s_extract = Seasonality()
                occ = s_extract.compute(scaled, idx)
                occ.reset_index(inplace=True)
                occ.rename({x_time:'payload_timestamp'}, inplace=True, axis=1)
                write_to_db(occ, row, y_column, datatype, db_stage)
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