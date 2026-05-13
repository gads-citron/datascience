# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 11:06:06 2025

@author: Myriam Lopez
"""
from dotenv import load_dotenv
import urllib.parse
import os
import clickhouse_connect
from clickhouse_connect.driver.httpclient import DatabaseError
import pandas as pd
from tqdm import tqdm

# tables
enedis = 'datamart_nrj.dataset_enedis_hour'
releves='datawarehouse_prod.fact_payload'
clients='datawarehouse_prod.dim_entity'
equipements='datawarehouse_prod.dim_device'
capteurs='datawarehouse_prod.dim_sensor'
patrimoine='datawarehouse_prod.device_patrimonial_data'

# variables
history = "toDateTime('2026-01-01 01:00:00')"
maxi = 10000000000000

water_vol_m3 = '63ea423a54a72b939f458aa0'
water_vol_L = '63ea427554a72bd68d458aa1'
water_idx_m3 = '63ea403fcb597437b92360b7'
water_cpt = '604266a583c2f34ea421ffb8'

# useful lists
power_sensor = ["6454b98a0dae31f561c6f68f", "60c0f3f14fb6db2a7f4f5467", "5e5d4f8cdbf60d60cabb5749"]
water_sensor = [water_vol_m3, water_idx_m3, water_vol_L, water_cpt, \
                "5a391b5d126e6e2424a237dc", "5dcc04f2bc7a8e442aceeb9a", "662bc1cab204a2e9b45cfda8",\
                    "662bc1cad146fde77f9fc03d", "662bc1cadb23311c9b7fea30"]
vol_sens = [water_vol_m3, water_vol_L, '662bc1cadb23311c9b7fea30', '662bc1cab204a2e9b45cfda8', \
            '5dcc04f2bc7a8e442aceeb9a']
# devices having payload on 604266a583c2f34ea421ffb8 but related to volume data
vol_dev = ['6042734d83c2f34ea4220621','6042735683c2f34ea4220633']

# devices having inconsistant data
to_delete_dev = ['6042734f83c2f34ea4220625','6042735383c2f34ea422062d']

#%% clickhouse env load
load_dotenv()
host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
client = clickhouse_connect.get_client(host=host, username=username,
                                       port = 443, password=password)
#%% functions
def volume_preprocess(df, b, dev):
    if water_vol_L in df.payload_metric_id.unique():
        df['payload_value'] = df.payload_value/1000
    resampled = df.resample('1h', on='payload_timestamp').payload_value.mean()
    coverage = resampled.dropna().shape[0]/(
        (df.payload_timestamp.max() - df.payload_timestamp.min()).days*24)
    if coverage < 0.6:
        return 0
    resampled = resampled.pivot_table(index="payload_timestamp", values="payload_value", 
                 columns='payload_metric_id').rename_axis(None, axis=1)
    resampled.dropna(inplace=True)
    resampled['building'] = b
    resampled['device_id'] = dev
    
def index_preprocess(df, b, dev):
    neg = df.payload_value.diff(1)
    if neg.min() < 0:
        for i, row in df[neg < 0].iterrows():
            if row.payload_value==0:
                df.loc[i, "payload_value"] = \
                    (df.loc[i-1].payload_value + df.loc[i+1].payload_value)/2
            else:
                df.loc[i:, 'payload_value'] = df.loc[i:].payload_value \
                + df.loc[i-1].payload_value - df.loc[i].payload_value
    df = df.pivot_table(index="payload_timestamp", values="payload_value", 
                 columns='payload_metric_id').rename_axis(None, axis=1)        
    resampled = df.resample('1h').last().diff()
    coverage = resampled.shape[0]/((resampled.index.max() - resampled.index.min()).days*24)
    if coverage < 0.6:
        return 0
    resampled.dropna(inplace=True)
    resampled['building'] = b
    resampled['device_id'] = dev
    return resampled

def collect_from_db(i, dev):
    try:
        df = client.query_df(f""" 
            SELECT device_id, payload_timestamp, payload_metric_id, 
            toFloat64(payload_value) as payload_value
            FROM {releves}
            WHERE device_id IN {dev}
                AND payload_metric_id IN {water_sensor}
                    AND payload_timestamp > {history}
            ORDER BY payload_metric_id ASC
            SETTINGS max_rows_to_read = {maxi};""")
    except DatabaseError:
        df = client.query_df(f"""
            SELECT device_id, payload_timestamp, payload_metric_id, 
                toFloat64(replace(payload_value, '"', '')) AS payload_value
            FROM {releves}
            WHERE device_id IN {dev}
                AND payload_metric_id IN {water_sensor}
                    AND payload_timestamp > {history}
            ORDER BY payload_metric_id ASC
            SETTINGS max_rows_to_read = {maxi};""")
    return df
        
def saveParquet(df, folder):
    df['year'] = df.index.year
    df['month'] = df.index.month
    df.reset_index(inplace=True)
    df.to_parquet('./occupancy_citron/data/'+folder+'/', partition_cols=['year', 'month'])
    
def detect_index(s):
    scaler = MinMaxScaler(feature_range=(0, 1))
    s['p_value'] = scaler.fit_transform(s[['payload_value']]).flatten()
    score = np.var(s.sort_values('payload_timestamp').p_value.diff(1)
    return score
#%% filtre des devices en fonction de leur fréquence
candidate_freq = []
for i, s in tqdm(devices[devices.Feature=='Eau'].groupby(['building', 'device_id'])):
    freq = client.query_df(f"""
    SELECT p.device_id, arrayReduce('avg', diffs) AS avgDiff 
        FROM(
            SELECT 
                p.device_id,
                arrayDifference(arraySort(groupArray(p.payload_timestamp))) AS diffs
            FROM {releves} p      
            WHERE device_id IN {s.device_id.to_list()}
            AND payload_metric_id IN {water_sensor}
            AND payload_timestamp > {history}
        GROUP BY 
            p.device_id) p
        ORDER BY avgDiff DESC
        settings max_rows_to_read = {maxi}""")
    if not freq.empty:
        freq['building'] = i[0]
        freq['device_id'] = i[1]
        candidate_freq.append(freq)
candidate_freq = pd.concat(candidate_freq)
#%% Collecte et nettoyage des données d'eau
zeros = []
errors = []
sensor_pb = []
cover_error = []
empty_dev = []
for i, s in tqdm(devices_water[devices_water.building.isin(dev_oneH.index)].groupby(['building', 'device_id'])):
    df = collect_from_db(i, s.device_id.to_list())
    if df.empty: # no data for this device 
        empty_dev.append({'building':i[0], 'device_id':i[1]})
        continue
    capteurs = df.payload_metric_id.unique()
    if len(capteurs) > 1: # device emits over more than 1 sensor
        if water_vol_m3 in capteurs:
            df = df[df.payload_metric_id==water_vol_m3]
            df.reset_index(drop=True, inplace=True)
        elif water_vol_L in capteurs:
            df = df[df.payload_metric_id==water_vol_L] 
            df.reset_index(drop=True, inplace=True)
        else:
            sensor_pb.append({'building':i[0], 'device_id':i[1]})
        if df.payload_value.sum()==0:
            zeros.append({'building':i[0], 'device_id':i[1]})
            continue
        capteurs = df.payload_metric_id.unique()
    try:
        if (capteurs[0] in vol_sens) or (i[1] in vol_dev):
            clean = volume_preprocess(df, i[0], i[1])
        else:
            clean = index_preprocess(df, i[0], i[1])
    except:
        print (f'An error occured on building {i[0]} and device {i[1]}')
        errors.append({'building':i[0], 'device_id':i[1]})
        continue
    if type(clean) == int:
        cover_error.append({'building':i[0], 'device_id':i[1]})
    else:
        saveParquet(df.set_index('payload_timestamp'), 'water_data')        
#%% collecte des données d'électricité
freq_power = pd.read_csv('./collection_payloads/frequence_device_power.csv')
candidates = pd.read_csv('./collection_payloads/Candidats_occupation_citron.csv', sep=";")
devices = pd.read_csv('./occupancy_citron/liste_des_equipements.csv')
freq_power = freq_power.merge(devices[['device_id', 'entity_name']])
freq_power_candidate = freq_power.merge(candidates[['entity_name', 'intérêt occupation']])
freq_power_candidate = freq_power_candidate[freq_power_candidate["intérêt occupation"]==1]
#%%
df = client.query_df(f""" 
    SELECT device_id, payload_datetime, consommation_kwh, entity_id, site_id
    FROM {enedis}
    WHERE device_id = '{dev}'
            AND payload_datetime > {history}
    SETTINGS max_rows_to_read = {maxi};""")
    
    SELECT * 
FROM datamart_nrj.dataset_enedis_hour
LIMIT 100
settings max_rows_to_read=10000000000000;
#%%
zeros = []
errors = []
sensor_pb = []
data_elec = []
empty_dev = []
for dev in tqdm(freq_power_candidate.device_id):
    # try:
    df = client.query_df(f""" 
             SELECT device_id, payload_datetime, consommation_kwh, entity_id, site_id
             FROM {enedis}
             WHERE device_id = '{dev}'
                     AND payload_datetime > {history}
             SETTINGS max_rows_to_read = {maxi};""")
    # except DatabaseError:
    #     df = client.query_df(f"""
    #         SELECT device_id, payload_timestamp, payload_metric_id, 
    #             toFloat64(replace(payload_value, '"', '')) AS payload_value
    #         FROM {releves}
    #         WHERE device_id IN {s.device_id.to_list()}
    #             AND payload_metric_id IN {water_sensor}
    #                 AND payload_timestamp > {history}
    #         ORDER BY payload_metric_id ASC
    #         SETTINGS max_rows_to_read = {maxi};""")
    if df.empty: # no data for this device 
        empty_dev.append({'device_id':dev})
        continue
    else:
        data_elec.append(df)