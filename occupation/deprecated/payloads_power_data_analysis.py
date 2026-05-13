# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 11:47:55 2025

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
import plotly.graph_objects as go
from plotly import io as pio
pio.renderers.default='svg'

#%%
orange = '#FF670F'
pink = '#FC006D'
blue = '#006CED'
green = '#2BD67C'
red = '#FF5D5D'
black = '#2B3034'

# tables
releves='datawarehouse_prod.fact_payload'
clients='datawarehouse_prod.dim_entity'
equipements='datawarehouse_prod.dim_device'
capteurs='datawarehouse_prod.dim_sensor'
patrimoine='datawarehouse_prod.device_patrimonial_data'
group = 'datawarehouse_prod.dim_group'

# variables
history = "toDateTime('2024-01-01 01:00:00')"
maxi = 10000000000000

# useful lists
w_cdc_power = ['6454b98a0dae31f561c6f693', '662bc1ca435092516520a135', '64ae9c2ce5464143c75735ce',
 '63ea5856cb5974b3842361d5', '63ea587854a72b5aee458c09', '686d327d5881188f74b5d528', '64b7b52de0097b2ab871042b',
 '67fce12d4e30e4cf43734576','6454b98a0dae31f561c6f68f']

w_idx_power = ['6454b98a0dae31f561c6f69f', '6454b98a0dae31f561c6f6a0', '6454b98a0dae31f561c6f69e', 
               '662bc1c9db23311c9b7fea25', '6454b98a0dae31f561c6f69b', '6454b98a0dae31f561c6f69c', 
               '6454b98a0dae31f561c6f69d', '6454b98a0dae31f561c6f69a', '6454b98a0dae31f561c6f699']
kw_cdc_power = ['662bc1ca3f0afb78ad059848']
kw_idx_power = ['59c3be73bee9f849f51461d3', '662bc1c93f0afb78ad05983d', '670e81d39f47470c6e108eaf']

power_sensors = w_cdc_power+w_idx_power+kw_cdc_power+kw_idx_power

#%% clickhouse env load
load_dotenv()
host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
client = clickhouse_connect.get_client(host=host, username=username,
                                       port = 443, password=password)
#%% functions
def plot_anomalies(time, y, anom):
    fig = go.Figure()
    fig.add_trace(go.Scatter(name="Truth", x=time, y=y,
                             mode="lines", line={'color': orange, 'width': 1}))
    fig.add_trace(go.Scatter(name="Anomaly", x=anom.payload_timestamp, y=anom.payload_value,
                             mode="markers", marker={'color': black, 'size': 4}))
    fig.update_layout(title="Anomaly detection",
                      xaxis_title="Time", yaxis_title="Water consumption (m3)",
                      template="simple_white")
    fig.show()

def plot_flow(time, y, yaxis, title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=y,
                             mode="lines", line={'color': orange, 'width': 1}))    
    fig.update_layout(title=title,
                      xaxis_title="Temps", yaxis_title=yaxis,
                      yaxis_range=[0, y.max()],
                      template="simple_white")
    fig.update_xaxes(ticks='outside',
                     dtick=str(30*24*60*60*1000),
                     tickangle=-45,
                     tickformat='%a-%b-%e')
    fig.show()

def compute_stats(smp):
    constant = smp.groupby('device_id')\
        .apply(lambda s: np.abs((s.payload_value.max() - s.payload_value.min())/s.payload_value.mean())).rename('constant')
    negatif = smp.groupby('device_id')\
        .apply(lambda s: s[s.payload_value <0].shape[0]).rename('negatif')
    discret = smp.groupby('device_id')\
        .apply(lambda s: len(s.payload_value.unique())).rename('discret')
    score = pd.concat([constant, discret, negatif], axis=1)
    return score

#%% Selection des équipements ayant une conso exploitable
freq_power = pd.read_csv('./collection_payloads/frequence_device_power.csv')
candidates = pd.read_csv('./collection_payloads/Candidats_occupation_citron.csv', sep=";")
devices = pd.read_csv('./occupancy_citron/liste_des_equipements.csv')
freq_power = freq_power.merge(devices[['device_id', 'entity_name']])
freq_power_candidate = freq_power.merge(candidates[['entity_name', 'intérêt occupation']])
freq_power_candidate = freq_power_candidate[freq_power_candidate["intérêt occupation"]==1]

#%%
# Collecte dans les fichiers parquets déjà stockés
data_prod = []
for dev in dev_prod.device_id:
    df= pd.read_parquet("./occupancy_citron/data/water_data/", 
                            filters=[("device_id", "=", dev)])
    data_prod.append(df)

data_prod = pd.concat(data_prod)
score = compute_stats(data_prod)
clean_data = data_prod[data_prod.device_id.isin(
    score[(score.constant >1) & (score.discret >10) & (score.negatif ==0)].index)]
clean_data = clean_data.merge(devices_water[['device_id', 'building', 'entity_name']])
#%%
root_consumption = water_data.groupby(['building', 'payload_timestamp'], as_index=False)\
    .agg({'entity_name':'min', 'payload_value':'sum'})
scaler = MinMaxScaler(feature_range=(0, 1))
for dv, s in root_consumption.groupby('building'):
# for dv, s in water_data.groupby(['building', 'device_id']):
    s['payload_timestamp'] = pd.to_datetime(s.payload_timestamp)
    s['payload_value'] = s.payload_value.astype(float)
    rs = s[['payload_timestamp', 'payload_value']].resample('1h', on='payload_timestamp').mean()
    rs['hour'] = rs.index.hour
    rs['day'] = rs.index.day_of_week
    week_type = rs.groupby(['day', 'hour'], as_index=False).payload_value.mean()
    scaled = scaler.fit_transform(week_type[['payload_value']]).flatten()
    plot_flow(week_type.index, week_type.payload_value, "Consommations d'eau (m3)", 
              s.entity_name.values[0])