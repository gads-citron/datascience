# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:59:24 2026

@author: Myriam Lopez
"""
import numpy as np
from dotenv import load_dotenv
import urllib.parse
import os
import clickhouse_connect
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
from plotly import io as pio
from tqdm import tqdm
pio.renderers.default='svg'
#%%
tab_anomalies = 'datamart_grdf_dev_mtahirintsoa.dataset_grdf_all'
tab_grdf='datamart_grdf.dataset_grdf_all'
tab_devices = 'datawarehouse_stage.dim_device'
maxi = 10000000000000
load_dotenv()
SIGERLY_id = '6808986418b9b4e0e0a053f1'
OGF_id = '64b8db49e0097b466c7104f9'
#%% methode de detection
def detect_outliers_iqr(data, k = 100):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    if iqr <1:
        iqr=1
    cutoff = iqr * k
    thres = q75 + cutoff
    return thres

def plot_conso(device, data, cons, time):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name = 'Consumption', x=data[time], y=data[cons], 
            mode='lines', line={'color': 'red', 'width':1}, showlegend=False
        )
    )
    fig.update_layout(title="Gaz consumption for " + device,
                      xaxis_title="Time", yaxis_title="Conso (kwh) informative",
                      template="simple_white")
    fig.show()

def plot_conso_anomaly(device, data, cons, anomaly, time, th=None):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name = 'Consumption', x=data[time], y=data[cons], 
            mode='lines', line={'color': 'red', 'width':1}, showlegend=False
        )
    )
    if th !=None : 
        fig.add_trace(
            go.Scatter(
                name = 'Threshold', x=data[time], y=data[th], 
                mode='lines', line={'color': 'black', 'width':1}, showlegend=False,
            )
        )
    fig.add_trace(go.Scatter(name="Anomaly", x=anomaly[time], 
                              y=anomaly[cons], mode="markers", marker={'color': 'blue', 'size': 4}))
    fig.update_layout(title="Anomaly detection for " + device,
                      xaxis_title="Time", yaxis_title="Conso (kwh) informative",
                      template="simple_white")
    fig.show()
    
def plot_violin_conso(device, data, cons):
    fig = go.Figure()
    fig.add_trace(
        go.Violin(y=data[cons], box_visible=True, 
                  line_color='black', meanline_visible=True, 
                  fillcolor='lightseagreen', opacity=0.6,
                  x0='Consommation'))
    fig.update_layout(title="Anomaly detection for " + device,
                      yaxis_title="Conso (kwh) informative",
                      template="simple_white")    
    fig.show()

#%% clickhouse env load
host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
client_prd = clickhouse_connect.get_client(host=host, username=username,
                                       port = 443, password=password)
#%%
host = urllib.parse.quote_plus(os.environ.get("clk_stg"))
username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
password = urllib.parse.quote_plus(os.environ.get('clk_stg_pass'))
client_stage = clickhouse_connect.get_client(host=host, username=username,
                                       port = 443, password=password)
#%% liste équipements OGF et SIGERLY uniquement
dev_ogf_sigerly = client_stage.query_df(f"""
        SELECT DISTINCT device_id, device_name
        FROM {tab_devices}
        WHERE iot_entity IN ('{SIGERLY_id}', '{OGF_id}')
        settings max_rows_to_read = {maxi}""")
#%% Liste anomalies pour OGF et SIGERLY
dev_anomaly = client_stage.query_df(f"""SELECT device_name, sum(beyond_threshold_hiver) AS total_anom
        FROM {tab_anomalies}
        WHERE device_id in {dev_ogf_sigerly.device_id.to_list()}
        GROUP BY device_name
        HAVING sum(beyond_threshold_hiver) >0""")
        
        
#%% récupération des données de ces équipements sur la période 2025
data_anom = client_stage.query_df(f"""SELECT device_name, beyond_threshold_hiver, threshold_hiver, consommation_kwh_informative, payload_time
        FROM {tab_anomalies}
        WHERE 1=1
            AND device_name in {dev_anomaly.device_name.to_list()}
            AND  payload_time > toDateTime('2024-01-01 01:00:00')
        """)
        
#%% calcul threshold sur un an 
anomaly_new_th = []
for d, test in data_anom.groupby('device_name'):
    test.reset_index(drop=True, inplace=True)
    for i in range (365, test.shape[0]):
        s = test.loc[:i]
        test.loc[i,'new_threshold'] = detect_outliers_iqr(s.consommation_kwh_informative.dropna())
    anomaly_new_th.append(test)
    
#%% calcul threshold fenêtre glissante 1 mois 
anomaly_new_th = []
for d, test in data_anom.groupby('device_name'):
    test.reset_index(drop=True, inplace=True)
    for i in range (15, test.shape[0]):
        s = test.loc[i-15:i+15]
        if s.consommation_kwh_informative.isna().sum()>10:
            continue
        test.loc[i,'new_threshold'] = detect_outliers_iqr(s.consommation_kwh_informative.dropna())
    anomaly_new_th.append(test)    
#%% Visualisation données brutes
for i, gr in data_anom.groupby('device_name'):    
    plot_conso(device=i, data=gr, cons='consommation_kwh_informative', time='payload_time')
#%% Visualisation données brutes avec anomalies
for i, gr in anomaly_new_th.groupby('device_name'):
    anomaly = gr[gr.consommation_kwh_informative > gr.new_threshold]
    plot_conso_anomaly(device=i, data=gr, 
                       cons='consommation_kwh_informative', 
                       anomaly=anomaly, time='payload_time')
    plot_violin_conso(device=i, data=gr, cons='consommation_kwh_informative')
#%% évaluation nouvelle méthode sur totalité données OGF SIGERLY
df_anomalies = []
df_stats_anom = []
for d in tqdm(dev_ogf_sigerly.device_name):
    df = client_stage.query_df(f"""SELECT device_name, beyond_threshold_hiver, 
                                      threshold_hiver, consommation_kwh_informative, payload_time
            FROM {tab_anomalies}
            WHERE 1=1
                AND device_name = '{d}'
                AND  payload_time > toDateTime('2023-12-31 01:00:00')
                AND  payload_time < toDateTime('2025-01-01 01:00:00')
            """)
    if df.empty:
        df_stats_anom.append({"device_name":d, "nb_anomalies":0, "shape":0})
        continue
    df['transformation'] = df.consommation_kwh_informative.apply(lambda x: np.log(x+1))
    if df.consommation_kwh_informative.sum()==0:
        df_stats_anom.append({"device_name":d, "nb_anomalies":0, "shape": df.shape[0]})
        continue
    df['date'] = df.payload_time.apply(lambda x: x.date())
    df['new_threshold'] = np.NaN
    for i in range(15,df.shape[0]):        
        t = df.loc[i, 'date']
        s = df.loc[i-15:i+14]
        if s.transformation.isna().sum()>10:
            continue
        df.loc[i, "new_threshold"] = detect_outliers_iqr(s.transformation.dropna(), k=1.3)
    nb_anom = df[df.transformation > df.new_threshold].shape[0]
    df_stats_anom.append({"device_name":d, "nb_anomalies":nb_anom, "shape": df.shape[0]})
    if nb_anom > 0:
        df_anomalies.append(df)
df_stats_anom = pd.DataFrame(df_stats_anom)
df_anomalies = pd.concat(df_anomalies)

#%%
for i, gr in df_anomalies.groupby('device_name'):
    anomaly = gr[gr.transformation > gr.new_threshold]
    if anomaly.empty:
        continue
    plot_conso_anomaly(device=i, data=gr, 
                       cons='consommation_kwh_informative', 
                       anomaly=anomaly, time='payload_time')
    
#%% changement de seuil k
k = 10
df_anomalies['th_k10'] = np.NaN
for d, gr in tqdm(df_anomalies.groupby('device_name')):
    for i in range(gr.index[0]+15,gr.index[0]+gr.shape[0]):        
        t = gr.loc[i, 'date']
        s = gr.loc[i-15:i+14]
        if s.transformation.isna().sum()>10:
            continue
        df_anomalies.loc[i, "th_k10"] = detect_outliers_iqr(s.transformation.dropna(), k=k)
        
#%% visualisation anomalies par paramètre k
devices = df_anomalies[df_anomalies.transformation>df_anomalies.th_k5].device_name
for i, gr in df_anomalies[df_anomalies.device_name.isin(devices)].groupby('device_name'):
    anomaly = gr[gr.transformation > gr.th_k5]
    if anomaly.empty:
        continue
    plot_conso_anomaly(device=i, data=gr, 
                       cons='consommation_kwh_informative', 
                       anomaly=anomaly, time='payload_time')