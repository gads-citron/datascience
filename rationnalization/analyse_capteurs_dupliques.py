# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 08:41:44 2026

@author: Myriam Lopez
"""
from dotenv import load_dotenv
import urllib.parse
import os
import clickhouse_connect
from clickhouse_connect.driver.httpclient import DatabaseError
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
client = clickhouse_connect.get_client(host=host, username=username,
                                       port = 443, password=password)
maxi = 10000000000000
#%%
sens = 'datawarehouse_prod.dim_sensor'
dev = 'datawarehouse_prod.dim_device'
releves = 'datawarehouse_prod.fact_payload'
entity = 'datawarehouse_prod.dim_entity'
networks_line = 'datamart_networks.line'
networks_col = 'datamart_networks.column'

feature = 'temperature_consigne'
#%%
capteur_list = pd.read_csv('./Liste_des_couples_capteur_equipement.csv')
feature_df = capteur_list[capteur_list['variable_table_reseau']==feature]

#%% Liste des équipements qui pointent sur les capteurs utilisés dans suivi des réseaux
capteurs = feature_df.sensor_id.drop_duplicates().to_list()
q = f"""SELECT distinct payload_metric_id, device_id, entity_id\
        FROM {releves}\
            WHERE payload_metric_id IN {capteurs}
        SETTINGS max_rows_to_read = {maxi};"""
payload_devices = client.query_df(q)
payload_devices = payload_devices.merge(feature_df[['sensor_id', 'sensor_name']].drop_duplicates(), 
                                        right_on='sensor_id', left_on='payload_metric_id', how = 'inner')
#%% Nom du client à rajouter 
q = f"""SELECT entity_id, entity_name \
        FROM {entity}\
            WHERE entity_id IN {payload_devices.entity_id.drop_duplicates().to_list()}
        SETTINGS max_rows_to_read = {maxi};"""
entity_info = client.query_df(q)
payload_devices = payload_devices.merge(entity_info)
#%% récupération infos du device
q = f"""SELECT device_id, device_type, device_name \
        FROM {dev}\
            WHERE device_id IN {payload_devices.device_id.to_list()}
        SETTINGS max_rows_to_read = {maxi};"""
dev_info = client.query_df(q)
payload_devices = payload_devices.merge(dev_info, how = 'left')

#%% Statistiques volume et historique par équipement
q = f"""SELECT payload_metric_id, device_id, min(payload_timestamp) start, max(payload_timestamp) end, count(payload_timestamp) nbPay
        FROM {releves}\
            WHERE payload_metric_id IN {capteurs}
        GROUP BY payload_metric_id, device_id
        SETTINGS max_rows_to_read = {maxi};"""
payload_nb = client.query_df(q)

synthese_table = payload_devices.merge(payload_nb, on=['device_id', 'payload_metric_id'], how='left')
#%% merger avec les infos d'equipement concernant le datamart_networks
a = synthese_table[['device_id', 'device_type','device_name', 'sensor_id', 'sensor_name', 'entity_id', 'entity_name', 'start', 'end', 'nbPay']]
feature_df['is_in_datamart'] = 1
b = feature_df[['device_id', 'sensor_id', 'sensor_name', 'unit', 'variable_table_reseau', 'is_in_datamart']]

synthese = a.merge(b, on=['device_id', 'sensor_id', 'sensor_name'], how='left')
# synthese.to_csv(f'./Documents/datascience/data_analyse/Suivi_des_reseaux/whole_devices_{feature}_with_stats.csv')
#%%
nb_dev_sensFam = synthese.groupby(['sensor_name', 'device_type'], group_keys=False).device_id.count()
nb_dev_famSens = synthese.groupby(['device_type','sensor_name'], group_keys=False).device_id.count()
#%% statistiques par capteur
capteurs = feature_df.sensor_id.drop_duplicates().to_list()
q = f"""SELECT payload_metric_id, device_id, min(toFloat64OrNull(payload_value)) minimal, 
                max(toFloat64OrNull(payload_value)) maximal,
                 avg(toFloat64OrNull(payload_value)) moy, 
                 quantileExact(toFloat64OrNull(payload_value)) mediane
        FROM {releves}\
            WHERE payload_metric_id IN {capteurs}
        GROUP BY payload_metric_id, device_id
        SETTINGS max_rows_to_read = {maxi};"""
stat_dev = client.query_df(q)
#%% ajouter les dates de début et fin
mean_time = []
for i, row in stat_dev.iterrows():
    sens = row.payload_metric_id
    dev = row.device_id
    q = f"""SELECT payload_metric_id, device_id, 
                avg(toUnixTimestamp(payload_timestamp)) avg_time, 
                quantileExact(toUnixTimestamp(payload_timestamp)) med_time 
            FROM {releves}\
                WHERE payload_metric_id = '{sens}'
                AND device_id = '{dev}'
            GROUP BY payload_metric_id, device_id
            SETTINGS max_rows_to_read = {maxi};"""
    datetime = client.query_df(q)
    mean_time.append({"payload_metric_id":sens,
                     "device_id":dev,
                     'avg_time':datetime.avg_time.values[0],
                     'med_time':datetime.med_time.values[0]})
mean_time = pd.DataFrame(mean_time)
#%% Index ou pas  ?
is_index = []
for i, row in stat_dev.iterrows():
    sens = row.payload_metric_id
    dev = row.device_id
    q = f"""SELECT countIf(diff < 0) AS nb_diff_negatifs
            FROM
            (
             SELECT payload_metric_id , device_id, payload_timestamp,
                toFloat64OrNull(payload_value) - lag(toFloat64OrNull(payload_value))
                      OVER (ORDER BY payload_timestamp) AS diff
                FROM {releves}
                WHERE payload_metric_id = '{sens}'
                  AND device_id = '{dev}')
            WHERE diff IS NOT NULL
            SETTINGS max_rows_to_read = {maxi}"""
    datetime = client.query_df(q)
    is_index.append({"payload_metric_id":sens, 
                     "device_id":dev,
                     'idx_score':datetime.nb_diff_negatifs.values[0]})
is_index = pd.DataFrame(is_index)
#%% rassemblement
time_idx = mean_time.merge(is_index)
stat_dev = stat_dev.merge(time_idx, on=['payload_metric_id', 'device_id'])
#%% Clustering sur device, maximal et minimal
def similarity_v0(dev, smp):
    sim_matrix = cosine_similarity(smp.drop(['payload_metric_id', 'device_id'], axis=1).values)
    n = sim_matrix.shape[0]
    sim = []
    for i in range(n):
        for j in range(i+1, n):
            sim.append({'device': dev, 
                        'Sensor1':smp.payload_metric_id.values[i],
                        'Sensor2':smp.payload_metric_id.values[j], 
                        'similarite': sim_matrix[i,j]})
    return sim

df_similarite = []
for dev, g in stat_dev.groupby('device_id', group_keys=False):
    g.dropna(inplace=True)
    if g.shape[0] >1:
        df_similarite = df_similarite+similarity_v0(dev, g)
        
df_similarite = pd.DataFrame(df_similarite)
df_similarite['similarite_rnd'] = df_similarite.similarite.round(1)
list_sensors_same = pd.DataFrame(df_similarite.groupby(['Sensor1', 'Sensor2', 'similarite_rnd']).device.apply(list))
list_sensors_same = list_sensors_same.reset_index()
list_sensors_same['nb_devices'] = list_sensors_same.device.apply(len)

synthese_sensor_sim = list_sensors_same.merge(capteur_list[['sensor_id', 'sensor_name']].drop_duplicates(), 
                                              right_on='sensor_id', left_on='Sensor1')
synthese_sensor_sim.columns = ['Sensor1', 'Sensor2', 'similarite_rnd', 
                               'device', 'nb_devices','sensor_id', 'Name1']
synthese_sensor_sim = synthese_sensor_sim.drop('sensor_id', axis=1).merge(capteur_list[['sensor_id', 'sensor_name']].drop_duplicates(), 
                                              right_on='sensor_id', left_on='Sensor2')
synthese_sensor_sim.columns = ['Sensor1', 'Sensor2', 'similarite_rnd', 'device', 'nb_devices', 'Name1',
       'sensor_id', 'Name2']
synthese_sensor_sim = synthese_sensor_sim.drop('sensor_id', axis=1)
#%% check ahead
row_nb = 0 # liste des equipements associés au premier couple de capteur similaire
dev_l = synthese_sensor_sim.loc[row_nb].device 
sens = synthese_sensor_sim[['Sensor1', 'Sensor2']].loc[row_nb].to_list()
test = stat_dev[(stat_dev.payload_metric_id.isin(sens))&(stat_dev.device_id.isin(dev_l))]
#%%
row_nb = 2 # le second equipement de la liste
dev = test.device_id.values[row_nb]
q = f"""SELECT payload_metric_id, payload_timestamp, toFloat64OrNull(payload_value) payVal
        FROM {releves}
        WHERE payload_metric_id IN {sens}
        AND device_id = '{dev}'
        SETTINGS max_rows_to_read = {maxi}"""
df = client.query_df(q)
df = df.merge(feature_df[['sensor_id', 'sensor_name']].drop_duplicates(), 
         right_on='sensor_id', left_on='payload_metric_id')
df = df.pivot_table(index='payload_timestamp', columns='sensor_name', values = 'payVal')

#%% check exact similarity
exact_similarity = []
for i, row in synthese_sensor_sim.iterrows():
    sens = [row.Sensor1, row.Sensor2]
    test = stat_dev[(stat_dev.payload_metric_id.isin(sens))&\
                    (stat_dev.device_id.isin(row.device))]
        # tester la similarité valeur par valeur
    for dev in test.device_id.values:    
        q = f"""SELECT payload_metric_id, payload_timestamp, toFloat64OrNull(payload_value) payVal
                FROM {releves}
                WHERE payload_metric_id IN {sens}
                AND device_id = '{dev}'
                SETTINGS max_rows_to_read = {maxi}"""
        df = client.query_df(q)
        df = df.merge(feature_df[['sensor_id', 'sensor_name']].drop_duplicates(), 
                 right_on='sensor_id', left_on='payload_metric_id')
        df = df.pivot_table(index='payload_timestamp', columns='sensor_name', values = 'payVal')
        df.dropna(inplace=True)
        if df.empty:
            continue
        similarity = np.sum(df.iloc[:,0]==df.iloc[:, 1])/df.shape[0]
        exact_similarity.append({'device_id':dev, 'sensors':sens, 'score':similarity})
exact_similarity = pd.DataFrame(exact_similarity)
