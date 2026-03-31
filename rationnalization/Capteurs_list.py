# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 14:44:07 2026

@author: Myriam Lopez
"""
from dotenv import load_dotenv
import urllib.parse
import os
import clickhouse_connect
from clickhouse_connect.driver.httpclient import DatabaseError
import pandas as pd
from tqdm import tqdm

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
#%%
q = f"""SELECT distinct metric_name \
        FROM {networks_line}\
        SETTINGS max_rows_to_read = {maxi};"""
sensors = client.query_df(q)
#%% Recoupement des infos dans dim_sensor
capteurs_info = []
for s in sensors.metric_name.to_list():
    q = f"""SELECT * \
            FROM {sens}\
                WHERE name = '{s}'
            SETTINGS max_rows_to_read = {maxi};"""
    info = client.query_df(q)
    if info.empty:
        print(s)
    capteurs_info.append(client.query_df(q))
capteurs_info = pd.concat(capteurs_info)
# colonnes : ['sensor_id', 'name', 'slug', 'unit', 'description']
#%% Récupération des listes des équipements dans le datamart network
q = f"""SELECT distinct device_id \
        FROM {networks_line}\
        SETTINGS max_rows_to_read = {maxi};"""
devices = client.query_df(q)
# 6707 équipements
#%% Requête dans la table des payloads pour les couples equipement capteur de la liste
q = f"""SELECT distinct device_id, payload_metric_id \
        FROM {releves}\
            WHERE device_id IN {devices.device_id.to_list()}
            AND payload_metric_id IN {capteurs_info.sensor_id.to_list()}
        SETTINGS max_rows_to_read = {maxi};"""
paritals = client.query_df(q)
# Au maxi 8 capteurs par équipement
# 1410 équipements sont connectés à plus d'un capteur
#%% Récupération du client associé à l'équipement
q = f"""SELECT device_id, iot_entity, device_type, device_name \
        FROM {dev}\
            WHERE device_id IN {devices.device_id.to_list()}
        SETTINGS max_rows_to_read = {maxi};"""
dev_info = client.query_df(q)
# Attention équipement 69837bff14b68f585436063c non retrouvé 
# colonnes = ['device_id', 'iot_entity', 'device_type']
#%% Nom du client à rajouter 
q = f"""SELECT entity_id, entity_name \
        FROM {entity}\
            WHERE entity_id IN {dev_info.iot_entity.drop_duplicates().to_list()}
        SETTINGS max_rows_to_read = {maxi};"""
entity_info = client.query_df(q)
#%% Croisement des informations
combined_df = paritals.merge(dev_info)\
    .merge(capteurs_info, right_on='sensor_id', left_on='payload_metric_id')\
        .merge(entity_info, right_on='entity_id', left_on='iot_entity')
# garder en tête qu'il y a toujours le device 69837bff14b68f585436063c dont on n'a pas l'info

combined_df = combined_df[['device_id', 'device_type', 'device_name',
       'sensor_id', 'sensor_name', 'unit','entity_id','entity_name']]

#%% Doublons par info ?
features = ['delta_temperature_depart_retour', 'pression', 'delta_temperature_retour_exterieur', \
            'temperature_exterieure', 'temperature_consigne', 'temperature_retour', \
                'temperature_depart', 'delta_temperature_retour_consigne']
# pression
q = f"""SELECT distinct pression_sensor_name \
        FROM {networks_col}\
        SETTINGS max_rows_to_read = {maxi};"""    
pre = client.query_df(q)
pre['feature'] = 'pression'
pre.columns=['sensor_name', 'feature']

# temperature exterieure
q = f"""SELECT distinct source_temperature_exterieure \
        FROM {networks_col}\
        SETTINGS max_rows_to_read = {maxi};"""    
temp = client.query_df(q)
temp['feature'] = 'temperature_exterieure'
temp.columns=['sensor_name', 'feature']

# temperature consigne
q = f"""SELECT distinct temperature_consigne_sensor_name \
        FROM {networks_col}\
        SETTINGS max_rows_to_read = {maxi};"""    
cons = client.query_df(q)
cons['feature'] = 'temperature_consigne'
cons.columns=['sensor_name', 'feature']

# temperature retour
q = f"""SELECT distinct temperature_retour_sensor_name \
        FROM {networks_col}\
        SETTINGS max_rows_to_read = {maxi};"""    
retour = client.query_df(q)
retour['feature'] = 'temperature_retour'
retour.columns=['sensor_name', 'feature']

# temperature départ
q = f"""SELECT distinct temperature_depart_sensor_name \
        FROM {networks_col}\
        SETTINGS max_rows_to_read = {maxi};"""    
depart = client.query_df(q)
depart['feature'] ='temperature_depart'
depart.columns=['sensor_name', 'feature']
#%%
sensors_family = pd.concat([pre, temp, cons, retour, depart])
sensors_family = sensors_family.reset_index(drop=True)
combined_df = combined_df.merge(sensors_family, how='left')
combined_df.to_csv('./Documents/datascience/data_analyse/Suivi_des_reseaux/Liste_des_couples_capteur_equipement.csv', index=False)