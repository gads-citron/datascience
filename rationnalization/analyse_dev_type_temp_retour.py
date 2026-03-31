# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 10:11:43 2026

@author: Myriam Lopez
"""

from dotenv import load_dotenv
import urllib.parse
import os
import clickhouse_connect
from clickhouse_connect.driver.httpclient import DatabaseError
import pandas as pd
from tqdm import tqdm
import datetime as dt
import re

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
capteur_list = pd.read_csv('./Liste_des_couples_capteur_equipement.csv')
temperature_retour = capteur_list[capteur_list['variable_table_reseau']=='temperature_retour']
#%% Liste des équipements qui pointent sur les capteurs utilisés dans suivi des réseaux
capteurs = temperature_retour.sensor_id.drop_duplicates().to_list()
q = f"""SELECT distinct payload_metric_id, device_id, entity_id\
        FROM {releves}\
            WHERE payload_metric_id IN {capteurs}
            AND payload_timestamp > toDateTime('2026-01-01 01:00:00')
        SETTINGS max_rows_to_read = {maxi};"""
payload_devices = client.query_df(q)
payload_devices = payload_devices.merge(temperature_retour[['sensor_id', 'sensor_name']].drop_duplicates(), 
                                        right_on='sensor_id', left_on='payload_metric_id', how = 'inner')
#%% Nom du client à rajouter 
q = f"""SELECT entity_id, entity_name \
        FROM {entity}\
            WHERE entity_id IN {payload_devices.entity_id.drop_duplicates().to_list()}
        SETTINGS max_rows_to_read = {maxi};"""
entity_info = client.query_df(q)
payload_devices = payload_devices.merge(entity_info)
#%% récupération infos du capteur
q = f"""SELECT device_id, device_type, device_uid, device_name, reference_source, reference_value \
        FROM {dev}\
            WHERE device_id IN {payload_devices.device_id.to_list()}\
        SETTINGS max_rows_to_read = {maxi};"""
dev_info = client.query_df(q)
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
temperature_retour['is_in_datamart'] = 1
b = temperature_retour[['device_id', 'sensor_id', 'sensor_name', 'unit', 'variable_table_reseau', 'is_in_datamart']]

synthese = a.merge(b, on=['device_id', 'sensor_id', 'sensor_name'], how='left')
# synthese.to_csv('./Documents/datascience/data_analyse/Suivi_des_reseaux/whole_devices_temp_retour_with_stats.csv')
#%%
nb_dev_sensFam = synthese.groupby(['sensor_name', 'device_type'], group_keys=False).device_id.count()
nb_dev_famSens = synthese.groupby(['device_type','sensor_name'], group_keys=False).device_id.count()

#%%
changement = {'Automate':'Chaufferie[IoT]',
              'CTA':'CTA',
              'Chaufferie[IoT]':'Chaufferie[IoT]',
              'Circuit chauffage':'Circuit de chauffage',
              'Circuit de chauffage':'Circuit de chauffage',
              'Circuit de chauffage primaire': 'Circuit de chauffage',
              'Circuit de chauffage secondaire': 'Circuit de chauffage',
              'Compteur Thermique': 'Compteur thermique',
              'Compteur thermique':'Compteur thermique',
              "Compteur d'énergie thermique":'Compteur thermique',
              "Groupe Froid":'inconnu',
              "Régulateur":"inconnu",
              'Circuit ECS':'Circuit ECS',
              'Réseau ECS':'Circuit ECS',
              'Réseau chauffage':'Circuit de chauffage',
              'Réseau chauffage primaire':'Circuit de chauffage',
              'Réseau chauffage secondaire':'Circuit de chauffage',
              'Réseau eau chaude':'Réseau eau chaude',
              'Réseau eau glacée':'Réseau eau glacée',
              'SONDES DOUBLE ECS ':'Circuit ECS',
              'Sonde':'inconnu',
              'Sonde ECS':'Circuit ECS',
              'Sonde contact':'inconnu',
              'Sondes ECS':'Circuit ECS'             
    }

synthese['new_fam'] = synthese.device_type.apply(lambda x: changement[x])
pattern = '((?<![A-Za-z])CTA(?![A-Za-z])'+\
    '|(?<![A-Za-z])ECS(?![A-Za-z])|'+\
        '(?<![A-Za-z])RCU(?![A-Za-z])'+\
            '|chaufferie|chauffage|echangeur|'+\
                '(?<![A-Za-z])CHA(?![A-Za-z])|'+\
                    '(?<![A-Za-z])EC(?![A-Za-z])|'+\
                        '(?<![A-Za-z])EG(?![A-Za-z])|'+\
                            '(?<![A-Za-z])VC(?![A-Za-z])|'+\
                                'station|froid|chaud'+\
                                    '(?<![A-Za-z])VMC(?![A-Za-z]))'
                        
synthese['keyword'] = synthese.device_name.str.findall(pattern, flags=re.IGNORECASE)
synthese['keyword'] = synthese.keyword.apply(
    lambda x: '.'.join([m.lower() for m in x]) if isinstance(x, list) else ''
)
