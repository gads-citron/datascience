# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:02:57 2025

@author: Myriam Lopez
"""
from dotenv import load_dotenv
import urllib.parse
import os
import clickhouse_connect
import pandas as pd
from tqdm import tqdm

orange = '#FF670F'
pink = '#FC006D'
blue = '#006CED'
green = '#2BD67C'
red = '#FF5D5D'
black = '#2B3034'

#tables
releves='datawarehouse_prod.fact_payload'
equipements='datawarehouse_prod.dim_device'
group = 'datawarehouse_prod.dim_group'
big_table = 'datawarehouse_prod.device_big_table'

maxi = 10000000000000

power_sensor = ["6454b98a0dae31f561c6f68f", "60c0f3f14fb6db2a7f4f5467", "5e5d4f8cdbf60d60cabb5749"]
water_sensor = ["63ea423a54a72b939f458aa0", "5a391b5d126e6e2424a237dc", "604266a583c2f34ea421ffb8"]

load_dotenv()
host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
client = clickhouse_connect.get_client(host=host, username=username, 
                                       port = 443, password=password)
#%%
# candidats sélectionné en amont sur la base de leur possession d'équipements relevant la consommation d'eau et d'électricité
candidates = pd.read_csv('./collection_payloads/Candidats_occupation_citron.csv', sep=";")
#%% Sélection des équipements liées à la relève de consommation d'eau et de consommation d'électricité
# electricité
dev_elec = client.query_df(f"""SELECT distinct device_type, parent, device_id, device_name, iot_entity
                                FROM {equipements}
                                WHERE match (device_type, '(?i).*lectr.*')
                                SETTINGS max_rows_to_read={maxi};""")
dev_elec['Feature'] = 'elec'
# eau
dev_water = client.query_df(f"""SELECT distinct device_type, parent, device_id, device_name, iot_entity
                                FROM {equipements}
                                WHERE match (device_type, '(?i).*eau.*')
                                SETTINGS max_rows_to_read={maxi};""")
dev_water['Feature'] = 'Eau'
devices = pd.concat([dev_elec, dev_water])
#%% Sélection des groupes 'site' ou 'batiment' apprtenant aux clients candidats
buildings = client.query_df(f""" SELECT group_id, iot_entity, group_genre, parent_id
    FROM {group}
    WHERE group_genre IN ['Site', 'Batiment']
    AND iot_entity IN {candidates.entity_id.to_list()}
    settings max_rows_to_read = {maxi};""")
#%% Sélection des équipements associés aux groupes obtenus en amont
dev_list = []
for b in tqdm(buildings.group_id.values[:]):
    devices = client.query_df(f"""SELECT device_id, id_compte_client
    FROM {big_table}
    WHERE match (parents_with_sharings, '.*{b}.*')
    settings max_rows_to_read = {maxi};""")
    devices['building'] = b
    dev_list.append(devices)
dev_list = pd.concat(dev_list)
#%% Parmis tous les devices qui récupèrent de l'eau et de l'électricité, on filtre ceux qui appartiennent aux groupes sélectionnés
devices_b = devices.merge(dev_list, on="device_id")
# pour ne garder vraiment que ce qui est lié aux candidats présélectionnés
devices_b = devices_b.merge(candidates[['entity_id', 'entity_name', 'Typologie de client']], 
                              right_on="entity_id", left_on="iot_entity")
final_devices = devices_b[['device_id', 'device_type', 'device_name', 'iot_entity', 
                            'Feature', 'building', 'entity_name', 'Typologie de client']]