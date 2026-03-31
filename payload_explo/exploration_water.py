# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 12:00:48 2025

@author: Myriam Lopez
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import urllib.parse
import os
import clickhouse_connect
import numpy as np
import pandas as pd
import datetime as dt
import re
import numpy as np

orange = '#FF670F'
pink = '#FC006D'
blue = '#006CED'
green = '#2BD67C'
red = '#FF5D5D'
black = '#2B3034'

#tables
grdf_all='datamart_grdf.dataset_grdf_all'
releves='datawarehouse_prod.fact_payload'
clients='datawarehouse_prod.dim_entity'
equipements='datawarehouse_prod.dim_device'
capteurs='datawarehouse_prod.dim_sensor'
patrimoine='datawarehouse_prod.device_patrimonial_data'

source="grdf"
maxi = 10000000000000

load_dotenv()
host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
client = clickhouse_connect.get_client(host=host, username=username, 
                                       port = 443, password=password)

#%%
# device type 'compteurs d'eau'
water_dev = client.query_df(f"""
            SELECT device_type, device_id, iot_entity
            FROM {equipements}
            WHERE match(device_type, '(?i).*compt.*eau.*')
            AND not match(device_type, '(?i).*chau.*')
            settings max_rows_to_read={maxi};""")
            
tertiaire_buildings = client.query_df(f"""
                SELECT device_id, site_typology, site_nature, sub_nature, sub_category
                FROM {patrimoine}
                WHERE site_typology = 'Tertiaire'                
                settings max_rows_to_read = {maxi};""")
#bât tertiaire où on récupère de l'eau
water_dev = water_dev.merge(tertiaire_buildings, on = 'device_id')

entity_df = client.query_df(f"""
            SELECT entity_id, entity_name
            FROM {clients}
            WHERE entity_id IN {water_dev.iot_entity.to_list()}
            settings max_rows_to_read={maxi};""")
# ajout du client ESSET
entity_df = pd.concat([entity_df, pd.DataFrame([{'entity_id':'6037dc5ff435564fd2458b88', 
                                   'entity_name':'ESSET'}])])
            
water_dev= water_dev.merge(entity_df, right_on="entity_id", left_on="iot_entity")

#%% Profondeur d'historique
capteurs_df = client.query_df(f"""
            SELECT distinct payload_metric_id
            FROM {releves}
            WHERE device_id IN {water_dev.device_id.to_list()}
            settings max_rows_to_read={maxi};""")
        
capteurs_df = capteurs_df.merge(client.query_df(f""" SELECT sensor_id, name
                                FROM {capteurs}
                                WHERE sensor_id IN {capteurs_df.payload_metric_id.to_list()}
                                settings max_rows_to_read={maxi};"""), 
                                right_on='sensor_id', left_on="payload_metric_id")
        
#%% définition des données sur le temps
profondeur_df = []
for capteur in capteurs_df.sensor_id.to_list():    
    profondeur_cp = client.query_df(f""" 
            SELECT payload_metric_id, device_id, 
            min(payload_timestamp) start, max(payload_timestamp) end,
            count(payload_timestamp) nbPay
            FROM {releves}
            WHERE device_id IN {water_dev.device_id.to_list()}
            AND payload_metric_id='{capteur}'
            GROUP BY payload_metric_id, device_id
            SETTINGS max_rows_to_read={maxi};""")
    profondeur_df.append(profondeur_cp)
    
profondeur_df = pd.concat(profondeur_df)
profondeur_df.pivot_table(index='device_id', columns = 'sensor_id', values = ')