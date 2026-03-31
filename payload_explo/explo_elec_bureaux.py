# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

orange = '#FF670F'
pink = '#FC006D'
purple = '#8659D9'
blue = '#006CED'
green = '#2BD67C'
red = '#FF5D5D'
black = '#2B3034'

#tables
releves='datawarehouse_prod.fact_payload'
enedis = 'datamart_nrj.dataset_enedis_hour'
clients = 'datawarehouse_prod.dim_entity'
equipements = 'datawarehouse_prod.dim_device'
capteurs = 'datawarehouse_prod.dim_sensor'
patrimoine = 'datawarehouse_prod.device_patrimonial_data'

#autres variables
maxi = 10000000000000
load_dotenv()
host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
client = clickhouse_connect.get_client(host=host, username=username,
                                       port = 443, password=password)
#%% liste des équipements associés à des bâtiments de type bureau
office_dev = pd.read_csv('./data/list_equipement_bureaux_elec.csv')

#%% nom des clients
client_list = office_dev.iot_entity.drop_duplicates().to_list()
entity_df = client.query_df(f"""
        SELECT entity_id, entity_name
        FROM {clients}
        WHERE entity_id IN {client_list}
        settings max_rows_to_read = {maxi};""")
office_dev = office_dev.merge(entity_df, right_on="entity_id", left_on='iot_entity')

#%% statistiques payloads dans la table enedis
stats = client.query_df(f"""SELECT device_id, min(payload_datetime) as debut, max(payload_datetime) as fin
                            FROM {enedis}
                            WHERE 1=1
                                AND device_id IN {office_dev.device_id.drop_duplicates().to_list()}                                 
                            GROUP BY device_id
                            settings max_rows_to_read = {maxi}""")
office_dev = stats.merge(office_dev)
#%%
df = client.query_df(f"""SELECT device_id, payload_datetime, payload_month, payload_hour, 
                            week_number, week_weekend, hot_dju, cold_dju, surface, consommation_kwh
                            FROM {enedis}
                            WHERE 1=1
                                AND device_id = '{dev}'
                                AND payload_datetime > toDateTime('2025-05-01 12:00:00')
                            settings max_rows_to_read = {maxi}""")


#%%
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='svg'
ambre = '#efc100'
groseille = '#d92338'
dev = '6675ae4353988e2522a4c493'

df = client.query_df(f"""SELECT device_id, payload_datetime, payload_month, payload_hour, 
                            week_number, week_weekend, hot_dju, cold_dju, surface, consommation_kwh
                            FROM {enedis}
                            WHERE 1=1
                                AND device_id = '{dev}'
                                AND payload_datetime > toDateTime('2025-05-01 12:00:00')
                            settings max_rows_to_read = {maxi}""")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df.payload_datetime,
        y=df.consommation_kwh,
        mode='lines',
        line={'color': ambre, 'width':1}, showlegend=False
    )
)
fig.update_layout(title="Consommation elec",
                  xaxis_title="time", yaxis_title="consommation kwh",
                  template="simple_white")
fig.show()


#%% Transformée de fourrier
from numpy import fft
import numpy as np
df = df.fillna(1e-6)
df['log10'] = df.consommation_kwh.apply(lambda x: np.log10(x)).astype(float)

X = fft.fft(df.consommation_kwh)
