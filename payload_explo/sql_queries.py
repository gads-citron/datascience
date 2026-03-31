# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 10:00:11 2025

@author: Myriam Lopez
"""
import plotly.graph_objects as go
import plotly.io as pio
from dotenv import load_dotenv
import urllib.parse
import os

orange = '#FF670F'
black = '#2B3034'
pio.renderers.default = 'png'

table='datawarehouse_prod.fact_payload' 
device="'6042702a83c2f34ea421ffbb'"
sort1 = 'payload_timestamp'
sort2= 'payload_metric_id'
maxi = 10000000000000
limit= (0,1000)

#%%
load_dotenv()
host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))

client = clickhouse_connect.get_client(host=host, username=username,
                                       port = 443, password=password)
# nb clients 
df = client.query_df(f'SELECT count(distinct(entity_id)) FROM {table} \
             settings max_rows_to_read = 10000000000000')
print(df)

# nb devices
df = client.query_df(f'SELECT count(distinct(device_id)) FROM {table} \
             settings max_rows_to_read = 10000000000000')
print(df)

#nb payloads / devices
df = client.query_df(f'SELECT device_id, count(payload_id) nb FROM {table} \
                     group by device_id  \
                         order by nb settings max_rows_to_read = 10000000000000')
                         
med = df[(df.nb < 1012113) & (df.nb >0)].nb.median()
fig = go.Figure()
fig.add_trace(
    go.Histogram(bingroup=10e6,
        x = df[(df.nb>0) & (df.nb < med)].nb,
        marker={'color': orange}
    )
)
fig.update_layout(title="Nombre de payloads par device",
                  xaxis_title="Quantité totale de payloads", 
                  yaxis_title="Nombre de devices",
                  font_family="Arial",
                    font_color=black,
                    title_font_family="Times New Roman",
                    title_font_color=black,
                  template="simple_white")
fig.show()

q3 = df[(df.nb < 1012113) & (df.nb >0)].nb.quantile(0.75)
fig = go.Figure()
fig.add_trace(
    go.Histogram(bingroup=10e6,
        x = df[(df.nb>med) & (df.nb < q3)].nb,
        marker={'color': orange}
    )
)
fig.update_layout(title="Nombre de payloads par device",
                  xaxis_title="Quantité totale de payloads", 
                  yaxis_title="Nombre de devices",
                  font_family="Arial",
                    font_color=black,
                    title_font_family="Times New Roman",
                    title_font_color=black,
                  template="simple_white")
fig.show()

q3 = df[(df.nb < 1012113) & (df.nb >0)].nb.quantile(0.75)
fig = go.Figure()
fig.add_trace(
    go.Histogram(bingroup=10e6,
        x = df[(df.nb>med) & (df.nb < q3)].nb,
        marker={'color': orange}
    )
)
fig.update_layout(title="Nombre de payloads par device",
                  xaxis_title="Quantité totale de payloads", 
                  yaxis_title="Nombre de devices",
                  font_family="Arial",
                    font_color=black,
                    title_font_family="Times New Roman",
                    title_font_color=black,
                  template="simple_white")
fig.show()