# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 11:06:06 2025

@author: Myriam Lopez
"""

from dotenv import load_dotenv
import urllib.parse
import os
import clickhouse_connect
import pandas as pd
import plotly.graph_objects as go

from tqdm import tqdm

# variables
maxi = 10000000000000
dev_type_elec = ['PDL Électricité', 'Centrale de mesure électrique', 'Compteur Électricité', \
            'Compteur d\\''électricité', 'Compteur électricité', 'Compteur électrique', \
            'Compteurs d’électricité']

dev_type_eau = ['Compteur d\\''Eau Chaude', 'Compteur d\\''Eau Froide', "Compteur d'Eau", \
            'Compteur eau', 'Compteur d’eau', 'Compteur d’Eau', \
            'Compteur eau chaude', 'Compteur d\\''eau', 'Compteur eau surchauffé']

site_typo = ['Tertiaire', 'Bureaux', '']
site_nat = ['Bâtiment tertiaire', 'Commerce', 'Enseignement',\
            'Bureaux', 'Santé', 'Bureau', 'Sport', 'Culture et spectacles', '']
# a checker : -Compteur EREF -Compteur EG -Compteur TD
# tables

temperature = 'datamart_comfort.base_hourly_temperature'
sous_comptage = 'datamart_sous_comptage.distributed_dataset_hourly_payloads_merged'
device_sous_comptage = 'datamart_sous_comptage.device_properties'
# %% clickhouse env load
load_dotenv()
host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
client = clickhouse_connect.get_client(host=host, username=username,
                                       port=443, password=password)
#%% sélection des équipements liés à l'elec dans le sous comptage
devices_elec = client.query_df(f"""
    SELECT site_typology, site_nature, site_sub_nature, meter_type, customer_account_name, site_id, device_id
    FROM {device_sous_comptage}
    WHERE 1=1
     AND match (device_type, '(?i).*lectr.*')
     AND site_typology IN {site_typo}
     AND site_nature IN {site_nat}
    settings max_rows_to_read = 10000000000000;""")

devices_elec = devices_elec.drop_duplicates()
group = devices_elec.groupby(['site_typology', 'site_nature', 'customer_account_name'])\
        .agg({'device_id':'count', 'site_id':'nunique'})

#%% Liste des site/client/device par type de client (collectivite/fonciere/tertiaire)
devices_elec = pd.read_csv('./data/device_sous_comptage_elec.csv', sep=';')
devices_elec = devices_elec[~devices_elec.Typologie_client.isna()]
devices_elec = devices_elec.drop_duplicates()
#%% Historique données électricité
df = client.query_df(f""" 
    SELECT device_id, min(payload_expected_time) as start_elec
    FROM {sous_comptage}
    WHERE 1=1
     AND device_id IN {devices_elec.device_id.to_list()}
    GROUP BY device_id
    settings max_rows_to_read = 10000000000000;""")

devices_elec = devices_elec.merge(df[['device_id', 'start_elec']])

#%% Données d'eau
devices_eau = client.query_df(f""" 
    SELECT device_type, site_typology, site_nature, site_sub_nature, meter_type, customer_account_name, site_id, device_id
    FROM {device_sous_comptage}
    WHERE 1=1
     AND match (device_type, '(?i).*eau.*')
     AND site_typology IN {site_typo}
     AND site_nature IN {site_nat}
    settings max_rows_to_read = 10000000000000;""")
devices_eau = devices_eau.drop_duplicates()

#%%
df = client.query_df(f""" 
    SELECT device_id, min(payload_expected_time) as start_eau
    FROM {sous_comptage}
    WHERE 1=1
     AND device_id IN {devices_eau.device_id.to_list()}
    GROUP BY device_id
    settings max_rows_to_read = 10000000000000;""")

#%% lesquels sont en commun avec l'elec
site_fields = devices_elec[['site_typology', 'site_nature', 'site_sub_nature', 'customer_account_name', 'site_id', 'Typologie_client']]
devices_eau['feature'] = 'eau'
devices_elec['feature'] = 'elec'
devices_list = pd.concat([devices_eau[['device_id', 'feature', 'site_id', 'meter_type', 'customer_account_name']], devices_elec[['device_id', 'feature', 'site_id', 'meter_type', 'customer_account_name']]])
eau_elec = devices_list.pivot_table(index = 'site_id', columns = 'feature', values = 'device_id', aggfunc='count').dropna()

#%%
eau_elec = eau_elec.merge(site_fields.drop_duplicates(), on = 'site_id', how = 'inner')
#%% Faire un box plot par device sur les valeurs des derniers mois
def plot_conso(df, dev):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.payload_expected_time, y=df.eval_prev_hour,
                             mode="lines", line={'color': 'orange', 'width': 1}))
    fig.update_layout(title=f'device_id: {dev}',
                      xaxis_title="Time", yaxis_title="eval_prev_hour_converted",
                      template="simple_white")
    fig.update_xaxes(ticks='outside',
                     dtick=str(24*60*60*1000),
                     tickangle=-45,
                     tickformat='%a-%b-%e')
    fig.show()

history = "toDateTime('2026-01-01 01:00:00')"
site_id = '6037dce8f435564fd2458b91'
dev = '6042702983c2f34ea421ffb9'
duplique = []
discrete = []

for dev in devices_elec[devices_elec.site_id==site_id].device_id.to_list():
    df = client.query_df(f"""
            SELECT device_id, payload_expected_time, sum_payload_value_hour_curr, eval_prev_hour_converted
            FROM {sous_comptage}
            WHERE 1=1
             AND device_id = '{dev}'
             AND payload_expected_time > {history}
            ORDER BY payload_expected_time DESC
            settings max_rows_to_read = 10000000000000;""")
    if df.empty:
        continue
    # nombre de dupliqués
    duplicatas = df.duplicated().sum()
    duplique.append({'device_id': dev, 'duplique': duplicatas})
    # valeurs discrete
    discrete.append({'device_id': dev, 'nb_values': len(df.eval_prev_hour_converted.unique())})
    # plot_occ(df.drop_duplicates(), dev)

#%% Pour l'occupation
# j'extrait le profil moyen de consommation heure par heure pour chaque site.
# je vérifie que le profil est exploitable
# je compare mes données de consommation à ce profil moyen.
# là où l'écart est important correspond aux faux positifs
# je compare les autres méthodes en utilisant ce profil moyen

# identifier les jours exceptionnels :
# -> est-ce que le pattern de moindre consommation est présent sur toute la journée et uniquement une journée ?

