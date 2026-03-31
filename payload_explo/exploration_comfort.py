# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 12:00:48 2025

@author: Myriam Lopez
"""
from dotenv import load_dotenv
import urllib.parse
import os
import clickhouse_connect
import pandas as pd

#tables
grdf_all='datamart_grdf.dataset_grdf_all'
releves='datawarehouse_prod.fact_payload'
clients='datawarehouse_prod.dim_entity'
equipements='datawarehouse_prod.dim_device'
capteurs='datawarehouse_prod.dim_sensor'
patrimoine='datawarehouse_prod.device_patrimonial_data'
group = 'datawarehouse_prod.dim_group'
elevator = 'datamart_movement.elevator_movement'

maxi = 10000000000000

load_dotenv()
host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
client = clickhouse_connect.get_client(host=host, username=username, 
                                       port = 443, password=password)
#%%
def segment_frequence(sample):
    max_val = sample.avgDiff.max()
    tickvals = [-float('inf'), 1, 60, 
                60*5, 60*10, 60*30, 60*45,
                3600, 3600*6, 3600*12, 
                86400, max_val]
    ticktext = ['< 1s', '< 1mn', '(1mn - 5mn]', '(5mn - 10mn]', '(10mn - 30mn]', 
               '(30mn - 45mn]', '(45mn - 1h]', '(1h - 6h]', '(6h - 12h]', 
               '(12h - 1jour]', '> 1jour']
    sample["interval"] = pd.cut(sample["avgDiff"], bins=tickvals, labels=ticktext)
    return sample
#%% table des clients
entity_df = client.query_df(f"""
            SELECT entity_id, entity_name
            FROM {clients}
            settings max_rows_to_read={maxi};""")
#%% Equipements sonde & sonde d'ambiance
dev_sonde = client.query_df(f"""SELECT distinct device_type, parent, device_id, device_name, iot_entity
                            FROM datawarehouse_prod.dim_device
                            WHERE 1=1
                            AND lower(device_type) IN 
                            ('sonde', 'sonde d''environnement', 'sondes d''ambiance', 'sonde d''ambiance')
                            OR lower(device_name) IN 
                            ('sonde', 'sonde d''environnement', 'sondes d''ambiance', 'sonde d''ambiance')
                            SETTINGS max_rows_to_read={maxi};""")
# Pour chercher la liste des capteurs liés aux équipements sonde
sensor_list = []
j = 0
i = 1000
while i < dev_sonde.shape[0]:
    sensor_list.append(
        client.query_df(f""" 
            SELECT distinct(payload_metric_id)
            FROM {releves}
            WHERE device_id IN {dev_sonde.device_id.to_list()[j:i]}
            SETTINGS max_rows_to_read={maxi};"""))
    j = i
    i += 1000
sensor_list = pd.concat(sensor_list)
sensor_list.drop_duplicates(inplace=True)
sensors_list = client.query_df(f""" 
    SELECT sensor_id, name, description
    FROM {capteurs}
    WHERE sensor_id IN {sensor_list.payload_metric_id.to_list()}
    SETTINGS max_rows_to_read={maxi};""")
    
# C02 (C° en Co2 et C02 en ppm): 
co2 = ['61e527f8109d4717ee2a3b9a', '662bc1ccd146fde77f9fc0a1']
humidity= '59c3bc64bee9f849f51461bb'
# pression
pression = ['5d1266e603635902ca74a434', '662bc1ccd146fde77f9fc08a']
# luminosité
luminosite = ['5fb6ab82451d617ac7ebd7f6', '662bc1cc5168048d008502ee']
#qualité de l'air interieur
QAI = '6230aa116b61bf66b7a8ce9f'
# temperature
temperature=['59c3b4b02e616c0008afdc8d', '59c3be80bee9f849f51461d4']
#%% Equipements ascenseur
dev_asc = client.query_df(f"""
            SELECT distinct device_type, parent, device_id, device_name, iot_entity
            FROM {equipements}
            WHERE match (device_type, '(?i).*ascens.*')
                OR match (device_type, '(?i).*epmr.*')
            settings max_rows_to_read={maxi};""")
dev_asc['Feature'] = 'Ascenseur'
#%% Equipements température
dev_temp = client.query_df(f"""
            SELECT distinct device_type, parent, device_id, device_name, iot_entity
            FROM {equipements}
            WHERE match (device_type, '(?i).*temp.*') 
            	AND not match (device_name, '(?i).*ext.*') 
            	AND not match (device_type, '(?i).*ecs.*') 
            	AND not match (device_type, '(?i).*chauf.*')
            settings max_rows_to_read={maxi};""")
dev_temp['Feature'] = 'Temperature'
#%% Sondes capable de capter le C02
c02_dev_list = client.query_df(f""" 
            SELECT distinct(device_id)
            FROM {releves}
            WHERE payload_metric_id IN {co2}
            SETTINGS max_rows_to_read={maxi};""")
# ajout des equipements associés aux sondes explicites Co2
df = client.query_df(f"""SELECT distinct device_type, parent, device_id, device_name, iot_entity
                                FROM datawarehouse_prod.dim_device
                                WHERE match (device_name, '(?i).*co2.*') AND match (device_type, '(?i).*sonde.*')
                                	OR match (device_type, '(?i).*co2.*')
                                SETTINGS max_rows_to_read={maxi};""")
c02_dev_list = pd.concat([c02_dev_list, df.device_id]).drop_duplicates()

# Ajout des informations sur les devices
dev_CO2 = client.query_df(f"""SELECT device_type, parent, device_id, device_name, iot_entity
                                FROM datawarehouse_prod.dim_device
                                WHERE device_id IN {c02_dev_list.device_id.to_list()}
                                SETTINGS max_rows_to_read={maxi};""")
dev_CO2['Feature'] = 'CO2'
#%% Equipements qui pourraient récupérer du bruit
# pas de capteur qui se rapportent au bruit, il faut se concentrer uniquement sur les équipements suivants
dev_bruit = client.query_df(f"""SELECT distinct device_type, parent, device_id, device_name, iot_entity
                                FROM datawarehouse_prod.dim_device
                                WHERE match (device_name, '(?i).*bruit.*') AND match (device_type, '(?i).*sonde.*')
                                	OR match (device_type, '(?i).*bruit.*')
                                SETTINGS max_rows_to_read={maxi};""")
dev_bruit['Feature'] = 'Bruit'
#%% Equipements qui récupèrent de l'électricité
dev_elec = client.query_df(f"""SELECT distinct device_type, parent, device_id, device_name, iot_entity
                                FROM {equipements}
                                WHERE match (device_type, '(?i).*lectr.*')
                                SETTINGS max_rows_to_read={maxi};""")
dev_elec['Feature'] = 'elec'
#%% Equipements qui récupèrent de l'eau
dev_water = client.query_df(f"""SELECT distinct device_type, parent, device_id, device_name, iot_entity
                                FROM datawarehouse_prod.dim_device
                                WHERE match (device_type, '(?i).*eau.*')
                                SETTINGS max_rows_to_read={maxi};""")
dev_water['Feature'] = 'Eau'
#%%
all_features = pd.concat([dev_elec, dev_water, dev_CO2, dev_temp, dev_bruit, dev_asc])
synth_table = all_features.groupby(['iot_entity', 'Feature'], as_index=False).device_id.count()
synth_table = synth_table.merge(entity_df, right_on = "entity_id", left_on="iot_entity").drop(['iot_entity', 'entity_id'], axis=1)
synth_table = synth_table.pivot_table(columns = 'Feature', values = 'device_id', index = ['entity_name']).reset_index()

df_clean = synth_table.replace('', pd.NA).notna().astype(int)
synth_table['combo'] = df_clean[['elec', 'Eau', 'CO2', 'Bruit', 'Temperature', 'Ascenseur']].apply(
    lambda row: '+'.join(df_clean[['elec', 'Eau', 'CO2', 'Bruit', 'Temperature', 'Ascenseur']].columns[row.eq(1)]), axis=1
)
#nombre de devices par feature
synth_table.groupby('combo').count()
#%% Ajout des infos sur les bâtiments
from tqdm import tqdm
synth_patrimoine = []
for i in tqdm(range(1,87000, 1000)):
    df = client.query_df(f"""SELECT group_id, site_typology, site_nature, id_du_site
                                FROM {patrimoine}
                                WHERE group_id IN {synth_table.parent.to_list()[i-1000:i]}
                                SETTINGS max_rows_to_read={maxi};""")
    synth_patrimoine.append(df)
#%% ajout infos bati
bat_id = client.query_df(f""" WITH (
                                SELECT groupArray(group_id)
                                FROM datawarehouse_prod.dim_group
                                WHERE group_genre IN ['Batiment']
                            ) AS batiment_ids
                        SELECT group_id, device_id, site_nature, site_typology, organization_group_genre
                        FROM datawarehouse_prod.device_big_table d
                        WHERE  1=1
                        AND arrayExists(
                            x -> x IS NOT NULL 
                                AND has(batiment_ids,x),
                            d.parents_inc_self)
                                    SETTINGS max_rows_to_read={maxi};""")
#%%merge_info and synthese
device_batiment = all_features.merge(bat_id)
device_batiment.to_csv('./data/list_equipement_batiment.csv')
batiment_synthese = device_batiment.groupby(['iot_entity', 'group_id', 'Feature'],
                                            group_keys=False)[['device_id']].count()\
    .pivot_table(columns='Feature', values='device_id', index=['iot_entity', 'group_id'])
#%% Jeu de données eau + electricité
sample = pd.concat([dev_water, dev_elec])
candidates = synth_table[~(synth_table.elec.isna())&~(synth_table.Eau.isna())]
dev_data = sample[sample.parent.isin(candidates.parent)]
# suppression du bâtiment thibaut test qui sert à rien 
dev_data = dev_data[~(dev_data.iot_entity == "665f0e73d89313291178a17e")]
#%% définition des données sur le temps
candidate_histo = []
for bat, s in dev_data.groupby('iot_entity'):    
    histo = client.query_df(f""" 
                SELECT device_id, min(payload_timestamp) as start, max(payload_timestamp) as end,
                count(payload_timestamp) nbPay
                FROM {releves}
                WHERE device_id IN {s.device_id.to_list()}
                GROUP BY device_id
                SETTINGS max_rows_to_read={maxi};""")
    if not histo.empty:
        histo['iot_entity'] = bat
        histo['parent'] = s.parent.values[0]
        candidate_histo.append(histo)
candidate_histo = pd.concat(candidate_histo)
#%% frequence theorique des équipements 
candidate_freq = []
for bat, s in dev_data.groupby('iot_entity'):
    freq = client.query_df(f"""
    SELECT p.device_id, arrayReduce('avg', diffs) AS avgDiff 
        FROM( 
            SELECT 
                p.device_id,
                arrayDifference(arraySort(groupArray(p.payload_timestamp))) AS diffs
            FROM {releves} p        
            WHERE device_id IN {s.device_id.to_list()}
        GROUP BY 
            p.device_id) p
        ORDER BY avgDiff DESC
        settings max_rows_to_read = {maxi}""")
    if not freq.empty:
        freq['parent'] = s.parent.values[0]
        freq['iot_entity'] = bat
        freq['Feature'] = s.Feature.values[0]
        candidate_freq.append(freq)
candidate_freq = pd.concat(candidate_freq)
candidate_freq = segment_frequence(candidate_freq)
#%%
candidate_sample = candidate_freq.merge(candidate_histo, on = ['device_id','parent', 'iot_entity'])

#stats à l'échelle du bâtiment
candidate_bat = candidate_sample.groupby(['iot_entity', 'parent'], as_index=False).agg({'avgDiff':'mean', 'start':'max', 'end':'min'})
candidate_bat = segment_frequence(candidate_bat)

candidate_bat = candidate_bat.merge(entity_df[['entity_id', 'entity_name']], right_on='entity_id', left_on='iot_entity')
candidate_bat = candidate_bat[['entity_name', 'parent', 'start', 'end', 'interval']]

# stats à l'échelle du client
candidate_ent = candidate_bat.groupby('iot_entity', as_index=False).agg({'avgDiff':'mean', 'start':'max', 'end':'min'})
candidate_ent = segment_frequence(candidate_ent)

candidate_ent = candidate_ent.merge(entity_df[['entity_id', 'entity_name']], right_on='entity_id', left_on='iot_entity')

#%%
patrimonial_data = client.query_df(f"""
            SELECT organization_group_id, organization_group_genre, site_typology, site_nature, sub_nature, group_id
            FROM {patrimoine}
            WHERE device_id IN {candidate_sample.device_id.to_list()}
            settings max_rows_to_read={maxi};""")
            
group_df = client.query_df(f"""
            SELECT group_id, group_genre
            FROM {group}
            WHERE group_id IN {patrimonial_data.group_id.to_list()}
            settings max_rows_to_read={maxi};""")
patrimonial_data = group_df.merge(patrimonial_data, on = 'group_id')


#%% Récupération données ascenseur
df_asc = []
for i in range(100, 17000, 100):
    df = client.query_df(f"""
            SELECT device_id, min(datetime) start, max(datetime) end, count(cab_movements_total) nb_mouv
            FROM {elevator}
            WHERE 1= 1
                AND device_id IN {dev_asc.device_id.drop_duplicates().to_list()[i-100:i]}
            GROUP BY device_id
            SETTINGS max_rows_to_read={maxi};""")
    if not df.empty :
        df_asc.append(df)
