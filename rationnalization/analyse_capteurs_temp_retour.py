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
#%% Récupération infos du capteur
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

#%% statistiques par capteur
capteurs = temperature_retour.sensor_id.drop_duplicates().to_list()
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
#%% rassemblement
time_idx = mean_time.merge(is_index)
stat_dev = stat_dev.merge(time_idx, on=['payload_metric_id', 'device_id'])
#%% Clustering sur device, maximal et minimal
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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

def similarity_v1(dev, smp):
    sim = []
    for i in range (smp.shape[0]):
        for j in range(i+1, smp.shape[0]):
            X = smp.drop(['payload_metric_id', 'device_id'], axis=1).values
            r = X[i]/X[j]
            sim.append({'device': dev, 
                        'Sensor1':smp.payload_metric_id.values[i],
                        'Sensor2':smp.payload_metric_id.values[j], 
                        'similarite': np.std(r)})
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
        df = df.merge(temperature_retour[['sensor_id', 'sensor_name']].drop_duplicates(), 
                 right_on='sensor_id', left_on='payload_metric_id')
        df = df.pivot_table(index='payload_timestamp', columns='sensor_name', values = 'payVal')
        df = df.dropna()
        if df.empty:
            continue
        similarity = np.sum(df.iloc[:,0]==df.iloc[:, 1])/df.shape[0]
        exact_similarity.append({'device_id':dev, 'sensors':sens, 'score':similarity})
exact_similarity = pd.DataFrame(exact_similarity)

#%% capteurs imprécis
synthese = synthese[synthese.is_in_datamart==1]
synthese = synthese[synthese.end > dt.datetime(2026,1,1)]
a_reclasser = synthese[synthese.sensor_name.isin(['Température retour', 'Température retour ECS'])]
a_reclasser = a_reclasser[a_reclasser.end > dt.datetime(2026,1,1)]
#%%Regroupemet par nom d'équipement
import re
pattern = '((?<![A-Za-z])CTA(?![A-Za-z])|(?<![A-Za-z])ECS(?![A-Za-z])|(?<![A-Za-z])RCU(?![A-Za-z])|chaufferie|primaire|chauffage|echangeur|(?<![A-Za-z])CHA(?![A-Za-z]))'
a_reclasser['device_keyword'] = a_reclasser.device_name.str.findall(pattern, flags=re.IGNORECASE)
a_reclasser['device_keyword'] = a_reclasser.device_keyword.apply(
    lambda x: '.'.join([m.lower() for m in x]) if isinstance(x, list) else ''
)
a_reclasser["texte_total"] = a_reclasser[["device_keyword", "device_type"]].fillna("").agg(" ".join, axis=1)
#%% tentative de clustering
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

synthese["texte_total"] = synthese[["device_keyword", "device_type", "entity_name"]].fillna("").agg(" ".join, axis=1)
vectorizer = TfidfVectorizer(stop_words=["french"])
X = vectorizer.fit_transform(synthese["texte_total"])
kmeans = KMeans(n_clusters=7, random_state=42)
synthese["cluster"] = kmeans.fit_predict(X)
test_kmeans = synthese[['device_type', 'device_name', 'sensor_name', 'cluster']]
test_kmeans.groupby('sensor_name').cluster.mean()

#%% KNN
from sklearn.neighbors import KNeighborsClassifier
X_train = pd.read_csv('./kmean_7clusters_modifie.csv', sep=";")# les groupes faits avant grâce à Kmeans
synthese = synthese.merge(X_train, how='left')
synthese.loc[~synthese['Nouveau capteur'].isna(), 'sensor_name'] = \
    synthese[~synthese['Nouveau capteur'].isna()]['Nouveau capteur']
    
synthese = synthese.drop('Nouveau capteur', axis=1)
idx_pred = synthese[synthese.sensor_name.isin(['Température retour', 
                                               'Température retour ECS'])].index
idx_train = synthese[~synthese.sensor_name.isin(['Température retour', 
                                               'Température retour ECS'])].index
vectorizer = TfidfVectorizer(stop_words=["french"])
X = vectorizer.fit_transform(synthese["texte_total"])

y_train = synthese.loc[idx_train].sensor_name
X_train = X[idx_train]
X_pred = X[idx_pred]

knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
knn.fit(X_train, y_train)
synthese.loc[idx_pred, "new_sensor"] = knn.predict(X_pred)