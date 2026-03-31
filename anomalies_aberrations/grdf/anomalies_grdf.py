# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:02:25 2025

@author: Myriam Lopez
"""
import numpy as np
from dotenv import load_dotenv
import urllib.parse
import os
import clickhouse_connect
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
from plotly import io as pio
pio.renderers.default='svg'
#%%
grdf_anomalies = 'datamart_grdf_dev_mtahirintsoa.dataset_grdf_all'
grdf_all='datamart_grdf.dataset_grdf_all'
# Capteurs de consommation
V_reel_info = '64f9dc4e964b1c6b3092aebe'
V_brut_info = '64f9dbc06bd36472e445b9c7'
date_fin_info = 'INFORM_CONSUMPTION_END_DATE'
date_debut_info = 'INFORM_CONSUMPTION_START_DATE'

coef_PTA_info = '64f9dd1fe1aa566bd5c539c8'
coef_PCS_info = '64f9dd576bd364b24d45b9c8'
coef_info = '64f9dd87e1aa563305c539c9'

maxi = 10000000000000
raison = '64f9db0d2ade76c96cb1e41c'

list_pb = ['64a432d810de337707f207a5', '64a435e010de33bbe3f20d91', '65d72c7244b489e0a9724cd0']
#%%
def detect_outliers_iqr(data, k = 1000):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    if iqr <1:
        iqr = 100
    cutoff = iqr * k
    thres = q75 + cutoff
    return (data > thres, thres)
#%% clickhouse env load
load_dotenv()
host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
client_prd = clickhouse_connect.get_client(host=host, username=username,
                                       port = 443, password=password)

#%%
load_dotenv()
host = urllib.parse.quote_plus(os.environ.get("clk_stg"))
username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
password = urllib.parse.quote_plus(os.environ.get('clk_stg_pass'))
client_stage = clickhouse_connect.get_client(host=host, username=username,
                                       port = 443, password=password)
#%%
dev_freq = client.query_df(f"""
        SELECT distinct entity_name, device_id, grdf_frequency as frequency
        FROM {grdf_all}
        settings max_rows_to_read = {maxi}""")
arr = dev_freq.index.to_list()
np.random.shuffle(arr)
#%%par année en rajoutant un minimum seuil pour supprimer les valeurs basses
q1 = 0.1
anom_year = []
for dev in dev_freq.loc[arr[:200]].device_id:
# for dev in list_pb:
    df = client.query_df(f"""SELECT device_id, consommation_kwh_informative,
                         payload_time
    FROM datamart_grdf_test_bo.dataset_grdf_all
    WHERE device_id = '{dev}'
    	AND toYear(payload_time) = 2025
    ORDER BY date_debut_consommation_informatif
    SETTINGS max_rows_to_read={maxi};""")
    if df.empty:
        continue
    min_val = client.query_df(f"""SELECT quantileExact({q1})(consommation_kwh_informative) as q1
            FROM datamart_grdf_test_bo.dataset_grdf_all
            WHERE device_id = '{dev}'
            AND toYear(payload_time) = 2025
            settings max_rows_to_read=10000000000000;""").values[0][0]
    thres = client.query_df(f"""SELECT quantileExact(0.75)(consommation_kwh_informative) + (quantileExact(0.75)(consommation_kwh_informative) -  quantileExact(0.25)(consommation_kwh_informative))*100 as thres
            FROM datamart_grdf_test_bo.dataset_grdf_all
            WHERE device_id = '{dev}'
            AND toYear(payload_time) = 2025
                AND consommation_kwh_informative >= {min_val}
            settings max_rows_to_read=10000000000000;""").values[0][0]
     
    df["anomaly"] = df.consommation_kwh_informative.values > thres
    anom_year.append(df)
anom_year = pd.concat(anom_year)

anom_year.groupby('device_id').anomaly.sum().sort_values()
anom_year.resample('1M', on='payload_time').anomaly.sum()

#%% par mois
scores_dev = []
score_month = []
for k in [100, 1000, 1e4, 1e5]:
    anom_month = []
    # for dev in dev_freq.loc[arr[:200]].device_id:
    for dev in list_pb:
        for m in [1,2,3,4,10,11,12]:
            df = client.query_df(f"""SELECT device_id, payload_time, consommation_kwh_informative, 
                                 risque_aberration_informatif, threshold, iqr
            FROM datamart_grdf_test_bo.dataset_grdf_all
            WHERE device_id = '{dev}'
            	AND toYear(payload_time) = 2024 AND toMonth(payload_time) = '{m}'
            ORDER BY date_debut_consommation_informatif
            SETTINGS max_rows_to_read={maxi};""")
            if df.empty:
                continue    
            thres = client.query_df(f"""SELECT quantileExact(0.75)(consommation_kwh_informative) + 
                                    (quantileExact(0.75)(consommation_kwh_informative) -  
                                         quantileExact(0.25)(consommation_kwh_informative))*{k} as thres
                    FROM datamart_grdf_test_bo.dataset_grdf_all
                    WHERE device_id = '{dev}'
                    AND toYear(payload_time) = 2025 AND toMonth(payload_time) = '{m}'
                    settings max_rows_to_read=10000000000000;""").values[0][0]                
            df["anomaly"] = df.consommation_kwh_informative.values > thres
            anom_month.append(df)
    anom_month = pd.concat(anom_month)
    perf_dev = anom_month.groupby('device_id').anomaly.sum().sort_values()
    perf_dev = pd.DataFrame(perf_dev)
    perf_dev['k_param'] = k
    scores_dev.append(perf_dev)
    perf_month = anom_month.resample('1M', on='payload_time').anomaly.sum()
    perf_month = pd.DataFrame(perf_month)
    perf_month['k_param'] = k
    score_month.append(perf_month)

#%% par année glissante
anom_sliding_year = []
for dev in dev_freq.loc[arr[:200]].device_id:
# for dev in list_pb:
    df = client.query_df(f"""SELECT device_id, payload_time, consommation_kwh_informative, 
    (quantileExact(0.75)(consommation_kwh_informative) 
        OVER (
            ORDER BY payload_time
            RANGE BETWEEN 31536000 PRECEDING AND CURRENT ROW
            ) + 
        ( quantileExact(0.75)(consommation_kwh_informative) 
            OVER (
                ORDER BY payload_time
                RANGE BETWEEN 31536000 PRECEDING AND CURRENT ROW
                ) -
            quantileExact(0.25)(consommation_kwh_informative) 
            OVER (
                    ORDER BY payload_time
                    RANGE BETWEEN 31536000 PRECEDING AND CURRENT ROW
                ))
        ) * 100 AS thres
        FROM datamart_grdf_test_bo.dataset_grdf_all
        WHERE device_id = '{dev}'
            AND toYear(payload_time) = 2025
        ORDER BY payload_time        
        SETTINGS max_rows_to_read={maxi};""")
    if df.empty:
        continue
    df["anomaly"] = df.dropna().consommation_kwh_informative > df.dropna().thres
    anom_sliding_year.append(df)
anom_sliding_year = pd.concat(anom_sliding_year)
anom_sliding_year.groupby('device_id').anomaly.sum().sort_values()
anom_sliding_year.resample('1M', on='payload_time').anomaly.sum()

#%%#%% par année glissante où les valeurs sont cappée en partie basse par le quantile 0.1
q = 1e-6
anom_sliding_year_cap = []
for dev in dev_freq.loc[arr[:200]].device_id:
# for dev in list_pb:
    df = client.query_df(f"""SELECT device_id, payload_time, consommation_kwh_informative, q1, 
        (quantileExactIf(0.75)(consommation_kwh_informative, consommation_kwh_informative >= q1) 
            OVER (
                ORDER BY payload_time
                RANGE BETWEEN 31536000 PRECEDING AND CURRENT ROW
                ) + 
            ( quantileExactIf(0.75)(consommation_kwh_informative, consommation_kwh_informative >= q1) 
                OVER (
                    ORDER BY payload_time
                    RANGE BETWEEN 31536000 PRECEDING AND CURRENT ROW
                    ) -
                quantileExactIf(0.25)(consommation_kwh_informative, consommation_kwh_informative >= q1) 
                OVER (
                        ORDER BY payload_time
                        RANGE BETWEEN 31536000 PRECEDING AND CURRENT ROW
                    ))
            ) * 100 AS thres
            FROM (
                SELECT
                    device_id, 
                    payload_time,
                    consommation_kwh_informative,
                    quantileExact({q})(consommation_kwh_informative) OVER (
                        ORDER BY payload_time
                        RANGE BETWEEN 31536000 PRECEDING AND CURRENT ROW
                    ) AS q1
                FROM datamart_grdf_test_bo.dataset_grdf_all
                WHERE device_id = '{dev}'
                    AND toYear(payload_time) = 2025
            )
            ORDER BY payload_time    
            SETTINGS max_rows_to_read={maxi};""")
    if df.empty:
        continue
    df["anomaly"] = df.dropna().consommation_kwh_informative > df.dropna().thres
    anom_sliding_year_cap.append(df)
anom_sliding_year_cap = pd.concat(anom_sliding_year_cap)
anom_sliding_year_cap.groupby('device_id').anomaly.sum().sort_values()
anom_sliding_year_cap.resample('1M', on='payload_time').anomaly.sum()
#%%#%% par année glissante où les valeurs sont cappée en partie basse par 0
anom_sliding_year_cap = []
k = 100
for dev in dev_freq.loc[arr[:200]].device_id:
# for dev in list_pb:
    min_date =  client.query_df(f"""SELECT min(payload_time)
        FROM datamart_grdf_test_bo.dataset_grdf_all
        WHERE device_id = '{dev}'
        settings max_rows_to_read={maxi};""")
    if min_date.values[0][0].date() > dt.datetime(2024,6,1).date() :
        continue
    df = client.query_df(f"""SELECT device_id, payload_time, consommation_kwh_informative,
        (quantileExact(0.75)(consommation_kwh_informative)
            OVER (
                ORDER BY payload_time
                RANGE BETWEEN 31536000 PRECEDING AND CURRENT ROW
                ) + 
            ( quantileExact(0.75)(consommation_kwh_informative) 
                OVER (
                    ORDER BY payload_time
                    RANGE BETWEEN 31536000 PRECEDING AND CURRENT ROW
                    ) -
                quantileExact(0.25)(consommation_kwh_informative) 
                OVER (
                        ORDER BY payload_time
                        RANGE BETWEEN 31536000 PRECEDING AND CURRENT ROW
                    ))
            ) * {k} AS thres
            FROM datamart_grdf_test_bo.dataset_grdf_all
            WHERE device_id = '{dev}'
                AND toYear(payload_time) = 2025
            ORDER BY payload_time
            SETTINGS max_rows_to_read={maxi};""")
    if df.empty:
        continue
    df["anomaly"] = df.dropna().consommation_kwh_informative > df.dropna().thres
    anom_sliding_year_cap.append(df)
anom_sliding_year_cap = pd.concat(anom_sliding_year_cap)
anom_sliding_year_cap.groupby('device_id').anomaly.sum().sort_values()
anom_sliding_year_cap.resample('1M', on='payload_time').anomaly.sum()

#%%
plot_flow(grdf_df[grdf_df.device_id==dev].index, grdf_df[grdf_df.device_id==dev][V_brut_info].astype(float), '', dev)

#%%
q = f"""WITH top_10_devices AS (
    SELECT
        device_name,
        sum(beyond_threshold_hiver) AS nb_depassement_total_hiver
    FROM {grdf_anomalies}
    WHERE threshold_hiver IS NOT NULL
    GROUP BY device_name
    ORDER BY nb_depassement_total_hiver DESC
    LIMIT 10
)
SELECT
    toYear(payload_date) AS annee,
    toMonth(payload_date) AS mois,
    d.device_name,
    t.nb_depassement_total_hiver,
    sum(consommation_kwh_informative > threshold_hiver) AS nb_depassements_reels,
    sum(consommation_kwh_informative > threshold_hiver) / toFloat64(count()) AS taux_depassements_reels,
    quantileExact(0.5)(consommation_kwh_informative) AS conso_mediane,
    quantileExact(0.9)(consommation_kwh_informative) AS conso_p90,
    quantileExact(0.5)(threshold_hiver) AS threshold_median,
    quantileExact(0.9)(threshold_hiver) AS threshold_p90,
    count() AS nb_obs
FROM {grdf_anomalies}
INNER JOIN top_10_devices t
    ON d.device_name = t.device_name
WHERE d.threshold_hiver IS NOT NULL
GROUP BY
    annee,
    mois,
    d.device_name,
    t.nb_depassement_total_hiver
ORDER BY
    t.nb_depassement_total_hiver DESC,
    d.device_name,
    annee DESC,
    mois DESC"""
    
#%%
top_10 = f"""SELECT device_name, sum(beyond_threshold_hiver) AS nb_depassement_total_hiver
        FROM {grdf_anomalies}
        WHERE threshold_hiver IS NOT NULL
        GROUP BY device_name
        ORDER BY nb_depassement_total_hiver DESC
        LIMIT 10"""
df = client.query_df(top_10)

values = client_stage.query_df(f"""SELECT device_name, beyond_threshold_hiver
        FROM {grdf_anomalies}
        WHERE threshold_hiver IS NOT NULL
        LIMIT 100""")
        
#%% Visualisation données brutes
dataset = client_stage.query_df(f"""SELECT device_name, payload_time, beyond_threshold_hiver, index_gaz_brut_m3
                                    FROM {grdf_anomalies}
                                    WHERE device_name = 'PCE_GAZ_GI153091'
                                    """)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=dataset.payload_timestamp,
        y=dataset.index_gaz_brut_m3, 
        mode='lines', 
        line={'color': red, 'width':1}, showlegend=False
    )
)

fig.show()

