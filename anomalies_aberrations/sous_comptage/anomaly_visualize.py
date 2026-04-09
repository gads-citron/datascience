from dotenv import load_dotenv
import urllib.parse
import os
import clickhouse_connect
from tqdm import tqdm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import citron_colors as col
from sklearn.preprocessing import MinMaxScaler
#%%
sous_comptage = 'datamart_sous_comptage.distributed_dataset_hourly_payloads_merged'
#%%
load_dotenv()
host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
client = clickhouse_connect.get_client(host=host, username=username,
                                       port=443, password=password)

#%%
def plot_conso(df, dev, conso):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.payload_expected_time, y=df[conso],
                             mode="lines", line={'color': col.orange, 'width': 1}))
    fig.update_layout(title=f'device_id: {dev}',
                      xaxis_title="Time", yaxis_title=conso, template="simple_white")
    fig.update_xaxes(ticks='outside', dtick=str(24*60*60*1000),
                     tickangle=-45, tickformat='%a-%b-%e')
    fig.show()

def serie_typology(df, tol=0.1):
    score = np.mean((df.shift() - df) < 0)  # combien de fois la valeur décrémente
    if score > tol:
        return 0  # volume
    else:
        return 1  # index
#%% liste des anomalies
start = "2025-01-01 00:00:00"
# start = "2025-06-30 23:59:00"
end = "2025-06-31 00:00:00"
anom = pd.read_csv('./data/anomalies_2025_sem1.csv')
anom = anom[anom.spikes>0]
df_analyse = []
cpt = 0
for row in tqdm(anom[:10].itertuples(index=False)):
    smp = client.query_df(f"""
                SELECT device_id, payload_expected_time, entity, eval_prev_hour, sum_payload_value_hour_curr
                FROM {sous_comptage}
                WHERE 1=1
                    AND payload_expected_time between '{start}' and '{end}'
                    AND month = '{row.month}'
                    AND sensor_name = '{row.sensor_name.replace("'", "''")}'
                    AND device_name = '{row.device_name.replace("'", "''")}'
                ORDER BY payload_expected_time
                LIMIT 100000""")
    if row.incremental_data:
        y_col = 'eval_prev_hour'
    else :
        y_col = 'sum_payload_value_hour_curr'
    smp[y_col] = smp[y_col].shift(1)
    smp['val_diff'] = smp[y_col].median()/smp[y_col]
    smp['smp_idx'] = cpt
    cpt += 1
    plot_conso(smp[~smp[y_col].isna()], row.device_name, y_col)
    df_analyse.append(smp)
df_analyse = pd.concat(df_analyse)
df_analyse.dropna(inplace=True)