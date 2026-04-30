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
def plot_conso(df, dev, conso, anom_iqr, anom_isof):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.payload_expected_time, y=df[conso],
                             mode="lines", line={'color': col.orange, 'width': 1}))
    fig.add_trace(go.Scatter(name = 'IQR', x=anom_iqr.payload_expected_time, y=anom_iqr['value'],
                                 mode="markers", marker={'color': 'black', 'size': 5, 'symbol': 'circle'},
                                 showlegend=True))
    fig.add_trace(go.Scatter(name='Isolation forest', x=anom_isof.payload_expected_time, y=anom_isof['value'],
                                 mode="markers", marker={'color': 'red', 'size': 5, 'symbol': 'circle'},
                                 showlegend=True))
    fig.update_layout(title=f'device_id: {dev}',
                      xaxis_title="Time", yaxis_title=conso, template="simple_white")
    fig.update_xaxes(ticks='outside', dtick=str(24*60*60*1000),
                     tickangle=-45, tickformat='%a-%b-%e')
    fig.show()

y_col = 'sum_payload_value_hour_curr'
#%% liste des anomalies
anomalies = pd.read_csv('./data/anomalies_comparison_2025_sem_1.csv')
# both = anomalies[anomalies.iqr  & anomalies.isof]
for i, g in anomalies.groupby(['device_id', 'sensor_id']):
    y_col = g.feature.values[0]
    start = pd.to_datetime(g.payload_expected_time.min()) - pd.Timedelta(hours=15*24)
    end = pd.to_datetime(g.payload_expected_time.max()) + pd.Timedelta(hours=15*24)
    smp = client.query(f"""
        SELECT device_id, payload_expected_time, entity, eval_prev_hour, sum_payload_value_hour_curr
        FROM {sous_comptage}
        WHERE 1=1
            AND payload_expected_time between '{start.strftime("%Y-%m-%d %H:%M:%S")}' and '{end.strftime("%Y-%m-%d %H:%M:%S")}'
            AND sensor_id = '{i[1].replace("'", "''")}'
            AND device_id = '{i[0].replace("'", "''")}'
        ORDER BY payload_expected_time
        LIMIT 100000""")
    plot_conso(smp[~smp[y_col].isna()], g.device_name.values[0], y_col, g[g.iqr], g[g.isof])

