from tqdm import tqdm
import os
import pandas as pd
import plotly.graph_objects as go
import citron_colors as col
from db_connector import ClickHouseClient
#%%
sous_comptage = 'datamart_sous_comptage.distributed_dataset_hourly_payloads_merged'
client = ClickHouseClient()
start = "2025-01-01 00:00:00"
end = "2025-06-30 00:00:00"
#%%
def plot_conso(df, dev, conso, anom_iqr, anom_isof, anom_stl, save_path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.payload_expected_time, y=df[conso],
                             mode="lines", line={'color': col.orange, 'width': 1}))
    fig.add_trace(go.Scatter(name = 'IQR', x=anom_iqr.payload_expected_time, y=anom_iqr['value'],
                                 mode="markers", marker={'color': 'black', 'size': 5, 'symbol': 'circle'},
                                 showlegend=True))
    fig.add_trace(go.Scatter(name='Isolation forest', x=anom_isof.payload_expected_time, y=anom_isof['value'],
                                 mode="markers", marker={'color': 'red', 'size': 5, 'symbol': 'circle'},
                                 showlegend=True))
    fig.add_trace(go.Scatter(name='stl', x=anom_stl.payload_expected_time, y=anom_stl['value'],
                                 mode="markers", marker={'color': 'blue', 'size': 5, 'symbol': 'circle'},
                                 showlegend=True))
    fig.update_layout(title=f'device_id: {dev}',
                      xaxis_title="Time", yaxis_title=conso, template="simple_white")
    fig.update_xaxes(ticks='outside', dtick=str(24*60*60*1000),
                     tickangle=-45, tickformat='%a-%b-%e')
    fig.show()

y_col = 'sum_payload_value_hour_curr'
#%% liste des anomalies
input_folder = './data/sample_anomalies_2025_sem_1'
anomalies = pd.read_csv(f'{input_folder}/method_comparison.csv')

os.makedirs(f'{input_folder}/plot', exist_ok=True)

for i, g in tqdm(anomalies.groupby(['device_id', 'sensor_id'])):
    y_col = g.feature.values[0]
    # start = pd.to_datetime(g.payload_expected_time.min()) - pd.Timedelta(hours=15*24)
    # end = pd.to_datetime(g.payload_expected_time.max()) + pd.Timedelta(hours=15*24)
    smp = client.query(f"""
        SELECT device_id, payload_expected_time, entity, eval_prev_hour, sum_payload_value_hour_curr
        FROM {sous_comptage}
        WHERE 1=1
            AND payload_expected_time between '{start}' and '{end}'
            AND sensor_id = '{i[1]}'
            AND device_id = '{i[0]}'
        ORDER BY payload_expected_time
        LIMIT 100000""")
    save_path = f"{input_folder}/plot/{g.customer_account_name.values[0]}_{i[0]}.png"
    plot_conso(smp, g.device_name.values[0], y_col, g[g.iqr], g[g.isof], g[g.stl], save_path)