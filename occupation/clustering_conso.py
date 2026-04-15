# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 17:36:30 2026

@author: Myriam Lopez
"""
from dotenv import load_dotenv
import urllib.parse
import os
import clickhouse_connect
import pandas as pd
import numpy as np
from tqdm import tqdm
import param as p
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm # to build a LOWESS model
from tslearn.metrics import cdist_dtw
from sklearn.cluster import AgglomerativeClustering

#%% functions
def db_connect(type='prod'):
    load_dotenv()
    username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
    host = ''
    password = ''
    if type=='prod':
        host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
        password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
    elif type == 'stage':
        host = urllib.parse.quote_plus(os.environ.get("clk_stg"))
        password = urllib.parse.quote_plus(os.environ.get('clk_stg_pass'))
    # si host = password = '' : lever une erreur
    conn = clickhouse_connect.get_client(host=host, username=username, port=443, password=password)
    return conn

def serie_typology(df, tol=0.1):
    score = np.mean((df.shift() - df) < 0) # combien de fois la valeur décrémente
    if score > tol:
        return 0 # volume
    else:
        return 1 # index

def plot_serie(x, y, dev):
    pink = '#FC006D'
    fig = go.Figure()
    fig.add_trace(go.Scatter(name = "occupancy", x=x, y=y,
                             mode="lines", line={'color': pink, 'width': 1}))
    fig.update_layout(title=f'Power week type, for device {dev}',
                      xaxis_title="Time", yaxis_title="Power week type",
                      template="simple_white")
    fig.show()

def encoding_cylical_ft(df):
    # From our earlier section "Extracting Datetime Components"
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    # Day of week has a cycle of 7 days
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    # Month has a cycle of 24 months
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    # Month has a cycle of 12 months
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

def detect_outliers_iqr(data, k = 100):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    if iqr <1:
        iqr = 10
    cutoff = iqr * k
    thres = q75 + cutoff
    num_outliers = np.sum(data > thres)
    return num_outliers
#%%
start = "2026-01-01 00:00:00"
end = "2026-06-31 23:59:59"
db_stage = db_connect(type='stage')
client = db_connect(type='prod')
#%% Liste des équipements déjà traités pour l'occupation
sample = db_stage.query_df(f"""
            SELECT distinct device_id, sensor_id, data_source
            FROM Myriam_test.dataset_occupancy
            WHERE 1=1
            settings max_rows_to_read = {p.maxi};""")
sample['y_column'] = sample.data_source.apply(lambda x: x[0].split('.')[-1])
#%% Collecte des données brutes et application du lissage de type LOESS
x_time = 'payload_expected_time'
X = []
x_info = []
for i, row in tqdm(sample[:].iterrows()):
        df = client.query_df(f"""
            SELECT device_id, {x_time}, {row.y_column}
            FROM {p.datasource}
            WHERE 1=1
            AND {x_time} between '{start}' and '{end}'
            AND device_id = '{row.device_id}'
            AND sensor_id = '{row.sensor_id}'
            ORDER BY {x_time}
            settings max_rows_to_read = {p.maxi};""")

        #moyenne glissante heure et jour
        df = encoding_cylical_ft(df.set_index('payload_expected_time'))
        df['time_seas'] = df.day_of_week + df.hour / 24
        df.loc[df[row.y_column] < 0, row.y_column] = 0
        scaler = MinMaxScaler(feature_range=(0, 1))
        df.dropna(inplace=True)
        df['scaled'] = scaler.fit_transform(pd.DataFrame(df[[row.y_column]])).flatten()

        lowess = sm.nonparametric.lowess
        # Generate y_hat values using lowess, try a couple values for hyperparameters
        y_hat = lowess(df['scaled'], df['time_seas'], frac=1/12)
        y_hat = pd.DataFrame(y_hat).groupby(0).mean()
        y_hat = MinMaxScaler(feature_range=(0, 1)).fit_transform(y_hat[[1]]).flatten()
        X.append(y_hat)
        x_info.append(row)
#%%
x_info= pd.DataFrame(x_info)
D = cdist_dtw(X)
model = AgglomerativeClustering(
    n_clusters=3,
    metric="cityblock",
    linkage="average"
)
x_info['label'] = model.fit_predict(X)
colors = {0:'#FC006D', 1:'#006CED', 2:'#2BD67C'}
for l, s in x_info.groupby('label'):
    fig = go.Figure()
    for r, row in s.iterrows():
        x = [i for i in range(len(X[r]))]
        y = X[r]
        fig.add_trace(go.Scatter(name = f"{row.device_id}x=x, y=y,
                                 mode="lines", line={'color': colors[1], 'width': 0.5}))
        fig.update_layout(title=f'Clustering, profiles for cluster {l}',
                          xaxis_title="Time", yaxis_title="Power week type",
                          template="simple_white")
    fig.show()

#%% test décompo de la conso
import numpy as np
y = df['scaled']
n = len(y)
# FFT
fft_vals = np.fft.rfft(y)
power = np.abs(fft_vals)**2
freqs = np.fft.rfftfreq(n)
peaks, _ = find_peaks(power, height=np.mean(power))
periods = 1 / freqs[peaks]


#%%
# Plot the distances
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
labels = ['profile 0', 'profile 1', 'profile 2'] #, 'profile 3', 'profile 4']
metric = 'cityblock'
n_clusters = 3
X = np.array(X)
avg_dist = np.zeros((n_clusters, n_clusters))
plt.figure(figsize=(5, 4.5))
for i in range(n_clusters):
    for j in range(n_clusters):
        avg_dist[i, j] = pairwise_distances(
            X[x_info.label == i], X[x_info.label == j], metric=metric
        ).mean()
avg_dist /= avg_dist.max()
for i in range(n_clusters):
    for j in range(n_clusters):
        t = plt.text(
            i,
            j,
            "%5.3f" % avg_dist[i, j],
            verticalalignment="center",
            horizontalalignment="center",
        )
        t.set_path_effects(
            [PathEffects.withStroke(linewidth=5, foreground="w", alpha=0.5)]
        )

    plt.imshow(avg_dist, interpolation="nearest", cmap="cividis", vmin=0)
    plt.xticks(range(n_clusters), labels, rotation=45)
    plt.yticks(range(n_clusters), labels)
    plt.colorbar()
    plt.suptitle("Interclass %s distances" % metric, size=18, y=1)
    plt.tight_layout()

