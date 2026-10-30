"""
Fonctions de graphiques plotly du dossier supervised_detection : contrôle visuel des séries
synthétiques (creation_rupture.py, creation_gaps.py, creation_constants.py) et des séries réelles
(plot_real_neg_samples.py).
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _scatter_raw_value(df, title):
    """Scatter plot de raw_value au cours du temps, commun aux séries synthétiques."""
    fig = px.scatter(
        df, x='payload_timestamp', y='raw_value', template='plotly_white', title=title,
        labels={'payload_timestamp': 'Date', 'raw_value': 'payload_value'},
    )
    fig.update_traces(marker=dict(size=4))
    return fig


def plot_ruptures(df, device_id, sensor_id):
    """Scatter plot de raw_value au cours du temps, avec les ruptures injectées en évidence."""
    fig = _scatter_raw_value(df, f"{device_id} / {sensor_id} — ruptures injectées")
    for ts in df.loc[df['is_injected_rupture'], 'payload_timestamp']:
        fig.add_vline(x=ts, line_dash='dash', line_color='red')
    fig.show()


def plot_gaps(df, device_id, sensor_id):
    """Scatter plot de raw_value au cours du temps, pour vérifier visuellement l'allure de la
    série et les trous laissés par les lignes supprimées."""
    _scatter_raw_value(
        df, f"{device_id} / {sensor_id} — série synthétique avec relevés supprimés").show()


def plot_constants(df, device_id, sensor_id):
    """Scatter plot de raw_value au cours du temps, pour vérifier visuellement l'allure en
    paliers de la série générée."""
    _scatter_raw_value(df, f"{device_id} / {sensor_id} — série synthétique en paliers").show()


def decimate_min_max(df, max_points):
    """Réduit df à environ max_points lignes : coupe la série en max_points // 2 tranches de
    lignes consécutives et garde le minimum et le maximum de payload_value de chacune, pour qu'un
    saut isolé ne disparaisse pas. Renvoie df tel quel s'il est déjà assez court."""
    if max_points <= 0 or len(df) <= max_points:
        return df
    n_buckets = max(1, max_points // 2)
    bucket = np.arange(len(df)) * n_buckets // len(df)
    values = df['payload_value'].reset_index(drop=True)
    keep = pd.concat([values.groupby(bucket).idxmin(), values.groupby(bucket).idxmax()])
    return df.iloc[np.sort(keep.unique())]


def build_figure(df, device_id, sensor_id, n_total):
    """Figure plotly (scatter) d'une série réelle de real_neg_samples ; n_total est le nombre de
    points avant décimation."""
    decimated = len(df) < n_total
    suffix = f" — décimé à {len(df)} / {n_total} points" if decimated else f" — {n_total} points"
    fig = go.Figure(go.Scattergl(
        x=df['payload_timestamp'], y=df['payload_value'], mode='markers',
        marker=dict(size=3, color='#2563eb'),
        hovertemplate='%{x|%d/%m/%Y %H:%M}<br>%{y:,.3f}<extra></extra>',
    ))
    fig.update_layout(
        title=f"{device_id} / {sensor_id}{suffix}", template='plotly_white', height=380,
        margin=dict(l=70, r=20, t=50, b=40), showlegend=False,
        xaxis_title='payload_timestamp', yaxis_title='payload_value',
    )
    return fig
