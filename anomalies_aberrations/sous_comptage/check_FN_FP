# streamlit run check_FN_FP.py
import glob
import duckdb
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import citron_colors as col

DATA_DIR = './data/sample_anomalies_2025_sem_1'

st.set_page_config(page_title='Revue FP / FN', layout='wide')

@st.cache_data(show_spinner='Chargement parquet…')
def load_series(site_id, device_id, sensor_id):
    pattern = str(Path(DATA_DIR) / f"site_id={site_id}" / f"device_id={device_id}" / f"sensor_id={sensor_id}" / "**" / "*.parquet")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return pd.DataFrame()
    df = duckdb.sql(f"SELECT * FROM read_parquet({[str(f) for f in files]})").df()
    df['payload_expected_time'] = pd.to_datetime(df['payload_expected_time'], utc=True).dt.tz_convert(None)
    return df.sort_values('payload_expected_time').reset_index(drop=True)

@st.cache_data
def load_cases(csv_name):
    path = Path(DATA_DIR) / csv_name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['payload_expected_time'] = pd.to_datetime(
        df['payload_expected_time'], format='mixed', utc=True).dt.tz_convert(None)
    return df.reset_index(drop=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('Revue FP / FN')
    choice = st.radio('Afficher', ['Faux Positifs', 'Faux Négatifs'])

csv_name = 'false_positives.csv' if choice == 'Faux Positifs' else 'false_negatives.csv'
cases = load_cases(csv_name)

if cases.empty:
    st.warning(f'{csv_name} introuvable ou vide dans {DATA_DIR}.')
    st.stop()

n = len(cases)

if 'idx' not in st.session_state:
    st.session_state.idx = 0

# reset l'index quand on change de catégorie
if st.session_state.get('_choice') != choice:
    st.session_state.idx = 0
    st.session_state['_choice'] = choice

idx = min(st.session_state.idx, n - 1)

with st.sidebar:
    st.divider()
    st.subheader(f'Cas  {idx + 1} / {n}')
    c1, c2 = st.columns(2)
    c1.button('◀ Préc.', disabled=(idx == 0),     use_container_width=True,
              on_click=lambda: st.session_state.update(idx=idx - 1))
    c2.button('Suiv. ▶', disabled=(idx == n - 1), use_container_width=True,
              on_click=lambda: st.session_state.update(idx=idx + 1))
    st.progress((idx + 1) / n)

# ── cas courant ───────────────────────────────────────────────────────────────
row        = cases.iloc[idx]
event_time = row['payload_expected_time']
label      = '🔴 Faux Positif' if choice == 'Faux Positifs' else '🟠 Faux Négatif'

st.markdown(f'## {label}')
m1, m2, m3 = st.columns(3)
m1.metric('site_id',   str(row['site_id']))
m2.metric('device_id', str(row['device_id']))
m3.metric('sensor_id', str(row['sensor_id']))
st.caption(f"Date : {event_time}")

# ── graphique ─────────────────────────────────────────────────────────────────
smp = load_series(row['site_id'], row['device_id'], row['sensor_id'])
fig = go.Figure()

if smp.empty:
    st.warning('Pas de données parquet pour ce capteur.')
else:
    series = smp.dropna(subset=['raw_value'])
    fig.add_trace(go.Scatter(
        x=series['payload_expected_time'], y=series['raw_value'],
        mode='lines', line={'color': col.orange, 'width': 1}, name='raw_value'))

    closest_idx = (series['payload_expected_time'] - event_time).abs().idxmin()
    y_val = series.loc[closest_idx, 'raw_value']

    fig.add_trace(go.Scatter(
        x=[event_time], y=[y_val],
        mode='markers',
        marker={'color': 'red', 'size': 10, 'symbol': 'circle',
                'line': {'color': 'black', 'width': 2}},
        name=label))

    fig.update_layout(
        template='simple_white', height=480,
        xaxis={
            'tickangle': -45,
            'tickformatstops': [
                {'dtickrange': [None, 3600000],       'value': '%H:%M\n%d %b'},
                {'dtickrange': [3600000, 86400000],   'value': '%Hh\n%d %b'},
                {'dtickrange': [86400000, 604800000], 'value': '%a %d %b'},
                {'dtickrange': [604800000, None],     'value': '%d %b'},
            ],
        },
        margin=dict(t=20, b=40, l=50, r=20),
        legend={'itemclick': False, 'itemdoubleclick': False})

    st.plotly_chart(fig, use_container_width=True)
