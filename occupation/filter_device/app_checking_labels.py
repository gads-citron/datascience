import citron_colors as col
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import boto3
import io

BUCKET = "gads-citron-sagemaker-custom-stage"
PREFIX = "data/occupation/processed/power/"
PRED_KEY = f"{PREFIX}predictions.csv"
x_time  = "payload_timestamp"
y_value = "occupancy_value"

st.set_page_config(page_title='Vérification prédictions', layout='wide')
s3 = boto3.client('s3')

LABELS = {1: 'Occupation', 0: 'Non-occupation'}
FILTER_OPTIONS = ['Tous', 'Faux positifs (prédit Occ, vrai Non-occ)',
                  'Faux négatifs (prédit Non-occ, vrai Occ)', 'Corrects']


def _build_path(site_id, device_id, sensor_id, parquet_file):
    segments = [f"site_id={site_id}", f"device_id={device_id}"]
    if pd.notna(sensor_id) and str(sensor_id).strip():
        segments.append(f"sensor_id={sensor_id}")
    segments.append(f"{parquet_file}.parquet")
    return f"s3://{BUCKET}/{PREFIX}" + "/".join(segments)


@st.cache_data(show_spinner='Chargement série…')
def load_series(path):
    key = path.replace(f's3://{BUCKET}/', '')
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
    df[x_time] = pd.to_datetime(df[x_time])
    df = df.sort_values(x_time)
    df[y_value] = df[y_value].rolling(window=3, center=True, min_periods=1).mean()
    return df


def _apply_filter(df, choice):
    if choice == 'Faux positifs (prédit Occ, vrai Non-occ)':
        return df[(df['prediction'] == 1) & (df['verite_terrain'] == 0)]
    if choice == 'Faux négatifs (prédit Non-occ, vrai Occ)':
        return df[(df['prediction'] == 0) & (df['verite_terrain'] == 1)]
    if choice == 'Corrects':
        return df[df['prediction'] == df['verite_terrain']]
    return df


@st.cache_data(show_spinner='Chargement predictions.csv…')
def load_predictions():
    obj = s3.get_object(Bucket=BUCKET, Key=PRED_KEY)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('Vérification prédictions')

try:
    pred_df = load_predictions()
except Exception as e:
    st.error(f'Impossible de charger `{PRED_KEY}` : {e}')
    st.stop()
pred_df['sensor_id'] = pred_df.get('sensor_id', pd.Series('')).fillna('')
pred_df['parquet_file'] = pred_df['parquet_file'].astype(str)
pred_df['path'] = pred_df.apply(
    lambda r: _build_path(r['site_id'], r['device_id'], r['sensor_id'], r['parquet_file']), axis=1
)

n_total = len(pred_df)
n_fp = int(((pred_df['prediction'] == 1) & (pred_df['verite_terrain'] == 0)).sum())
n_fn = int(((pred_df['prediction'] == 0) & (pred_df['verite_terrain'] == 1)).sum())
n_ok = int((pred_df['prediction'] == pred_df['verite_terrain']).sum())

with st.sidebar:
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric('Corrects', n_ok)
    col2.metric('Faux pos.', n_fp)
    col3.metric('Faux nég.', n_fn)

    st.divider()
    filter_choice = st.selectbox('Filtrer les échantillons', FILTER_OPTIONS)

filtered_df = _apply_filter(pred_df, filter_choice).reset_index(drop=True)
n_samples = len(filtered_df)

# reset index quand le filtre change
if st.session_state.get('last_filter') != filter_choice:
    st.session_state.pred_idx = 0
    st.session_state.last_filter = filter_choice

if 'pred_idx' not in st.session_state:
    st.session_state.pred_idx = 0

idx = min(st.session_state.pred_idx, max(n_samples - 1, 0))

with st.sidebar:
    st.divider()
    st.subheader(f'Échantillon  {idx + 1} / {n_samples}')
    c1, c2 = st.columns(2)
    c1.button('◀', disabled=(idx == 0), use_container_width=True,
              on_click=lambda: st.session_state.update(pred_idx=idx - 1))
    c2.button('▶', disabled=(idx >= n_samples - 1), use_container_width=True,
              on_click=lambda: st.session_state.update(pred_idx=idx + 1))

if n_samples == 0:
    st.warning(f'Aucun échantillon pour le filtre « {filter_choice} ».')
    st.stop()

row = filtered_df.iloc[idx]

# ── header ────────────────────────────────────────────────────────────────────
st.markdown(f"## Site `{row['site_id']}`")
st.markdown(f"`{row['device_id']}` — `sensor_id={row['sensor_id']}` — fichier `{row['parquet_file']}`")

vt   = int(row['verite_terrain'])
pred = int(row['prediction'])
match = vt == pred

m1, m2, m3 = st.columns(3)
m1.metric('Vérité terrain',    LABELS.get(vt,   '—'))
m2.metric('Prédiction modèle', LABELS.get(pred, '—'))
m3.metric('Résultat', '✓ Correct' if match else '✗ Erreur',
          delta=None, delta_color='off')

# ── graphique série temporelle ────────────────────────────────────────────────
smp = load_series(row['path'])
fig = go.Figure()
if not smp.empty:
    fig.add_trace(go.Scatter(
        x=smp[x_time], y=smp[y_value],
        mode='lines', line={'color': col.orange, 'width': 1}, name=y_value))

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

# ── profil moyen (heure × jour de semaine) ───────────────────────────────────
if not smp.empty:
    smp['hour']    = smp[x_time].dt.hour
    smp['weekday'] = smp[x_time].dt.weekday

    DAY_NAMES = {0: 'Lun', 1: 'Mar', 2: 'Mer', 3: 'Jeu', 4: 'Ven', 5: 'Sam', 6: 'Dim'}
    pivot = smp.groupby(['weekday', 'hour'])[y_value].mean().unstack(level='hour')
    pivot.index = pivot.index.map(DAY_NAMES)

    fig2 = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f'{h:02d}h' for h in pivot.columns],
        y=pivot.index,
        colorscale='OrRd',
        hovertemplate='%{y} %{x} : %{z:.2f}<extra></extra>',
    ))
    fig2.update_layout(
        template='simple_white', height=280,
        xaxis_title='Heure', yaxis_title='',
        margin=dict(t=10, b=40, l=50, r=20))
    st.plotly_chart(fig2, use_container_width=True)
