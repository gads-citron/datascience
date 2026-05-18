
import citron_colors as col
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import boto3
import io
from botocore.exceptions import ClientError

BUCKET  = "gads-citron-sagemaker-custom-stage"
PREFIX  = "data/occupation/processed/power/"
CSV_KEY = f"{PREFIX}verite_terrain.csv"

st.set_page_config(page_title='Labelling occupation', layout='wide')

s3 = boto3.client('s3')


@st.cache_data(show_spinner='Scan des fichiers…')
def load_parquet_files():
    files = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith('.parquet'):
                continue
            parts = {p.split('=')[0]: p.split('=')[1]
                     for p in key.split('/') if '=' in p}
            if 'device_id' in parts:
                files.append({
                    'site_id':   parts.get('site_id', ''),
                    'device_id': parts.get('device_id', ''),
                    'sensor_id': parts.get('sensor_id', ''),
                    'path':      f's3://{BUCKET}/{key}',
                })
    return pd.DataFrame(files)


@st.cache_data(show_spinner='Chargement série…')
def load_series(path):
    key = path.replace(f's3://{BUCKET}/', '')
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
    df['payload_timestamp'] = pd.to_datetime(df['payload_timestamp'])
    df = df.sort_values('payload_timestamp')
    df['raw_value'] = df['raw_value'].rolling(window=3, center=True, min_periods=1).mean()
    return df


# ── données ──────────────────────────────────────────────────────────────────
all_files  = load_parquet_files()

# dossiers = combinaisons uniques (site_id, device_id, sensor_id)
folders_df = all_files[['site_id', 'device_id', 'sensor_id']].drop_duplicates().reset_index(drop=True)
n_folders  = len(folders_df)

# devices = pour le suivi du labelling
devices_df = all_files[['site_id', 'device_id']].drop_duplicates().reset_index(drop=True)
n_devices  = len(devices_df)


def _load_selections():
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=CSV_KEY)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        for c in ('VT_Myriam', 'parquet_file'):
            if c not in df.columns:
                df[c] = pd.NA
        return df
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            df = devices_df.copy()
            df['VT_Myriam']    = pd.NA
            df['parquet_file'] = pd.NA
            return df
        raise


def save():
    buf = io.StringIO()
    st.session_state.selections.to_csv(buf, index=False)
    s3.put_object(Bucket=BUCKET, Key=CSV_KEY, Body=buf.getvalue().encode())


if 'selections' not in st.session_state:
    st.session_state.selections = _load_selections()

if 'folder_idx' not in st.session_state:
    st.session_state.folder_idx = 0
    st.session_state.file_idx   = 0


def prev_folder():
    st.session_state.folder_idx -= 1
    st.session_state.file_idx    = 0

def next_folder():
    st.session_state.folder_idx += 1
    st.session_state.file_idx    = 0


def label(val, path):
    folder = folders_df.iloc[st.session_state.folder_idx]
    sel    = st.session_state.selections
    mask   = (
        (sel['site_id'].astype(str)   == str(folder['site_id'])) &
        (sel['device_id'].astype(str) == str(folder['device_id']))
    )
    st.session_state.selections.loc[mask, 'VT_Myriam']    = val
    st.session_state.selections.loc[mask, 'parquet_file'] = path.split('/')[-1].removesuffix('.parquet')
    save()
    if st.session_state.folder_idx < n_folders - 1:
        next_folder()


# ── état courant ──────────────────────────────────────────────────────────────
if n_folders == 0:
    st.warning('Aucun fichier parquet trouvé dans le préfixe S3.')
    st.stop()

folder_idx     = min(st.session_state.folder_idx, n_folders - 1)
current_folder = folders_df.iloc[folder_idx]

# fichiers du dossier courant
folder_files = all_files[
    (all_files['site_id'].astype(str)   == str(current_folder['site_id'])) &
    (all_files['device_id'].astype(str) == str(current_folder['device_id'])) &
    (all_files['sensor_id'].astype(str) == str(current_folder['sensor_id']))
].reset_index(drop=True)
n_files  = len(folder_files)
file_idx = min(st.session_state.file_idx, n_files - 1)

n_sensors_for_device = folders_df[
    (folders_df['site_id'].astype(str)   == str(current_folder['site_id'])) &
    (folders_df['device_id'].astype(str) == str(current_folder['device_id']))
].shape[0]

def advance_file():
    if st.session_state.file_idx < n_files - 1:
        st.session_state.file_idx += 1
    elif n_sensors_for_device == 1 and st.session_state.folder_idx < n_folders - 1:
        next_folder()

current_path = folder_files.iloc[file_idx]['path']

sel  = st.session_state.selections
mask = (
    (sel['site_id'].astype(str)   == str(current_folder['site_id'])) &
    (sel['device_id'].astype(str) == str(current_folder['device_id']))
)

def _first_valid(series):
    vals = series.dropna()
    return int(vals.iloc[0]) if len(vals) else None

current_vt_myriam = _first_valid(sel.loc[mask, 'VT_Myriam'])

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('Labelling occupation')

    n_vt_myriam = st.session_state.selections['VT_Myriam'].notna().sum()
    st.progress(int(n_vt_myriam) / n_devices if n_devices else 0,
                text=f'{int(n_vt_myriam)} / {n_devices} devices validés')
    st.divider()

    # navigation fichiers dans le dossier courant
    st.subheader(f'Fichier  {file_idx + 1} / {n_files}')
    c1, c2 = st.columns(2)
    c1.button('◀', key='prev_file', disabled=(file_idx == 0),
              use_container_width=True,
              on_click=lambda: st.session_state.update(file_idx=file_idx - 1))
    next_file_disabled = (file_idx == n_files - 1) and (
        n_sensors_for_device > 1 or folder_idx == n_folders - 1)
    c2.button('▶', key='next_file', disabled=next_file_disabled,
              use_container_width=True, on_click=advance_file)

    # navigation dossiers sensor_id (uniquement si plusieurs sensors pour ce device)
    if n_sensors_for_device > 1:
        st.subheader(f'Dossier sensor  {folder_idx + 1} / {n_folders}')
        s1, s2 = st.columns(2)
        s1.button('◀', key='prev_folder', disabled=(folder_idx == 0),
                  use_container_width=True, on_click=prev_folder)
        s2.button('▶', key='next_folder', disabled=(folder_idx == n_folders - 1),
                  use_container_width=True, on_click=next_folder)

    st.divider()
    st.subheader('VT_Myriam')
    st.button('Occupation', type='primary', use_container_width=True,
              on_click=label, args=(1, current_path))
    st.button('Non-occupation', use_container_width=True,
              on_click=label, args=(0, current_path))

# ── header ────────────────────────────────────────────────────────────────────
st.markdown(f"## Site `{current_folder['site_id']}`")
st.markdown(f"`{current_folder['device_id']}` — `sensor_id={current_folder['sensor_id']}`")

vt_label = {1: '✅ Occupation', 0: '❌ Non-occupation'}.get(current_vt_myriam, '—')
st.markdown(f"**VT_Myriam :** {vt_label}")

# ── graphique ─────────────────────────────────────────────────────────────────
smp = load_series(current_path)
fig = go.Figure()
if not smp.empty:
    fig.add_trace(go.Scatter(
        x=smp['payload_expected_time'], y=smp['raw_value'],
        mode='lines', line={'color': col.orange, 'width': 1}, name='raw_value'))

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
    smp['hour']    = smp['payload_expected_time'].dt.hour
    smp['weekday'] = smp['payload_expected_time'].dt.weekday  # 0=lundi … 6=dimanche

    pivot = smp.groupby(['weekday', 'hour'])['raw_value'].mean().unstack(level='hour')
    pivot.index = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']

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
