
from utils import citron_colors as col
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import boto3
import io
from botocore.exceptions import ClientError

INPUT_FOLDER = 's3://YOUR-BUCKET/YOUR-PREFIX/'            # à remplacer
CSV_PATH     = 's3://YOUR-BUCKET/YOUR-PREFIX/selections.csv'  # à remplacer

st.set_page_config(page_title='Labelling occupation', layout='wide')

s3 = boto3.client('s3')


def _parse_s3(path):
    without_scheme = path.replace('s3://', '')
    bucket, _, key = without_scheme.partition('/')
    return bucket, key

@st.cache_data(show_spinner='Scan des fichiers…')
def load_parquet_files(folder):
    files = []
    bucket, prefix = _parse_s3(folder)
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
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
                    'path':      f's3://{bucket}/{key}',
                })
    return pd.DataFrame(files)


@st.cache_data(show_spinner='Chargement série…')
def load_series(path):
    bucket, key = _parse_s3(path)
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
    df['payload_expected_time'] = pd.to_datetime(df['payload_expected_time'])
    df = df.sort_values('payload_expected_time')
    df['raw_value'] = df['raw_value'].rolling(window=3, center=True, min_periods=1).mean()
    return df


# ── données ──────────────────────────────────────────────────────────────────
all_files = load_parquet_files(INPUT_FOLDER)

def _load_selections():
    bucket, key = _parse_s3(CSV_PATH)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        for c in ['average_pattern', 'classif_svm', 'VT_Myriam']:
            if c not in df.columns:
                df[c] = pd.NA
        return df
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return pd.DataFrame(columns=['site_id', 'device_id', 'sensor_id', 'selection',
                                          'average_pattern', 'classif_svm', 'VT_Myriam'])
        raise

needs_reload = (
    'selections' not in st.session_state
    or 'classif_svm' not in st.session_state.selections.columns
    or st.session_state.selections['classif_svm'].isna().all()
)
if needs_reload:
    st.session_state.selections = _load_selections()

# filtrer : uniquement les profils avec une prédiction SVM
sel_svm      = st.session_state.selections
svm_rows     = sel_svm[sel_svm['classif_svm'].notna()]
svm_keys = set(zip(
    svm_rows['site_id'].astype(str),
    svm_rows['device_id'].astype(str),
))

files_df = all_files[
    all_files.apply(
        lambda r: (str(r['site_id']), str(r['device_id'])) in svm_keys,
        axis=1)
].reset_index(drop=True)
n_profiles = len(files_df)

if 'idx' not in st.session_state:
    vt_done = set(zip(
        sel_svm.loc[sel_svm['VT_Myriam'].notna(), 'site_id'].astype(str),
        sel_svm.loc[sel_svm['VT_Myriam'].notna(), 'device_id'].astype(str),
    ))
    st.session_state.idx = next(
        (i for i, r in files_df.iterrows()
         if (str(r['site_id']), str(r['device_id'])) not in vt_done),
        0)


def save():
    bucket, key = _parse_s3(CSV_PATH)
    buf = io.StringIO()
    st.session_state.selections.to_csv(buf, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue().encode())


def label(col, val):
    row = files_df.iloc[st.session_state.idx]
    sel  = st.session_state.selections
    mask = (
        (sel['site_id']   == row['site_id']) &
        (sel['device_id'] == row['device_id'])
    )
    if mask.any():
        st.session_state.selections.loc[mask, col] = val
    else:
        st.session_state.selections = pd.concat(
            [sel, pd.DataFrame([{
                'site_id':    row['site_id'],
                'device_id':  row['device_id'],
                'sensor_id':  row['sensor_id'],
                'selection':  pd.NA,
                'average_pattern': pd.NA,
                col:          val,
            }])], ignore_index=True)
    save()
    if st.session_state.idx < n_profiles - 1:
        st.session_state.idx += 1


# ── état courant ──────────────────────────────────────────────────────────────
if n_profiles == 0:
    st.warning('Aucun profil avec prédiction SVM trouvé. Lance d\'abord clustering_profiles.py.')
    st.stop()

idx     = min(st.session_state.idx, n_profiles - 1)
current = files_df.iloc[idx]

sel  = st.session_state.selections
mask = (
    (sel['site_id']   == current['site_id']) &
    (sel['device_id'] == current['device_id'])
)
def _first_valid(series):
    vals = series.dropna()
    return int(vals.iloc[0]) if len(vals) else None

current_label           = _first_valid(sel.loc[mask, 'selection'])
current_average_pattern = _first_valid(sel.loc[mask, 'average_pattern'])
current_classif_svm     = _first_valid(sel.loc[mask, 'classif_svm'])
current_vt_myriam       = _first_valid(sel.loc[mask, 'VT_Myriam'])

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('Labelling occupation')

    sel_ = st.session_state.selections
    n_vt_myriam = sel_['VT_Myriam'].notna().sum() if 'VT_Myriam' in sel_.columns else 0
    st.progress(int(n_vt_myriam) / n_profiles if n_profiles else 0,
                text=f'{int(n_vt_myriam)} / {n_profiles} profils validés (VT_Myriam)')
    st.divider()

    st.subheader(f'Profil  {idx + 1} / {n_profiles}')
    c1, c2 = st.columns(2)
    c1.button('◀', disabled=(idx == 0), use_container_width=True,
              on_click=lambda: st.session_state.update(idx=idx - 1))
    c2.button('▶', disabled=(idx == n_profiles - 1), use_container_width=True,
              on_click=lambda: st.session_state.update(idx=idx + 1))

    st.divider()
    st.subheader('VT_Myriam')
    st.button('Occupation', type='primary', use_container_width=True,
              on_click=label, args=('VT_Myriam', 1))
    st.button('Non-occupation',             use_container_width=True,
              on_click=label, args=('VT_Myriam', 0))

# ── header ────────────────────────────────────────────────────────────────────
st.markdown(f"## Site `{current['site_id']}`")
st.markdown(f"`{current['device_id']}` / `{current['sensor_id']}`")

c1, c2, c3 = st.columns(3)
svm_label = {1: '🤖 Occupation', 0: '🤖 Non-occupation'}.get(current_classif_svm, '—')
vt_label  = {1: '✅ Occupation', 0: '❌ Non-occupation'}.get(current_vt_myriam, '—')
c1.markdown(f"**Prédit SVM :** {svm_label}")
c2.markdown(f"**VT_Myriam :** {vt_label}")
c3.markdown(f"**average_pattern :** `{current_average_pattern}`" if current_average_pattern is not None else '**average_pattern :** —')

# ── graphique ─────────────────────────────────────────────────────────────────
smp = load_series(current['path'])
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
