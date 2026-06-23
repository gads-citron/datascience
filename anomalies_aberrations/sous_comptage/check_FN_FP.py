# streamlit run check_FN_FP.py
import glob
import boto3
import duckdb
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import citron_colors as col

CSV_DIR  = 's3://sagemaker-eu-west-3-585762237892/workspaces/default/1/4adf6a370c71483c895ddb9786e1292e/artifacts'
DATA_DIR = 's3://gads-citron-sagemaker-custom-stage/data/anomaly/tertiary_IQ/2026'
VT_COL   = 'VT_myriam'

def _find_vt_files(data_dir):
    if _is_s3(data_dir):
        bucket, prefix = _s3_parts(data_dir)
        resp = boto3.client('s3').list_objects_v2(Bucket=bucket, Prefix=prefix)
        return sorted(
            f"s3://{bucket}/{obj['Key']}"
            for obj in resp.get('Contents', [])
            if 'verite_terrain' in obj['Key'] and obj['Key'].endswith('.csv')
        )
    return sorted(glob.glob(str(Path(data_dir) / 'verite_terrain*.csv')))

def _is_s3(path):
    return str(path).startswith('s3://')

def _s3_parts(path):
    without = str(path).replace('s3://', '', 1)
    bucket, prefix = without.split('/', 1)
    return bucket, prefix.rstrip('/')

VT_FILES = _find_vt_files(DATA_DIR)

st.set_page_config(page_title='Revue FP / FN', layout='wide')

@st.cache_data(show_spinner='Chargement parquet…')
def load_series(site_id, device_id):
    sub = f"site_id={site_id}/device_id={device_id}/*.parquet"
    if _is_s3(DATA_DIR):
        path = f"{DATA_DIR}/{sub}"
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs; CALL load_aws_credentials();")
        df = con.execute(f"SELECT * FROM read_parquet('{path}')").df()
    else:
        files = glob.glob(str(Path(DATA_DIR) / sub))
        if not files:
            return pd.DataFrame()
        df = duckdb.sql(f"SELECT * FROM read_parquet({[str(f) for f in files]})").df()
    df['payload_expected_time'] = pd.to_datetime(df['payload_expected_time'], utc=True).dt.tz_convert(None)
    return df.sort_values('payload_expected_time').reset_index(drop=True)

@st.cache_data
def load_cases(csv_name, label):
    if _is_s3(CSV_DIR):
        bucket, prefix = _s3_parts(CSV_DIR)
        obj = boto3.client('s3').get_object(Bucket=bucket, Key=f"{prefix}/{csv_name}")
        df = pd.read_csv(obj['Body'])
    else:
        path = Path(CSV_DIR) / csv_name
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
    df['_type'] = label
    df['payload_expected_time'] = pd.to_datetime(
        df['payload_expected_time'], format='mixed', utc=True).dt.tz_convert(None)
    return df

@st.cache_data
def load_all_cases():
    fp = load_cases('false_positives.csv', 'Faux Positif')
    fn = load_cases('false_negatives.csv', 'Faux Négatif')
    dfs = [d for d in [fp, fn] if not d.empty]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def load_vt(path):
    if _is_s3(path):
        bucket, key = _s3_parts(path)
        obj = boto3.client('s3').get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'])
    else:
        df = pd.read_csv(path)
    df['payload_expected_time'] = pd.to_datetime(
        df['payload_expected_time'], utc=True).dt.tz_convert(None)
    return df

def save_vt(df, path):
    if _is_s3(path):
        bucket, key = _s3_parts(path)
        boto3.client('s3').put_object(
            Bucket=bucket, Key=key, Body=df.to_csv(index=False).encode())
    else:
        df.to_csv(path, index=False)

# ── chargement ────────────────────────────────────────────────────────────────
cases = load_all_cases()

if cases.empty:
    st.warning(f'Aucun fichier false_positives.csv / false_negatives.csv trouvé dans {CSV_DIR}.')
    st.stop()

n = len(cases)

if 'idx' not in st.session_state:
    st.session_state.idx = 0

idx = min(st.session_state.idx, n - 1)

# ── cas courant ───────────────────────────────────────────────────────────────
row        = cases.iloc[idx]
event_time = row['payload_expected_time']
is_fp      = row['_type'] == 'Faux Positif'
label      = '🔴 Faux Positif' if is_fp else '🟠 Faux Négatif'

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('Revue FP / FN')
    st.divider()
    st.subheader(f'Cas  {idx + 1} / {n}')
    c1, c2 = st.columns(2)
    c1.button('◀ Préc.', disabled=(idx == 0),     use_container_width=True,
              on_click=lambda: st.session_state.update(idx=idx - 1))
    c2.button('Suiv. ▶', disabled=(idx == n - 1), use_container_width=True,
              on_click=lambda: st.session_state.update(idx=idx + 1))
    st.progress((idx + 1) / n)

    st.divider()
    st.subheader('Vérité terrain')
    if VT_FILES:
        vt_path = st.selectbox('Fichier VT', VT_FILES, label_visibility='collapsed')
        vt_key  = f'vt_{vt_path}'
        if vt_key not in st.session_state:
            st.session_state[vt_key] = load_vt(vt_path)
        vt_df = st.session_state[vt_key]

        mask = (
            (vt_df['site_id'].astype(str) == str(row['site_id'])) &
            (vt_df['device_id'].astype(str) == str(row['device_id'])) &
            (vt_df['payload_expected_time'] == event_time)
        )
        if mask.any():
            vt_idx  = vt_df[mask].index[0]
            current = int(vt_df.loc[vt_idx, VT_COL])
            new_val = st.radio(
                'Label',
                [0, 1],
                index=current,
                format_func=lambda x: ('✅ Anomalie' if x else '❌ Pas anomalie'),
                horizontal=True,
                key=f'vt_radio_{idx}',
            )
            if st.button('💾 Sauvegarder', use_container_width=True):
                st.session_state[vt_key].loc[vt_idx, VT_COL] = new_val
                save_vt(st.session_state[vt_key], vt_path)
                st.success('Sauvegardé !')
        else:
            st.caption('Ce cas n\'est pas dans le fichier VT sélectionné.')
    else:
        st.info('Aucun fichier verite_terrain trouvé.')

# ── en-tête principal ─────────────────────────────────────────────────────────
st.markdown(f'## {label}')
m1, m2 = st.columns(2)
m1.metric('site_id',   str(row['site_id']))
m2.metric('device_id', str(row['device_id']))
st.caption(f"Date : {event_time}")

# ── graphique ─────────────────────────────────────────────────────────────────
smp = load_series(row['site_id'], row['device_id'])
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
