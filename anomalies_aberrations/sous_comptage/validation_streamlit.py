# streamlit run validation_streamlit.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import citron_colors as col
from db_connector import ClickHouseClient

# ── config ────────────────────────────────────────────────────────────────────
INPUT_FOLDER  = './data/sample_anomalies_2025_sem_1'
CSV_PATH      = f'{INPUT_FOLDER}/verite_terrain.csv'
SOUS_COMPTAGE = 'datamart_sous_comptage.distributed_dataset_hourly_payloads_merged'
START         = '2025-01-01 00:00:00'
END           = '2025-12-31 00:00:00'

st.set_page_config(page_title='Validation manuelle des anomalies', layout='wide')

# ── client + série cachée par capteur ────────────────────────────────────────
@st.cache_resource
def get_client():
    return ClickHouseClient()

@st.cache_data(show_spinner='Chargement série…')
def load_series(device_id, sensor_id):
    smp = get_client().query(f"""
        SELECT payload_expected_time, entity, eval_prev_hour, sum_payload_value_hour_curr
        FROM {SOUS_COMPTAGE}
        WHERE payload_expected_time BETWEEN '{START}' AND '{END}'
          AND device_id = '{device_id}' AND sensor_id = '{sensor_id}'
        ORDER BY payload_expected_time LIMIT 100000""")
    if smp.empty:
        return smp
    smp = smp[smp['entity'] == smp['entity'].value_counts().index[0]]
    smp['payload_expected_time'] = pd.to_datetime(smp['payload_expected_time'], utc=True).dt.tz_convert(None)
    return smp.reset_index(drop=True)

# ── session state ─────────────────────────────────────────────────────────────
if 'anomalies' not in st.session_state:
    df = pd.read_csv(CSV_PATH)
    if 'VT_myriam' not in df.columns:
        df['VT_myriam'] = pd.NA
    st.session_state.anomalies = df

for key, default in [('g_idx', 0), ('a_idx', 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

def save():
    st.session_state.anomalies.to_csv(CSV_PATH, index=False)

# ── groupes (device_id, sensor_id) ───────────────────────────────────────────
anomalies = st.session_state.anomalies
groups    = [(key, grp) for key, grp in anomalies.groupby(['device_id', 'sensor_id'])]
n_groups  = len(groups)

g_idx = min(st.session_state.g_idx, n_groups - 1)
(device_id, sensor_id), grp = groups[g_idx]
n_anom = len(grp)
a_idx  = min(st.session_state.a_idx, n_anom - 1)
row    = grp.iloc[a_idx]
y_col  = row['feature']

# ── callbacks ─────────────────────────────────────────────────────────────────
def prev_group():
    st.session_state.g_idx -= 1
    st.session_state.a_idx  = 0

def next_group():
    untagged = grp.index[st.session_state.anomalies.loc[grp.index, 'VT_myriam'].isna()]
    st.session_state.anomalies.loc[untagged, 'VT_myriam'] = 0
    save()
    st.session_state.g_idx += 1
    st.session_state.a_idx  = 0

def validate(val):
    st.session_state.anomalies.loc[row.name, 'VT_myriam'] = val
    save()
    if a_idx < n_anom - 1:
        st.session_state.a_idx = a_idx + 1
    elif g_idx < n_groups - 1:
        st.session_state.g_idx = g_idx + 1
        st.session_state.a_idx = 0

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('Validation VT_myriam')

    validated = anomalies['VT_myriam'].notna().sum()
    st.progress(int(validated) / len(anomalies) if len(anomalies) else 0,
                text=f'{int(validated)} / {len(anomalies)} lignes validées')
    st.divider()

    st.subheader(f'Capteur  {g_idx + 1} / {n_groups}')
    c1, c2 = st.columns(2)
    c1.button('◀ Capteur', disabled=(g_idx == 0),            use_container_width=True, on_click=prev_group)
    c2.button('Capteur ▶', disabled=(g_idx == n_groups - 1), use_container_width=True, on_click=next_group)

    st.subheader(f'Anomalie  {a_idx + 1} / {n_anom}')
    c3, c4 = st.columns(2)
    c3.button('◀ Anom.', disabled=(a_idx == 0),          use_container_width=True,
              on_click=lambda: st.session_state.update(a_idx=a_idx - 1))
    c4.button('Anom. ▶', disabled=(a_idx == n_anom - 1), use_container_width=True,
              on_click=lambda: st.session_state.update(a_idx=a_idx + 1))

    st.divider()

    st.subheader('Validation')
    st.button('1 — Vraie anomalie', type='primary', use_container_width=True,
              on_click=validate, args=(1,))
    st.button('0 — Fausse alerte',               use_container_width=True,
              on_click=validate, args=(0,))

    st.divider()

    with st.expander('+ Ajouter une anomalie au groupe courant'):
        with st.form('add_row', clear_on_submit=True):
            # champs fixes — pré-remplis depuis le groupe, non éditables
            st.text_input('customer_account_name', value=str(row['customer_account_name']), disabled=True)
            st.text_input('site_id',               value=str(row['site_id']),               disabled=True)
            st.text_input('device_id',             value=str(device_id),                    disabled=True)
            st.text_input('sensor_id',             value=str(sensor_id),                    disabled=True)
            st.text_input('device_name',           value=str(row['device_name']),           disabled=True)
            st.text_input('sensor_name',           value=str(row['sensor_name']),           disabled=True)
            # champs à saisir
            f_ts      = st.text_input('payload_expected_time (YYYY-MM-DD HH:MM:SS)')
            st.selectbox('feature', ['eval_prev_hour', 'sum_payload_value_hour_curr'],
                         index=0 if y_col == 'eval_prev_hour' else 1, disabled=True)
            f_value   = st.number_input('value', value=0.0)
            f_iqr     = st.checkbox('iqr')
            f_isof    = st.checkbox('isof')
            f_stl     = st.checkbox('stl')
            f_vt      = st.selectbox('VT_myriam', ['—', '1', '0'])

            if st.form_submit_button('Ajouter', use_container_width=True):
                new_row = {
                    'customer_account_name': row['customer_account_name'],
                    'site_id':               row['site_id'],
                    'device_id':             device_id,
                    'sensor_id':             sensor_id,
                    'device_name':           row['device_name'],
                    'sensor_name':           row['sensor_name'],
                    'payload_expected_time': f_ts,
                    'feature':               y_col,
                    'value':                 f_value,
                    'iqr':                   f_iqr,
                    'isof':                  f_isof,
                    'stl':                   f_stl,
                    'VT_myriam':             int(f_vt) if f_vt in ('0', '1') else pd.NA,
                }
                st.session_state.anomalies = pd.concat(
                    [anomalies, pd.DataFrame([new_row])], ignore_index=True)
                save()
                st.rerun()

# ── header ────────────────────────────────────────────────────────────────────
vt_val  = row['VT_myriam']
vt_disp = (f"{'✅' if int(vt_val) == 1 else '❌'} {int(vt_val)}") if pd.notna(vt_val) else '—'

st.markdown(f"## {row['customer_account_name']}")
st.markdown(f"`{row['device_name']}` / `{row['sensor_name']}`")

m1, m2, m3, m4 = st.columns(4)
m1.metric('IQR',       '✓' if row['iqr']  else '·')
m2.metric('ISOF',      '✓' if row['isof'] else '·')
m3.metric('STL',       '✓' if row['stl']  else '·')
m4.metric('VT_myriam', vt_disp)

# ── graphique — série chargée une fois par capteur, marker sur l'anomalie courante
smp = load_series(str(device_id), str(sensor_id))

others = grp[grp.index != row.name]

fig = go.Figure()
if not smp.empty:
    series = smp.dropna(subset=[y_col])
    fig.add_trace(go.Scatter(
        x=series['payload_expected_time'], y=series[y_col],
        mode='lines', line={'color': col.orange, 'width': 1}, name=y_col))

# autres anomalies du groupe — petits markers gris
if not others.empty:
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(others['payload_expected_time'], format='mixed', utc=True).dt.tz_convert(None), y=others['value'],
        mode='markers',
        marker={'color': 'gray', 'size': 8, 'symbol': 'circle', 'opacity': 0.7},
        name='autres anomalies'))

# anomalie courante — marker rouge bien visible
fig.add_trace(go.Scatter(
    x=[pd.to_datetime(row['payload_expected_time'], format='mixed', utc=True).tz_convert(None)], y=[row['value']],
    mode='markers',
    marker={'color': 'red', 'size': 8, 'symbol': 'circle',
            'line': {'color': 'black', 'width': 2}},
    name='anomalie courante'))

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
