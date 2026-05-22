from dotenv import load_dotenv
import urllib.parse
from io import BytesIO
import pandas as pd
import os
import clickhouse_connect
import boto3
import datetime as dt
import clickhouse_connect
import numpy as np
import mlflow
import mlflow.sklearn

#%% functions
def get_clickhouse_credentials(db='stage'):
    if db == 'stage':
        secret_name = "AmazonSageMaker-clickhouse-user-1"
    elif db == 'prod':
        secret_name = "AmazonSageMaker-clickhouse-user-prod"
    region_name = "eu-west-3"
    session = boto3.session.Session()
    sm_client = session.client(
        service_name = 'secretsmanager',
        region_name = region_name
    )
    get_secret_value_response = sm_client.get_secret_value(
            SecretId=secret_name)
    secret = json.loads(get_secret_value_response['SecretString'])
    return secret

def db_connect(type='prod', envir='local'):
    if envir == 'local':
        load_dotenv()
        username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
        if type=='prod':
            host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
            password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
        elif type == 'stage':
            host = urllib.parse.quote_plus(os.environ.get("clk_stg"))
            password = urllib.parse.quote_plus(os.environ.get('clk_stg_pass'))
    elif envir == 'sagemaker':
        secret = get_clickhouse_credentials(db=type)
        host = secret['host']
        username = secret['username']
        password = secret['password']
    conn = clickhouse_connect.get_client(host=host, username=username, port=443, password=password)
    return conn

def write_to_db(s_occ, r, y_col, datat, db):
    s_occ['device_id'] = r.device_id
    s_occ['sensor_name'] = r.sensor_name
    s_occ['sensor_id'] = r.sensor_id
    s_occ['site_id'] = r.site_id
    s_occ['data_source'] = [['.'.join([p.datasource, y_col])].copy() for _ in range(len(s_occ))]
    s_occ['model_version'] = p.model_version
    s_occ['valueType'] = datat
    s_occ['ingested_at'] = dt.datetime.now()
    db.insert_df(p.dataout, occ)

def _s3_prefix(*parts):
    """Construit le prefix S3 en joignant S3_PREFIX et les parties non-vides (sans double-slash)."""
    return "/".join(p.strip('/') for p in [S3_PREFIX, *parts] if p)

def load_parquet_files():
    """Liste tous les fichiers parquet dans S3 (structure site/device[/sensor]/date.parquet)."""
    prefix = _s3_prefix() + "/"
    paginator = _s3.get_paginator('list_objects_v2')
    files = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith('.parquet'):
                continue
            rel = key[len(prefix):]
            parts = rel.split('/')
            # site/device/sensor/date.parquet  OU  site/device/date.parquet
            if len(parts) == 4:
                files.append({'site_id': parts[0], 'device_id': parts[1],
                              'sensor_id': parts[2], 'path': key})
            elif len(parts) == 3:
                files.append({'site_id': parts[0], 'device_id': parts[1],
                              'sensor_id': '', 'path': key})
    return pd.DataFrame(files)

def get_parquet_path(site_id, device_id, sensor_id, parquet_file):
    """Construit la clé S3 du fichier parquet labelisé (colonne parquet_file du CSV)."""
    segments = [f"site_id={site_id}", f"device_id={device_id}"]
    if sensor_id:
        segments.append(f"sensor_id={sensor_id}")
    segments.append(f"{parquet_file}.parquet")
    return _s3_prefix(*segments)

def compute_raw_series(s3_key):
    """Série horaire lissée (ROLLING), premiers N_HOURS points, interpolée si lacunes."""
    buf = BytesIO()
    _s3.download_fileobj(S3_BUCKET, s3_key, buf)
    buf.seek(0)
    df = pd.read_parquet(buf)
    df['payload_timestamp'] = pd.to_datetime(df['payload_timestamp'])
    df = df.sort_values('payload_timestamp').set_index('payload_timestamp')
    df['occupancy_value'] = df['occupancy_value'].rolling(window=ROLLING, center=True, min_periods=1).mean()
    hourly = df['occupancy_value'].resample('1h').mean()
    hourly = hourly.iloc[:N_HOURS].interpolate(method='time').fillna(0)
    if len(hourly) < N_HOURS:
        full_index = pd.date_range(hourly.index[0], periods=N_HOURS, freq='1h')
        hourly = hourly.reindex(full_index, fill_value=0)
    return hourly.iloc[:N_HOURS]

def compute_fft_features(series):
    """Spectre de puissance normalisé sur les FFT_COMPONENTS premières fréquences (hors DC)."""
    mag = np.abs(np.fft.rfft(series))
    total_power = mag[1:].sum() + 1e-8
    return mag[1:FFT_COMPONENTS + 1] / total_power

#%% DtB connect
# prod_client = db_connect(type='prod', envir=p.envir)
# stage_client = db_connect(type='stage', envir=p.envir)

#
S3_BUCKET  = "gads-citron-sagemaker-custom-stage"
S3_PREFIX  = "data/occupation/processed/power/"
SELECTIONS_CSV = "verite_terrain.csv"
ROLLING = 3
N_WEEKS = 3
N_HOURS = N_WEEKS * 7 * 24  # 504 points (échantillons de 3 semaines)
FFT_IDX_DAY = N_WEEKS * 7   # 21 → 1 cycle/jour sur 504h
FFT_COMPONENTS = N_WEEKS * 30  # 90
ml_flow_experiment = 'SVM-RBF_w3-1'

_s3 = boto3.client('s3')
DAILY_POWER_THRESH = 0.01
RDM = 42 # seed

experiment = mlflow.get_experiment_by_name("Occupation-device-filter")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"run_name = '{ml_flow_experiment}'",
    order_by=["start_time DESC"],
)
svm_model = mlflow.sklearn.load_model(f"runs:/{runs.iloc[0].run_id}/model")

prefix = S3_PREFIX.rstrip('/') + '/'
response = _s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, Delimiter='/')
site_prefixes = [
    cp['Prefix'] for cp in response.get('CommonPrefixes', [])
    if cp['Prefix'].rstrip('/').split('/')[-1].startswith('site_id=')
]
LISTE = {}  # ici liste ce sont des séries de 168*50 
rejected = []

for site_prefix in site_prefixes[:2]:
    site_id = site_prefix.rstrip('/').split('/')[-1]
    site_records = []
    sub_resp = _s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=site_prefix, Delimiter='/')
    N_TOTAL = len(sub_resp.get('CommonPrefixes', []))

    paginator = _s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=site_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith('.parquet'):
                continue
            series = compute_raw_series(key)
            features = compute_fft_features(series)
            rel = key[len(prefix):]
            parts = rel.split('/')
            if svm_model.predict([features])[0] == 1:
                device = parts[1].split('=', 1)[-1]
                sensor = parts[2].split('=', 1)[-1] if len(parts) == 4 else ''
                series_norm = (series - series.min()) / (series.max() - series.min() + 1e-8)
                df_device = series_norm.to_frame(name='occupancy_value')
                df_device['tag'] = f"{device}__{sensor}"
                site_records.append(df_device)
            else:
                rejected.append({
                    'site_id':      parts[0].split('=', 1)[-1],
                    'device_id':    parts[1].split('=', 1)[-1],
                    'sensor_id':    parts[2].split('=', 1)[-1] if len(parts) == 4 else '',
                    'parquet_file': parts[-1],
                })

    if site_records:
        df_long = pd.concat(site_records)
        df_long.index = pd.to_datetime(df_long.index)
        result = df_long.groupby(df_long.index).agg(
            mean_occupancy=('occupancy_value', 'mean'),
            n_unique_tags=('tag', 'nunique'),
        )
        result['pct_devices_selected'] = result['n_unique_tags'] / N_TOTAL
        LISTE[site_id] = result

pd.DataFrame(rejected).to_csv('rejected_by_svm.csv', index=False)