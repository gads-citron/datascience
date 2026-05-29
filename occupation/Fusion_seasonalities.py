from dotenv import load_dotenv
import urllib.parse
from io import BytesIO
import json
import pandas as pd
import os
import clickhouse_connect
import boto3
import datetime as dt
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

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
    db.insert_df(p.dataout, s_occ)

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

def compute_raw_series(key):
    """Série horaire lissée (ROLLING), premiers N_HOURS points, interpolée si lacunes."""
    if LOCAL_TEST:
        df = pd.read_parquet(key)
    else:
        buf = BytesIO()
        _s3.download_fileobj(S3_BUCKET, key, buf)
        buf.seek(0)
        df = pd.read_parquet(buf)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).set_index(TIME_COL)
    df[VALUE_COL] = df[VALUE_COL].rolling(window=ROLLING, center=True, min_periods=1).mean()
    hourly = df[VALUE_COL].resample('1h').mean()
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

#
LOCAL_TEST     = False
LOCAL_DATA_DIR = './data/sample_s3'
LOCAL_MODEL    = './data/SVM_RBF_w3.pkl'

S3_BUCKET         = "gads-citron-sagemaker-custom-stage"
S3_PREFIX         = "data/occupation/processed/power/"
S3_OUTPUT_PREFIX  = "data/occupation/site_occupancy/"
TIME_COL = 'payload_timestamp' #'payload_expected_time'
VALUE_COL = 'occupancy_value' # 'raw_value'
ROLLING = 3
N_WEEKS = 3
N_HOURS = N_WEEKS * 7 * 24  # 504 points (échantillons de 3 semaines)
FFT_COMPONENTS = N_WEEKS * 30  # 90
ml_flow_experiment = 'SVM-RBF_w3-1'

_s3 = boto3.client('s3')

if LOCAL_TEST:
    svm_model = joblib.load(LOCAL_MODEL)
    prefix = LOCAL_DATA_DIR.rstrip('/').replace('\\', '/') + '/'
    site_prefixes = [
        prefix + d + '/'
        for d in os.listdir(LOCAL_DATA_DIR)
        if os.path.isdir(os.path.join(LOCAL_DATA_DIR, d)) and d.startswith('site_id=')
    ]
else:
    mlflow.set_tracking_uri("arn:aws:sagemaker:eu-west-3:585762237892:mlflow-app/app-N4LROSLQN5BN")
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
# Récupération des customer_account_name depuis ClickHouse prod
site_ids = [p.rstrip('/').split('/')[-1].split('=', 1)[-1] for p in site_prefixes]
envir = 'local' if LOCAL_TEST else 'sagemaker'
prod_client  = db_connect(type='prod',  envir=envir)
stage_client = db_connect(type='stage', envir=envir)
site_id_list = ', '.join(f"'{s}'" for s in site_ids)
account_df = prod_client.query_df(f"""
    SELECT site_id, site_name, customer_account_name
    FROM datamart_sous_comptage.distributed_dataset_hourly_payloads_merged
    WHERE site_id IN ({site_id_list})
    GROUP BY site_id, site_name, customer_account_name
""")
account_map   = dict(zip(account_df['site_id'].astype(str), account_df['customer_account_name']))
site_name_map = dict(zip(account_df['site_id'].astype(str), account_df['site_name']))

# LISTE : {site_id: DataFrame cumulant toutes les périodes, avec colonne 'couverture'}
LISTE = {}
rejected = []

for site_prefix in site_prefixes[:2]:
    site_id = site_prefix.rstrip('/').split('/')[-1]

    # 1. Grouper les fichiers parquet par période (= nom du fichier sans extension)
    parquet_by_period = {}
    if LOCAL_TEST:
        for root, _, files in os.walk(site_prefix.rstrip('/')):
            for fname in files:
                if fname.endswith('.parquet'):
                    key = os.path.join(root, fname).replace('\\', '/')
                    period = fname.replace('.parquet', '')
                    parquet_by_period.setdefault(period, []).append(key)
    else:
        paginator = _s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=site_prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if not key.endswith('.parquet'):
                    continue
                period = key.rsplit('/', 1)[-1].replace('.parquet', '')
                parquet_by_period.setdefault(period, []).append(key)

    site_periods = []

    for period, keys in parquet_by_period.items():
        N_TOTAL = len(keys)
        period_records = []

        # 2. Filtrer chaque fichier de la période avec le SVM
        for key in keys:
            rel = key[len(prefix):]
            parts = rel.split('/')
            series = compute_raw_series(key)
            features = compute_fft_features(series)

            if svm_model.predict([features])[0] == 1:
                device = parts[1].split('=', 1)[-1]
                sensor = parts[2].split('=', 1)[-1] if len(parts) >= 4 else ''
                # 3. Normaliser chaque fichier accepté
                series_norm = (series - series.min()) / (series.max() - series.min() + 1e-8)
                df_dev = series_norm.reset_index()
                df_dev.columns = [TIME_COL, VALUE_COL]
                df_dev['tag'] = f"{device}__{sensor}"
                period_records.append(df_dev)
            else:
                rejected.append({
                    'site_id':      parts[0].split('=', 1)[-1],
                    'device_id':    parts[1].split('=', 1)[-1],
                    'sensor_id':    parts[2].split('=', 1)[-1] if len(parts) >= 4 else '',
                    'parquet_file': key.rsplit('/', 1)[-1],
                })

        # 4. Fusion : moyenne heure par heure + couverture fixe pour la période
        if period_records:
            couverture = len(period_records) / N_TOTAL * 100
            df_all = pd.concat(period_records, ignore_index=True)
            df_all[TIME_COL] = pd.to_datetime(df_all[TIME_COL])
            fused = (
                df_all
                .groupby(df_all[TIME_COL].dt.floor('1h'))
                .agg(mean_occupancy=(VALUE_COL, 'mean'))
                .rename_axis(TIME_COL)
            )
            fused = fused.rename_axis('payload_timestamp')
            fused['coverage'] = couverture
            fused['customer_account_name'] = account_map.get(site_id.split('=', 1)[-1], None)
            fused['period'] = period
            site_id_val = site_id.split('=', 1)[-1]
            df_out = fused.reset_index()
            if LOCAL_TEST:
                out_dir = os.path.join('./data/sample_s3/site_occupancy', site_id)
                os.makedirs(out_dir, exist_ok=True)
                df_out.to_parquet(os.path.join(out_dir, f'{period}.parquet'), index=False)
            else:
                buf = BytesIO()
                df_out.to_parquet(buf, index=False)
                buf.seek(0)
                s3_key = f"{S3_OUTPUT_PREFIX}{site_id}/{period}.parquet"
                _s3.upload_fileobj(buf, S3_BUCKET, s3_key)
            # Écriture dans ClickHouse stage
            df_ck = pd.DataFrame({
                'payload_timestamp': df_out['payload_timestamp'],
                'site_id':           site_id_val,
                'site_name':         site_name_map.get(site_id_val, None),
                'customer_account_name': account_map.get(site_id_val, None),
                'mean_occupancy':    df_out['mean_occupancy'],
                'coverage':          df_out['coverage'],
            })
            stage_client.insert_df('Myriam_test.site_occupancy', df_ck)
            site_periods.append(fused)

    if site_periods:
        LISTE[site_id] = pd.concat(site_periods)

if LOCAL_TEST:
    os.makedirs('./data/sample_s3/site_occupancy', exist_ok=True)
    pd.DataFrame(rejected).to_csv('./data/rejected_by_svm.csv', index=False)
else:
    buf = BytesIO()
    pd.DataFrame(rejected).to_csv(buf, index=False)
    buf.seek(0)
    _s3.upload_fileobj(buf, S3_BUCKET, f"{S3_OUTPUT_PREFIX}rejected_by_svm.csv")