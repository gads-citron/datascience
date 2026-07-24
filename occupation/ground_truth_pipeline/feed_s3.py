import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])

import pandas as pd
from pathlib import Path
import datetime as dt
import numpy as np
import clickhouse_connect
from dotenv import load_dotenv
import urllib
import os
import json
import boto3
import logging
from tqdm import tqdm

#%% collect parameters
start = "2025-01-01 00:00:00"
end = "2026-06-30 23:59:59"
x_time = 'payload_expected_time'
datasource = 'datamart_sous_comptage.distributed_dataset_hourly_payloads_merged'
output_dir = './data_collect/sample_s3/'

#%% S3 destination (aligné sur run_sagemaker.py)
S3_BUCKET = os.environ.get("S3_BUCKET", "gads-citron-sagemaker-custom-stage")
S3_PREFIX = os.environ.get("S3_PREFIX", "data/power_clk_collect/")
UPLOAD_TO_S3 = os.environ.get("UPLOAD_TO_S3", "1") == "1"
s3_client = boto3.client("s3") if UPLOAD_TO_S3 else None

#%% liste des sites IQ (ayant eau + elec)
candidates = pd.read_csv('./data_collect/list_site_eau_and_elec_IQ.csv')

def serie_typology(df, tol=0.1):
    score = np.mean((df.shift() - df) < 0) # combien de fois la valeur décrémente
    if score > tol:
        return 0 # volume
    else:
        return 1 # index

def check_problems(s, errors, y_col):
    """
        Parameters
        ----------
        s : sample data to be checked
        errors : log map to be filled
        y_col : col in "datamart_sous_comptage" used to read data
        Returns
        -------
        errors : map of error description
        flag_return : 0 if no problem; 1 else
        y_col : same as input if no change
        datat : datatype for tracking purpose (data index or volume)
    """
    datat = 'volume'
    flag_return = 0
    if s.empty:
        errors['error'] = "no_data"
        flag_return = 1
        return errors, flag_return, y_col, datat
    if s[y_col].isna().mean() > 0.1:
        errors['error'] = "too_many_NAN"
        flag_return = 1
    if serie_typology(s[y_col]):
        y_col = 'eval_prev_hour'
        datat = 'index'
        if serie_typology(s[y_col]):
            errors['error'] = "index_serie_on_eval_prev"
            flag_return = 1
    if s[y_col].astype(float).sum() == 0:
        errors['error'] = "zero_values"
        flag_return = 1
    if s[y_col].value_counts().shape[0] < 100:
        errors['error'] = "continuous"
        flag_return = 1
    if s.shape[0] < 168 * 3:
        errors['error'] = "history_not_enough"
        flag_return = 1
    return errors, flag_return, y_col, datat

def upload_to_s3(local_path, site_id, device_id, sensor_id, file_name):
    """Upload un parquet local vers s3://{S3_BUCKET}/{S3_PREFIX}site_id=.../device_id=.../sensor_id=.../."""
    if not UPLOAD_TO_S3:
        return
    key = (
        f"{S3_PREFIX}"
        f"site_id={site_id}/"
        f"device_id={device_id}/"
        f"sensor_id={sensor_id}/"
        f"{file_name}"
    )
    s3_client.upload_file(str(local_path), S3_BUCKET, key)

def export_to_parquet(df, y_col, start_date, end_date,
                            site_id, device_id, sensor_id, output_dir):
    date_col = 'payload_expected_time'
    if df[date_col].dt.tz is not None:
        df[date_col] = df[date_col].dt.tz_localize(None)
    df.sort_values(date_col, inplace=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    start = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    partition_dir = (
            output_path /
            f"site_id={site_id}" /
            f"device_id={device_id}" /
            f"sensor_id={sensor_id}"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df['raw_value'] = df[y_col]
    df['site_id'] = site_id
    cols = ['site_id', 'device_id', 'sensor_id', 'payload_expected_time', 'raw_value']

    for year, df_year in df.groupby(df[date_col].dt.year):
        file_name = f"data_{year}.parquet"
        file_path = partition_dir / file_name
        df_year[cols].to_parquet(file_path, index=False)
        upload_to_s3(file_path, site_id, device_id, sensor_id, file_name)

def db_connect(type='prod'):    
    load_dotenv()
    username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
    if type=='prod':
        host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
        password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
    elif type == 'stage':
        host = urllib.parse.quote_plus(os.environ.get("clk_stg"))
        password = urllib.parse.quote_plus(os.environ.get('clk_stg_pass'))
    conn = clickhouse_connect.get_client(host=host, username=username, port=443, password=password)
    return conn

prod_client = db_connect(type='prod')
#%% devices from which collect is to be done:
devices_list = prod_client.query_df(f"""
    SELECT distinct site_id, device_id, sensor_id
    FROM {datasource}
    WHERE 1=1
    AND {x_time} between '{start}' and '{end}'
    AND site_id IN {candidates.site_id.drop_duplicates().to_list()}
    settings max_rows_to_read = 10000000000000;""")

log_path = Path(output_dir) / 'process_log.log'
log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
for i, row in tqdm(devices_list.iterrows(), total=len(devices_list)):
    error_dict = {"device":row.device_id, "start":start, 'end':end,
                    "sensor_name":row.sensor_id}
    df = prod_client.query_df(f"""
        SELECT site_id, device_id, {x_time}, sum_payload_value_hour_curr, eval_prev_hour, sensor_id, entity
        FROM {datasource}
        WHERE 1=1
        AND {x_time} between '{start}' and '{end}'
        AND device_id = '{row.device_id}'
        AND sensor_id = '{row.sensor_id}'
        ORDER BY {x_time} DESC
        settings max_rows_to_read = 10000000000000;""")
    if df.empty:
        error_dict['error'] = "no_data"
        logger.info(json.dumps(error_dict))
        continue
    uniq_ent = df.entity.value_counts().index[0] # filter to remove duplicatas from multiple entities
    df = df[df.entity==uniq_ent]
    y_column = 'sum_payload_value_hour_curr'
    error_dict, flag, y_column, datatype = check_problems(df, error_dict, y_column)
    if not flag :
        export_to_parquet(df, y_column, start, end,
                                row.site_id, row.device_id, row.sensor_id, output_dir)
    logger.info(json.dumps(error_dict))