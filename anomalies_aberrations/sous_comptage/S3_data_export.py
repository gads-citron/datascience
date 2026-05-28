# import subprocess, sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])

import json
import os
import boto3
import logging
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from db_connector import ClickHouseClient
from anomaly_detector import DataProcessor

#%% S3 destination
S3_BUCKET = os.environ.get("S3_BUCKET", "gads-citron-sagemaker-custom-stage")
S3_PREFIX = os.environ.get("S3_PREFIX", "data/anomaly/tertiary_IQ/")
UPLOAD_TO_S3 = os.environ.get("UPLOAD_TO_S3", "1") == "1"
s3_client = boto3.client("s3") if UPLOAD_TO_S3 else None
prod_client = ClickHouseClient()
y_column = 'eval_prev_hour'
x_time = 'payload_expected_time'
sensor_idx = '662bc1c93f0afb78ad05983d'# Index consommation électricité (kWh)

def upload_to_s3(local_path, site_id, device_id, file_name):
    """Upload un parquet local vers s3://{S3_BUCKET}/{S3_PREFIX}site_id=.../device_id=.../sensor_id=.../."""
    if not UPLOAD_TO_S3:
        return
    key = (
        f"{S3_PREFIX}"
        f"site_id={site_id}/"
        f"device_id={device_id}/"
        f"{file_name}"
    )
    s3_client.upload_file(str(local_path), S3_BUCKET, key)

def export_to_parquet(df, y_col, site_id, device_id, period, output_dir):
    date_col = 'payload_expected_time'
    if df[date_col].dt.tz is not None:
        df[date_col] = df[date_col].dt.tz_convert(None)
    df.sort_values(date_col, inplace=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    partition_dir = (
            output_path /
            f"site_id={site_id}" /
            f"device_id={device_id}"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df['raw_value'] = df[y_col]
    df['site_id'] = site_id
    cols = ['site_id', 'device_id', 'sensor_id', 'payload_expected_time', 'raw_value']

    for year, df_year in df.groupby(df[date_col].dt.year):
        file_name = f"data_{period}.parquet"
        file_path = partition_dir / file_name
        df_year[cols].to_parquet(file_path, index=False)
        upload_to_s3(file_path, site_id, device_id, file_name)

def main(start, end, period, input_dir, output_dir):
    log_path = Path(output_dir) / 'process_log.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)
    sous_comptage = 'datamart_sous_comptage.distributed_dataset_hourly_payloads_merged'
    x_time = 'payload_expected_time'

    list_anom = pd.read_csv(f'{input_dir}/iq_spot_detection_stl_isof_2026.csv')
    candidates = list_anom[['customer_account_name', 'device_id', 'sensor_id']].drop_duplicates()
    for i, row in tqdm(candidates.iterrows(), total=candidates.shape[0]):
        record = {
                "customer_account_name": row.customer_account_name,
                "device_id": row.device_id,
                "sensor_id": row.sensor_id,
            }
        q = f"""SELECT customer_account_name, site_id, device_id, sensor_id, entity, {x_time}, eval_prev_hour, sum_payload_value_hour_curr,
                        device_name, sensor_name
                FROM {sous_comptage}
                WHERE 1=1
                    AND payload_expected_time between '{start}' and '{end}'
                    AND customer_account_name = '{row.customer_account_name.replace("'", "''")}'
                    AND device_id = '{row.device_id}'
                    AND sensor_id  = '{sensor_idx}'
                    ORDER BY payload_expected_time DESC"""
        smp = prod_client.query(q)
        if smp.empty:
            record['error'] = 'No data.'
            logger.info(json.dumps(record))
            continue
        else:
            device_id = smp.device_id.values[0]
            site_id = smp.site_id.values[0]
            smp = smp[smp.entity == smp.entity.value_counts().index[0]]  # dédupliquer la données par client
            smp = smp.copy()
            smp[x_time] = pd.to_datetime(smp[x_time])
            smp[y_column] = smp[y_column].interpolate()
            # smp = smp[~smp['sum_payload_value_hour_curr'].isna()]
            # checking = DataProcessor(record, smp)
            # if 'index' in row.sensor_name.lower():
            #     y_column = 'eval_prev_hour'
            # elif not checking.serie_typology('sum_payload_value_hour_curr', tol=0.3):
            #     y_column = 'sum_payload_value_hour_curr'
            # else:
            #     y_column = 'eval_prev_hour'
            # record['y_column'] = y_column
            try:
                export_to_parquet(smp, y_column, site_id, device_id, period, output_dir)
            except Exception as e:
                record['error'] = str(e)
        logger.info(json.dumps(record))

#%% Collect parameters
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Export données ClickHouse vers Parquet/S3")
    parser.add_argument("--start",      default="2026-01-01 00:00:00", help="Date de début (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--end",        default="2026-06-01 23:59:59", help="Date de fin   (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--period",     default="2026_sem1", help="Nom de la période traitée par les données.")
    parser.add_argument("--output_dir", default="./data/sample_anomalies_IQ_2026", dest="output_dir", help="Dossier de sortie local")
    parser.add_argument("--input_dir",  default="./data/sample_anomalies_IQ_2026", dest="input_dir", help="Dossier de sortie local")
    args = parser.parse_args()
    start      = args.start
    end        = args.end
    output_dir = args.output_dir
    input_dir = args.input_dir
    period = args.period
    main(start, end, period, input_dir, output_dir)