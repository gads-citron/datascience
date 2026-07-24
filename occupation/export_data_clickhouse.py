import io
import boto3
import os
import pandas as pd
import json
import clickhouse_connect
from tqdm import tqdm

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

def db_connect(dtb='prod'):  
    secret = get_clickhouse_credentials(db=dtb)
    host = secret['host']
    username = secret['username']
    password = secret['password']
    conn = clickhouse_connect.get_client(host=host, username=username, port=443, password=password)
    return conn

def read_parquet_from_s3(key: str) -> pd.DataFrame:
    response = _s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(response['Body'].read()))

S3_BUCKET = "gads-citron-sagemaker-custom-stage"
S3_PREFIX = "data/occupation/site_occupancy/"
OUTPUT_DATASET = 'Myriam_test.site_occupancy'
_s3 = boto3.client('s3')

prefix = S3_PREFIX.rstrip('/') + '/'
response = _s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, Delimiter='/')
site_prefixes = [
    cp['Prefix'] for cp in response.get('CommonPrefixes', [])
    if cp['Prefix'].rstrip('/').split('/')[-1].startswith('site_id=')
]
# Récupération des customer_account_name depuis ClickHouse prod
site_ids = [p.rstrip('/').split('=', 1)[-1] for p in site_prefixes]
clk_stg = db_connect('stage')
# clk_prd  = db_connect('prod')
# print('prod connectée')
account_df = clk_stg.query_df(f"""
     SELECT id_du_site site_id, site_name, customer_account_name
     FROM datawarehouse_stage.device_big_table
     WHERE id_du_site IN {site_ids}
     GROUP BY id_du_site, site_name, customer_account_name
""")
account_map   = dict(zip(account_df['site_id'].astype(str), account_df['customer_account_name']))
site_name_map = dict(zip(account_df['site_id'].astype(str), account_df['site_name']))

paginator = _s3.get_paginator('list_objects_v2')

for site_prefix in tqdm(site_prefixes):
    site_id = site_prefix.rstrip('/').split('/')[-1].split('=', 1)[-1]
    site_name = site_name_map[site_id]
    cust_name = account_map[site_id]

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=site_prefix):
        for obj in page.get('Contents', []):
            try:
                df = read_parquet_from_s3(obj['Key'])
            except Exception as e:
                print('An error occurred during data reading from S3', e)
                continue
            df['site_id'] = site_id
            df['site_name'] = site_name
            df['customer_account_name'] = cust_name
            # Écriture dans ClickHouse stage
            try:
                clk_stg.insert_df(OUTPUT_DATASET, df.drop('period', axis=1))
            except Exception as e:
                print('An error occurred during data upload to clickhouse', e)
