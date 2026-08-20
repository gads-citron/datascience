"""
Connexion ClickHouse et collecte des données brutes de puissance.

@author: Myriam Lopez
"""

import os
import urllib.parse
import clickhouse_connect
from dotenv import load_dotenv


def db_connect(type='prod', envir='local'):
    load_dotenv()
    username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
    if type == 'prod':
        host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
        password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
    elif type == 'stage':
        host = urllib.parse.quote_plus(os.environ.get("clk_stg"))
        password = urllib.parse.quote_plus(os.environ.get('clk_stg_pass'))
    conn = clickhouse_connect.get_client(host=host, username=username, port=443, password=password)
    return conn


def collect_power_data(site_id, start, end):
    conn = db_connect()
    query = f"""
        SELECT payload_expected_time, eval_prev_hour as raw_value, device_id
        FROM datamart_sous_comptage.distributed_dataset_hourly_payloads_merged
        WHERE site_id = '{site_id}'
        and sensor_name like 'Index %lectr%'
        and payload_expected_time between '{start}' and '{end}'
        settings max_rows_to_read=10000000000000
    """
    df = conn.query_df(query)
    df['payload_expected_time'] = df['payload_expected_time'].dt.tz_localize(None)
    df = df.groupby('payload_expected_time', as_index=False)['raw_value'].sum()
    return df
