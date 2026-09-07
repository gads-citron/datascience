import clickhouse_connect
import boto3
import json
from dotenv import load_dotenv
import os
import urllib.parse
import param as p

class ClickHouseClient:
    def __init__(self):
        secret_name = p.SECRET
        host, username, password = self.sage_secret(secret_name)
        # host, username, password = self.local_secret()
        self.client = clickhouse_connect.get_client(
                        host=host, username=username, 
                        port=443, password=password)
        
    def local_secret(self):
        load_dotenv()
        host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
        username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
        password = urllib.parse.quote_plus(os.environ.get("clk_prod_pass"))
        return host, username, password

    def sage_secret(self, secret_name):
        region_name = "eu-west-3"
        session = boto3.session.Session()
        sm_client = session.client(
            service_name = 'secretsmanager',
            region_name = region_name
        )
        get_secret_value_response = sm_client.get_secret_value(
                SecretId=secret_name)
        secret = json.loads(get_secret_value_response['SecretString'])
        host = secret['host']
        username = secret['username']
        password = secret['password']
        return host, username, password
    
    def query(self, q):
        return self.client.query_df(q)

    def insert_df(self, table, df):
        return self.client.insert_df(table, df)

class Collector:
    def __init__(self, client):
        self.client = client

    def query_serie(self, dev_id, sens_id, start, end):
        # dataset = 'datawarehouse_prod.fact_payload'
        dataset = 'datawarehouse_stage_fact_payload'
        q = f"""SELECT
            device_id,
            payload_metric_id AS sensor_id,
            toUnixTimestamp(fp.payload_timestamp) AS ts,
            payload_timestamp,
            toFloat32OrNull(fp.payload_value) AS raw_value
        FROM {dataset} AS fp
        WHERE fp.device_id = '{dev_id}'
        AND fp.payload_metric_id = '{sens_id}'
        AND fp.payload_timestamp BETWEEN '{start}' AND '{end}'
        ORDER BY fp.payload_timestamp;"""
        return self.client.query(q)

    def patrimonial(self, dev_id, sens_id):
        dataset = 'datamart_sous_comptage.device_properties'
        q = f"""SELECT distinct customer_account_name, site_name, site_id,
            device_name, device_id, sensor_name, sensor_id
            FROM {dataset}
            WHERE device_id = '{dev_id}'
            AND sensor_id = '{sens_id}'"""
        return self.client.query(q)

    def get_first_payload(self, dev_id, sens_id):
        dataset = 'datawarehouse_stage_fact_payload'
        # dataset = 'datawarehouse_prod.fact_payload'
        q = f"""SELECT min(payload_timestamp)            
                FROM {dataset} AS fp
                WHERE fp.device_id = '{dev_id}'
                AND fp.payload_metric_id = '{sens_id}'"""
        return self.client.query(q)


class Writer:
    def __init__(self, client):
        self.client = client

    def write(self, df, out_table):
        try:
            self.client.insert_df(out_table, df)
            return 1
        except Exception as e:
            print(f"""An error occurred during data upload to table {out_table}""", e)
            return 0

