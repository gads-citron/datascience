import json
import logging
import os

import boto3
import clickhouse_connect
from dotenv import load_dotenv

import param as p

logger = logging.getLogger(__name__)


class ClickHouseClient:
    def __init__(self):
        secret_name = p.SECRET
        host, username, password = self.sage_secret(secret_name)
        # host, username, password = self.local_secret()
        self.client = clickhouse_connect.get_client(
            host=host,
            username=username,
            port=443,
            password=password,
            secure=True,
        )

    def local_secret(self):
        load_dotenv()
        values = {
            "clk_prod": os.environ.get("clk_prod"),
            "USER_NAME": os.environ.get("USER_NAME"),
            "clk_prod_pass": os.environ.get("clk_prod_pass"),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise RuntimeError(
                "Variables d'environnement manquantes : " + ", ".join(missing)
            )
        return values["clk_prod"], values["USER_NAME"], values["clk_prod_pass"]

    def sage_secret(self, secret_name):
        region_name = "eu-west-3"
        session = boto3.session.Session()
        sm_client = session.client(
            service_name="secretsmanager",
            region_name=region_name,
        )
        get_secret_value_response = sm_client.get_secret_value(SecretId=secret_name)
        secret = json.loads(get_secret_value_response["SecretString"])
        return secret["host"], secret["username"], secret["password"]

    def query(self, q, parameters=None):
        return self.client.query_df(q, parameters=parameters)

    def insert_df(self, table, df):
        return self.client.insert_df(table, df)

    def command(self, q):
        return self.client.command(q)


class Collector:
    def __init__(self, client):
        self.client = client

    def get_site_pairs(self, site_id):
        query = f"""SELECT DISTINCT customer_account_name, site_name, site_id, device_id, device_name, sensor_id, sensor_name
        FROM {p.DEVICE_PROPERTIES}
        WHERE site_id = {{site_id:String}}
        AND lower(device_type) LIKE '%lectr%'
        AND lower(sensor_name) LIKE '%index%'
        AND lower(sensor_name) NOT LIKE '%(cons)%'"""
        return query, {"site_id": str(site_id)}

    def query_by_site(self, pairs_df, start, end):
        device_ids = [str(d) for d in pairs_df["device_id"].dropna().unique()]
        sensor_ids = [str(s) for s in pairs_df["sensor_id"].dropna().unique()]
        query = f"""SELECT device_id, payload_metric_id AS sensor_id, payload_timestamp,
            toFloat64OrNull(payload_value) AS raw_value
        FROM {p.FACT_TABLE}
        WHERE payload_metric_id IN {{sensor_ids:Array(String)}}
        AND device_id IN {{device_ids:Array(String)}}
        AND payload_timestamp BETWEEN {{start:String}} AND {{end:String}}
        ORDER BY payload_timestamp"""
        return query, {
            "sensor_ids": sensor_ids,
            "device_ids": device_ids,
            "start": start,
            "end": end,
        }

    def get_first_payloads(self, pairs_df):
        device_ids = [str(d) for d in pairs_df["device_id"].dropna().unique()]
        sensor_ids = [str(s) for s in pairs_df["sensor_id"].dropna().unique()]
        query = f"""SELECT device_id, payload_metric_id AS sensor_id,
                minOrNull(payload_timestamp) AS first_ts
            FROM {p.FACT_TABLE}
            WHERE device_id IN {{device_ids:Array(String)}}
            AND payload_metric_id IN {{sensor_ids:Array(String)}}
            AND toFloat64OrNull(payload_value) != 0
            GROUP BY device_id, payload_metric_id"""
        return self.client.query(
            query,
            {"device_ids": device_ids, "sensor_ids": sensor_ids},
        )


class Writer:
    def __init__(self, client):
        self.client = client

    def write(self, df, out_table):
        try:
            self.client.insert_df(out_table, df)
        except Exception:
            logger.exception("Écriture dans %s : échec (%d lignes)", out_table, len(df))
            return 0
        logger.info("Écriture dans %s : succès (%d lignes)", out_table, len(df))
        return 1
