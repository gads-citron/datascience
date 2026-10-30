from dotenv import load_dotenv
import json
import os
import boto3
import clickhouse_connect

from paths import LOCAL  # interrupteur local / SageMaker, défini une seule fois dans paths.py

REGION_NAME = "eu-west-3"
SECRETS = {
    "stage": "AmazonSageMaker-clickhouse-user-1",
    "prod": "AmazonSageMaker-clickhouse-user-prod",
}
# Variables d'environnement (.env) lues en local : (hôte, utilisateur, mot de passe)
LOCAL_VARS = {
    "stage": ("clk_stg", "usr_name", "clk_stg_pass"),
    "prod": ("clk_prod", "usr_name", "clk_prod_pass"),
}


class ClickHouseClient:
    def __init__(self, env, local=LOCAL):
        """env : 'prod' ou 'stage'. local=True lit les identifiants dans le .env / l'environnement,
        local=False dans le secret AWS Secrets Manager (SageMaker)."""
        self.env = 'prod' if env == 'prod' else 'stage'
        host, username, password = self.local_secret() if local else self.sagemaker_secret(SECRETS[self.env])
        self.client = clickhouse_connect.get_client(
            host=host, username=username, port=443, password=password)

    def local_secret(self):
        load_dotenv()
        names = LOCAL_VARS[self.env]
        values = {name: os.environ.get(name) for name in names}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise RuntimeError("Variables d'environnement manquantes : " + ", ".join(missing))
        return tuple(values[name] for name in names)

    def sagemaker_secret(self, secret_name):
        session = boto3.session.Session()
        sm_client = session.client(service_name="secretsmanager", region_name=REGION_NAME)
        response = sm_client.get_secret_value(SecretId=secret_name)
        secret = json.loads(response["SecretString"])
        return secret["host"], secret["username"], secret["password"]

    def query(self, q):
        return self.client.query_df(q)

    def command(self, q):
        return self.client.command(q)
