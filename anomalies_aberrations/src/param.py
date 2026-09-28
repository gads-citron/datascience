ENV = "stage"  # "stage" ou "prod"

_SECRETS = {
    "stage": "AmazonSageMaker-clickhouse-user-1",
    "prod": "AmazonSageMaker-clickhouse-user-prod",
}
_FACT_TABLES = {
    "stage": "datawarehouse_stage.fact_payload",
    "prod": "datawarehouse_prod.fact_payload",
}
SECRET = _SECRETS[ENV]
FACT_TABLE = _FACT_TABLES[ENV]
DEVICE_PROPERTIES = "datamart_sous_comptage.device_properties"
OUT_TABLE = "Myriam_test.anomalie_v2"
W = 7  # nombre de points avant pos utilisés pour les features de lag (valeur, delta_t, delta_v)
T = 50  # nombre de points avant pos utilisés pour la médiane de delta_t
M = 5  # taille des fenêtres utilisées pour les moyennes quadratiques de la valeur
NORM_MAX = 100  # borne haute de l'échelle sur laquelle on ramène la fenêtre de T points
