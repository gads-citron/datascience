import os
import experiment_run as exp

ANOM_DIR = os.environ.get(
    "ANOM_DIR",
    "s3://gads-citron-sagemaker-custom-stage/data/anomaly/tertiary_IQ/2026"
)

BASE_DESCRIPTION = (
    "Analyse sur les données de sous-comptage clients IQ, année 2026 uniquement, tous sites confondus. "
    "Les anomalies sont détectées avec la méthode {method} (params : {params}) avec transformation log10"
    "sur des périodes glissantes de {window} semaines."
)
URI_MLFLOW = 'arn:aws:sagemaker:eu-west-3:585762237892:mlflow-app/app-N4LROSLQN5BN'
EXP_NAME = "Sous-Comptage_IQ_2026_Anomaly-Detection"

WINDOW = 8
configs = [
    {"method": "iso_forest", "param": {"th": -0.6}},
    {"method": "iso_forest", "param": {"th": -0.5}},
    {"method": "iso_forest", "param": {"th": -0.4}},
    {"method": "iqr",        "param": {"k": 1.5}},
    {"method": "iqr",        "param": {"k": 5}},
    {"method": "iqr",        "param": {"k": 10}},
    {"method": "stl",        "param": {"k": 1.5}},
    {"method": "stl",        "param": {"k": 3}},
    {"method": "stl",        "param": {"k": 10}},
    {"method": "arima",      "param": {"p": 6, "d": 1, "q": 1}},
    {"method": "arima",      "param": {"p": 12, "d": 1, "q": 2}},
    {"method": "arima",      "param": {"p": 24, "d": 1, "q": 1}},
]

def run_cfg(cfg):
    params_str = ", ".join(f"{k}={v}" for k, v in cfg["param"].items())
    description = BASE_DESCRIPTION.format(
        method=cfg["method"],
        params=params_str,
        window=WINDOW,
    )
    exp.run(
        exp_name=EXP_NAME,
        uri_mlflow=URI_MLFLOW,
        method=cfg["method"],
        window=WINDOW,
        param=cfg["param"],
        anom_dir=ANOM_DIR,
        description=description,
        log10=True,
    )
for cfg in configs:
    run_cfg(cfg)
