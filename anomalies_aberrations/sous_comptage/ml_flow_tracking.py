import os
import experiment_run as exp

YEAR = 2026

ANOM_DIR = os.environ.get(
    "ANOM_DIR",
    f"s3://gads-citron-sagemaker-custom-stage/data/anomaly/tertiary_IQ/{YEAR}"
)

BASE_DESCRIPTION = (
    f"Analyse sur les données de sous-comptage clients IQ, année {YEAR} uniquement, tous sites confondus. "
    "Les anomalies sont détectées avec la méthode {method} (params : {params}) avec transformation log10"
    "sur des périodes glissantes de {window} semaines."
)
URI_MLFLOW = 'arn:aws:sagemaker:eu-west-3:585762237892:mlflow-app/app-CZPP4GNRF7VT'
EXP_NAME = f"Sous-Comptage_IQ_{YEAR}_Anomaly-Detection"

# WINDOW = 8
configs = [
    # {"method": "iso_forest", "param": {"th": -0.6}},
    # {"method": "iso_forest", "param": {"th": -0.5}},
    # {"method": "iso_forest", "param": {"th": -0.4}},
    {"method": "iqr",        "param": {"k": 0.2}, "window" : 8},
    {"method": "iqr",        "param": {"k": 0.5}, "window" : 8},
    {"method": "iqr",        "param": {"k": 0.8}, "window" : 8},
    {"method": "iqr",        "param": {"k": 1}, "window" : 8},
    {"method": "iqr",        "param": {"k": 0.2}, "window" : 4},
    {"method": "iqr",        "param": {"k": 0.5}, "window" : 4},
    {"method": "iqr",        "param": {"k": 0.8}, "window" : 4},
    {"method": "iqr",        "param": {"k": 1}, "window" : 4},
    {"method": "iqr",        "param": {"k": 0.2}, "window" : 12},
    {"method": "iqr",        "param": {"k": 0.5}, "window" : 12},
    {"method": "iqr",        "param": {"k": 0.8}, "window" : 12},
    {"method": "iqr",        "param": {"k": 1}, "window" : 12},
    
    # {"method": "stl",        "param": {"k": 1.5}},
    # {"method": "stl",        "param": {"k": 3}},
    # {"method": "stl",        "param": {"k": 10}},
    # {"method": "arima",      "param": {"p": 6, "d": 1, "q": 1}},
    # {"method": "arima",      "param": {"p": 12, "d": 1, "q": 2}},
    # {"method": "arima",      "param": {"p": 24, "d": 1, "q": 1}},
]

def run_cfg(cfg):
    params_str = ", ".join(f"{k}={v}" for k, v in cfg["param"].items())
    description = BASE_DESCRIPTION.format(
        method=cfg["method"],
        params=params_str,
        window=cfg['window'],
    )
    exp.run(
        exp_name=EXP_NAME,
        uri_mlflow=URI_MLFLOW,
        method=cfg["method"],
        window=cfg['window'],
        param=cfg["param"],
        anom_dir=ANOM_DIR,
        description=description,
        log10=True,
    )
for cfg in configs:
    run_cfg(cfg)
