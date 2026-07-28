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
    # --- IQR, log10=True, k=[0.3;0.4;0.5;0.7;0.8] ---
    {"method": "iqr", "param": {"k": 0.3}, "window": 2, "log10": True},
    {"method": "iqr", "param": {"k": 0.4}, "window": 2, "log10": True},
    {"method": "iqr", "param": {"k": 0.5}, "window": 2, "log10": True},
    {"method": "iqr", "param": {"k": 0.7}, "window": 2, "log10": True},
    {"method": "iqr", "param": {"k": 0.8}, "window": 2, "log10": True},
    {"method": "iqr", "param": {"k": 0.3}, "window": 4, "log10": True},
    {"method": "iqr", "param": {"k": 0.4}, "window": 4, "log10": True},
    {"method": "iqr", "param": {"k": 0.5}, "window": 4, "log10": True},
    {"method": "iqr", "param": {"k": 0.7}, "window": 4, "log10": True},
    {"method": "iqr", "param": {"k": 0.8}, "window": 4, "log10": True},
    {"method": "iqr", "param": {"k": 0.3}, "window": 8, "log10": True},
    {"method": "iqr", "param": {"k": 0.4}, "window": 8, "log10": True},
    {"method": "iqr", "param": {"k": 0.5}, "window": 8, "log10": True},
    {"method": "iqr", "param": {"k": 0.7}, "window": 8, "log10": True},
    {"method": "iqr", "param": {"k": 0.8}, "window": 8, "log10": True},
    {"method": "iqr", "param": {"k": 0.3}, "window": 12, "log10": True},
    {"method": "iqr", "param": {"k": 0.4}, "window": 12, "log10": True},
    {"method": "iqr", "param": {"k": 0.5}, "window": 12, "log10": True},
    {"method": "iqr", "param": {"k": 0.7}, "window": 12, "log10": True},
    {"method": "iqr", "param": {"k": 0.8}, "window": 12, "log10": True},

    # --- IQR, log10=False, k=[1.5;2;4;10;100;1e3] ---
    {"method": "iqr", "param": {"k": 1.5}, "window": 2, "log10": False},
    {"method": "iqr", "param": {"k": 2}, "window": 2, "log10": False},
    {"method": "iqr", "param": {"k": 4}, "window": 2, "log10": False},
    {"method": "iqr", "param": {"k": 10}, "window": 2, "log10": False},
    {"method": "iqr", "param": {"k": 100}, "window": 2, "log10": False},
    {"method": "iqr", "param": {"k": 1e3}, "window": 2, "log10": False},
    {"method": "iqr", "param": {"k": 1.5}, "window": 4, "log10": False},
    {"method": "iqr", "param": {"k": 2}, "window": 4, "log10": False},
    {"method": "iqr", "param": {"k": 4}, "window": 4, "log10": False},
    {"method": "iqr", "param": {"k": 10}, "window": 4, "log10": False},
    {"method": "iqr", "param": {"k": 100}, "window": 4, "log10": False},
    {"method": "iqr", "param": {"k": 1e3}, "window": 4, "log10": False},
    {"method": "iqr", "param": {"k": 1.5}, "window": 8, "log10": False},
    {"method": "iqr", "param": {"k": 2}, "window": 8, "log10": False},
    {"method": "iqr", "param": {"k": 4}, "window": 8, "log10": False},
    {"method": "iqr", "param": {"k": 10}, "window": 8, "log10": False},
    {"method": "iqr", "param": {"k": 100}, "window": 8, "log10": False},
    {"method": "iqr", "param": {"k": 1e3}, "window": 8, "log10": False},
    {"method": "iqr", "param": {"k": 1.5}, "window": 12, "log10": False},
    {"method": "iqr", "param": {"k": 2}, "window": 12, "log10": False},
    {"method": "iqr", "param": {"k": 4}, "window": 12, "log10": False},
    {"method": "iqr", "param": {"k": 10}, "window": 12, "log10": False},
    {"method": "iqr", "param": {"k": 100}, "window": 12, "log10": False},
    {"method": "iqr", "param": {"k": 1e3}, "window": 12, "log10": False},

    # --- iso_forest, log10=True, th=[-0.2;-0.3;-0.35;-0.4;-0.5] ---
    {"method": "iso_forest", "param": {"th": -0.2}, "window": 2, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.3}, "window": 2, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.35}, "window": 2, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.4}, "window": 2, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.5}, "window": 2, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.2}, "window": 4, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.3}, "window": 4, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.35}, "window": 4, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.4}, "window": 4, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.5}, "window": 4, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.2}, "window": 8, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.3}, "window": 8, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.35}, "window": 8, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.4}, "window": 8, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.5}, "window": 8, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.2}, "window": 12, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.3}, "window": 12, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.35}, "window": 12, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.4}, "window": 12, "log10": True},
    {"method": "iso_forest", "param": {"th": -0.5}, "window": 12, "log10": True},

    # --- stl, log10=True et log10=False, k=[5;10;100;1e3] ---
    {"method": "stl", "param": {"k": 5}, "window": 2, "log10": True},
    {"method": "stl", "param": {"k": 10}, "window": 2, "log10": True},
    {"method": "stl", "param": {"k": 100}, "window": 2, "log10": True},
    {"method": "stl", "param": {"k": 1e3}, "window": 2, "log10": True},
    {"method": "stl", "param": {"k": 5}, "window": 2, "log10": False},
    {"method": "stl", "param": {"k": 10}, "window": 2, "log10": False},
    {"method": "stl", "param": {"k": 100}, "window": 2, "log10": False},
    {"method": "stl", "param": {"k": 1e3}, "window": 2, "log10": False},
    {"method": "stl", "param": {"k": 5}, "window": 4, "log10": True},
    {"method": "stl", "param": {"k": 10}, "window": 4, "log10": True},
    {"method": "stl", "param": {"k": 100}, "window": 4, "log10": True},
    {"method": "stl", "param": {"k": 1e3}, "window": 4, "log10": True},
    {"method": "stl", "param": {"k": 5}, "window": 4, "log10": False},
    {"method": "stl", "param": {"k": 10}, "window": 4, "log10": False},
    {"method": "stl", "param": {"k": 100}, "window": 4, "log10": False},
    {"method": "stl", "param": {"k": 1e3}, "window": 4, "log10": False},
    {"method": "stl", "param": {"k": 5}, "window": 8, "log10": True},
    {"method": "stl", "param": {"k": 10}, "window": 8, "log10": True},
    {"method": "stl", "param": {"k": 100}, "window": 8, "log10": True},
    {"method": "stl", "param": {"k": 1e3}, "window": 8, "log10": True},
    {"method": "stl", "param": {"k": 5}, "window": 8, "log10": False},
    {"method": "stl", "param": {"k": 10}, "window": 8, "log10": False},
    {"method": "stl", "param": {"k": 100}, "window": 8, "log10": False},
    {"method": "stl", "param": {"k": 1e3}, "window": 8, "log10": False},
    {"method": "stl", "param": {"k": 5}, "window": 12, "log10": True},
    {"method": "stl", "param": {"k": 10}, "window": 12, "log10": True},
    {"method": "stl", "param": {"k": 100}, "window": 12, "log10": True},
    {"method": "stl", "param": {"k": 1e3}, "window": 12, "log10": True},
    {"method": "stl", "param": {"k": 5}, "window": 12, "log10": False},
    {"method": "stl", "param": {"k": 10}, "window": 12, "log10": False},
    {"method": "stl", "param": {"k": 100}, "window": 12, "log10": False},
    {"method": "stl", "param": {"k": 1e3}, "window": 12, "log10": False},
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
        log10=cfg['log10'],
    )
for cfg in configs:
    run_cfg(cfg)
