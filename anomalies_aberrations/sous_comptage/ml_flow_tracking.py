import os
import experiment_run as exp

YEAR = 2026

ANOM_DIR = os.environ.get(
    "ANOM_DIR",
    f"s3://gads-citron-sagemaker-custom-stage/data/anomaly/tertiary_IQ/{YEAR}"
)

BASE_DESCRIPTION = (
    f"Analyse sur les données de sous-comptage clients IQ, année {YEAR} uniquement, tous sites confondus. "
)
URI_MLFLOW = 'arn:aws:sagemaker:eu-west-3:585762237892:mlflow-app/app-CZPP4GNRF7VT'
EXP_NAME = f"Sous-Comptage_IQ_{YEAR}_Anomaly-Detection"

# WINDOW = 8
configs = [
    # --- iqr_dynamic_k, a=[0.5;1;1.5;2;5;10], window=[2;4;8;12], log10=[True;False] ---
    {"method": "iqr_dynamic_k", "param": {"a": 0.5}, "window": 2, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 0.5}, "window": 2, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 0.5}, "window": 4, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 0.5}, "window": 4, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 0.5}, "window": 8, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 0.5}, "window": 8, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 0.5}, "window": 12, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 0.5}, "window": 12, "log10": False},

    {"method": "iqr_dynamic_k", "param": {"a": 1}, "window": 2, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 1}, "window": 2, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 1}, "window": 4, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 1}, "window": 4, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 1}, "window": 8, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 1}, "window": 8, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 1}, "window": 12, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 1}, "window": 12, "log10": False},

    {"method": "iqr_dynamic_k", "param": {"a": 1.5}, "window": 2, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 1.5}, "window": 2, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 1.5}, "window": 4, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 1.5}, "window": 4, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 1.5}, "window": 8, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 1.5}, "window": 8, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 1.5}, "window": 12, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 1.5}, "window": 12, "log10": False},

    {"method": "iqr_dynamic_k", "param": {"a": 2}, "window": 2, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 2}, "window": 2, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 2}, "window": 4, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 2}, "window": 4, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 2}, "window": 8, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 2}, "window": 8, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 2}, "window": 12, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 2}, "window": 12, "log10": False},

    {"method": "iqr_dynamic_k", "param": {"a": 5}, "window": 2, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 5}, "window": 2, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 5}, "window": 4, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 5}, "window": 4, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 5}, "window": 8, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 5}, "window": 8, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 5}, "window": 12, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 5}, "window": 12, "log10": False},

    {"method": "iqr_dynamic_k", "param": {"a": 10}, "window": 2, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 10}, "window": 2, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 10}, "window": 4, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 10}, "window": 4, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 10}, "window": 8, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 10}, "window": 8, "log10": False},
    {"method": "iqr_dynamic_k", "param": {"a": 10}, "window": 12, "log10": True},
    {"method": "iqr_dynamic_k", "param": {"a": 10}, "window": 12, "log10": False},
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
