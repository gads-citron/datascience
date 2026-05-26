import os
from joblib import Parallel, delayed
import experiment_run as exp

ANOM_DIR = os.environ.get(
    "ANOM_DIR",
    "s3://gads-citron-sagemaker-custom-stage/data/anomaly/tertiary"
)

BASE_DESCRIPTION = (
    "Analyse sur les données de sous-comptage 1er semestre 2025 uniquement, tous sites confondus. "
    "Les anomalies sont détectées avec la méthode {method} (params : {params}) "
    "sur des périodes glissantes de {window} semaines."
)
WINDOW = 12

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
        method=cfg["method"],
        window=WINDOW,
        param=cfg["param"],
        anom_dir=ANOM_DIR,
        description=description,
    )
Parallel(n_jobs=3, backend="loky")(delayed(run_cfg)(cfg) for cfg in configs)
