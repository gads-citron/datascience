import mlflow
import pandas as pd
import pyarrow.dataset as ds
from anomaly_detector import Detector

def load_all_parquets(folder: str) -> pd.DataFrame:
    """ charge tous les fichiers parquet partitionnés depuis un dossier local ou S3 (s3://bucket/path/) """
    dataset = ds.dataset(folder, format="parquet", partitioning="hive")
    return dataset.to_table().to_pandas()

def classify_data(smp, method, w, param):
    """ avec un jeu de données """
    detector = Detector(smp, 'raw_value')
    smp['raw_value'] smp.raw_value.interpolate()
    for start in range(0, len(smp), w):
        window = smp.iloc[start:start + w].copy()
        detect_fn = getattr(detector, method)
        window['anomaly'] = detect_fn(window, param)
        mask = window['anomaly'] == 1

        anomalies.append(
            window.loc[mask, ['customer_account_name', 'site_id', 'device_id', 'sensor_id', 'device_name', 'sensor_name',
                           'payload_expected_time', 'feature', 'value', 'med_max20', 'anomaly']])


#%% load data and process anomaly classification experiment
METHOD = 
PARAM = 
W = 
df = load_all_parquets("s3://mon-bucket/mon-dossier/")
anomalies = [] # c'est une liste qui va à la fin contenir tous les df d'anomalies détectées au cours de la boucle qui itère sur les fichiers parquets

for i, g in df.groupby(['site_id', 'device_id', 'sensor_id']):
    classify_data(g, METHOD, W, PARAM)

mlflow.set_tracking_uri("URI_MLFLOW")
mlflow.set_experiment("MLflow Quickstart")
with mlflow.start_run(run_name="test_logit_dataset_iris"):
    mlflow.set_tag("mlflow.note.content", """
    Test classification .
    - Données : dataset iris open source de scikit learn
    - Objectif : classer les individus sur la base des features 
    - Params : train/test split avec une proportion 80/20
    - Observation : juste pour l'experience
    """)
    mlflow.sklearn.autolog()
    mlflow.log_param("method", method)
    mlflow.log_param("param", param)
    mlflow.log_metric("precision", precision_score(y_true, df['anomaly']))
    mlflow.log_metric("recall", recall_score(y_true, df['anomaly']))
    mlflow.log_metric("f1", f1_score(y_true, df['anomaly']))