import argparse
import mlflow
import pandas as pd
import pyarrow.dataset as ds
from anomaly_detector import Detector
from sklearn.metrics import precision_score, recall_score, f1_score

def load_all_parquets(folder: str) -> pd.DataFrame:
    """ charge tous les fichiers parquet partitionnés depuis un dossier local ou S3 (s3://bucket/path/) """
    dataset = ds.dataset(folder, format="parquet", partitioning="hive")
    return dataset.to_table().to_pandas()

def classify_data(smp, method, w, param):
    """ avec un jeu de données """
    detector = Detector(method, param)
    smp['raw_value'] = smp.raw_value.interpolate()
    anomalies = []
    smp = smp.sort_values('payload_expected_time')
    delta = pd.Timedelta(weeks=w)
    current = smp['payload_expected_time'].min()
    end_time = smp['payload_expected_time'].max()
    while current < end_time:
        mask_window = (smp['payload_expected_time'] >= current) & (smp['payload_expected_time'] < current + delta)
        window = smp.loc[mask_window].copy()
        if not window.empty:
            detector.predict(window, "raw_value")
            mask = window['anomaly'] == 1
            anomalies.append(
                window.loc[mask, ['customer_account_name', 'site_id', 'device_id', 'sensor_id', 'device_name', 'sensor_name',
                               'payload_expected_time', 'feature', 'value', 'med_max20', 'anomaly']])
        current += delta
    return pd.concat(anomalies, ignore_index=True) if anomalies else pd.DataFrame()

def run(method, window, param, anom_dir, description=""):
    #%% load data and process anomaly classification experiment
    df = load_all_parquets(anom_dir)
    anom_df = pd.concat(
        [classify_data(g, method, window, param) for _, g in df.groupby(['site_id', 'device_id', 'sensor_id'])],
        ignore_index=True
    )
    vt = pd.read_csv(f"{anom_dir}/verite_terrain.csv")
    merged = vt.merge(
        anom_df[['site_id', 'device_id', 'sensor_id', 'payload_expected_time', 'anomaly']],
        on=['site_id', 'device_id', 'sensor_id', 'payload_expected_time'],
        how='left'
    )
    merged['anomaly'] = merged['anomaly'].fillna(0).astype(int)
    mlflow.set_tracking_uri("URI_MLFLOW")
    mlflow.set_experiment("Sous-Comptage_Anomaly-Detection")
    datetime = pd.Timestamp.now().strftime("%m%d-%H%M")
    with mlflow.start_run(run_name=f"{method}-{param}-{datetime}"):
        mlflow.set_tag("mlflow.note.content", description)
        mlflow.sklearn.autolog()
        mlflow.log_param("method", method)
        mlflow.log_param("param", param)
        mlflow.log_metric("precision", precision_score(merged['VT_myriam'], merged['anomaly']))
        mlflow.log_metric("recall", recall_score(merged['VT_myriam'], merged['anomaly']))
        mlflow.log_metric("f1", f1_score(merged['VT_myriam'], merged['anomaly']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default="iso_forest", help="iqr, stl, iso_forest")
    parser.add_argument('--param', type=lambda x: {x.split('=')[0]: float(x.split('=')[1])}, default={}, help="ex: k=5 ou th=2")
    parser.add_argument('--window', type=int, default=4, help="fenêtre en nombre de semaines")
    parser.add_argument('--anom_dir', type=str, required=True)
    parser.add_argument('--description', type=str, default="")
    args = parser.parse_args()
    run(args.method, args.window, args.param, args.anom_dir, args.description)