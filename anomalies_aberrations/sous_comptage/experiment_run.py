import argparse
import contextlib
import mlflow
import boto3
import glob
import tempfile
import os
from pathlib import Path
import duckdb
import pandas as pd
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from anomaly_detector import Detector
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class _Callback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Callback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()

def _is_s3(folder: str) -> bool:
    return folder.startswith("s3://")

def _duckdb_con(folder: str):
    con = duckdb.connect()
    if _is_s3(folder):
        con.execute("INSTALL httpfs; LOAD httpfs; CALL load_aws_credentials();")
    return con

def load_all_parquets(folder: str, site_ids: list) -> pd.DataFrame:
    """ charge tous les fichiers parquet partitionnés depuis un dossier local ou S3 (s3://bucket/path/) """
    if _is_s3(folder):
        con = _duckdb_con(folder)
        paths = [f"{folder}/site_id={s}/**/*.parquet" for s in site_ids]
        return con.execute(f"SELECT * FROM read_parquet({paths}, hive_partitioning=true)").df()
    base = Path(__file__).parent / folder
    files = [f for s in site_ids for f in glob.glob(str(base / f"site_id={s}" / "**" / "*.parquet"), recursive=True)]
    return duckdb.sql(f"SELECT * FROM read_parquet({[str(f) for f in files]})").df()

def classify_data(smp, method, w, param, window_type="tumbling", log10=False):
    """ avec un jeu de données """
    detector = Detector(method, param, log10=log10)
    smp['raw_value'] = smp.raw_value.interpolate()
    smp = smp.sort_values('payload_expected_time')
    delta = pd.Timedelta(weeks=w)
    anomalies = []

    if window_type == "centered":
        half_delta = delta / 2
        for _, row in smp.iterrows():
            t = row['payload_expected_time']
            mask_window = (smp['payload_expected_time'] >= t - half_delta) & \
                          (smp['payload_expected_time'] <= t + half_delta)
            ctx = smp.loc[mask_window].copy()
            if ctx.empty:
                continue
            ctx = detector.predict(ctx, "raw_value")
            center = ctx.loc[ctx['payload_expected_time'] == t]
            if not center.empty and center['anomaly'].iloc[0] == 1:
                anomalies.append(center)
    else:
        current = smp['payload_expected_time'].min()
        end_time = smp['payload_expected_time'].max()
        while current < end_time:
            mask_window = (smp['payload_expected_time'] >= current) & (smp['payload_expected_time'] < current + delta)
            window = smp.loc[mask_window].copy()
            if not window.empty:
                window = detector.predict(window, "raw_value")
                mask = window['anomaly'] == 1
                anomalies.append(window.loc[mask])
            current += delta

    return pd.concat(anomalies, ignore_index=True) if anomalies else pd.DataFrame()

def run(exp_name, uri_mlflow, method, window, param, anom_dir, description="", window_type="tumbling", log10=False):
    if _is_s3(anom_dir):
        parts = anom_dir.replace("s3://", "", 1).split("/", 1)
        bucket, prefix = parts[0], parts[1].rstrip("/")
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=f"{prefix}/verite_terrain.csv")
        vt = pd.read_csv(obj["Body"])
    else:
        vt = pd.read_csv(f"{anom_dir}/verite_terrain.csv")
    vt['payload_expected_time'] = pd.to_datetime(vt['payload_expected_time'],
                                                  format='mixed', utc=True).dt.tz_convert(None)
    site_ids = vt.loc[vt['VT_myriam'] == 1, 'site_id'].unique().tolist()
    df = load_all_parquets(anom_dir, site_ids=site_ids)
    groups = list(df.groupby(['site_id', 'device_id', 'sensor_id']))
    with tqdm_joblib(tqdm(total=len(groups), desc="Classification")):
        results = Parallel(n_jobs=-1)(
            delayed(classify_data)(g, method, window, param, window_type, log10)
            for _, g in groups
        )
    anom_df = pd.concat([r for r in results if not r.empty], ignore_index=True)
    merged = vt.merge(
        anom_df[['site_id', 'device_id', 'sensor_id', 'payload_expected_time', 'anomaly']],
        on=['site_id', 'device_id', 'sensor_id', 'payload_expected_time'],
        how='left'
    )
    merged['anomaly'] = merged['anomaly'].fillna(0).astype(int)
    tn, fp, fn, tp = confusion_matrix(merged['VT_myriam'], merged['anomaly']).ravel()
    mlflow.set_tracking_uri(uri_mlflow)
    mlflow.set_experiment(exp_name)
    datetime = pd.Timestamp.now().strftime("%m%d-%H%M")
    with mlflow.start_run(run_name=f"{method}-{param}-{datetime}"):
        mlflow.set_tag("mlflow.note.content", description)
        mlflow.sklearn.autolog()
        mlflow.log_param("method", method)
        mlflow.log_param("window", window)
        mlflow.log_param("window_type", window_type)
        mlflow.log_param("param", param)
        mlflow.log_param("log10", log10)
        mlflow.log_metric("precision", precision_score(merged['VT_myriam'], merged['anomaly']))
        mlflow.log_metric("recall", recall_score(merged['VT_myriam'], merged['anomaly']))
        mlflow.log_metric("f1", f1_score(merged['VT_myriam'], merged['anomaly']))
        n_pos = tp + fn
        n_neg = tn + fp
        mlflow.log_metric("TP", tp)
        mlflow.log_metric("taux_FN", fn / n_pos)
        mlflow.log_metric("TN", tn)
        mlflow.log_metric("taux_FP", fp / n_neg)

        ref_cols = ['site_id', 'device_id', 'sensor_id', 'payload_expected_time']
        fp_df = merged.loc[(merged['VT_myriam'] == 0) & (merged['anomaly'] == 1), ref_cols]
        fn_df = merged.loc[(merged['VT_myriam'] == 1) & (merged['anomaly'] == 0), ref_cols]

        with tempfile.TemporaryDirectory() as tmp:
            fp_path = os.path.join(tmp, "false_positives.csv")
            fn_path = os.path.join(tmp, "false_negatives.csv")
            fp_df.to_csv(fp_path, index=False)
            fn_df.to_csv(fn_path, index=False)
            mlflow.log_artifact(fp_path)
            mlflow.log_artifact(fn_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default="iso_forest", help="iqr, stl, iso_forest")
    parser.add_argument('--param', type=lambda x: {x.split('=')[0]: float(x.split('=')[1])}, default={}, help="ex: k=5 ou th=2")
    parser.add_argument('--window', type=int, default=4, help="fenêtre en nombre de semaines")
    parser.add_argument('--anom_dir', type=str, required=True)
    parser.add_argument('--window_type', type=str, default="tumbling", choices=["tumbling", "centered"])
    parser.add_argument('--description', type=str, default="")
    parser.add_argument('--log10', action='store_true', help="applique log10(x+1) avant détection")
    args = parser.parse_args()
    run(args.method, args.window, args.param, args.anom_dir, args.description, args.window_type, args.log10)