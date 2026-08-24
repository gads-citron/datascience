"""
Construction des datasets d'entraînement/test et des séquences pour le TCN-GRU.

@author: Myriam Lopez
"""

import io
import os
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import boto3

from db import collect_power_data
from features import PREPROCESSORS, feature_creator

LOCAL_DIR = "./data/site_occupancy"

# déclarée dans train_model.py : lecture locale (False) ou sur S3/SageMaker (True)
SAGE_ENV = False
S3_BUCKET = os.environ.get("S3_BUCKET", "gads-citron-sagemaker-custom-stage")
S3_PREFIX = "data/occupation/site_occupancy/"
# donnée brute élec, produite par ground_truth_pipeline/feed_s3.py (mélange eau/élec, non filtré ici)
POWER_S3_PREFIX = "data/occupation/power_clk_collect/"
# devices écartés par le filtre SVM (ground_truth_pipeline/Fusion_seasonalities.py)
REJECTED_SVM_LOCAL = "../rejected_by_svm.csv"

s3_client = boto3.client("s3")
_rejected_df = None

def _read_parquets_from_s3(prefix):
    keys = [
        obj["Key"]
        for obj in s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix).get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]
    if not keys:
        return pd.DataFrame()
    return pd.concat(
        [
            pd.read_parquet(io.BytesIO(s3_client.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()))
            for key in keys
        ],
        ignore_index=True,
    )


def _load_rejected_df():
    global _rejected_df
    if _rejected_df is None:
        if SAGE_ENV:
            key = f"{POWER_S3_PREFIX}rejected_by_svm.csv"
            body = s3_client.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
            _rejected_df = pd.read_csv(io.BytesIO(body))
        else:
            _rejected_df = pd.read_csv(REJECTED_SVM_LOCAL)
    return _rejected_df


def collect_power_data_s3(site_id, start, end):
    df = _read_parquets_from_s3(f"{POWER_S3_PREFIX}site_id={site_id}/")
    if df.empty:
        return pd.DataFrame({
            'payload_expected_time': pd.Series(dtype='datetime64[ns]'),
            'raw_value': pd.Series(dtype='float64'),
        })

    rejected_df = _load_rejected_df()
    rejected_devices = set(rejected_df.loc[rejected_df['site_id'] == site_id, 'device_id'])
    df = df[~df['device_id'].isin(rejected_devices)]

    df = df[(df['payload_expected_time'] >= start) & (df['payload_expected_time'] <= end)]
    return df.groupby('payload_expected_time', as_index=False)['raw_value'].sum()

def list_site_ids():
    if SAGE_ENV:
        site_ids = set()
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX, Delimiter="/"):
            for common_prefix in page.get("CommonPrefixes", []):
                folder = common_prefix["Prefix"].rstrip("/").rsplit("/", 1)[-1]
                if folder.startswith("site_id="):
                    site_ids.add(folder.split("=", 1)[-1])
        return sorted(site_ids)
    return [
        entry.split('=', 1)[-1]
        for entry in os.listdir(LOCAL_DIR)
        if entry.startswith('site_id=') and os.path.isdir(os.path.join(LOCAL_DIR, entry))
    ]


def build_train_set(site_id, preprocess='log_zscore'):
    if preprocess not in PREPROCESSORS:
        raise ValueError(f"preprocess doit être parmi {list(PREPROCESSORS)}, reçu '{preprocess}'")

    if SAGE_ENV:
        occ_df = _read_parquets_from_s3(f"{S3_PREFIX}site_id={site_id}/")
    else:
        parquet_paths = glob.glob(os.path.join(LOCAL_DIR, f"site_id={site_id}", "*.parquet"))
        occ_df = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)

    start = occ_df['payload_timestamp'].min()
    end = occ_df['payload_timestamp'].max()
    raw_data_df = collect_power_data_s3(site_id, start, end) if SAGE_ENV else collect_power_data(site_id, start, end)

    train_df = occ_df.merge(
        raw_data_df,
        left_on='payload_timestamp',
        right_on='payload_expected_time',
        how='left',
    )
    train_df['site_id'] = site_id
    train_df = train_df[['site_id', 'payload_timestamp', 'raw_value', 'mean_occupancy']]
    return PREPROCESSORS[preprocess](train_df)


def build_dataset(site_ids, preprocess='log_zscore'):
    dfs = [feature_creator(build_train_set(site_id, preprocess=preprocess)) for site_id in tqdm(site_ids)]
    return pd.concat(dfs, ignore_index=True)


def build_train_test_datasets(preprocess='log_zscore', test_ratio=0.3, seed=142):
    site_ids = list_site_ids()

    # split par site (pas par date) : le modèle doit être évalué sur des bâtiments jamais vus en entraînement
    rng = random.Random(seed)
    train_site_ids = rng.sample(site_ids, k=round((1 - test_ratio) * len(site_ids)))
    test_site_ids = [s for s in site_ids if s not in train_site_ids]

    train_df = build_dataset(train_site_ids, preprocess=preprocess)
    test_df = build_dataset(test_site_ids, preprocess=preprocess)
    return train_df, test_df


def build_sequences(df, feature_cols, target_col, seq_len=168, stride=24, return_groups=False):
    # une séquence par bâtiment (jamais de fenêtre à cheval sur deux sites)
    X, y, mask, site_ids, timestamps = [], [], [], [], []
    for site_id, site_df in df.groupby('site_id'):
        site_df = site_df.sort_values('payload_timestamp')
        feats = site_df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
        target = site_df[target_col].to_numpy(dtype=np.float32)
        valid = ~np.isnan(target)
        ts = site_df['payload_timestamp'].to_numpy()

        for start in range(0, len(site_df) - seq_len + 1, stride):
            end = start + seq_len
            X.append(feats[start:end])
            y.append(np.nan_to_num(target[start:end]))
            mask.append(valid[start:end])
            if return_groups:
                site_ids.append(np.full(seq_len, site_id))
                timestamps.append(ts[start:end])

    if return_groups:
        return np.stack(X), np.stack(y), np.stack(mask), np.stack(site_ids), np.stack(timestamps)
    return np.stack(X), np.stack(y), np.stack(mask)


class SequenceDataset(Dataset):
    def __init__(self, X, y, mask):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.mask = torch.from_numpy(mask)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]
