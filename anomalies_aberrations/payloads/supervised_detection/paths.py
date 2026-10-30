"""
Chemins d'accès aux données (séries parquet, vérité terrain, modèles) adaptatifs : la racine est
la variable d'environnement DATA_ROOT si elle est définie (URL s3://bucket/prefix, lue/écrite via
boto3), sinon le dossier de ce module (local). Les mêmes sous-dossiers (training_data/, models/,
...) sont utilisés partout, en local comme sur S3.
Tous les accès fichiers passent par les helpers ci-dessous (join, glob, exists, read_parquet,
to_parquet, read_csv, to_csv, load_joblib, dump_joblib), qui acceptent indifféremment un chemin
local ou une URL s3://.
"""
import fnmatch
import glob as _glob
import io
import os

import joblib
import pandas as pd

# Interrupteur unique local / SageMaker : True = données dans ce dossier et identifiants ClickHouse
# du .env ; False = données sur S3 (S3_ROOT) et identifiants du secret AWS (voir db_connector.py).
LOCAL = True
S3_ROOT = 's3://gads-citron-sagemaker-custom-stage/data/anomaly'  # racine des données sur S3
DATA_ROOT = os.path.dirname(os.path.abspath(__file__)) if LOCAL else S3_ROOT

_s3_client = None

def _s3():
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client('s3')
    return _s3_client


def is_remote(path):
    return str(path).startswith('s3://')


def _split(url):
    bucket, _, key = str(url)[len('s3://'):].partition('/')
    return bucket, key


def join(*parts):
    """os.path.join pour un chemin local, jointure avec '/' pour une URL."""
    if is_remote(parts[0]):
        return '/'.join([str(parts[0]).rstrip('/')] + [str(p).strip('/') for p in parts[1:]])
    return os.path.join(*parts)


def data_path(*parts):
    """Chemin sous DATA_ROOT, ex: data_path('training_data', 'ruptures')."""
    return join(DATA_ROOT, *parts)


def basename(path):
    return str(path).rstrip('/').rsplit('/', 1)[-1] if is_remote(path) else os.path.basename(path)


def dirname(path):
    return str(path).rstrip('/').rsplit('/', 1)[0] if is_remote(path) else os.path.dirname(path)


def same_dir(a, b):
    """True si a et b désignent le même dossier."""
    if is_remote(a) or is_remote(b):
        return str(a).rstrip('/') == str(b).rstrip('/')
    return os.path.abspath(a) == os.path.abspath(b)


def glob(pattern):
    """Chemins (ou URL s3://) correspondant au motif, triés."""
    if not is_remote(pattern):
        return sorted(_glob.glob(pattern))
    bucket, key_pattern = _split(pattern)
    prefix = key_pattern.split('*', 1)[0].split('?', 1)[0].split('[', 1)[0]
    matches = []
    for page in _s3().get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            if fnmatch.fnmatchcase(obj['Key'], key_pattern):
                matches.append(f"s3://{bucket}/{obj['Key']}")
    return sorted(matches)


def exists(path):
    if not is_remote(path):
        return os.path.exists(path)
    from botocore.exceptions import ClientError
    bucket, key = _split(path)
    try:
        _s3().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] in ('404', 'NoSuchKey', 'NotFound'):
            return False
        raise


def makedirs(path):
    """Crée le dossier en local ; sans objet sur S3 (les préfixes existent dès qu'un objet y est écrit)."""
    if not is_remote(path):
        os.makedirs(path, exist_ok=True)


def read_bytes(path):
    if not is_remote(path):
        with open(path, 'rb') as f:
            return f.read()
    bucket, key = _split(path)
    return _s3().get_object(Bucket=bucket, Key=key)['Body'].read()


def write_bytes(path, data):
    if not is_remote(path):
        with open(path, 'wb') as f:
            f.write(data)
        return
    bucket, key = _split(path)
    _s3().put_object(Bucket=bucket, Key=key, Body=data)


def read_parquet(path, **kwargs):
    if not is_remote(path):
        return pd.read_parquet(path, **kwargs)
    return pd.read_parquet(io.BytesIO(read_bytes(path)), **kwargs)


def to_parquet(df, path, **kwargs):
    if not is_remote(path):
        df.to_parquet(path, **kwargs)
        return
    buffer = io.BytesIO()
    df.to_parquet(buffer, **kwargs)
    write_bytes(path, buffer.getvalue())


def read_csv(path, **kwargs):
    if not is_remote(path):
        return pd.read_csv(path, **kwargs)
    return pd.read_csv(io.BytesIO(read_bytes(path)), **kwargs)


def to_csv(df, path, **kwargs):
    if not is_remote(path):
        df.to_csv(path, **kwargs)
        return
    write_bytes(path, df.to_csv(**kwargs).encode('utf-8'))


def load_joblib(path):
    if not is_remote(path):
        return joblib.load(path)
    return joblib.load(io.BytesIO(read_bytes(path)))


def dump_joblib(obj, path):
    if not is_remote(path):
        joblib.dump(obj, path)
        return
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    write_bytes(path, buffer.getvalue())
