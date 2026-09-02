import os
import urllib.parse

import clickhouse_connect
import pandas as pd
from dotenv import load_dotenv
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

def db_connect():
    load_dotenv()
    host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
    username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
    password = urllib.parse.quote_plus(os.environ.get("clk_prod_pass"))
    return clickhouse_connect.get_client(host=host, username=username, password=password, port=443)

def query_serie(dev_id, sens_id, start, end):
    dataset = 'datawarehouse_prod.fact_payload'
    query = f"""SELECT
        toUnixTimestamp(fp.payload_timestamp) AS ts,
        fp.payload_timestamp
    FROM {dataset} AS fp
    WHERE fp.device_id = '{dev_id}'
    AND fp.payload_metric_id = '{sens_id}'
    AND fp.payload_timestamp BETWEEN '{start}' AND '{end}'
    ORDER BY fp.payload_timestamp;"""
    return query

def detection(df, e=1):
    delta = df['ts'].diff()
    T = delta.median()
    r = delta % T
    proportions = delta[r > 0].value_counts(normalize=True)
    proportions = proportions[proportions >= 0.2]
    if len(proportions) > 1:
        if proportions.index.min() <= 5:
            return proportions
        if np.sum(proportions.index % proportions.index.min()) > 0:
            return proportions
    return []

def main():
    to_check = [
        ('6042702983c2f34ea421ffb9', '59c3be73bee9f849f51461d3'),
        ('67583e8f0fe5a964c753aa8f', '662bc1c93f0afb78ad05983d'),
        ('67447ea5ff9b1dc035478f2d', '662bc1c93f0afb78ad05983d'),
        ('67447ea5ff9b1dc035478f47', '662bc1c93f0afb78ad05983d'),
        ('63e50fae9100e8c12004d4d1', '59c3be73bee9f849f51461d3'),
        ('6a0d578c911b6b8a63c95a67', '59c3be73bee9f849f51461d3'),
        ('6422ab5002c384efc000f8f9', '662bc1c93f0afb78ad05983d'),
        ('674f3bd8ab8c74c94d1f6ec9', '674f3b03d28a13012e91bdc2'),
        ('63ea5b177be5fb02d2f6f415', '59c3be73bee9f849f51461d3'),
        ('674f3bd7ab8c74c94d1f6e92', '674f3b03d28a13012e91bdc2'),
        ('63ea58facb597418302361d9', '59c3be73bee9f849f51461d3'),
        ('63ea5b197be5fb6076f6f417', '59c3be73bee9f849f51461d3'),
        ('63ea5aa054a72b52a7458c90', '59c3be73bee9f849f51461d3'),
        ('674f3acb0c858cbee19e98b4', '674f3b03d28a13012e91bdc2'),
        ('674f3acb0c858cbee19e98b4', '662bc1c9db23311c9b7fea25'),
        ('68499e90f178c8ec8dc8297d', '662bc1c93f0afb78ad05983d'),
        ('68499e93f178c8ec8dc82aaa', '662bc1c93f0afb78ad05983d'),
        ('6422ab5002c384efc000f8f9', '59c3be73bee9f849f51461d3'),
        ('699f0aad999b02cb5278aaaf', '662bc1c93f0afb78ad05983d'),
        ('69bbfe19ded74906ae3dd4e9', '662bc1c93f0afb78ad05983d'),
        ('67583e8e0fe5a964c753aa56', '662bc1c93f0afb78ad05983d'),
        ('67583e900fe5a964c753aab9', '662bc1c93f0afb78ad05983d'),
        ('67583e8e0fe5a964c753aa23', '662bc1c93f0afb78ad05983d'),
        ('67583e8e0fe5a964c753aa59', '662bc1c93f0afb78ad05983d'),
        ('67583e8e0fe5a964c753aa54', '662bc1c93f0afb78ad05983d'),
        ('67583e8e0fe5a964c753aa1f', '662bc1c93f0afb78ad05983d'),
        ('67583e900fe5a964c753aaa3', '662bc1c93f0afb78ad05983d'),
        ('67ab45ffbc31f7da6041c896', '662bc1c93f0afb78ad05983d'),
        ('67ab4600bc31f7da6041c961', '662bc1c93f0afb78ad05983d'),
        ('67583e8e0fe5a964c753aa21', '662bc1c93f0afb78ad05983d'),
        ('67583e8f0fe5a964c753aa8d', '662bc1c93f0afb78ad05983d'),
        ('67583e8f0fe5a964c753aa8a', '662bc1c93f0afb78ad05983d'),
        ('684aee4fb808d8009572da31', '6454b98a0dae31f561c6f69a'),
    ]
    start = '2025-01-01 00:00:00'
    end = '2025-12-31 23:59:59'

    client = db_connect()
    results = []
    for device_id, sensor_id in to_check:
        df = client.query_df(query_serie(device_id, sensor_id, start, end))
        if df.empty:
            print(f"Aucune donnée pour {device_id} / {sensor_id} entre {start} et {end}")
            continue
        periodes = detection(df)
        if len(periodes)==0:
            continue
        if periodes.index.max() - periodes.index.min() < 10:
            continue
        results.append({
            'device_id': device_id,
            'sensor_id': sensor_id,
            'periodes': list(periodes.index),
        })

    results_df = pd.DataFrame(results, columns=['device_id', 'sensor_id', 'periodes'])
    # results_df.to_csv("./pdm_multi_freq.csv")
    print(results_df)

if __name__ == "__main__":
    main()
