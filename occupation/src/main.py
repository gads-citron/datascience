import sys
import argparse
import subprocess

#%% install dependencies
#subprocess.check_call([sys.executable, "-m", "pip", "install", "clickhouse-connect"])

#%% packages
import boto3
import clickhouse_connect
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# custom
import frstl.configs as c
import failures as e
from frstl.decompose import fast_robustSTL as cp


#%% functions
def get_clickhouse_credentials():
    secret_name = "AmazonSageMaker-clickhouse-user-1"
    region_name = "eu-west-3"
    session = boto3.session.Session()
    sm_client = session.client(
        service_name = 'secretsmanager',
        region_name = region_name
    )
    get_secret_value_response = sm_client.get_secret_value(
            SecretId=secret_name)
    secret = json.loads(get_secret_value_response['SecretString'])
    return secret

def clk_db_connect(host, usr, pswd):
    db_client = clickhouse_connect.get_client(
        host = host,
        port = 443,
        username = usr,
        password = pswd)
    return db_client

def index_preprocess(df):
    neg = df.payload_value.diff(1)
    if neg.min() < 0:
        for i, row in df[neg < 0].iterrows():
            # index back to 0
            if row.payload_value==0:
                df.loc[i, "payload_value"] = \
                    (df.loc[i-1].payload_value + df.loc[i+1].payload_value)/2
            else:
                df.loc[i:, 'payload_value'] = df.loc[i:].payload_value \
                + df.loc[i-1].payload_value - df.loc[i].payload_value
    return df

def convert_kwh(sens, df):
    if sens =='59c3be73bee9f849f51461d3':
            smp = index_preprocess(df)
            return smp
    elif sens == '6167da8e41d2727f9844a0af':
            return df.payload_value.astype(float)
            # puissance active donc pas sûr que ce ne soit pas de l'index
    elif sens == '6454b98a0dae31f561c6f68f':
            return df.payload_value.astype(float)/1000

def postprocess(s, t, idx):
    """
    post process data :
        - by taking min value between estimated occupancy and real water data.
        - normalize values to be > 0
        - return only the last week
    Arguments:
        - s [np.array] {p.K} weeks water seasonnality
        - t [np.array] {p.K} weeks water trend
        - idx [np.array] sample index    
    returns:
    occ [pandas.DataFrame] with columns={'occupancy'}, index=timestamp  occupancy values (m3/h)
    """
    fix = [x if x < 0 else 0 for x in (pd.Series(
        t - np.median(t)).shift(-int(H)).fillna(0).values)]
    adjusted = [x - np.median(s)+y if -1*x < y else 0 for (x, y) in zip(s, fix)]
    _df = pd.DataFrame(adjusted, columns=["value"],
                       index=idx).shift(int(H)).fillna(0)
    # Cause only we're only interested in the last week:
    occ = _df.iloc[-7*24:]
    return occ

#%% main 
def main(args):
    end = args.end
    start = args.start
    dev_id = args.dev_id
    # get clickhouse credentials from AWS secret manager
    clk_creds = get_clickhouse_credentials()
    # open clickhouse client connexion
    db_client = clk_db_connect(host=clk_creds['host'], usr=clk_creds['username'], pswd=clk_creds['password'])
    try:        
        start_date = f"toDateTime('{start} 01:00:00')"
        end_date = f"toDateTime('{end} 01:00:00')"
        df = db_client.query_df(f""" 
                SELECT device_id, payload_timestamp, payload_value, payload_metric_id
                FROM {c.releves}
                WHERE device_id = '{dev_id}'
                        AND payload_metric_id IN {c.enedis_sensors}
                        AND payload_timestamp > {start_date}
                        AND payload_timestamp < {end_date}
                SETTINGS max_rows_to_read = {c.maxi};""")
        if df.empty:
            raise e.notEnoughData(dev_id, start_date, end_date)
        if len(df.payload_metric_id.unique()):
            raise e.SeveralSensors(dev_id, start_date, end_date)
        sensor = df.payload_metric_id.values[0]
        df['power_kwh'] = convert_kwh(sensor, df)
        scaler = MinMaxScaler(feature_range=(0, 1))
        df = df.resample('1h', on='payload_timestamp')['power_kwh'].sum()
        scaled = scaler.fit_transform(pd.DataFrame(df)[['power_kwh']]).flatten()
        raw, t, s, r = fast_robustSTL(scaled[:168*3], season_len, trend_regs,
                                  season_regs, alpha, z, denoise_ds, season_ds, K, H)
        occ = postprocess(s, t, df.index[168*3])
        occ.to_parquet("/opt/ml/processing/output/test_occupancy/")
             
    except (e.notEnoughData, e.SeveralSensors) as err:
        print(err)
#%% main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dev_id', 
        help='device_id', 
        default = '67ab7251bc31f7da6046f44f', 
        type=str)
    parser.add_argument(
      '--start',
      help='Date as YYY-MM-DD from which new prediction are required',
      default='2025-01-01',
      type=str)
    parser.add_argument(
        '--end',
        help='Date as YYY-MM-DD until which new prediction are required',
        default='2025-01-31',
        type=str)
    args = parser.parse_args()
    main(args)