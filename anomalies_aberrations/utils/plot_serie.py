import argparse

import pandas as pd
import plotly.express as px

from db_connector import ClickHouseClient

def query_serie(device_id, sensor_id, start, end):
    dataset = 'datawarehouse_prod.fact_payload'
    return f"""SELECT
        device_id,
        payload_metric_id AS sensor_id,
        payload_timestamp,
        toFloat32OrNull(payload_value) AS raw_value
    FROM {dataset}
    WHERE device_id = '{device_id}'
    AND payload_metric_id = '{sensor_id}'
    AND payload_timestamp BETWEEN '{start}' AND '{end}'
    ORDER BY payload_timestamp;"""

def main():
    parser = argparse.ArgumentParser(description="Scatter plot de payload_value en fonction du temps pour un device/sensor.")
    parser.add_argument('--device_id')
    parser.add_argument('--sensor_id')
    parser.add_argument('--start', help="Date de début, ex: '2025-07-01 00:00:00'")
    parser.add_argument('--end', help="Date de fin, ex: '2025-07-31 23:59:59'")
    args = parser.parse_args()

    client = ClickHouseClient('prod')
    df = client.query(query_serie(args.device_id, args.sensor_id, args.start, args.end))
    if df.empty:
        print(f"Aucune donnée pour {args.device_id} / {args.sensor_id} entre {args.start} et {args.end}")
        return

    df['date'] = pd.to_datetime(df['payload_timestamp'], unit='s', utc=True).dt.tz_convert('Europe/Paris')

    fig = px.scatter(
        df, x='date', y='raw_value', template='plotly_white',
        title=f"{args.device_id} / {args.sensor_id}",
    )
    fig.show()


if __name__ == "__main__":
    main()
