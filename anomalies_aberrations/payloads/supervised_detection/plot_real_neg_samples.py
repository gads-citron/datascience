"""
Affiche, pour chaque ligne de real_neg_samples.csv (device_id, sensor_id, start, end), un scatter
plot plotly de payload_value en fonction de payload_timestamp, collecté dans
datawarehouse_stage.fact_payload (payload_metric_id = sensor_id). Tous les graphiques sont écrits
dans une seule page html. Les séries de plus de --max-points points sont décimées par min/max par
tranche de temps (les sauts et les ruptures restent visibles) ; --max-points 0 désactive la décimation.
Usage: python plot_real_neg_samples.py [--csv real_neg_samples.csv] [--out real_neg_samples.html]
"""
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
from db_connector import ClickHouseClient
from plotting import build_figure, decimate_min_max
from real_neg_samples import CSV_PATH, read_requests

FACT_TABLE = 'datawarehouse_stage.fact_payload'
OUT_PATH = os.path.join(os.path.dirname(__file__), 'real_neg_samples.html')
MAX_POINTS = 6000


def query_serie(device_id, sensor_id, start, end):
    return f"""SELECT
        payload_timestamp,
        toFloat64OrNull(payload_value) AS payload_value
    FROM {FACT_TABLE}
    WHERE device_id = '{device_id}'
    AND payload_metric_id = '{sensor_id}'
    AND payload_timestamp BETWEEN '{start:%Y-%m-%d %H:%M:%S}' AND '{end:%Y-%m-%d %H:%M:%S}'
    AND deleted_at IS NULL
    ORDER BY payload_timestamp;"""


def main():
    parser = argparse.ArgumentParser(
        description="Scatter plot plotly de chaque série listée dans le csv, collectée dans "
                    f"{FACT_TABLE}.")
    parser.add_argument('--csv', default=CSV_PATH, help=f"Csv des séries (défaut: {CSV_PATH})")
    parser.add_argument('--out', default=OUT_PATH, help=f"Page html de sortie (défaut: {OUT_PATH})")
    parser.add_argument('--max-points', type=int, default=MAX_POINTS,
                        help=f"Points maximum par graphique, 0 pour ne pas décimer (défaut: {MAX_POINTS})")
    args = parser.parse_args()

    requests = read_requests(args.csv)
    client = ClickHouseClient('stage')
    sections, empty = [], []
    for i, row in enumerate(requests.itertuples(index=False), start=1):
        df = client.query(query_serie(row.device_id, row.sensor_id, row.start, row.end))
        df = df.dropna(subset=['payload_value']) if not df.empty else df
        if df.empty:
            print(f"[{i}/{len(requests)}] vide : {row.device_id} / {row.sensor_id}")
            empty.append((row.device_id, row.sensor_id))
            continue
        df = df.sort_values('payload_timestamp').reset_index(drop=True)
        n_total = len(df)
        fig = build_figure(decimate_min_max(df, args.max_points), row.device_id, row.sensor_id, n_total)
        sections.append(fig.to_html(full_html=False, include_plotlyjs='cdn' if not sections else False))
        print(f"[{i}/{len(requests)}] {row.device_id} / {row.sensor_id} : {n_total} point(s)")

    header = (f"<h2>{len(sections)} série(s) tracée(s) sur {len(requests)} — "
              f"{FACT_TABLE}</h2><p>Sans donnée : {len(empty)}</p>")
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('<html><head><meta charset="utf-8"><title>real_neg_samples</title></head><body>'
                + header + '\n'.join(sections) + '</body></html>')
    print(f"{len(sections)} graphique(s) écrit(s) dans {args.out}, {len(empty)} série(s) vide(s)")


if __name__ == "__main__":
    main()
