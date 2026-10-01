import argparse
import logging
import time
from datetime import datetime

import pandas as pd

import detector as dt
import param as p
from db_client import ClickHouseClient as clk
from db_client import Collector
from db_client import Writer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FINAL_COLUMNS = [
    "customer_account_name",
    "site_name",
    "site_id",
    "device_name",
    "device_id",
    "sensor_name",
    "sensor_id",
    "anomaly_type",
    "payload_timestamp",
    "run_date",
]


def finalize_anomalies(anomalies, patrimonial_df, run_date):
    if not anomalies:
        return pd.DataFrame(columns=FINAL_COLUMNS)
    df = pd.concat(anomalies, ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS)
    df = df.merge(patrimonial_df, on=["device_id", "sensor_id"], how="left")
    df["run_date"] = run_date
    return df[FINAL_COLUMNS]


def _first_payload_lookup(first_df):
    if first_df.empty:
        return {}
    return {
        (row.device_id, row.sensor_id): row.first_ts
        for row in first_df.itertuples(index=False)
    }


def main(site_id, start, end):
    logger.info("Treatment starting ... site_id=%s start=%s end=%s", site_id, start, end)
    run_date = datetime.now()
    run_start = time.perf_counter()
    try:
        anomalies = []
        client = clk()
        col = Collector(client)
        model = dt.load()
        pairs = client.query(*col.get_site_pairs(site_id))
        if pairs.empty:
            raise ValueError(f"Pas de PDM elec / capteur index pour le site {site_id}")
        sample = client.query(*col.query_by_site(pairs, start, end))
        if sample.empty:
            site_name = pairs.site_name.unique()[0]
            client_name = pairs.customer_account_name.unique()[0]
            raise ValueError(
                f"Aucune donnée pour le site {site_name}, client {client_name} entre {start} et {end}"
            )
        first_ts_by_pair = _first_payload_lookup(col.get_first_payloads(pairs, end))
        n_pairs = 0
        loop_start = time.perf_counter()
        for (device_id, sensor_id), df in sample.groupby(["device_id", "sensor_id"]):
            anomalies.extend(
                dt.find_anomalies(
                    df, model, first_ts_by_pair.get((device_id, sensor_id))
                )
            )
            n_pairs += 1
        loop_duration = time.perf_counter() - loop_start
        avg_per_pair = loop_duration / n_pairs if n_pairs else 0
        result = finalize_anomalies(anomalies, pairs, run_date)
        site_name = pairs.site_name.unique()[0]
        client_name = pairs.customer_account_name.unique()[0]
        logger.info(
            "Détection terminée pour le site %s, client %s : %d anomalies détectées",
            site_name,
            client_name,
            len(result),
        )
        logger.info(
            "Temps d'exécution total : %.2fs (%d paires device_id/sensor_id, %.3fs/paire en moyenne)",
            time.perf_counter() - run_start, n_pairs, avg_per_pair,
        )
        if not result.empty:
            Writer(client).write(result, p.OUT_TABLE)
    except Exception:
        logger.exception(
            "Erreur pendant le traitement site_id=%s start=%s end=%s",
            site_id,
            start,
            end,
        )
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--site_id", required=True, help="site_id to query")
    parser.add_argument("--start", required=True, help="Start datetime (e.g. 2026-01-01 00:00:00)")
    parser.add_argument("--end", required=True, help="End datetime (e.g. 2026-01-31 23:59:59)")
    args = parser.parse_args()
    main(args.site_id, args.start, args.end)
