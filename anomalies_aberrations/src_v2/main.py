import logging
import time
from db_client import ClickHouseClient as clk
from db_client import Collector
from db_client import Writer
import detector as dt
import argparse
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OUT_TABLE = 'Myriam_test.sous_comptage_anomalies_v2'
# PUNCTUAL_TYPES = {'commissioning', 'reset', 'is_null', 'rupture'}
FINAL_COLUMNS = ['customer_account_name', 'site_name', 'site_id', 'device_name', 'device_id',
                  'sensor_name', 'sensor_id', 'anomaly_type', 'date_start', 'date_stop']

def finalize_anomalies(anomalies, patrimonial_df):
    if not anomalies:
        return pd.DataFrame(columns=FINAL_COLUMNS)
    df = pd.concat(anomalies, ignore_index=True)
    df = df.rename(columns={'date_end': 'date_stop'})
    df = df.merge(patrimonial_df, on=['device_id', 'sensor_id'], how='left')
    return df[FINAL_COLUMNS]

def main(site_id, start, end):
    logger.info("Treatment starting ... site_id=%s start=%s end=%s", site_id, start, end)
    run_start = time.perf_counter()
    try:
        anomalies = []
        client = clk()
        col = Collector(client)
        model = dt.load()
        pairs = client.query(col.get_site_pairs(site_id))
        if pairs.empty :
            raise ValueError(f"Pas de PDM elec / capteur index pour le site {site_id} entre {start} et {end}")
        sample = client.query(col.query_by_site(pairs, start, end))
        if sample.empty:
            raise ValueError(f"Aucune donnée pour le site {pairs.site_name.unique()[0]}, \
                             client {pairs.customer_account_name.unique()[0]} entre {start} et {end}")
        n_pairs = 0
        loop_start = time.perf_counter()
        for (device_id, sensor_id), df in sample.groupby(['device_id', 'sensor_id']):
            anomalies.extend(dt.find_anomalies(df, model))
            n_pairs += 1
        loop_duration = time.perf_counter() - loop_start
        avg_per_pair = loop_duration / n_pairs if n_pairs else 0
        result = finalize_anomalies(anomalies, pairs)
        logger.info(f"Détection terminée pour le site {pairs.site_name.unique()[0]}, \
                    client {pairs.customer_account_name.unique()[0]} : {len(result)} anomalies détectées")
        logger.info(
            "Temps d'exécution total : %.2fs (%d paires device_id/sensor_id, %.3fs/paire en moyenne)",
            time.perf_counter() - run_start, n_pairs, avg_per_pair,
        )
        if not result.empty:
            write_status = Writer(client).write(result, OUT_TABLE)
            logger.info("Écriture dans %s : %s", OUT_TABLE, "succès" if write_status else "échec")
            return write_status
    except Exception:
        logger.exception(f"Erreur pendant le traitement site_id={site_id} start={start} end={end}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--site_id", required=True, help="site_id to query")
    parser.add_argument("--start", required=True, help="Start datetime (e.g. 2026-01-01 00:00:00)")
    parser.add_argument("--end", required=True, help="End datetime (e.g. 2026-01-31 23:59:59)")
    args = parser.parse_args()
    return_value = main(args.site_id, args.start, args.end)