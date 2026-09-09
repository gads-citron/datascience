import logging
from db_client import ClickHouseClient as clk
from db_client import Collector
from db_client import Writer
import detector as dt
import argparse
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OUT_TABLE = 'Myriam_test.sous_comptage_anomalies'

def convert_timestamp(df):
    df['payload_timestamp'] = pd.to_datetime(df['ts'], unit='s', utc=True).dt.tz_convert('Europe/Paris')
    return df

PUNCTUAL_TYPES = {'commissioning', 'rupture'}

FINAL_COLUMNS = ['customer_account_name', 'site_name', 'site_id', 'device_name', 'device_id',
                  'sensor_name', 'sensor_id', 'anomaly_type', 'date_start', 'date_stop']

def finalize_anomalies(anomalies, patrimonial_df):
    if not anomalies:
        return pd.DataFrame(columns=FINAL_COLUMNS)
    df = pd.concat(anomalies, ignore_index=True)
    is_punctual = df['anomaly_type'].isin(PUNCTUAL_TYPES)
    df['date_start'] = df['segment_start'].where(~is_punctual, df['date'])
    df['date_stop'] = df['segment_end'].where(~is_punctual, df['date'])
    df = df.merge(patrimonial_df, on=['device_id', 'sensor_id'], how='left')
    return df[FINAL_COLUMNS]

def main(dev_id, sens_id, start, end):
    logger.info("Début du traitement device_id=%s sensor_id=%s start=%s end=%s", dev_id, sens_id, start, end)
    try:
        anomalies = []
        client = clk()
        col = Collector(client)
        patrimonial_df = col.patrimonial(dev_id, sens_id)
        df = col.query_serie(dev_id, sens_id, start, end)
        if df.empty:
            raise ValueError(f"Aucune donnée pour device_id={dev_id}, sensor_id={sens_id} entre {start} et {end}")
        df = convert_timestamp(df)
        commissioning_result = dt.commissioning(df, start)
        if not commissioning_result.empty:
            anomalies.append(commissioning_result)
            df = df[df['ts'] >= commissioning_result['ts'].iloc[0]]
        multi_freq_result = dt.multi_frequence(df, start, end)
        if not multi_freq_result.empty:
            anomalies.append(multi_freq_result)
            logger.info("Multi-fréquence détectée pour device_id=%s sensor_id=%s", dev_id, sens_id)
        else:
            segment, null_range_list = dt.segmentize(df)
            for _, segment_df in segment.groupby('segment_id'):
                is_rupture = dt.rupture_detection(segment_df)
                segment_start = pd.to_datetime(segment_df['ts'].min(), unit='s', utc=True).tz_convert('Europe/Paris')
                segment_end = pd.to_datetime(segment_df['ts'].max(), unit='s', utc=True).tz_convert('Europe/Paris')
                rupture_rows = segment_df.loc[is_rupture, ['device_id', 'sensor_id', 'ts', 'raw_value']].copy()
                rupture_rows['date'] = pd.to_datetime(rupture_rows['ts'], unit='s', utc=True).dt.tz_convert('Europe/Paris')
                anomalies.append(rupture_rows.assign(anomaly_type='rupture', segment_start=segment_start, segment_end=segment_end))

            if null_range_list:
                null_df = pd.DataFrame(null_range_list, columns=['segment_start', 'segment_end'])
                null_df['segment_start'] = pd.to_datetime(null_df['segment_start'], unit='s', utc=True).dt.tz_convert('Europe/Paris')
                null_df['segment_end'] = pd.to_datetime(null_df['segment_end'], unit='s', utc=True).dt.tz_convert('Europe/Paris')
                null_df['device_id'] = dev_id
                null_df['sensor_id'] = sens_id
                null_df['anomaly_type'] = 'null_period'
                anomalies.append(null_df)

        result = finalize_anomalies(anomalies, patrimonial_df)
        logger.info("Détection terminée pour device_id=%s sensor_id=%s : %d anomalies détectées", dev_id, sens_id, len(result))
        if not result.empty:
            write_status = Writer(client).write(result, OUT_TABLE)
            logger.info("Écriture dans %s : %s", OUT_TABLE, "succès" if write_status else "échec")
            return write_status
    except Exception:
        logger.exception("Erreur pendant le traitement device_id=%s sensor_id=%s", dev_id, sens_id)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", required=True, help="Device_id to query")
    parser.add_argument("--sensor_id", required=True, help="sensor_id to query")
    parser.add_argument("--start", required=True, help="Start datetime (e.g. 2026-01-01 00:00:00)")
    parser.add_argument("--end", required=True, help="End datetime (e.g. 2026-01-31 23:59:59)")
    args = parser.parse_args()
    return_value = main(args.device_id, args.sensor_id, args.start, args.end)