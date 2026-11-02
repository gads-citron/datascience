SELECT DISTINCT
    device_id,
    sensor_id,
    payload_timestamp,
    anomaly_type
FROM {out_table}
WHERE
    (device_id, sensor_id) IN {pairs:Array(Tuple(String, String))}
    AND payload_timestamp BETWEEN {min_ts:DateTime64(3)} AND {max_ts:DateTime64(3)}
