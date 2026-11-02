-- DATASET: payloads whose payload_timestamp is in [{ctx_start}, {newest_ts}] (oldest and
-- newest rows of the sample, see queries.context_start / window_bounds.sql). Rows of the
-- sample ({date_column} or payload_timestamp in [start, end], and payload_timestamp > floor) are flagged
-- in_window; the others only provide context and are never reported. The history before
-- the sample is trimmed in Python (detector.keep_context).
-- FINAL: unmerged duplicates would give time_delta_seconds = 0 and double is_null reports.
SELECT
    device_id,
    payload_metric_id AS sensor_id,
    payload_timestamp,
    toFloat64OrNull(payload_value) AS raw_value,
    (
        (
            {date_column} BETWEEN {start:String} AND {end:String}
            OR payload_timestamp BETWEEN {start:String} AND {end:String}
        )
        AND payload_timestamp > {floor:String}
    ) AS in_window
FROM {fact_table} FINAL
WHERE
    (device_id, payload_metric_id) IN {pairs:Array(Tuple(String, String))}
    AND payload_timestamp BETWEEN {ctx_start:String} AND {newest_ts:String}
ORDER BY payload_timestamp
SETTINGS do_not_merge_across_partitions_select_final = 1
