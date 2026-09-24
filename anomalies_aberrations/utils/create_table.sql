CREATE OR REPLACE TABLE Myriam_test.sous_comptage_anomalies ON CLUSTER 'all-replicated'(
    customer_account_name LowCardinality(String),
    site_name              LowCardinality(String),
    site_id                LowCardinality(String),
    device_name            LowCardinality(String),
    device_id              LowCardinality(String),
    sensor_name            LowCardinality(String),
    sensor_id              LowCardinality(String),
    anomaly_type                   LowCardinality(String),
    date_start              DateTime,
    date_stop               DateTime,
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/{installation}/{cluster}/{uuid}', '{replica}')
PARTITION BY toYear(date_start)
ORDER BY (device_id, sensor_id, anomaly_type, date_start)


/*------- Seconde table dédiée au modèle supervisé -------*/
CREATE OR REPLACE TABLE Myriam_test.anomalie_v2 ON CLUSTER 'all-replicated'(
    customer_account_name LowCardinality(String),
    site_name              LowCardinality(String),
    site_id                LowCardinality(String),
    device_name            LowCardinality(String),
    device_id              LowCardinality(String),
    sensor_name            LowCardinality(String),
    sensor_id              LowCardinality(String),
    anomaly_type           LowCardinality(String),
    payload_timestamp      DateTime,
    run_date			DateTime
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/{installation}/{cluster}/{uuid}', '{replica}')
PARTITION BY toYear(payload_timestamp)
ORDER BY (device_id, sensor_id, anomaly_type, date_start);