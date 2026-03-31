CREATE DATABASE Myriam_test ON CLUSTER 'all-replicated';

SHOW DATABASES;

CREATE OR REPLACE TABLE Myriam_test.dataset_occupancy ON CLUSTER 'all-replicated' (
    payload_timestamp DateTime,
    device_id         LowCardinality(String),
    site_id           LowCardinality(String),
    occupancy_value   Nullable(Decimal64(3)),
    ingested_at       DateTime,
    sensor_name       LowCardinality(String),
    sensor_id         LowCardinality(String),
    model_version     LowCardinality(String), 
	data_source       Array(LowCardinality(String))
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/{installation}/{cluster}/{uuid}', '{replica}')
PARTITION BY toYear(payload_timestamp)
ORDER BY (toYYYYMM(payload_timestamp), device_id, payload_timestamp);


-- INSERT
INSERT INTO Myriam_test.dataset_occupancy (payload_timestamp, device_id, site_id,
	occupancy_value, ingested_at, sensor_name, sensor_id, device_type, device_name, model_version) 
VALUES (toDateTime('2026-01-15 00:00:00'), '663a510cbdbe96ae42ad0048', '6037dce8f435564fd2458b91', 0.258, now(),
'Index Consommation Electrique', '59c3be73bee9f849f51461d3', 'Compteur Electrique', 'PDL', 'test');

-- SUPPRIMER
ALTER TABLE Myriam_test.dataset_occupancy ON CLUSTER 'all-replicated' DELETE
WHERE device_id = '663a510cbdbe96ae42ad0048';

-- CHECK
SHOW TABLES FROM Myriam_test;
SELECT * FROM Myriam_test.dataset_occupancy;


