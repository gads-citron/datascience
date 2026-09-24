
select *
from Myriam_test.sous_comptage_anomalies
WHERE site_id = '69afff3c2151de008abf3c7e';

select *
from Myriam_test.anomalie_v2
WHERE site_id = '656dd11ffc49c29c9e6cdc71';


CREATE OR REPLACE TABLE Myriam_test.sous_comptage_anomalies ON CLUSTER 'all-replicated'(
    customer_account_name LowCardinality(String),
    site_name              LowCardinality(String),
    site_id                LowCardinality(String),
    device_name            LowCardinality(String),
    device_id              LowCardinality(String),
    sensor_name            LowCardinality(String),
    sensor_id              LowCardinality(String),
    anomaly_type           LowCardinality(String),
    payload_timestamp      DateTime,
    run_date				DateTime
)
ENGINE = ReplicatedReplacingMergeTree('/clickhouse/{installation}/{cluster}/{uuid}', '{replica}')
PARTITION BY toYear(payload_timestamp)
ORDER BY (device_id, sensor_id, anomaly_type, payload_timestamp, run_date);

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
ORDER BY (device_id, sensor_id, anomaly_type, payload_timestamp, run_date);


-- par client
select customer_account_name, count(payload_timestamp) nb
from Myriam_test.anomalie_v2
WHERE anomaly_type = 'rupture'
GROUP BY customer_account_name
ORDER BY nb DESC ;

--par client et par type
select customer_account_name, anomaly_type, count(payload_timestamp) nb
from Myriam_test.anomalie_v2
GROUP BY customer_account_name, anomaly_type
ORDER BY nb DESC ;

select device_name, site_name, device_id, sensor_id, count(payload_timestamp) nb
from Myriam_test.anomalie_v2
WHERE 1=1
AND anomaly_type = 'reset'
AND customer_account_name = 'CD91 - Conseil départemental de l\'Essonne'
GROUP BY site_name, device_name, device_id, sensor_id
ORDER BY nb DESC ;

-- site ayant eu à la fois valeurs nulle et reset dans la période
SELECT customer_account_name,
    device_name,
    site_name,
    device_id,
    sensor_id,
    countIf(anomaly_type = 'reset') AS nb_reset,
    countIf(anomaly_type = 'is_null') AS nb_null_value
FROM Myriam_test.anomalie_v2
GROUP BY
	customer_account_name,
    site_name,
    device_name,
    device_id,
    sensor_id
HAVING
    nb_reset > 0
    AND nb_null_value > 0
ORDER BY nb_reset DESC;

-- liste des anomalies
select device_id, sensor_id, payload_timestamp, anomaly_type
from Myriam_test.anomalie_v2
WHERE device_id = '67bc832905fcd3f6eae214de'
AND sensor_id = '59c3be73bee9f849f51461d3';

-- anomalies détectées par l'ancienne version mais pas par la nouvele
SELECT
    s.customer_account_name,
    s.device_id,
    s.sensor_id,
    count() AS nb_anom
FROM Myriam_test.sous_comptage_anomalies AS s
LEFT ANTI JOIN Myriam_test.anomalie_v2 AS a
    ON s.device_id = a.device_id
   AND s.sensor_id = a.sensor_id
   AND s.date_start = a.payload_timestamp
GROUP BY s.customer_account_name, s.device_id, s.sensor_id
ORDER BY nb_anom DESC