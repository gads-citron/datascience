/* Sur l'historique */
SELECT query_id, type, query_start_time, 
formatReadableSize(memory_usage) memory_usage, 
formatReadableTimeDelta(query_duration_ms/1000) duration
FROM clusterAllReplicas('all-replicated', system.query_log)
where query_start_time > toDateTime('2026-10-01 14:32:00') and initial_user = 'sagemaker'
ORDER BY query_start_time DESC 
limit 100;

SELECT
    toDate(query_start_time) AS jour,
    count() AS nb_queries,
    formatReadableSize(sum(memory_usage)) AS total_memory
FROM clusterAllReplicas('all-replicated', system.query_log)
WHERE query_start_time > toDateTime('2026-10-01 14:32:00')
  AND initial_user = 'sagemaker'
  AND type = 'QueryFinish'
GROUP BY jour
ORDER BY jour DESC;

select min(payload_value)
from datawarehouse_stage.fact_payload
where payload_metric_id = '662bc1c93f0afb78ad05983d'
limit 100;

select distinct site_id
from Myriam_test.sous_comptage_anomalies
where run_date > '2026-10-13 16:30:00'
LIMIT 100;

select distinct customer_account_name
from Myriam_test.payload_view


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
from datamart_sous_comptage.electric_anomalies
WHERE anomaly_type = 'rupture'
GROUP BY customer_account_name
ORDER BY nb DESC ;

-- par device_id, sensor_id
select customer_account_name, site_name, device_name, device_id, sensor_id, run_date, count(payload_timestamp) nb, max(payload_timestamp)
from datamart_sous_comptage.electric_anomalies
WHERE anomaly_type = 'rupture'
GROUP BY customer_account_name, site_name, device_name,device_id, sensor_id, run_date
ORDER BY nb DESC ;

select *
from datamart_sous_comptage.electric_anomalies
where anomaly_type = 'rupture'
and device_id = '6808f9c4857032e9e9d4ca1f';

--par client et par type
select customer_account_name, anomaly_type, count(payload_timestamp) nb
from datamart_sous_comptage.electric_anomalies
GROUP BY customer_account_name, anomaly_type
ORDER BY nb DESC;

select *
from datamart_sous_comptage.electric_anomalies
WHERE anomaly_type = 'rupture'
AND site_name like 'A_005 Complexe culturel Saint Louis'
AND device_name like 'CPT PAC Pompe 2';
--AND customer_account_name = 'Réseau ETE - Nouvelle Aquitaine';
--GROUP BY site_name, device_name, device_id, sensor_id
--ORDER BY nb DESC ;


SELECT
    device_id,
    device_name,
    sensor_id,
    customer_account_name,
    site_name,
    anomaly_type
FROM Myriam_test.sous_comptage_anomalies sca 

SELECT *
FROM datamart_sous_comptage.electric_anomalies
WHERE payload_timestamp > '2026-09-01 00:00:00'
ORDER BY payload_timestamp DESC
limit 100;

SELECT *
FROM datawarehouse_stage.fact_payload
WHERE device_id = '67474bc6c2131d520bddb799'
AND payload_timestamp > '2026-01-21 00:00:00';

-- site ayant eu à la fois valeurs nulle et reset dans la période
SELECT customer_account_name,
    device_name,
    site_name,
    device_id,
    sensor_id,
    sensor_name,
    countIf(anomaly_type = 'reset') AS nb_reset,
    countIf(anomaly_type = 'is_null') AS nb_null_value,
    countIf(anomaly_type = 'rupture') AS nb_rupture
FROM Myriam_test.sous_comptage_anomalies
GROUP BY
	customer_account_name,
    site_name,
    device_name,
    device_id,
    sensor_id,
    sensor_name
HAVING
	nb_null_value > 20
    AND nb_rupture < 10
ORDER BY nb_rupture DESC;

-- requête payloads
WITH
window_bounds AS (
    SELECT
        device_id,
        payload_metric_id,
        min(payload_timestamp) AS min_ts,
        max(payload_timestamp) AS max_ts
    FROM datawarehouse.fact_payload FINAL
    WHERE
        (device_id, payload_metric_id) IN {pairs:Array(Tuple(String, String))}
        AND {date_column} BETWEEN {start:String} AND {end:String}
    GROUP BY device_id, payload_metric_id
),

context_bounds AS (
    -- pairs without history get no row here: the default (epoch) start keeps every row
    SELECT
        device_id,
        payload_metric_id,
        min(payload_timestamp) AS ctx_start
    FROM (
        SELECT
            f.device_id AS device_id,
            f.payload_metric_id AS payload_metric_id,
            f.payload_timestamp AS payload_timestamp,
            row_number() OVER (
                PARTITION BY f.device_id, f.payload_metric_id
                ORDER BY f.payload_timestamp DESC
            ) AS rn
        FROM datawarehouse.fact_payload AS f FINAL
        INNER JOIN window_bounds AS w
            ON f.device_id = w.device_id AND f.payload_metric_id = w.payload_metric_id
        WHERE f.payload_timestamp < w.min_ts
    )
    WHERE rn <= {margin:UInt32}
    GROUP BY device_id, payload_metric_id
)

SELECT
    f.device_id AS device_id,
    f.payload_metric_id AS sensor_id,
    f.payload_timestamp AS payload_timestamp,
    toFloat64OrNull(f.payload_value) AS raw_value,
    f.{date_column} BETWEEN {start:String} AND {end:String} AS in_window
FROM {fact_table} AS f FINAL
INNER JOIN window_bounds AS w
    ON f.device_id = w.device_id AND f.payload_metric_id = w.payload_metric_id
LEFT JOIN context_bounds AS c
    ON f.device_id = c.device_id AND f.payload_metric_id = c.payload_metric_id
WHERE
    f.payload_timestamp >= c.ctx_start
    AND f.payload_timestamp <= w.max_ts
ORDER BY f.payload_timestamp
SETTINGS do_not_merge_across_partitions_select_final = 1

-- liste des anomalies
select device_name, device_id, sensor_id, payload_timestamp, anomaly_type
from Myriam_test.sous_comptage_anomalies
WHERE device_id = '6042723d83c2f34ea4220415'
AND sensor_id = '59c3be73bee9f849f51461d3'
--AND payload_timestamp > '2026-01-01 00:00:00'
ORDER BY payload_timestamp;

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

select *
from Myriam_test.payload_view
limit 100;


select distinct customer_account_name, device_type, sensor_name
FROM datamart_sous_comptage.device_properties
WHERE lower(device_type) LIKE '%lectr%'
AND lower(sensor_name) LIKE '%index%'
AND lower(sensor_name) NOT LIKE '%cons%';

-- creation de vue sur payload
CREATE OR REPLACE VIEW Myriam_test.payload_view ON CLUSTER 'all-replicated' AS
WITH dev_list AS (
    SELECT
        device_id,
        sensor_id,
        any(device_name) AS device_name,
        any(customer_account_name) AS customer_account_name,
        any(site_name) AS site_name
    FROM datamart_sous_comptage.electric_anomalies
    GROUP BY device_id, sensor_id
),
anomalies AS (
    SELECT
        device_id,
        sensor_id,
        payload_timestamp,
        any(anomaly_type) AS anomaly_type
    FROM datamart_sous_comptage.electric_anomalies
    GROUP BY device_id, sensor_id, payload_timestamp
)
SELECT
    p.device_id as device_id,
    p.payload_metric_id as payload_metric_id,
    p.payload_timestamp as payload_timestamp,
    toFloat32OrNull(p.payload_value) AS payload_value,
    a.anomaly_type as anomaly_type,
    d.device_name as device_name,
    d.customer_account_name as customer_account_name,
    d.site_name as site_name
FROM (
    SELECT
        device_id,
        payload_metric_id,
        payload_timestamp,
        any(payload_value) AS payload_value
    FROM datawarehouse_stage.fact_payload
    WHERE (payload_metric_id, device_id) IN (SELECT sensor_id, device_id FROM dev_list)
    GROUP BY device_id, payload_metric_id, payload_timestamp
) AS p
INNER JOIN dev_list AS d
    ON p.device_id = d.device_id
    AND p.payload_metric_id = d.sensor_id
LEFT JOIN anomalies AS a
    ON p.device_id = a.device_id
    AND p.payload_metric_id = a.sensor_id
    AND p.payload_timestamp = a.payload_timestamp;

