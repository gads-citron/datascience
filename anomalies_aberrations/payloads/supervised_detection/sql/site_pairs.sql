SELECT DISTINCT
    customer_account_name,
    site_name,
    site_id,
    device_id,
    device_name,
    sensor_id,
    sensor_name
FROM datamart_sous_comptage.device_properties
WHERE
    site_id = {site_id:String}
    AND lower(device_type) LIKE '%lectr%'
    AND lower(sensor_name) LIKE '%index%'
    AND lower(sensor_name) NOT LIKE '%(cons)%'
