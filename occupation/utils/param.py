# source des données
datasource = 'datamart_sous_comptage.distributed_dataset_hourly_payloads_merged'
dataout = 'Myriam_test.dataset_occupancy'
dev_type_elec = '%lectr%'
dev_type_water = '%eau%'
start = "2026-01-01 00:00:00"
end = "2026-04-10 23:59:59"
# model_version=  "V0.water"
model_version=  "V0.elec"
maxi = 10000000000000
envir = 'local'# vs 'sagemaker'
