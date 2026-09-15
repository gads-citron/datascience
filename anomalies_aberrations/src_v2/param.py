SECRET = "AmazonSageMaker-clickhouse-user-1" # STAGE environment
#SECRET = "AmazonSageMaker-clickhouse-user-prod" # PROD environment
OUT_TABLE = 'Myriam_test.anomalie_v2'
W = 7  # nombre de points avant pos utilisés pour les features de lag (valeur, delta_t, delta_v)
T = 50  # nombre de points avant pos utilisés pour la médiane de delta_t
M = 5  # taille des fenêtres utilisées pour les moyennes quadratiques de la valeur