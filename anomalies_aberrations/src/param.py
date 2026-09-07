PEN = 0.01  # pénalité PELT (modèle "l2" sur le signal log10) : plus elle est haute, moins il y a de ruptures détectées
MIN_RATIO = 5.  # si défini (float) : une rupture n'est gardée que si elle passe le test défini par RATIO_BASIS ; si None, aucun filtre, toutes les ruptures trouvées par PELT sont gardées
SMOOTHING_HOURS = 0  # nombre d'heures précédentes utilisées pour lisser le signal avant détection ; mettre à 0 pour désactiver le lissage
CLIP_NEGATIVE = True  # True : ramène les valeurs négatives à 0 avant transformation (comportement historique) ; False : conserve les valeurs négatives telles quelles
APPLY_LOG = False  # True : applique une compression log10 (signée si CLIP_NEGATIVE=False, pour rester définie sur les négatifs) ; False : pas de transformation logarithmique
MIN_SIZE = 2  # nombre minimum de points entre deux ruptures PELT
LINEAR_OUTLIER_THRESHOLD = 100  # un point est marqué aberrant si son résidu au modèle linéaire dépasse LINEAR_OUTLIER_THRESHOLD x la MAD des résidus
MIN_SEGMENT_DURATION_HOURS = 24  # durée minimale (en heures) d'un segment pour être conservé par segmentize
PAIRS_SOURCE = 'manual_pairs'  # fonction utilisée par main() pour récupérer les paires (device_id, sensor_id) : 'sample_1', 'sample_1bis', 'sample_2' ou 'manual_pairs'

SECRET = "AmazonSageMaker-clickhouse-user-1" # STAGE environment
#SECRET = "AmazonSageMaker-clickhouse-user-prod" # PROD environment
# OUTPUT_FILE = 'payload_anomaly_client_IQ_2026.csv'