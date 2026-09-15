"""
Génère des ruptures artificielles sur la série cumulative d'un device/sensor, pour
constituer des jeux de test avec vérité terrain connue (détection par PELT notamment).
Choici au hazard N_SITES parmi ceux du fichier device_sensor_index_pairs.csv 
(sous_comptage, équipements electrique et capteur index uniquement, candidatas enedis sortis de l'échantillon)
renvoie une vérité terrain synthétique (timestamps auxquels on a une rupture et timestamps pour lesquels il n'y en a pas)
et on stocke la donné modifiée dans des fichiers parquet.
Usage: python creation_rupture.py <device_id> <sensor_id> <start> <end>
"""
import argparse
import glob
import os
import random
import sys

import pandas as pd
import plotly.express as px

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from db_connector import ClickHouseClient

MIN_GAP_ROWS = 5  # écart minimum (en nombre de lignes) entre deux ruptures
MAX_GAP_ROWS = 10000  # écart maximum (en nombre de lignes) entre deux ruptures
MIN_JUMP = 100  # saut minimum appliqué à une rupture
MAX_JUMP = 1e4  # saut maximum appliqué à une rupture
MIN_NEG = 5  # nombre minimum de négatifs (simple sampling, label=0) tirés par couple device/sensor
MAX_NEG = 10  # nombre maximum de négatifs (simple sampling, label=0) tirés par couple device/sensor
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'synthetic_data', 'ruptures')
VT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'verite_terrain.csv')  # vérité terrain cumulée sur tous les runs
N_SITES = 20
MIN_SITE_GAP = 2  # écart minimum (en position dans le fichier) entre deux sites choisis, pour ne jamais en prendre deux qui se suivent

PAIRS_SOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', 'device_sensor_index_pairs.csv')
INDEX_SENSOR_IDS = ['59c3be73bee9f849f51461d3', '61669dcf81278b679e1bc513',
                    '61669de581278b679e1bc514', '61669df381278b679e1bc515',
                    '61669e0081278b679e1bc516', '662bc1c93f0afb78ad05983d',
                    '662bc1c9db23311c9b7fea25']

def query_serie(device_id, sensor_id, start, end):
    dataset = 'datawarehouse_prod.fact_payload'
    return f"""SELECT
        device_id,
        payload_metric_id AS sensor_id,
        payload_timestamp,
        toFloat32OrNull(payload_value) AS raw_value
    FROM {dataset}
    WHERE device_id = '{device_id}'
    AND payload_metric_id = '{sensor_id}'
    AND payload_timestamp BETWEEN '{start}' AND '{end}'
    ORDER BY payload_timestamp;"""

def generate_rupture_indices(n_rows, min_gap=MIN_GAP_ROWS, max_gap=MAX_GAP_ROWS):
    """Tire au hasard les positions (index de ligne, hors première/dernière ligne) où
    injecter une rupture, espacées les unes des autres de min_gap à max_gap lignes
    (écart lui-même tiré au hasard). Le nombre de ruptures découle de la série tirée."""
    indices = []
    position = random.randint(min_gap, max_gap)
    while position < n_rows:
        indices.append(position)
        position += random.randint(min_gap, max_gap)
    if not indices and n_rows > 1:
        indices = [random.randint(1, n_rows - 1)]
    return indices

def apply_ruptures(df, indices, min_jump=MIN_JUMP, max_jump=MAX_JUMP):
    """A chaque index de rupture, tire un saut au hasard et l'ajoute à toutes les
    valeurs à partir de cet index. Les sauts s'accumulent pour une rupture suivante,
    ce qui reste cohérent avec une série cumulative monotone."""
    df = df.copy()
    df['is_injected_rupture'] = False
    df['injected_jump'] = 0.0
    for idx in indices:
        jump = random.uniform(min_jump, max_jump)
        tail = df.index[idx:]
        df.loc[tail, 'raw_value'] += jump
        df.loc[tail, 'injected_jump'] += jump
        df.loc[df.index[idx], 'is_injected_rupture'] = True
    return df

def save_ground_truth(df, indices, device_id, sensor_id, negative_timestamps=(), path=VT_OUTPUT_PATH):
    """Ajoute au fichier de vérité terrain une ligne par timestamp : label=1 pour les
    ruptures injectées (indices), label=0 pour les timestamps de simple sampling
    (negative_timestamps), en complétant le fichier existant s'il y en a un."""
    positive_timestamps = df.loc[indices, 'payload_timestamp'].to_numpy()
    vt_df = pd.DataFrame({
        'device_id': device_id,
        'sensor_id': sensor_id,
        'payload_timestamp': list(positive_timestamps) + list(negative_timestamps),
        'label': [1] * len(positive_timestamps) + [0] * len(negative_timestamps),
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vt_df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return vt_df

def load_serie(device_id, sensor_id, output_dir=OUTPUT_DIR):
    """Charge la colonne payload_timestamp de tous les fichiers parquet correspondant
    au couple device_id/sensor_id (il peut y en avoir plusieurs, un par run/période)."""
    pattern = os.path.join(output_dir, f"{device_id}_{sensor_id}_*.parquet")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return pd.Series(dtype='datetime64[ns]')
    return pd.concat(
        [pd.read_parquet(path, columns=['payload_timestamp']) for path in paths],
        ignore_index=True,
    )['payload_timestamp']

def sample_negative_timestamps(device_id, sensor_id, positive_timestamps):
    """Tire au hasard entre MIN_NEG et MAX_NEG payload_timestamp de la série
    device_id/sensor_id, parmi ceux qui n'apparaissent pas déjà dans positive_timestamps."""
    serie = load_serie(device_id, sensor_id)
    if serie.empty:
        print(f"Aucun fichier parquet trouvé pour {device_id} / {sensor_id}")
        return []
    available = serie[~serie.isin(positive_timestamps)]
    n = min(random.randint(MIN_NEG, MAX_NEG), len(available))
    if n < MIN_NEG:
        print(f"Seulement {n} timestamp(s) négatif(s) disponible(s) pour {device_id} / {sensor_id}")
    if n == 0:
        return []
    return available.sample(n=n).tolist()

def select_random_pairs(pairs_path=PAIRS_SOURCE_PATH, sensor_ids=INDEX_SENSOR_IDS,
                         n_sites=N_SITES, min_gap=MIN_SITE_GAP, vt_path=VT_OUTPUT_PATH):
    """Sélectionne au hasard n_sites sites, piochés un peu partout dans device_sensor_index_pairs
    (jamais deux sites qui se suivent dans le fichier, pour ne pas tomber deux fois sur le même
    client), écarte ceux déjà présents dans Myriam_test.sous_comptage_anomalies ainsi que ceux déjà
    traités lors d'un run précédent (retrouvés via vt_path), puis tire un couple device_id/sensor_id
    au hasard par site retenu. Renvoie la liste finale de couples."""
    pairs_df = pd.read_csv(pairs_path, dtype=str)
    pairs_df = pairs_df[pairs_df['site_id'].notna() & (pairs_df['site_id'] != '')]
    pairs_df = pairs_df[pairs_df['sensor_id'].isin(sensor_ids)]

    already_treated_sites = set()
    if os.path.exists(vt_path):
        treated_devices = pd.read_csv(vt_path, dtype=str)['device_id'].unique()
        already_treated_sites = set(pairs_df.loc[pairs_df['device_id'].isin(treated_devices), 'site_id'])

    ordered_sites = [site_id for site_id in pairs_df['site_id'].drop_duplicates().tolist()
                      if site_id not in already_treated_sites]

    positions = list(range(len(ordered_sites)))
    random.shuffle(positions)
    selected_positions = []
    for pos in positions:
        if all(abs(pos - other) >= min_gap for other in selected_positions):
            selected_positions.append(pos)
        if len(selected_positions) == n_sites:
            break
    selected_sites = [ordered_sites[pos] for pos in selected_positions]

    client = ClickHouseClient('stage')
    existing = client.query("SELECT DISTINCT site_id FROM Myriam_test.sous_comptage_anomalies;")
    existing_sites = set(existing['site_id']) if not existing.empty else set()
    selected_sites = [site_id for site_id in selected_sites if site_id not in existing_sites]

    result_pairs = []
    for site_id in selected_sites:
        site_rows = pairs_df[pairs_df['site_id'] == site_id]
        chosen = site_rows.sample(n=1).iloc[0]
        result_pairs.append((chosen['device_id'], chosen['sensor_id']))
    return result_pairs

def plot_result(df, device_id, sensor_id):
    """Scatter plot de raw_value au cours du temps, avec les ruptures injectées en évidence."""
    fig = px.scatter(
        df, x='payload_timestamp', y='raw_value', template='plotly_white',
        title=f"{device_id} / {sensor_id} — ruptures injectées",
        labels={'payload_timestamp': 'Date', 'raw_value': 'payload_value'},
    )
    fig.update_traces(marker=dict(size=4))
    for ts in df.loc[df['is_injected_rupture'], 'payload_timestamp']:
        fig.add_vline(x=ts, line_dash='dash', line_color='red')
    fig.show()

def device_sensor_treat(device_id, sensor_id, start, end):
    client = ClickHouseClient('prod')
    df = client.query(query_serie(device_id, sensor_id, start, end))
    if df.empty:
        print(f"Aucune donnée pour {device_id} / {sensor_id} entre {start} et {end}")
        return 0, 0
    df = df.sort_values('payload_timestamp').reset_index(drop=True)
    indices = generate_rupture_indices(len(df))
    df = apply_ruptures(df, indices)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_tag = start[:4]
    output_path = os.path.join(OUTPUT_DIR, f"{device_id}_{sensor_id}_{date_tag}.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Sauvegardé : {output_path}")
    negative_timestamps = sample_negative_timestamps(device_id, sensor_id, df.loc[indices, 'payload_timestamp'])
    save_ground_truth(df, indices, device_id, sensor_id, negative_timestamps)
    print(f"Vérité terrain mise à jour : {VT_OUTPUT_PATH}")
    plot_result(df, device_id, sensor_id)
    return len(indices), len(negative_timestamps)

def main():
    parser = argparse.ArgumentParser(
        description="Génère des ruptures artificielles sur la série d'un device/sensor "
                    "et sauvegarde le résultat en parquet dans ./data/ruptures/.")
    parser.add_argument('--start', required=True, help="Date de début, ex: '2026-01-01 00:00:00'")
    parser.add_argument('--end', required=True, help="Date de fin, ex: '2026-09-10 23:59:59'")
    args = parser.parse_args()
    pairs = select_random_pairs()
    total_positives = 0
    total_negatives = 0
    for device_id, sensor_id in pairs:
        n_positive, n_negative = device_sensor_treat(device_id, sensor_id, args.start, args.end)
        total_positives += n_positive
        total_negatives += n_negative
    total = total_positives + total_negatives
    if total == 0:
        print(f"Aucun exemple généré sur {len(pairs)} couple(s) device/sensor")
    else:
        pct_positive = 100 * total_positives / total
        pct_negative = 100 * total_negatives / total
        print(f"{pct_positive:.1f}% positifs (ruptures) / {pct_negative:.1f}% négatifs "
              f"sur {total} exemple(s) générés pour {len(pairs)} couple(s) device/sensor")

if __name__ == "__main__":
    main()
