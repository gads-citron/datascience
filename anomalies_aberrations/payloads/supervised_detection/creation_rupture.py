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

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
from db_connector import ClickHouseClient

MIN_GAP_ROWS = 5  # écart minimum (en nombre de lignes) entre deux ruptures
MAX_GAP_ROWS = 10000  # écart maximum (en nombre de lignes) entre deux ruptures
MIN_JUMP = 1000  # saut minimum appliqué à une rupture
MAX_JUMP = 1e4  # saut maximum appliqué à une rupture
MIN_NEG = 5  # nombre minimum de négatifs (simple sampling, label=0) tirés par couple device/sensor
MAX_NEG = 10  # nombre maximum de négatifs (simple sampling, label=0) tirés par couple device/sensor
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'synthetic_data', 'ruptures')
VT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'verite_terrain.csv')  # vérité terrain cumulée sur tous les runs
N_SITES = 20
MIN_SITE_GAP = 2  # écart minimum (en position dans le fichier) entre deux sites choisis, pour ne jamais en prendre deux qui se suivent

PAIRS_SOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'utils', 'data', 'device_sensor_index_pairs.csv')
INDEX_SENSOR_IDS = ['59c3be73bee9f849f51461d3', '61669dcf81278b679e1bc513',
                    '61669de581278b679e1bc514', '61669df381278b679e1bc515',
                    '61669e0081278b679e1bc516', '662bc1c93f0afb78ad05983d',
                    '662bc1c9db23311c9b7fea25']
TARGET_CUSTOMER_ACCOUNTS = ['ENAC', 'Looping Experiences', 'ISAE - SUPAERO',
                             "Mairie d'Argenteuil", 'ESSET', 'Réseau ETE - Nouvelle Aquitaine']

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

def pick_spaced_positions(n_positions, n_sites, min_gap=MIN_SITE_GAP):
    """Tire n_sites positions parmi range(n_positions), en essayant de les espacer d'au
    moins min_gap. Si la contrainte d'écart ne permet pas d'en trouver assez (vivier trop
    petit), on complète sans contrainte plutôt que de renvoyer moins de sites."""
    positions = list(range(n_positions))
    random.shuffle(positions)
    selected = []
    for pos in positions:
        if len(selected) == n_sites:
            break
        if all(abs(pos - other) >= min_gap for other in selected):
            selected.append(pos)
    if len(selected) < n_sites:
        remaining = [pos for pos in positions if pos not in set(selected)]
        fill = remaining[:n_sites - len(selected)]
        if fill:
            print(f"Contrainte d'écart min_gap={min_gap} non tenable : {len(fill)} site(s) "
                  f"ajouté(s) sans contrainte d'espacement")
        selected.extend(fill)
    return selected

def select_random_pairs(pairs_path=PAIRS_SOURCE_PATH, sensor_ids=INDEX_SENSOR_IDS,
                         customer_account_names=TARGET_CUSTOMER_ACCOUNTS,
                         n_sites=N_SITES, min_gap=MIN_SITE_GAP, vt_path=VT_OUTPUT_PATH):
    """Sélectionne au hasard n_sites sites, piochés un peu partout dans device_sensor_index_pairs
    (en évitant autant que possible deux sites qui se suivent dans le fichier, pour ne pas tomber
    deux fois sur le même client), parmi les couples appartenant à customer_account_names. Les sites
    déjà présents dans Myriam_test.sous_comptage_anomalies et ceux déjà traités lors d'un run
    précédent (retrouvés via vt_path) sont écartés AVANT le tirage, pour que le tirage porte bien sur
    n_sites sites utilisables. Puis tire un couple device_id/sensor_id au hasard par site retenu."""
    pairs_df = pd.read_csv(pairs_path, dtype=str)
    pairs_df = pairs_df[pairs_df['site_id'].notna() & (pairs_df['site_id'] != '')]
    print(f"Vivier : {pairs_df['site_id'].nunique()} site(s) dans {os.path.basename(pairs_path)}")
    pairs_df = pairs_df[pairs_df['sensor_id'].isin(sensor_ids)]
    print(f"  après filtre sensor_id (capteurs index) : {pairs_df['site_id'].nunique()} site(s)")
    pairs_df = pairs_df[pairs_df['customer_account_name'].isin(customer_account_names)]
    print(f"  après filtre customer_account_name : {pairs_df['site_id'].nunique()} site(s)")

    already_treated_sites = set()
    if os.path.exists(vt_path):
        treated_devices = pd.read_csv(vt_path, dtype=str)['device_id'].unique()
        already_treated_sites = set(pairs_df.loc[pairs_df['device_id'].isin(treated_devices), 'site_id'])

    client = ClickHouseClient('stage')
    existing = client.query("SELECT DISTINCT site_id FROM Myriam_test.sous_comptage_anomalies;")
    existing_sites = set(existing['site_id']) if not existing.empty else set()

    excluded_sites = already_treated_sites | existing_sites
    ordered_sites = [site_id for site_id in pairs_df['site_id'].drop_duplicates().tolist()
                      if site_id not in excluded_sites]
    print(f"  après exclusion des runs précédents ({len(already_treated_sites)} via {os.path.basename(vt_path)}) "
          f"et de sous_comptage_anomalies ({len(existing_sites)} site(s) en base) : "
          f"{len(ordered_sites)} site(s) disponible(s)")

    if len(ordered_sites) < n_sites:
        print(f"ATTENTION : seulement {len(ordered_sites)} site(s) disponible(s) pour {n_sites} demandé(s)")

    selected_positions = pick_spaced_positions(len(ordered_sites), n_sites, min_gap)
    selected_sites = [ordered_sites[pos] for pos in selected_positions]

    result_pairs = []
    for site_id in selected_sites:
        site_rows = pairs_df[pairs_df['site_id'] == site_id]
        chosen = site_rows.sample(n=1).iloc[0]
        result_pairs.append((chosen['device_id'], chosen['sensor_id']))
    print(f"{len(result_pairs)} couple(s) device/sensor sélectionné(s)")
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

def existing_parquet_pairs(output_dir=OUTPUT_DIR):
    """Couples (device_id, sensor_id) qui ont déjà au moins un parquet {device_id}_{sensor_id}_*.parquet
    dans output_dir (suppose qu'aucun des deux ids ne contient de '_')."""
    pairs = set()
    for path in glob.glob(os.path.join(output_dir, '*.parquet')):
        parts = os.path.basename(path).split('_', 2)
        if len(parts) == 3:
            pairs.add((parts[0], parts[1]))
    return pairs

def collect_negative_only_site(site_id, start, end, pairs_path=PAIRS_SOURCE_PATH):
    """Collecte, pour la période start/end, les séries des couples device_id/sensor_id du site
    site_id (listés dans device_sensor_index_pairs.csv) qui n'ont pas encore de parquet dans
    OUTPUT_DIR, et les sauvegarde sans injecter de rupture (exemples négatifs purs). Les fichiers ont
    le même format et le même nom que ceux de device_sensor_treat, avec is_injected_rupture=False et
    injected_jump=0. Rien n'est écrit dans la vérité terrain : training.py retrouve ces couples via
    les parquet. Retourne la liste des chemins écrits."""
    pairs_df = pd.read_csv(pairs_path, dtype=str)
    site_pairs = pairs_df.loc[pairs_df['site_id'] == site_id, ['device_id', 'sensor_id']].drop_duplicates()
    if site_pairs.empty:
        print(f"Site {site_id} introuvable dans {os.path.basename(pairs_path)}")
        return []
    existing = existing_parquet_pairs()
    todo = [(d, s) for d, s in site_pairs.itertuples(index=False, name=None) if (d, s) not in existing]
    print(f"Site {site_id} : {len(site_pairs)} couple(s), {len(site_pairs) - len(todo)} déjà présent(s) "
          f"en parquet, {len(todo)} à collecter")

    client = ClickHouseClient('prod')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    written = []
    for device_id, sensor_id in todo:
        df = client.query(query_serie(device_id, sensor_id, start, end))
        if df.empty:
            print(f"Aucune donnée pour {device_id} / {sensor_id} entre {start} et {end}")
            continue
        df = df.sort_values('payload_timestamp').reset_index(drop=True)
        df['is_injected_rupture'] = False
        df['injected_jump'] = 0.0
        output_path = os.path.join(OUTPUT_DIR, f"{device_id}_{sensor_id}_{start[:4]}.parquet")
        df.to_parquet(output_path, index=False)
        print(f"Sauvegardé : {output_path}")
        written.append(output_path)
    return written

def main():
    parser = argparse.ArgumentParser(
        description="Génère des ruptures artificielles sur la série d'un device/sensor "
                    "et sauvegarde le résultat en parquet dans ./data/ruptures/.")
    parser.add_argument('--start', required=True, help="Date de début, ex: '2026-01-01 00:00:00'")
    parser.add_argument('--end', required=True, help="Date de fin, ex: '2026-09-10 23:59:59'")
    parser.add_argument('--n-sites', type=int, default=N_SITES,
                        help=f"Nombre de sites à tirer (défaut: {N_SITES})")
    parser.add_argument('--negative-site', default=None, metavar='SITE_ID',
                        help="Collecte sans rupture (exemples négatifs purs) les couples de ce site "
                             "qui n'ont pas encore de parquet, puis s'arrête (--n-sites est ignoré)")
    args = parser.parse_args()
    if args.negative_site:
        written = collect_negative_only_site(args.negative_site, args.start, args.end)
        print(f"{len(written)} fichier(s) parquet écrit(s) pour le site {args.negative_site}")
        return
    pairs = select_random_pairs(n_sites=args.n_sites)
    total_positives = 0
    total_negatives = 0
    n_empty = 0
    for device_id, sensor_id in pairs:
        n_positive, n_negative = device_sensor_treat(device_id, sensor_id, args.start, args.end)
        if n_positive == 0 and n_negative == 0:
            n_empty += 1
        total_positives += n_positive
        total_negatives += n_negative
    if n_empty:
        print(f"{n_empty} couple(s) sans donnée sur la période {args.start} → {args.end}")
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
