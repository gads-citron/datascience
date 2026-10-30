"""
Génère des séries 100% synthétiques (pas de requête ClickHouse, timestamps et valeurs
fabriqués) qui imitent un compteur cumulatif strictement croissant mais à faible résolution :
la valeur reste constante pendant un palier de plusieurs lignes, puis augmente d'un petit saut
pour le palier suivant. Exemple : 5 lignes à 588126, puis 8 lignes à 588127, puis 3 lignes à
588132, etc. Ces séries servent d'exemples négatifs supplémentaires (pas de vérité terrain
associée) pour éviter que le modèle de détection de ruptures confonde ce motif de paliers avec
une vraie rupture.
Usage: python creation_constants.py --n-series 20 --n-rows 2000
"""
import argparse
import random

import numpy as np
import pandas as pd

import paths
from plotting import plot_constants

MIN_STEP = 1  # saut minimum appliqué entre deux paliers
MAX_STEP = 10  # saut maximum appliqué entre deux paliers
MIN_PLATEAU = 1  # nombre minimum de lignes consécutives à la même valeur
MAX_PLATEAU = 20  # nombre maximum de lignes consécutives à la même valeur
MIN_START_VALUE = 1e5  # valeur de départ minimale du compteur
MAX_START_VALUE = 9e5  # valeur de départ maximale du compteur
N_SERIES = 20  # nombre de séries synthétiques générées par run
N_ROWS = 2000  # nombre de lignes par série
START_DATE = '2026-01-01 00:00:00'  # date de départ des timestamps synthétiques
FREQ_SECONDS = 600  # écart en secondes entre deux payloads synthétiques
OUTPUT_DIR = paths.data_path('training_data', 'constants')
# Ni device_id ni sensor_id ne doivent contenir de '_' : list_series_pairs (training.py) découpe
# le nom de fichier {device_id}_{sensor_id}_{tag}.parquet sur les 2 premiers '_' pour retrouver
# le couple, un '_' dans l'un des deux ids casserait ce découpage.
DEVICE_ID_PREFIX = 'syntheticconst'
SENSOR_ID = 'conststep'


def generate_step_series(n_rows, start_value, min_step=MIN_STEP, max_step=MAX_STEP,
                          min_plateau=MIN_PLATEAU, max_plateau=MAX_PLATEAU):
    """Construit une série de n_rows valeurs strictement croissante par paliers : chaque palier
    répète une valeur constante pendant un nombre de lignes tiré entre min_plateau et
    max_plateau, puis la valeur augmente d'un saut tiré entre min_step et max_step pour le
    palier suivant."""
    values = np.empty(n_rows, dtype=float)
    pos = 0
    current_value = start_value
    while pos < n_rows:
        plateau_len = min(random.randint(min_plateau, max_plateau), n_rows - pos)
        values[pos:pos + plateau_len] = current_value
        pos += plateau_len
        current_value += random.randint(min_step, max_step)
    return values


def generate_timestamps(n_rows, start=START_DATE, freq_seconds=FREQ_SECONDS):
    """Génère n_rows timestamps régulièrement espacés de freq_seconds secondes, à partir de
    start."""
    return pd.date_range(start=start, periods=n_rows, freq=pd.Timedelta(seconds=freq_seconds))


def build_series(device_id, sensor_id, n_rows=N_ROWS, start=START_DATE, freq_seconds=FREQ_SECONDS,
                  min_step=MIN_STEP, max_step=MAX_STEP, min_plateau=MIN_PLATEAU,
                  max_plateau=MAX_PLATEAU, min_start_value=MIN_START_VALUE,
                  max_start_value=MAX_START_VALUE):
    """Assemble une série synthétique complète (device_id, sensor_id, payload_timestamp,
    raw_value) avec le même schéma que les parquet produits par creation_rupture.py
    (is_injected_rupture et injected_jump à False/0, aucune rupture n'étant injectée ici)."""
    start_value = random.uniform(min_start_value, max_start_value)
    raw_value = generate_step_series(n_rows, start_value, min_step, max_step, min_plateau, max_plateau)
    df = pd.DataFrame({
        'device_id': device_id,
        'sensor_id': sensor_id,
        'payload_timestamp': generate_timestamps(n_rows, start, freq_seconds),
        'raw_value': raw_value,
    })
    df['is_injected_rupture'] = False
    df['injected_jump'] = 0.0
    return df


def save_series(df, device_id, sensor_id, output_dir=OUTPUT_DIR):
    """Sauvegarde la série en parquet, avec le même format de nom que creation_rupture.py
    ({device_id}_{sensor_id}_{tag}.parquet) pour rester chargeable par un SeriesStore pointé
    sur output_dir."""
    paths.makedirs(output_dir)
    tag = df['payload_timestamp'].iloc[0].strftime('%Y%m%d')
    output_path = paths.join(output_dir,f"{device_id}_{sensor_id}_{tag}.parquet")
    paths.to_parquet(df, output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Génère des séries synthétiques strictement croissantes par paliers "
                    "(compteur à faible résolution) et les sauvegarde en parquet dans "
                    "./training_data/constants/.")
    parser.add_argument('--n-series', type=int, default=N_SERIES,
                        help=f"Nombre de séries à générer (défaut: {N_SERIES})")
    parser.add_argument('--n-rows', type=int, default=N_ROWS,
                        help=f"Nombre de lignes par série (défaut: {N_ROWS})")
    parser.add_argument('--start', default=START_DATE,
                        help=f"Date de départ des timestamps, ex: '2026-01-01 00:00:00' (défaut: {START_DATE})")
    parser.add_argument('--freq-seconds', type=int, default=FREQ_SECONDS,
                        help=f"Écart en secondes entre deux payloads (défaut: {FREQ_SECONDS})")
    parser.add_argument('--min-step', type=int, default=MIN_STEP,
                        help=f"Saut minimum entre deux paliers (défaut: {MIN_STEP})")
    parser.add_argument('--max-step', type=int, default=MAX_STEP,
                        help=f"Saut maximum entre deux paliers (défaut: {MAX_STEP})")
    parser.add_argument('--min-plateau', type=int, default=MIN_PLATEAU,
                        help=f"Longueur minimum d'un palier, en lignes (défaut: {MIN_PLATEAU})")
    parser.add_argument('--max-plateau', type=int, default=MAX_PLATEAU,
                        help=f"Longueur maximum d'un palier, en lignes (défaut: {MAX_PLATEAU})")
    parser.add_argument('--plot', action='store_true',
                        help="Affiche un scatter plot de la première série générée")
    args = parser.parse_args()

    written = []
    for i in range(args.n_series):
        device_id = f"{DEVICE_ID_PREFIX}_{i:04d}"
        df = build_series(
            device_id, SENSOR_ID, n_rows=args.n_rows, start=args.start,
            freq_seconds=args.freq_seconds, min_step=args.min_step, max_step=args.max_step,
            min_plateau=args.min_plateau, max_plateau=args.max_plateau,
        )
        output_path = save_series(df, device_id, SENSOR_ID)
        written.append(output_path)
        print(f"Sauvegardé : {output_path}")
        if args.plot and i == 0:
            plot_constants(df, device_id, SENSOR_ID)

    print(f"{len(written)} série(s) synthétique(s) générée(s) dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
