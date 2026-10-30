"""
Génère des séries 100% synthétiques (pas de requête ClickHouse) qui imitent un compteur
cumulatif monotone croissant à la consommation plausible (profil journalier + bruit), puis
supprime des relevés dans chaque série : de 1 à 5 trous contigus d'une durée de 1 h à 5 jours
(tirée en échelle logarithmique, donc surtout quelques heures et parfois plusieurs jours), et
entre 1 et 50 lignes éparses. Les lignes supprimées créent des
écarts de temps (delta_t) variables entre deux relevés consécutifs, et donc des sauts de valeur
(delta_v) plus grands que la normale sans qu'il y ait de rupture. Ces séries servent d'exemples
négatifs supplémentaires (pas de vérité terrain associée) pour que le modèle de détection de
ruptures n'interprète pas un trou de mesure comme une rupture.
Deux modes (--mode) : 'large' (défaut) supprime des trous contigus de durée variable décrits
ci-dessus ; 'regular' supprime des relevés selon un motif qui se répète, soit périodique (k lignes
toutes les p lignes), soit une fenêtre horaire fixe chaque jour (par exemple aucun relevé la nuit).
Usage: python creation_gaps.py --n-series 20 --n-rows 5000
       python creation_gaps.py --mode regular --n-series 20
"""
import argparse
import random

import numpy as np
import pandas as pd

import paths
from plotting import plot_gaps

MIN_REMOVED = 1  # nombre minimum de relevés supprimés par série
MAX_REMOVED = 50  # nombre maximum de relevés supprimés par série
MIN_START_VALUE = 1e5  # valeur de départ minimale du compteur
MAX_START_VALUE = 9e5  # valeur de départ maximale du compteur
MIN_BASE_RATE = 1.0  # consommation moyenne minimale du compteur, en unités par heure
MAX_BASE_RATE = 60.0  # consommation moyenne maximale du compteur, en unités par heure
DAILY_AMPLITUDE = 0.6  # amplitude relative du profil journalier (0 = consommation plate)
NOISE_SIGMA = 0.15  # écart-type du bruit multiplicatif (log-normal) sur la consommation
MIN_GAPS = 1  # nombre minimum de trous contigus par série
MAX_GAPS = 5  # nombre maximum de trous contigus par série
MIN_GAP_SECONDS = 3600  # durée minimale d'un trou contigu (1 h)
MAX_GAP_SECONDS = 5 * 86400  # durée maximale d'un trou contigu (5 jours)
MAX_GAP_FRACTION = 0.2  # un trou ne peut pas dépasser cette part des lignes de la série
WARMUP_ROWS = 60  # lignes de début de série jamais touchées par un trou (historique des features)
REGULAR_MIN_PERIOD = 2  # mode regular : période minimale du motif, en lignes
REGULAR_MAX_PERIOD = 12  # mode regular : période maximale du motif, en lignes
REGULAR_MAX_WINDOW_HOURS = 12  # mode regular : durée maximale de la fenêtre horaire sans relevé
DAILY_FREQ_SECONDS = 86400  # cadence à partir de laquelle une fenêtre horaire n'a plus de sens
N_SERIES = 20  # nombre de séries synthétiques générées par run
N_ROWS = 5000  # nombre de lignes par série, avant suppression
START_DATE = '2026-01-01 00:00:00'  # date de départ des timestamps synthétiques
# Cadences (écart de temps minimal entre deux relevés) : 5 min, 15 min, 30 min, 1 h, 1 jour.
# Les séries parcourent cette liste à tour de rôle pour que chaque cadence soit représentée à
# parts égales.
FREQ_CHOICES_SECONDS = [300, 900, 1800, 3600, 86400]
OUTPUT_DIR = paths.data_path('synthetic_data', 'gaps')
# Ni device_id ni sensor_id ne doivent contenir de '_' : list_series_pairs (training.py) découpe
# le nom de fichier {device_id}_{sensor_id}_{tag}.parquet sur les 2 premiers '_' pour retrouver
# le couple, un '_' dans l'un des deux ids casserait ce découpage.
DEVICE_ID_PREFIX = 'syntheticvardt'
REGULAR_DEVICE_ID_PREFIX = 'syntheticreg'  # préfixe distinct pour ne pas écraser les séries du mode large
SENSOR_ID = 'variabledt'


def generate_timestamps(n_rows, start=START_DATE, freq_seconds=600):
    """Génère n_rows timestamps régulièrement espacés de freq_seconds secondes, à partir de
    start."""
    return pd.date_range(start=start, periods=n_rows, freq=pd.Timedelta(seconds=freq_seconds))


def generate_monotonic_values(timestamps, start_value, base_rate, daily_amplitude=DAILY_AMPLITUDE,
                               noise_sigma=NOISE_SIGMA, rng=None):
    """Construit un compteur cumulatif monotone croissant : la consommation instantanée (unités
    par heure) suit un profil journalier sinusoïdal autour de base_rate, avec un bruit
    multiplicatif log-normal, et la valeur à chaque timestamp est start_value plus la somme de
    consommation x durée écoulée depuis le relevé précédent. La consommation reste positive,
    donc la série est croissante."""
    rng = rng or np.random.default_rng()
    hours = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60
    profile = 1 + daily_amplitude * np.sin(2 * np.pi * (hours - 6) / 24)
    rate = base_rate * np.clip(profile, 0.05, None) * rng.lognormal(0, noise_sigma, len(timestamps))
    dt_hours = np.diff(timestamps.to_numpy()).astype('timedelta64[s]').astype(float) / 3600
    increments = rate[1:] * dt_hours
    values = start_value + np.concatenate([[0.0], np.cumsum(increments)])
    return np.round(values, 3)


def remove_random_rows(df, min_removed=MIN_REMOVED, max_removed=MAX_REMOVED, rng=None):
    """Supprime au hasard entre min_removed et max_removed lignes de df (la première est
    conservée pour garder la date de départ de la série) et renvoie le dataframe restant ainsi
    que le nombre de lignes supprimées. Les valeurs restantes ne sont pas modifiées : la série
    reste croissante, mais l'écart de temps et de valeur entre les lignes voisines d'un trou
    augmente."""
    rng = rng or np.random.default_rng()
    n_removed = int(rng.integers(min_removed, max_removed + 1))
    n_removed = min(n_removed, len(df) - 1)
    removed = rng.choice(np.arange(1, len(df)), size=n_removed, replace=False)
    return df.drop(df.index[removed]).reset_index(drop=True), n_removed


def remove_gaps(df, freq_seconds, min_gaps=MIN_GAPS, max_gaps=MAX_GAPS,
                 min_gap_seconds=MIN_GAP_SECONDS, max_gap_seconds=MAX_GAP_SECONDS,
                 max_gap_fraction=MAX_GAP_FRACTION, warmup_rows=WARMUP_ROWS, rng=None):
    """Supprime entre min_gaps et max_gaps blocs de lignes consécutives de df : la durée de
    chaque trou est tirée en échelle logarithmique entre min_gap_seconds et max_gap_seconds (donc
    plutôt des trous d'une à quelques heures, parfois de plusieurs jours), convertie en nombre
    de lignes selon freq_seconds et plafonnée à max_gap_fraction de la série. Les warmup_rows
    premières lignes ne sont jamais supprimées, pour laisser aux features de lag leur historique.
    Deux trous qui se chevauchent fusionnent en un seul, plus long. Renvoie le dataframe restant
    et le nombre de lignes supprimées."""
    rng = rng or np.random.default_rng()
    n_rows = len(df)
    max_gap_rows = max(1, int(n_rows * max_gap_fraction))
    removed = np.zeros(n_rows, dtype=bool)
    for _ in range(int(rng.integers(min_gaps, max_gaps + 1))):
        duration = np.exp(rng.uniform(np.log(min_gap_seconds), np.log(max_gap_seconds)))
        gap_rows = min(max(1, round(duration / freq_seconds)), max_gap_rows)
        start = int(rng.integers(warmup_rows, n_rows - gap_rows))
        removed[start:start + gap_rows] = True
    return df[~removed].reset_index(drop=True), int(removed.sum())


def remove_regular_rows(df, freq_seconds, min_period=REGULAR_MIN_PERIOD, max_period=REGULAR_MAX_PERIOD,
                         max_window_hours=REGULAR_MAX_WINDOW_HOURS, warmup_rows=WARMUP_ROWS, rng=None):
    """Supprime des lignes de df selon un motif qui se répète, tiré au hasard : soit périodique
    (run lignes supprimées toutes les period lignes, period entre min_period et max_period et run
    entre 1 et period // 2, ce qui laisse toujours au moins la moitié des lignes), soit une
    fenêtre horaire fixe chaque jour (de 1 h à max_window_hours heures sans aucun relevé, à
    une heure de début tirée au hasard). La fenêtre horaire n'a pas de sens pour une cadence
    journalière : on y force le motif périodique. Les warmup_rows premières lignes ne sont jamais
    supprimées. Renvoie le dataframe restant, le nombre de lignes supprimées et une description
    du motif."""
    rng = rng or np.random.default_rng()
    n_rows = len(df)
    index = np.arange(n_rows)
    use_window = freq_seconds < DAILY_FREQ_SECONDS and rng.random() < 0.5
    if use_window:
        window_hours = int(rng.integers(1, max_window_hours + 1))
        start_hour = int(rng.integers(0, 24))
        hours = df['payload_timestamp'].dt.hour.to_numpy()
        removed = ((hours - start_hour) % 24) < window_hours
        description = f"fenêtre quotidienne {start_hour:02d}h + {window_hours}h"
    else:
        period = int(rng.integers(min_period, max_period + 1))
        run = int(rng.integers(1, max(1, period // 2) + 1))
        removed = ((index - warmup_rows) % period) < run
        description = f"{run} ligne(s) toutes les {period}"
    removed = removed & (index >= warmup_rows)
    return df[~removed].reset_index(drop=True), int(removed.sum()), description


def build_series(device_id, sensor_id, freq_seconds, n_rows=N_ROWS, start=START_DATE,
                  min_removed=MIN_REMOVED, max_removed=MAX_REMOVED,
                  min_gaps=MIN_GAPS, max_gaps=MAX_GAPS,
                  min_gap_seconds=MIN_GAP_SECONDS, max_gap_seconds=MAX_GAP_SECONDS,
                  min_start_value=MIN_START_VALUE, max_start_value=MAX_START_VALUE,
                  min_base_rate=MIN_BASE_RATE, max_base_rate=MAX_BASE_RATE, mode='large', rng=None):
    """Assemble une série synthétique complète (device_id, sensor_id, payload_timestamp,
    raw_value) avec le même schéma que les parquet produits par creation_rupture.py
    (is_injected_rupture et injected_jump à False/0, aucune rupture n'étant injectée ici).
    En mode 'large', supprime d'abord des trous contigus (remove_gaps) puis quelques lignes
    éparses (remove_random_rows) ; en mode 'regular', supprime des lignes selon un motif qui se
    répète (remove_regular_rows). freq_seconds est l'écart de temps entre deux relevés avant
    suppression. Renvoie aussi le nombre total de lignes supprimées, la cadence de la série, le
    plus grand écart de temps obtenu en secondes et une description du motif (vide en mode
    'large')."""
    rng = rng or np.random.default_rng()
    timestamps = generate_timestamps(n_rows, start, freq_seconds)
    raw_value = generate_monotonic_values(
        timestamps, random.uniform(min_start_value, max_start_value),
        random.uniform(min_base_rate, max_base_rate), rng=rng)
    df = pd.DataFrame({
        'device_id': device_id,
        'sensor_id': sensor_id,
        'payload_timestamp': timestamps,
        'raw_value': raw_value,
    })
    description = ''
    if mode == 'regular':
        df, n_removed, description = remove_regular_rows(df, freq_seconds, rng=rng)
    else:
        df, n_gap_rows = remove_gaps(df, freq_seconds, min_gaps, max_gaps, min_gap_seconds,
                                      max_gap_seconds, rng=rng)
        df, n_scattered = remove_random_rows(df, min_removed, max_removed, rng)
        n_removed = n_gap_rows + n_scattered
    df['is_injected_rupture'] = False
    df['injected_jump'] = 0.0
    max_dt = df['payload_timestamp'].diff().dt.total_seconds().max()
    return df, n_removed, freq_seconds, max_dt, description


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
        description="Génère des séries synthétiques monotones croissantes dont 1 à 50 relevés "
                    "sont supprimés au hasard (écarts de temps variables, sans rupture) et les "
                    "sauvegarde en parquet dans ./synthetic_data/gaps/.")
    parser.add_argument('--n-series', type=int, default=N_SERIES,
                        help=f"Nombre de séries à générer (défaut: {N_SERIES})")
    parser.add_argument('--n-rows', type=int, default=N_ROWS,
                        help=f"Nombre de lignes par série avant suppression (défaut: {N_ROWS})")
    parser.add_argument('--start', default=START_DATE,
                        help=f"Date de départ des timestamps, ex: '2026-01-01 00:00:00' (défaut: {START_DATE})")
    parser.add_argument('--min-removed', type=int, default=MIN_REMOVED,
                        help=f"Nombre minimum de relevés supprimés par série (défaut: {MIN_REMOVED})")
    parser.add_argument('--max-removed', type=int, default=MAX_REMOVED,
                        help=f"Nombre maximum de relevés supprimés par série (défaut: {MAX_REMOVED})")
    parser.add_argument('--min-gaps', type=int, default=MIN_GAPS,
                        help=f"Nombre minimum de trous contigus par série (défaut: {MIN_GAPS})")
    parser.add_argument('--max-gaps', type=int, default=MAX_GAPS,
                        help=f"Nombre maximum de trous contigus par série (défaut: {MAX_GAPS})")
    parser.add_argument('--min-gap-seconds', type=int, default=MIN_GAP_SECONDS,
                        help=f"Durée minimale d'un trou contigu, en secondes (défaut: {MIN_GAP_SECONDS})")
    parser.add_argument('--max-gap-seconds', type=int, default=MAX_GAP_SECONDS,
                        help=f"Durée maximale d'un trou contigu, en secondes (défaut: {MAX_GAP_SECONDS})")
    parser.add_argument('--freq-seconds', type=int, nargs='+', default=FREQ_CHOICES_SECONDS,
                        help="Cadence(s) des séries en secondes, parcourues à tour de rôle "
                             f"(défaut: {' '.join(map(str, FREQ_CHOICES_SECONDS))})")
    parser.add_argument('--mode', choices=['large', 'regular'], default='large',
                        help="'large' : trous contigus de durée variable (défaut) ; 'regular' : "
                             "motif de suppression qui se répète (périodique ou fenêtre horaire "
                             "quotidienne), avec le préfixe de device syntheticreg")
    parser.add_argument('--seed', type=int, default=None,
                        help="Graine aléatoire, pour reproduire les mêmes séries (défaut: aucune)")
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                        help=f"Dossier de sortie des parquet (défaut: {OUTPUT_DIR})")
    parser.add_argument('--plot', action='store_true',
                        help="Affiche un scatter plot de la première série générée")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    prefix = REGULAR_DEVICE_ID_PREFIX if args.mode == 'regular' else DEVICE_ID_PREFIX
    written = []
    for i in range(args.n_series):
        device_id = f"{prefix}{i:04d}"
        df, n_removed, freq_seconds, max_dt, description = build_series(
            device_id, SENSOR_ID, args.freq_seconds[i % len(args.freq_seconds)],
            n_rows=args.n_rows, start=args.start,
            min_removed=args.min_removed, max_removed=args.max_removed,
            min_gaps=args.min_gaps, max_gaps=args.max_gaps,
            min_gap_seconds=args.min_gap_seconds, max_gap_seconds=args.max_gap_seconds,
            mode=args.mode, rng=rng,
        )
        output_path = save_series(df, device_id, SENSOR_ID, args.output_dir)
        written.append(output_path)
        print(f"Sauvegardé : {output_path} ({freq_seconds}s, {n_removed} relevé(s) supprimé(s), "
              f"{len(df)} ligne(s), plus grand trou {max_dt / 3600:.1f}h"
              f"{', ' + description if description else ''})")
        if args.plot and i == 0:
            plot_gaps(df, device_id, SENSOR_ID)

    print(f"{len(written)} série(s) synthétique(s) générée(s) dans {args.output_dir}")


if __name__ == "__main__":
    main()
