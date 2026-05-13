"""
Supprime les fichiers parquet (et leurs dossiers site/device) quand
aucune verite_terrain n'est présente dans occupancy_profile_vt_labels.csv.

Usage :
  python cleanup_unlabeled.py          # dry-run (affiche sans supprimer)
  python cleanup_unlabeled.py --delete # supprime réellement
"""
import sys
import shutil
import pandas as pd
from pathlib import Path

INPUT_FOLDER = './data_collect/sample_s3/'
VT_CSV       = f'{INPUT_FOLDER}occupancy_profile_vt_labels.csv'
DRY_RUN      = '--delete' not in sys.argv

vt = pd.read_csv(VT_CSV)
if 'sensor_id' not in vt.columns:
    vt['sensor_id'] = ''
vt['sensor_id'] = vt['sensor_id'].fillna('')

labeled_keys = set(zip(
    vt.loc[vt['verite_terrain'].notna(), 'site_id'],
    vt.loc[vt['verite_terrain'].notna(), 'device_id'],
    vt.loc[vt['verite_terrain'].notna(), 'sensor_id'],
))

root = Path(INPUT_FOLDER)
to_delete = []

for parquet in sorted(root.rglob('*.parquet')):
    parts = {p.split('=')[0]: p.split('=')[1] for p in parquet.parts if '=' in p}
    if 'device_id' not in parts:
        continue
    key = (parts.get('site_id', ''), parts.get('device_id', ''), parts.get('sensor_id', ''))
    if key not in labeled_keys:
        # dossier device_id à supprimer (contient sensor/ et parquet)
        device_dir = next(p for p in parquet.parents if f"device_id={key[1]}" in p.name)
        if device_dir not in to_delete:
            to_delete.append(device_dir)

if not to_delete:
    print('Aucun dossier à supprimer.')
    sys.exit(0)

print(f'{"[DRY-RUN] " if DRY_RUN else ""}Dossiers à supprimer ({len(to_delete)}) :')
for d in to_delete:
    print(f'  {d}')

if DRY_RUN:
    print('\nRelance avec --delete pour supprimer réellement.')
    sys.exit(0)

confirm = input(f'\nSupprimer {len(to_delete)} dossiers ? (oui/non) : ').strip().lower()
if confirm != 'oui':
    print('Annulé.')
    sys.exit(0)

deleted, errors = 0, 0
site_dirs = set()
for d in to_delete:
    try:
        shutil.rmtree(d)
        site_dirs.add(d.parent)
        deleted += 1
    except Exception as e:
        print(f'  ERREUR {d} : {e}')
        errors += 1

# nettoie les dossiers site_id devenus vides
for site_dir in site_dirs:
    if site_dir.exists() and not any(site_dir.iterdir()):
        site_dir.rmdir()
        print(f'Dossier site vide supprimé : {site_dir}')

print(f'\n{deleted} dossiers supprimés, {errors} erreur(s).')
