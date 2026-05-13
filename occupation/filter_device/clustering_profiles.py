import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import plotly.graph_objects as go
import mlflow

mlflow.set_tracking_uri("arn:aws:sagemaker:eu-west-3:585762237892:mlflow-app/app-N4LROSLQN5BN")

INPUT_FOLDER = './data/sample_s3/'
SELECTIONS_CSV = f'./filter_device/occupancy_profile_vt_labels.csv'
ROLLING = 3
N_WEEKS = 4
N_HOURS = N_WEEKS * 7 * 24 # 672 points
FFT_IDX_DAY = N_WEEKS * 7  # 28 → 1 cycle/jour sur 672h
FFT_COMPONENTS = N_WEEKS * 30  # 120
DAILY_POWER_THRESH = 0.01
RDM = 42 # seed

#%% fonctions
def log_classifier_run(model, X_train, y_train, X_test, y_test, run_name, note=None, y_pred=None, extra_figures=None, dataset_tags=None):
    if y_pred is None:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    method = type(model).__name__ if model else "naive"
    report = classification_report(y_test, y_pred, target_names=['Non-occupation', 'Occupation'])
    mlflow.set_experiment("Occupation-device-filter")
    with mlflow.start_run(run_name=run_name):
        if note:
            mlflow.set_tag("mlflow.note.content", note)
        if dataset_tags:
            for k, v in dataset_tags.items():
                mlflow.set_tag(k, v)
        mlflow.log_param("method", method)
        if model:
            mlflow.log_params(model.get_params())
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric_name}", value)
        cm = confusion_matrix(y_test, y_pred)
        fig = go.Figure(go.Heatmap(
            z=cm,
            x=['Prédit Non-occ', 'Prédit Occ'],
            y=['Vrai Non-occ', 'Vrai Occ'],
            colorscale='Blues',
            text=cm, texttemplate='%{text}',
            hovertemplate='%{y} → %{x} : %{z}<extra></extra>',
        ))
        fig.update_layout(
            title=f'Matrice de confusion — {method}',
            template='simple_white', height=350,
            margin=dict(t=50, b=50, l=100, r=20),
        )
        mlflow.log_figure(fig, "confusion_matrix.png")
        if extra_figures:
            for filename, fig_extra in extra_figures.items():
                mlflow.log_figure(fig_extra, filename)
    return y_pred

def make_filter_figure(power_daily, power_subdaily, y_pred, threshold):
    fig = go.Figure()
    for flag, color, name in [(True, '#457b9d', 'Conservé'), (False, '#f4a261', 'Retiré')]:
        mask = y_pred.astype(bool) == flag
        fig.add_trace(go.Scatter(
            x=power_daily[mask], y=power_subdaily[mask],
            mode='markers',
            marker=dict(color=color, size=7, opacity=0.7),
            name=name,
            hovertemplate=f'daily=%{{x:.3f}} subdaily=%{{y:.3f}}<extra>{name}</extra>',
        ))
    fig.add_vline(x=threshold, line_dash='dash', line_color='grey',
                  annotation_text=f'seuil={threshold}')
    fig.add_shape(type='line', x0=0, y0=0, x1=0.5, y1=0.5,
                  line=dict(dash='dot', color='grey'))
    fig.update_layout(
        title='Filtre jour/nuit — puissance journalière vs sub-journalière (test)',
        template='simple_white', height=500,
        xaxis_title='Puissance relative 1 cycle/jour (24h)',
        yaxis_title='Puissance relative cycles sub-journaliers (<24h)',
    )
    return fig

def load_parquet_files(folder):
    files = []
    for path in sorted(Path(folder).rglob('*.parquet')):
        parts = {p.split('=')[0]: p.split('=')[1] for p in path.parts if '=' in p}
        if 'device_id' in parts:
            files.append({
                'site_id':   parts.get('site_id', ''),
                'device_id': parts.get('device_id', ''),
                'sensor_id': parts.get('sensor_id', ''),
                'path':      str(path),
            })
    return pd.DataFrame(files)

def compute_raw_series(path):
    """Série horaire lissée (ROLLING), premiers N_HOURS points, interpolée si lacunes."""
    df = pd.read_parquet(path)
    df['payload_expected_time'] = pd.to_datetime(df['payload_expected_time'])
    df = df.sort_values('payload_expected_time').set_index('payload_expected_time')
    df['raw_value'] = df['raw_value'].rolling(window=ROLLING, center=True, min_periods=1).mean()
    hourly = df['raw_value'].resample('1h').mean()
    hourly = hourly.iloc[:N_HOURS].interpolate(method='time').fillna(0)
    if len(hourly) < N_HOURS:
        hourly = hourly.reindex(range(N_HOURS), fill_value=0)
    return hourly.values[:N_HOURS]

def compute_fft_features(series):
    """Spectre de puissance normalisé sur les FFT_COMPONENTS premières fréquences (hors DC)."""
    mag = np.abs(np.fft.rfft(series))
    total_power = mag[1:].sum() + 1e-8
    return mag[1:FFT_COMPONENTS + 1] / total_power

def get_parquet_path(folder, site_id, device_id, sensor_id):
    pattern = f"site_id={site_id}/device_id={device_id}"
    if sensor_id:
        pattern += f"/sensor_id={sensor_id}"
    paths = list(Path(folder).glob(f"{pattern}/**/*.parquet")) or list(Path(folder).glob(f"{pattern}/*.parquet"))
    return str(paths[0]) if paths else None

#%% Chargement de la vérité terrain
files_df = pd.read_csv(SELECTIONS_CSV)
if 'sensor_id' not in files_df.columns:
    files_df['sensor_id'] = ''
files_df['sensor_id'] = files_df['sensor_id'].fillna('')
files_df = (files_df
    .drop_duplicates(subset=['site_id', 'device_id', 'sensor_id'], keep='last')
    .dropna(subset=['verite_terrain'])
    .copy()
)
files_df['verite_terrain'] = files_df['verite_terrain'].astype(int)
files_df['path'] = files_df.apply(
    lambda r: get_parquet_path(INPUT_FOLDER, r['site_id'], r['device_id'], r['sensor_id']), axis=1
)
files_df = files_df.reset_index(drop=True)

#%% Build feature matrix with fourier transform 
raw_series = []
for _, row in files_df.iterrows():
    raw_series.append(compute_raw_series(row['path']))
X_fft = np.vstack([compute_fft_features(s) for s in raw_series])

print(f'{len(files_df)} profils labellisés  '
      f"({(files_df['verite_terrain'] == 1).sum()} occupation, "
      f"{(files_df['verite_terrain'] == 0).sum()} non-occupation)")

if (files_df['verite_terrain'] == 0).sum() < 1:
    raise ValueError('Pas assez de cas non-occupation (au moins 1 requis)')

vt = files_df['verite_terrain'].values

#%% train / test split
n_occ = (vt == 1).sum()
n_non = (vt == 0).sum()
print(f'\n{len(vt)} profils pour la classification ({n_occ} occupation, {n_non} non-occupation)')

idx_all = np.arange(len(X_fft))
idx_train, idx_test, y_train, y_test = train_test_split(
    idx_all, vt, test_size=0.3, random_state=RDM, stratify=vt)
X_train, X_test = X_fft[idx_train], X_fft[idx_test]

for cls, cls_name in [(0, 'non-occupation'), (1, 'occupation')]:
    pct = (y_train == cls).mean()
    if pct < 0.10:
        print(f'AVERTISSEMENT : {cls_name} représente seulement {pct:.0%} du train (< 10%)')

dataset_tags = {
    "dataset.n_total": len(files_df),
    "dataset.n_occupation": int((files_df['verite_terrain'] == 1).sum()),
    "dataset.n_non_occupation": int((files_df['verite_terrain'] == 0).sum()),
}

#%% Baseline : filtre naïf sur le marquage de la variation jour/nuit et l'existence d'un cycle 24h dominant ───────────────
power_daily_test = X_test[:, FFT_IDX_DAY - 1]
power_subdaily_test = X_test[:, FFT_IDX_DAY:].max(axis=1)
y_pred_naive = ((power_daily_test > DAILY_POWER_THRESH) & (power_daily_test > power_subdaily_test)).astype(int)

fig_filter = make_filter_figure(power_daily_test, power_subdaily_test, y_pred_naive, DAILY_POWER_THRESH)

log_classifier_run(
    model=None, X_train=None, y_train=None, X_test=None, y_test=y_test,
    run_name="Baseline_filtre_journuit",
    note="Méthode naïve : valid_mask=True → occupation, False → non-occupation.",
    y_pred=y_pred_naive,
    extra_figures={"filter_journuit.png": fig_filter},
    dataset_tags=dataset_tags,
)

#%% Linear Discriminant Analysis
lda_clf = LinearDiscriminantAnalysis(solver='lsqr', priors=[0.5, 0.5])
y_pred_lda = log_classifier_run(
    lda_clf, X_train, y_train, X_test, y_test,
    run_name="LDA_Sample_labeled_1305",
    note="Test sur échantillon de données labelisées en local avec "
         "une séparation stratifiée 70/30 pour les ensembles d'entraînement et de test.",
    dataset_tags=dataset_tags,
)

#%% SVM à noyau RBF
svm_clf = SVC(kernel='rbf', class_weight='balanced', random_state=RDM)
y_pred_svm = log_classifier_run(
    svm_clf, X_train, y_train, X_test, y_test,
    run_name="SVM-RBF_Sample_labeled_1305",
    note="Test sur échantillon de données labelisées en local avec "
         "une séparation stratifiée 70/30 pour les ensembles d'entraînement et de test.",
    dataset_tags=dataset_tags,
)

# pred_df = files_df.iloc[idx_test][['site_id', 'device_id', 'sensor_id']].copy().reset_index(drop=True)
# pred_df['verite_terrain'] = y_test
# pred_df['classif_svm']    = y_pred_svm
# pred_df.to_csv(f'./prediction.csv', index=False)
# print(f'Prédictions test : ./prediction.csv ({len(pred_df)} lignes)')

# MODEL_DIR  = Path('./model')
# MODEL_DIR.mkdir(exist_ok=True)
# model_path = MODEL_DIR / f'svm_{date.today()}.pkl'
# joblib.dump(svm_clf, model_path)
# print(f'Modèle SVM sauvegardé : {model_path}')

# ── prédiction SVM sur les fichiers sans verite_terrain ──────────────────────
# labeled_keys = set(zip(files_df['site_id'], files_df['device_id'], files_df['sensor_id']))
# all_files    = load_parquet_files(INPUT_FOLDER)
# unlabeled_df = all_files[
#     ~all_files.apply(lambda r: (r['site_id'], r['device_id'], r['sensor_id']) in labeled_keys, axis=1)
# ].reset_index(drop=True)
# print(f'\n{len(unlabeled_df)} fichiers sans verite_terrain à prédire')

# if len(unlabeled_df):
#     raw_unlabeled = [compute_raw_series(row['path']) for _, row in unlabeled_df.iterrows()]
#     X_unlabeled   = np.vstack([compute_fft_features(s) for s in raw_unlabeled])
#     preds         = svm_clf.predict(X_unlabeled)

#     new_rows = unlabeled_df[['site_id', 'device_id', 'sensor_id']].copy()
#     new_rows['verite_terrain'] = pd.NA
#     new_rows['classif_svm']    = preds

#     selections_updated = pd.concat([files_df, new_rows], ignore_index=True)
#     selections_updated.to_csv(SELECTIONS_CSV, index=False)
#     print(f'{(preds == 1).sum()} occupation, {(preds == 0).sum()} non-occupation prédits')
#     print(f'occupancy_profile_vt_labels.csv mis à jour ({len(selections_updated)} lignes total)')
# else:
#     print('Aucun fichier non labellisé trouvé.')
