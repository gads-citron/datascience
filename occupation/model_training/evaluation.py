"""
Métriques d'évaluation et visualisations des prédictions d'occupation.

@author: Myriam Lopez
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tslearn.metrics import dtw as tslearn_dtw

import training as trn


HOUR_STRATA = [
    ('23h-4h', lambda h: (h >= 23) | (h < 4)),
    ('4h-9h', lambda h: (h >= 4) & (h < 9)),
    ('9h-14h', lambda h: (h >= 9) & (h < 14)),
    ('14h-19h', lambda h: (h >= 14) & (h < 19)),
    ('19h-23h', lambda h: (h >= 19) & (h < 23)),
]

DAY_STRATA = [('semaine', False), ('week-end', True)]

WEEKDAY_LABELS = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']

def symmetric_mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, epsilon)
    return np.mean(np.abs(y_true - y_pred) / denom)

def normalised_mean_absolute_error(y_true, y_pred, epsilon=1e-8):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = max(np.abs(y_true).mean(), epsilon)
    return mean_absolute_error(y_true, y_pred) / denom

def mean_bias_error(y_true, y_pred):
    # positif = surestimation (y_pred > y_true), négatif = sous-estimation
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(y_pred - y_true)

def bias_amplitude_by_hour(y_true, y_pred, groups):
    df = pd.DataFrame({
        'hour': groups['payload_timestamp'].dt.hour.reset_index(drop=True),
        'y_true': np.asarray(y_true),
        'y_pred': np.asarray(y_pred),
    })
    rows = [
        {
            'hour': hour,
            'mbe': mean_bias_error(hour_df['y_true'], hour_df['y_pred']),
            'nmae': normalised_mean_absolute_error(hour_df['y_true'], hour_df['y_pred']),
            'n': len(hour_df),
        }
        for hour, hour_df in df.groupby('hour')
    ]
    return pd.DataFrame(rows).sort_values('hour').reset_index(drop=True)

def bias_amplitude_by_strata(y_true, y_pred, groups):
    df = pd.DataFrame({
        'hour': groups['payload_timestamp'].dt.hour.reset_index(drop=True),
        'is_weekend': groups['payload_timestamp'].dt.dayofweek.isin([5, 6]).reset_index(drop=True),
        'y_true': np.asarray(y_true),
        'y_pred': np.asarray(y_pred),
    })
    rows = []
    for day_label, is_weekend in DAY_STRATA:
        day_df = df[df['is_weekend'] == is_weekend]
        for hour_label, in_strate in HOUR_STRATA:
            strate_df = day_df[in_strate(day_df['hour'])]
            rows.append({
                'jour': day_label,
                'strate': hour_label,
                'mbe': mean_bias_error(strate_df['y_true'], strate_df['y_pred']),
                'nmae': normalised_mean_absolute_error(strate_df['y_true'], strate_df['y_pred']),
                'n': len(strate_df),
            })
    return pd.DataFrame(rows)

def fwhm(values):
    # largeur (en heures) du pic journalier : nombre d'heures au-dessus du demi-maximum,
    # demi-maximum pris par rapport au minimum de la journée (pas 0, pour absorber un bruit de fond nocturne)
    values = np.asarray(values, dtype=np.float64)
    baseline = values.min()
    peak = values.max()
    if peak - baseline <= 1e-8:
        return np.nan
    half_max = baseline + (peak - baseline) / 2
    return int(np.sum(values >= half_max))

def fwhm_by_day(y_true, y_pred, groups):
    df = groups[['site_id', 'payload_timestamp']].reset_index(drop=True).copy()
    df['y_true'] = np.asarray(y_true)
    df['y_pred'] = np.asarray(y_pred)
    df['date'] = df['payload_timestamp'].dt.date

    rows = []
    for (site_id, date), day_df in df.groupby(['site_id', 'date']):
        fwhm_true = fwhm(day_df['y_true'])
        fwhm_pred = fwhm(day_df['y_pred'])
        rows.append({
            'site_id': site_id,
            'date': date,
            'fwhm_true': fwhm_true,
            'fwhm_pred': fwhm_pred,
            'diff': fwhm_pred - fwhm_true,
            'n': len(day_df),
        })
    return pd.DataFrame(rows)

def decalage_phase(y_true, y_pred, max_shift=6):
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    maxi = 0
    horaire = 0
    for d in range(0, max_shift + 1):
        y_pred_shifted = y_pred.shift(d)
        valid = y_pred_shifted.notna()
        score = y_pred_shifted[valid].corr(y_true[valid])
        if score > maxi:
            maxi = score
            horaire = d
    return maxi, horaire

def dtw_score(y_true, y_pred, window=6):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1, 1)
    return tslearn_dtw(y_true, y_pred, global_constraint="sakoe_chiba", sakoe_chiba_radius=window)

def _weekly_score(y_true, y_pred, groups, score_fn):
    # une série continue par (site, semaine) : DTW/RMSE n'ont de sens que sur une plage temporelle contiguë
    df = groups[['site_id', 'payload_timestamp']].reset_index(drop=True).copy()
    df['y_true'] = np.asarray(y_true)
    df['y_pred'] = np.asarray(y_pred)
    df['année'] = df['payload_timestamp'].dt.year
    df['week_number'] = df['payload_timestamp'].dt.isocalendar().week

    scores = []
    for _, week_df in df.groupby(['site_id', 'année', 'week_number']):
        if len(week_df) < 2:
            continue
        week_df = week_df.sort_values('payload_timestamp')
        scores.append(score_fn(week_df['y_true'], week_df['y_pred']))
    return float(np.mean(scores)) if scores else np.nan

def dtw_score_weekly(y_true, y_pred, groups, window=6):
    return _weekly_score(y_true, y_pred, groups, lambda t, p: dtw_score(t, p, window=window))

def rmse_score_weekly(y_true, y_pred, groups):
    return _weekly_score(y_true, y_pred, groups, lambda t, p: mean_squared_error(t, p) ** 0.5)

def fwhm_by_weekday(y_true, y_pred, groups):
    daily = fwhm_by_day(y_true, y_pred, groups)
    daily['weekday'] = pd.to_datetime(daily['date']).dt.dayofweek

    rows = []
    for weekday, label in enumerate(WEEKDAY_LABELS):
        day_rows = daily[daily['weekday'] == weekday]
        rows.append({
            'jour': label,
            'fwhm_true': day_rows['fwhm_true'].mean(),
            'fwhm_pred': day_rows['fwhm_pred'].mean(),
            'diff': day_rows['diff'].mean(),
            'n_jours': len(day_rows),
        })
    return pd.DataFrame(rows)

def fwhm_diff_summary(fwhm_by_weekday_df):
    # moyenne pondérée par n_jours (plutôt qu'une moyenne simple des 7 moyennes journalières déjà agrégées)
    df = fwhm_by_weekday_df
    weekday_mask = df['jour'].isin(['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi'])
    weekend_mask = df['jour'].isin(['samedi', 'dimanche'])

    def weighted_mean_diff(sub_df):
        return (sub_df['diff'] * sub_df['n_jours']).sum() / sub_df['n_jours'].sum()

    return {
        'fwhm_diff_global': weighted_mean_diff(df),
        'fwhm_diff_semaine': weighted_mean_diff(df[weekday_mask]),
        'fwhm_diff_weekend': weighted_mean_diff(df[weekend_mask]),
    }

def bias_direction_summary(bias_by_strata_df):
    # moyenne du mbe des strates en sur-estimation (mbe > 0) et en sous-estimation (mbe < 0), semaine vs week-end
    df = bias_by_strata_df
    result = {}
    for jour_label, key in (('semaine', 'semaine'), ('week-end', 'weekend')):
        day_df = df[df['jour'] == jour_label]
        over = day_df.loc[day_df['mbe'] > 0, 'mbe']
        under = day_df.loc[day_df['mbe'] < 0, 'mbe']
        result[f'bias_overestimation_{key}'] = over.mean() if len(over) else np.nan
        result[f'bias_underestimation_{key}'] = under.mean() if len(under) else np.nan
    return result

def compute_metrics(y_true, y_pred, groups=None):
    metrics = {
        'rmse': mean_squared_error(y_true, y_pred) ** 0.5,
        'nmae': normalised_mean_absolute_error(y_true, y_pred),
    }
    if groups is not None:
        metrics['dtw'] = dtw_score_weekly(y_true, y_pred, groups, window=6)
        metrics['rmse_weekly'] = rmse_score_weekly(y_true, y_pred, groups)
    else:
        metrics['dtw'] = dtw_score(y_true, y_pred)
    return metrics

def plot_prediction_example(model, test_df, site_id=None, target_col='mean_occupancy'):
    if site_id is None:
        site_id = test_df['site_id'].sample(1, random_state=142).iloc[0]

    site_df = test_df[test_df['site_id'] == site_id].dropna(subset=[target_col]).sort_values('payload_timestamp')
    feature_cols = [c for c in site_df.columns if c not in (target_col, 'site_id', 'payload_timestamp')]

    y_true = site_df[target_col]
    y_pred = np.clip(model.predict(site_df[feature_cols]), 0, 1)
    y_pred = np.where(site_df['raw_value'].isna(), np.nan, y_pred)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(site_df['payload_timestamp'], y_true, label='vérité terrain')
    ax.plot(site_df['payload_timestamp'], y_pred, label='prédiction')
    ax.set_title(f"Occupation prédite vs réelle - site {site_id}")
    ax.set_xlabel('date')
    ax.set_ylabel("taux d'occupation")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_prediction_example_tcn_gru(model, test_df, feature_cols, site_id=None,
                                     target_col='mean_occupancy', seq_len=168, device=None):
    # test_df doit être le df déjà normalisé renvoyé par train_tcn_gru (mêmes moyenne/écart-type que l'entraînement)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if site_id is None:
        site_id = test_df['site_id'].sample(1, random_state=142).iloc[0]

    site_df = test_df[test_df['site_id'] == site_id].sort_values('payload_timestamp')
    n_windows = len(site_df) // seq_len
    site_df = site_df.iloc[: n_windows * seq_len]

    feats = site_df[feature_cols].fillna(0).to_numpy(dtype=np.float32).reshape(n_windows, seq_len, -1)
    X = torch.from_numpy(feats).to(device)

    model.eval()
    with torch.no_grad():
        y_pred = model(X).cpu().numpy().reshape(-1)

    y_true = site_df[target_col].to_numpy()
    timestamps = site_df['payload_timestamp']

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(timestamps, y_true, label='vérité terrain')
    ax.plot(timestamps, y_pred, label='prédiction TCN-GRU')
    ax.set_title(f"Occupation prédite (TCN-GRU) vs réelle - site {site_id}")
    ax.set_xlabel('date')
    ax.set_ylabel("taux d'occupation")
    ax.legend()
    fig.tight_layout()
    return fig

def evaluate_model(model_type, model, artifacts, target_col='mean_occupancy'):
    if model_type == 'lightgbm':
        y_true, y_pred, groups = trn.predict_lightgbm(model, artifacts['test_df'], target_col)
    elif model_type == 'tcn_gru':
        y_true, y_pred, groups = trn.predict_tcn_gru(
            model, artifacts['test_df'], artifacts['feature_cols'],
            target_col, artifacts['seq_len'], artifacts['stride'],
        )
    else:
        raise ValueError(f"model_type doit être parmi {trn.MODEL_TYPES}, reçu '{model_type}'")

    # prédictions volontairement NaN (raw_value manquant, cf. predict_lightgbm) : à exclure des métriques,
    # sklearn ne tolère aucun NaN dans mean_squared_error/etc.
    valid = ~np.isnan(y_pred)
    if not valid.all():
        print(f"{(~valid).sum()} prédictions ignorées pour le calcul des métriques (raw_value manquant) sur {len(y_pred)}")
        y_true, y_pred, groups = y_true[valid], y_pred[valid], groups[valid].reset_index(drop=True)

    metrics = compute_metrics(y_true, y_pred, groups)
    bias_by_strata = bias_amplitude_by_strata(y_true, y_pred, groups)
    fwhm_by_weekday_ = fwhm_by_weekday(y_true, y_pred, groups)
    fwhm_diff_summary_ = fwhm_diff_summary(fwhm_by_weekday_)
    bias_direction_summary_ = bias_direction_summary(bias_by_strata)

    print(metrics)
    print(bias_by_strata)
    print(fwhm_by_weekday_)
    print(fwhm_diff_summary_)
    print(bias_direction_summary_)

    return {
        'metrics': metrics,
        'bias_by_hour': bias_amplitude_by_hour(y_true, y_pred, groups),
        'bias_by_strata': bias_by_strata,
        'fwhm_by_day': fwhm_by_day(y_true, y_pred, groups),
        'fwhm_by_weekday': fwhm_by_weekday_,
        'fwhm_diff_summary': fwhm_diff_summary_,
        'bias_direction_summary': bias_direction_summary_,
    }

def plot_example(model_type, model, artifacts, site_id=None, target_col='mean_occupancy'):
    if model_type == 'lightgbm':
        return plot_prediction_example(model, artifacts['test_df'], site_id, target_col)
    elif model_type == 'tcn_gru':
        return plot_prediction_example_tcn_gru(
            model, artifacts['test_df'], artifacts['feature_cols'],
            site_id, target_col, artifacts['seq_len'],
        )
    else:
        raise ValueError(f"model_type doit être parmi {trn.MODEL_TYPES}, reçu '{model_type}'")
