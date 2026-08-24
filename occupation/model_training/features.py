"""
Feature engineering et preprocessing du signal de consommation.

@author: Myriam Lopez
"""

import numpy as np
import holidays

def feature_creator(df):
    df = df.sort_values(['site_id', 'payload_timestamp']).reset_index(drop=True)

    # encodage cyclique de l'heure (0-23) : sin/cos pour que minuit et 23h restent proches dans l'espace des features
    hour = df['payload_timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # encodage cyclique du jour de la semaine (0=lundi ... 6=dimanche)
    dow = df['payload_timestamp'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    df['is_weekend'] = dow.isin([5, 6]).astype(int)

    fr_holidays = holidays.France()
    df['is_holiday'] = df['payload_timestamp'].dt.date.isin(fr_holidays).astype(int)

    # lags et stats glissantes de la conso brute, calculés par bâtiment (site_id) pour ne pas mélanger les séries entre sites
    grouped = df.groupby('site_id')['raw_value']
    df['lag_1h'] = grouped.shift(1)
    df['lag_24h'] = grouped.shift(24)
    df['lag_168h'] = grouped.shift(168)
    df['rolling_mean_3h'] = grouped.transform(lambda s: s.rolling(3).mean())
    df['rolling_mean_24h'] = grouped.transform(lambda s: s.rolling(24).mean())
    df['rolling_std_24h'] = grouped.transform(lambda s: s.rolling(24).std())

    # dérivée de la conso : variation par rapport à l'heure précédente, utile pour repérer les transitions d'occupation
    df['delta_raw_value'] = grouped.diff()

    # TODO: température extérieure pas encore branchée (source pas encore définie) — à ajouter ici
    # (ex: brute + moyenne glissante + degrés-heure de chauffe/clim)

    # float32 plutôt que float64 : divise par ~2 la mémoire du dataset (utile sur les petites instances SageMaker)
    float_cols = df.select_dtypes(include='float64').columns
    df[float_cols] = df[float_cols].astype('float32')

    return df

def preprocess_none(df):
    return df

def preprocess_log_zscore(df):
    df['raw_value'] = np.log1p(df['raw_value'])
    grouped = df.groupby('site_id')['raw_value']
    df['raw_value'] = (df['raw_value'] - grouped.transform('mean')) / grouped.transform('std')
    return df

def preprocess_zscore(df):
    # comme log_zscore mais sans log1p : évite d'écraser le haut de la distribution de conso
    grouped = df.groupby('site_id')['raw_value']
    df['raw_value'] = (df['raw_value'] - grouped.transform('mean')) / grouped.transform('std')
    return df

def preprocess_baseload_ratio(df, window=24 * 30, quantile=0.05):
    # exprime la conso relativement au baseload du bâtiment (percentile bas glissant, proche de la conso hors occupation)
    grouped = df.groupby('site_id')['raw_value']
    baseload = grouped.transform(lambda s: s.rolling(window, min_periods=1).quantile(quantile))
    df['raw_value'] = df['raw_value'] / baseload.replace(0, np.nan)
    return df


PREPROCESSORS = {
    'none': preprocess_none,
    'log_zscore': preprocess_log_zscore,
    'zscore': preprocess_zscore,
    'baseload_ratio': preprocess_baseload_ratio,
}


def normalize_features(train_df, test_df, feature_cols):
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[feature_cols] = (train_df[feature_cols] - mean) / std
    test_df[feature_cols] = (test_df[feature_cols] - mean) / std
    return train_df, test_df
