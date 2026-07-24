import os
import glob
import urllib.parse
import clickhouse_connect
import numpy as np
import pandas as pd
import holidays
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv
from tqdm import tqdm
import random
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader

LOCAL_DIR = "./data/site_occupancy"

def db_connect(type='prod', envir='local'):
    load_dotenv()
    username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
    if type=='prod':
        host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
        password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
    elif type == 'stage':
        host = urllib.parse.quote_plus(os.environ.get("clk_stg"))
        password = urllib.parse.quote_plus(os.environ.get('clk_stg_pass'))
    conn = clickhouse_connect.get_client(host=host, username=username, port=443, password=password)
    return conn

def collect_power_data(site_id, start, end):
    conn = db_connect()
    query = f"""
        SELECT payload_expected_time, eval_prev_hour as raw_value, device_id
        FROM datamart_sous_comptage.distributed_dataset_hourly_payloads_merged
        WHERE site_id = '{site_id}'
        and sensor_name like 'Index %lectr%'
        and payload_expected_time between '{start}' and '{end}'
        settings max_rows_to_read=10000000000000
    """
    df = conn.query_df(query)
    df['payload_expected_time'] = df['payload_expected_time'].dt.tz_localize(None)
    df = df.groupby('payload_expected_time', as_index=False)['raw_value'].sum()
    return df

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
    return df

def preprocess_none(df):
    return df

def preprocess_log_zscore(df):
    df['raw_value'] = np.log1p(df['raw_value'])
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
    'baseload_ratio': preprocess_baseload_ratio,
}

def build_train_set(site_id, preprocess='log_zscore'):
    if preprocess not in PREPROCESSORS:
        raise ValueError(f"preprocess doit être parmi {list(PREPROCESSORS)}, reçu '{preprocess}'")

    parquet_paths = glob.glob(os.path.join(LOCAL_DIR, f"site_id={site_id}", "*.parquet"))
    occ_df = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)

    start = occ_df['payload_timestamp'].min()
    end = occ_df['payload_timestamp'].max()
    raw_data_df = collect_power_data(site_id, start, end)

    train_df = occ_df.merge(
        raw_data_df,
        left_on='payload_timestamp',
        right_on='payload_expected_time',
        how='left',
    )
    train_df['site_id'] = site_id
    train_df = train_df[['site_id', 'payload_timestamp', 'raw_value', 'mean_occupancy']]
    return PREPROCESSORS[preprocess](train_df)

def symmetric_mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, epsilon)
    return np.mean(np.abs(y_true - y_pred) / denom)

def train_lightgbm(train_df, test_df, target_col='mean_occupancy', params=None):
    # site_id exclu des features : l'ensemble d'entraînement ne voit jamais tous les bâtiments
    # (nouveaux sites ajoutés régulièrement), le modèle doit généraliser sans identifiant de bâtiment
    feature_cols = [c for c in train_df.columns if c not in (target_col, 'site_id', 'payload_timestamp')]

    train_df = train_df.dropna(subset=[target_col])
    test_df = test_df.dropna(subset=[target_col])

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    default_params = dict(objective='regression', metric='mae', n_estimators=500, learning_rate=0.05)
    if params:
        default_params.update(params)

    model = lgb.LGBMRegressor(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )
    return model


def predict_lightgbm(model, df, target_col='mean_occupancy'):
    df = df.dropna(subset=[target_col])
    feature_cols = [c for c in df.columns if c not in (target_col, 'site_id', 'payload_timestamp')]
    y_pred = np.clip(model.predict(df[feature_cols]), 0, 1)
    return df[target_col].to_numpy(), y_pred


class Chomp1d(nn.Module):
    # retire le padding "futur" ajouté par la conv dilatée pour garder la convolution causale
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, n_inputs, channels, kernel_size=3, dropout=0.2):
        super().__init__()
        blocks = []
        for i, n_channels in enumerate(channels):
            in_channels = n_inputs if i == 0 else channels[i - 1]
            blocks.append(TemporalBlock(in_channels, n_channels, kernel_size, dilation=2 ** i, dropout=dropout))
        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        return self.network(x)

class TCNGRU(nn.Module):
    # TCN (convolutions causales dilatées) pour capter les motifs locaux/périodiques, suivi d'un GRU
    # pour la dépendance longue portée -> une prédiction d'occupation par pas de temps de la fenêtre
    def __init__(self, n_features, tcn_channels=(32, 32, 32), kernel_size=3, gru_hidden=32, dropout=0.2):
        super().__init__()
        self.tcn = TCN(n_features, list(tcn_channels), kernel_size, dropout)
        self.gru = nn.GRU(tcn_channels[-1], gru_hidden, batch_first=True)
        self.head = nn.Linear(gru_hidden, 1)

    def forward(self, x):
        x = x.transpose(1, 2)              # (batch, n_features, seq_len) pour Conv1d
        x = self.tcn(x)
        x = x.transpose(1, 2)              # (batch, seq_len, tcn_channels) pour le GRU
        x, _ = self.gru(x)
        return torch.sigmoid(self.head(x)).squeeze(-1)   # occupation bornée entre 0 et 1


def normalize_features(train_df, test_df, feature_cols):
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[feature_cols] = (train_df[feature_cols] - mean) / std
    test_df[feature_cols] = (test_df[feature_cols] - mean) / std
    return train_df, test_df


def build_sequences(df, feature_cols, target_col, seq_len=168, stride=24):
    # une séquence par bâtiment (jamais de fenêtre à cheval sur deux sites)
    X, y, mask = [], [], []
    for _, site_df in df.groupby('site_id'):
        site_df = site_df.sort_values('payload_timestamp')
        feats = site_df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
        target = site_df[target_col].to_numpy(dtype=np.float32)
        valid = ~np.isnan(target)

        for start in range(0, len(site_df) - seq_len + 1, stride):
            end = start + seq_len
            X.append(feats[start:end])
            y.append(np.nan_to_num(target[start:end]))
            mask.append(valid[start:end])

    return np.stack(X), np.stack(y), np.stack(mask)


class SequenceDataset(Dataset):
    def __init__(self, X, y, mask):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.mask = torch.from_numpy(mask)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]


def _masked_mae(pred, target, mask):
    return (torch.abs(pred - target) * mask).sum() / mask.sum().clamp(min=1)


def train_tcn_gru(train_df, test_df, target_col='mean_occupancy', seq_len=168, stride=24,
                   batch_size=64, epochs=30, lr=1e-3, patience=5, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    feature_cols = [c for c in train_df.columns if c not in (target_col, 'site_id', 'payload_timestamp')]

    train_df, test_df = normalize_features(train_df, test_df, feature_cols)

    X_train, y_train, mask_train = build_sequences(train_df, feature_cols, target_col, seq_len, stride)
    X_test, y_test, mask_test = build_sequences(test_df, feature_cols, target_col, seq_len, stride)

    train_loader = DataLoader(SequenceDataset(X_train, y_train, mask_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SequenceDataset(X_test, y_test, mask_test), batch_size=batch_size, shuffle=False)

    model = TCNGRU(n_features=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad()
            loss = _masked_mae(model(xb), yb, mb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, mb in test_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                val_loss += _masked_mae(model(xb), yb, mb).item() * xb.size(0)
        val_loss /= len(test_loader.dataset)
        print(f"epoch {epoch + 1}/{epochs} - train_mae={train_loss:.4f} - val_mae={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stopping à l'epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    return model, test_df, feature_cols


def predict_tcn_gru(model, df, feature_cols, target_col='mean_occupancy', seq_len=168, stride=24, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    X, y, mask = build_sequences(df, feature_cols, target_col, seq_len, stride)
    loader = DataLoader(SequenceDataset(X, y, mask), batch_size=64, shuffle=False)

    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for xb, yb, mb in loader:
            pred = model(xb.to(device)).cpu().numpy()
            yb, mb = yb.numpy(), mb.numpy().astype(bool)
            y_true_all.append(yb[mb])
            y_pred_all.append(pred[mb])
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


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

    plt.figure(figsize=(14, 5))
    plt.plot(timestamps, y_true, label='vérité terrain')
    plt.plot(timestamps, y_pred, label='prédiction TCN-GRU')
    plt.title(f"Occupation prédite (TCN-GRU) vs réelle - site {site_id}")
    plt.xlabel('date')
    plt.ylabel("taux d'occupation")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_prediction_example(model, test_df, site_id=None, target_col='mean_occupancy'):
    if site_id is None:
        site_id = test_df['site_id'].sample(1, random_state=142).iloc[0]

    site_df = test_df[test_df['site_id'] == site_id].dropna(subset=[target_col]).sort_values('payload_timestamp')
    feature_cols = [c for c in site_df.columns if c not in (target_col, 'site_id', 'payload_timestamp')]

    y_true = site_df[target_col]
    y_pred = np.clip(model.predict(site_df[feature_cols]), 0, 1)

    plt.figure(figsize=(14, 5))
    plt.plot(site_df['payload_timestamp'], y_true, label='vérité terrain')
    plt.plot(site_df['payload_timestamp'], y_pred, label='prédiction')
    plt.title(f"Occupation prédite vs réelle - site {site_id}")
    plt.xlabel('date')
    plt.ylabel("taux d'occupation")
    plt.legend()
    plt.tight_layout()
    plt.show()

def list_site_ids():
    return [
        entry.split('=', 1)[-1]
        for entry in os.listdir(LOCAL_DIR)
        if entry.startswith('site_id=') and os.path.isdir(os.path.join(LOCAL_DIR, entry))
    ]

def build_dataset(site_ids, preprocess='log_zscore'):
    dfs = [feature_creator(build_train_set(site_id, preprocess=preprocess)) for site_id in tqdm(site_ids)]
    return pd.concat(dfs, ignore_index=True)


def build_train_test_datasets(preprocess='log_zscore', test_ratio=0.3, seed=142):
    site_ids = list_site_ids()

    # split par site (pas par date) : le modèle doit être évalué sur des bâtiments jamais vus en entraînement
    rng = random.Random(seed)
    train_site_ids = rng.sample(site_ids, k=round((1 - test_ratio) * len(site_ids)))
    test_site_ids = [s for s in site_ids if s not in train_site_ids]

    train_df = build_dataset(train_site_ids, preprocess=preprocess)
    test_df = build_dataset(test_site_ids, preprocess=preprocess)
    return train_df, test_df


def decalage_phase(y_true, y_pred, max_shift=12):
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


def compute_metrics(y_true, y_pred):
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': mean_squared_error(y_true, y_pred) ** 0.5,
        'smape': symmetric_mean_absolute_percentage_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'pearson_shift': decalage_phase(y_true, y_pred),
    }


MODEL_TYPES = ('lightgbm', 'tcn_gru')


def train_model(model_type, train_df, test_df, **kwargs):
    if model_type == 'lightgbm':
        model = train_lightgbm(train_df, test_df, **kwargs)
        artifacts = {'test_df': test_df}
    elif model_type == 'tcn_gru':
        model, test_df_norm, feature_cols = train_tcn_gru(train_df, test_df, **kwargs)
        artifacts = {
            'test_df': test_df_norm,
            'feature_cols': feature_cols,
            'seq_len': kwargs.get('seq_len', 168),
            'stride': kwargs.get('stride', 24),
        }
    else:
        raise ValueError(f"model_type doit être parmi {MODEL_TYPES}, reçu '{model_type}'")
    return model, artifacts


def evaluate_model(model_type, model, artifacts, target_col='mean_occupancy'):
    if model_type == 'lightgbm':
        y_true, y_pred = predict_lightgbm(model, artifacts['test_df'], target_col)
    elif model_type == 'tcn_gru':
        y_true, y_pred = predict_tcn_gru(
            model, artifacts['test_df'], artifacts['feature_cols'],
            target_col, artifacts['seq_len'], artifacts['stride'],
        )
    else:
        raise ValueError(f"model_type doit être parmi {MODEL_TYPES}, reçu '{model_type}'")

    metrics = compute_metrics(y_true, y_pred)
    print(metrics)
    return metrics


def plot_example(model_type, model, artifacts, site_id=None, target_col='mean_occupancy'):
    if model_type == 'lightgbm':
        plot_prediction_example(model, artifacts['test_df'], site_id, target_col)
    elif model_type == 'tcn_gru':
        plot_prediction_example_tcn_gru(
            model, artifacts['test_df'], artifacts['feature_cols'],
            site_id, target_col, artifacts['seq_len'],
        )
    else:
        raise ValueError(f"model_type doit être parmi {MODEL_TYPES}, reçu '{model_type}'")


if __name__ == '__main__':
    train_df, test_df = build_train_test_datasets(preprocess='log_zscore')

    model_type = 'tcn_gru'  # ou 'tcn_gru' / lightgbm
    model, artifacts = train_model(model_type, train_df, test_df)
    evaluate_model(model_type, model, artifacts)
    plot_example(model_type, model, artifacts)