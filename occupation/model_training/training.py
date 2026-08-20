"""
Entraînement et prédiction des modèles d'occupation (LightGBM & TCN-GRU).

@author: Myriam Lopez
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
from torch.utils.data import DataLoader

from datasets import build_sequences, SequenceDataset
from features import normalize_features
from model import TCNGRU

MODEL_TYPES = ('lightgbm', 'tcn_gru')


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
    groups = df[['site_id', 'payload_timestamp']].reset_index(drop=True)
    return df[target_col].to_numpy(), y_pred, groups


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
    X, y, mask, site_ids, timestamps = build_sequences(
        df, feature_cols, target_col, seq_len, stride, return_groups=True,
    )
    loader = DataLoader(SequenceDataset(X, y, mask), batch_size=64, shuffle=False)

    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for xb, yb, mb in loader:
            pred = model(xb.to(device)).cpu().numpy()
            yb, mb = yb.numpy(), mb.numpy().astype(bool)
            y_true_all.append(yb[mb])
            y_pred_all.append(pred[mb])

    groups = pd.DataFrame({
        'site_id': site_ids[mask],
        'payload_timestamp': timestamps[mask],
    })
    return np.concatenate(y_true_all), np.concatenate(y_pred_all), groups


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
