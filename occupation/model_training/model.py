import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


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

