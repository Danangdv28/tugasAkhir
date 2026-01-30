# lstm_dataset.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

class SNRSequenceDataset(Dataset):
    def __init__(self, csv_file, features, lookback=20, horizons=[1,5,10]):
        df = pd.read_csv(csv_file)
        df["fog_visibility_m"] = df["fog_visibility_m"].fillna(0)

        self.features = features
        self.lookback = lookback
        self.horizons = horizons

        # target matrix
        targets = []
        for h in horizons:
            targets.append(df["snr_db"].shift(-h))
        df_targets = pd.concat(targets, axis=1)
        df_targets.columns = [f"snr_t+{h}" for h in horizons]

        df = pd.concat([df, df_targets], axis=1)
        df = df.dropna().reset_index(drop=True)

        X = df[features].values
        y = df[df_targets.columns].values

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.X_seq, self.y_seq = [], []
        for i in range(len(X) - lookback):
            self.X_seq.append(X[i:i+lookback])
            self.y_seq.append(y[i+lookback])

        self.X_seq = torch.tensor(self.X_seq, dtype=torch.float32)
        self.y_seq = torch.tensor(self.y_seq, dtype=torch.float32)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.y_seq[idx]