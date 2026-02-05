# Dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class SNRSequenceDataset(Dataset):
    def __init__(self, csv_file, features, lookback=10, scale=True):
        df = pd.read_csv(csv_file)
        df["fog_visibility_m"] = df["fog_visibility_m"].fillna(0)

        # One-hot CCS
        if "ccs" in df.columns:
            df = pd.get_dummies(df, columns=["ccs"], prefix="ccs", drop_first=True)

        # === INI HARUS DULU ===
        X = df[features].values.astype(np.float32)
        y = df["snr_db"].shift(-1).values.astype(np.float32)

        # Drop last NaN target
        X = X[:-1]
        y = y[:-1]

        # Optional scaling (AMAN)
        if scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        self.X = X
        self.y = y
        self.lookback = lookback

    def __len__(self):
        return len(self.X) - self.lookback

    def __getitem__(self, idx):
        X_seq = self.X[idx : idx + self.lookback]
        y = self.y[idx + self.lookback]
        return torch.tensor(X_seq), torch.tensor(y)
