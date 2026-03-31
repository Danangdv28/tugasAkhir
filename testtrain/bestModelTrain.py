# ==========================================================
# 1. IMPORT
# ==========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==========================================================
# 2. LOAD DATASET BARU
# ==========================================================
df = pd.read_csv("testtrain/bestModeTesting/urban_140GHz.csv")  # GANTI dengan dataset test kamu

# DROP timestamp
df = df.drop(columns=["timestamp"])


# ==========================================================
# 3. FEATURE & TARGET
# ==========================================================
features = [
    "temperature_C",
    "humidity_percent",
    "rain_rate_mmhr",
    "wind_speed_ms",
    "pressure_hPa",
    "hour_of_day",
    "day_of_week",
    "month"
]

target = "snr_dB"

X = df[features].values
y = df[target].values


# ==========================================================
# 4. LOAD SCALER (PENTING)
# ==========================================================
scaler = joblib.load("testtrain/scaler_140GHz.save")

X_scaled = scaler.transform(X)  # ❌ JANGAN FIT ULANG


# ==========================================================
# 5. SEQUENCE ENGINEERING
# ==========================================================
def create_sequences(X, y, window):
    X_seq, y_seq = [], []

    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])

    return np.array(X_seq), np.array(y_seq)


window_size = 20  # HARUS SAMA DENGAN TRAIN
X_seq, y_seq = create_sequences(X_scaled, y, window_size)


# ==========================================================
# 6. DATASET CLASS
# ==========================================================
class ChannelDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


test_loader = DataLoader(ChannelDataset(X_seq, y_seq), batch_size=64, shuffle=False)


# ==========================================================
# 7. MODEL CLASS (HARUS SAMA)
# ==========================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size, 64, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(64, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze()
    


# ==========================================================
# 8. LOAD MODEL
# ==========================================================
checkpoint = torch.load("testtrain/lstm_model_140GHz.pth")

model = LSTMModel(input_size=checkpoint["input_size"])
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

# ================================
# ZERO WEATHER FEATURES
# ================================
X_zero = X_seq.copy()

# nolkan weather (kolom 0–4)
X_zero[:, :, 0:5] = 0

X_tensor = torch.tensor(X_zero, dtype=torch.float32)

zero_pred = []

with torch.no_grad():
    for i in range(0, len(X_tensor), 64):
        batch = X_tensor[i:i+64]
        out = model(batch)
        zero_pred.extend(out.numpy())

zero_pred = np.array(zero_pred)

r2_zero = r2_score(y_seq, zero_pred)

print("\nZERO WEATHER TEST")
print("R2 no weather:", r2_zero)

# ================================
# SHUFFLE TIME TEST
# ================================
X_seq_shuffled = X_seq.copy()

for seq in X_seq_shuffled:
    np.random.shuffle(seq)  # acak urutan dalam window

X_tensor = torch.tensor(X_seq_shuffled, dtype=torch.float32)

shuffled_pred = []

with torch.no_grad():
    for i in range(0, len(X_tensor), 64):
        batch = X_tensor[i:i+64]
        out = model(batch)
        shuffled_pred.extend(out.numpy())

shuffled_pred = np.array(shuffled_pred)

r2_shuffled = r2_score(y_seq, shuffled_pred)

print("\nSHUFFLE TIME TEST")
print("R2 shuffled:", r2_shuffled)

# ==========================================================
# 9. PREDICTION
# ==========================================================
predictions = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        output = model(X_batch)
        predictions.extend(output.numpy())

predictions = np.array(predictions)


# ==========================================================
# 10. EVALUATION
# ==========================================================
mse = mean_squared_error(y_seq, predictions)
r2  = r2_score(y_seq, predictions)

print("\nTEST ON NEW DATASET")
print("MSE:", mse)
print("R2 :", r2)


# ==========================================================
# 11. PLOT
# ==========================================================
plt.figure(figsize=(12,5))

plt.plot(y_seq[:10000], label="True")
plt.plot(predictions[:10000], label="Pred")

plt.legend()
plt.title("Generalization Test")

plt.show()

# ================================
# NAIVE BASELINE
# ================================
naive_pred = y_seq[:-1]
true_naive = y_seq[1:]

r2_naive = r2_score(true_naive, naive_pred)

print("\nNAIVE BASELINE")
print("R2 Naive:", r2_naive)

print("\nCOMPARISON")
print("R2 LSTM :", r2)
print("R2 Naive:", r2_naive)