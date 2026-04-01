# ==========================================================
# 1. IMPORT
# ==========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib


# ==========================================================
# 2. LOAD DATASET
# ==========================================================
df = pd.read_csv("urban_140GHz.csv")  # ganti 220 jika mau

# ==========================================================
# 3. PREPROCESSING
# ==========================================================

# DROP timestamp
df = df.drop(columns=["timestamp"])

# FEATURE & TARGET
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
# 4. SCALING
# ==========================================================

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)


# ==========================================================
# 5. SEQUENCE ENGINEERING
# ==========================================================

def create_sequences(X, y, window):
    X_seq, y_seq = [], []

    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])

    return np.array(X_seq), np.array(y_seq)


window_size = 20
X_seq, y_seq = create_sequences(X_scaled, y, window_size)


# ==========================================================
# 6. SPLIT (TIME SERIES)
# ==========================================================

train_split = int(len(X_seq) * 0.7)
val_split   = int(len(X_seq) * 0.85)

X_train = X_seq[:train_split]
y_train = y_seq[:train_split]

X_val = X_seq[train_split:val_split]
y_val = y_seq[train_split:val_split]

X_test = X_seq[val_split:]
y_test = y_seq[val_split:]


# ==========================================================
# 7. DATASET CLASS
# ==========================================================

class ChannelDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(ChannelDataset(X_train, y_train), batch_size=64, shuffle=False)
val_loader   = DataLoader(ChannelDataset(X_val, y_val), batch_size=64, shuffle=False)
test_loader  = DataLoader(ChannelDataset(X_test, y_test), batch_size=64, shuffle=False)

# Scaler

joblib.dump(scaler, "scaler_140GHz.save")

print("Scaler saved!")

# ==========================================================
# 8. MODEL
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


model = LSTMModel(X_train.shape[2])


# ==========================================================
# 9. TRAIN SETUP
# ==========================================================

criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50


# ==========================================================
# 10. TRAINING LOOP
# ==========================================================

train_losses = []
val_losses = []

for epoch in range(epochs):

    model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")


# ==========================================================
# 11. TEST
# ==========================================================

model.eval()
predictions = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        predictions.extend(model(X_batch).numpy())

predictions = np.array(predictions)

mse = mean_squared_error(y_test, predictions)
r2  = r2_score(y_test, predictions)

print("\nTEST PERFORMANCE")
print("MSE:", mse)
print("R2 :", r2)


# ==========================================================
# 12. PLOT
# ==========================================================

plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.legend()
plt.title("Training Curve")
plt.show()


plt.figure(figsize=(12,5))
plt.plot(y_test[:500], label="True")
plt.plot(predictions[:500], label="Pred")
plt.legend()
plt.title("Prediction")
plt.show()

# ==========================================================
# SAVE MODEL
# ==========================================================

torch.save({
    "model_state_dict": model.state_dict(),
    "input_size": X_train.shape[2],
    "window_size": window_size
}, "lstm_model_140GHz.pth")

print("Model saved!")

