import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Using device:", device)

# ======================
# 1. PATH CONFIGURATION
# ======================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
CSV_FILE = os.path.join(ROOT_DIR, "simulator", "hasil_simulasi", "single_140GHz_60days_urban.csv")

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Dataset tidak ditemukan di: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)

# ======================
# 2. CLEAN & ENCODE DATA
# ======================

# Drop timestamp if exists
if 'timestamp' in df.columns:
    df = df.drop(columns=['timestamp'])

# One-hot encode categorical column
if 'channel_state' in df.columns:
    df = pd.get_dummies(df, columns=['channel_state'], drop_first=True)

# Ensure all data numeric
for col in df.columns:
    if df[col].dtype == 'object':
        raise ValueError(f"Kolom {col} masih bertipe object. Periksa kembali.")

# ======================
# 3. DEFINE TARGET & FEATURES
# ======================

target_col = "snr_db"
feature_cols = [col for col in df.columns if col != target_col]

X_raw = df[feature_cols].values
y_raw = df[target_col].values.reshape(-1, 1)

# ======================
# 4. TIME-BASED SPLIT 70 / 15 / 15
# ======================

n_total = len(X_raw)

train_end = int(n_total * 0.70)
val_end   = int(n_total * 0.85)

X_train_raw = X_raw[:train_end]
X_val_raw   = X_raw[train_end:val_end]
X_test_raw  = X_raw[val_end:]

y_train_raw = y_raw[:train_end]
y_val_raw   = y_raw[train_end:val_end]
y_test_raw  = y_raw[val_end:]

# ======================
# 5. SCALING (FIT ONLY TRAIN)
# ======================

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_val_scaled   = scaler_X.transform(X_val_raw)
X_test_scaled  = scaler_X.transform(X_test_raw)

y_train_scaled = scaler_y.fit_transform(y_train_raw)
y_val_scaled   = scaler_y.transform(y_val_raw)
y_test_scaled  = scaler_y.transform(y_test_raw)

# ======================
# 6. CREATE SEQUENCES
# ======================

lookback = 240
horizon  = 240

def create_sequences(X, y, lookback, horizon):
    Xs, ys = [], []
    for i in range(len(X) - lookback - horizon):
        Xs.append(X[i:i+lookback])
        ys.append(y[i+lookback:i+lookback+horizon])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(
    X_train_scaled, y_train_scaled, lookback, horizon
)

X_val_seq, y_val_seq = create_sequences(
    X_val_scaled, y_val_scaled, lookback, horizon
)

X_test_seq, y_test_seq = create_sequences(
    X_test_scaled, y_test_scaled, lookback, horizon
)

print("Train shape:", X_train_seq.shape)
print("Val shape  :", X_val_seq.shape)
print("Test shape :", X_test_seq.shape)

# Convert to tensors
X_train = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train_seq, dtype=torch.float32).squeeze(-1).to(device)

X_val = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val_seq, dtype=torch.float32).squeeze(-1).to(device)

X_test = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test_seq, dtype=torch.float32).squeeze(-1).to(device)


# ======================
# 7. BUILD MODEL
# ======================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, horizon=240):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

model = LSTMModel(input_size=X_train.shape[2], horizon=horizon).to(device)

criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

# ======================
# 8. TRAINING LOOP
# ======================

epochs = 80
batch_size = 64
patience = 10
best_val_loss = np.inf
patience_counter = 0

for epoch in range(epochs):

    model.train()
    train_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        batch_x = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= (X_train.size(0) / batch_size)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val).item()

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_lstm_140GHz.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

model.load_state_dict(torch.load("best_lstm_140GHz.pt"))

# ======================
# 9. PREDICTION
# ======================

model.eval()
with torch.no_grad():
    y_pred = model(X_test).cpu().numpy()

y_test_np = y_test.cpu().numpy()

# Inverse scaling
y_pred_real = scaler_y.inverse_transform(
    y_pred.reshape(-1,1)
).reshape(y_pred.shape)

y_test_real = scaler_y.inverse_transform(
    y_test_np.reshape(-1,1)
).reshape(y_test_np.shape)

# ======================
# 10. GLOBAL METRICS
# ======================

y_pred_flat = y_pred_real.flatten()
y_test_flat = y_test_real.flatten()

mse = mean_squared_error(y_test_flat, y_pred_flat)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_flat, y_pred_flat)

print("\n===== GLOBAL METRICS =====")
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2)

print("\nSNR Statistics:")
print("Mean SNR:", np.mean(y_test_flat))
print("Std SNR :", np.std(y_test_flat))

# ======================
# 11. RMSE PER HORIZON
# ======================

horizon_rmse = []

for h in range(horizon):
    rmse_h = np.sqrt(mean_squared_error(
        y_test_real[:, h],
        y_pred_real[:, h]
    ))
    horizon_rmse.append(rmse_h)

plt.figure(figsize=(10,4))
plt.plot(horizon_rmse)
plt.title("RMSE per Forecast Horizon (140 GHz)")
plt.xlabel("Minute Ahead")
plt.ylabel("RMSE")
plt.show()

# ======================
# 12. SAMPLE PLOT
# ======================

plt.figure(figsize=(12,5))
plt.plot(y_test_real[0], label="True")
plt.plot(y_pred_real[0], label="Predicted")
plt.legend()
plt.title("4-Hour Ahead Prediction Sample (140 GHz)")
plt.show()
