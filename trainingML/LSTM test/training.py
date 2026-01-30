# train_lstm_regression.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from Dataset import SNRSequenceDataset
from model import LSTMRegressor

# =========================
# CONFIG
# =========================
CSV_FILE = "single_140GHz_14days_urban_CCS_FULL.csv"
FEATURES = [
    "snr_db",
    "rain_rate",
    "humidity_percent",
    "temperature_c",
    "fog_visibility_m",
    "path_loss_db"
]

LOOKBACK = 10
HORIZONS = [1, 3, 5]
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3

# =========================
# DATA
# =========================
dataset = SNRSequenceDataset(
    CSV_FILE,
    FEATURES,
    lookback=LOOKBACK,
    horizons=HORIZONS
)

N = len(dataset)
train_end = int(0.7 * N)
val_end = int(0.85 * N)

train_set = torch.utils.data.Subset(dataset, range(0, train_end))
val_set   = torch.utils.data.Subset(dataset, range(train_end, val_end))
test_set  = torch.utils.data.Subset(dataset, range(val_end, N))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMRegressor(num_features=len(FEATURES)).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAIN
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1:02d} | Train MSE: {total_loss/len(train_loader):.4f}")

# =========================
# EVALUATE
# =========================
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        pred = model(X).cpu().numpy()
        y_true.append(y.numpy())
        y_pred.append(pred)

y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

print("\n=== LSTM REGRESSION RESULTS ===")
for i, h in enumerate(HORIZONS):
    rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
    r2 = r2_score(y_true[:, i], y_pred[:, i])
    print(f"SNR(t+{h}) → RMSE: {rmse:.3f} dB | R²: {r2:.4f}")