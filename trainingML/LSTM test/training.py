# training.py
import torch
import numpy as np
import os
import random
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Dataset import SNRSequenceDataset
from model import LSTMRegressor
import matplotlib.pyplot as plt

# =========================
# REPRODUCIBILITY
# =========================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =========================
# CONFIG
# =========================
BASE_DIR = "hasil_simulasi"
CSV_FILE = os.path.join(BASE_DIR, "single_220GHz_30days_urban_CCS_FULL.csv")

FEATURES = [
    "rain_rate",
    "humidity_percent",
    "temperature_c",
    "fog_visibility_m",
    "path_loss_db",
    "ccs_1",
    "ccs_2"
]

LOOKBACK = 5
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
PATIENCE = 7   # early stopping patience

# =========================
# DATASET
# =========================
dataset = SNRSequenceDataset(CSV_FILE, FEATURES, LOOKBACK)

N = len(dataset)
train_end = int(0.7 * N)
val_end   = int(0.85 * N)

train_set = Subset(dataset, range(0, train_end))
val_set   = Subset(dataset, range(train_end, val_end))
test_set  = Subset(dataset, range(val_end, N))

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

train_rmse_history = []
val_rmse_history = []

best_val_rmse = np.inf
best_state = None
patience_counter = 0

# =========================
# TRAIN (WITH EARLY STOPPING)
# =========================
for epoch in range(EPOCHS):
    # ===== TRAIN =====
    model.train()
    train_loss = 0.0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_rmse = np.sqrt(train_loss / len(train_loader))
    train_rmse_history.append(train_rmse)

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += criterion(pred, y).item()

    val_rmse = np.sqrt(val_loss / len(val_loader))
    val_rmse_history.append(val_rmse)

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train RMSE: {train_rmse:.4f} | "
        f"Val RMSE: {val_rmse:.4f}"
    )

    # ===== EARLY STOPPING =====
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹ Early stopping at epoch {epoch+1}")
            break

# =========================
# LOAD BEST MODEL
# =========================
model.load_state_dict(best_state)
model.eval()

# =========================
# TEST EVALUATION (FINAL)
# =========================
y_true, y_pred = [], []

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        pred = model(X).cpu().numpy().ravel()

        y_true.append(y.numpy().ravel())
        y_pred.append(pred)

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

# =========================
# PLOT TRAINING CURVE
# =========================
epochs = range(1, len(train_rmse_history) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_rmse_history, label="Train RMSE", linewidth=2)
plt.plot(epochs, val_rmse_history, label="Validation RMSE", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("RMSE (dB)")
plt.title("LSTM Training vs Validation RMSE")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lstm_training_curve.png", dpi=300)
plt.show()

# =========================
# RESULTS
# =========================
print("\n=== LSTM RESULTS (TEST SET) ===")
print(f"RMSE : {rmse:.4f} dB")
print(f"MAE  : {mae:.4f} dB")
print(f"R²   : {r2:.4f}")
print(f"LSTM,TEST,{rmse:.4f},{mae:.4f},{r2:.4f}")
