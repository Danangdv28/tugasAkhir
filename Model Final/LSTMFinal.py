# =============================================================================
# LSTMFinal.py  —  v3 (Deep Fix)
# Stacked LSTM (PyTorch) — Prediksi SNR Sinyal 6G (140 GHz - Urban)
#
# Perbaikan dari v2 (R²=0.77):
#   [1] LOOKBACK 30 → 60
#   [2] Fitur turunan: snr_rolling_mean_10, snr_rolling_std_10,
#                      snr_rolling_mean_30, distance_snr_ratio
#   [3] Weighted Huber Loss → sampel SNR rendah (outlier ekstrem) diberi bobot ×3
#   [4] Arsitektur: 4-layer LSTM dengan Residual Connection
#   [5] Optimizer: AdamW + CosineAnnealingWarmRestarts scheduler
#   [6] Epochs 150 → 200, Patience 20 → 25
# =============================================================================

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ─────────────────────────────────────────────
# 0. SEED & CONFIG
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOOKBACK    = 60        # ↑ dari 30 → 60
LAG_STEPS   = 5         # ↑ dari 3 → 5
BATCH_SIZE  = 64
EPOCHS      = 200       # ↑ dari 150 → 200
LR          = 0.0005
PATIENCE    = 25        # ↑ dari 20 → 25
WEIGHT_LOW_SNR = 3.0    # bobot ekstra untuk sampel SNR rendah (ekstrem)
SNR_LOW_THRESHOLD = 35.0  # dB — di bawah ini dianggap kondisi ekstrem

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

print("=" * 65)
print("  Stacked LSTM v3 (PyTorch) — Prediksi SNR Sinyal 6G 140 GHz")
print("=" * 65)
print(f"  Device   : {DEVICE}")
print(f"  Lookback : {LOOKBACK}  |  Lag Steps : {LAG_STEPS}")
print(f"  LR       : {LR}  |  Epochs : {EPOCHS}  |  Patience : {PATIENCE}")
print(f"  Weighted Loss: SNR < {SNR_LOW_THRESHOLD} dB → bobot ×{WEIGHT_LOW_SNR}\n")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(
    BASE_DIR, '..', 'simulator', 'hasil_simulasi',
    'single_140GHz_60days_urban.csv'
)

print(f"[1] Loading data:\n    {os.path.normpath(CSV_PATH)}\n")
df = pd.read_csv(CSV_PATH)
print(f"    Shape awal : {df.shape}\n")

# ─────────────────────────────────────────────
# 2. PREPROCESSING + FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("[2] Preprocessing & Feature Engineering ...\n")

FEATURE_COLS = [
    'time_min', 'day', 'distance_m',
    'path_loss_db', 'rsrp_dbm', 'noise_dbm',
    'blockage_loss_db', 'beamwidth_deg',
    'temperature_c', 'humidity_percent',
    'is_raining', 'rain_rate',
    'fog_visibility_m',
    'path_loss_anomaly', 'excess_loss_db', 'fading_gain',
    'channel_state'
]
TARGET_COL = 'snr_db'

missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Kolom tidak ditemukan: {missing}")

# -- Handle NaN
df['fog_visibility_m'] = pd.to_numeric(df['fog_visibility_m'], errors='coerce')
df['fog_visibility_m'].fillna(df['fog_visibility_m'].median(), inplace=True)

for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# -- Encode channel_state
le = LabelEncoder()
df['channel_state'] = le.fit_transform(df['channel_state'].astype(str))
print(f"    channel_state mapping : {dict(zip(le.classes_, le.transform(le.classes_)))}")

# -- Encode boolean
for col in ['is_raining', 'path_loss_anomaly']:
    df[col] = df[col].astype(str).str.lower().map({'true': 1, 'false': 0}).fillna(0).astype(int)

# -- Sort temporal
df = df.sort_values(['day', 'time_min']).reset_index(drop=True)

# ── [BARU v3] Rolling statistics SNR
#    Membantu model mengenali distribusi lokal sebelum prediksi
df['snr_rolling_mean_10'] = df[TARGET_COL].rolling(window=10, min_periods=1).mean()
df['snr_rolling_std_10']  = df[TARGET_COL].rolling(window=10, min_periods=1).std().fillna(0)
df['snr_rolling_mean_30'] = df[TARGET_COL].rolling(window=30, min_periods=1).mean()
df['snr_rolling_std_30']  = df[TARGET_COL].rolling(window=30, min_periods=1).std().fillna(0)

# ── [BARU v3] Fitur interaksi: distance vs path_loss
df['distance_pathloss_ratio'] = df['distance_m'] / (df['path_loss_db'] + 1e-6)

rolling_cols    = ['snr_rolling_mean_10', 'snr_rolling_std_10',
                   'snr_rolling_mean_30', 'snr_rolling_std_30',
                   'distance_pathloss_ratio']

# ── Lag SNR (t-1 ... t-LAG_STEPS)
lag_cols = []
for lag in range(1, LAG_STEPS + 1):
    col_name = f'snr_lag_{lag}'
    df[col_name] = df[TARGET_COL].shift(lag)
    lag_cols.append(col_name)

for col in lag_cols:
    df[col].fillna(df[col].iloc[LAG_STEPS], inplace=True)

FEATURE_COLS_FINAL = FEATURE_COLS + rolling_cols + lag_cols

print(f"    Rolling features      : {rolling_cols}")
print(f"    Lag features          : {lag_cols}")
print(f"    Total fitur           : {len(FEATURE_COLS_FINAL)}")
print(f"    NaN tersisa           : {df[FEATURE_COLS_FINAL + [TARGET_COL]].isnull().sum().sum()}")
print(f"    Shape bersih          : {df.shape}\n")

# -- X, y
X_raw = df[FEATURE_COLS_FINAL].values.astype(np.float32)
y_raw = df[[TARGET_COL]].values.astype(np.float32)

# ─────────────────────────────────────────────
# 3. SPLIT
# ─────────────────────────────────────────────
n       = len(X_raw)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)
n_test  = n - n_train - n_val

print(f"[3] Split data  (total={n})")
print(f"    Train : {n_train}  |  Val : {n_val}  |  Test : {n_test}\n")

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_X.fit(X_raw[:n_train])
scaler_y.fit(y_raw[:n_train])

X_scaled = scaler_X.transform(X_raw)
y_scaled = scaler_y.transform(y_raw)

# ─────────────────────────────────────────────
# 4. SLIDING WINDOW
# ─────────────────────────────────────────────
def create_sequences(X, y, y_orig, lookback):
    """Buat sekuens + simpan y_orig untuk weighted loss."""
    Xs, ys, y_origs = [], [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i : i + lookback])
        ys.append(y[i + lookback])
        y_origs.append(y_orig[i + lookback])
    return (np.array(Xs, dtype=np.float32),
            np.array(ys, dtype=np.float32),
            np.array(y_origs, dtype=np.float32))

X_seq, y_seq, y_seq_orig = create_sequences(X_scaled, y_scaled, y_raw, LOOKBACK)

n_train_seq = n_train - LOOKBACK
n_val_seq   = n_val

X_train  = X_seq[:n_train_seq]
y_train  = y_seq[:n_train_seq]
yo_train = y_seq_orig[:n_train_seq]   # SNR asli (dB) untuk weight

X_val    = X_seq[n_train_seq : n_train_seq + n_val_seq]
y_val    = y_seq[n_train_seq : n_train_seq + n_val_seq]
yo_val   = y_seq_orig[n_train_seq : n_train_seq + n_val_seq]

X_test   = X_seq[n_train_seq + n_val_seq :]
y_test   = y_seq[n_train_seq + n_val_seq :]

print(f"[4] Shape sekuens (LOOKBACK={LOOKBACK})")
print(f"    X_train : {X_train.shape}  |  X_val : {X_val.shape}  |  X_test : {X_test.shape}\n")

# ─────────────────────────────────────────────
# 5. DATASET (dengan sample weight)
# ─────────────────────────────────────────────
class SNRDataset(Dataset):
    def __init__(self, X, y, y_orig_db, low_thr=SNR_LOW_THRESHOLD, w_low=WEIGHT_LOW_SNR):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        # Buat weight: sampel dengan SNR asli < threshold dapat bobot lebih besar
        weights = np.where(y_orig_db.flatten() < low_thr, w_low, 1.0).astype(np.float32)
        self.w = torch.tensor(weights)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]

train_ds = SNRDataset(X_train, y_train, yo_train)
val_ds   = SNRDataset(X_val,   y_val,   yo_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Test loader tanpa weight
class SNRTestDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

test_loader = DataLoader(SNRTestDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────────
# 6. ARSITEKTUR: 4-Layer LSTM + Residual Connection
# ─────────────────────────────────────────────
class ResidualLSTMBlock(nn.Module):
    """
    LSTM block dengan residual connection.
    Jika in_size != hidden_size, pakai linear projection untuk residual.
    """
    def __init__(self, in_size, hidden_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_size, hidden_size, batch_first=True)
        self.bn   = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout(dropout)

        # Projection untuk residual jika dimensi berbeda
        self.proj = nn.Linear(in_size, hidden_size) if in_size != hidden_size else nn.Identity()

    def forward(self, x, return_seq=True):
        lstm_out, _ = self.lstm(x)              # (batch, seq, hidden)

        if return_seq:
            # Residual pada seluruh sekuens
            res = self.proj(x)                  # (batch, seq, hidden)
            out = lstm_out + res
            # BatchNorm1d: permute ke (batch, hidden, seq)
            out = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
            out = self.drop(out)
        else:
            # Ambil timestep terakhir untuk output akhir
            last = lstm_out[:, -1, :]           # (batch, hidden)
            res  = self.proj(x[:, -1, :])       # (batch, hidden)
            out  = last + res
            out  = self.bn(out)
            out  = self.drop(out)

        return out


class StackedLSTMv3(nn.Module):
    """
    4-layer Stacked LSTM dengan Residual Connection:
      Block1: LSTM(256) + residual  → full seq
      Block2: LSTM(256) + residual  → full seq
      Block3: LSTM(128) + residual  → full seq
      Block4: LSTM(64)  + residual  → last timestep

      FC: 64 → 128 → ReLU → Dropout → 64 → ReLU → 1
    """
    def __init__(self, input_size, dropout=0.3):
        super().__init__()

        self.block1 = ResidualLSTMBlock(input_size, 256, dropout)
        self.block2 = ResidualLSTMBlock(256,        256, dropout * 0.8)
        self.block3 = ResidualLSTMBlock(256,        128, dropout * 0.7)
        self.block4 = ResidualLSTMBlock(128,         64, dropout * 0.5)

        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out = self.block1(x,   return_seq=True)
        out = self.block2(out, return_seq=True)
        out = self.block3(out, return_seq=True)
        out = self.block4(out, return_seq=False)   # → (batch, 64)
        return self.fc(out)                        # → (batch, 1)


n_features = X_train.shape[2]
model = StackedLSTMv3(input_size=n_features, dropout=0.3).to(DEVICE)

print("[5] Arsitektur Model (v3 — Residual LSTM):")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n    Total trainable parameter : {total_params:,}\n")

# ─────────────────────────────────────────────
# 7. LOSS FUNCTION (Weighted Huber)
# ─────────────────────────────────────────────
def weighted_huber_loss(pred, target, weight, delta=1.0):
    """Huber loss dengan per-sample weight."""
    err  = target - pred
    abs_err = err.abs()
    huber = torch.where(
        abs_err <= delta,
        0.5 * err ** 2,
        delta * (abs_err - 0.5 * delta)
    )
    return (huber * weight.unsqueeze(1)).mean()

# ─────────────────────────────────────────────
# 8. OPTIMIZER & SCHEDULER
#    AdamW + CosineAnnealingWarmRestarts
# ─────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

# ─────────────────────────────────────────────
# 9. TRAINING LOOP
# ─────────────────────────────────────────────
print("[6] Training ...\n")

history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 'lr': []}
best_val_loss    = float('inf')
patience_counter = 0
best_weights     = None

SAVE_DIR        = BASE_DIR
BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'best_lstm_snr.pt')

for epoch in range(1, EPOCHS + 1):

    # ── Train
    model.train()
    train_loss, train_mae = 0.0, 0.0

    for X_batch, y_batch, w_batch in train_loader:
        X_batch  = X_batch.to(DEVICE)
        y_batch  = y_batch.to(DEVICE)
        w_batch  = w_batch.to(DEVICE)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss   = weighted_huber_loss(y_pred, y_batch, w_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * len(X_batch)
        train_mae  += torch.mean(torch.abs(y_pred.detach() - y_batch)).item() * len(X_batch)

    train_loss /= len(train_loader.dataset)
    train_mae  /= len(train_loader.dataset)

    # ── Validasi
    model.eval()
    val_loss, val_mae = 0.0, 0.0

    with torch.no_grad():
        for X_batch, y_batch, w_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            w_batch = w_batch.to(DEVICE)
            y_pred  = model(X_batch)
            loss    = weighted_huber_loss(y_pred, y_batch, w_batch)
            val_loss += loss.item() * len(X_batch)
            val_mae  += torch.mean(torch.abs(y_pred - y_batch)).item() * len(X_batch)

    val_loss /= len(val_loader.dataset)
    val_mae  /= len(val_loader.dataset)

    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(epoch)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_mae'].append(train_mae)
    history['val_mae'].append(val_mae)
    history['lr'].append(current_lr)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        best_weights     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_weights, BEST_MODEL_PATH)
    else:
        patience_counter += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch [{epoch:3d}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}  "
              f"MAE: {train_mae:.4f}/{val_mae:.4f}  "
              f"LR: {current_lr:.6f}")

    if patience_counter >= PATIENCE:
        print(f"\n  Early stopping di epoch {epoch} "
              f"(val_loss tidak membaik selama {PATIENCE} epoch)\n")
        break

model.load_state_dict(best_weights)
print(f"\n  Best Val Loss : {best_val_loss:.6f}\n")

# ─────────────────────────────────────────────
# 10. EVALUASI TEST SET
# ─────────────────────────────────────────────
print("[7] Evaluasi pada Test Set ...\n")

model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        y_pred  = model(X_batch).cpu().numpy()
        all_preds.append(y_pred)
        all_targets.append(y_batch.numpy())

y_pred_scaled = np.concatenate(all_preds,   axis=0)
y_test_scaled = np.concatenate(all_targets, axis=0)

y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)
y_test_orig = scaler_y.inverse_transform(y_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
mae  = mean_absolute_error(y_test_orig, y_pred_orig)
r2   = r2_score(y_test_orig, y_pred_orig)
mape = np.mean(np.abs((y_test_orig - y_pred_orig) / (y_test_orig + 1e-8))) * 100

print("=" * 45)
print(f"  RMSE : {rmse:.4f} dB")
print(f"  MAE  : {mae:.4f} dB")
print(f"  R²   : {r2:.4f}")
print(f"  MAPE : {mape:.2f} %")
print("=" * 45)

# ─────────────────────────────────────────────
# 11. VISUALISASI
# ─────────────────────────────────────────────
print("\n[8] Menyimpan plot ...\n")

fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    f'Stacked LSTM v3 (PyTorch) — Residual LSTM — SNR 6G 140 GHz Urban\n'
    f'LOOKBACK={LOOKBACK} | LAG={LAG_STEPS} | Weighted Huber | AdamW+Cosine | '
    f'RMSE={rmse:.3f} dB | R²={r2:.4f}',
    fontsize=11, fontweight='bold'
)

ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 1, 2)

# Loss
ax1.plot(history['train_loss'], label='Train', color='steelblue')
ax1.plot(history['val_loss'],   label='Val',   color='tomato')
ax1.set_title('Loss (Weighted Huber)')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.legend(); ax1.grid(True, alpha=0.3)

# MAE
ax2.plot(history['train_mae'], label='Train MAE', color='steelblue')
ax2.plot(history['val_mae'],   label='Val MAE',   color='tomato')
ax2.set_title('MAE Curve (scaled)')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('MAE')
ax2.legend(); ax2.grid(True, alpha=0.3)

# Scatter
# Warna berbeda untuk outlier (SNR aktual < threshold)
mask_low = y_test_orig.flatten() < SNR_LOW_THRESHOLD
ax3.scatter(y_test_orig[~mask_low], y_pred_orig[~mask_low],
            alpha=0.3, s=8,  color='steelblue', label='Normal')
ax3.scatter(y_test_orig[mask_low],  y_pred_orig[mask_low],
            alpha=0.7, s=20, color='red',       label=f'SNR<{SNR_LOW_THRESHOLD}dB (ekstrem)', zorder=5)
min_v = min(y_test_orig.min(), y_pred_orig.min())
max_v = max(y_test_orig.max(), y_pred_orig.max())
ax3.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=1.5, label='Ideal')
ax3.set_title(f'Scatter  (R² = {r2:.4f})')
ax3.set_xlabel('Actual SNR (dB)'); ax3.set_ylabel('Predicted SNR (dB)')
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

# Actual vs Predicted
n_show = min(600, len(y_test_orig))
ax4.plot(y_test_orig[:n_show], label='Actual SNR',    color='steelblue', alpha=0.85, linewidth=0.8)
ax4.plot(y_pred_orig[:n_show], label='Predicted SNR', color='tomato',    alpha=0.85, linewidth=0.8, linestyle='--')
ax4.fill_between(
    range(n_show),
    y_test_orig[:n_show].flatten(),
    y_pred_orig[:n_show].flatten(),
    alpha=0.15, color='orange', label='Error gap'
)
ax4.set_title(f'Actual vs Predicted SNR (pertama {n_show} sampel test)')
ax4.set_xlabel('Sampel'); ax4.set_ylabel('SNR (dB)')
ax4.legend(loc='upper right'); ax4.grid(True, alpha=0.3)

plt.tight_layout()
PLOT_PATH = os.path.join(SAVE_DIR, 'lstm_snr_results.png')
plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
print(f"    Plot disimpan : {PLOT_PATH}")
plt.show()

# ─────────────────────────────────────────────
# 12. SIMPAN MODEL & HASIL
# ─────────────────────────────────────────────
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, 'LSTMFinal_model.pt')
torch.save({
    'model_state_dict' : model.state_dict(),
    'scaler_X'         : scaler_X,
    'scaler_y'         : scaler_y,
    'feature_cols'     : FEATURE_COLS_FINAL,
    'lookback'         : LOOKBACK,
    'lag_steps'        : LAG_STEPS,
    'metrics'          : {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}
}, FINAL_MODEL_PATH)
print(f"\n[9] Model disimpan  : {FINAL_MODEL_PATH}")

result_df = pd.DataFrame({
    'actual_snr_db':    y_test_orig.flatten(),
    'predicted_snr_db': y_pred_orig.flatten(),
    'error_db':         (y_test_orig - y_pred_orig).flatten()
})
RESULT_PATH = os.path.join(SAVE_DIR, 'lstm_prediction_results.csv')
result_df.to_csv(RESULT_PATH, index=False)
print(f"    Hasil prediksi  : {RESULT_PATH}")

print("\n✅ Selesai! (v3 — Residual LSTM + Weighted Loss)\n")