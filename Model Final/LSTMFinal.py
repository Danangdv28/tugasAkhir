# =============================================================================
# LSTMFinal.py  —  v4.1 (Sidang-Safe)
# Stacked LSTM (PyTorch) — Prediksi SNR Sinyal 6G (140 GHz - Urban)
#
# Strategi v4 (tetap):
#   [1] SNR clipping: nilai < 10 dB di-clip ke 10 dB saat training
#   [2] LOOKBACK = 30
#   [3] Klasifikasi channel quality (Good/Fair/Poor/Critical)
#   [4] Arsitektur: 3-layer Residual LSTM (256→128→64)
#   [5] Loss: MSE murni
#   [6] Evaluasi: clipped / all / normal-only
#
# Perbaikan v4.1 (3 fix wajib untuk sidang):
#   [FIX 1] shuffle=False pada train_loader  ← metodologis time-series
#   [FIX 2] Persistence Baseline             ← buktikan LSTM lebih baik
#   [FIX 3] Classification Report lengkap    ← precision/recall/F1 per kelas
#            + Confusion Matrix heatmap
# =============================================================================

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ─────────────────────────────────────────────
# 0. SEED & CONFIG
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOOKBACK   = 30
LAG_STEPS  = 5
BATCH_SIZE = 64
EPOCHS     = 150
LR         = 0.0005
PATIENCE   = 20

SNR_CLIP_MIN = 10.0

CQ_THRESHOLDS = {'Good': 50, 'Fair': 35, 'Poor': 20}
CQ_COLORS     = {'Good': 'steelblue', 'Fair': 'gold',
                 'Poor': 'orange',    'Critical': 'red'}
CQ_ORDER      = ['Good', 'Fair', 'Poor', 'Critical']

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

print("=" * 65)
print("  Stacked LSTM v4.1 — Prediksi SNR + Channel Quality 6G 140GHz")
print("=" * 65)
print(f"  Device      : {DEVICE}")
print(f"  Lookback    : {LOOKBACK}  |  Lag : {LAG_STEPS}")
print(f"  SNR Clip    : nilai < {SNR_CLIP_MIN} dB → di-clip ke {SNR_CLIP_MIN} dB")
print(f"  LR          : {LR}  |  Epochs : {EPOCHS}  |  Patience : {PATIENCE}")
print(f"  [FIX 1] shuffle=False  [FIX 2] Persistence Baseline  [FIX 3] Full CQ Report\n")

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
TARGET_COL     = 'snr_db'
TARGET_CLIPPED = 'snr_db_clipped'

missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Kolom tidak ditemukan: {missing}")

# -- Handle NaN
df['fog_visibility_m'] = pd.to_numeric(df['fog_visibility_m'], errors='coerce')
df['fog_visibility_m'].fillna(df['fog_visibility_m'].median(), inplace=True)
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# -- Encode
le = LabelEncoder()
df['channel_state'] = le.fit_transform(df['channel_state'].astype(str))
print(f"    channel_state  : {dict(zip(le.classes_, le.transform(le.classes_)))}")

for col in ['is_raining', 'path_loss_anomaly']:
    df[col] = df[col].astype(str).str.lower().map(
        {'true': 1, 'false': 0}).fillna(0).astype(int)

# -- Sort temporal
df = df.sort_values(['day', 'time_min']).reset_index(drop=True)

# ── SNR CLIPPING
df[TARGET_CLIPPED] = df[TARGET_COL].clip(lower=SNR_CLIP_MIN)
n_clipped = (df[TARGET_COL] < SNR_CLIP_MIN).sum()
print(f"    SNR clipping   : {n_clipped} sampel ({n_clipped/len(df)*100:.2f}%) "
      f"di-clip dari < {SNR_CLIP_MIN} dB → {SNR_CLIP_MIN} dB")

# ── Rolling statistics
df['snr_rolling_mean_10'] = df[TARGET_CLIPPED].rolling(10, min_periods=1).mean()
df['snr_rolling_std_10']  = df[TARGET_CLIPPED].rolling(10, min_periods=1).std().fillna(0)
df['snr_rolling_mean_30'] = df[TARGET_CLIPPED].rolling(30, min_periods=1).mean()
df['snr_rolling_std_30']  = df[TARGET_CLIPPED].rolling(30, min_periods=1).std().fillna(0)
df['distance_pathloss_ratio'] = df['distance_m'] / (df['path_loss_db'] + 1e-6)

rolling_cols = ['snr_rolling_mean_10', 'snr_rolling_std_10',
                'snr_rolling_mean_30', 'snr_rolling_std_30',
                'distance_pathloss_ratio']

# ── Lag SNR
lag_cols = []
for lag in range(1, LAG_STEPS + 1):
    col = f'snr_lag_{lag}'
    df[col] = df[TARGET_CLIPPED].shift(lag).fillna(method='bfill')
    lag_cols.append(col)

FEATURE_COLS_FINAL = FEATURE_COLS + rolling_cols + lag_cols

print(f"    Total fitur    : {len(FEATURE_COLS_FINAL)}")
print(f"    NaN tersisa    : {df[FEATURE_COLS_FINAL].isnull().sum().sum()}\n")

# ── Channel Quality label (dari SNR ASLI)
def get_channel_quality(snr):
    if snr >= CQ_THRESHOLDS['Good']:   return 'Good'
    elif snr >= CQ_THRESHOLDS['Fair']: return 'Fair'
    elif snr >= CQ_THRESHOLDS['Poor']: return 'Poor'
    else:                               return 'Critical'

df['channel_quality'] = df[TARGET_COL].apply(get_channel_quality)
print("    Distribusi Channel Quality (SNR asli):")
cq_counts = df['channel_quality'].value_counts()
for cq, cnt in cq_counts.items():
    print(f"      {cq:<10}: {cnt:>7,} sampel ({cnt/len(df)*100:.2f}%)")
print()

X_raw    = df[FEATURE_COLS_FINAL].values.astype(np.float32)
y_raw    = df[[TARGET_CLIPPED]].values.astype(np.float32)
y_orig   = df[[TARGET_COL]].values.astype(np.float32)
cq_label = df['channel_quality'].values

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
def create_sequences(X, y, y_orig_arr, cq, lookback):
    Xs, ys, y_origs, cqs = [], [], [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i : i + lookback])
        ys.append(y[i + lookback])
        y_origs.append(y_orig_arr[i + lookback])
        cqs.append(cq[i + lookback])
    return (np.array(Xs, dtype=np.float32),
            np.array(ys, dtype=np.float32),
            np.array(y_origs, dtype=np.float32),
            np.array(cqs))

X_seq, y_seq, y_seq_orig, cq_seq = create_sequences(
    X_scaled, y_scaled, y_orig, cq_label, LOOKBACK
)

n_train_seq = n_train - LOOKBACK
n_val_seq   = n_val

X_train, y_train = X_seq[:n_train_seq],                          y_seq[:n_train_seq]
X_val,   y_val   = X_seq[n_train_seq:n_train_seq+n_val_seq],     y_seq[n_train_seq:n_train_seq+n_val_seq]
X_test,  y_test  = X_seq[n_train_seq+n_val_seq:],                y_seq[n_train_seq+n_val_seq:]
y_test_orig      = y_seq_orig[n_train_seq+n_val_seq:]
cq_test          = cq_seq[n_train_seq+n_val_seq:]

# Simpan juga y_orig seluruh test untuk persistence baseline
y_orig_test_full = y_orig[n_train + n_val:]   # belum di-sequence, untuk persistence

print(f"[4] Shape sekuens (LOOKBACK={LOOKBACK})")
print(f"    X_train : {X_train.shape}  X_val : {X_val.shape}  X_test : {X_test.shape}\n")

# ─────────────────────────────────────────────
# 5. DATASET & DATALOADER
# ─────────────────────────────────────────────
class SNRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ── [FIX 1] shuffle=False — konsisten dengan prinsip time-series
train_loader = DataLoader(SNRDataset(X_train, y_train), BATCH_SIZE, shuffle=False, num_workers=0)
val_loader   = DataLoader(SNRDataset(X_val,   y_val),   BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(SNRDataset(X_test,  y_test),  BATCH_SIZE, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────
# 6. ARSITEKTUR: 3-Layer Residual LSTM
# ─────────────────────────────────────────────
class ResidualLSTMBlock(nn.Module):
    def __init__(self, in_size, hidden_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_size, hidden_size, batch_first=True)
        self.bn   = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(in_size, hidden_size) if in_size != hidden_size else nn.Identity()

    def forward(self, x, return_seq=True):
        out, _ = self.lstm(x)
        if return_seq:
            out = out + self.proj(x)
            out = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
            out = self.drop(out)
        else:
            out = out[:, -1, :] + self.proj(x[:, -1, :])
            out = self.bn(out)
            out = self.drop(out)
        return out


class StackedLSTMv4(nn.Module):
    """
    3-layer Residual LSTM:
      Block1 : LSTM(256) → full seq
      Block2 : LSTM(128) → full seq
      Block3 : LSTM(64)  → last timestep
      FC     : 64 → 128 → 64 → 1
    """
    def __init__(self, input_size, dropout=0.25):
        super().__init__()
        self.b1 = ResidualLSTMBlock(input_size, 256, dropout)
        self.b2 = ResidualLSTMBlock(256,        128, dropout * 0.8)
        self.b3 = ResidualLSTMBlock(128,         64, dropout * 0.6)
        self.fc = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.b1(x, return_seq=True)
        x = self.b2(x, return_seq=True)
        x = self.b3(x, return_seq=False)
        return self.fc(x)


n_features = X_train.shape[2]
model = StackedLSTMv4(input_size=n_features, dropout=0.25).to(DEVICE)

print("[5] Arsitektur Model (v4.1 — 3-layer Residual LSTM):")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n    Total parameter : {total_params:,}\n")

# ─────────────────────────────────────────────
# 7. LOSS, OPTIMIZER, SCHEDULER
# ─────────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

# ─────────────────────────────────────────────
# 8. TRAINING LOOP
# ─────────────────────────────────────────────
print("[6] Training ...\n")

history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
best_val_loss    = float('inf')
patience_counter = 0
best_weights     = None
BEST_MODEL_PATH  = os.path.join(BASE_DIR, 'best_lstm_snr.pt')

for epoch in range(1, EPOCHS + 1):

    model.train()
    t_loss, t_mae = 0.0, 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss += loss.item() * len(Xb)
        t_mae  += torch.mean(torch.abs(pred.detach() - yb)).item() * len(Xb)
    t_loss /= len(train_loader.dataset)
    t_mae  /= len(train_loader.dataset)

    model.eval()
    v_loss, v_mae = 0.0, 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            pred   = model(Xb)
            loss   = criterion(pred, yb)
            v_loss += loss.item() * len(Xb)
            v_mae  += torch.mean(torch.abs(pred - yb)).item() * len(Xb)
    v_loss /= len(val_loader.dataset)
    v_mae  /= len(val_loader.dataset)

    scheduler.step(v_loss)

    history['train_loss'].append(t_loss)
    history['val_loss'].append(v_loss)
    history['train_mae'].append(t_mae)
    history['val_mae'].append(v_mae)

    if v_loss < best_val_loss:
        best_val_loss    = v_loss
        patience_counter = 0
        best_weights     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_weights, BEST_MODEL_PATH)
    else:
        patience_counter += 1

    if epoch % 10 == 0 or epoch == 1:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch [{epoch:3d}/{EPOCHS}]  "
              f"Loss: {t_loss:.6f}/{v_loss:.6f}  "
              f"MAE: {t_mae:.4f}/{v_mae:.4f}  LR: {lr_now:.6f}")

    if patience_counter >= PATIENCE:
        print(f"\n  Early stopping epoch {epoch} (patience={PATIENCE})\n")
        break

model.load_state_dict(best_weights)
print(f"\n  Best Val Loss : {best_val_loss:.6f}\n")

# ─────────────────────────────────────────────
# 9. EVALUASI TEST SET
# ─────────────────────────────────────────────
print("[7] Evaluasi pada Test Set ...\n")

model.eval()
preds, targets = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        preds.append(model(Xb.to(DEVICE)).cpu().numpy())
        targets.append(yb.numpy())

y_pred_scaled = np.concatenate(preds,   axis=0)
y_test_scaled = np.concatenate(targets, axis=0)

y_pred_clipped = scaler_y.inverse_transform(y_pred_scaled)
y_test_clipped = scaler_y.inverse_transform(y_test_scaled)

# Evaluasi 1: vs CLIPPED
rmse_c = np.sqrt(mean_squared_error(y_test_clipped, y_pred_clipped))
mae_c  = mean_absolute_error(y_test_clipped, y_pred_clipped)
r2_c   = r2_score(y_test_clipped, y_pred_clipped)

# Evaluasi 2: vs ASLI (semua)
rmse_o = np.sqrt(mean_squared_error(y_test_orig, y_pred_clipped))
mae_o  = mean_absolute_error(y_test_orig, y_pred_clipped)
r2_o   = r2_score(y_test_orig, y_pred_clipped)

# Evaluasi 3: vs ASLI (normal only, SNR ≥ 10 dB)
mask_normal = y_test_orig.flatten() >= SNR_CLIP_MIN
rmse_n = np.sqrt(mean_squared_error(y_test_orig[mask_normal], y_pred_clipped[mask_normal]))
mae_n  = mean_absolute_error(y_test_orig[mask_normal], y_pred_clipped[mask_normal])
r2_n   = r2_score(y_test_orig[mask_normal], y_pred_clipped[mask_normal])
mape_n = np.mean(np.abs((y_test_orig[mask_normal] - y_pred_clipped[mask_normal])
                         / (y_test_orig[mask_normal] + 1e-8))) * 100

print("=" * 60)
print("  EVALUASI vs SNR CLIPPED (training target):")
print(f"    RMSE : {rmse_c:.4f} dB  |  MAE : {mae_c:.4f} dB  |  R² : {r2_c:.4f}")
print()
print("  EVALUASI vs SNR ASLI (semua, termasuk ekstrem):")
print(f"    RMSE : {rmse_o:.4f} dB  |  MAE : {mae_o:.4f} dB  |  R² : {r2_o:.4f}")
print()
print(f"  EVALUASI vs SNR ASLI (hanya SNR ≥ {SNR_CLIP_MIN} dB):")
print(f"    RMSE : {rmse_n:.4f} dB  |  MAE : {mae_n:.4f} dB")
print(f"    R²   : {r2_n:.4f}       |  MAPE: {mape_n:.2f} %")
print("=" * 60)

# ── Channel Quality classify helper
def classify_cq(snr_arr):
    out = []
    for v in snr_arr.flatten():
        if v >= CQ_THRESHOLDS['Good']:   out.append('Good')
        elif v >= CQ_THRESHOLDS['Fair']: out.append('Fair')
        elif v >= CQ_THRESHOLDS['Poor']: out.append('Poor')
        else:                             out.append('Critical')
    return np.array(out)

cq_actual = cq_test
cq_pred   = classify_cq(y_pred_clipped)

# ── Critical detection (tetap dari v4)
mask_crit_actual = cq_actual == 'Critical'
mask_crit_pred   = cq_pred   == 'Critical'
if mask_crit_actual.sum() > 0:
    recall_crit    = (mask_crit_actual & mask_crit_pred).sum() / mask_crit_actual.sum()
    precision_crit = (mask_crit_actual & mask_crit_pred).sum() / (mask_crit_pred.sum() + 1e-8)
    print(f"\n  Channel 'Critical' Detection:")
    print(f"    Actual Critical : {mask_crit_actual.sum()} sampel")
    print(f"    Pred Critical   : {mask_crit_pred.sum()} sampel")
    print(f"    Recall          : {recall_crit:.3f}")
    print(f"    Precision       : {precision_crit:.3f}")

overall_acc = (cq_actual == cq_pred).mean()
print(f"\n  Channel Quality Accuracy (4 kelas): {overall_acc*100:.2f}%\n")

# ─────────────────────────────────────────────
# [FIX 2] PERSISTENCE BASELINE
# ─────────────────────────────────────────────
print("=" * 60)
print("  [FIX 2] PERSISTENCE BASELINE COMPARISON")
print("=" * 60)

# Persistence: y_hat(t) = y(t-1)
# Sejajarkan: buang elemen pertama aktual, buang elemen terakhir prediksi
y_actual_t    = y_test_orig[1:].flatten()
y_persist     = y_test_orig[:-1].flatten()
y_lstm_align  = y_pred_clipped[1:].flatten()

# Normal only mask (disejajarkan)
mask_n_align  = y_actual_t >= SNR_CLIP_MIN

# Persistence metrics — normal only
rmse_p = np.sqrt(mean_squared_error(y_actual_t[mask_n_align], y_persist[mask_n_align]))
mae_p  = mean_absolute_error(y_actual_t[mask_n_align], y_persist[mask_n_align])
r2_p   = r2_score(y_actual_t[mask_n_align], y_persist[mask_n_align])

# LSTM metrics — normal only (aligned, konsisten dengan persistence)
rmse_l = np.sqrt(mean_squared_error(y_actual_t[mask_n_align], y_lstm_align[mask_n_align]))
mae_l  = mean_absolute_error(y_actual_t[mask_n_align], y_lstm_align[mask_n_align])
r2_l   = r2_score(y_actual_t[mask_n_align], y_lstm_align[mask_n_align])

rmse_improve = (rmse_p - rmse_l) / rmse_p * 100
mae_improve  = (mae_p  - mae_l)  / mae_p  * 100

print(f"\n  {'Metrik':<12} {'Persistence':>14} {'LSTM v4.1':>14} {'Improve':>10}")
print(f"  {'-'*54}")
print(f"  {'RMSE (dB)':<12} {rmse_p:>14.4f} {rmse_l:>14.4f} {rmse_improve:>9.2f}%")
print(f"  {'MAE  (dB)':<12} {mae_p:>14.4f} {mae_l:>14.4f} {mae_improve:>9.2f}%")
print(f"  {'R²':<12} {r2_p:>14.4f} {r2_l:>14.4f} {'—':>10}")
print(f"\n  * Evaluasi pada SNR ≥ {SNR_CLIP_MIN} dB")

if rmse_improve > 20:
    verdict = "✅ LSTM secara signifikan lebih baik dari baseline naive."
elif rmse_improve > 5:
    verdict = "✅ LSTM lebih baik dari baseline — improvement moderat."
else:
    verdict = "⚠️  LSTM hanya sedikit lebih baik. Justifikasi arsitektur perlu diperkuat."
print(f"\n  Verdict: {verdict}")
print("=" * 60)

# ─────────────────────────────────────────────
# [FIX 3] CLASSIFICATION REPORT LENGKAP
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [FIX 3] CHANNEL QUALITY CLASSIFICATION REPORT (4 Kelas)")
print("=" * 60)

report_str = classification_report(
    cq_actual, cq_pred,
    labels=CQ_ORDER,
    target_names=CQ_ORDER,
    digits=4,
    zero_division=0
)
print(report_str)

# Per-class detail
prec, rec, f1, sup = precision_recall_fscore_support(
    cq_actual, cq_pred, labels=CQ_ORDER, zero_division=0
)
print(f"  {'Kelas':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print(f"  {'-'*52}")
for i, cq in enumerate(CQ_ORDER):
    print(f"  {cq:<12} {prec[i]:>10.4f} {rec[i]:>10.4f} {f1[i]:>10.4f} {sup[i]:>10}")
print("=" * 60)

# ─────────────────────────────────────────────
# 10. VISUALISASI
# ─────────────────────────────────────────────
print("\n[8] Menyimpan plot ...\n")

# ── Plot utama (sama persis dengan v4)
fig = plt.figure(figsize=(20, 14))
fig.suptitle(
    f'Stacked LSTM v4.1 — SNR + Channel Quality 6G 140 GHz Urban\n'
    f'LOOKBACK={LOOKBACK} | Clip≥{SNR_CLIP_MIN}dB | MSE | AdamW '
    f'| RMSE={rmse_n:.3f}dB | R²={r2_n:.4f} (SNR≥{SNR_CLIP_MIN}dB)',
    fontsize=12, fontweight='bold'
)

ax1 = fig.add_subplot(3, 3, 1)
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 1, 2)
ax5 = fig.add_subplot(3, 1, 3)

# Loss
ax1.plot(history['train_loss'], label='Train', color='steelblue')
ax1.plot(history['val_loss'],   label='Val',   color='tomato')
ax1.set_title('Loss (MSE)'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.legend(); ax1.grid(True, alpha=0.3)

# MAE
ax2.plot(history['train_mae'], label='Train MAE', color='steelblue')
ax2.plot(history['val_mae'],   label='Val MAE',   color='tomato')
ax2.set_title('MAE Curve (scaled)'); ax2.set_xlabel('Epoch')
ax2.legend(); ax2.grid(True, alpha=0.3)

# Scatter per channel quality
for cq in CQ_ORDER:
    mask = cq_actual == cq
    if mask.sum() > 0:
        ax3.scatter(y_test_orig[mask], y_pred_clipped[mask],
                    alpha=0.4, s=10, color=CQ_COLORS[cq], label=cq)
min_v = min(y_test_orig.min(), y_pred_clipped.min())
max_v = max(y_test_orig.max(), y_pred_clipped.max())
ax3.plot([min_v, max_v], [min_v, max_v], 'k--', lw=1.5, label='Ideal')
ax3.axhline(SNR_CLIP_MIN, color='purple', linestyle=':', lw=1.2,
            label=f'Clip floor ({SNR_CLIP_MIN} dB)')
ax3.set_title(f'Scatter per Channel Quality  (R²={r2_n:.4f})')
ax3.set_xlabel('Actual SNR (dB)'); ax3.set_ylabel('Predicted SNR (dB)')
ax3.legend(fontsize=7, loc='upper left'); ax3.grid(True, alpha=0.3)

# Actual vs Predicted + background channel quality
n_show = min(500, len(y_test_orig))
x_idx  = np.arange(n_show)

prev_cq, prev_i = cq_actual[0], 0
for i in range(1, n_show + 1):
    cur_cq = cq_actual[i] if i < n_show else None
    if cur_cq != prev_cq or i == n_show:
        ax4.axvspan(prev_i, i, alpha=0.12,
                    color=CQ_COLORS[prev_cq], label=prev_cq)
        prev_cq, prev_i = cur_cq, i

ax4.plot(x_idx, y_test_orig[:n_show],    label='Actual SNR',
         color='navy', alpha=0.85, lw=0.8)
ax4.plot(x_idx, y_pred_clipped[:n_show], label='Predicted SNR',
         color='tomato', alpha=0.85, lw=0.8, linestyle='--')
ax4.axhline(SNR_CLIP_MIN, color='purple', linestyle=':', lw=1,
            label=f'Clip min ({SNR_CLIP_MIN} dB)')

patches = [mpatches.Patch(color=CQ_COLORS[cq], alpha=0.4, label=cq)
           for cq in CQ_ORDER]
handles, labels_leg = ax4.get_legend_handles_labels()
ax4.legend(handles=handles + patches, fontsize=7,
           loc='upper right', ncol=4)
ax4.set_title(f'Actual vs Predicted SNR + Channel Quality Background ({n_show} sampel)')
ax4.set_xlabel('Sampel'); ax4.set_ylabel('SNR (dB)')
ax4.grid(True, alpha=0.3)

# Channel Quality bar chart
cq_cats = CQ_ORDER
actual_counts = [np.sum(cq_actual[:n_show] == c) for c in cq_cats]
pred_counts   = [np.sum(cq_pred[:n_show]   == c) for c in cq_cats]
x_bar = np.arange(len(cq_cats))
w = 0.35
ax5.bar(x_bar - w/2, actual_counts, w, label='Actual',    color='steelblue', alpha=0.8)
ax5.bar(x_bar + w/2, pred_counts,   w, label='Predicted', color='tomato',    alpha=0.8)
ax5.set_title(f'Distribusi Channel Quality — Actual vs Predicted ({n_show} sampel)')
ax5.set_xticks(x_bar); ax5.set_xticklabels(cq_cats)
ax5.set_ylabel('Jumlah Sampel')
ax5.legend(); ax5.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
PLOT_PATH = os.path.join(BASE_DIR, 'lstm_snr_results.png')
plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
print(f"    Plot utama disimpan : {PLOT_PATH}")
plt.show()

# ── [FIX 3 cont.] Confusion Matrix — plot terpisah
fig_cm, axes_cm = plt.subplots(1, 2, figsize=(14, 5))
fig_cm.suptitle('Channel Quality Classification — Confusion Matrix',
                fontsize=13, fontweight='bold')

cm_raw  = confusion_matrix(cq_actual, cq_pred, labels=CQ_ORDER)
cm_norm = cm_raw.astype(float) / (cm_raw.sum(axis=1, keepdims=True) + 1e-8)

sns.heatmap(cm_raw,  annot=True, fmt='d',    cmap='Blues',
            xticklabels=CQ_ORDER, yticklabels=CQ_ORDER,
            ax=axes_cm[0], linewidths=0.5)
axes_cm[0].set_title('Confusion Matrix (Count)')
axes_cm[0].set_xlabel('Predicted'); axes_cm[0].set_ylabel('Actual')

sns.heatmap(cm_norm, annot=True, fmt='.3f',  cmap='YlOrRd',
            xticklabels=CQ_ORDER, yticklabels=CQ_ORDER,
            ax=axes_cm[1], linewidths=0.5, vmin=0, vmax=1)
axes_cm[1].set_title('Confusion Matrix (Normalized — Recall per Kelas)')
axes_cm[1].set_xlabel('Predicted'); axes_cm[1].set_ylabel('Actual')

plt.tight_layout()
CM_PATH = os.path.join(BASE_DIR, 'channel_quality_confusion_matrix.png')
plt.savefig(CM_PATH, dpi=150, bbox_inches='tight')
print(f"    Confusion matrix    : {CM_PATH}")
plt.show()

# ── [FIX 2 cont.] Baseline comparison bar chart
fig_bl, ax_bl = plt.subplots(figsize=(8, 5))
metrics_name = ['RMSE (dB)', 'MAE (dB)']
vals_persist = [rmse_p, mae_p]
vals_lstm    = [rmse_l, mae_l]
x_bl = np.arange(len(metrics_name))
w_bl = 0.3
ax_bl.bar(x_bl - w_bl/2, vals_persist, w_bl, label='Persistence Baseline',
          color='gray', alpha=0.8)
ax_bl.bar(x_bl + w_bl/2, vals_lstm,    w_bl, label='LSTM v4.1',
          color='steelblue', alpha=0.85)
ax_bl.set_xticks(x_bl); ax_bl.set_xticklabels(metrics_name)
ax_bl.set_title(f'LSTM vs Persistence Baseline (SNR ≥ {SNR_CLIP_MIN} dB)\n'
                f'RMSE improve: {rmse_improve:.2f}%  |  MAE improve: {mae_improve:.2f}%')
ax_bl.set_ylabel('Error (dB)')
ax_bl.legend(); ax_bl.grid(True, alpha=0.3, axis='y')
for bar in ax_bl.patches:
    ax_bl.annotate(f'{bar.get_height():.4f}',
                   (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9)
plt.tight_layout()
BL_PATH = os.path.join(BASE_DIR, 'baseline_comparison.png')
plt.savefig(BL_PATH, dpi=150, bbox_inches='tight')
print(f"    Baseline comparison : {BL_PATH}")
plt.show()

# ─────────────────────────────────────────────
# 11. SIMPAN MODEL & HASIL
# ─────────────────────────────────────────────
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X'        : scaler_X,
    'scaler_y'        : scaler_y,
    'feature_cols'    : FEATURE_COLS_FINAL,
    'lookback'        : LOOKBACK,
    'snr_clip_min'    : SNR_CLIP_MIN,
    'metrics': {
        'rmse_clipped'           : rmse_c,
        'r2_clipped'             : r2_c,
        'rmse_normal'            : rmse_n,
        'mae_normal'             : mae_n,
        'r2_normal'              : r2_n,
        'mape_normal'            : mape_n,
        'rmse_persistence_normal': rmse_p,   # [FIX 2]
        'rmse_lstm_normal'       : rmse_l,
        'rmse_improve_pct'       : rmse_improve,
        'cq_overall_accuracy'    : overall_acc,
    }
}, os.path.join(BASE_DIR, 'LSTMFinal_model.pt'))
print(f"\n[9] Model disimpan : LSTMFinal_model.pt")

result_df = pd.DataFrame({
    'actual_snr_db'             : y_test_orig.flatten(),
    'predicted_snr_db'          : y_pred_clipped.flatten(),
    'error_db'                  : (y_test_orig - y_pred_clipped).flatten(),
    'channel_quality_actual'    : cq_actual,
    'channel_quality_predicted' : cq_pred
})
RESULT_PATH = os.path.join(BASE_DIR, 'lstm_prediction_results.csv')
result_df.to_csv(RESULT_PATH, index=False)
print(f"    Hasil prediksi : lstm_prediction_results.csv")

print(f"\n✅ Selesai! (v4.1 — shuffle=False + Persistence Baseline + Full CQ Report)\n")