import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIG
# =========================
CSV_FILE = "single_140GHz_14days_urban_CCS_FULL.csv"
LOOKBACK = 20
HORIZON = 1

FEATURES_X = [
    "snr_mean_10",
    "snr_std_10",
    "path_loss_db",
    "rain_rate",
    "humidity_percent",
    "temperature_c",
    "fog_visibility_m"
]

TARGET = "ccs"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_FILE)

# basic cleaning
df["fog_visibility_m"] = df["fog_visibility_m"].fillna(0)
df = df.dropna().reset_index(drop=True)

# =========================
# NORMALIZATION (X ONLY)
# =========================
scaler = StandardScaler()
df[FEATURES_X] = scaler.fit_transform(df[FEATURES_X])

# =========================
# BUILD SEQUENCES
# =========================
X, y = [], []

for i in range(len(df) - LOOKBACK - HORIZON + 1):
    X_seq = df.loc[i:i+LOOKBACK-1, FEATURES_X].values
    y_target = df.loc[i+LOOKBACK+HORIZON-1, TARGET]

    X.append(X_seq)
    y.append(y_target)

X = np.array(X)
y = np.array(y)

# =========================
# SAVE
# =========================
np.save("X_lstm.npy", X)
np.save("y_lstm.npy", y)

print("✓ Dataset siap")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("CCS distribution:", np.bincount(y.astype(int)))
