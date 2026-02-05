import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

BASE_DIR = "hasil_simulasi"
RAW_FILE = "single_140GHz_30days_urban.csv"

input_path = os.path.join(BASE_DIR, RAW_FILE)

TARGET = "snr_db"
SHIFT = 1
TRAIN_RATIO = 0.7

FEATURES = [
    "rain_rate",
    "humidity_percent",
    "temperature_c",
    "fog_visibility_m",
    "path_loss_db"
]

df = pd.read_csv(input_path)
df["fog_visibility_m"] = df["fog_visibility_m"].fillna(0)

df["target"] = df[TARGET].shift(-SHIFT)
df = df.dropna().reset_index(drop=True)

X = df[FEATURES].values
y = df["target"].values

split = int(len(X) * TRAIN_RATIO)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

model = LinearRegression()
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)

print("\n=== LN – RAW ===")
print(f"RMSE : {np.sqrt(mean_squared_error(y_te, y_pred)):.3f} dB")
print(f"MAE  : {mean_absolute_error(y_te, y_pred):.3f} dB")
print(f"R²   : {r2_score(y_te, y_pred):.4f}")