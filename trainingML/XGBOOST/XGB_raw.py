import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RAW_FILE = "single_140GHz_14days_urban.csv"
TARGET = "snr_db"
SHIFT = 1
TRAIN_RATIO = 0.7

FEATURES = [
    "snr_db",
    "rain_rate",
    "humidity_percent",
    "temperature_c",
    "fog_visibility_m",
    "path_loss_db"
]

df = pd.read_csv(RAW_FILE)
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

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)

print("\n=== XGB – RAW ===")
print(f"RMSE : {np.sqrt(mean_squared_error(y_te, y_pred)):.3f} dB")
print(f"MAE  : {mean_absolute_error(y_te, y_pred):.3f} dB")
print(f"R²   : {r2_score(y_te, y_pred):.4f}")