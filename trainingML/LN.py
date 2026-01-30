import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("single_140GHz_14days_urban.csv")

# =========================
# BUILD REGRESSION DATASET
# =========================
FEATURES = [
    "snr_db",             # SNR(t-1)
    # "ccs",
    "rain_rate",
    "humidity_percent",
    "temperature_c",
    "path_loss_db"
]

TARGET = "snr_db"

# shift target → predict future SNR
df["snr_future"] = df["snr_db"].shift(-1)

df = df.dropna().reset_index(drop=True)

X = df[FEATURES]
y = df["snr_future"]

# =========================
# TRAIN / TEST SPLIT (TIME-SERIES!)
# =========================
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# =========================
# NORMALIZATION
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# =========================
# REGRESSION MODEL
# =========================
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# =========================
# PREDICTION
# =========================
y_pred = model.predict(X_test_scaled)

# =========================
# METRICS (INI YANG DOSEN MAU)
# =========================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("=== SNR REGRESSION RESULTS ===")
print(f"RMSE : {rmse:.3f} dB")
print(f"MAE  : {mae:.3f} dB")
print(f"R²   : {r2:.4f}")