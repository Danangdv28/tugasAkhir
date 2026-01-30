import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("single_140GHz_14days_urban_CCS_FULL.csv")

# shift target
df["snr_next"] = df["snr_db"].shift(-1)
df = df.dropna().reset_index(drop=True)

FEATURES = [
    "snr_db",
    "ccs",
    "rain_rate",
    "humidity_percent",
    "temperature_c",
    "path_loss_db"
]

X = df[FEATURES].values
y = df["snr_next"].values

# =====================
# TIME-BASED SPLIT
# =====================
N = len(df)
train_end = int(N * 0.7)
val_end   = int(N * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_test,  y_test  = X[val_end:],  y[val_end:]

# =====================
# MODEL
# =====================
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

# =====================
# EVALUATION
# =====================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("XGBoost Regression Results")
print(f"RMSE : {rmse:.3f} dB")
print(f"MAE  : {mae:.3f} dB")
print(f"R²   : {r2:.4f}")