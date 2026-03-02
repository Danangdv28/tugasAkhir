import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf, adfuller
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# 1. LOAD DATA
# =====================================================
file_path = "simulator/hasil_simulasi/single_140GHz_60days_urban.csv"
df = pd.read_csv(file_path)

print("Shape dataset:", df.shape)
print("\nKolom tersedia:")
print(df.columns)

# =====================================================
# 2. DETEKSI KOLOM TIME (jika ada)
# =====================================================
time_cols = [col for col in df.columns if "time" in col.lower() 
                                      or "date" in col.lower()]

if len(time_cols) > 0:
    print("\nKolom waktu terdeteksi:", time_cols)
    df[time_cols[0]] = pd.to_datetime(df[time_cols[0]], errors='coerce')
    df = df.sort_values(by=time_cols[0])
else:
    print("\nTidak ada kolom waktu terdeteksi. Pastikan data sudah urut waktu.")

# =====================================================
# 3. AMBIL KOLOM NUMERIK
# =====================================================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("\nKolom numerik:")
print(numeric_cols)

# Buang kolom konstan
numeric_cols = [col for col in numeric_cols if df[col].nunique() > 10]

print("\nKolom numerik valid (>10 unique values):")
print(numeric_cols)

# =====================================================
# 4. PILIH TARGET OTOMATIS (atau ganti manual)
# =====================================================
target_col = numeric_cols[0]  # bisa kamu ganti manual

print("\nTarget yang dianalisis:", target_col)

series = df[target_col].dropna().values

print("Jumlah data:", len(series))

# =====================================================
# 5. CEK STATIONARITY (ADF TEST)
# =====================================================
adf_result = adfuller(series)

print("\nADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

if adf_result[1] < 0.05:
    print("Data STATIONARY (p < 0.05)")
else:
    print("Data NON-STATIONARY (p >= 0.05)")
    print("Disarankan lakukan differencing sebelum interpretasi ACF.")

# =====================================================
# 6. PLOT ACF
# =====================================================
plt.figure(figsize=(12,5))
plot_acf(series, lags=180)
plt.title(f"ACF - {target_col}")
plt.show()

# =====================================================
# 7. PLOT PACF
# =====================================================
plt.figure(figsize=(12,5))
plot_pacf(series, lags=50, method='ywm')
plt.title(f"PACF - {target_col}")
plt.show()

# =====================================================
# 8. HITUNG LAG SIGNIFIKAN ACF
# =====================================================
acf_vals, confint = acf(series, nlags=180, alpha=0.05)

lower = confint[:,0] - acf_vals
upper = confint[:,1] - acf_vals

significant_lags_acf = []

for i in range(1, len(acf_vals)):
    if acf_vals[i] < lower[i] or acf_vals[i] > upper[i]:
        significant_lags_acf.append(i)

print("\nLag signifikan ACF:")
print(significant_lags_acf)
print("Jumlah lag signifikan ACF:", len(significant_lags_acf))

# =====================================================
# 9. HITUNG LAG SIGNIFIKAN PACF
# =====================================================
pacf_vals, confint_pacf = pacf(series, nlags=50, alpha=0.05)

lower_p = confint_pacf[:,0] - pacf_vals
upper_p = confint_pacf[:,1] - pacf_vals

significant_lags_pacf = []

for i in range(1, len(pacf_vals)):
    if pacf_vals[i] < lower_p[i] or pacf_vals[i] > upper_p[i]:
        significant_lags_pacf.append(i)

print("\nLag signifikan PACF:")
print(significant_lags_pacf)
print("Jumlah lag signifikan PACF:", len(significant_lags_pacf))

# =====================================================
# 10. REKOMENDASI LOOKBACK
# =====================================================
if len(significant_lags_acf) > 20:
    print("\nREKOMENDASI:")
    print("Temporal memory panjang. LSTM justified.")
    print("Lookback yang masuk akal: 20–50")
elif len(significant_lags_acf) > 5:
    print("\nREKOMENDASI:")
    print("Temporal memory sedang.")
    print("Lookback 10–20 cukup.")
else:
    print("\nREKOMENDASI:")
    print("Temporal memory pendek.")
    print("LSTM besar kemungkinan overkill.")
    print("Coba model sederhana (AR / Linear / XGBoost).")
    
series_diff = np.diff(series)
plt.figure(figsize=(12,5))
plot_acf(series_diff, lags=180)
plt.title("ACF after differencing")
plt.show()

plt.figure(figsize=(12,5))
plot_pacf(series_diff, lags=50, method='ywm')
plt.title("PACF after differencing")
plt.show()