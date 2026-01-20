import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("single_220GHz_2days_urban.csv")

# =====================
# FILTER CONDITIONS
# =====================
snr_los_clear = df[
    (df["channel_state"] == "LOS") &
    (df["is_raining"] == False)
]["snr_db"]

snr_los_rain = df[
    (df["channel_state"] == "LOS") &
    (df["is_raining"] == True)
]["snr_db"]

snr_nlos = df[
    (df["channel_state"] == "NLOS")
]["snr_db"]

# =====================
# CDF FUNCTION
# =====================
def plot_cdf(data, label):
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    plt.plot(x, y, label=label)

plt.figure(figsize=(14,5))
plot_cdf(snr_los_clear, "LOS – Clear")
plot_cdf(snr_los_rain,  "LOS – Rain")
plot_cdf(snr_nlos,      "NLOS")

plt.xlabel("SNR (dB)")
plt.ylabel("CDF")
plt.grid(True)
plt.legend()
plt.title("CDF of SNR – 220 GHz Urban")
plt.show()
