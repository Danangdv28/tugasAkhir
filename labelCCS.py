import pandas as pd
import numpy as np
import os

BASE_DIR = "hasil_simulasi"

input_file = os.path.join(
    BASE_DIR,
    "single_140GHz_30days_urban.csv"
)

df = pd.read_csv(input_file)


WINDOW = 10

df["snr_mean_10"] = df["snr_db"].rolling(WINDOW, min_periods=1).mean()
df["snr_std_10"]  = df["snr_db"].rolling(WINDOW, min_periods=1).std()

p70 = df["snr_mean_10"].quantile(0.70)
p30 = df["snr_mean_10"].quantile(0.30)

def label_ccs(row):
    if row["snr_mean_10"] >= p70:
        return 0      # GOOD
    elif row["snr_mean_10"] >= p30:
        return 1      # DEGRADED
    else:
        return 2      # SEVERE

df["ccs"] = df.apply(label_ccs, axis=1)

# explicit NaN handling (BENAR, jangan drop)
df["fog_visibility_m"] = df["fog_visibility_m"].fillna(0)
df = df.ffill().bfill()

output_file = os.path.join(
    BASE_DIR,
    "single_140GHz_30days_urban_CCS_FULL.csv"
)

df.to_csv(output_file, index=False)

print("Total rows:", len(df))
print(df["ccs"].value_counts())
