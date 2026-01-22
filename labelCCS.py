import pandas as pd
import numpy as np

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("single_140GHz_14days_urban.csv")

# =====================
# ROLLING FEATURES
# =====================
WINDOW = 10  # 10 minutes

df["snr_mean_10"] = df["snr_db"].rolling(WINDOW).mean()
df["snr_std_10"]  = df["snr_db"].rolling(WINDOW).std()

# =====================
# PERCENTILE THRESHOLD
# =====================
p70 = df["snr_mean_10"].quantile(0.70)
p30 = df["snr_mean_10"].quantile(0.30)


def label_ccs(row):
    if np.isnan(row["snr_mean_10"]):
        return np.nan

    if row["snr_mean_10"] >= p70:
        return 0      # GOOD
    elif row["snr_mean_10"] >= p30:
        return 1      # DEGRADED
    else:
        return 2      # SEVERE


df["ccs"] = df.apply(label_ccs, axis=1)

# =====================
# CLEAN & SAVE
# =====================
df_labeled = df.dropna().reset_index(drop=True)
df_labeled.to_csv("single_140GHz_14days_urban_CCS.csv", index=False)

print("✓ CCS labeling selesai")
print(df_labeled["ccs"].value_counts())
