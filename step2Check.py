import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
CSV_PATH = "single_220GHz_14days_urban_CCS.csv"   # GANTI JIKA PERLU
SHOW_PLOT = True                                 # False jika tidak mau plot

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(CSV_PATH)

print("="*80)
print("STEP-2 DATA VALIDATION")
print(f"File : {CSV_PATH}")
print(f"Total samples : {len(df)}")
print("="*80)

# =========================================================
# STEP-2A : MONOTONICITY CHECK (SNR vs CCS)
# =========================================================
print("\n[STEP-2A] MONOTONICITY CHECK")

stats = df.groupby("ccs")["snr_db"].describe()[["mean", "50%", "std", "count"]]
print(stats)

med = stats["50%"]

if 0 in med and 1 in med and 2 in med and (med.loc[0] > med.loc[1] > med.loc[2]):
    print("✅ PASS: CCS monotonic terhadap SNR")
else:
    print("❌ FAIL: CCS TIDAK monotonic terhadap SNR")

# =========================================================
# STEP-2B : TEMPORAL CONSISTENCY (CCS JUMP)
# =========================================================
print("\n[STEP-2B] TEMPORAL CONSISTENCY")

df["ccs_prev"] = df["ccs"].shift(1)
df["jump"] = abs(df["ccs"] - df["ccs_prev"])

jump_counts = df["jump"].value_counts().sort_index()
print("\nJump distribution:")
print(jump_counts)

jump_02 = jump_counts.get(2, 0)
total_jump = jump_counts.sum()

ratio = jump_02 / total_jump if total_jump > 0 else 0
print(f"\nExtreme jump (0↔2): {jump_02} / {total_jump} ({ratio*100:.2f}%)")

if ratio < 0.05:
    print("✅ PASS: Temporal CCS stabil")
elif ratio < 0.10:
    print("⚠ WARNING: CCS agak noisy")
else:
    print("❌ FAIL: CCS terlalu fluktuatif")

# =========================================================
# STEP-2C : DWELL TIME (RUN LENGTH)
# =========================================================
print("\n[STEP-2C] DWELL TIME CHECK")

runs = []
current = df.loc[0, "ccs"]
length = 1

for i in range(1, len(df)):
    if df.loc[i, "ccs"] == current:
        length += 1
    else:
        runs.append((current, length))
        current = df.loc[i, "ccs"]
        length = 1

runs.append((current, length))

run_df = pd.DataFrame(runs, columns=["ccs", "run_length"])

dwell_stats = run_df.groupby("ccs")["run_length"].describe()[["mean", "50%", "min"]]
print(dwell_stats)

if (dwell_stats["50%"] >= 5).all():
    print("✅ PASS: CCS cukup stabil untuk LSTM")
else:
    print("⚠ WARNING: Beberapa CCS terlalu singkat")

# =========================================================
# OPTIONAL PLOT
# =========================================================
if SHOW_PLOT:
    plt.figure(figsize=(14,4))
    plt.plot(df["ccs"], lw=1)
    plt.yticks([0,1,2], ["GOOD","DEGRADED","SEVERE"])
    plt.title("Temporal CCS Behavior")
    plt.xlabel("Time index (minute)")
    plt.grid(True)
    plt.show()

print("\n" + "="*80)
print("VALIDATION SELESAI")
print("Silakan kirim output terminal ke saya untuk interpretasi final.")
print("="*80)
