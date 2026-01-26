import pandas as pd
import numpy as np

df = pd.read_csv("single_220GHz_14days_urban_CCS_FULL.csv")

print("Total rows:", len(df))
print("\nCCS distribution:")
print(df["ccs"].value_counts(normalize=True))

print("\nCheck NaN:")
print(df.isna().sum())

# CCS transition check
df["ccs_diff"] = df["ccs"].diff().abs()
print("\nAbrupt transitions (>1):")
print((df["ccs_diff"] > 1).sum())
