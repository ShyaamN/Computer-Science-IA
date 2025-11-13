import pandas as pd
import numpy as np

df = pd.read_csv('data/training/kepler.csv', comment='#')
target_col = 'koi_disposition'

for col in df.columns:
    if df[col].dtype == 'object' and col != target_col:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=[target_col])
Y = df[target_col]

ordered = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
label_map = {v: i for i, v in enumerate(ordered) if v in Y.unique()}
enc = Y.map(label_map)

corrs = []
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) >= 3:
            corr = df[col].corr(enc)
            if not pd.isna(corr):
                corrs.append((col, abs(corr)))

corrs.sort(key=lambda x: x[1], reverse=True)

print(f"Total features with valid correlation: {len(corrs)}")
print(f"\nTop 15 features:")
for i, (col, corr) in enumerate(corrs[:15], 1):
    print(f"  {i}. {col}: {corr:.4f}")

print(f"\nFeatures with correlation >= 0.30: {sum(1 for c in corrs if c[1] >= 0.30)}")
print(f"Features with correlation >= 0.20: {sum(1 for c in corrs if c[1] >= 0.20)}")
print(f"Features with correlation >= 0.10: {sum(1 for c in corrs if c[1] >= 0.10)}")
