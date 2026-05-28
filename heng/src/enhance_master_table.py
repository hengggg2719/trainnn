import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import numpy as np
import pandas as pd

# ============================================================
# PATHS - works from pacrise/trainnn/heng/src/
# ============================================================
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH   = os.path.join(REPO_ROOT, 'output', 'master_training_table_v2_with_aadt.csv')
OUT_PATH  = os.path.join(REPO_ROOT, 'output', 'master_training_table_v5.csv')

print("Loading master_training_table_v2_with_aadt.csv...")
df = pd.read_csv(IN_PATH, low_memory=False)
print(f"  Loaded {len(df):,} rows x {len(df.columns)} columns")

# Fix severe_crash and sample weights
print("\nFixing severe_crash and sample weights...")
df['severe_crash']  = df['severe_crash'].astype(int)
df['sample_weight'] = df['severe_crash'].map({1: 1.5, 0: 1.0})
print(f"  severe=1: {(df['severe_crash']==1).sum():,} rows weight=1.5")
print(f"  severe=0: {(df['severe_crash']==0).sum():,} rows weight=1.0")

# Data quality fixes
print("\nData quality fixes...")
if 'SPEED_ZONE' in df.columns:
    bad = df['SPEED_ZONE'].isin([777, 888]).sum()
    df.loc[df['SPEED_ZONE'].isin([777, 888]), 'SPEED_ZONE'] = 60
    print(f"  SPEED_ZONE: fixed {bad} sentinel values")
if 'DAY_OF_WEEK' in df.columns:
    bad = (df['DAY_OF_WEEK'] == 0).sum()
    df.loc[df['DAY_OF_WEEK'] == 0, 'DAY_OF_WEEK'] = 4
    print(f"  DAY_OF_WEEK: fixed {bad} invalid zeros")
if 'NODE_TYPE' in df.columns:
    bad = (df['NODE_TYPE'] == 'U').sum()
    df.loc[df['NODE_TYPE'] == 'U', 'NODE_TYPE'] = 'N'
    print(f"  NODE_TYPE: merged {bad} U into N")
if 'NO_OF_VEHICLES' in df.columns:
    bad = (df['NO_OF_VEHICLES'] == 0).sum()
    df.loc[df['NO_OF_VEHICLES'] == 0, 'NO_OF_VEHICLES'] = 1
    print(f"  NO_OF_VEHICLES: fixed {bad} zeros")

# Feature engineering
print("\nEngineering features...")
SPEED_RISK_MAP = {0:0, 40:1, 50:2, 60:3, 75:4, 80:4, 90:5, 100:6, 110:6}
DARKNESS_MAP   = {1:0.00, 2:0.30, 3:0.60, 4:0.85, 5:1.00, 6:0.70, 9:0.40}
df['speed_risk']     = df['SPEED_ZONE'].map(SPEED_RISK_MAP).fillna(3).astype(int)
df['darkness_score'] = df['LIGHT_CONDITION'].map(DARKNESS_MAP).fillna(0.40)
df['hour_sin']       = np.sin(2 * np.pi * df['crash_hour'] / 24)
df['hour_cos']       = np.cos(2 * np.pi * df['crash_hour'] / 24)
df['is_peak_hour']   = df['crash_hour'].isin([7,8,9,16,17,18]).astype(int)
df['day_sin']        = np.sin(2 * np.pi * df['DAY_OF_WEEK'] / 7)
df['day_cos']        = np.cos(2 * np.pi * df['DAY_OF_WEEK'] / 7)
print("  speed_risk, darkness_score, hour_sin/cos, day_sin/cos, is_peak_hour: done")

# Crash rate (Ben's advice)
print("\nCalculating crash_rate...")
if 'aadt_volume' in df.columns:
    df['_road_key'] = df['aadt_volume'].astype(str) + '_' + df['road_class'].astype(str)
    cpk = df.groupby('_road_key').size().reset_index()
    cpk.columns = ['_road_key', '_crashes_on_road']
    df = df.merge(cpk, on='_road_key', how='left')
    total_passes = df['aadt_volume'] * 365 * 7
    df['crash_rate'] = (df['_crashes_on_road'] / total_passes.replace(0, np.nan) * 1_000_000).fillna(0)
    df = df.drop(columns=['_road_key', '_crashes_on_road'])
    print(f"  crash_rate range: {df['crash_rate'].min():.2f} to {df['crash_rate'].max():.2f}")
else:
    df['crash_rate'] = 0.0
    print("  WARNING: aadt_volume missing - crash_rate set to 0")

# Final column selection
# NOTE: weather columns removed - applied as live multipliers at prediction time
print("\nSelecting final columns...")
FINAL_COLUMNS = [
    'severe_crash', 'sample_weight',
    'is_weekend', 'is_peak_hour', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'SPEED_ZONE', 'speed_risk', 'LIGHT_CONDITION', 'darkness_score',
    'ROAD_GEOMETRY_DESC', 'DISTANCE_LOCATION', 'NODE_TYPE', 'road_class',
    'wet_road', 'LGA_NAME', 'DEG_URBAN_NAME',
    'NO_OF_VEHICLES', 'has_pedestrian', 'has_cyclist',
    'aadt_volume', 'crash_rate', 'nearest_school_dist_m',
]

missing = [c for c in FINAL_COLUMNS if c not in df.columns]
if missing:
    print(f"  WARNING: Missing columns filled with 0: {missing}")
    for c in missing:
        df[c] = 0

df_out = df[FINAL_COLUMNS].copy()

# Fill nulls
print("\nFilling nulls...")
for col in df_out.select_dtypes(include=[np.number]).columns:
    if df_out[col].isnull().sum() > 0:
        df_out[col] = df_out[col].fillna(df_out[col].median())
for col in df_out.select_dtypes(include=['object']).columns:
    if df_out[col].isnull().sum() > 0:
        df_out[col] = df_out[col].fillna('Unknown')

assert df_out.isnull().sum().sum() == 0, "Nulls remain"
print("  Nulls: 0")

# Save
print(f"\nSaving to {OUT_PATH}...")
df_out.to_csv(OUT_PATH, index=False)

print(f"\n=== SUMMARY ===")
print(f"Shape: {df_out.shape}")
print(f"Columns ({len(df_out.columns)}):")
for i, col in enumerate(df_out.columns):
    print(f"  {i+1:2d}. {col}")
print(f"Nulls: {df_out.isnull().sum().sum()}")
print(f"severe=1: {(df_out['severe_crash']==1).sum():,} ({(df_out['severe_crash']==1).mean()*100:.1f}%)")
print(f"severe=0: {(df_out['severe_crash']==0).sum():,} ({(df_out['severe_crash']==0).mean()*100:.1f}%)")
print(f"aadt_volume zeros: {(df_out['aadt_volume']==0).sum():,}")
print(f"crash_rate zeros:  {(df_out['crash_rate']==0).sum():,}")
print("DONE - enhance_master_table.py complete")
print("Next step: run train_model.py")