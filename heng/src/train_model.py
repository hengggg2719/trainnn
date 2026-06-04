import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import math
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # no GUI - saves to file instead of opening a window
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    f1_score, recall_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

# ============================================================
# PATHS - works from pacrise/trainnn/heng/src/
# ============================================================
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(REPO_ROOT, 'output', 'master_training_table_v8.csv')
MODEL_PATH = os.path.join(REPO_ROOT, 'output', 'model.pkl')
ENC_PATH   = os.path.join(REPO_ROOT, 'output', 'encoders.pkl')
FEAT_PATH  = os.path.join(REPO_ROOT, 'output', 'feature_names.txt')

TARGET       = 'severe_crash'
WEIGHT_COL   = 'sample_weight'
RANDOM_STATE = 42

CATEGORICAL_COLS = [
    'ROAD_GEOMETRY_DESC',
    'LGA_NAME',
    'DEG_URBAN_NAME',
    'NODE_TYPE',
    'road_class',
]

# ============================================================
# STEP 1 - Load data
# ============================================================
print("Loading master_training_table_v8.csv...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Shape: {df.shape}")
print(f"  Nulls: {df.isnull().sum().sum()}")
assert df.isnull().sum().sum() == 0, "Nulls found - fix the table first"

print(f"\n  Columns ({len(df.columns)}):")
for i, col in enumerate(df.columns):
    print(f"    {i+1:2d}. {col}")

print(f"\n  severe=1: {(df[TARGET]==1).sum():,} ({(df[TARGET]==1).mean()*100:.1f}%)")
print(f"  severe=0: {(df[TARGET]==0).sum():,} ({(df[TARGET]==0).mean()*100:.1f}%)")

# ============================================================
# STEP 2 - Separate features, target, weights
# ============================================================
print("\nPreparing features...")
weights = df[WEIGHT_COL].values
y       = df[TARGET].values
X       = df.drop(columns=[TARGET, WEIGHT_COL])

print(f"  Features: {X.shape[1]}")

# ============================================================
# STEP 3 - Label encode categorical columns
# ============================================================
print("\nEncoding categoricals...")
encoders = {}
for col in CATEGORICAL_COLS:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        print(f"  {col}: {len(le.classes_)} classes")

# ============================================================
# STEP 4 - Train/val/test split 70/15/15
# ============================================================
print("\nSplitting 70/15/15...")
X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(
    X, y, weights, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
    X_temp, y_temp, w_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)
print(f"  Train: {len(X_train):,}")
print(f"  Val:   {len(X_val):,}")
print(f"  Test:  {len(X_test):,}")

# ============================================================
# STEP 5 - Train
# ============================================================
print("\nTraining RandomForest (300 trees)...")
model = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    min_samples_leaf=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)
model.fit(X_train, y_train, sample_weight=w_train)
print("  Training complete")

# ============================================================
# STEP 6 - Evaluate
# ============================================================
print("\n=== VALIDATION ===")
y_val_prob = model.predict_proba(X_val)[:,1]
y_val_pred = model.predict(X_val)
val_auc = roc_auc_score(y_val, y_val_prob)
print(f"  AUC:    {val_auc:.4f}")
print(f"  Acc:    {accuracy_score(y_val, y_val_pred)*100:.1f}%")
print(f"  F1:     {f1_score(y_val, y_val_pred):.3f}")
print(f"  Recall: {recall_score(y_val, y_val_pred)*100:.1f}%")

print("\n=== TEST ===")
y_test_prob = model.predict_proba(X_test)[:,1]
y_test_pred = model.predict(X_test)
test_auc    = roc_auc_score(y_test, y_test_prob)
test_recall = recall_score(y_test, y_test_pred)
test_acc    = accuracy_score(y_test, y_test_pred)
print(f"  AUC:    {test_auc:.4f}")
print(f"  Acc:    {test_acc*100:.1f}%")
print(f"  F1:     {f1_score(y_test, y_test_pred):.3f}")
print(f"  Recall: {test_recall*100:.1f}%")
print()
print(classification_report(y_test, y_test_pred, target_names=['non-severe','severe']))

# --- Confusion Matrix breakdown ---
cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix Breakdown:")
print(f"  True Negatives  (correctly said non-severe): {tn:,}")
print(f"  False Positives (wrongly flagged as severe):  {fp:,}")
print(f"  False Negatives (missed actual severe):       {fn:,}  <-- keep this low")
print(f"  True Positives  (correctly caught severe):    {tp:,}")

# --- Save Confusion Matrix Plot ---
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Severe", "Severe"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix - RF300-v8")
plt.tight_layout()
cm_path = os.path.join(REPO_ROOT, 'output', 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150)
plt.clf()
print(f"\nSaved: output/confusion_matrix.png")

# --- Save ROC Curve Plot ---
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, y_test_prob, name=f"RF300-v8 (AUC={test_auc:.3f})", ax=ax)
ax.plot([0,1],[0,1], 'k--', linewidth=0.8, label="Random guess")
ax.set_title("ROC Curve - RF300-v8")
ax.legend()
plt.tight_layout()
roc_path = os.path.join(REPO_ROOT, 'output', 'roc_curve.png')
plt.savefig(roc_path, dpi=150)
plt.clf()
print(f"Saved: output/roc_curve.png")

# ============================================================
# STEP 7 - Feature importances
# ============================================================
print("\n=== TOP 22 FEATURE IMPORTANCES ===")
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
for i, (feat, imp) in enumerate(importances.items()):
    bar = "X" * int(imp * 100 / 0.5)
    print(f"  {i+1:2d}. {feat:<30} {imp*100:.2f}%  {bar}")

# --- Save Feature Importances Bar Chart ---
fig, ax = plt.subplots(figsize=(9, 7))
colors = ["#d62728" if imp > 0.08 else "#1f77b4" for imp in importances.values[::-1]]
ax.barh(importances.index[::-1], importances.values[::-1] * 100, color=colors)
ax.set_xlabel("Importance (%)")
ax.set_title("Feature Importances - RF300-v8\n(red = top contributors above 8%)")
ax.axvline(x=5, color='gray', linestyle='--', linewidth=0.8, label="5% line")
ax.legend()
plt.tight_layout()
fi_path = os.path.join(REPO_ROOT, 'output', 'feature_importances.png')
plt.savefig(fi_path, dpi=150)
plt.clf()
print(f"Saved: output/feature_importances.png")

# ============================================================
# STEP 8 - Sanity check predictions
# ============================================================
print("\n=== SANITY CHECK ===")
print("  Do scores vary meaningfully between safe and dangerous?")

def enc(col, val):
    if col in encoders and val in encoders[col].classes_:
        return int(encoders[col].transform([val])[0])
    return 0

feature_names = list(X.columns)

def make_row(is_wknd, is_peak, hour, dow, spd_risk, dark,
             road_geom, dist_loc, node, rd_cls, wet,
             lga, deg, vehicles, aadt, cr, school,
             pub_hol=0, sch_hol=0, dls=0):
    row = {
        'is_weekend':            is_wknd,
        'is_peak_hour':          is_peak,
        'hour_sin':              math.sin(2*math.pi*hour/24),
        'hour_cos':              math.cos(2*math.pi*hour/24),
        'day_sin':               math.sin(2*math.pi*dow/7),
        'day_cos':               math.cos(2*math.pi*dow/7),
        'speed_risk':            spd_risk,
        'darkness_score':        dark,
        'ROAD_GEOMETRY_DESC':    enc('ROAD_GEOMETRY_DESC', road_geom),
        'DISTANCE_LOCATION':     dist_loc,
        'NODE_TYPE':             enc('NODE_TYPE', node),
        'road_class':            enc('road_class', rd_cls),
        'wet_road':              wet,
        'LGA_NAME':              enc('LGA_NAME', lga),
        'DEG_URBAN_NAME':        enc('DEG_URBAN_NAME', deg),
        'NO_OF_VEHICLES':        vehicles,
        'aadt_volume':           aadt,
        'crash_rate':            cr,
        'nearest_school_dist_m': school,
        'is_public_holiday':     pub_hol,
        'is_school_holiday':     sch_hol,
        'is_daylight_saving':    dls,
    }
    return [row[f] for f in feature_names]

cases = [
    ("SAFE  - sunny midday 60kmh urban busy road low crash rate",
     make_row(0, 0, 12, 4, 3, 0.0,
              'T intersection', 10, 'I', 'Local_Road', 0,
              'MELBOURNE', 'MELB_URBAN', 1, 8000, 0.2, 800)),
    ("MOD   - peak hour 80kmh main road",
     make_row(0, 1, 8, 3, 4, 0.0,
              'Not at intersection', 50, 'N', 'Main_Road', 0,
              'MONASH', 'MELB_URBAN', 2, 12000, 1.5, 500)),
    ("RISKY - night wet rural 100kmh high crash rate",
     make_row(0, 0, 22, 4, 6, 0.6,
              'Not at intersection', 200, 'N', 'Freeway_Highway', 1,
              'WODONGA', 'RURAL_VICTORIA', 1, 0, 8.0, 3000)),
    ("DANGER - 1am wet dark rural weekend public holiday",
     make_row(1, 0, 1, 7, 6, 0.85,
              'Not at intersection', 500, 'N', 'Local_Road', 1,
              'MILDURA', 'RURAL_VICTORIA', 1, 0, 15.0, 5000,
              pub_hol=1)),
]

for label, row in cases:
    prob  = model.predict_proba([row])[0][1]
    score = prob * 100
    band  = 'LOW' if score<=30 else 'MEDIUM' if score<=60 else 'HIGH' if score<=80 else 'CRITICAL'
    print(f"  {label}")
    print(f"    -> {score:.1f} [{band}]")

# ============================================================
# STEP 9 - Save
# ============================================================
print("\nSaving model artifacts...")
joblib.dump(model, MODEL_PATH)
print(f"  model.pkl: {os.path.getsize(MODEL_PATH)/1024/1024:.1f} MB")
joblib.dump(encoders, ENC_PATH)
print(f"  encoders.pkl saved")
with open(FEAT_PATH, 'w') as f:
    f.write('\n'.join(feature_names))
print(f"  feature_names.txt: {len(feature_names)} features")

print(f"\n=== FINAL SUMMARY ===")
print(f"Model: RF300-v8-AUC{test_auc:.3f}")
print(f"Features: {len(feature_names)}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Recall: {test_recall*100:.1f}%")
print(f"Test Accuracy: {test_acc*100:.1f}%")
print(f"crash_rate importance: {importances.get('crash_rate',0)*100:.2f}%")
print(f"aadt_volume importance: {importances.get('aadt_volume',0)*100:.2f}%")
print(f"\nSaved to: {REPO_ROOT}/output/")
print("  model.pkl")
print("  encoders.pkl")
print("  feature_names.txt")
print("  confusion_matrix.png")
print("  roc_curve.png")
print("  feature_importances.png")
print("DONE - train_model.py complete")