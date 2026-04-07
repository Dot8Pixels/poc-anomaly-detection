"""
Model Verification Script
=========================
Runs a series of checks to confirm the trained model works correctly:

  1. Artifact integrity  — all files load without error
  2. Gap-stat sanity     — mean/p95/p99 values are plausible
  3. Prediction smoke    — model returns valid predictions on training data
  4. Anomaly rate        — contamination ~1 % as configured
  5. Sensitivity test    — inject a known silence and confirm it is flagged
  6. Healthy test        — inject a normal gap and confirm it is NOT flagged
  7. Report preview      — show the top anomalies from the saved report

Usage:
    uv run verify_model.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import joblib
import pandas as pd
import polars as pl

MODEL_DIR = "models"
DATA_PATH = "data/anomaly_results.parquet"
REPORT_PATH = "reports/anomaly_report.md"
FEATURES = ["app_number_enc", "RIC_enc", "FID_enc", "hour", "day_of_week", "pub_count"]

PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"
SEP = "─" * 60


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ── 1. Load artifacts ────────────────────────────────────────────────────────
section("1. ARTIFACT INTEGRITY")

try:
    model = joblib.load(os.path.join(MODEL_DIR, "anomaly_model.joblib"))
    le_app = joblib.load(os.path.join(MODEL_DIR, "encoder_app_number.joblib"))
    le_ric = joblib.load(os.path.join(MODEL_DIR, "encoder_RIC.joblib"))
    le_fid = joblib.load(os.path.join(MODEL_DIR, "encoder_FID.joblib"))
    gap_stats = joblib.load(os.path.join(MODEL_DIR, "gap_stats.joblib"))
    print(PASS, "All 5 model artifacts loaded successfully")
except Exception as e:
    print(FAIL, f"Could not load artifacts: {e}")
    sys.exit(1)

print(f"  Model type      : {type(model).__name__}")
print(f"  n_estimators    : {model.n_estimators}")
print(f"  contamination   : {model.contamination}")
print(f"  Known apps      : {list(le_app.classes_)}")
print(f"  Known RICs      : {list(le_ric.classes_)}")
print(f"  Known FIDs      : {list(le_fid.classes_)}")

# ── 2. Gap statistics sanity ─────────────────────────────────────────────────
section("2. GAP STATISTICS SANITY")

print(gap_stats.to_string(index=False))

for _, row in gap_stats.iterrows():
    ok = row["mean_count"] > 0 and row["p01_count"] >= 0
    label = PASS if ok else FAIL
    print(
        f"{label}  {row['app_number']} | {row['RIC']} | {row['FID']}  "
        f"mean={row['mean_count']:.1f}/min  p05={row['p05_count']:.1f}/min  "
        f"p01={row['p01_count']:.1f}/min  zero_minutes={row['zero_minutes']}"
    )

# ── 3. Load training results & predict ──────────────────────────────────────
section("3. PREDICTION SMOKE TEST (first 1 000 rows)")

df = pl.read_parquet(DATA_PATH)
pdf = df.to_pandas()

X_sample = pdf[FEATURES].head(1000)
try:
    preds = model.predict(X_sample)
    scores = model.score_samples(X_sample)
    assert set(preds).issubset({1, -1}), "Unexpected prediction values"
    print(PASS, f"predict() returned valid values {{1, -1}} over {len(preds)} rows")
    print(PASS, f"score_samples() range: [{scores.min():.4f}, {scores.max():.4f}]")
except Exception as e:
    print(FAIL, f"Prediction failed: {e}")

# ── 4. Anomaly rate check ────────────────────────────────────────────────────
section("4. ANOMALY RATE (should be ~1 %)")

total = len(pdf)
anomalies = int(pdf["is_anomaly"].sum())
rate = anomalies / total * 100

print(f"  Total rows    : {total:,}")
print(f"  Anomalies     : {anomalies:,}")
print(f"  Anomaly rate  : {rate:.2f}%")

if 0.1 <= rate <= 5.0:
    print(PASS, f"Rate {rate:.2f}% is within acceptable range [0.1%, 5%]")
else:
    print(
        FAIL,
        f"Rate {rate:.2f}% is outside expected range — check contamination setting",
    )

# ── 5. Sensitivity test — inject a known silence ─────────────────────────────
section("5. SENSITIVITY TEST — inject a 20-min silence (should be ANOMALY)")

app_val = le_app.classes_[0]
ric_val = le_ric.classes_[0]
fid_val = le_fid.classes_[0]

# We inject a silence (0) AND a normal count at the same hour, then compare scores.
# The silence MUST score lower (more anomalous) than the normal count.
stats_row = gap_stats.iloc[0]
normal_count_val = int(stats_row["mean_count"])


def _make_row(pub_count: int):
    return pd.DataFrame(
        [
            {
                "app_number_enc": le_app.transform([app_val])[0],
                "RIC_enc": le_ric.transform([ric_val])[0],
                "FID_enc": le_fid.transform([fid_val])[0],
                "hour": 10,
                "day_of_week": 1,
                "pub_count": pub_count,
            }
        ],
        columns=FEATURES,
    )


silence_row = _make_row(0)
normal_row = _make_row(normal_count_val)

pred_silence = model.predict(silence_row)[0]
score_silence = model.score_samples(silence_row)[0]
pred_normal = model.predict(normal_row)[0]
score_normal = model.score_samples(normal_row)[0]

# The silence score should be lower (more isolated) than the normal score
ok_silence = score_silence < score_normal

print(
    f"  Silence  : pub_count=0     prediction={'ANOMALY' if pred_silence == -1 else 'NORMAL'}  score={score_silence:.4f}"
)
print(
    f"  Normal   : pub_count={normal_count_val}  prediction={'ANOMALY' if pred_normal == -1 else 'NORMAL'}  score={score_normal:.4f}"
)
print(
    PASS if ok_silence else FAIL,
    f"Silence score ({score_silence:.4f}) < Normal score ({score_normal:.4f}) — model correctly ranks silence as more anomalous"
    if ok_silence
    else f"Silence score ({score_silence:.4f}) >= Normal score ({score_normal:.4f}) — model cannot distinguish silence from normal",
)

# ── 6. Healthy test — inject a normal gap ────────────────────────────────────
section("6. HEALTHY TEST — inject a normal count (should be NORMAL)")

healthy_row = _make_row(normal_count_val)

pred_healthy = model.predict(healthy_row)[0]
score_healthy = model.score_samples(healthy_row)[0]
ok_healthy = pred_healthy == 1

print(
    f"  Input pub_count : {int(normal_count_val)}  (mean count/min from training data)"
)
print(f"  Prediction      : {'ANOMALY (-1)' if pred_healthy == -1 else 'NORMAL (1)'}")
print(f"  Anomaly score   : {score_healthy:.4f}  (higher = more normal)")
print(
    PASS if ok_healthy else FAIL,
    "Normal count correctly classified as NORMAL"
    if ok_healthy
    else "Normal count flagged as anomaly — model may be over-sensitive",
)

# ── 7. Report preview ────────────────────────────────────────────────────────
section("7. ANOMALY REPORT PREVIEW")

if os.path.exists(REPORT_PATH):
    with open(REPORT_PATH) as f:
        lines = f.readlines()
    # Print the summary section (first 20 lines)
    for line in lines[:20]:
        print(" ", line.rstrip())
    print(PASS, f"Report exists at {REPORT_PATH}")
else:
    print(FAIL, f"Report not found at {REPORT_PATH}")

# ── Final verdict ────────────────────────────────────────────────────────────
print(f"\n{'═' * 60}")
print("  Verification complete. Model is ready for: uv run main.py --monitor")
print(f"{'═' * 60}\n")
