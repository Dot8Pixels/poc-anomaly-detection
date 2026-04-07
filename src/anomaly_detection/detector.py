import os
import time

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder


def detect_silence_anomalies(
    data_path="data/mock_data.parquet",
    results_path="data/anomaly_results.parquet",
    report_path="reports/anomaly_report.md",
    model_dir="models",
):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading data from {data_path}...")
    df = pl.read_parquet(
        data_path, columns=["app_number", "timestamp", "RIC", "FID"]
    ).with_columns(
        pl.col("timestamp").str.to_datetime(
            "%Y-%m-%dT%H:%M:%S%.fZ", time_unit="us", time_zone="UTC"
        )
    )

    # ── 1. BUILD 1-MINUTE BUCKET GRID ──────────────────────────────────────────
    # Truncate every timestamp to the minute, then count publications per
    # (app, RIC, FID, minute).  A zero-count minute means silence.
    print("Building 1-minute publication buckets...")
    df = df.with_columns(pl.col("timestamp").dt.truncate("1m").alias("minute"))

    counts = (
        df.group_by(["app_number", "RIC", "FID", "minute"])
        .agg(pl.len().alias("pub_count"))
        .sort(["app_number", "RIC", "FID", "minute"])
    )

    # Build a full grid: every combination × every minute in range
    print("Zero-filling silent minutes...")
    min_ts = df["minute"].min()
    max_ts = df["minute"].max()
    assert min_ts is not None and max_ts is not None
    all_minutes = pl.DataFrame(
        {
            "minute": pl.datetime_range(
                min_ts, max_ts, interval="1m", time_zone="UTC", eager=True
            )
        }
    )
    streams = counts.select(["app_number", "RIC", "FID"]).unique()
    full_grid = streams.join(all_minutes, how="cross")

    grid = (
        full_grid.join(counts, on=["app_number", "RIC", "FID", "minute"], how="left")
        .with_columns(pl.col("pub_count").fill_null(0))
        .sort(["app_number", "RIC", "FID", "minute"])
    )

    # ── 2. FEATURE ENGINEERING ──────────────────────────────────────────────────
    print("Engineering time-based features...")
    grid = grid.with_columns(
        [
            pl.col("minute").dt.hour().alias("hour"),
            pl.col("minute").dt.weekday().alias("day_of_week"),
        ]
    )

    pdf = grid.to_pandas()

    # ── 3. ENCODE IDENTITIES ────────────────────────────────────────────────────
    print("Encoding application and instrument identities...")
    os.makedirs(model_dir, exist_ok=True)
    for col in ["app_number", "RIC", "FID"]:
        le = LabelEncoder()
        pdf[f"{col}_enc"] = le.fit_transform(pdf[col])
        joblib.dump(le, os.path.join(model_dir, f"encoder_{col}.joblib"))

    # ── 4. TRAIN ISOLATION FOREST ───────────────────────────────────────────────
    # Features: Identity + Time-of-day + Publication count per minute
    # A silence (pub_count=0) at a normally busy time will be isolated.
    print("Training Isolation Forest on bucket counts...")
    features = [
        "app_number_enc",
        "RIC_enc",
        "FID_enc",
        "hour",
        "day_of_week",
        "pub_count",
    ]
    X = pdf[features]

    # contamination ≈ fraction of silence minutes we expect in 10 days.
    # ~20-min silences / (10 days × 16 active hrs × 60 min) ≈ few per ~9600 minutes.
    model = IsolationForest(n_estimators=200, contamination=0.005, random_state=42)
    model.fit(X)

    joblib.dump(model, os.path.join(model_dir, "anomaly_model.joblib"))
    print(f"Model saved to {model_dir}/")

    # ── 5. SAVE PER-STREAM COUNT STATISTICS ─────────────────────────────────────
    print("Computing per-stream count statistics for alert context...")
    # Only compute stats on non-zero minutes (normal activity)
    active = pdf[pdf["pub_count"] > 0]
    grp = active.groupby(["app_number", "RIC", "FID"])["pub_count"]
    count_stats = grp.mean().rename("mean_count").reset_index()
    count_stats = (
        count_stats.merge(
            grp.std().rename("std_count").reset_index(), on=["app_number", "RIC", "FID"]
        )
        .merge(
            grp.quantile(0.05).rename("p05_count").reset_index(),
            on=["app_number", "RIC", "FID"],
        )
        .merge(
            grp.quantile(0.01).rename("p01_count").reset_index(),
            on=["app_number", "RIC", "FID"],
        )
    )
    # Also track total zero-minutes per stream
    zero_counts = (
        pdf[pdf["pub_count"] == 0]
        .groupby(["app_number", "RIC", "FID"])
        .size()
        .rename("zero_minutes")
        .reset_index()
    )
    count_stats = count_stats.merge(
        zero_counts, on=["app_number", "RIC", "FID"], how="left"
    )
    count_stats["zero_minutes"] = count_stats["zero_minutes"].fillna(0).astype(int)
    joblib.dump(count_stats, os.path.join(model_dir, "gap_stats.joblib"))
    print(f"Count statistics saved to {model_dir}/gap_stats.joblib")

    # ── 6. LABEL & SAVE RESULTS ─────────────────────────────────────────────────
    pdf["ml_score"] = model.predict(X)
    pdf["is_anomaly"] = pdf["ml_score"] == -1

    results = pl.from_pandas(pdf)
    results.write_parquet(results_path)

    anomalies = results.filter(pl.col("is_anomaly"))
    summary_md = f"""# Bucket-Count Publication Analysis
Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}

## ML Strategy
- **Feature**: `pub_count` — publications per 1-minute window (0 = silence).
- **Grid**: Full timeline zero-filled so silent minutes appear explicitly.
- **Goal**: Learn the normal publication frequency per App/RIC/FID/hour and flag deviations.

## Summary
- **Total 1-min Windows Analyzed:** {len(results):,}
- **Anomalous Windows:** {len(anomalies):,}
- **Silent Windows (pub_count=0):** {int(results.filter(pl.col("pub_count") == 0)["pub_count"].len()):,}

## Top Anomalous Windows (Lowest Counts)
| App | RIC | FID | Minute | Count | Score |
|-----|-----|-----|--------|-------|-------|
"""
    for row in anomalies.sort("pub_count").head(20).iter_rows(named=True):
        summary_md += (
            f"| {row['app_number']} | {row['RIC']} | {row['FID']} "
            f"| {row['minute']} | {row['pub_count']} | {row['ml_score']} |\n"
        )

    with open(report_path, "w") as f:
        f.write(summary_md)
    print(f"Baseline analysis complete. Report: {report_path}")


if __name__ == "__main__":
    detect_silence_anomalies()
