"""
24/7 Real-Time Publication Monitor
====================================
Replays historical parquet data minute-by-minute (as if it were a live feed).
For every completed 1-minute window it asks the Isolation Forest:
  "Is this publication count normal for this App/RIC/FID at this hour?"

A zero-count or suspiciously low count triggers a detailed alert.

Run standalone:
    python src/anomaly_detection/monitor.py

Or via main.py:
    python main.py --monitor
"""

import os
import time
from datetime import timezone

import joblib
import pandas as pd
import polars as pl

# ── Formatting helpers ──────────────────────────────────────────────────────

_SEP = "─" * 70
_DSEP = "═" * 70


def _ts(dt) -> str:
    return dt.strftime("%Y-%m-%d %H:%M UTC") if dt else "N/A"


def _fmt_count(n: int) -> str:
    return str(n) if n > 0 else "0  ← SILENCE"


# ── Model loader ────────────────────────────────────────────────────────────


def _load_artifacts(model_dir: str):
    required = [
        "anomaly_model.joblib",
        "encoder_app_number.joblib",
        "encoder_RIC.joblib",
        "encoder_FID.joblib",
        "gap_stats.joblib",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts in '{model_dir}': {missing}\n"
            "Run the training pipeline first: python main.py --train"
        )
    model = joblib.load(os.path.join(model_dir, "anomaly_model.joblib"))
    encoders = {
        "app_number": joblib.load(os.path.join(model_dir, "encoder_app_number.joblib")),
        "RIC": joblib.load(os.path.join(model_dir, "encoder_RIC.joblib")),
        "FID": joblib.load(os.path.join(model_dir, "encoder_FID.joblib")),
    }
    count_stats = joblib.load(os.path.join(model_dir, "gap_stats.joblib"))
    return model, encoders, count_stats


# ── Stats lookup ─────────────────────────────────────────────────────────────


def _get_stream_stats(count_stats: pd.DataFrame, app, ric: str, fid: str) -> dict:
    row = count_stats[
        (count_stats["app_number"] == app)
        & (count_stats["RIC"] == ric)
        & (count_stats["FID"] == fid)
    ]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "mean_count": r.get("mean_count", None),
        "p05_count": r.get("p05_count", None),
        "p01_count": r.get("p01_count", None),
        "zero_minutes": r.get("zero_minutes", None),
    }


# ── Alert / normal printers ──────────────────────────────────────────────────


def _print_alert(
    stream_key: tuple, minute, pub_count: int, anomaly_score: float, stats: dict
):
    app, ric, fid = stream_key
    print(f"\n{'🚨 ANOMALY ALERT':^70}")
    print(_SEP)
    print(f"  Stream      : App {app}  |  RIC: {ric}  |  FID: {fid}")
    print(f"  Minute      : {_ts(minute)}")
    print(f"  Publications: {_fmt_count(pub_count)}")
    if stats:
        print(
            f"  Normal range: avg {stats['mean_count']:.0f}/min  "
            f"| p05 (low) {stats['p05_count']:.0f}/min  "
            f"| p01 (very low) {stats['p01_count']:.0f}/min"
        )
        if stats.get("zero_minutes") is not None:
            print(
                f"  Training zeros: {int(stats['zero_minutes'])} silent minutes seen in history"
            )
    print(f"  ML Score    : {anomaly_score:.4f}  (lower = more anomalous)")
    verdict = (
        "⛔ SILENT — no publications this minute"
        if pub_count == 0
        else "⚠️  VERY LOW — far below normal frequency"
    )
    print(f"  Verdict     : {verdict}")
    print(_SEP)


def _print_normal(stream_key: tuple, minute, pub_count: int, anomaly_score: float):
    app, ric, fid = stream_key
    print(
        f"  ✅ [{_ts(minute)}]  App {app} | {ric} | {fid}"
        f"  — {pub_count} pub/min  score {anomaly_score:.4f}"
    )


# ── Core monitor ─────────────────────────────────────────────────────────────


def monitor_24_7(
    data_path: str = "data/mock_data.parquet",
    model_dir: str = "models",
    speed_factor: float = 500.0,
    verbose_normal: bool = False,
):
    """
    Replay historical parquet data minute-by-minute as a real-time stream
    and predict publication anomalies using the trained Isolation Forest.

    Parameters
    ----------
    data_path      : Path to the raw parquet data file.
    model_dir      : Directory containing trained model artifacts.
    speed_factor   : Time-compression.  500 = each simulated minute plays in
                     60/500 = 0.12 real seconds.  Use 1.0 for true real-time.
    verbose_normal : If True, also print healthy windows (very verbose).
    """
    print(_DSEP)
    print(f"{'24/7 PUBLICATION ANOMALY MONITOR':^70}")
    print(_DSEP)
    print(f"  Data source  : {data_path}")
    print(f"  Model dir    : {model_dir}")
    print(
        f"  Speed factor : {speed_factor}x  "
        f"(1 sim-minute = {60 / speed_factor:.3f} real-sec)"
    )
    print(_DSEP)

    # ── 1. Load model ────────────────────────────────────────────────────────
    print("\nLoading model artifacts...")
    try:
        model, encoders, count_stats = _load_artifacts(model_dir)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return
    print("  ✔ Model, encoders and count statistics loaded.")

    # ── 2. Load & prepare data ────────────────────────────────────────────────
    if not os.path.exists(data_path):
        print(f"\n❌ Data file not found: {data_path}")
        return

    print(f"\nLoading and bucketing data from {data_path}...")
    df = (
        pl.read_parquet(data_path, columns=["app_number", "timestamp", "RIC", "FID"])
        .with_columns(
            pl.col("timestamp").str.to_datetime(
                "%Y-%m-%dT%H:%M:%S%.fZ", time_unit="us", time_zone="UTC"
            )
        )
        .with_columns(pl.col("timestamp").dt.truncate("1m").alias("minute"))
    )

    # Count publications per (app, RIC, FID, minute)
    counts = (
        df.group_by(["app_number", "RIC", "FID", "minute"])
        .agg(pl.len().alias("pub_count"))
        .sort("minute")
    )

    # Zero-fill: build full grid so silent minutes are explicit
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
        .with_columns(
            [
                pl.col("minute").dt.hour().alias("hour"),
                pl.col("minute").dt.weekday().alias("day_of_week"),
            ]
        )
        .sort("minute")
    )

    total_windows = len(grid)
    print(f"  ✔ {total_windows:,} minute-windows ready. Starting replay...\n")

    # ── 3. Replay loop ────────────────────────────────────────────────────────
    features = [
        "app_number_enc",
        "RIC_enc",
        "FID_enc",
        "hour",
        "day_of_week",
        "pub_count",
    ]
    alert_count = 0
    normal_count = 0
    unknown_count = 0
    prev_minute = None

    print("─" * 70)
    print(f"{'LIVE FEED':^70}")
    print("─" * 70 + "\n")

    for row in grid.iter_rows(named=True):
        app = row["app_number"]
        ric = row["RIC"]
        fid = row["FID"]
        minute = row["minute"]
        if minute.tzinfo is None:
            minute = minute.replace(tzinfo=timezone.utc)

        stream_key = (app, ric, fid)

        # Pace replay — sleep proportional to simulated time elapsed
        if prev_minute is not None:
            sim_delta = (minute - prev_minute).total_seconds()
            real_sleep = sim_delta / speed_factor
            if real_sleep > 0:
                time.sleep(real_sleep)
        prev_minute = minute

        # Encode stream identity
        try:
            app_enc = encoders["app_number"].transform([app])[0]
            ric_enc = encoders["RIC"].transform([ric])[0]
            fid_enc = encoders["FID"].transform([fid])[0]
        except ValueError:
            print(f"  ⚠️  Unknown stream App {app} | {ric} | {fid} — needs retraining")
            unknown_count += 1
            continue

        X = pd.DataFrame(
            [
                {
                    "app_number_enc": app_enc,
                    "RIC_enc": ric_enc,
                    "FID_enc": fid_enc,
                    "hour": row["hour"],
                    "day_of_week": row["day_of_week"],
                    "pub_count": row["pub_count"],
                }
            ],
            columns=features,
        )

        prediction = model.predict(X)[0]  # 1=normal, -1=anomaly
        anomaly_score = model.score_samples(X)[0]  # lower = more anomalous
        is_anomaly = prediction == -1

        if is_anomaly:
            stats = _get_stream_stats(count_stats, app, ric, fid)
            _print_alert(stream_key, minute, row["pub_count"], anomaly_score, stats)
            alert_count += 1
        else:
            normal_count += 1
            if verbose_normal:
                _print_normal(stream_key, minute, row["pub_count"], anomaly_score)

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print(f"\n{_DSEP}")
    print(f"{'REPLAY COMPLETE':^70}")
    print(_DSEP)
    print(f"  Total windows processed : {total_windows:,}")
    print(f"  Normal windows          : {normal_count:,}")
    print(f"  Anomalies detected      : {alert_count:,}  🚨")
    print(f"  Unknown streams skipped : {unknown_count:,}")
    print(_DSEP)


if __name__ == "__main__":
    monitor_24_7()
