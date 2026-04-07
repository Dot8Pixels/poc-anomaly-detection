"""
24/7 Real-Time Publication Monitor
====================================
Reads the live feed parquet (written by live_feed.py) and checks
every N seconds whether each known App/RIC/FID stream has published
within the last minute.

If a stream that *should* be active goes silent, the trained
Isolation Forest model is asked: "Is a zero-count this unusual?"
If yes → alert with full details.

Two run modes
─────────────
1. Live simulation (default):
       reads  data/live_feed.parquet  (written in real-time by live_feed.py)

2. Historical replay  (--monitor flag via main.py):
       reads  data/mock_data.parquet  and replays bucket-by-bucket

Usage:
    uv run main.py --simulate          # live mode (recommended)
    uv run main.py --monitor           # historical replay
    python src/anomaly_detection/monitor.py   # live mode standalone
"""

import multiprocessing
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import joblib
import pandas as pd
import polars as pl
import yaml

CONFIG_PATH = "config/data_config.yaml"
_DSEP = "═" * 70
_SEP = "─" * 70


# ── helpers ──────────────────────────────────────────────────────────────────


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── artifact loader ───────────────────────────────────────────────────────────


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
            "Run the training pipeline first:  uv run main.py --train"
        )
    model = joblib.load(os.path.join(model_dir, "anomaly_model.joblib"))
    encoders = {
        "app_number": joblib.load(os.path.join(model_dir, "encoder_app_number.joblib")),
        "RIC": joblib.load(os.path.join(model_dir, "encoder_RIC.joblib")),
        "FID": joblib.load(os.path.join(model_dir, "encoder_FID.joblib")),
    }
    count_stats = joblib.load(os.path.join(model_dir, "gap_stats.joblib"))
    return model, encoders, count_stats


# ── stats lookup ──────────────────────────────────────────────────────────────


def _stream_stats(count_stats: pd.DataFrame, app, ric: str, fid: str) -> dict:
    row = count_stats[
        (count_stats["app_number"] == app)
        & (count_stats["RIC"] == ric)
        & (count_stats["FID"] == fid)
    ]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "mean_count": float(r.get("mean_count", 0)),
        "p05_count": float(r.get("p05_count", 0)),
        "p01_count": float(r.get("p01_count", 0)),
        "zero_minutes": int(r.get("zero_minutes", 0)),
    }


# ── predict one window ────────────────────────────────────────────────────────

FEATURES = ["app_number_enc", "RIC_enc", "FID_enc", "hour", "day_of_week", "pub_count"]


def _predict(
    model, encoders, app, ric: str, fid: str, pub_count: int, check_time: datetime
):
    """
    Returns (is_anomaly: bool, score: float) or raises ValueError for unknown streams.
    """
    X = pd.DataFrame(
        [
            {
                "app_number_enc": encoders["app_number"].transform([app])[0],
                "RIC_enc": encoders["RIC"].transform([ric])[0],
                "FID_enc": encoders["FID"].transform([fid])[0],
                "hour": check_time.hour,
                "day_of_week": check_time.weekday(),
                "pub_count": pub_count,
            }
        ],
        columns=FEATURES,
    )
    pred = model.predict(X)[0]
    score = model.score_samples(X)[0]
    return pred == -1, score


# ── alert printer ─────────────────────────────────────────────────────────────


def _alert(
    app,
    ric: str,
    fid: str,
    pub_count: int,
    window_start: datetime,
    last_seen: datetime | None,
    score: float,
    stats: dict,
):
    now_str = _ts(datetime.now(tz=timezone.utc))
    window_str = window_start.strftime("%H:%M UTC")
    silence_sec = (
        (datetime.now(tz=timezone.utc) - last_seen).total_seconds()
        if last_seen
        else None
    )

    print(f"\n{'🚨  ANOMALY ALERT':^70}")
    print(_SEP)
    print(f"  Detected at  : {now_str}")
    print(f"  Stream       : App {app}  |  RIC: {ric}  |  FID: {fid}")
    print(f"  Window       : {window_str}  (last completed minute)")
    print(
        f"  Publications : {pub_count}  {'← SILENT' if pub_count == 0 else '← VERY LOW'}"
    )
    if last_seen:
        print(f"  Last seen    : {_ts(last_seen)}")
        if silence_sec is not None:
            m, s = divmod(int(silence_sec), 60)
            print(f"  Silent for   : {m}m {s}s")
    if stats:
        print(
            f"  Normal range : avg {stats['mean_count']:.0f}/min  "
            f"| p05 {stats['p05_count']:.0f}/min  "
            f"| p01 {stats['p01_count']:.0f}/min"
        )
    print(f"  ML score     : {score:.4f}  (lower = more anomalous)")
    verdict = (
        "⛔  NO PUBLICATIONS — stream has gone SILENT"
        if pub_count == 0
        else "⚠️   VERY LOW COUNT — far below expected frequency"
    )
    print(f"  Verdict      : {verdict}")
    print(_SEP)


def _ok(app, ric: str, fid: str, pub_count: int, window_start: datetime, score: float):
    print(
        f"  ✅  {window_start.strftime('%H:%M')}  "
        f"App {app} | {ric:<8} | {fid:<5}  "
        f"{pub_count:>5} pub/min   score={score:.4f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# LIVE MONITOR  (reads live_feed.parquet, checks every N seconds)
# ══════════════════════════════════════════════════════════════════════════════


def monitor_live(
    data_path: str = "data/live_feed.parquet",
    model_dir: str = "models",
    config_path: str = CONFIG_PATH,
    verbose: bool = False,
    stop_after_seconds: float = 0,
    dashboard_queue: Optional[multiprocessing.Queue] = None,
):
    """
    Tail the live feed parquet and alert when a stream goes silent.

    Logic per check cycle
    ─────────────────────
    1. Read the entire live_feed.parquet.
    2. Find the last *completed* 1-minute window
       (current_minute - 1, so the window is fully closed).
    3. For every expected stream (from config), count events in that window.
    4. Run the ML model on that (stream, hour, count) tuple.
    5. Alert if anomaly.
    """
    cfg = _load_config(config_path)
    mon_cfg = cfg.get("monitor", {})
    lf_cfg = cfg.get("live_feed", {})
    instr = cfg.get("instruments", {})
    app_numbers = cfg.get("app_numbers", [101])

    check_interval = float(mon_cfg.get("check_interval_seconds", 5))
    alert_after_silent = int(mon_cfg.get("alert_after_silent_minutes", 2))
    rics = instr.get("RICs", [])
    fids = list(instr.get("FIDs", {}).keys())

    # All expected streams
    all_streams = [
        (app, ric, fid) for app in app_numbers for ric in rics for fid in fids
    ]

    print(_DSEP)
    print(f"{'🔍  LIVE PUBLICATION MONITOR':^70}")
    print(_DSEP)
    print(f"  Feed file      : {data_path}")
    print(f"  Model dir      : {model_dir}")
    print(f"  Check interval : {check_interval}s")
    print(f"  Alert after    : {alert_after_silent} silent min")
    print(
        f"  Watching       : {len(all_streams)} streams  "
        f"({len(app_numbers)} apps × {len(rics)} RICs × {len(fids)} FIDs)"
    )
    print(_DSEP)

    # Load model
    print("\nLoading model artifacts...")
    try:
        model, encoders, count_stats = _load_artifacts(model_dir)
    except FileNotFoundError as e:
        print(f"\n❌  {e}")
        return
    print("  ✔  Model loaded.\n")

    # last_seen[(app, ric, fid)] = last datetime we actually saw an event
    last_seen: dict[tuple, datetime] = {}
    # already_alerted[(app, ric, fid, window)] = True  → suppress duplicate alerts
    already_alerted: set[tuple] = set()

    alert_total = 0
    check_count = 0
    start_wall = time.monotonic()

    print(f"  Waiting for feed file: {data_path}")
    print(f"  {'─' * 66}")

    try:
        while True:
            # Wait for the publisher to create the file
            if not os.path.exists(data_path):
                print(
                    f"  ⏳  [{datetime.now(tz=timezone.utc).strftime('%H:%M:%S')}]"
                    f"  Feed not yet available..."
                )
                time.sleep(check_interval)
                continue

            now = datetime.now(tz=timezone.utc)
            # Last completed 1-minute window
            cur_minute = now.replace(second=0, microsecond=0)
            check_window = cur_minute - timedelta(minutes=1)

            check_count += 1
            print(
                f"\n  [{now.strftime('%H:%M:%S')} UTC]  "
                f"Check #{check_count}  —  window: "
                f"{check_window.strftime('%H:%M')}–{cur_minute.strftime('%H:%M')}"
            )

            # Read the live feed
            try:
                df = pl.read_parquet(data_path).with_columns(
                    pl.col("timestamp").str.to_datetime(
                        "%Y-%m-%dT%H:%M:%S%.fZ", time_unit="us", time_zone="UTC"
                    )
                )
            except Exception as e:
                print(f"  ⚠️   Could not read feed: {e}")
                time.sleep(check_interval)
                continue

            # Update last_seen from entire history
            latest = df.group_by(["app_number", "RIC", "FID"]).agg(
                pl.col("timestamp").max().alias("last_ts")
            )
            for r in latest.iter_rows(named=True):
                key = (r["app_number"], r["RIC"], r["FID"])
                ts = r["last_ts"]
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if key not in last_seen or ts > last_seen[key]:
                    last_seen[key] = ts

            # Count events in the closed window
            window_df = df.filter(
                (pl.col("timestamp") >= check_window)
                & (pl.col("timestamp") < cur_minute)
            )
            window_counts: dict[tuple, int] = {}
            for r in (
                window_df.group_by(["app_number", "RIC", "FID"])
                .agg(pl.len().alias("cnt"))
                .iter_rows(named=True)
            ):
                window_counts[(r["app_number"], r["RIC"], r["FID"])] = r["cnt"]

            # Evaluate every expected stream
            alerts_this_cycle = 0
            for stream in all_streams:
                app, ric, fid = stream
                pub_count = window_counts.get(stream, 0)

                # Skip streams the model never saw
                try:
                    is_anomaly, score = _predict(
                        model, encoders, app, ric, fid, pub_count, check_window
                    )
                except ValueError:
                    print(f"  ⚠️   Unknown stream {stream} — needs retraining")
                    continue

                alert_key = (*stream, check_window)

                if is_anomaly and alert_key not in already_alerted:
                    stats = _stream_stats(count_stats, app, ric, fid)
                    _alert(
                        app,
                        ric,
                        fid,
                        pub_count,
                        check_window,
                        last_seen.get(stream),
                        score,
                        stats,
                    )
                    already_alerted.add(alert_key)
                    alert_total += 1
                    alerts_this_cycle += 1
                    if dashboard_queue is not None:
                        ls = last_seen.get(stream)
                        silent_for = None
                        if ls:
                            diff = int(
                                (datetime.now(tz=timezone.utc) - ls).total_seconds()
                            )
                            m, s = divmod(diff, 60)
                            silent_for = f"{m}m {s}s"
                        try:
                            dashboard_queue.put_nowait(
                                {
                                    "type": "alert",
                                    "app": app,
                                    "ric": ric,
                                    "fid": fid,
                                    "pub_count": pub_count,
                                    "mean_count": round(stats.get("mean_count", 0)),
                                    "score": round(score, 4),
                                    "silent_for": silent_for,
                                    "ts": check_window.strftime("%H:%M UTC"),
                                }
                            )
                        except Exception:
                            pass
                elif verbose or pub_count > 0:
                    _ok(app, ric, fid, pub_count, check_window, score)

            if alerts_this_cycle == 0:
                healthy = sum(1 for s in all_streams if window_counts.get(s, 0) > 0)
                print(
                    f"  ✅  All {healthy}/{len(all_streams)} active streams healthy "
                    f"this window."
                )

            # Push check summary to dashboard
            if dashboard_queue is not None:
                healthy = sum(1 for s in all_streams if window_counts.get(s, 0) > 0)
                # streams that recovered this cycle (had alert before but OK now)
                recovered = [
                    f"{a}|{r}|{f}"
                    for a, r, f in all_streams
                    if window_counts.get((a, r, f), 0) > 0
                ]
                try:
                    dashboard_queue.put_nowait(
                        {
                            "type": "check",
                            "healthy": healthy,
                            "total": len(all_streams),
                            "recovered": recovered,
                            "ts": datetime.now(tz=timezone.utc).strftime("%H:%M:%S"),
                        }
                    )
                except Exception:
                    pass

            # Stop condition
            if stop_after_seconds > 0:
                if time.monotonic() - start_wall >= stop_after_seconds:
                    print(f"\n  ⏹  Monitor stopped after {stop_after_seconds}s.")
                    break

            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n  ⏹  Monitor stopped by user.")

    print(f"\n{_DSEP}")
    print(f"{'MONITOR SUMMARY':^70}")
    print(_DSEP)
    print(f"  Total checks    : {check_count}")
    print(f"  Total alerts    : {alert_total}  🚨")
    print(_DSEP)


# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL REPLAY MONITOR  (used by  uv run main.py --monitor)
# ══════════════════════════════════════════════════════════════════════════════


def monitor_24_7(
    data_path: str = "data/mock_data.parquet",
    model_dir: str = "models",
    speed_factor: float = 500.0,
    verbose_normal: bool = False,
):
    """
    Replay historical parquet data minute-by-minute and predict anomalies.
    Used for offline verification of the trained model.
    For the real simulation, use monitor_live() instead.
    """
    print(_DSEP)
    print(f"{'📼  HISTORICAL REPLAY MONITOR':^70}")
    print(_DSEP)
    print(f"  Data source  : {data_path}")
    print(f"  Model dir    : {model_dir}")
    print(
        f"  Speed factor : {speed_factor}x  "
        f"(1 sim-minute = {60 / speed_factor:.3f} real-sec)"
    )
    print(_DSEP)

    print("\nLoading model artifacts...")
    try:
        model, encoders, count_stats = _load_artifacts(model_dir)
    except FileNotFoundError as e:
        print(f"\n❌  {e}")
        return
    print("  ✔  Model, encoders and count statistics loaded.")

    if not os.path.exists(data_path):
        print(f"\n❌  Data file not found: {data_path}")
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

    counts = (
        df.group_by(["app_number", "RIC", "FID", "minute"])
        .agg(pl.len().alias("pub_count"))
        .sort("minute")
    )

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
    print(f"  ✔  {total_windows:,} minute-windows ready. Starting replay...\n")

    alert_count = 0
    normal_count = 0
    unknown_count = 0
    prev_minute = None

    print("─" * 70)
    print(f"{'REPLAY FEED':^70}")
    print("─" * 70 + "\n")

    for row in grid.iter_rows(named=True):
        app = row["app_number"]
        ric = row["RIC"]
        fid = row["FID"]
        minute = row["minute"]
        if minute.tzinfo is None:
            minute = minute.replace(tzinfo=timezone.utc)

        if prev_minute is not None:
            sim_delta = (minute - prev_minute).total_seconds()
            real_sleep = sim_delta / speed_factor
            if real_sleep > 0:
                time.sleep(real_sleep)
        prev_minute = minute

        try:
            is_anomaly, score = _predict(
                model, encoders, app, ric, fid, row["pub_count"], minute
            )
        except ValueError:
            unknown_count += 1
            continue

        if is_anomaly:
            stats = _stream_stats(count_stats, app, ric, fid)
            _alert(app, ric, fid, row["pub_count"], minute, None, score, stats)
            alert_count += 1
        else:
            normal_count += 1
            if verbose_normal:
                _ok(app, ric, fid, row["pub_count"], minute, score)

    print(f"\n{_DSEP}")
    print(f"{'REPLAY COMPLETE':^70}")
    print(_DSEP)
    print(f"  Total windows : {total_windows:,}")
    print(f"  Normal        : {normal_count:,}")
    print(f"  Anomalies     : {alert_count:,}  🚨")
    print(f"  Skipped       : {unknown_count:,}")
    print(_DSEP)


if __name__ == "__main__":
    monitor_live()
