"""
Live Feed Publisher
===================
Simulates a real-time application that continuously publishes
(App, RIC, FID, value) rows to a shared Parquet file.

Two parallel processes run during simulation:
  • This publisher  → writes to  data/live_feed.parquet
  • The monitor     → reads from data/live_feed.parquet

Silence anomalies are randomly injected per-stream so the monitor
has something real to detect.

Usage (standalone):
    python src/anomaly_detection/live_feed.py

Via main.py (recommended — runs both together):
    uv run main.py --simulate
"""

import multiprocessing
import os
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import polars as pl
import yaml

CONFIG_PATH = "config/data_config.yaml"
_DSEP = "═" * 70
_SEP = "─" * 70


def _load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_live_feed(
    config_path: str = CONFIG_PATH,
    stop_after_seconds: float = 0,
    dashboard_queue: Optional[multiprocessing.Queue] = None,
):
    """
    Continuously emit publication events to the live feed parquet.

    Parameters
    ----------
    config_path         : Path to data_config.yaml.
    stop_after_seconds  : If > 0, stop after this many real seconds.
                          0 = run forever (Ctrl-C to stop).
    """
    cfg = _load_config(config_path)
    lf_cfg = cfg.get("live_feed", {})
    instr = cfg.get("instruments", {})
    app_numbers = cfg.get("app_numbers", [101])

    output_file = lf_cfg.get("output_file", "data/live_feed.parquet")
    batch_interval = float(lf_cfg.get("batch_interval_seconds", 1.0))
    events_per_batch = int(lf_cfg.get("events_per_batch", 20))
    silence_prob = float(lf_cfg.get("silence_probability", 0.005))
    silence_duration_m = int(lf_cfg.get("silence_duration_minutes", 5))

    rics = instr.get("RICs", [])
    fids_cfg = instr.get("FIDs", {})

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Delete stale feed from a previous run
    if os.path.exists(output_file):
        os.remove(output_file)

    # Track per-stream silence windows: (app, ric, fid) -> silence_until datetime
    silence_until: dict[tuple, datetime] = {}

    print(_DSEP)
    print(f"{'📡  LIVE FEED PUBLISHER':^70}")
    print(_DSEP)
    print(f"  Output       : {output_file}")
    print(f"  Apps         : {app_numbers}")
    print(f"  RICs         : {rics}")
    print(f"  FIDs         : {list(fids_cfg.keys())}")
    print(f"  Batch rate   : {events_per_batch} events / {batch_interval}s")
    print(f"  Silence prob : {silence_prob * 100:.2f}% per batch")
    print(f"  Silence len  : {silence_duration_m} min")
    print(_DSEP)
    print("  Publishing... (Ctrl-C to stop)\n")

    all_rows: list[dict] = []
    batch_num = 0
    start_wall = time.monotonic()

    try:
        while True:
            now = datetime.now(tz=timezone.utc)
            batch_num += 1
            new_rows: list[dict] = []
            silenced_this_batch: list[tuple] = []

            for _ in range(events_per_batch):
                app = random.choice(app_numbers)
                ric = random.choice(rics)
                fid = random.choice(list(fids_cfg.keys()))
                key = (app, ric, fid)

                # Check / trigger silence
                if key in silence_until and now < silence_until[key]:
                    continue  # this stream is silenced

                # Random new silence trigger
                if random.random() < silence_prob:
                    until = now + timedelta(minutes=silence_duration_m)
                    silence_until[key] = until
                    label = f"App {app} | {ric} | {fid}"
                    if label not in [lbl for lbl, _ in silenced_this_batch]:
                        silenced_this_batch.append((label, key))
                    if dashboard_queue is not None:
                        try:
                            dashboard_queue.put_nowait(
                                {
                                    "type": "silence",
                                    "app": app,
                                    "ric": ric,
                                    "fid": fid,
                                    "duration_min": silence_duration_m,
                                    "ts": now.strftime("%H:%M:%S"),
                                }
                            )
                        except Exception:
                            pass
                    continue

                f_cfg = fids_cfg[fid]
                row = {
                    "app_number": app,
                    "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                    "RIC": ric,
                    "FID": fid,
                    "value": round(random.uniform(f_cfg["min"], f_cfg["max"]), 2),
                }
                new_rows.append(row)
                if dashboard_queue is not None:
                    try:
                        dashboard_queue.put_nowait(
                            {
                                "type": "event",
                                "app": app,
                                "ric": ric,
                                "fid": fid,
                                "value": row["value"],
                                "ts": now.strftime("%H:%M:%S"),
                            }
                        )
                    except Exception:
                        pass

            if new_rows:
                all_rows.extend(new_rows)
                pl.from_dicts(all_rows).write_parquet(output_file)

            # Console status
            active_silences = sum(1 for v in silence_until.values() if now < v)
            status = (
                f"  [Batch {batch_num:>5}]  {now.strftime('%H:%M:%S')} UTC  "
                f"total={len(all_rows):>7,}  +{len(new_rows):>3}  "
                f"silences_active={active_silences}"
            )
            print(status)

            for label, skey in silenced_this_batch:
                remaining = (silence_until[skey] - now).total_seconds() / 60
                print(
                    f"    🔇 SILENCE INJECTED → {label}  "
                    f"(~{silence_duration_m} min, "
                    f"{remaining:.1f} min remaining)"
                )

            # Stop condition
            if stop_after_seconds > 0:
                if time.monotonic() - start_wall >= stop_after_seconds:
                    print(f"\n  ⏹  Publisher stopped after {stop_after_seconds}s.")
                    break

            time.sleep(batch_interval)

    except KeyboardInterrupt:
        print("\n  ⏹  Publisher stopped by user.")

    print(f"  Total events written: {len(all_rows):,}  →  {output_file}")


if __name__ == "__main__":
    run_live_feed()
