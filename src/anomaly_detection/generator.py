"""
Mock data generator — realistic per-stream publication profiles
===============================================================
Each (app, RIC, FID) stream has its own distinct baseline:

  RIC profiles  (events/minute during peak market hours)
  ────────────────────────────────────────────────────────
  TRI.N   → high-frequency news feed   ~200–300 pub/min
  AAPL.O  → busy equity               ~120–180 pub/min
  MSFT.O  → moderate equity           ~60–100  pub/min

  FID multipliers
  ────────────────────────────────────────────────────────
  LAST    → 1.0×  (primary trade price — most active)
  BID     → 0.6×  (quote update — less frequent)

  App multipliers
  ────────────────────────────────────────────────────────
  App 101 → 1.0×  (primary feed)
  App 102 → 0.75× (backup / secondary feed)

  Time-of-day shape (multiplier applied each hour)
  ────────────────────────────────────────────────────────
  Pre-open  (07–09)  0.15×   trickle
  Open      (09–10)  1.40×   spike at market open
  Morning   (10–12)  1.00×   normal
  Lunch     (12–13)  0.60×   quiet
  Afternoon (13–16)  1.10×   steady
  Close     (16–17)  1.30×   spike at close
  After-hrs (17–20)  0.20×   trickle
  Overnight (20–07)  0.05×   near-silent

  Day-of-week multiplier
  ────────────────────────────────────────────────────────
  Mon  0.90×   Tue–Thu  1.00×   Fri  0.85×
  Sat  0.10×   Sun  0.08×
"""

import os
import random
from datetime import datetime, timedelta, timezone

import polars as pl
import yaml
from tqdm import tqdm


def load_config(config_path="config/data_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── per-stream base rates (pub/minute at peak) ────────────────────────────────
_RIC_BASE = {
    "TRI.N": 250.0,
    "AAPL.O": 150.0,
    "MSFT.O": 80.0,
}
_DEFAULT_RIC_BASE = 100.0

_FID_MULT = {
    "LAST": 1.00,
    "BID": 0.60,
}
_DEFAULT_FID_MULT = 0.80

_APP_MULT = {
    101: 1.00,
    102: 0.75,
}
_DEFAULT_APP_MULT = 1.00

# hour-of-day multiplier (index 0 = midnight)
_HOUR_MULT = [
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,  # 00–06
    0.15,
    0.15,  # 07–08  pre-open
    1.40,  # 09     open
    1.00,
    1.00,  # 10–11
    0.60,  # 12     lunch
    1.10,
    1.10,
    1.10,  # 13–15
    1.30,  # 16     close
    0.20,
    0.20,
    0.20,  # 17–19
    0.05,
    0.05,
    0.05,
    0.05,  # 20–23
]

# day-of-week multiplier (0=Mon … 6=Sun)
_DOW_MULT = [0.90, 1.00, 1.00, 1.00, 0.85, 0.10, 0.08]


def _stream_rate(app: int, ric: str, fid: str, hour: int, dow: int) -> float:
    """Return expected publications-per-minute for one stream at a given time."""
    base = _RIC_BASE.get(ric, _DEFAULT_RIC_BASE)
    base *= _FID_MULT.get(fid, _DEFAULT_FID_MULT)
    base *= _APP_MULT.get(app, _DEFAULT_APP_MULT)
    base *= _HOUR_MULT[hour % 24]
    base *= _DOW_MULT[dow % 7]
    # add ±15% jitter so minute-to-minute counts are not perfectly flat
    base *= random.uniform(0.85, 1.15)
    return max(base, 0.0)


def generate_mock_data(config_path="config/data_config.yaml"):
    config = load_config(config_path)

    num_days = config.get("num_days", 10)
    output_file = config.get("output_file", "data/mock_data.parquet")
    app_numbers = config.get("app_numbers", [101])
    time_settings = config.get("time_settings", {})
    instruments = config.get("instruments", {})
    anomalies_cfg = config.get("anomalies", {})

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    rics = instruments.get("RICs", [])
    fids_config = instruments.get("FIDs", {})
    fids = list(fids_config.keys())

    start_str = time_settings.get("start_time", "2026-04-01T00:00:00Z")
    start_time = datetime.fromisoformat(start_str.replace("Z", "+00:00"))

    # silence state: (app, ric, fid) → silence_until datetime
    silence_until: dict[tuple, datetime] = {}
    for app in app_numbers:
        for ric in rics:
            for fid in fids:
                silence_until[(app, ric, fid)] = start_time

    all_rows: list[dict] = []

    for day in tqdm(range(num_days), desc="Days"):
        day_start = start_time + timedelta(days=day)
        dow = day_start.weekday()  # 0=Mon … 6=Sun

        # Simulate each minute of the day for each stream independently
        for minute_offset in range(24 * 60):
            ts_minute = day_start + timedelta(minutes=minute_offset)
            hour = ts_minute.hour

            for app in app_numbers:
                for ric in rics:
                    for fid in fids:
                        key = (app, ric, fid)

                        # Still in a silence window?
                        if ts_minute < silence_until[key]:
                            continue

                        # Maybe trigger a new silence anomaly
                        if random.random() < anomalies_cfg.get("probability", 0):
                            anom_types = anomalies_cfg.get("types", [])
                            if anom_types:
                                chosen = random.choice(anom_types)
                                if chosen["name"] == "silence":
                                    dur = chosen.get("duration_minutes", 10)
                                    silence_until[key] = ts_minute + timedelta(
                                        minutes=dur
                                    )
                                    continue

                        # Draw pub count from Poisson around expected rate
                        rate = _stream_rate(app, ric, fid, hour, dow)
                        count = int(random.gauss(rate, rate * 0.12 + 1))
                        count = max(count, 0)

                        # Spread `count` events uniformly across the minute
                        f_cfg = fids_config[fid]
                        for _ in range(count):
                            sec_offset = random.uniform(0, 59.9)
                            ts = ts_minute + timedelta(seconds=sec_offset)
                            all_rows.append(
                                {
                                    "app_number": app,
                                    "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[
                                        :-3
                                    ]
                                    + "Z",
                                    "RIC": ric,
                                    "FID": fid,
                                    "value": round(
                                        random.uniform(f_cfg["min"], f_cfg["max"]), 2
                                    ),
                                }
                            )

    print(f"Saving {len(all_rows):,} rows to {output_file}…")
    (
        pl.from_dicts(all_rows)
        .with_columns(pl.col("app_number").cast(pl.Int64))
        .write_parquet(output_file)
    )
    print("Done.")


if __name__ == "__main__":
    generate_mock_data()
