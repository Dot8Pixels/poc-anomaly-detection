# Anomaly Detection — Publication Heartbeat Monitor

Real-time anomaly detection for financial application publication heartbeats.  
Uses **Unsupervised Machine Learning** (Isolation Forest) to learn the normal publication frequency of every App / RIC / FID stream and alert when a stream goes silent or drops below its expected rate.

---

## 📌 Project Purpose

In real-time financial environments, applications continuously publish market data (RICs and FIDs) to a unified database. If an application stops publishing — or its frequency drops significantly — it indicates a failure that needs immediate attention.

This tool replaces manual thresholds with an ML model that learns exactly what "normal" looks like for **every unique stream**, including its time-of-day shape, day-of-week pattern, and relative publication rate.

---

## 🔍 What Is Monitored

The model focuses on **publication frequency metadata** only — it never looks at data values:

| Feature       | Description                                      |
| ------------- | ------------------------------------------------ |
| `app_number`  | Which application is publishing                  |
| `RIC`         | Reuters Instrument Code (e.g. `TRI.N`, `AAPL.O`) |
| `FID`         | Field ID (e.g. `LAST` price, `BID` quote)        |
| `hour`        | Hour of day (0–23)                               |
| `day_of_week` | Monday–Sunday                                    |
| `pub_count`   | Publications in a 1-minute window (0 = silence)  |

---

## 🧠 ML Model: Isolation Forest

### How It Works

1. **1-Minute Bucket Grid** — Every timestamp is truncated to the minute and counted per `(app, RIC, FID, minute)`. The full timeline is zero-filled so silent minutes appear explicitly as `pub_count = 0`.

2. **Feature Space** — Each minute is a 6-dimensional point: identity (app/RIC/FID encodings) + time context (hour, day-of-week) + frequency (pub_count).

3. **Isolation Logic**
   - *Normal points* cluster tightly — e.g. `TRI.N · LAST · App101` at 09:30 always has ~250 pub/min and is hard to isolate.
   - *Anomalous points* are sparse outliers — a `pub_count = 0` at 10:00 for a normally-busy stream is instantly separable from its cluster.

4. **Per-Stream Profiles** — Because identity is a feature, the model learns that `MSFT.O · BID` publishes at ~48 pub/min while `TRI.N · LAST` publishes at ~250 pub/min. A "low" count is judged relative to that stream's own history.

### Advantages over Rule-Based Thresholds

- **No manual configuration** — no need to define active hours or per-stream thresholds
- **Time-aware** — automatically handles open/close spikes, lunch dips, weekend quiet
- **Scalable** — handles any number of RICs and FIDs without per-instrument tuning

---

## 📊 Per-Stream Publication Profiles (Training Data)

The generator produces realistic, distinct profiles per stream:

| Stream                   | Peak pub/min | Pattern                  |
| ------------------------ | ------------ | ------------------------ |
| `TRI.N · LAST · App101`  | ~250         | High-frequency news feed |
| `TRI.N · BID  · App101`  | ~150         |                          |
| `AAPL.O · LAST · App101` | ~150         | Busy equity              |
| `AAPL.O · BID  · App101` | ~90          |                          |
| `MSFT.O · LAST · App101` | ~80          | Moderate equity          |
| `MSFT.O · BID  · App101` | ~48          |                          |
| App 102 streams          | 75% of above | Secondary/backup feed    |

**Intraday shape** (multiplier on peak rate):

| Time window | Multiplier | Notes                  |
| ----------- | ---------- | ---------------------- |
| 00:00–07:00 | 0.05×      | Near-silent overnight  |
| 07:00–09:00 | 0.15×      | Pre-open trickle       |
| 09:00–10:00 | 1.40×      | **Market open spike**  |
| 10:00–12:00 | 1.00×      | Normal morning         |
| 12:00–13:00 | 0.60×      | Lunch dip              |
| 13:00–16:00 | 1.10×      | Steady afternoon       |
| 16:00–17:00 | 1.30×      | **Market close spike** |
| 17:00–20:00 | 0.20×      | After-hours trickle    |

Weekend volumes are further reduced (Sat 10%, Sun 8% of weekday peak).

---

## 🚀 How to Run

### Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) package manager

### Install dependencies

```powershell
uv sync
```

### Step 1 — Train the model *(required once before anything else)*

```powershell
uv run main.py --train
```

Runs the full pipeline: **Generate → Detect → Visualize**

- Generates 30 days of synthetic publication data (`data/mock_data.parquet`)
- Trains Isolation Forest and saves artifacts to `models/`
- Writes per-stream count stats to `models/gap_stats.joblib`
- Saves detection results to `data/anomaly_results.parquet`
- Produces HTML visualizations in `reports/visualizations/`

### Step 2 — Run the live web dashboard *(recommended)*

```powershell
uv run main.py --web
```

Starts **3 parallel processes** and auto-opens `http://127.0.0.1:8765` in your browser:

| Process   | Role                                                       |
| --------- | ---------------------------------------------------------- |
| Publisher | Generates synthetic live events → `data/live_feed.parquet` |
| Monitor   | Reads live feed, calls model, raises alerts                |
| Dashboard | FastAPI + WebSocket server, streams updates to browser     |

Press **Ctrl-C** to stop all processes cleanly.

**Options:**

```powershell
uv run main.py --web --port 9000          # custom port
uv run main.py --web --stop-after 120     # auto-stop after 120 s (testing)
```

### Other run modes

```powershell
# Terminal-only simulation (no browser)
uv run main.py --simulate

# Historical replay of training data through the model
uv run main.py --monitor
uv run main.py --monitor --speed 1000     # 1000× time compression
uv run main.py --monitor --verbose        # also print healthy events
```

### All CLI flags

| Flag                  | Default       | Description                               |
| --------------------- | ------------- | ----------------------------------------- |
| *(none)* or `--train` | —             | Run full training pipeline                |
| `--simulate`          | —             | Publisher + Monitor (console only)        |
| `--web`               | —             | Publisher + Monitor + Dashboard (browser) |
| `--monitor`           | —             | Historical replay through trained model   |
| `--port PORT`         | `8765`        | Dashboard port (used with `--web`)        |
| `--stop-after SECS`   | `0` (forever) | Auto-stop after N seconds                 |
| `--speed X`           | `500`         | Time-compression for `--monitor` replay   |
| `--verbose`           | off           | Print healthy events in monitor mode      |

---

## 🌐 Web Dashboard

The dashboard is a single-page app with a dark theme served at `http://localhost:8765`.

### Left panel — Stream Grid

One card per stream (12 total: 2 apps × 3 RICs × 2 FIDs).  
Each card shows real-time status, last event timestamp, current pub/min, and ML score.

| Card state            | Meaning                              |
| --------------------- | ------------------------------------ |
| 🟢 Green dot           | Normal — publishing as expected      |
| 🟠 Orange dot + banner | Silent — publisher stopped           |
| 🔴 Red dot + pulsing   | Anomaly — ML model flagged low count |

### Right panel — Tabs

| Tab               | Content                                                        |
| ----------------- | -------------------------------------------------------------- |
| **Event Log**     | Live stream of all events, silences, alerts, and check results |
| **Alerts**        | Persistent cards for every anomaly detected this session       |
| **Train Profile** | Per-stream training baseline charts (see below)                |
| **Config**        | Current simulation configuration values                        |

### Train Profile Tab

Select **App / RIC / FID** and click **Load** to compare a stream's training baseline against live behaviour.

**Timeframe toggles** — Daily / Weekly / Monthly — slice the training data window before computing the charts.

| Chart                          | Description                                                                                          |
| ------------------------------ | ---------------------------------------------------------------------------------------------------- |
| **Hourly Publication Profile** | Bar chart: avg pub/min by hour-of-day (0–23) from training data                                      |
| **pub/min Distribution**       | Histogram: count of Normal (green) vs Anomaly (red) minutes per pub/min bucket                       |
| **Live vs Train avg**          | Line chart: live pub/min readings this session vs the trained mean (appears after first check cycle) |

**Stats row**: Avg pub/min · p05 · p01 · Zero-minute %

---

## 💾 Model Artifacts

All saved to `models/` after `--train`:

| File                        | Contents                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------- |
| `anomaly_model.joblib`      | Trained `IsolationForest(n_estimators=200, contamination=0.005)`                      |
| `encoder_app_number.joblib` | `LabelEncoder` for app numbers                                                        |
| `encoder_RIC.joblib`        | `LabelEncoder` for RIC strings                                                        |
| `encoder_FID.joblib`        | `LabelEncoder` for FID strings                                                        |
| `gap_stats.joblib`          | Per-stream stats: `mean_count`, `std_count`, `p05_count`, `p01_count`, `zero_minutes` |

---

## 📂 Project Structure

```text
anomaly_detection/
├── main.py                          # CLI entry point (--train / --simulate / --web / --monitor)
├── pyproject.toml
├── config/
│   └── data_config.yaml             # Training + simulation parameters
├── data/
│   ├── mock_data.parquet            # Generated training data (30 days)
│   ├── anomaly_results.parquet      # Labeled detection results
│   └── live_feed.parquet            # Written by publisher during simulation
├── models/
│   ├── anomaly_model.joblib
│   ├── encoder_*.joblib
│   └── gap_stats.joblib
├── reports/
│   ├── anomaly_report.md
│   └── visualizations/
│       ├── timeseries_anomalies.html
│       └── value_distribution.html
└── src/anomaly_detection/
    ├── generator.py                 # Synthetic data generator (per-stream profiles)
    ├── detector.py                  # Isolation Forest training + labelling
    ├── visualizer.py                # Plotly HTML report generation
    ├── monitor.py                   # Live monitor + historical replay
    ├── live_feed.py                 # Live publisher with silence injection
    └── dashboard.py                 # FastAPI + WebSocket web dashboard
```

---

## ⚙️ Configuration (`config/data_config.yaml`)

```yaml
num_days: 30                         # days of training data to generate
app_numbers: [101, 102]

time_settings:
  start_time: "2026-03-08T00:00:00Z" # training window start (midnight)

instruments:
  RICs: ["TRI.N", "AAPL.O", "MSFT.O"]
  FIDs:
    LAST: { min: 100.0, max: 1500.0 }
    BID:  { min:  99.0, max: 1499.0 }

anomalies:
  probability: 0.00003               # silence injection rate during training
  types:
    - name: silence
      duration_minutes: 20

live_feed:
  batch_interval_seconds: 1.0        # how often the publisher writes a batch
  events_per_batch: 30
  silence_probability: 0.005         # random silence injection per batch
  silence_duration_minutes: 5

monitor:
  check_interval_seconds: 5          # how often the monitor evaluates each stream
  alert_after_silent_minutes: 2
```

---

## 🛠️ Tech Stack

| Layer                   | Library                                          |
| ----------------------- | ------------------------------------------------ |
| ML                      | `scikit-learn` — Isolation Forest                |
| Data I/O                | `polars` (fast Parquet), `pandas` (ML interface) |
| Web server              | `fastapi` + `uvicorn`                            |
| Real-time push          | WebSockets (`websockets`)                        |
| Frontend charts         | Chart.js 4 (CDN, no build step)                  |
| Config                  | `pyyaml`                                         |
| Serialisation           | `joblib`                                         |
| Visualisation (reports) | `plotly`, `matplotlib`, `seaborn`                |
