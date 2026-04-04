# Anomaly Detection for Application Publication (Liveness & Periodicity)

This application is designed to monitor the publication "heartbeat" of multiple apps. It detects anomalies based on the **periodicity** of data publication to ensure that expected data is arriving consistently.

## 📌 Project Purpose
In a real-time environment, applications publish data (RICs and FIDs) to a database. If an application stops publishing or lags significantly, it indicates a failure. This tool establishes a baseline of "normal" publication frequency and alerts when a "silence" or "delayed" period is detected.

## 🔍 Monitoring Criteria
The model ignores data **values** (as they are random/variable) and focuses strictly on the following metadata:
- **App Number**: The unique identifier for the source application.
- **Timestamp**: Used to calculate the gap (period) between publications.
- **RIC (Instrument Name)**: The specific financial instrument being published.
- **FID (Field Name)**: The specific field (e.g., BID, ASK, LAST) being published.

### **Detection Logic**
- **Windowing**: Data is aggregated into **1-minute windows**.
- **Max Gap Analysis**: For every window, the system calculates the longest time gap between two consecutive messages for the same App/RIC/FID.
- **Baseline Learning**: The system calculates the `Median Max Gap` (Normal Period) for each stream during active hours (08:00 - 18:00).
- **Anomaly Trigger**: 
    - **SILENCE**: A window contains 0 messages for an active stream.
    - **DELAYED**: The `Max Gap` in a window exceeds **3x the Normal Period**.

## 🚀 How to Run

### **Prerequisites**
- Python 3.12+
- `uv` (Fast Python package manager)

### **Quick Start**
Run the entire pipeline (Generate -> Detect -> Visualize) with a single command:
```powershell
uv run main.py
```

### **Manual Steps**
If you wish to run components individually:
1. **Configure**: Edit `config/data_config.yaml` to change RICs, Apps, or Anomaly frequency.
2. **Generate Data**: `uv run src/anomaly_detection/generator.py`
3. **Run Detection**: `uv run src/anomaly_detection/detector.py`
4. **Visualize**: `uv run src/anomaly_detection/visualizer.py`

## 📊 How to Read Reports
All results are stored in the `reports/` directory:

### **1. Text Summary (`reports/anomaly_report.md`)**
- Provides a high-level count of how many periodicity violations were found.
- Lists a sample table of anomalies including the **Max Gap** (in seconds) vs. the **Normal Gap**.
- **Status "SILENCE"**: 0 messages arrived in that minute.
- **Status "DELAYED"**: Messages arrived, but there was an unusually long pause between them.

### **2. Interactive Visualizations (`reports/visualizations/`)**
Open these `.html` files in any web browser:
- **`timeseries_anomalies.html`**: A bar chart showing the **Max Gap** per minute.
    - **Blue Bars**: Normal heartbeat.
    - **Red Bars**: Periodicity/Silence anomaly.
    - **Orange Line**: The calculated threshold (3x the normal period).
- **`value_distribution.html`**: A scatter plot showing the relationship between message count and the maximum gap, helping to identify outliers.

## 📂 Project Structure
- `config/`: Contains `data_config.yaml`.
- `data/`: Stores raw and processed `.parquet` files (efficient for millions of rows).
- `reports/`: Contains the Markdown report and Plotly HTML visualizations.
- `src/anomaly_detection/`: The core Python package containing the logic.
- `main.py`: The root orchestrator script.
