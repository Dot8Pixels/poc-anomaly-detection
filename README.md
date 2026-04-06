# Anomaly Detection for Application Publication (Liveness & Periodicity)

This application is designed to monitor the publication "heartbeat" of multiple apps. It uses **Unsupervised Machine Learning** to detect anomalies where data should have been published but was not, based on unique historical patterns for each application.

## 📌 Project Purpose
In real-time financial environments, applications publish data (RICs and FIDs) to a unified database. If an application stops publishing or its frequency drops significantly, it indicates a failure. This tool replaces manual thresholds with an AI model that learns exactly what "normal" looks like for every specific App, RIC, and FID.

## 🔍 Monitoring Criteria
The model focuses strictly on metadata and ignores data **values**:
- **Identity**: App Number, RIC (Instrument Name), FID (Field Name).
- **Time Context**: Hour of day, Minute, and Day of the Week.
- **Heartbeat**: The count of publications per 1-minute window.

## 🧠 Model Detail: Isolation Forest
The system utilizes a **Pure Machine Learning** approach (Isolation Forest) to detect "Missing Data" without hardcoded rules.

### **How it Works**
1.  **Zero-Fill Grid**: The system creates a full timeline of every minute for every App/RIC/FID. If no data arrived, it records a `0`. This "teaches" the model to recognize silence.
2.  **Multi-Dimensional Clustering**: The model maps every minute into a high-dimensional space (Identity + Time + Frequency). 
3.  **The "Isolation" Logic**: 
    - **Normal Points**: Healthy publication minutes form dense "clusters" (e.g., App 101 usually has 130 messages at 10 AM). These are hard to separate from each other.
    - **Anomalous Points**: A `0` count occurring when an app is usually busy is mathematically "isolated" from its normal cluster. The model can separate these points very quickly.
4.  **Behavioral Learning**: Because `App Number` and `RIC` are features, the model learns a unique profile for each stream. It automatically knows that a `0` is normal for a batch app at noon, but highly suspicious for a real-time app.

### **Advantages over Rules**
- **Dynamic Thresholds**: No need to manually define "active hours" or "low counts."
- **Pattern Recognition**: Automatically handles complex schedules (e.g., higher traffic on market open/close).
- **Scalability**: Handles thousands of unique RICs and FIDs without manual configuration.

## 💾 Model Export & Inference
When the detector runs, it saves the learned "brain" to the `models/` directory:
- `anomaly_model.joblib`: The trained Isolation Forest model.
- `encoder_*.joblib`: Label encoders for App, RIC, and FID identities.

Other applications (like a real-time monitor) can load these files to check if live data is healthy without needing to retrain. See `src/anomaly_detection/monitor.py` for a code example.

## 🚀 How to Run

### **Prerequisites**
- Python 3.12+
- `uv` (Fast Python package manager)

### **Quick Start**
Run the entire pipeline (Generate -> Detect -> Visualize) with a single command:
```powershell
uv run main.py
```

## 📊 How to Read Reports
All results are stored in the `reports/` directory:

### **1. ML Summary Report (`reports/anomaly_report.md`)**
- Lists windows that the ML model identified as **Isolated** (Anomalous).
- Displays the **App**, **RIC**, and **Timestamp** of the failure.
- **Reason "Isolated Pattern"**: Means the model expected activity based on historical time/identity clusters, but found a significant deviation (usually silence).

### **2. Interactive Visualizations (`reports/visualizations/`)**
- **`timeseries_anomalies.html`**: A bar chart of publication frequency.
    - **Blue Bars**: Normal clustered behavior.
    - **Red Bars**: ML-detected missing or irregular publication periods.
- **`value_distribution.html`**: A bar chart showing the total count of Normal vs. Missing slots per RIC.

## 📂 Project Structure
- `config/`: Contains `data_config.yaml`.
- `data/`: Stores raw and processed `.parquet` files.
- `reports/`: Contains the Markdown report and Plotly HTML visualizations.
- `src/anomaly_detection/`: The core logic (Generator, Detector, Visualizer).
- `main.py`: The root orchestrator script.
