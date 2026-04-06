import polars as pl
import numpy as np
import time
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

def detect_silence_anomalies(data_path="data/mock_data.parquet", 
                             results_path="data/anomaly_results.parquet",
                             report_path="reports/anomaly_report.md",
                             model_dir="models",
                             window_minutes=1):
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading data from {data_path}...")
    df = pl.read_parquet(data_path, columns=["app_number", "timestamp", "RIC", "FID"]).with_columns([
        pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%.fZ")
    ])
    
    print("Generating comprehensive timeline grid...")
    start_time = df["timestamp"].min().replace(second=0, microsecond=0)
    end_time = df["timestamp"].max().replace(second=0, microsecond=0)
    time_grid = pl.datetime_range(start_time, end_time, f"{window_minutes}m", eager=True).alias("timestamp").to_frame()
    unique_streams = df.select(["app_number", "RIC", "FID"]).unique()
    full_grid = time_grid.join(unique_streams, how="cross")

    print(f"Aggregating actual publication heartbeat...")
    actual_agg = df.sort("timestamp").group_by_dynamic(
        "timestamp",
        every=f"{window_minutes}m",
        group_by=["app_number", "RIC", "FID"]
    ).agg([
        pl.len().alias("message_count")
    ]).with_columns(pl.col("timestamp").dt.truncate(f"{window_minutes}m"))

    dataset = full_grid.join(actual_agg, on=["timestamp", "app_number", "RIC", "FID"], how="left")\
                       .with_columns([
                           pl.col("message_count").fill_null(0)
                       ])

    print("Engineering features for ML...")
    dataset = dataset.with_columns([
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.minute().alias("minute"),
        pl.col("timestamp").dt.weekday().alias("day_of_week")
    ])

    pdf = dataset.to_pandas()
    
    # 1. ENCODE AND SAVE ENCODERS
    print("Encoding categories and saving artifacts...")
    os.makedirs(model_dir, exist_ok=True)
    
    encoders = {}
    for col in ['app_number', 'RIC', 'FID']:
        le = LabelEncoder()
        pdf[f'{col}_enc'] = le.fit_transform(pdf[col])
        encoders[col] = le
        # Save each encoder so other apps can map RIC names to the same numbers
        joblib.dump(le, os.path.join(model_dir, f"encoder_{col}.joblib"))

    # 2. TRAIN AND SAVE MODEL
    print("Training Isolation Forest...")
    features = ['app_number_enc', 'RIC_enc', 'FID_enc', 'hour', 'minute', 'day_of_week', 'message_count']
    X = pdf[features]

    model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    pdf['ml_score'] = model.fit_predict(X)
    pdf['is_anomaly'] = pdf['ml_score'] == -1
    
    # Save the model
    joblib.dump(model, os.path.join(model_dir, "anomaly_model.joblib"))
    print(f"Model and Encoders saved to {model_dir}/")

    results = pl.from_pandas(pdf)
    anomalies = results.filter(pl.col("is_anomaly"))
    
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results.write_parquet(results_path)
    
    summary_md = f"""# ML-Based Publication Anomaly Report
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Machine Learning Artifacts
- **Model Path**: `{model_dir}/anomaly_model.joblib`
- **Encoders**: Saved for App, RIC, and FID identity mapping.

## Summary
- **Total Windows Modeled:** {len(results):,}
- **ML Detected Anomalies:** {len(anomalies):,}

## Top ML Anomalies
| App | RIC | FID | Timestamp | Count |
|-----|-----|-----|-----------|-------|
"""
    for row in anomalies.sort("message_count").head(20).iter_rows(named=True):
        summary_md += f"| {row['app_number']} | {row['RIC']} | {row['FID']} | {row['timestamp']} | {row['message_count']} |\n"
        
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(summary_md)
    print(f"ML Detection complete. Results: {results_path}")

if __name__ == "__main__":
    detect_silence_anomalies()
