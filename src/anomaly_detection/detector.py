import polars as pl
import numpy as np
import time
import os

def detect_silence_anomalies(data_path="data/mock_data.parquet", 
                             results_path="data/anomaly_results.parquet",
                             report_path="reports/anomaly_report.md",
                             window_minutes=1):
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run generator first.")
        return

    print(f"Loading data from {data_path}...")
    df = pl.read_parquet(data_path).with_columns([
        pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%.fZ")
    ])
    
    print(f"Aggregating data into {window_minutes}-min windows...")
    aggregated = df.sort("timestamp").group_by_dynamic(
        "timestamp",
        every=f"{window_minutes}m",
        group_by=["app_number", "RIC", "FID"]
    ).agg([
        pl.len().alias("message_count")
    ])

    print("Generating active hour time grid...")
    time_grid = pl.datetime_range(df["timestamp"].min(), df["timestamp"].max(), f"{window_minutes}m", eager=True).alias("timestamp").to_frame()
    time_grid = time_grid.with_columns([pl.col("timestamp").dt.hour().alias("hour")])\
                         .filter((pl.col("hour") >= 8) & (pl.col("hour") <= 18))
    
    unique_streams = df.select(["app_number", "RIC", "FID"]).unique()
    full_grid = time_grid.join(unique_streams, how="cross")
    
    full_results = full_grid.join(aggregated, on=["timestamp", "app_number", "RIC", "FID"], how="left")\
                             .with_columns([pl.col("message_count").fill_null(0)])

    print("Calculating baselines and flagging anomalies...")
    stats = full_results.group_by(["app_number", "RIC", "FID"]).agg([
        pl.col("message_count").mean().alias("mean_count")
    ])
    
    results = full_results.join(stats, on=["app_number", "RIC", "FID"])
    results = results.with_columns([
        ((pl.col("mean_count") > 2.0) & (pl.col("message_count") == 0)).alias("is_silence_anomaly")
    ])
    
    anomalies = results.filter(pl.col("is_silence_anomaly"))
    
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results.write_parquet(results_path)
    
    summary_md = f"""# Liveness Anomaly Report
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Windows Analyzed:** {len(results):,}
- **Anomalies Detected:** {len(anomalies):,}

## Sample Anomalies
| App | RIC | FID | Window Start | Count | Mean |
|-----|-----|-----|--------------|-------|------|
"""
    for row in anomalies.sort("timestamp").head(10).iter_rows(named=True):
        summary_md += f"| {row['app_number']} | {row['RIC']} | {row['FID']} | {row['timestamp']} | {row['message_count']} | {row['mean_count']:.2f} |\n"
        
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(summary_md)
    print(f"Results saved to {results_path} and {report_path}")

if __name__ == "__main__":
    detect_silence_anomalies()
