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
    # Criteria: app_number, timestamp, RIC, FID (Values ignored)
    df = pl.read_parquet(data_path, columns=["app_number", "timestamp", "RIC", "FID"]).with_columns([
        pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%.fZ")
    ]).sort("timestamp")
    
    print(f"Analyzing periodicity in {window_minutes}-min windows...")
    
    # Calculate Gaps between messages to understand 'Periodicity'
    df = df.with_columns(
        pl.col("timestamp").diff().over(["app_number", "RIC", "FID"]).dt.total_seconds().alias("gap_seconds")
    )

    # Aggregate by Window: Track both Count and the Worst (Max) Gap
    aggregated = df.group_by_dynamic(
        "timestamp",
        every=f"{window_minutes}m",
        group_by=["app_number", "RIC", "FID"]
    ).agg([
        pl.len().alias("message_count"),
        pl.col("gap_seconds").max().alias("max_gap_in_window")
    ])

    print("Generating active hour time grid...")
    start_time = df["timestamp"].min().replace(second=0, microsecond=0)
    end_time = df["timestamp"].max().replace(second=0, microsecond=0)
    
    time_grid = pl.datetime_range(start_time, end_time, f"{window_minutes}m", eager=True).alias("timestamp").to_frame()
    time_grid = time_grid.with_columns([pl.col("timestamp").dt.hour().alias("hour")])\
                         .filter((pl.col("hour") >= 8) & (pl.col("hour") <= 18))
    
    unique_streams = df.select(["app_number", "RIC", "FID"]).unique()
    full_grid = time_grid.join(unique_streams, how="cross")
    
    aggregated = aggregated.with_columns(pl.col("timestamp").dt.truncate(f"{window_minutes}m"))
    full_grid = full_grid.with_columns(pl.col("timestamp").dt.truncate(f"{window_minutes}m"))

    full_results = full_grid.join(aggregated, on=["timestamp", "app_number", "RIC", "FID"], how="left")\
                             .with_columns([
                                 pl.col("message_count").fill_null(0),
                                 pl.col("max_gap_in_window").fill_null(float(window_minutes * 60)) # If 0 messages, the gap is the whole window
                             ])

    print("Calculating publication period baselines...")
    stats = full_results.group_by(["app_number", "RIC", "FID"]).agg([
        pl.col("message_count").mean().alias("avg_messages_per_min"),
        pl.col("max_gap_in_window").median().alias("normal_max_gap")
    ])
    
    results = full_results.join(stats, on=["app_number", "RIC", "FID"])
    
    # ANOMALY CRITERIA: 
    # 1. Absolute Silence (count is 0)
    # 2. Period Violation (max gap in window is > 3x the normal max gap)
    results = results.with_columns([
        (
            (pl.col("avg_messages_per_min") > 1.0) & # Only monitor active streams
            (
                (pl.col("message_count") == 0) | 
                (pl.col("max_gap_in_window") > (pl.col("normal_max_gap") * 3))
            )
        ).alias("is_period_anomaly")
    ])
    
    anomalies = results.filter(pl.col("is_period_anomaly"))
    
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results.write_parquet(results_path)
    
    summary_md = f"""# Publication Periodicity Report
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Monitoring Criteria
- **App Number, RIC, FID**: Tracked as unique publication streams.
- **Period Detection**: Monitoring Max Gap between messages.
- **Value Check**: Skipped (Liveness only).

## Summary
- **Total Windows Analyzed:** {len(results):,}
- **Periodicity Anomalies Found:** {len(anomalies):,}

## Sample Period Anomalies (Missing/Delayed Publication)
| App | RIC | FID | Window Start | Max Gap (sec) | Normal Gap (sec) | Status |
|-----|-----|-----|--------------|---------------|------------------|--------|
"""
    for row in anomalies.sort("timestamp").head(15).iter_rows(named=True):
        status = "SILENCE" if row['message_count'] == 0 else "DELAYED"
        summary_md += f"| {row['app_number']} | {row['RIC']} | {row['FID']} | {row['timestamp']} | {row['max_gap_in_window']:.1f} | {row['normal_max_gap']:.1f} | {status} |\n"
        
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(summary_md)
    print(f"Results saved to {results_path} and {report_path}")

if __name__ == "__main__":
    detect_silence_anomalies()
