import polars as pl
import pandas as pd
import plotly.express as px
import os

def generate_visualizations(results_path="data/anomaly_results.parquet",
                            output_dir="reports/visualizations"):
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run detector first.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from {results_path}...")
    df = pl.read_parquet(results_path)
    pdf = df.to_pandas()
    pdf['Status'] = pdf['is_silence_anomaly'].map({False: 'Normal', True: 'Silence Anomaly'})
    
    # 1. Publication distribution
    print("Generating distribution boxplot...")
    fig_box = px.box(pdf, x="RIC", y="message_count", color="Status", 
                     title="Publication Frequency by RIC (1-min Windows)")
    fig_box.write_html(os.path.join(output_dir, "value_distribution.html"))

    # 2. Time series for a specific stream
    target_ric = "TRI.N"
    target_fid = "LAST"
    subset = pdf[(pdf['RIC'] == target_ric) & (pdf['FID'] == target_fid)].sort_values('timestamp')
    
    if len(subset) > 0:
        print(f"Generating time series for {target_ric} {target_fid}...")
        fig_ts = px.bar(subset, x="timestamp", y="message_count", color="Status",
                        title=f"Heartbeat: {target_ric} {target_fid}",
                        color_discrete_map={'Normal': 'blue', 'Silence Anomaly': 'red'})
        fig_ts.write_html(os.path.join(output_dir, "timeseries_anomalies.html"))
    
    print(f"Visualizations saved to {output_dir}/")

if __name__ == "__main__":
    generate_visualizations()
