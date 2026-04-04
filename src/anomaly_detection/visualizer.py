import polars as pl
import pandas as pd
import plotly.express as px
import os

def generate_visualizations(results_path="data/anomaly_results.parquet",
                            output_dir="reports/visualizations"):
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from {results_path}...")
    df = pl.read_parquet(results_path)
    pdf = df.to_pandas()
    
    # Map the ML status
    pdf['Status'] = pdf['is_anomaly'].map({False: 'Normal (Found)', True: 'Missing (Expected)'})
    
    # 1. Expected vs Missing Chart (ML Perspective)
    print("Generating distribution chart...")
    fig_dist = px.bar(pdf.groupby(['RIC', 'Status']).size().reset_index(name='count'), 
                      x="RIC", y="count", color="Status", 
                      title="ML Detected: Normal vs Missing Publications",
                      color_discrete_map={'Normal (Found)': 'blue', 'Missing (Expected)': 'red'})
    fig_dist.write_html(os.path.join(output_dir, "value_distribution.html"))

    # 2. Time series for a specific stream
    target_ric = "TRI.N"
    target_fid = "LAST"
    subset = pdf[(pdf['RIC'] == target_ric) & (pdf['FID'] == target_fid)].sort_values('timestamp')
    
    if len(subset) > 0:
        print(f"Generating time series for {target_ric} {target_fid}...")
        fig_ts = px.bar(subset, x="timestamp", y="message_count", color="Status",
                        title=f"ML Publication Timeline - {target_ric} {target_fid}",
                        labels={"message_count": "Messages per Min"},
                        color_discrete_map={'Normal (Found)': 'blue', 'Missing (Expected)': 'red'})
        
        fig_ts.write_html(os.path.join(output_dir, "timeseries_anomalies.html"))
    
    print(f"Visualizations saved to {output_dir}/")

if __name__ == "__main__":
    generate_visualizations()
