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
    pdf['Status'] = pdf['is_period_anomaly'].map({False: 'Normal Heartbeat', True: 'Periodicity Anomaly'})
    
    # 1. Heartbeat Analysis (Max Gap vs Count)
    print("Generating distribution chart...")
    fig_dist = px.scatter(pdf, x="message_count", y="max_gap_in_window", color="Status", 
                          title="Publication Health: Count vs. Max Gap (1-min Windows)",
                          labels={"message_count": "Messages per Min", "max_gap_in_window": "Max Gap between Messages (sec)"})
    fig_dist.write_html(os.path.join(output_dir, "value_distribution.html"))

    # 2. Time series for a specific stream
    target_ric = "TRI.N"
    target_fid = "LAST"
    subset = pdf[(pdf['RIC'] == target_ric) & (pdf['FID'] == target_fid)].sort_values('timestamp')
    
    if len(subset) > 0:
        print(f"Generating time series for {target_ric} {target_fid}...")
        # Show Max Gap over time
        fig_ts = px.bar(subset, x="timestamp", y="max_gap_in_window", color="Status",
                        title=f"Publication Periodicity (Max Gap) - {target_ric} {target_fid}",
                        labels={"max_gap_in_window": "Max Gap in Window (sec)"},
                        color_discrete_map={'Normal Heartbeat': 'blue', 'Periodicity Anomaly': 'red'})
        
        # Add normal baseline
        fig_ts.add_hline(y=subset['normal_max_gap'].iloc[0] * 3, 
                         line_dash="dot", 
                         annotation_text="Anomaly Threshold (3x Median)", 
                         line_color="orange")

        fig_ts.write_html(os.path.join(output_dir, "timeseries_anomalies.html"))
    
    print(f"Visualizations saved to {output_dir}/")

if __name__ == "__main__":
    generate_visualizations()
