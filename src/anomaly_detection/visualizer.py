import os

import pandas as pd
import plotly.express as px
import polars as pl


def generate_visualizations(
    results_path="data/anomaly_results.parquet", output_dir="reports/visualizations"
):

    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from {results_path}...")
    df = pl.read_parquet(results_path)
    pdf = df.to_pandas()

    # Map the ML status
    pdf["Status"] = pdf["is_anomaly"].map({False: "Normal", True: "Anomaly"})

    # 1. Normal vs Anomaly count per RIC
    print("Generating distribution chart...")
    fig_dist = px.bar(
        pdf.groupby(["RIC", "Status"]).size().reset_index(name="count"),
        x="RIC",
        y="count",
        color="Status",
        title="ML Detected: Normal vs Anomalous 1-min Windows",
        color_discrete_map={"Normal": "steelblue", "Anomaly": "crimson"},
    )
    fig_dist.write_html(os.path.join(output_dir, "value_distribution.html"))

    # 2. Time series of pub_count for a specific stream
    target_ric = "TRI.N"
    target_fid = "LAST"
    subset = pdf[(pdf["RIC"] == target_ric) & (pdf["FID"] == target_fid)].sort_values(
        "minute"
    )

    if len(subset) > 0:
        print(f"Generating time series for {target_ric} {target_fid}...")
        fig_ts = px.bar(
            subset,
            x="minute",
            y="pub_count",
            color="Status",
            title=f"ML Publication Timeline - {target_ric} {target_fid}",
            labels={"pub_count": "Publications per Minute", "minute": "Time"},
            color_discrete_map={"Normal": "steelblue", "Anomaly": "crimson"},
        )
        fig_ts.write_html(os.path.join(output_dir, "timeseries_anomalies.html"))

    print(f"Visualizations saved to {output_dir}/")


if __name__ == "__main__":
    generate_visualizations()
