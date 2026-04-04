import sys
import os
# Ensure the src directory is in the python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from anomaly_detection.generator import generate_mock_data
from anomaly_detection.detector import detect_silence_anomalies
from anomaly_detection.visualizer import generate_visualizations
import time

def run_pipeline():
    start_time = time.time()
    print("--- Phase 1: Generating Mock Data ---")
    generate_mock_data()

    print("\n--- Phase 2: Detecting Silence Anomalies ---")
    detect_silence_anomalies()

    print("\n--- Phase 3: Generating Visualizations ---")
    generate_visualizations()

    duration = time.time() - start_time
    print(f"\nPipeline completed in {duration:.2f} seconds.")
    print("Check 'reports/' for the anomaly report and visualizations.")

if __name__ == "__main__":
    run_pipeline()
