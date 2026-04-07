import argparse
import os
import sys

# Ensure the src directory is in the python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import time

from anomaly_detection.detector import detect_silence_anomalies
from anomaly_detection.generator import generate_mock_data
from anomaly_detection.monitor import monitor_24_7
from anomaly_detection.visualizer import generate_visualizations


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


def run_monitor(speed_factor: float = 500.0, verbose: bool = False):
    monitor_24_7(speed_factor=speed_factor, verbose_normal=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anomaly Detection — Publication Heartbeat Monitor",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Run full training pipeline: Generate data → Train model → Visualize.",
    )
    mode_group.add_argument(
        "--monitor",
        action="store_true",
        help="Start the 24/7 real-time monitor using the trained model.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=500.0,
        metavar="X",
        help=(
            "Time-compression factor for monitor replay.\n"
            "  500  = 1 simulated second plays in 1/500 real second (default)\n"
            "  1    = true real-time replay (very slow for historical data)\n"
            "  5000 = ultra-fast replay"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In monitor mode, also print every normal (healthy) event.",
    )

    args = parser.parse_args()

    if args.monitor:
        run_monitor(speed_factor=args.speed, verbose=args.verbose)
    else:
        # Default behaviour (no flag, or --train): run the training pipeline
        run_pipeline()
