import argparse
import multiprocessing
import os
import sys

# Ensure the src directory is in the python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import time

from anomaly_detection.dashboard import run_dashboard
from anomaly_detection.detector import detect_silence_anomalies
from anomaly_detection.generator import generate_mock_data
from anomaly_detection.live_feed import run_live_feed
from anomaly_detection.monitor import monitor_24_7, monitor_live
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


def _feed_worker(config_path: str, stop_after: float):
    """Subprocess target: runs the live publisher."""
    run_live_feed(config_path=config_path, stop_after_seconds=stop_after)


def _monitor_worker(config_path: str, stop_after: float):
    """Subprocess target: runs the live monitor."""
    monitor_live(config_path=config_path, stop_after_seconds=stop_after)


def _feed_worker_web(config_path: str, stop_after: float, q: multiprocessing.Queue):
    """Subprocess target: publisher that also pushes to dashboard queue."""
    run_live_feed(
        config_path=config_path, stop_after_seconds=stop_after, dashboard_queue=q
    )


def _monitor_worker_web(config_path: str, stop_after: float, q: multiprocessing.Queue):
    """Subprocess target: monitor that also pushes to dashboard queue."""
    monitor_live(
        config_path=config_path, stop_after_seconds=stop_after, dashboard_queue=q
    )


def _dashboard_worker(q: multiprocessing.Queue, host: str, port: int):
    """Subprocess target: FastAPI dashboard server."""
    run_dashboard(q=q, host=host, port=port)


def run_web(
    config_path: str = "config/data_config.yaml",
    stop_after: float = 0,
    host: str = "127.0.0.1",
    port: int = 8765,
):
    """
    Launch publisher + monitor + web dashboard as three parallel processes.
    Open http://localhost:8765 in a browser to see the live dashboard.
    Press Ctrl-C to stop all three.
    """
    import webbrowser

    q: multiprocessing.Queue = multiprocessing.Queue(maxsize=2000)

    print("\n" + "═" * 70)
    print(f"{'🌐  STARTING WEB SIMULATION':^70}")
    print("═" * 70)
    print("  Process 1 → Publisher   (writes data/live_feed.parquet)")
    print("  Process 2 → Monitor     (reads feed, predicts, alerts)")
    print(f"  Process 3 → Dashboard   http://{host}:{port}")
    print("  Press Ctrl-C to stop all processes.")
    print("═" * 70 + "\n")

    p_feed = multiprocessing.Process(
        target=_feed_worker_web,
        args=(config_path, stop_after, q),
        name="Publisher",
    )
    p_monitor = multiprocessing.Process(
        target=_monitor_worker_web,
        args=(config_path, stop_after, q),
        name="Monitor",
    )
    p_dashboard = multiprocessing.Process(
        target=_dashboard_worker,
        args=(q, host, port),
        name="Dashboard",
    )

    p_dashboard.start()
    time.sleep(1.5)  # Let uvicorn start before opening browser
    p_feed.start()
    p_monitor.start()

    webbrowser.open(f"http://{host}:{port}")

    try:
        p_feed.join()
        p_monitor.join()
        p_dashboard.join()
    except KeyboardInterrupt:
        print("\n\n⏹  Ctrl-C received — stopping all processes...")
        for p in (p_feed, p_monitor, p_dashboard):
            p.terminate()
            p.join()
        print("  All processes stopped cleanly.\n")


def run_simulate(config_path: str = "config/data_config.yaml", stop_after: float = 0):
    """
    Launch the live publisher and live monitor as two parallel processes.
    Press Ctrl-C to stop both.

    Process 1 — Publisher  (live_feed.py):
        Generates synthetic events and writes them to data/live_feed.parquet.

    Process 2 — Monitor  (monitor.py):
        Reads the live feed, calls the trained model, and prints alerts.
    """
    print("═" * 70)
    print(f"{'🚀  STARTING SIMULATION':^70}")
    print("═" * 70)
    print("  Process 1 → Publisher  (writes data/live_feed.parquet)")
    print("  Process 2 → Monitor    (reads feed, predicts, alerts)")
    print("  Press Ctrl-C to stop both processes.")
    print("═" * 70 + "\n")

    p_feed = multiprocessing.Process(
        target=_feed_worker, args=(config_path, stop_after), name="Publisher"
    )
    p_monitor = multiprocessing.Process(
        target=_monitor_worker, args=(config_path, stop_after), name="Monitor"
    )

    p_feed.start()
    p_monitor.start()

    try:
        p_feed.join()
        p_monitor.join()
    except KeyboardInterrupt:
        print("\n\n⏹  Ctrl-C received — stopping both processes...")
        p_feed.terminate()
        p_monitor.terminate()
        p_feed.join()
        p_monitor.join()
        print("  Both processes stopped cleanly.\n")


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
    mode_group.add_argument(
        "--simulate",
        action="store_true",
        help=(
            "Run a real-time simulation:\n"
            "  - Publisher process writes events to data/live_feed.parquet\n"
            "  - Monitor process reads the feed and prints anomaly alerts\n"
            "  Press Ctrl-C to stop both processes."
        ),
    )
    mode_group.add_argument(
        "--web",
        action="store_true",
        help=(
            "Run the simulation with a live browser dashboard:\n"
            "  - Publisher + Monitor (same as --simulate)\n"
            "  - FastAPI dashboard auto-opens at http://localhost:8765\n"
            "  - Real-time stream grid, event log, and anomaly alert panel\n"
            "  Press Ctrl-C to stop all processes."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        metavar="PORT",
        help="Dashboard port (default: 8765). Only used with --web.",
    )
    parser.add_argument(
        "--stop-after",
        type=float,
        default=0,
        metavar="SECS",
        help=(
            "Auto-stop simulate/monitor after this many seconds (0 = run forever).\n"
            "Useful for automated testing."
        ),
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

    if args.web:
        run_web(stop_after=args.stop_after, port=args.port)
    elif args.simulate:
        run_simulate(stop_after=args.stop_after)
    elif args.monitor:
        run_monitor(speed_factor=args.speed, verbose=args.verbose)
    else:
        # Default behaviour (no flag, or --train): run the training pipeline
        run_pipeline()
