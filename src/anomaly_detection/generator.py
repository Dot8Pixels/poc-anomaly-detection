import polars as pl
import yaml
import random
from datetime import datetime, timedelta
from tqdm import tqdm
import os

def load_config(config_path="config/data_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_mock_data(config_path="config/data_config.yaml"):
    config = load_config(config_path)
    
    num_days = config.get("num_days", 1)
    events_per_day = config.get("events_per_day", 100)
    output_file = config.get("output_file", "data/mock_data.parquet")
    app_numbers = config.get("app_numbers", [101])
    time_settings = config.get("time_settings", {})
    instruments = config.get("instruments", {})
    anomalies_cfg = config.get("anomalies", {})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    rics = instruments.get("RICs", [])
    fids_config = instruments.get("FIDs", {})
    
    start_time_base_str = time_settings.get("start_time", "2026-04-01T08:00:00Z")
    start_time_base = datetime.fromisoformat(start_time_base_str.replace("Z", "+00:00"))
    min_gap = time_settings.get("min_gap_seconds", 0.1)
    max_gap = time_settings.get("max_gap_seconds", 0.5)

    all_rows = []
    app_silence_until = {app: start_time_base for app in app_numbers}

    for day in range(num_days):
        current_day_start = start_time_base + timedelta(days=day)
        current_time = current_day_start
        for app in app_numbers:
            app_silence_until[app] = current_day_start
            
        print(f"Generating data for Day {day + 1}/{num_days}: {current_day_start.strftime('%Y-%m-%d')}")

        for i in tqdm(range(events_per_day), desc=f"Day {day+1}"):
            current_time += timedelta(seconds=random.uniform(min_gap, max_gap))
            timestamp_str = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            app_num = random.choice(app_numbers)
            
            if current_time < app_silence_until[app_num]:
                continue

            is_anomaly_trigger = random.random() < anomalies_cfg.get("probability", 0)
            if is_anomaly_trigger:
                anomaly_type = random.choice(anomalies_cfg.get("types", []))
                if anomaly_type["name"] == "silence":
                    duration = anomaly_type.get("duration_minutes", 10)
                    app_silence_until[app_num] = current_time + timedelta(minutes=duration)
                    continue

            selected_rics = random.sample(rics, k=random.randint(1, min(2, len(rics))))
            for ric in selected_rics:
                selected_fids = random.sample(list(fids_config.keys()), k=random.randint(1, len(fids_config)))
                for fid in selected_fids:
                    f_cfg = fids_config[fid]
                    all_rows.append({
                        "app_number": app_num,
                        "timestamp": timestamp_str,
                        "RIC": ric,
                        "FID": fid,
                        "value": round(random.uniform(f_cfg["min"], f_cfg["max"]), 2)
                    })

    print(f"Saving to {output_file}...")
    pl.from_dicts(all_rows).write_parquet(output_file)
    print(f"Generated {len(all_rows)} rows.")

if __name__ == "__main__":
    generate_mock_data()
