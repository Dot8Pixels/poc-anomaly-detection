import sys

sys.path.insert(0, "src")
import yaml

from anomaly_detection.dashboard import _build_html

cfg = yaml.safe_load(open("config/data_config.yaml"))
instr = cfg.get("instruments", {})
apps = cfg.get("app_numbers", [101])
rics = instr.get("RICs", [])
fids = list(instr.get("FIDs", {}).keys())
streams = [(a, r, f) for a in apps for r in rics for f in fids]
html = _build_html(streams, cfg)

idx = html.find('class="panel-tab"')
print("panel-tab found at index:", idx)
print(html[idx : idx + 500])
print("---")
print("Train Profile present:", "Train Profile" in html)
print("tab-profile present:", 'id="tab-profile"' in html)
