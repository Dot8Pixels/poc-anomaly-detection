"""
Real-Time Simulation Dashboard
================================
FastAPI + WebSocket server that renders a live browser dashboard for
the anomaly-detection simulation.

Architecture
────────────
  Publisher process  ──┐
                        ├──► multiprocessing.Queue ──► Dashboard process
  Monitor  process  ──┘                               │
                                                       ├── /ws  (WebSocket broadcast)
                                                       └── /    (HTML page)

Message types sent over the queue (dicts):
  {"type": "event",   "app": 101, "ric": "TRI.N", "fid": "LAST",
   "value": 123.4, "ts": "15:30:01"}

  {"type": "silence", "app": 101, "ric": "AAPL.O", "fid": "BID",
   "duration_min": 5, "ts": "15:30:02"}

  {"type": "alert",   "app": 101, "ric": "AAPL.O", "fid": "BID",
   "pub_count": 0, "mean_count": 197, "score": -0.42,
   "silent_for": "3m 12s", "ts": "15:30:05"}

  {"type": "check",   "ts": "15:30:05", "healthy": 11, "total": 12}

Usage (via main.py — recommended):
    uv run main.py --web

Standalone (for dev / testing):
    uv run uvicorn src.anomaly_detection.dashboard:app --reload --port 8765
"""

import asyncio
import json
import multiprocessing
import os
from typing import Optional

import joblib
import pandas as pd
import polars as pl
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

CONFIG_PATH = "config/data_config.yaml"

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Anomaly Detection Dashboard")

# These are set during startup inside the running event loop.
_broadcast_q: Optional[asyncio.Queue] = None
_loop: Optional[asyncio.AbstractEventLoop] = None
_connected_clients: list[WebSocket] = []

# The inter-process queue is injected by run_dashboard() before uvicorn starts.
_mp_queue: Optional[multiprocessing.Queue] = None


# ── WebSocket endpoint ────────────────────────────────────────────────────────


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    _connected_clients.append(websocket)
    try:
        while True:
            # Keep connection alive; sending is done by the broadcaster task
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _connected_clients:
            _connected_clients.remove(websocket)


async def _broadcaster():
    """Drain _broadcast_q and fan-out JSON messages to all connected clients."""
    while True:
        if _broadcast_q is None:
            await asyncio.sleep(0.05)
            continue
        try:
            msg = _broadcast_q.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.05)
            continue
        dead = []
        for ws in list(_connected_clients):
            try:
                await ws.send_text(json.dumps(msg))
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in _connected_clients:
                _connected_clients.remove(ws)


async def _poll_mp_queue():
    """Drain the inter-process multiprocessing.Queue into the asyncio queue."""

    def _try_get():
        """Non-blocking attempt to get one message; returns None if empty."""
        if _mp_queue is None:
            return None
        try:
            return _mp_queue.get_nowait()
        except Exception:
            return None

    while True:
        msg = _try_get()
        if msg is not None and _broadcast_q is not None:
            await _broadcast_q.put(msg)
        else:
            await asyncio.sleep(0.05)


@app.on_event("startup")
async def _startup():
    global _broadcast_q, _loop
    _broadcast_q = asyncio.Queue()
    _loop = asyncio.get_running_loop()
    asyncio.create_task(_broadcaster())
    asyncio.create_task(_poll_mp_queue())


# ── HTML page ─────────────────────────────────────────────────────────────────


def _build_html(streams: list[tuple], cfg: dict) -> str:
    """Return the full single-page dashboard HTML."""
    stream_ids = [f"{app}|{ric}|{fid}" for app, ric, fid in streams]
    stream_ids_js = json.dumps(stream_ids)

    apps = sorted(set(a for a, _, _ in streams))
    rics = sorted(set(r for _, r, _ in streams))
    fids = sorted(set(f for _, _, f in streams))

    lf_cfg = cfg.get("live_feed", {})
    mon_cfg = cfg.get("monitor", {})
    batch_s = lf_cfg.get("batch_interval_seconds", 1)
    check_s = mon_cfg.get("check_interval_seconds", 5)
    sil_prob = lf_cfg.get("silence_probability", 0.005) * 100
    sil_dur = lf_cfg.get("silence_duration_minutes", 5)

    # Build stream-card HTML (one per stream, filled by JS)
    cards_html = ""
    for app, ric, fid in streams:
        sid = f"{app}|{ric}|{fid}"
        cards_html += f"""
        <div class="stream-card" id="card-{sid.replace("|", "-")}">
          <div class="card-header">
            <span class="app-badge">App {app}</span>
            <span class="ric-label">{ric}</span>
            <span class="fid-badge">{fid}</span>
          </div>
          <div class="card-body">
            <div class="status-dot normal" id="dot-{sid.replace("|", "-")}"></div>
            <div class="card-stats">
              <div class="stat-row">
                <span class="stat-label">Last event</span>
                <span class="stat-val" id="last-{sid.replace("|", "-")}">—</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">pub/min</span>
                <span class="stat-val" id="cnt-{sid.replace("|", "-")}">—</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">ML score</span>
                <span class="stat-val" id="score-{sid.replace("|", "-")}">—</span>
              </div>
            </div>
          </div>
          <div class="silence-bar" id="sbar-{sid.replace("|", "-")}" style="display:none">
            🔇 SILENT — <span id="sdur-{sid.replace("|", "-")}"></span>
          </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Anomaly Detection — Live Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:       #0d1117;
    --surface:  #161b22;
    --border:   #30363d;
    --text:     #e6edf3;
    --muted:    #8b949e;
    --green:    #3fb950;
    --red:      #f85149;
    --orange:   #d29922;
    --blue:     #58a6ff;
    --purple:   #bc8cff;
    --font:     'Segoe UI', system-ui, sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--font); min-height: 100vh; }}

  /* ── header ── */
  header {{
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 14px 24px; display: flex; align-items: center; gap: 16px;
  }}
  header h1 {{ font-size: 1.15rem; font-weight: 600; color: var(--blue); }}
  .header-meta {{ margin-left: auto; font-size: 0.78rem; color: var(--muted); display: flex; gap: 18px; }}
  .conn-badge {{ padding: 3px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 600; }}
  .conn-badge.online  {{ background: #1a3a1f; color: var(--green); }}
  .conn-badge.offline {{ background: #3a1a1a; color: var(--red); }}

  /* ── layout ── */
  .layout {{ display: grid; grid-template-columns: 1fr 370px; gap: 0; height: calc(100vh - 53px); }}

  /* ── left panel ── */
  .left-panel {{ padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 18px; }}

  /* ── stats bar ── */
  .stats-bar {{
    display: flex; gap: 12px; flex-wrap: wrap;
  }}
  .stat-box {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 18px; min-width: 120px;
  }}
  .stat-box .val {{ font-size: 1.4rem; font-weight: 700; color: var(--blue); }}
  .stat-box .lbl {{ font-size: 0.72rem; color: var(--muted); margin-top: 2px; }}
  .stat-box.red  .val {{ color: var(--red); }}
  .stat-box.green .val {{ color: var(--green); }}
  .stat-box.orange .val {{ color: var(--orange); }}

  /* ── stream grid ── */
  .section-title {{ font-size: 0.78rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }}
  .stream-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 10px;
  }}
  .stream-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden; transition: border-color .3s;
  }}
  .stream-card.anomaly {{ border-color: var(--red); box-shadow: 0 0 12px #f8514930; }}
  .stream-card.silent  {{ border-color: var(--orange); }}
  .card-header {{
    padding: 8px 12px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 6px; font-size: 0.8rem;
  }}
  .app-badge {{
    background: #1f3a5f; color: var(--blue); padding: 2px 7px;
    border-radius: 10px; font-size: 0.68rem; font-weight: 700;
  }}
  .ric-label {{ font-weight: 600; font-size: 0.82rem; }}
  .fid-badge {{
    background: #2a2040; color: var(--purple); padding: 2px 7px;
    border-radius: 10px; font-size: 0.68rem; font-weight: 700; margin-left: auto;
  }}
  .card-body {{ padding: 10px 12px; display: flex; align-items: flex-start; gap: 10px; }}
  .status-dot {{
    width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; margin-top: 4px;
    transition: background .4s;
  }}
  .status-dot.normal  {{ background: var(--green); box-shadow: 0 0 6px var(--green); }}
  .status-dot.anomaly {{ background: var(--red);   box-shadow: 0 0 8px var(--red); animation: pulse 1s infinite; }}
  .status-dot.silent  {{ background: var(--orange); box-shadow: 0 0 6px var(--orange); }}
  @keyframes pulse {{ 0%,100% {{ opacity:1 }} 50% {{ opacity:.4 }} }}
  .card-stats {{ flex: 1; }}
  .stat-row {{ display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 3px; }}
  .stat-label {{ color: var(--muted); }}
  .stat-val {{ color: var(--text); font-weight: 500; }}
  .silence-bar {{
    background: #3a2a00; color: var(--orange); font-size: 0.72rem;
    padding: 5px 12px; text-align: center; font-weight: 600;
  }}

  /* ── right panel ── */
  .right-panel {{
    border-left: 1px solid var(--border); display: flex; flex-direction: column;
    overflow: hidden;
  }}
  .panel-tab {{
    display: flex; flex-wrap: wrap; border-bottom: 1px solid var(--border);
  }}
  .tab-btn {{
    flex: 1; padding: 7px 4px; font-size: 0.72rem; font-weight: 600;
    background: none; border: none; color: var(--muted); cursor: pointer;
    border-bottom: 2px solid transparent; white-space: nowrap;
  }}
  .tab-btn.active {{ color: var(--blue); border-bottom-color: var(--blue); }}
  .panel-body {{ flex: 1; overflow-y: auto; padding: 10px; display: none; }}
  .panel-body.active {{ display: block; }}

  /* ── event log ── */
  .log-entry {{
    font-size: 0.73rem; padding: 4px 6px; border-radius: 4px;
    margin-bottom: 3px; line-height: 1.4; font-family: 'Consolas', monospace;
    border-left: 3px solid transparent;
  }}
  .log-entry.event   {{ border-left-color: var(--green); background: #0d1f0d; }}
  .log-entry.silence {{ border-left-color: var(--orange); background: #1f1a0d; }}
  .log-entry.alert   {{ border-left-color: var(--red); background: #1f0d0d; font-weight: 600; }}
  .log-entry.check   {{ border-left-color: var(--border); color: var(--muted); }}
  .log-ts {{ color: var(--muted); margin-right: 6px; }}

  /* ── alert panel ── */
  .alert-card {{
    background: #1a0d0d; border: 1px solid #5a2020;
    border-radius: 8px; padding: 12px; margin-bottom: 8px;
  }}
  .alert-card h4 {{ color: var(--red); font-size: 0.82rem; margin-bottom: 6px; }}
  .alert-row {{ display: flex; justify-content: space-between; font-size: 0.73rem; margin-bottom: 3px; }}
  .alert-row .lbl {{ color: var(--muted); }}
  .alert-count {{ position: absolute; top: -6px; right: -6px; background: var(--red); color: #fff; font-size: 0.65rem; font-weight: 700; padding: 1px 5px; border-radius: 10px; }}
  .tab-btn {{ position: relative; }}
  #alert-badge {{ display: none; background: var(--red); color: #fff; font-size: 0.65rem; font-weight: 700; padding: 1px 5px; border-radius: 10px; margin-left: 4px; vertical-align: middle; }}

  /* ── config panel ── */
  .cfg-row {{ display: flex; justify-content: space-between; font-size: 0.75rem; padding: 5px 0; border-bottom: 1px solid var(--border); }}
  .cfg-row:last-child {{ border-bottom: none; }}
  .cfg-row .k {{ color: var(--muted); }}
  .cfg-row .v {{ color: var(--text); font-weight: 500; }}

  /* ── profile tab ── */
  .profile-controls {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; align-items: center; }}
  .profile-controls select {{
    background: var(--surface); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 8px; font-size: 0.75rem; flex: 1; min-width: 120px;
  }}
  .profile-controls button {{
    background: var(--blue); color: #0d1117; border: none; border-radius: 6px;
    padding: 5px 14px; font-size: 0.75rem; font-weight: 700; cursor: pointer;
  }}
  .chart-wrap {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px; margin-bottom: 12px; position: relative;
  }}
  .chart-wrap h4 {{
    font-size: 0.75rem; color: var(--muted); margin-bottom: 8px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .06em;
  }}
  .chart-wrap canvas {{ width: 100% !important; }}
  .profile-stats-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 12px;
  }}
  .pstat {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 7px 10px;
  }}
  .pstat .pv {{ font-size: 1.1rem; font-weight: 700; color: var(--blue); }}
  .pstat .pk {{ font-size: 0.68rem; color: var(--muted); margin-top: 1px; }}
  #profile-placeholder {{
    color: var(--muted); font-size: 0.78rem; text-align: center;
    margin-top: 40px;
  }}
  .tf-btn-group {{ display: flex; gap: 6px; margin-bottom: 12px; }}
  .tf-btn {{
    flex: 1; padding: 5px 0; font-size: 0.72rem; font-weight: 600;
    background: var(--surface); color: var(--muted);
    border: 1px solid var(--border); border-radius: 6px; cursor: pointer;
  }}
  .tf-btn.active {{ background: var(--blue); color: #0d1117; border-color: var(--blue); }}

  /* scrollbar */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: var(--surface); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
</style>
</head>
<body>

<header>
  <span style="font-size:1.4rem">📡</span>
  <h1>Anomaly Detection — Live Simulation Dashboard</h1>
  <div class="header-meta">
    <span>{len(apps)} apps &nbsp;·&nbsp; {len(rics)} RICs &nbsp;·&nbsp; {len(fids)} FIDs &nbsp;·&nbsp; {len(streams)} streams</span>
    <span id="clock">—</span>
    <span class="conn-badge offline" id="conn-badge">⬤ Connecting…</span>
  </div>
</header>

<div class="layout">
  <!-- LEFT PANEL -->
  <div class="left-panel">
    <!-- Summary stats -->
    <div class="stats-bar">
      <div class="stat-box green">
        <div class="val" id="stat-healthy">—</div>
        <div class="lbl">Healthy streams</div>
      </div>
      <div class="stat-box red">
        <div class="val" id="stat-alerts">0</div>
        <div class="lbl">Active alerts</div>
      </div>
      <div class="stat-box orange">
        <div class="val" id="stat-silent">0</div>
        <div class="lbl">Silent streams</div>
      </div>
      <div class="stat-box">
        <div class="val" id="stat-events">0</div>
        <div class="lbl">Events received</div>
      </div>
      <div class="stat-box">
        <div class="val" id="stat-checks">0</div>
        <div class="lbl">Model checks</div>
      </div>
    </div>

    <!-- Stream grid -->
    <div class="section-title">Stream Status</div>
    <div class="stream-grid">
      {cards_html}
    </div>
  </div>

  <!-- RIGHT PANEL -->
  <div class="right-panel">
    <div class="panel-tab">
      <button class="tab-btn active" onclick="showTab('log',this)">Event Log</button>
      <button class="tab-btn" onclick="showTab('alerts',this)">
        Alerts <span id="alert-badge">0</span>
      </button>
      <button class="tab-btn" onclick="showTab('profile',this)">Train Profile</button>
      <button class="tab-btn" onclick="showTab('config',this)">Config</button>
    </div>

    <div class="panel-body active" id="tab-log"></div>

    <div class="panel-body" id="tab-alerts">
      <div style="font-size:0.78rem;color:var(--muted);margin-bottom:8px;">Most recent alerts first.</div>
      <div id="alert-list"></div>
    </div>

    <div class="panel-body" id="tab-profile">
      <div class="profile-controls">
        <select id="prof-app"><option value="">App…</option></select>
        <select id="prof-ric"><option value="">RIC…</option></select>
        <select id="prof-fid"><option value="">FID…</option></select>
        <button onclick="loadProfile()">Load</button>
      </div>
      <div id="profile-placeholder">← Select App / RIC / FID and click Load<br>to compare the training profile against live detections.</div>
      <div id="profile-content" style="display:none">
        <div class="tf-btn-group">
          <button class="tf-btn" id="tf-daily"   onclick="setTimeframe('daily'  ,this)">Daily</button>
          <button class="tf-btn" id="tf-weekly"  onclick="setTimeframe('weekly' ,this)">Weekly</button>
          <button class="tf-btn active" id="tf-monthly" onclick="setTimeframe('monthly',this)">Monthly</button>
        </div>
        <div class="profile-stats-grid">
          <div class="pstat"><div class="pv" id="ps-mean">—</div><div class="pk">Avg pub/min (train)</div></div>
          <div class="pstat"><div class="pv" id="ps-p05">—</div><div class="pk">p05 pub/min</div></div>
          <div class="pstat"><div class="pv" id="ps-p01">—</div><div class="pk">p01 pub/min</div></div>
          <div class="pstat"><div class="pv" id="ps-zeros">—</div><div class="pk">Zero-min %</div></div>
        </div>
        <div class="chart-wrap">
          <h4>Hourly Publication Profile (train avg pub/min by hour)</h4>
          <canvas id="chart-hourly" height="160"></canvas>
        </div>
        <div class="chart-wrap">
          <h4>pub/min Distribution (train data)</h4>
          <canvas id="chart-hist" height="140"></canvas>
        </div>
        <div class="chart-wrap" id="live-compare-wrap" style="display:none">
          <h4>Live detections this session vs train avg</h4>
          <canvas id="chart-live" height="140"></canvas>
        </div>
      </div>
    </div>

    <div class="panel-body" id="tab-config">
      <div class="cfg-row"><span class="k">Apps</span><span class="v">{", ".join(str(a) for a in apps)}</span></div>
      <div class="cfg-row"><span class="k">RICs</span><span class="v">{", ".join(rics)}</span></div>
      <div class="cfg-row"><span class="k">FIDs</span><span class="v">{", ".join(fids)}</span></div>
      <div class="cfg-row"><span class="k">Total streams</span><span class="v">{len(streams)}</span></div>
      <div class="cfg-row"><span class="k">Batch interval</span><span class="v">{batch_s}s</span></div>
      <div class="cfg-row"><span class="k">Monitor check interval</span><span class="v">{check_s}s</span></div>
      <div class="cfg-row"><span class="k">Silence probability</span><span class="v">{sil_prob:.2f}% / batch</span></div>
      <div class="cfg-row"><span class="k">Silence duration</span><span class="v">{sil_dur} min</span></div>
    </div>
  </div>
</div>

<script>
const STREAMS = {stream_ids_js};
const MAX_LOG = 300;

// State
const state = {{
  healthy: {len(streams)},
  alerts:  0,
  silent:  0,
  events:  0,
  checks:  0,
  alertCount: 0,
  streamState: {{}}   // sid -> "normal"|"silent"|"anomaly"
}};
STREAMS.forEach(s => state.streamState[s] = "normal");

function slugify(sid) {{ return sid.split("|").join("-"); }}
function cardId(sid)  {{ return "card-"  + slugify(sid); }}
function dotId(sid)   {{ return "dot-"   + slugify(sid); }}
function lastId(sid)  {{ return "last-"  + slugify(sid); }}
function cntId(sid)   {{ return "cnt-"   + slugify(sid); }}
function scoreId(sid) {{ return "score-" + slugify(sid); }}
function sbarId(sid)  {{ return "sbar-"  + slugify(sid); }}
function sdurId(sid)  {{ return "sdur-"  + slugify(sid); }}

function setStreamState(sid, newState) {{
  const old = state.streamState[sid] || "normal";
  if (old !== newState) {{
    if (old === "anomaly") state.alerts  = Math.max(0, state.alerts  - 1);
    if (old === "silent")  state.silent  = Math.max(0, state.silent  - 1);
    if (newState === "anomaly") state.alerts++;
    if (newState === "silent")  state.silent++;
    state.streamState[sid] = newState;
  }}
  const card = document.getElementById(cardId(sid));
  const dot  = document.getElementById(dotId(sid));
  const sbar = document.getElementById(sbarId(sid));
  if (!card) return;
  card.className = "stream-card " + (newState !== "normal" ? newState : "");
  dot.className  = "status-dot " + newState;
  sbar.style.display = newState === "silent" ? "" : "none";
  updateStats();
}}

function updateStats() {{
  const healthy = STREAMS.filter(s => state.streamState[s] === "normal").length;
  document.getElementById("stat-healthy").textContent = healthy;
  document.getElementById("stat-alerts").textContent  = state.alerts;
  document.getElementById("stat-silent").textContent  = state.silent;
  document.getElementById("stat-events").textContent  = state.events;
  document.getElementById("stat-checks").textContent  = state.checks;
}}

function addLog(cls, html) {{
  const el = document.createElement("div");
  el.className = "log-entry " + cls;
  el.innerHTML = html;
  const log = document.getElementById("tab-log");
  log.insertBefore(el, log.firstChild);
  while (log.children.length > MAX_LOG) log.removeChild(log.lastChild);
}}

function addAlertCard(msg) {{
  state.alertCount++;
  const badge = document.getElementById("alert-badge");
  badge.style.display = "inline";
  badge.textContent = state.alertCount;

  const card = document.createElement("div");
  card.className = "alert-card";
  card.innerHTML = `
    <h4>🚨 ${{msg.app}} | ${{msg.ric}} | ${{msg.fid}}</h4>
    <div class="alert-row"><span class="lbl">Time</span><span>${{msg.ts}}</span></div>
    <div class="alert-row"><span class="lbl">pub/min</span><span>${{msg.pub_count}}</span></div>
    <div class="alert-row"><span class="lbl">Expected avg</span><span>${{msg.mean_count}}</span></div>
    <div class="alert-row"><span class="lbl">ML score</span><span>${{msg.score}}</span></div>
    ${{msg.silent_for ? `<div class="alert-row"><span class="lbl">Silent for</span><span>${{msg.silent_for}}</span></div>` : ""}}
    <div class="alert-row"><span class="lbl">Verdict</span><span style="color:var(--red)">${{msg.pub_count==0?"⛔ SILENT":"⚠️ VERY LOW"}}</span></div>
  `;
  const list = document.getElementById("alert-list");
  list.insertBefore(card, list.firstChild);
}}

function showTab(name, btn) {{
  document.querySelectorAll(".panel-body").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("tab-" + name).classList.add("active");
  btn.classList.add("active");
  if (name === "alerts") {{
    const badge = document.getElementById("alert-badge");
    badge.style.display = "none";
  }}
}}

// ── Profile tab ──
const STREAMS_LIST = {stream_ids_js};
let _hourlyChart = null;
let _histChart   = null;
let _liveChart   = null;
let _liveData    = {{}}; // sid -> list of {{ts, pub_count}}
let _currentSid  = null;
let _timeframe   = 'monthly';

function setTimeframe(tf, btn) {{
  _timeframe = tf;
  document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  if (_currentSid) loadProfile();
}}

// Populate dropdowns from known streams
(function initDropdowns() {{
  const appSel = document.getElementById('prof-app');
  const ricSel = document.getElementById('prof-ric');
  const fidSel = document.getElementById('prof-fid');
  const apps = [...new Set(STREAMS_LIST.map(s => s.split('|')[0]))];
  const rics = [...new Set(STREAMS_LIST.map(s => s.split('|')[1]))];
  const fids = [...new Set(STREAMS_LIST.map(s => s.split('|')[2]))];
  apps.forEach(a => appSel.add(new Option('App ' + a, a)));
  rics.forEach(r => ricSel.add(new Option(r, r)));
  fids.forEach(f => fidSel.add(new Option(f, f)));
}})();

function destroyChart(c) {{ if (c) {{ c.destroy(); }} return null; }}

function loadProfile() {{
  const app = document.getElementById('prof-app').value;
  const ric = document.getElementById('prof-ric').value;
  const fid = document.getElementById('prof-fid').value;
  if (!app || !ric || !fid) {{ alert('Please select App, RIC and FID'); return; }}
  _currentSid = app + '|' + ric + '|' + fid;

  fetch('/api/profile?app=' + app + '&ric=' + encodeURIComponent(ric) + '&fid=' + encodeURIComponent(fid) + '&timeframe=' + _timeframe)
    .then(r => r.json())
    .then(d => {{
      if (d.error) {{ alert(d.error); return; }}
      document.getElementById('profile-placeholder').style.display = 'none';
      document.getElementById('profile-content').style.display = '';

      // Stats
      document.getElementById('ps-mean').textContent  = d.mean_count.toFixed(1);
      document.getElementById('ps-p05').textContent   = d.p05_count.toFixed(1);
      document.getElementById('ps-p01').textContent   = d.p01_count.toFixed(1);
      document.getElementById('ps-zeros').textContent = d.zero_pct.toFixed(1) + '%';

      // Hourly chart
      _hourlyChart = destroyChart(_hourlyChart);
      const hCtx = document.getElementById('chart-hourly').getContext('2d');
      _hourlyChart = new Chart(hCtx, {{
        type: 'bar',
        data: {{
          labels: d.hourly.map(h => h.hour + ':00'),
          datasets: [{{
            label: 'avg pub/min',
            data: d.hourly.map(h => h.avg),
            backgroundColor: 'rgba(88,166,255,0.6)',
            borderColor: '#58a6ff',
            borderWidth: 1,
          }}]
        }},
        options: {{
          responsive: true,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }} }} , grid: {{ color: '#30363d' }} }},
            y: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }} }} , grid: {{ color: '#30363d' }} }}
          }}
        }}
      }});

      // Distribution histogram
      _histChart = destroyChart(_histChart);
      const hiCtx = document.getElementById('chart-hist').getContext('2d');
      _histChart = new Chart(hiCtx, {{
        type: 'bar',
        data: {{
          labels: d.hist.map(b => b.bin),
          datasets: [
            {{
              label: 'Normal',
              data: d.hist.map(b => b.normal),
              backgroundColor: 'rgba(63,185,80,0.55)',
              borderColor: '#3fb950', borderWidth: 1,
            }},
            {{
              label: 'Anomaly',
              data: d.hist.map(b => b.anomaly),
              backgroundColor: 'rgba(248,81,73,0.7)',
              borderColor: '#f85149', borderWidth: 1,
            }}
          ]
        }},
        options: {{
          responsive: true,
          plugins: {{ legend: {{ labels: {{ color: '#e6edf3', font: {{ size: 9 }} }} }} }},
          scales: {{
            x: {{ stacked: false, ticks: {{ color: '#8b949e', font: {{ size: 8 }} }} , grid: {{ color: '#30363d' }} }},
            y: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }} }} , grid: {{ color: '#30363d' }} }}
          }}
        }}
      }});

      // Live compare chart (if we already have live data for this sid)
      refreshLiveChart();
    }})
    .catch(e => alert('Error loading profile: ' + e));
}}

function refreshLiveChart() {{
  if (!_currentSid || !_liveData[_currentSid] || _liveData[_currentSid].length === 0) {{
    document.getElementById('live-compare-wrap').style.display = 'none';
    return;
  }}
  document.getElementById('live-compare-wrap').style.display = '';
  const pts = _liveData[_currentSid];
  const meanVal = parseFloat(document.getElementById('ps-mean').textContent) || 0;

  _liveChart = destroyChart(_liveChart);
  const ctx = document.getElementById('chart-live').getContext('2d');
  _liveChart = new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: pts.map(p => p.ts),
      datasets: [
        {{
          label: 'Live pub/min',
          data: pts.map(p => p.pub_count),
          borderColor: '#bc8cff', backgroundColor: 'rgba(188,140,255,0.12)',
          borderWidth: 1.5, pointRadius: 2, tension: 0.3, fill: true,
        }},
        {{
          label: 'Train avg',
          data: pts.map(() => meanVal),
          borderColor: '#58a6ff', borderDash: [4,3],
          borderWidth: 1, pointRadius: 0,
        }}
      ]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ labels: {{ color: '#e6edf3', font: {{ size: 9 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e', font: {{ size: 8 }}, maxTicksLimit: 8 }} , grid: {{ color: '#30363d' }} }},
        y: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }} }} , grid: {{ color: '#30363d' }} }}
      }}
    }}
  }});
}}

// ── WebSocket ──
function connect() {{
  const ws = new WebSocket("ws://" + location.host + "/ws");

  ws.onopen = () => {{
    document.getElementById("conn-badge").textContent = "⬤ Live";
    document.getElementById("conn-badge").className = "conn-badge online";
  }};

  ws.onclose = () => {{
    document.getElementById("conn-badge").textContent = "⬤ Disconnected";
    document.getElementById("conn-badge").className = "conn-badge offline";
    setTimeout(connect, 2000);
  }};

  ws.onmessage = (e) => {{
    const msg = JSON.parse(e.data);
    const sid = msg.app + "|" + msg.ric + "|" + msg.fid;

    if (msg.type === "event") {{
      state.events++;
      document.getElementById(lastId(sid)).textContent = msg.ts;
      if (state.streamState[sid] !== "anomaly") setStreamState(sid, "normal");
      addLog("event",
        `<span class="log-ts">${{msg.ts}}</span>` +
        `<b>App ${{msg.app}}</b> ${{msg.ric}} ${{msg.fid}} ` +
        `<span style="color:var(--green)">${{msg.value}}</span>`
      );

    }} else if (msg.type === "silence") {{
      setStreamState(sid, "silent");
      document.getElementById(sdurId(sid)).textContent = msg.duration_min + " min injected";
      addLog("silence",
        `<span class="log-ts">${{msg.ts}}</span>` +
        `🔇 SILENCE → <b>App ${{msg.app}}</b> ${{msg.ric}} ${{msg.fid}} ` +
        `(~${{msg.duration_min}} min)`
      );

    }} else if (msg.type === "alert") {{
      setStreamState(sid, "anomaly");
      document.getElementById(cntId(sid)).textContent   = msg.pub_count + " / min";
      document.getElementById(scoreId(sid)).textContent = msg.score.toFixed(4);
      addLog("alert",
        `<span class="log-ts">${{msg.ts}}</span>` +
        `🚨 ALERT <b>App ${{msg.app}}</b> ${{msg.ric}} ${{msg.fid}} ` +
        `pub=${{msg.pub_count}} score=${{msg.score.toFixed(3)}}`
      );
      addAlertCard(msg);

    }} else if (msg.type === "check") {{
      state.checks++;
      updateStats();
      // Reset anomaly/silent flags for streams that recovered
      if (msg.recovered) {{
        msg.recovered.forEach(s => {{
          if (state.streamState[s] !== "normal") setStreamState(s, "normal");
        }});
      }}
      // Collect live pub counts for the profile compare chart
      if (msg.stream_counts) {{
        Object.entries(msg.stream_counts).forEach(([sid, cnt]) => {{
          if (!_liveData[sid]) _liveData[sid] = [];
          _liveData[sid].push({{ ts: msg.ts, pub_count: cnt }});
          if (_liveData[sid].length > 120) _liveData[sid].shift();
        }});
        if (_currentSid) refreshLiveChart();
      }}
      addLog("check",
        `<span class="log-ts">${{msg.ts}}</span>` +
        `✅ Check #${{state.checks}}  healthy=${{msg.healthy}}/${{msg.total}}`
      );
    }}
  }};
}}

// clock
setInterval(() => {{
  document.getElementById("clock").textContent =
    new Date().toUTCString().slice(17, 25) + " UTC";
}}, 1000);

connect();
updateStats();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    cfg = yaml.safe_load(open(CONFIG_PATH))
    instr = cfg.get("instruments", {})
    apps = cfg.get("app_numbers", [101])
    rics = instr.get("RICs", [])
    fids = list(instr.get("FIDs", {}).keys())
    streams = [(a, r, f) for a in apps for r in rics for f in fids]
    return HTMLResponse(_build_html(streams, cfg))


# ── Training profile API ──────────────────────────────────────────────────────

_GAP_STATS_PATH = os.path.join("models", "gap_stats.joblib")
_RESULTS_PATH = os.path.join("data", "anomaly_results.parquet")
_HIST_BINS = 20  # number of bins for pub/min distribution


@app.get("/api/profile")
async def api_profile(app: str, ric: str, fid: str, timeframe: str = "monthly"):
    """Return training baseline stats + hourly profile + pub/min histogram for one stream.

    timeframe: 'daily' = last 1 day, 'weekly' = last 7 days, 'monthly' = all data (up to 30 days)
    """
    # ── load models ────────────────────────────────────────────────────────────
    if not os.path.exists(_GAP_STATS_PATH):
        return JSONResponse(
            {"error": f"Gap-stats not found at {_GAP_STATS_PATH}. Run --train first."},
            status_code=404,
        )
    if not os.path.exists(_RESULTS_PATH):
        return JSONResponse(
            {
                "error": f"Anomaly results not found at {_RESULTS_PATH}. Run --train first."
            },
            status_code=404,
        )

    app_n = int(app)
    gap_stats: pd.DataFrame = joblib.load(_GAP_STATS_PATH)

    row = gap_stats[
        (gap_stats["app_number"] == app_n)
        & (gap_stats["RIC"] == ric)
        & (gap_stats["FID"] == fid)
    ]
    if row.empty:
        return JSONResponse(
            {"error": f"No training stats for {app}|{ric}|{fid}"}, status_code=404
        )
    row = row.iloc[0]

    mean_count = float(row["mean_count"])
    p05_count = float(row["p05_count"])
    p01_count = float(row["p01_count"])
    zero_minutes = int(row["zero_minutes"])

    # ── load anomaly results ─────────────────────────────────────────────────
    df = pl.read_parquet(_RESULTS_PATH).filter(
        (pl.col("app_number") == app_n)
        & (pl.col("RIC") == ric)
        & (pl.col("FID") == fid)
    )

    # Apply timeframe filter — slice to last N days of available data
    _TF_DAYS = {"daily": 1, "weekly": 7, "monthly": 30}
    days_back = _TF_DAYS.get(timeframe, 30)
    if len(df) > 0:
        max_ts = df["minute"].max()
        cutoff = max_ts - pl.duration(days=days_back)  # type: ignore[operator]
        df = df.filter(pl.col("minute") >= cutoff)

    # Zero-minute percentage across training data
    total_minutes = len(df)
    zero_pct = (zero_minutes / total_minutes * 100.0) if total_minutes > 0 else 0.0

    # ── hourly profile ────────────────────────────────────────────────────────
    hourly_df = (
        df.group_by("hour").agg(pl.col("pub_count").mean().alias("avg")).sort("hour")
    )
    # Fill all 24 hours (some may be missing in short training windows)
    hourly_map = {
        int(r["hour"]): float(r["avg"]) for r in hourly_df.iter_rows(named=True)
    }
    hourly = [{"hour": h, "avg": round(hourly_map.get(h, 0.0), 1)} for h in range(24)]

    # ── pub/min distribution histogram ───────────────────────────────────────
    counts_normal = df.filter(pl.col("is_anomaly") == False)["pub_count"].to_list()  # noqa: E712
    counts_anomaly = df.filter(pl.col("is_anomaly") == True)["pub_count"].to_list()  # noqa: E712
    all_counts = df["pub_count"].to_list()

    if all_counts:
        min_v = min(all_counts)
        max_v = max(all_counts)
        bin_width = max((max_v - min_v) / _HIST_BINS, 1)

        def _bucket(v: float) -> int:
            return min(int((v - min_v) / bin_width), _HIST_BINS - 1)

        bin_norm = [0] * _HIST_BINS
        bin_anom = [0] * _HIST_BINS
        for v in counts_normal:
            bin_norm[_bucket(v)] += 1
        for v in counts_anomaly:
            bin_anom[_bucket(v)] += 1

        hist = [
            {
                "bin": f"{round(min_v + i * bin_width)}-{round(min_v + (i + 1) * bin_width)}",
                "normal": bin_norm[i],
                "anomaly": bin_anom[i],
            }
            for i in range(_HIST_BINS)
        ]
    else:
        hist = []

    return JSONResponse(
        {
            "mean_count": mean_count,
            "p05_count": p05_count,
            "p01_count": p01_count,
            "zero_minutes": zero_minutes,
            "zero_pct": round(zero_pct, 2),
            "hourly": hourly,
            "hist": hist,
        }
    )


# ── run_dashboard entry point ────────────────────────────────────────────────


def run_dashboard(
    q: multiprocessing.Queue,
    host: str = "127.0.0.1",
    port: int = 8765,
):
    """
    Entry point for the dashboard subprocess.
    Injects the inter-process queue, then launches uvicorn.
    The _poll_mp_queue asyncio task (started in _startup) drains it.
    """
    global _mp_queue
    _mp_queue = q

    print(f"\n  🌐  Dashboard → http://{host}:{port}\n")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="warning",
    )
