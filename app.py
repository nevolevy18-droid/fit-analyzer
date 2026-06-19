"""
🏃 Advanced Running Telemetry Dashboard
========================================
A neutral, maximum-detail analytics dashboard for raw .fit running files.
"""

import io
import math
import hmac
from pathlib import Path
import fitparse 
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fitparse import FitFile

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Running Telemetry Dashboard", page_icon="🏃", layout="wide")

SEMICIRCLE_TO_DEG = 180.0 / (2 ** 31)
RUNS_DIR = Path(__file__).resolve().parent / "runs"

CURATED_METRICS = [
    ("pace_min_km", "Pace", "min/km", "#d62728", True),
    ("heart_rate", "Heart Rate", "bpm", "#e377c2", False),
    ("cadence_spm", "Cadence", "spm", "#2ca02c", False),
    ("altitude_m", "Altitude", "m", "#8c564b", False),
    ("power", "Power", "W", "#ff7f0e", False),
    ("vertical_oscillation", "Vertical Oscillation", "mm", "#9467bd", False),
    ("stance_time", "Ground Contact Time", "ms", "#1f77b4", False),
    ("vertical_ratio", "Vertical Ratio", "%", "#17becf", False),
    ("step_length", "Step Length", "mm", "#bcbd22", False),
    ("stance_time_balance", "GCT Balance (L/R)", "%", "#7f7f7f", False),
    ("stance_time_percent", "Stance Time", "%", "#aec7e8", False),
    ("speed_kmh", "Speed", "km/h", "#c49c94", False),
    ("temperature", "Temperature", "°C", "#ffbb78", False),
    ("grade", "Grade", "%", "#98df8a", False),
    ("respiration_rate", "Respiration Rate", "brpm", "#ff9896", False),
    ("gps_accuracy", "GPS Accuracy", "m", "#c5b0d5", False),
]

def check_password() -> None:
    if st.session_state.get("authenticated"): return
    st.title("🔒 Restricted Analytics Dashboard")
    pw = st.text_input("Password", type="password")
    if st.button("Unlock"):
        try:
            if hmac.compare_digest(pw, str(st.secrets["app_password"])):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        except: st.error("Secrets not configured."); st.stop()
    st.stop()

def fmt_duration(seconds) -> str:
    if seconds is None or not np.isfinite(seconds): return "—"
    s = int(round(seconds))
    h, m = divmod(s, 3600)
    m, s = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def fmt_pace(pace_min_km) -> str:
    if pace_min_km is None or not np.isfinite(pace_min_km): return "—"
    m = int(pace_min_km)
    s = int(round((pace_min_km - m) * 60))
    return f"{m}:{s:02d}"

def get_file_info(file_path):
    """Extracts the start date from the FIT file metadata."""
    try:
        fitfile = fitparse.FitFile(file_path)
        for record in fitfile.get_messages("file_id"):
            for data in record:
                if data.name == "time_created":
                    return data.value.strftime("%Y-%m-%d")
    except: pass
    return "Unknown Date"

def first_present(df: pd.DataFrame, *candidates):
    for c in candidates:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any(): return c
    return None

@st.cache_data(show_spinner="Parsing .fit telemetry…")
def parse_fit(file_bytes: bytes):
    fit = FitFile(io.BytesIO(file_bytes))
    records, others, msg_counts = [], {}, {}
    for msg in fit.get_messages():
        name = msg.name
        msg_counts[name] = msg_counts.get(name, 0) + 1
        if name == "record": records.append(msg.get_values())
        elif name in ("session", "lap", "sport", "file_id", "device_info"):
            others.setdefault(name, []).append(msg.get_values())
    return pd.DataFrame(records), others, msg_counts

def prepare(df: pd.DataFrame):
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    else: df["elapsed_s"] = np.arange(len(df), dtype=float)

    sp = first_present(df, "enhanced_speed", "speed")
    df["speed_ms"] = pd.to_numeric(df[sp], errors="coerce") if sp else np.nan
    df["speed_kmh"] = df["speed_ms"] * 3.6
    pace = 1000.0 / (df["speed_ms"] * 60.0)
    pace[df["speed_ms"] <= 0.3] = np.nan
    df["pace_min_km"] = pace.replace([np.inf, -np.inf], np.nan)

    if "distance" in df.columns:
        df["distance_km"] = pd.to_numeric(df["distance"], errors="coerce").ffill() / 1000.0
        has_distance = True
    else:
        step = df["speed_ms"].fillna(0) * df["elapsed_s"].diff().fillna(0)
        df["distance_km"] = step.clip(lower=0).cumsum() / 1000.0
        has_distance = df["distance_km"].max() > 0 if len(df) else False

    al = first_present(df, "enhanced_altitude", "altitude")
    if al: df["altitude_m"] = pd.to_numeric(df[al], errors="coerce")
    
    if "cadence" in df.columns:
        cad = pd.to_numeric(df["cadence"], errors="coerce")
        frac = pd.to_numeric(df.get("fractional_cadence", 0), errors="coerce").fillna(0)
        df["cadence_spm"] = (cad + frac) * 2

    if "position_lat" in df.columns:
        df["lat"] = pd.to_numeric(df["position_lat"], errors="coerce") * SEMICIRCLE_TO_DEG
        df["lon"] = pd.to_numeric(df["position_long"], errors="coerce") * SEMICIRCLE_TO_DEG

    x_col = "distance_km" if has_distance else "elapsed_s"
    return df, x_col, "Distance (km)" if has_distance else "Elapsed Time (s)", has_distance

def moving_stats(df: pd.DataFrame, threshold: float):
    dt = df["elapsed_s"].diff().fillna(0).clip(lower=0)
    moving = df["speed_ms"] > threshold
    moving_time = float(dt[moving].sum())
    total_km = float(df["distance_km"].max()) if len(df) and "distance_km" in df.columns else 0.0
    avg_pace = (moving_time / 60.0) / total_km if (total_km > 0 and moving_time > 0) else np.nan
    return moving_time, total_km, avg_pace

def build_master_chart(df, x_col, x_label, metrics):
    n = len(metrics)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=[m['label'] for m in metrics])
    for i, m in enumerate(metrics, start=1):
        fig.add_trace(go.Scatter(x=df[x_col], y=df[m["col"]], name=m["label"], mode="lines", 
                                 line=dict(width=1.4, color=m["colour"]), connectgaps=True,
                                 hovertemplate=f"%{{y:.1f}} {m['unit']}<extra>{m['label']}</extra>"), row=i, col=1)
        if m["invert"]: fig.update_yaxes(autorange="reversed", row=i, col=1)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="#888")
    fig.update_layout(height=max(300, 200 * n), hovermode="x unified", showlegend=False, margin=dict(l=60, r=20, t=40, b=50))
    return fig

def main():
    check_password()
    st.title("🏃 Advanced Running Telemetry Dashboard")
    RUNS_DIR.mkdir(exist_ok=True)
    fit_files = [p for p in RUNS_DIR.iterdir() if p.suffix.lower() == ".fit"]

    with st.sidebar:
        st.header("📁 Activity")
        if fit_files:
            # MAP: "Date | Filename" -> FilePath
            options = {f"{get_file_info(p)} | {p.name}": p for p in fit_files}
            chosen_label = st.selectbox("Select a run:", list(options.keys()))
            selected_path = options[chosen_label]
        else:
            selected_path = None
            st.warning("No .fit files in `runs/`.")

    if selected_path:
        raw_df, others, _ = parse_fit(selected_path.read_bytes())
        df, x_col, x_label, _ = prepare(raw_df)
        m_time, tot_km, avg_pace = moving_stats(df, 0.3)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Distance", f"{tot_km:.2f} km")
        c2.metric("Moving Time", fmt_duration(m_time))
        c3.metric("Avg Pace", f"{fmt_pace(avg_pace)} /km")
        
        metrics = [{"col": c, "label": l, "unit": u, "colour": col, "invert": inv} 
                   for c, l, u, col, inv in CURATED_METRICS if c in df.columns]
        st.plotly_chart(build_master_chart(df, x_col, x_label, metrics), use_container_width=True)

if __name__ == "__main__":
    main()
