"""
🏃 Advanced Running Telemetry Dashboard
========================================
A neutral, maximum-detail analytics dashboard for raw .fit running files.

- Password-gated startup (Streamlit secrets).
- Reads .fit files from a repo-hosted runs/ folder (no upload, no database).
- Extracts *every* available record stream + advanced running dynamics.
- Moving-time vs elapsed-time computed strictly from telemetry velocity.
- Master chart: one shared-X subplot per metric with a unified spike-line.
- GPS map colour-coded by Heart Rate or Pace.
- Top-line summary metrics + detailed per-kilometre splits table.
"""

import io
import math
import hmac
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fitparse import FitFile

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Running Telemetry Dashboard",
    page_icon="🏃",
    layout="wide",
)

# Semicircle -> degree conversion factor for Garmin GPS coordinates
SEMICIRCLE_TO_DEG = 180.0 / (2 ** 31)

# Folder (next to this file) where .fit files are committed in the GitHub repo.
RUNS_DIR = Path(__file__).resolve().parent / "runs"

# Friendly labels / units / colours for the curated metrics (in display order).
# col, label, unit, colour, invert_y
CURATED_METRICS = [
    ("pace_min_km",          "Pace",                 "min/km", "#d62728", True),
    ("heart_rate",           "Heart Rate",           "bpm",    "#e377c2", False),
    ("cadence_spm",          "Cadence",              "spm",    "#2ca02c", False),
    ("altitude_m",           "Altitude",             "m",      "#8c564b", False),
    ("power",                "Power",                "W",      "#ff7f0e", False),
    ("vertical_oscillation", "Vertical Oscillation", "mm",     "#9467bd", False),
    ("stance_time",          "Ground Contact Time",  "ms",     "#1f77b4", False),
    ("vertical_ratio",       "Vertical Ratio",       "%",      "#17becf", False),
    ("step_length",          "Step Length",          "mm",     "#bcbd22", False),
    ("stance_time_balance",  "GCT Balance (L/R)",    "%",      "#7f7f7f", False),
    ("stance_time_percent",  "Stance Time",          "%",      "#aec7e8", False),
    ("speed_kmh",            "Speed",                "km/h",   "#c49c94", False),
    ("temperature",          "Temperature",          "°C",     "#ffbb78", False),
    ("grade",                "Grade",                "%",      "#98df8a", False),
    ("respiration_rate",     "Respiration Rate",     "brpm",   "#ff9896", False),
    ("gps_accuracy",         "GPS Accuracy",         "m",      "#c5b0d5", False),
]


# --------------------------------------------------------------------------- #
# Security: password gate
# --------------------------------------------------------------------------- #
def check_password() -> None:
    """Blocks the entire app until the correct password is supplied."""
    if st.session_state.get("authenticated"):
        return

    st.title("🔒 Restricted Analytics Dashboard")
    st.caption("Enter the access password to load the application.")
    pw = st.text_input("Password", type="password", label_visibility="collapsed")
    unlock = st.button("Unlock", type="primary")

    if unlock or pw:
        try:
            correct = str(st.secrets["app_password"])
        except Exception:
            st.error(
                "⚠️ No `app_password` configured. Add it to "
                "`.streamlit/secrets.toml` (local) or the app's *Secrets* "
                "settings (Streamlit Cloud)."
            )
            st.stop()

        if hmac.compare_digest(pw, correct):
            st.session_state["authenticated"] = True
            st.rerun()
        elif pw:
            st.error("Incorrect password.")

    st.stop()


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_duration(seconds) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "—"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_pace(pace_min_km) -> str:
    if pace_min_km is None or not np.isfinite(pace_min_km):
        return "—"
    m = int(pace_min_km)
    s = int(round((pace_min_km - m) * 60))
    if s == 60:
        m, s = m + 1, 0
    return f"{m}:{s:02d}"


def first_present(df: pd.DataFrame, *candidates):
    """Return the first candidate column that exists and holds real data."""
    for c in candidates:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return None


# --------------------------------------------------------------------------- #
# FIT parsing (cached on raw bytes)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Parsing .fit telemetry…")
def parse_fit(file_bytes: bytes):
    """Parse a .fit byte buffer fully in memory."""
    fit = FitFile(io.BytesIO(file_bytes))

    records, others, msg_counts = [], {}, {}
    for msg in fit.get_messages():
        name = msg.name
        msg_counts[name] = msg_counts.get(name, 0) + 1
        values = msg.get_values()
        if name == "record":
            records.append(values)
        elif name in ("session", "lap", "sport", "file_id", "device_info"):
            others.setdefault(name, []).append(values)

    df = pd.DataFrame(records)
    return df, others, msg_counts


# --------------------------------------------------------------------------- #
# Derive analysis columns
# --------------------------------------------------------------------------- #
def prepare(df: pd.DataFrame):
    df = df.copy()

    # --- Time base -------------------------------------------------------- #
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    else:
        df = df.reset_index(drop=True)
        df["elapsed_s"] = np.arange(len(df), dtype=float)

    # --- Speed & pace ----------------------------------------------------- #
    sp = first_present(df, "enhanced_speed", "speed")
    if sp:
        df["speed_ms"] = pd.to_numeric(df[sp], errors="coerce")
    elif "distance" in df.columns:
        dd = pd.to_numeric(df["distance"], errors="coerce").diff()
        dt = df["elapsed_s"].diff()
        df["speed_ms"] = (dd / dt).replace([np.inf, -np.inf], np.nan)
    else:
        df["speed_ms"] = np.nan

    df["speed_kmh"] = df["speed_ms"] * 3.6
    pace = 1000.0 / (df["speed_ms"] * 60.0)          # min per km
    pace[df["speed_ms"] <= 0.3] = np.nan             # treat near-stops as gaps
    df["pace_min_km"] = pace.replace([np.inf, -np.inf], np.nan)

    # --- Distance --------------------------------------------------------- #
    if "distance" in df.columns and pd.to_numeric(df["distance"], errors="coerce").notna().any():
        df["distance_km"] = pd.to_numeric(df["distance"], errors="coerce").ffill() / 1000.0
        has_distance = True
    else:
        step = df["speed_ms"].fillna(0) * df["elapsed_s"].diff().fillna(0)
        df["distance_km"] = step.clip(lower=0).cumsum() / 1000.0
        has_distance = df["distance_km"].iloc[-1] > 0 if len(df) else False

    # --- Altitude --------------------------------------------------------- #
    al = first_present(df, "enhanced_altitude", "altitude")
    if al:
        df["altitude_m"] = pd.to_numeric(df[al], errors="coerce")

    # --- Cadence -> steps per minute ------------------------------------- #
    if "cadence" in df.columns:
        cad = pd.to_numeric(df["cadence"], errors="coerce")
        if "fractional_cadence" in df.columns:
            frac = pd.to_numeric(df["fractional_cadence"], errors="coerce").fillna(0)
        else:
            frac = 0
        df["cadence_spm"] = (cad + frac) * 2

    # --- GPS coordinates (semicircles -> degrees) ------------------------- #
    if "position_lat" in df.columns and "position_long" in df.columns:
        df["lat"] = pd.to_numeric(df["position_lat"], errors="coerce") * SEMICIRCLE_TO_DEG
        df["lon"] = pd.to_numeric(df["position_long"], errors="coerce") * SEMICIRCLE_TO_DEG

    if has_distance:
        x_col, x_label = "distance_km", "Distance (km)"
    else:
        df["elapsed_min"] = df["elapsed_s"] / 60.0
        x_col, x_label = "elapsed_min", "Elapsed Time (min)"

    return df, x_col, x_label, has_distance


# --------------------------------------------------------------------------- #
# Moving-time statistics (strictly velocity based)
# --------------------------------------------------------------------------- #
def moving_stats(df: pd.DataFrame, threshold: float):
    dt = df["elapsed_s"].diff().fillna(0).clip(lower=0)
    moving = df["speed_ms"] > threshold
    moving_time = float(dt[moving].sum())
    elapsed = float(df["elapsed_s"].iloc[-1] - df["elapsed_s"].iloc[0]) if len(df) else 0.0
    total_km = float(df["distance_km"].max()) if len(df) and "distance_km" in df.columns else 0.0
    avg_pace = (moving_time / 60.0) / total_km if (total_km > 0 and moving_time > 0) else np.nan
    return moving_time, elapsed, total_km, avg_pace


def elevation_totals(df: pd.DataFrame):
    if "altitude_m" not in df.columns or not df["altitude_m"].notna().any():
        return np.nan, np.nan
    alt = df["altitude_m"].interpolate().rolling(5, center=True, min_periods=1).mean()
    diffs = alt.diff()
    ascent = float(diffs.clip(lower=0).sum())
    descent = float(-diffs.clip(upper=0).sum())
    return ascent, descent


# --------------------------------------------------------------------------- #
# Per-kilometre splits
# --------------------------------------------------------------------------- #
def compute_splits(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["distance_km", "elapsed_s"]).sort_values("distance_km")
    dist = d["distance_km"].values
    t = d["elapsed_s"].values
    if len(dist) < 2 or dist[-1] <= 0:
        return pd.DataFrame()

    total = dist[-1]
    boundaries = list(range(0, int(np.floor(total)) + 1))
    if total > boundaries[-1] + 1e-6:
        boundaries.append(round(total, 3))     # final partial km

    rows = []
    for i in range(1, len(boundaries)):
        lo, hi = boundaries[i - 1], boundaries[i]
        t_lo, t_hi = np.interp(lo, dist, t), np.interp(hi, dist, t)
        lap_time = t_hi - t_lo
        seg_dist = hi - lo

        idx = np.nonzero((dist >= lo) & (dist <= hi))[0]
        seg = d.iloc[idx]

        avg_pace = (lap_time / 60.0) / seg_dist if seg_dist > 0 else np.nan
        max_pace = seg["pace_min_km"].min() if "pace_min_km" in seg else np.nan

        if "altitude_m" in seg and seg["altitude_m"].notna().any():
            ad = seg["altitude_m"].diff()
            gain = float(ad.clip(lower=0).sum())
            loss = float(-ad.clip(upper=0).sum())
        else:
            gain = loss = np.nan

        rows.append({
            "Split":          f"{i}" if hi == int(hi) else f"{i} (partial)",
            "Dist (km)":      round(hi, 2),
            "Lap Time":       fmt_duration(lap_time),
            "Avg Pace":       fmt_pace(avg_pace),
            "Best Pace":      fmt_pace(max_pace),
            "Avg HR":         round(seg["heart_rate"].mean(), 0) if "heart_rate" in seg else np.nan,
            "Max HR":         round(seg["heart_rate"].max(), 0) if "heart_rate" in seg else np.nan,
            "Elev +/- (m)":   f"+{gain:.0f} / -{loss:.0f}" if np.isfinite(gain) else "—",
            "Avg Cad (spm)":  round(seg["cadence_spm"].mean(), 0) if "cadence_spm" in seg else np.nan,
            "Avg Power (W)":  round(seg["power"].mean(), 0) if "power" in seg else np.nan,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Master chart
# --------------------------------------------------------------------------- #
def assemble_metrics(df: pd.DataFrame, x_col: str):
    metrics, consumed = [], set()
    for col, label, unit, colour, invert in CURATED_METRICS:
        if col in df.columns and df[col].notna().any():
            metrics.append({"col": col, "label": label, "unit": unit,
                            "colour": colour, "invert": invert})
            consumed.add(col)

    # Skip raw source columns already represented or used as axes/coords.
    skip = {
        x_col, "elapsed_s", "elapsed_min", "timestamp", "distance", "distance_km",
        "speed", "enhanced_speed", "speed_ms", "altitude", "enhanced_altitude",
        "position_lat", "position_long", "lat", "lon", "cadence", "fractional_cadence",
    } | consumed

    # Add every remaining numeric stream so nothing is omitted.
    for col in df.select_dtypes(include="number").columns:
        if col not in skip and df[col].notna().any():
            metrics.append({"col": col, "label": col.replace("_", " ").title(),
                            "unit": "", "colour": None, "invert": False})
    return metrics


def build_master_chart(df: pd.DataFrame, x_col: str, x_label: str, metrics):
    n = len(metrics)
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True,
        vertical_spacing=min(0.02, 0.6 / max(n, 1)),
        subplot_titles=[f"{m['label']} ({m['unit']})" if m["unit"] else m["label"]
                        for m in metrics],
    )

    for i, m in enumerate(metrics, start=1):
        fig.add_trace(
            go.Scatter(
                x=df[x_col], y=df[m["col"]], name=m["label"], mode="lines",
                line=dict(width=1.4, color=m["colour"]), connectgaps=True,
                hovertemplate=f"%{{y:.1f}} {m['unit']}<extra>{m['label']}</extra>",
            ),
            row=i, col=1,
        )
        fig.update_yaxes(title_text=m["unit"], row=i, col=1)
        if m["invert"]:
            fig.update_yaxes(autorange="reversed", row=i, col=1)

    # Unified vertical spike-line across every subplot.
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikecolor="#888", spikedash="solid")
    fig.update_xaxes(title_text=x_label, row=n, col=1)
    fig.update_layout(
        height=max(320, 235 * n),
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=60, r=20, t=42, b=50),
    )
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=12)
    return fig


# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #
def _zoom_for(lat: pd.Series, lon: pd.Series) -> float:
    span_km = max(abs(lat.max() - lat.min()), abs(lon.max() - lon.min())) * 111.0
    zoom = 13.0 - math.log2(max(span_km, 0.3))
    return float(min(16, max(3, zoom)))


def build_map(df: pd.DataFrame, color_by: str):
    g = df.dropna(subset=["lat", "lon"]).copy()
    if g.empty:
        return None

    if color_by == "Heart Rate" and "heart_rate" in g and g["heart_rate"].notna().any():
        cval, cbar, reverse = g["heart_rate"], "HR (bpm)", False
    else:
        cval, cbar, reverse = g["pace_min_km"], "Pace (min/km)", True

    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=g["lat"], lon=g["lon"], mode="lines",
        line=dict(width=2, color="rgba(140,140,140,0.4)"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scattermapbox(
        lat=g["lat"], lon=g["lon"], mode="markers",
        marker=dict(size=8, color=cval, colorscale="Turbo", reversescale=reverse,
                    colorbar=dict(title=cbar), showscale=True),
        customdata=g[["distance_km"]].values,
        hovertemplate="%{customdata[0]:.2f} km<br>" + cbar + ": %{marker.color:.0f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=g["lat"].mean(), lon=g["lon"].mean()),
                    zoom=_zoom_for(g["lat"], g["lon"])),
        height=620, margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def render_dict_table(records):
    if not records:
        return
    table = pd.DataFrame(records).T
    table.columns = [f"#{i+1}" for i in range(table.shape[1])]
    st.dataframe(table, use_container_width=True)


# --------------------------------------------------------------------------- #
# Main application
# --------------------------------------------------------------------------- #
def main():
    check_password()

    st.title("🏃 Advanced Running Telemetry Dashboard")
    st.caption("Select a .fit file from the runs/ folder to extract and visualise every data point.")

    # Ensure the runs/ folder exists so the app never crashes on a fresh repo.
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    fit_files = sorted(
        (p for p in RUNS_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".fit"),
        key=lambda p: p.name.lower(),
    )

    with st.sidebar:
        st.header("📁 Activity")
        if fit_files:
            chosen = st.selectbox("Select a .fit file", [p.name for p in fit_files])
            selected_path = next(p for p in fit_files if p.name == chosen)
        else:
            selected_path = None
            st.warning("No .fit files found in the `runs/` folder.")

        st.divider()
        st.header("⚙️ Settings")
        move_thr = st.slider(
            "Moving threshold (m/s)", 0.0, 2.0, 0.3, 0.1,
            help="Velocity below this is treated as a stop and excluded from moving time.",
        )
        map_color = st.radio("Colour map by", ["Heart Rate", "Pace"], horizontal=True)
        if st.button("Log out"):
            st.session_state.clear()
            st.rerun()

    if selected_path is None:
        st.info(
            f"⬆️ Add `.fit` files to the **`runs/`** folder in your repository, "
            f"then pick one from the sidebar.\n\n_Looking in:_ `{RUNS_DIR}`"
        )
        st.stop()

    # Read the raw bytes from disk and feed them into the existing parser.
    raw_df, others, msg_counts = parse_fit(selected_path.read_bytes())
    if raw_df.empty:
        st.error("No `record` messages found in this file.")
        st.stop()

    df, x_col, x_label, has_distance = prepare(raw_df)
    moving_time, elapsed, total_km, avg_pace = moving_stats(df, move_thr)
    ascent, descent = elevation_totals(df)

    # ----------------------- Top-line summary --------------------------- #
    st.subheader("📊 Summary")
    r1 = st.columns(6)
    r1[0].metric("Distance", f"{total_km:.2f} km")
    r1[1].metric("Moving Time", fmt_duration(moving_time))
    r1[2].metric("Elapsed Time", fmt_duration(elapsed))
    r1[3].metric("Avg Moving Pace", f"{fmt_pace(avg_pace)} /km")
    r1[4].metric("Total Ascent", f"{ascent:.0f} m" if np.isfinite(ascent) else "—")
    r1[5].metric("Total Descent", f"{descent:.0f} m" if np.isfinite(descent) else "—")

    r2 = st.columns(6)
    if "heart_rate" in df:
        r2[0].metric("Avg HR", f"{df['heart_rate'].mean():.0f} bpm")
        r2[1].metric("Max HR", f"{df['heart_rate'].max():.0f} bpm")
    if "cadence_spm" in df:
        r2[2].metric("Avg Cadence", f"{df['cadence_spm'].mean():.0f} spm")
        r2[3].metric("Max Cadence", f"{df['cadence_spm'].max():.0f} spm")
    if "power" in df and df["power"].notna().any():
        r2[4].metric("Avg Power", f"{df['power'].mean():.0f} W")
        r2[5].metric("Max Power", f"{df['power'].max():.0f} W")

    # ----------------------- Master chart ------------------------------- #
    st.subheader("📈 Master Telemetry Chart")
    st.caption("All metrics share the X-axis. Hover to read every stream at the same point.")
    metrics = assemble_metrics(df, x_col)
    if metrics:
        st.plotly_chart(build_master_chart(df, x_col, x_label, metrics),
                        use_container_width=True)
    else:
        st.warning("No plottable numeric streams were found.")

    # ----------------------- Map ---------------------------------------- #
    st.subheader("🗺️ Route Map")
    map_fig = build_map(df, map_color)
    if map_fig is not None:
        st.plotly_chart(map_fig, use_container_width=True)
    else:
        st.info("No GPS coordinates present in this file.")

    # ----------------------- Splits ------------------------------------- #
    st.subheader("🔢 Kilometre Splits")
    splits = compute_splits(df) if has_distance else pd.DataFrame()
    if not splits.empty:
        st.dataframe(splits, use_container_width=True, hide_index=True)
    else:
        st.info("Splits require a distance stream.")

    # ----------------------- Raw data & metadata ----------------------- #
    with st.expander("🧬 Raw data, detected fields & file metadata"):
        st.write("**Message types found in file:**")
        st.json(msg_counts)
        st.write(f"**Record fields detected ({len(raw_df.columns)}):**")
        st.write(sorted(raw_df.columns.tolist()))
        if "session" in others:
            st.write("**Session summary:**")
            render_dict_table(others["session"])
        st.write("**Full parsed record stream:**")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Download parsed data (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_path.stem}_parsed.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
