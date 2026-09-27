"""
ThermaTwin SRP Digital Twin — Dual-Mode Dashboard.

Supports:
  - Synthetic Demo: Static 30-day simulation with day slider
  - Live SCADA: Real-time InfluxDB polling with VFD write-back control

Both modes output identical data formats for downstream physics modules.
"""

import os

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from core.data_gen import generate_css_cycle
from core.physics_engine import calculate_downhole_card
from core.diagnostics import detect_rod_float
from core.optimizer import calculate_safe_spm
from core.db_client import get_historical_thermal_data, get_live_dyno_card, get_latest_sensors
from components.charts import plot_dyno_overlay, plot_thermal_decay
from components.metrics import render_metric, render_warning_banner

# ── MQTT client for write-back (lazy, non-blocking) ─────────────────────────

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
VFD_COMMAND_TOPIC = "baghewala/well_01/vfd/set_hz"

mqtt_client = None


def _init_mqtt():
    """Initialize MQTT client lazily. Returns None if broker unavailable."""
    global mqtt_client
    if mqtt_client is not None:
        return mqtt_client
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
        client.loop_start()
        mqtt_client = client
        return client
    except Exception:
        return None


def publish_vfd_setpoint(frequency_hz: float) -> bool:
    """Publish recommended VFD frequency to the command topic."""
    client = _init_mqtt()
    if client is None:
        return False
    try:
        result = client.publish(VFD_COMMAND_TOPIC, payload=f"{frequency_hz:.2f}", qos=1)
        return result.rc == mqtt.MQTT_ERR_SUCCESS
    except Exception:
        return False


# ── Data Source Functions ───────────────────────────────────────────────────

@st.cache_data
def load_synthetic_data() -> pd.DataFrame:
    """Cache the synthetic dataset for faster reloads."""
    return generate_css_cycle()


def get_synthetic_day_data(df: pd.DataFrame, day: int) -> dict:
    """
    Extract data for a specific day from synthetic dataset.

    Returns dict with: surface_load, surface_pos, viscosity, temperature
    """
    day_data = df[df["day"] == day]
    if day_data.empty:
        return None
    row = day_data.iloc[0]
    return {
        "surface_load": day_data["surface_load_n"].values,
        "surface_pos": day_data["surface_pos_m"].values,
        "viscosity": float(row["viscosity_cp"]),
        "temperature": float(row["temperature_c"]),
    }


def get_live_scada_data() -> dict | None:
    """
    Fetch live SCADA data from InfluxDB.

    Returns dict with: surface_load, surface_pos, viscosity, temperature
    or None if data unavailable.
    """
    try:
        dyno_data = get_live_dyno_card()
        sensors = get_latest_sensors()
    except Exception:
        return None

    if dyno_data is None:
        return None

    surface_load, surface_pos = dyno_data
    temp = sensors.get("wellhead_temp_c", 100)
    viscosity = 500 * np.exp(0.05 * (200 - temp))

    return {
        "surface_load": surface_load,
        "surface_pos": surface_pos,
        "viscosity": viscosity,
        "temperature": temp,
    }


# ── Dashboard ──────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="ThermaTwin SRP — Dual Mode", layout="wide")
    st.title("ThermaTwin — Sucker-Rod Pump Digital Twin")
    st.caption("Baghewala Field | Dual-Mode: Synthetic Demo & Live SCADA")

    # ── Sidebar: Mode Selection ──
    st.sidebar.header("Operating Mode")
    mode = st.sidebar.radio("Operating Mode", ["Synthetic Demo", "Live SCADA"])

    # ── Mode: Synthetic Demo ──
    if mode == "Synthetic Demo":
        st.sidebar.markdown("---")
        st.sidebar.header("Simulation Controls")
        day = st.sidebar.slider("Day Since CSS Cycle Start", 0, 29, 0)

        df = load_synthetic_data()
        data = get_synthetic_day_data(df, day)

        if data is None:
            st.error(f"No data for day {day}")
            return

        st.sidebar.markdown(f"**Day {day} Conditions:**")
        st.sidebar.markdown(f"- Temperature: **{data['temperature']:.1f} °C**")
        st.sidebar.markdown(f"- Viscosity: **{data['viscosity']:,.0f} cP**")

        surface_load = data["surface_load"]
        surface_pos = data["surface_pos"]
        viscosity = data["viscosity"]

    # ── Mode: Live SCADA ──
    else:
        # Activate auto-refresh for live mode
        st_autorefresh(interval=5000, key="scada_refresh")

        st.sidebar.markdown("---")
        st.sidebar.header("Live Sensors")

        try:
            sensors = get_latest_sensors()
        except Exception:
            sensors = {}

        if sensors:
            st.sidebar.markdown(f"- Wellhead Temp: **{sensors['wellhead_temp_c']:.1f} °C**")
            st.sidebar.markdown(f"- VFD Frequency: **{sensors['vfd_hz']:.1f} Hz**")
            st.sidebar.markdown(f"- Last Update: **{sensors['timestamp']}**")
        else:
            st.sidebar.warning("No sensor data available — is InfluxDB running?")

        st.sidebar.markdown("---")
        st.sidebar.header("Closed-Loop Control")

        data = get_live_scada_data()

        if data is None:
            st.warning("No live dynamometer data available. Is the ingestion worker running?")
            st.info("Start the worker with: `./run.sh ingestion`")
            st.stop()

        surface_load = data["surface_load"]
        surface_pos = data["surface_pos"]
        viscosity = data["viscosity"]

    # ── Common Pipeline (agnostic to data source) ──

    surface_card = np.column_stack([surface_load, surface_pos])
    downhole_card = calculate_downhole_card(surface_load, surface_pos, viscosity)
    diag = detect_rod_float(downhole_card, surface_card, viscosity)
    opt = calculate_safe_spm(viscosity)

    # ── Main Dashboard ──

    # Row 1: Metric cards
    st.subheader("Operating Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric(
            "Max Safe SPM",
            f"{opt['max_safe_spm']:.1f}",
            "SPM",
            delta=f"Current: {sensors.get('vfd_hz', 0) / 2:.1f} SPM" if mode == "Live SCADA" else None,
            alert=diag["severity"] == "high",
        )
    with col2:
        render_metric(
            "Recommended VFD Frequency",
            f"{opt['recommended_vfd_hz']:.1f}",
            "Hz",
            delta=f"Est. Production: {opt['production_bpd_estimate']:.0f} BPD",
        )
    with col3:
        render_metric(
            "Diagnostic Severity",
            diag["severity"].upper(),
            unit="",
            alert=diag["severity"] == "high",
        )

    # Warning banner
    render_warning_banner(diag)

    # VFD write-back button (Live SCADA mode only)
    if mode == "Live SCADA":
        if st.sidebar.button("Apply Recommended VFD Setpoint", type="primary"):
            success = publish_vfd_setpoint(opt["recommended_vfd_hz"])
            if success:
                st.sidebar.success(f"Published {opt['recommended_vfd_hz']:.1f} Hz to {VFD_COMMAND_TOPIC}")
            else:
                st.sidebar.error("Failed to publish — is the MQTT broker running?")

    st.markdown("---")

    # Row 2: Charts
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Thermal Decay Profile")
        if mode == "Synthetic Demo":
            fig_thermal = plot_thermal_decay(df, highlight_day=day)
            st.plotly_chart(fig_thermal, width='stretch')
        else:
            try:
                thermal_df = get_historical_thermal_data(days=30)
                if not thermal_df.empty:
                    fig_thermal = plot_thermal_decay(thermal_df)
                    st.plotly_chart(fig_thermal, width='stretch')
                else:
                    st.info("No historical thermal data available yet.")
            except Exception:
                st.info("InfluxDB not available — thermal history will appear once connected.")

    with col_right:
        st.subheader("Dynamometer Cards")
        fig_dyno = plot_dyno_overlay(
            surface_card,
            downhole_card,
            rod_float_detected=diag["rod_float"],
        )
        st.plotly_chart(fig_dyno, width='stretch')

    st.markdown("---")

    # Row 3: Diagnostics summary
    st.subheader("Diagnostics Summary")
    diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)
    with diag_col1:
        st.metric("Rod Float Detected", "YES" if diag["rod_float"] else "NO")
    with diag_col2:
        st.metric("Impact Loading", "YES" if diag["impact_loading"] else "NO")
    with diag_col3:
        st.metric("Load Transfer Delay", f"{diag['delay_pct']:.1f}%")
    with diag_col4:
        st.metric("Est. Viscosity", f"{viscosity:,.0f} cP")


if __name__ == "__main__":
    main()
