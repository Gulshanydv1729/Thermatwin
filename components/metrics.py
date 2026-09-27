"""UI metric card and warning banner renderers for Streamlit."""

import streamlit as st


def render_metric(
    label: str,
    value: str | float,
    unit: str,
    delta: str | None = None,
    alert: bool = False,
) -> None:
    """
    Render a single KPI metric card.

    Parameters
    ----------
    label : str
        Metric label (e.g., "Max Safe SPM").
    value : str | float
        Metric value.
    unit : str
        Unit string (e.g., "SPM", "Hz").
    delta : str, optional
        Delta indicator text.
    alert : bool
        If True, render with red alert styling.
    """
    border_color = "#ff4b4b" if alert else "#e0e0e0"
    bg_color = "#fff0f0" if alert else "#f8f9fa"

    delta_html = f'<div style="font-size: 12px; color: #666; margin-top: 4px;">{delta}</div>' if delta else ""

    st.markdown(
        f"""
        <div style="
            border: 2px solid {border_color};
            border-radius: 8px;
            padding: 16px;
            background-color: {bg_color};
            text-align: center;
        ">
            <div style="font-size: 14px; color: #666;">{label}</div>
            <div style="font-size: 28px; font-weight: bold; color: #333;">{value}</div>
            <div style="font-size: 12px; color: #999;">{unit}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_warning_banner(diagnostics: dict) -> None:
    """
    Render a warning banner based on diagnostic severity.

    Parameters
    ----------
    diagnostics : dict
        Output from detect_rod_float().
    """
    severity = diagnostics.get("severity", "low")

    if severity == "high":
        st.error(
            f"CRITICAL: Rod float detected! "
            f"Load transfer delay: {diagnostics['delay_pct']:.1f}%. "
            f"Impact loading: {'Yes' if diagnostics['impact_loading'] else 'No'}. "
            f"Reduce SPM immediately."
        )
    elif severity == "medium":
        st.warning(
            f"WARNING: Potential rod float. "
            f"Load transfer delay: {diagnostics['delay_pct']:.1f}%. "
            f"Monitor closely."
        )
    # "low" severity: no banner
