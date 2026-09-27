"""Plotly figure builders for ThermaTwin dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_dyno_overlay(
    surface_card: np.ndarray,
    downhole_card: np.ndarray,
    title: str = "Dynamometer Cards",
    rod_float_detected: bool = False,
) -> go.Figure:
    """
    Plot surface and downhole dynamometer cards overlaid.

    Parameters
    ----------
    surface_card : np.ndarray
        Surface card array of shape (N, 2) [load, position].
    downhole_card : np.ndarray
        Downhole card array of shape (N, 2) [load, position].
    title : str
        Chart title.
    rod_float_detected : bool
        If True, color downhole card red.

    Returns
    -------
    go.Figure
    """
    downhole_color = "red" if rod_float_detected else "green"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=surface_card[:, 1],
        y=surface_card[:, 0],
        mode="lines",
        name="Surface Card",
        line=dict(color="blue", dash="dash", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=downhole_card[:, 1],
        y=downhole_card[:, 0],
        mode="lines",
        name="Downhole Card",
        line=dict(color=downhole_color, width=2),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Position (m)",
        yaxis_title="Load (N)",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=30, t=50, b=50),
        height=400,
    )

    return fig


def plot_thermal_decay(
    df: pd.DataFrame,
    highlight_day: int | None = None,
    highlight_time: str | None = None,
) -> go.Figure:
    """
    Plot temperature decay curve from synthetic or live data.

    Parameters
    ----------
    df : pd.DataFrame
        Synthetic: columns [day, temperature_c, viscosity_cp]
        Live: columns [time, wellhead_temp_c]
    highlight_day : int, optional
        Day to highlight (synthetic mode).
    highlight_time : str, optional
        ISO timestamp to highlight (live mode).

    Returns
    -------
    go.Figure
    """
    # Detect data mode from columns
    is_synthetic = "day" in df.columns

    if is_synthetic:
        return _plot_synthetic_thermal(df, highlight_day)
    else:
        return _plot_live_thermal(df, highlight_time)


def _plot_synthetic_thermal(df: pd.DataFrame, highlight_day: int | None) -> go.Figure:
    """Plot synthetic thermal decay with dual y-axes."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["day"],
            y=df["temperature_c"],
            mode="lines",
            name="Temperature (°C)",
            line=dict(color="orange", width=2),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["day"],
            y=df["viscosity_cp"],
            mode="lines",
            name="Viscosity (cP)",
            line=dict(color="purple", width=2),
        ),
        secondary_y=True,
    )

    if highlight_day is not None:
        fig.add_vline(
            x=highlight_day,
            line=dict(color="red", dash="dot", width=2),
            annotation_text=f"Day {highlight_day}",
            annotation_position="top",
        )

    fig.update_layout(
        title="Reservoir Thermal Decay & Viscosity Spike",
        xaxis_title="Day Since CSS Cycle Start",
        margin=dict(l=60, r=60, t=50, b=50),
        height=400,
        legend=dict(x=0.02, y=0.98),
    )

    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig.update_yaxes(title_text="Viscosity (cP)", type="log", secondary_y=True)

    return fig


def _plot_live_thermal(df: pd.DataFrame, highlight_time: str | None) -> go.Figure:
    """Plot live wellhead temperature decay."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["wellhead_temp_c"],
            mode="lines",
            name="Wellhead Temperature (°C)",
            line=dict(color="orange", width=2),
        ),
    )

    if highlight_time is not None:
        fig.add_vline(
            x=highlight_time,
            line=dict(color="red", dash="dot", width=2),
            annotation_text="Latest",
            annotation_position="top",
        )

    fig.update_layout(
        title="Wellhead Temperature Decay (Live)",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        margin=dict(l=60, r=30, t=50, b=50),
        height=400,
        legend=dict(x=0.02, y=0.98),
    )

    return fig
