"""Diagnostics and anomaly detection for dynamometer cards."""

import numpy as np


def detect_rod_float(
    downhole_card: np.ndarray,
    surface_card: np.ndarray,
    viscosity: float,
    viscosity_threshold: float = 5000.0,
    delay_threshold_pct: float = 5.0,
) -> dict:
    """
    Detect rod float and mechanical impact loading from dyno card shapes.

    Parameters
    ----------
    downhole_card : np.ndarray
        Downhole card array of shape (N, 2) [load, position].
    surface_card : np.ndarray
        Surface card array of shape (N, 2) [load, position].
    viscosity : float
        Fluid viscosity in cP.
    viscosity_threshold : float
        Viscosity threshold for rod float detection (cP).
    delay_threshold_pct : float
        Minimum load transfer delay percentage to flag rod float.

    Returns
    -------
    dict
        {
            "rod_float": bool,
            "impact_loading": bool,
            "delay_pct": float,
            "severity": "low" | "medium" | "high"
        }
    """
    downhole_load = downhole_card[:, 0]
    surface_load = surface_card[:, 0]
    n = len(surface_load)

    # Find index of minimum load in each card
    surface_min_idx = int(np.argmin(surface_load))
    downhole_min_idx = int(np.argmin(downhole_load))

    # Compute load transfer delay
    delay_pct = abs(downhole_min_idx - surface_min_idx) / n * 100.0

    # Rod float detection
    rod_float = (delay_pct > delay_threshold_pct) and (viscosity > viscosity_threshold)

    # Mechanical impact loading: sharp inflection on downstroke
    # Compute second derivative of downhole load
    if n > 2:
        second_deriv = np.diff(downhole_load, n=2)
        impact_threshold = np.std(second_deriv) * 3.0
        impact_loading = bool(np.max(np.abs(second_deriv)) > impact_threshold)
    else:
        impact_loading = False

    # Severity assessment
    flags_sum = int(rod_float) + int(impact_loading)
    if flags_sum == 2:
        severity = "high"
    elif flags_sum == 1:
        severity = "medium"
    else:
        severity = "low"

    return {
        "rod_float": rod_float,
        "impact_loading": impact_loading,
        "delay_pct": round(delay_pct, 2),
        "severity": severity,
    }
