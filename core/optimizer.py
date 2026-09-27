"""Optimization controller for VFD/SPM recommendations."""

import numpy as np


def calculate_safe_spm(
    viscosity: float,
    current_spm: float = 6.0,
    max_spm: float = 12.0,
    min_spm: float = 1.5,
    viscosity_ref: float = 1000.0,
    motor_pole_pairs: int = 2,
    safety_factor: float = 0.9,
    pump_displacement: float = 0.5,  # bbl/stroke
) -> dict:
    """
    Calculate maximum safe SPM and recommended VFD frequency.

    Parameters
    ----------
    viscosity : float
        Fluid viscosity in cP.
    current_spm : float
        Current operating SPM.
    max_spm : float
        Maximum allowable SPM.
    min_spm : float
        Minimum allowable SPM.
    viscosity_ref : float
        Reference viscosity for normalization (cP).
    motor_pole_pairs : int
        Number of motor pole pairs (4-pole motor = 2).
    safety_factor : float
        Safety margin factor (0-1).
    pump_displacement : float
        Pump displacement in bbl/stroke.

    Returns
    -------
    dict
        {
            "max_safe_spm": float,
            "recommended_vfd_hz": float,
            "production_bpd_estimate": float
        }
    """
    # Safe SPM scales inversely with sqrt of viscosity ratio
    safe_spm = max_spm * (viscosity_ref / viscosity) ** 0.5

    # Clamp to operating range
    safe_spm = max(min_spm, min(max_spm, safe_spm))

    # Convert SPM to VFD frequency
    # freq_hz = spm / 60 * motor_pole_pairs * 60 = spm * motor_pole_pairs
    safe_freq = safe_spm * motor_pole_pairs

    # Apply safety factor
    recommended_vfd_hz = safety_factor * safe_freq

    # Production estimate: SPM * displacement * 1440 min/day
    production_bpd = safe_spm * pump_displacement * 1440

    return {
        "max_safe_spm": round(safe_spm, 2),
        "recommended_vfd_hz": round(recommended_vfd_hz, 2),
        "production_bpd_estimate": round(production_bpd, 1),
    }
