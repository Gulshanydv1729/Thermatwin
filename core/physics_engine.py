"""Physics engine for downhole dynamometer card generation."""

import numpy as np
from scipy.signal import savgol_filter

# Module-level constants
VISCOSITY_THRESHOLD = 5000.0  # cP


def calculate_downhole_card(
    surface_load: np.ndarray,
    surface_pos: np.ndarray,
    viscosity: float,
    rod_length: float = 1500.0,
    wave_speed: float = 5000.0,
    damping_coeff: float | None = None,
) -> np.ndarray:
    """
    Convert surface card to bottom-hole dynamometer card.

    Parameters
    ----------
    surface_load : np.ndarray
        Surface polished rod load values (N).
    surface_pos : np.ndarray
        Surface polished rod position values (m).
    viscosity : float
        Fluid viscosity in cP.
    rod_length : float
        Rod string length in meters.
    wave_speed : float
        Wave propagation speed in rod string (m/s).
    damping_coeff : float, optional
        Override damping coefficient. If None, computed from viscosity.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) where column 0 = load (N), column 1 = position (m).
    """
    surface_load = np.asarray(surface_load, dtype=float)
    surface_pos = np.asarray(surface_pos, dtype=float)
    n = len(surface_load)

    # Compute damping from viscosity
    if damping_coeff is None:
        damping = (viscosity / VISCOSITY_THRESHOLD) * 0.1
    else:
        damping = damping_coeff

    # Apply wave propagation attenuation
    attenuation = np.exp(-damping * rod_length / wave_speed)
    downhole_load = surface_load * attenuation

    # Apply phase delay (wave travel time)
    delay_samples = int(damping * n * 0.05)
    if delay_samples > 0:
        downhole_load = np.roll(downhole_load, delay_samples)

    # If viscosity exceeds threshold, introduce rod-float distortion
    if viscosity >= VISCOSITY_THRESHOLD:
        # Stretch downstroke horizontally (fluid drag slows load transfer)
        warp_factor = 1.0 + (viscosity - VISCOSITY_THRESHOLD) / 50000.0
        # Apply time-warp to the load signal
        original_indices = np.arange(n)
        warped_indices = original_indices / warp_factor
        warped_indices = np.clip(warped_indices, 0, n - 1)
        downhole_load = np.interp(original_indices, warped_indices, downhole_load)

    # Smooth the result
    window = min(11, n // 2 * 2 + 1)  # Must be odd and <= n
    if window >= 5:
        downhole_load = savgol_filter(downhole_load, window_length=window, polyorder=3)

    # Position is same as surface (simplified 1D model)
    downhole_pos = surface_pos.copy()

    return np.column_stack([downhole_load, downhole_pos])
