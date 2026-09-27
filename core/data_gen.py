"""Synthetic data generator for Baghewala field CSS cycle."""

import numpy as np
import pandas as pd


def generate_css_cycle(
    days: int = 30,
    initial_temp: float = 200.0,
    final_temp: float = 50.0,
    initial_viscosity: float = 500.0,
    final_viscosity: float = 50000.0,
    samples_per_day: int = 100,
) -> pd.DataFrame:
    """
    Generate synthetic 30-day thermal/viscosity dataset.

    Returns DataFrame with columns:
        - day: int (0..days-1)
        - temperature_c: float (initial_temp -> final_temp, exponential decay)
        - viscosity_cp: float (initial_viscosity -> final_viscosity, exponential spike)
        - surface_load_n: float (polished rod load, N)
        - surface_pos_m: float (polished rod position, m)
    """
    t = np.linspace(0, days, days * samples_per_day, endpoint=False)

    # Temperature: exponential decay
    k = 0.08
    temperature = final_temp + (initial_temp - final_temp) * np.exp(-k * t)

    # Viscosity: Arrhenius-like exponential spike
    T_kelvin = temperature + 273.15
    T_initial_kelvin = initial_temp + 273.15
    T_final_kelvin = final_temp + 273.15
    alpha = np.log(final_viscosity / initial_viscosity) / (1.0 / T_final_kelvin - 1.0 / T_initial_kelvin)
    viscosity = initial_viscosity * np.exp(alpha * (1.0 / T_kelvin - 1.0 / T_initial_kelvin))

    # Surface load: sinusoidal with upstroke peak ~80 kN, downstroke trough ~20 kN
    stroke_phase = 2 * np.pi * t
    surface_load = 50000 + 30000 * np.sin(stroke_phase)

    # Surface position: triangular wave 0 -> 3 m stroke length
    position_phase = t % 1.0
    surface_pos = 3.0 * np.where(position_phase < 0.5, 2 * position_phase, 2 - 2 * position_phase)

    day_idx = np.floor(t).astype(int)

    df = pd.DataFrame({
        "day": day_idx,
        "temperature_c": np.round(temperature, 2),
        "viscosity_cp": np.round(viscosity, 2),
        "surface_load_n": np.round(surface_load, 2),
        "surface_pos_m": np.round(surface_pos, 4),
    })

    return df


if __name__ == "__main__":
    df = generate_css_cycle()
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"\nTemperature range: {df['temperature_c'].min():.1f} - {df['temperature_c'].max():.1f} C")
    print(f"Viscosity range: {df['viscosity_cp'].min():.1f} - {df['viscosity_cp'].max():.1f} cP")
