"""Integration tests for dual-mode pipeline."""

import numpy as np
from core.data_gen import generate_css_cycle
from core.physics_engine import calculate_downhole_card
from core.diagnostics import detect_rod_float
from core.optimizer import calculate_safe_spm


def test_synthetic_mode_pipeline():
    """Pipeline should work with synthetic data (Synthetic Demo mode)."""
    df = generate_css_cycle()
    day_data = df[df["day"] == 15]
    row = day_data.iloc[0]

    surface_load = day_data["surface_load_n"].values
    surface_pos = day_data["surface_pos_m"].values
    viscosity = float(row["viscosity_cp"])

    surface_card = np.column_stack([surface_load, surface_pos])
    downhole_card = calculate_downhole_card(surface_load, surface_pos, viscosity)
    diag = detect_rod_float(downhole_card, surface_card, viscosity)
    opt = calculate_safe_spm(viscosity)

    assert "rod_float" in diag
    assert "severity" in diag
    assert "max_safe_spm" in opt
    assert "recommended_vfd_hz" in opt


def test_live_mode_pipeline():
    """Pipeline should work with live SCADA data structure."""
    # Simulate live data (same structure as MQTT payloads)
    surface_load = np.array([20000, 30000, 40000, 50000, 60000, 70000, 80000, 70000, 60000, 50000,
                             40000, 30000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 70000])
    surface_pos = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35,
                            1.5, 1.65, 1.8, 1.95, 2.1, 2.25, 2.4, 2.55, 2.7, 2.85])
    viscosity = 15000.0

    surface_card = np.column_stack([surface_load, surface_pos])
    downhole_card = calculate_downhole_card(surface_load, surface_pos, viscosity)
    diag = detect_rod_float(downhole_card, surface_card, viscosity)
    opt = calculate_safe_spm(viscosity)

    assert diag["severity"] in ("low", "medium", "high")
    assert opt["max_safe_spm"] >= 1.5


def test_pipeline_output_format_consistency():
    """Both modes should output identical format for downstream functions."""
    # Synthetic data
    df = generate_css_cycle()
    day_data = df[df["day"] == 10]
    syn_load = day_data["surface_load_n"].values
    syn_pos = day_data["surface_pos_m"].values
    syn_visc = float(day_data.iloc[0]["viscosity_cp"])

    # Live data (simulated)
    live_load = np.linspace(20000, 80000, 100)
    live_pos = np.linspace(0, 3, 100)
    live_visc = 10000.0

    # Both should produce valid outputs
    for load, pos, visc in [(syn_load, syn_pos, syn_visc), (live_load, live_pos, live_visc)]:
        surface_card = np.column_stack([load, pos])
        downhole_card = calculate_downhole_card(load, pos, visc)
        diag = detect_rod_float(downhole_card, surface_card, visc)
        opt = calculate_safe_spm(visc)

        assert isinstance(diag["rod_float"], bool)
        assert isinstance(diag["impact_loading"], bool)
        assert diag["severity"] in ("low", "medium", "high")
        assert opt["max_safe_spm"] >= 1.5
        assert opt["recommended_vfd_hz"] > 0
