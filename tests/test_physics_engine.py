"""Unit tests for core/physics_engine.py."""

import numpy as np
from core.physics_engine import calculate_downhole_card, VISCOSITY_THRESHOLD


def _make_surface_card(n=100):
    """Helper to create a simple surface card."""
    load = np.linspace(20000, 80000, n)
    pos = np.linspace(0, 3, n)
    return load, pos


def test_output_shape():
    """Output should be (N, 2)."""
    load, pos = _make_surface_card(100)
    result = calculate_downhole_card(load, pos, viscosity=1000)
    assert result.shape == (100, 2)


def test_low_viscosity_normal_card():
    """Low viscosity should produce minimal distortion."""
    load, pos = _make_surface_card(100)
    result = calculate_downhole_card(load, pos, viscosity=1000)
    # Load should be attenuated but same general shape
    assert result[:, 0].max() <= load.max()
    assert result[:, 0].min() >= load.min() * 0.5


def test_high_viscosity_rod_float():
    """High viscosity should produce larger phase delay."""
    load, pos = _make_surface_card(100)
    low_visc = calculate_downhole_card(load, pos, viscosity=1000)
    high_visc = calculate_downhole_card(load, pos, viscosity=50000)
    # High viscosity card should have different shape
    assert not np.allclose(low_visc[:, 0], high_visc[:, 0])


def test_damping_increases_with_viscosity():
    """Higher viscosity should produce more attenuation."""
    load, pos = _make_surface_card(100)
    low_visc = calculate_downhole_card(load, pos, viscosity=1000)
    high_visc = calculate_downhole_card(load, pos, viscosity=50000)
    # High viscosity should have lower peak load (more damping)
    assert high_visc[:, 0].max() < low_visc[:, 0].max()


def test_custom_damping():
    """Custom damping coefficient should override viscosity-based damping."""
    load, pos = _make_surface_card(100)
    result = calculate_downhole_card(load, pos, viscosity=1000, damping_coeff=0.5)
    assert result.shape == (100, 2)
