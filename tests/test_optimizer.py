"""Unit tests for core/optimizer.py."""

import pytest
from core.optimizer import calculate_safe_spm


def test_safe_spm_decreases_with_viscosity():
    """Higher viscosity should give lower safe SPM."""
    low_visc = calculate_safe_spm(viscosity=1000)
    high_visc = calculate_safe_spm(viscosity=50000)
    assert high_visc["max_safe_spm"] < low_visc["max_safe_spm"]


def test_safe_spm_clamped_max():
    """Safe SPM should never exceed max_spm."""
    result = calculate_safe_spm(viscosity=100, max_spm=12.0)
    assert result["max_safe_spm"] <= 12.0


def test_safe_spm_clamped_min():
    """Safe SPM should never go below min_spm."""
    result = calculate_safe_spm(viscosity=1000000, min_spm=1.5)
    assert result["max_safe_spm"] >= 1.5


def test_frequency_conversion():
    """VFD frequency should equal SPM * motor_pole_pairs * safety_factor."""
    result = calculate_safe_spm(viscosity=1000, motor_pole_pairs=2, safety_factor=0.9)
    expected_freq = result["max_safe_spm"] * 2 * 0.9
    assert abs(result["recommended_vfd_hz"] - expected_freq) < 0.1


def test_safety_factor_applied():
    """Recommended frequency should be less than raw safe frequency."""
    result = calculate_safe_spm(viscosity=1000, safety_factor=0.9)
    raw_freq = result["max_safe_spm"] * 2  # motor_pole_pairs=2
    assert result["recommended_vfd_hz"] < raw_freq


def test_production_estimate_positive():
    """Production estimate should always be positive."""
    result = calculate_safe_spm(viscosity=50000)
    assert result["production_bpd_estimate"] > 0


def test_reference_viscosity():
    """At reference viscosity, safe SPM should equal max_spm."""
    result = calculate_safe_spm(viscosity=1000, viscosity_ref=1000, max_spm=12.0)
    assert abs(result["max_safe_spm"] - 12.0) < 0.1
