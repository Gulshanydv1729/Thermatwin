"""Unit tests for core/data_gen.py — restored for Synthetic Demo mode."""

import numpy as np
import pytest
from core.data_gen import generate_css_cycle


def test_dataframe_shape():
    """Default params should produce (3000, 5) DataFrame."""
    df = generate_css_cycle()
    assert df.shape == (3000, 5)


def test_temperature_range():
    """Temperature should decay from ~200 to ~50."""
    df = generate_css_cycle()
    assert df["temperature_c"].min() >= 49.0
    assert df["temperature_c"].max() <= 201.0


def test_viscosity_monotonic():
    """Viscosity should increase monotonically over time."""
    df = generate_css_cycle()
    daily_viscosity = df.groupby("day")["viscosity_cp"].mean()
    assert daily_viscosity.is_monotonic_increasing


def test_no_nan_values():
    """No NaN values in any column."""
    df = generate_css_cycle()
    assert not df.isnull().any().any()


def test_custom_days():
    """Custom days parameter should adjust row count."""
    df = generate_css_cycle(days=10)
    assert df.shape == (1000, 5)
    assert df["day"].nunique() == 10


def test_viscosity_range():
    """Viscosity should span from ~500 to near 50000 cP."""
    df = generate_css_cycle()
    assert df["viscosity_cp"].min() >= 400.0
    assert df["viscosity_cp"].max() >= 25000.0
