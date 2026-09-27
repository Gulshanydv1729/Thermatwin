"""Unit tests for stream processor."""

from backend.ingestion.stream_processor import StreamProcessor


def test_load_conversion():
    sp = StreamProcessor()
    assert abs(sp._convert_load(1000.0) - 4448.22) < 0.1


def test_temp_conversion():
    sp = StreamProcessor()
    assert abs(sp._convert_temp(212.0) - 100.0) < 0.01


def test_pressure_conversion():
    sp = StreamProcessor()
    assert abs(sp._convert_pressure(14.696) - 101.325) < 0.1


def test_outlier_filter():
    sp = StreamProcessor()
    raw = {"surface_load_lbf": 50000.0}  # ~222 kN, exceeds 200 kN threshold
    result = sp.process(raw)
    assert result is None


def test_valid_message():
    sp = StreamProcessor()
    raw = {
        "timestamp": "2026-09-28T12:00:00Z",
        "well_id": "123e4567-e89b-12d3-a456-426614174000",
        "surface_load_lbf": 10000.0,
        "surface_pos_m": 1.5,
        "spm": 6.0,
        "vfd_freq_hz": 30.0,
        "motor_current_a": 50.0,
        "motor_voltage_v": 480.0,
        "wellhead_temp_f": 150.0,
        "wellhead_pressure_psi": 100.0,
    }
    result = sp.process(raw)
    assert result is not None
    assert result["well_id"] == "123e4567-e89b-12d3-a456-426614174000"
    assert result["spm"] == 6.0
