"""Unit tests for ORM models."""

import uuid
from datetime import datetime

from backend.models.well import Well
from backend.models.css_cycle import CssCycle


def test_well_model_creation():
    well = Well(well_name="BHW-001", field_name="Baghewala", pump_type="SRP")
    assert well.well_name == "BHW-001"
    assert well.field_name == "Baghewala"
    assert well.pump_type == "SRP"
    # well_id is populated at flush time by the database default
    assert well.well_id is None or isinstance(well.well_id, uuid.UUID)


def test_well_model_with_full_data():
    well = Well(
        well_name="BHW-002",
        api_number="API-12345",
        latitude=28.12345678,
        longitude=72.87654321,
        total_depth_m=1500.0,
        perforation_top_m=1200.0,
        perforation_bottom_m=1450.0,
        pump_displacement=0.5,
    )
    assert well.api_number == "API-12345"
    assert float(well.total_depth_m) == 1500.0


def test_css_cycle_model_creation():
    well_id = uuid.uuid4()
    cycle = CssCycle(
        well_id=well_id,
        cycle_number=1,
        injection_start=datetime(2026, 1, 1),
        initial_temp_c=200.0,
        target_temp_c=50.0,
        status="active",
    )
    assert cycle.cycle_number == 1
    assert cycle.status == "active"
    assert float(cycle.initial_temp_c) == 200.0
