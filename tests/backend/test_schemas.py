"""Unit tests for Pydantic schemas."""

import uuid
from datetime import datetime

from backend.schemas.well import WellCreate, WellUpdate
from backend.schemas.css_cycle import CssCycleCreate
from backend.schemas.diagnostics import DiagnosticsResult
from backend.schemas.optimization import OptimizationResult


def test_well_create_schema():
    data = WellCreate(well_name="BHW-001")
    assert data.well_name == "BHW-001"
    assert data.field_name == "Baghewala"
    assert data.pump_type == "SRP"


def test_well_update_schema():
    data = WellUpdate(well_name="BHW-002")
    assert data.well_name == "BHW-002"
    # Unset fields should be None
    assert data.field_name is None


def test_css_cycle_create_schema():
    data = CssCycleCreate(
        well_id=uuid.uuid4(),
        cycle_number=1,
        injection_start=datetime(2026, 1, 1),
    )
    assert data.cycle_number == 1
    assert data.status == "active"


def test_diagnostics_result_schema():
    data = DiagnosticsResult(
        rod_float=True,
        impact_loading=False,
        delay_pct=7.5,
        severity="medium",
    )
    assert data.rod_float is True
    assert data.severity == "medium"


def test_optimization_result_schema():
    data = OptimizationResult(
        max_safe_spm=4.5,
        recommended_vfd_hz=8.1,
        production_bpd_estimate=3240.0,
    )
    assert data.max_safe_spm == 4.5
    assert data.recommended_vfd_hz == 8.1
