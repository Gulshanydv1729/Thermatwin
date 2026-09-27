"""Pydantic schemas for SCADA telemetry API."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ScadaSurfacePoint(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    time: datetime
    well_id: uuid.UUID
    cycle_id: uuid.UUID | None = None
    surface_load_n: float | None = None
    surface_pos_m: float | None = None
    spm: float | None = None
    vfd_frequency_hz: float | None = None
    motor_current_a: float | None = None
    motor_voltage_v: float | None = None
    wellhead_temp_c: float | None = None
    wellhead_pressure_kpa: float | None = None


class ScadaDownholePoint(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    time: datetime
    well_id: uuid.UUID
    cycle_id: uuid.UUID | None = None
    downhole_load_n: float | None = None
    downhole_pos_m: float | None = None
    temperature_c: float | None = None
    viscosity_cp: float | None = None
    rod_float_flag: bool = False
    impact_loading_flag: bool = False
