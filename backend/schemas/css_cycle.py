"""Pydantic schemas for CSS Cycle API."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class CssCycleBase(BaseModel):
    cycle_number: int
    injection_start: datetime
    injection_end: datetime | None = None
    soak_start: datetime | None = None
    soak_end: datetime | None = None
    production_start: datetime | None = None
    production_end: datetime | None = None
    steam_injected_bbl: float | None = None
    steam_quality: float | None = None
    injection_pressure_kpa: float | None = None
    reservoir_pressure_kpa: float | None = None
    initial_temp_c: float | None = None
    target_temp_c: float | None = None
    status: str = "active"


class CssCycleCreate(CssCycleBase):
    well_id: uuid.UUID


class CssCycleUpdate(BaseModel):
    cycle_number: int | None = None
    injection_start: datetime | None = None
    injection_end: datetime | None = None
    soak_start: datetime | None = None
    soak_end: datetime | None = None
    production_start: datetime | None = None
    production_end: datetime | None = None
    steam_injected_bbl: float | None = None
    steam_quality: float | None = None
    injection_pressure_kpa: float | None = None
    reservoir_pressure_kpa: float | None = None
    initial_temp_c: float | None = None
    target_temp_c: float | None = None
    status: str | None = None


class CssCycleResponse(CssCycleBase):
    model_config = ConfigDict(from_attributes=True)

    cycle_id: uuid.UUID
    well_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
