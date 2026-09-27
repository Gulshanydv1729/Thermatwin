"""Pydantic schemas for Well API."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class WellBase(BaseModel):
    well_name: str
    field_name: str = "Baghewala"
    api_number: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    total_depth_m: float | None = None
    perforation_top_m: float | None = None
    perforation_bottom_m: float | None = None
    pump_type: str = "SRP"
    rod_string_config: dict[str, Any] | None = None
    pump_displacement: float | None = None


class WellCreate(WellBase):
    pass


class WellUpdate(BaseModel):
    well_name: str | None = None
    field_name: str | None = None
    api_number: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    total_depth_m: float | None = None
    perforation_top_m: float | None = None
    perforation_bottom_m: float | None = None
    pump_type: str | None = None
    rod_string_config: dict[str, Any] | None = None
    pump_displacement: float | None = None


class WellResponse(WellBase):
    model_config = ConfigDict(from_attributes=True)

    well_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
