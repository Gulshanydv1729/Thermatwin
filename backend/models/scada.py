"""SCADA time-series ORM models for TimescaleDB."""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Numeric, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.session import Base


class ScadaSurface(Base):
    __tablename__ = "scada_surface"

    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    well_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    cycle_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("css_cycles.cycle_id"), nullable=True)
    surface_load_n: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    surface_pos_m: Mapped[float | None] = mapped_column(Numeric(6, 3), nullable=True)
    spm: Mapped[float | None] = mapped_column(Numeric(4, 2), nullable=True)
    vfd_frequency_hz: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    motor_current_a: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    motor_voltage_v: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    wellhead_temp_c: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    wellhead_pressure_kpa: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)


class ScadaDownhole(Base):
    __tablename__ = "scada_downhole"

    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    well_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    cycle_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("css_cycles.cycle_id"), nullable=True)
    downhole_load_n: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    downhole_pos_m: Mapped[float | None] = mapped_column(Numeric(6, 3), nullable=True)
    temperature_c: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    viscosity_cp: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    rod_float_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    impact_loading_flag: Mapped[bool] = mapped_column(Boolean, default=False)
