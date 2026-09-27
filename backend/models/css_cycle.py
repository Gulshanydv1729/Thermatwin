"""CSS Cycle ORM model."""

import uuid
from datetime import datetime

from sqlalchemy import String, Integer, Numeric, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.session import Base


class CssCycle(Base):
    __tablename__ = "css_cycles"

    cycle_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    well_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("wells.well_id", ondelete="CASCADE"), nullable=False)
    cycle_number: Mapped[int] = mapped_column(Integer, nullable=False)
    injection_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    injection_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    soak_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    soak_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    production_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    production_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    steam_injected_bbl: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    steam_quality: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)
    injection_pressure_kpa: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)
    reservoir_pressure_kpa: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)
    initial_temp_c: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    target_temp_c: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    well: Mapped["Well"] = relationship(back_populates="css_cycles")  # noqa: F821
