"""Well ORM model."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import String, Numeric, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.session import Base


class Well(Base):
    __tablename__ = "wells"

    well_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    well_name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    field_name: Mapped[str] = mapped_column(String(100), nullable=False, default="Baghewala")
    api_number: Mapped[str | None] = mapped_column(String(20), unique=True, nullable=True)
    latitude: Mapped[float | None] = mapped_column(Numeric(10, 8), nullable=True)
    longitude: Mapped[float | None] = mapped_column(Numeric(11, 8), nullable=True)
    total_depth_m: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)
    perforation_top_m: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)
    perforation_bottom_m: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)
    pump_type: Mapped[str] = mapped_column(String(50), default="SRP")
    rod_string_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    pump_displacement: Mapped[float | None] = mapped_column(Numeric(6, 4), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    css_cycles: Mapped[list["CssCycle"]] = relationship(back_populates="well", cascade="all, delete-orphan")  # noqa: F821
