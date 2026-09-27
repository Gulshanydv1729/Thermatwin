"""Equipment specifications ORM model."""

import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import String, Date, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.session import Base


class EquipmentSpec(Base):
    __tablename__ = "equipment_specs"

    spec_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    well_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("wells.well_id", ondelete="CASCADE"), nullable=False)
    equipment_type: Mapped[str] = mapped_column(String(50), nullable=False)
    component_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    manufacturer: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    specifications: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    installed_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
