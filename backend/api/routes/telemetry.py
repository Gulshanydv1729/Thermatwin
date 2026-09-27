"""Telemetry API routes."""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.schemas.scada import ScadaSurfacePoint, ScadaDownholePoint

router = APIRouter(prefix="/api/v1/telemetry", tags=["telemetry"])


@router.get("/surface", response_model=list[ScadaSurfacePoint])
async def get_surface_telemetry(
    well_id: uuid.UUID,
    start: datetime | None = Query(None),
    end: datetime | None = Query(None),
    limit: int = Query(1000, le=10000),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select
    from backend.models.scada import ScadaSurface

    stmt = select(ScadaSurface).where(ScadaSurface.well_id == well_id)
    if start is not None:
        stmt = stmt.where(ScadaSurface.time >= start)
    if end is not None:
        stmt = stmt.where(ScadaSurface.time <= end)
    stmt = stmt.order_by(ScadaSurface.time.desc()).limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/downhole", response_model=list[ScadaDownholePoint])
async def get_downhole_telemetry(
    well_id: uuid.UUID,
    start: datetime | None = Query(None),
    end: datetime | None = Query(None),
    limit: int = Query(1000, le=10000),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select
    from backend.models.scada import ScadaDownhole

    stmt = select(ScadaDownhole).where(ScadaDownhole.well_id == well_id)
    if start is not None:
        stmt = stmt.where(ScadaDownhole.time >= start)
    if end is not None:
        stmt = stmt.where(ScadaDownhole.time <= end)
    stmt = stmt.order_by(ScadaDownhole.time.desc()).limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())
