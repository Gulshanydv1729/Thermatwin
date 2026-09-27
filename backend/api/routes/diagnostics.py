"""Diagnostics API routes."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.schemas.diagnostics import DiagnosticsResult
from backend.services.diagnostics_service import get_latest_diagnostics

router = APIRouter(prefix="/api/v1/diagnostics", tags=["diagnostics"])


@router.get("/latest/{well_id}", response_model=DiagnosticsResult)
async def get_latest(well_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await get_latest_diagnostics(db, str(well_id))
    if result is None:
        raise HTTPException(status_code=404, detail="No telemetry data for this well")
    return result
