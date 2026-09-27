"""Optimization API routes."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.schemas.optimization import OptimizationResult
from backend.services.optimization_service import get_optimization_recommendation

router = APIRouter(prefix="/api/v1/optimization", tags=["optimization"])


@router.get("/recommend/{well_id}", response_model=OptimizationResult)
async def get_recommendation(well_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await get_optimization_recommendation(db, str(well_id))
    if result is None:
        raise HTTPException(status_code=404, detail="No viscosity data for this well")
    return result
