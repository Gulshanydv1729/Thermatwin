"""Well API routes."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.schemas.well import WellCreate, WellResponse, WellUpdate
from backend.services import well_service

router = APIRouter(prefix="/api/v1/wells", tags=["wells"])


@router.get("", response_model=list[WellResponse])
async def list_wells(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    return await well_service.list_wells(db, skip=skip, limit=limit)


@router.get("/{well_id}", response_model=WellResponse)
async def get_well(well_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    well = await well_service.get_well(db, well_id)
    if well is None:
        raise HTTPException(status_code=404, detail="Well not found")
    return well


@router.post("", response_model=WellResponse, status_code=201)
async def create_well(data: WellCreate, db: AsyncSession = Depends(get_db)):
    return await well_service.create_well(db, data)


@router.put("/{well_id}", response_model=WellResponse)
async def update_well(well_id: uuid.UUID, data: WellUpdate, db: AsyncSession = Depends(get_db)):
    well = await well_service.update_well(db, well_id, data)
    if well is None:
        raise HTTPException(status_code=404, detail="Well not found")
    return well


@router.delete("/{well_id}", status_code=204)
async def delete_well(well_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    deleted = await well_service.delete_well(db, well_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Well not found")
