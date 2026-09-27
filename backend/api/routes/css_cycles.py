"""CSS Cycle API routes."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.schemas.css_cycle import CssCycleCreate, CssCycleResponse, CssCycleUpdate
from backend.services.well_service import get_well

router = APIRouter(prefix="/api/v1/css-cycles", tags=["css-cycles"])


@router.get("", response_model=list[CssCycleResponse])
async def list_css_cycles(well_id: uuid.UUID | None = None, skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    from backend.models.css_cycle import CssCycle

    stmt = select(CssCycle)
    if well_id is not None:
        stmt = stmt.where(CssCycle.well_id == well_id)
    stmt = stmt.offset(skip).limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/{cycle_id}", response_model=CssCycleResponse)
async def get_css_cycle(cycle_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    from backend.models.css_cycle import CssCycle

    cycle = await db.get(CssCycle, cycle_id)
    if cycle is None:
        raise HTTPException(status_code=404, detail="CSS cycle not found")
    return cycle


@router.post("", response_model=CssCycleResponse, status_code=201)
async def create_css_cycle(data: CssCycleCreate, db: AsyncSession = Depends(get_db)):
    from backend.models.css_cycle import CssCycle

    # Verify well exists
    well = await get_well(db, data.well_id)
    if well is None:
        raise HTTPException(status_code=404, detail="Well not found")

    cycle = CssCycle(**data.model_dump())
    db.add(cycle)
    await db.flush()
    await db.refresh(cycle)
    return cycle


@router.put("/{cycle_id}", response_model=CssCycleResponse)
async def update_css_cycle(cycle_id: uuid.UUID, data: CssCycleUpdate, db: AsyncSession = Depends(get_db)):
    from backend.models.css_cycle import CssCycle

    cycle = await db.get(CssCycle, cycle_id)
    if cycle is None:
        raise HTTPException(status_code=404, detail="CSS cycle not found")
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(cycle, key, value)
    await db.flush()
    await db.refresh(cycle)
    return cycle


@router.delete("/{cycle_id}", status_code=204)
async def delete_css_cycle(cycle_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    from backend.models.css_cycle import CssCycle

    cycle = await db.get(CssCycle, cycle_id)
    if cycle is None:
        raise HTTPException(status_code=404, detail="CSS cycle not found")
    await db.delete(cycle)
