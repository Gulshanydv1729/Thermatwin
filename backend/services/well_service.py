"""Well service — CRUD operations for wells."""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.models.well import Well
from backend.schemas.well import WellCreate, WellUpdate


async def list_wells(db: AsyncSession, skip: int = 0, limit: int = 100) -> list[Well]:
    result = await db.execute(select(Well).offset(skip).limit(limit))
    return list(result.scalars().all())


async def get_well(db: AsyncSession, well_id: uuid.UUID) -> Well | None:
    return await db.get(Well, well_id)


async def create_well(db: AsyncSession, data: WellCreate) -> Well:
    well = Well(**data.model_dump())
    db.add(well)
    await db.flush()
    await db.refresh(well)
    return well


async def update_well(db: AsyncSession, well_id: uuid.UUID, data: WellUpdate) -> Well | None:
    well = await db.get(Well, well_id)
    if well is None:
        return None
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(well, key, value)
    await db.flush()
    await db.refresh(well)
    return well


async def delete_well(db: AsyncSession, well_id: uuid.UUID) -> bool:
    well = await db.get(Well, well_id)
    if well is None:
        return False
    await db.delete(well)
    return True
