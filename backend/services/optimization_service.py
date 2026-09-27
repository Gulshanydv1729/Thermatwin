"""Optimization service — async wrapper around core/optimizer.py."""

from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.models.scada import ScadaDownhole
from backend.schemas.optimization import OptimizationResult


async def get_optimization_recommendation(
    db: AsyncSession,
    well_id: str,
) -> OptimizationResult | None:
    """Fetch latest viscosity and compute SPM/VFD recommendation."""
    from core.optimizer import calculate_safe_spm

    result = await db.execute(
        select(ScadaDownhole)
        .where(ScadaDownhole.well_id == well_id)
        .order_by(ScadaDownhole.time.desc())
        .limit(1)
    )
    record = result.scalar_one_or_none()
    if record is None or record.viscosity_cp is None:
        return None

    opt = calculate_safe_spm(float(record.viscosity_cp))
    return OptimizationResult(**opt)
