"""Diagnostics service — async wrapper around core/diagnostics.py."""

from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy import func

from backend.models.scada import ScadaSurface, ScadaDownhole
from backend.schemas.diagnostics import DiagnosticsResult

if TYPE_CHECKING:
    import numpy as np


async def get_latest_diagnostics(
    db: AsyncSession,
    well_id: str,
) -> DiagnosticsResult | None:
    """
    Fetch latest downhole and surface card data for a well,
    then run diagnostics from core/diagnostics.py.
    """
    import numpy as np
    from core.diagnostics import detect_rod_float
    from core.physics_engine import calculate_downhole_card

    # Fetch latest downhole record
    result = await db.execute(
        select(ScadaDownhole)
        .where(ScadaDownhole.well_id == well_id)
        .order_by(ScadaDownhole.time.desc())
        .limit(1)
    )
    downhole_record = result.scalar_one_or_none()
    if downhole_record is None:
        return None

    # Fetch corresponding surface records (last 1 second of data)
    result = await db.execute(
        select(ScadaSurface)
        .where(ScadaSurface.well_id == well_id)
        .order_by(ScadaSurface.time.desc())
        .limit(100)
    )
    surface_records = result.scalars().all()
    if not surface_records:
        return None

    surface_load = np.array([float(r.surface_load_n or 0) for r in reversed(surface_records)])
    surface_pos = np.array([float(r.surface_pos_m or 0) for r in reversed(surface_records)])
    surface_card = np.column_stack([surface_load, surface_pos])

    downhole_card = calculate_downhole_card(
        surface_load,
        surface_pos,
        float(downhole_record.viscosity_cp or 1000),
    )

    diag = detect_rod_float(downhole_card, surface_card, float(downhole_record.viscosity_cp or 1000))
    return DiagnosticsResult(**diag)
