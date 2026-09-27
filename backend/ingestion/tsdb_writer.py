"""TimescaleDB batch writer — bulk inserts processed SCADA data."""

import logging
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.session import async_session_factory

logger = logging.getLogger(__name__)


async def insert_surface_batch(records: list[dict]) -> int:
    """
    Bulk insert surface SCADA records into TimescaleDB.
    Returns number of rows inserted.
    """
    if not records:
        return 0

    async with async_session_factory() as session:
        for record in records:
            await session.execute(
                text("""
                    INSERT INTO scada_surface (time, well_id, cycle_id, surface_load_n, surface_pos_m,
                                              spm, vfd_frequency_hz, motor_current_a, motor_voltage_v,
                                              wellhead_temp_c, wellhead_pressure_kpa)
                    VALUES (:time, :well_id, :cycle_id, :surface_load_n, :surface_pos_m,
                            :spm, :vfd_frequency_hz, :motor_current_a, :motor_voltage_v,
                            :wellhead_temp_c, :wellhead_pressure_kpa)
                """),
                {
                    "time": record.get("time"),
                    "well_id": record.get("well_id"),
                    "cycle_id": record.get("cycle_id"),
                    "surface_load_n": record.get("surface_load_n"),
                    "surface_pos_m": record.get("surface_pos_m"),
                    "spm": record.get("spm"),
                    "vfd_frequency_hz": record.get("vfd_frequency_hz"),
                    "motor_current_a": record.get("motor_current_a"),
                    "motor_voltage_v": record.get("motor_voltage_v"),
                    "wellhead_temp_c": record.get("wellhead_temp_c"),
                    "wellhead_pressure_kpa": record.get("wellhead_pressure_kpa"),
                },
            )
    return len(records)
