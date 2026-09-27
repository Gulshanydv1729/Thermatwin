"""Create scada_surface hypertable.

Revision ID: 003
Revises: 002
Create Date: 2026-09-28
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scada_surface",
        sa.Column("time", sa.TIMESTAMPTZ(), nullable=False),
        sa.Column("well_id", UUID(as_uuid=True), nullable=False),
        sa.Column("cycle_id", UUID(as_uuid=True), nullable=True),
        sa.Column("surface_load_n", sa.Numeric(10, 2)),
        sa.Column("surface_pos_m", sa.Numeric(6, 3)),
        sa.Column("spm", sa.Numeric(4, 2)),
        sa.Column("vfd_frequency_hz", sa.Numeric(5, 2)),
        sa.Column("motor_current_a", sa.Numeric(6, 2)),
        sa.Column("motor_voltage_v", sa.Numeric(6, 2)),
        sa.Column("wellhead_temp_c", sa.Numeric(6, 2)),
        sa.Column("wellhead_pressure_kpa", sa.Numeric(8, 2)),
    )
    op.execute("SELECT create_hypertable('scada_surface', 'time')")
    op.create_index("idx_scada_well_time", "scada_surface", ["well_id", sa.text("time DESC")])


def downgrade() -> None:
    op.drop_index("idx_scada_well_time", table_name="scada_surface")
    op.drop_table("scada_surface")
