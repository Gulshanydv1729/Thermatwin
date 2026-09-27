"""Create scada_downhole hypertable.

Revision ID: 004
Revises: 003
Create Date: 2026-09-28
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scada_downhole",
        sa.Column("time", sa.TIMESTAMPTZ(), nullable=False),
        sa.Column("well_id", UUID(as_uuid=True), nullable=False),
        sa.Column("cycle_id", UUID(as_uuid=True), nullable=True),
        sa.Column("downhole_load_n", sa.Numeric(10, 2)),
        sa.Column("downhole_pos_m", sa.Numeric(6, 3)),
        sa.Column("temperature_c", sa.Numeric(6, 2)),
        sa.Column("viscosity_cp", sa.Numeric(10, 2)),
        sa.Column("rod_float_flag", sa.Boolean(), server_default=sa.false()),
        sa.Column("impact_loading_flag", sa.Boolean(), server_default=sa.false()),
    )
    op.execute("SELECT create_hypertable('scada_downhole', 'time')")
    op.create_index("idx_downhole_well_time", "scada_downhole", ["well_id", sa.text("time DESC")])


def downgrade() -> None:
    op.drop_index("idx_downhole_well_time", table_name="scada_downhole")
    op.drop_table("scada_downhole")
