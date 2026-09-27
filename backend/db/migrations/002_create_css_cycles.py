"""Create css_cycles table.

Revision ID: 002
Revises: 001
Create Date: 2026-09-28
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "css_cycles",
        sa.Column("cycle_id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("well_id", UUID(as_uuid=True), sa.ForeignKey("wells.well_id", ondelete="CASCADE"), nullable=False),
        sa.Column("cycle_number", sa.Integer(), nullable=False),
        sa.Column("injection_start", sa.TIMESTAMPTZ(), nullable=False),
        sa.Column("injection_end", sa.TIMESTAMPTZ()),
        sa.Column("soak_start", sa.TIMESTAMPTZ()),
        sa.Column("soak_end", sa.TIMESTAMPTZ()),
        sa.Column("production_start", sa.TIMESTAMPTZ()),
        sa.Column("production_end", sa.TIMESTAMPTZ()),
        sa.Column("steam_injected_bbl", sa.Numeric(10, 2)),
        sa.Column("steam_quality", sa.Numeric(4, 3)),
        sa.Column("injection_pressure_kpa", sa.Numeric(8, 2)),
        sa.Column("reservoir_pressure_kpa", sa.Numeric(8, 2)),
        sa.Column("initial_temp_c", sa.Numeric(6, 2)),
        sa.Column("target_temp_c", sa.Numeric(6, 2)),
        sa.Column("status", sa.String(20), server_default="'active'"),
        sa.Column("created_at", sa.TIMESTAMPTZ(), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.TIMESTAMPTZ(), server_default=sa.text("NOW()")),
        sa.UniqueConstraint("well_id", "cycle_number", name="uq_well_cycle"),
    )
    op.create_index("idx_css_well", "css_cycles", ["well_id"])
    op.create_index("idx_css_status", "css_cycles", ["status"])
    op.create_index("idx_css_dates", "css_cycles", ["production_start", "production_end"])


def downgrade() -> None:
    op.drop_index("idx_css_dates", table_name="css_cycles")
    op.drop_index("idx_css_status", table_name="css_cycles")
    op.drop_index("idx_css_well", table_name="css_cycles")
    op.drop_table("css_cycles")
