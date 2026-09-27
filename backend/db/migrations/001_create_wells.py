"""Create wells table.

Revision ID: 001
Revises:
Create Date: 2026-09-28
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "wells",
        sa.Column("well_id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("well_name", sa.String(50), nullable=False, unique=True),
        sa.Column("field_name", sa.String(100), nullable=False, server_default="'Baghewala'"),
        sa.Column("api_number", sa.String(20), unique=True),
        sa.Column("latitude", sa.Numeric(10, 8)),
        sa.Column("longitude", sa.Numeric(11, 8)),
        sa.Column("total_depth_m", sa.Numeric(8, 2)),
        sa.Column("perforation_top_m", sa.Numeric(8, 2)),
        sa.Column("perforation_bottom_m", sa.Numeric(8, 2)),
        sa.Column("pump_type", sa.String(50), server_default="'SRP'"),
        sa.Column("rod_string_config", JSONB),
        sa.Column("pump_displacement", sa.Numeric(6, 4)),
        sa.Column("created_at", sa.TIMESTAMPTZ(), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.TIMESTAMPTZ(), server_default=sa.text("NOW()")),
    )
    op.create_index("idx_wells_field", "wells", ["field_name"])


def downgrade() -> None:
    op.drop_index("idx_wells_field", table_name="wells")
    op.drop_table("wells")
