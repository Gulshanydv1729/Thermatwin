"""Pydantic schemas for Diagnostics API."""

from pydantic import BaseModel


class DiagnosticsResult(BaseModel):
    rod_float: bool
    impact_loading: bool
    delay_pct: float
    severity: str  # "low" | "medium" | "high"
