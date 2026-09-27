"""Pydantic schemas for Optimization API."""

from pydantic import BaseModel


class OptimizationResult(BaseModel):
    max_safe_spm: float
    recommended_vfd_hz: float
    production_bpd_estimate: float
