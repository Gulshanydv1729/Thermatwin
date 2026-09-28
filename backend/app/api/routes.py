"""REST endpoints and telemetry ingestion for the ThermaTwin service.

All routes are mounted under ``/api/v1``.  The WebSocket stream lives in
:mod:`backend.app.api.websocket`; this module owns the request/response side and
the shared application state.
"""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.app.ai.dyno_classifier import (
    CardLabel,
    DynoCardClassifier,
    synthesize_card,
)
from backend.app.ai.sor_optimizer import CSSSchedule, CSOScheduleOptimizer
from backend.app.core.config import PUMPING_UNIT, RESERVOIR, get_config
from backend.app.engine.digital_twin import DigitalTwin
from backend.app.engine.state_estimator import TelemetryPacket
from backend.app.physics.gibbs_solver import baghewala_rod_string
from backend.app.physics.hydraulics import compute_traverse
from backend.app.physics.rheology import baghewala_crude
from backend.app.physics.thermal_reservoir import ThermalReservoirModel

__all__ = [
    "router",
    "get_twin",
    "get_optimizer",
    "get_classifier",
    "SetpointRequest",
    "AutonomyRequest",
    "DiagnoseRequest",
    "TraverseRequest",
    "ViscosityRequest",
]

router = APIRouter(prefix="/api/v1", tags=["thermatwin"])

_TWIN: Optional[DigitalTwin] = None
_OPTIMIZER: Optional[CSOScheduleOptimizer] = None
_CLASSIFIER: Optional[DynoCardClassifier] = None
_OPTIMIZER_LOCK = asyncio.Lock()


def get_twin() -> DigitalTwin:
    """Return the process-wide twin, constructing it on first use.

    The twin is created lazily so that importing the application does not pay
    the cost of building the physics model, and so that the classifier is only
    trained when a caller actually needs a diagnosis.
    """
    global _TWIN
    if _TWIN is None:
        _TWIN = DigitalTwin()
    return _TWIN


def get_optimizer() -> CSOScheduleOptimizer:
    """Return the CSS schedule optimiser, constructing it on first use."""
    global _OPTIMIZER
    if _OPTIMIZER is None:
        _OPTIMIZER = CSOScheduleOptimizer()
    return _OPTIMIZER


def get_classifier() -> DynoCardClassifier:
    """Return the trained dyno card classifier, training it if required."""
    global _CLASSIFIER
    if _CLASSIFIER is None:
        config = get_config()
        _CLASSIFIER = DynoCardClassifier(train=False)
        weights = config.weights_dir / "dyno_classifier.pt"
        if weights.exists():
            _CLASSIFIER.load(weights)
        else:
            _CLASSIFIER.train_and_save(weights, epochs=10, samples_per_label=250)
    return _CLASSIFIER


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SetpointRequest(BaseModel):
    """Operator- or autonomous-approved VFD setpoint."""

    spm: float = Field(..., description="Target strokes per minute")
    stroke_length_m: float = Field(..., description="Target stroke length, m")
    autonomous: bool = Field(False, description="Whether the twin may apply it")


class AutonomyRequest(BaseModel):
    """Autonomous closed-loop toggle."""

    enabled: bool = Field(..., description="Enable the autonomous VFD loop")


class DiagnoseRequest(BaseModel):
    """Surface card to be inverted and classified."""

    position_m: List[float] = Field(..., min_length=8)
    load_n: List[float] = Field(..., min_length=8)
    duration_s: float = Field(6.0, gt=0.0, description="Length of the crank cycle")
    temperature_c: Optional[float] = Field(
        None, description="Local tubing temperature; defaults to the twin state"
    )


class TraverseRequest(BaseModel):
    """Wellbore traverse query."""

    mass_flow_kg_s: Optional[float] = Field(
        None, gt=0.0, description="Defaults to the twin's current rate"
    )
    segments: int = Field(40, ge=10, le=200)
    water_cut: float = Field(0.12, ge=0.0, le=1.0)


class ViscosityRequest(BaseModel):
    """Temperature-viscosity query."""

    temperature_c: float = Field(..., ge=-20.0, le=400.0)


class TelemetrySample(BaseModel):
    """One SCADA scan replayed from a historian or the synthetic streamer."""

    timestamp_s: float
    cycle: int = Field(1, ge=1)
    phase: str = Field("PRODUCTION")
    day: float = 0.0
    spm: float = Field(..., gt=0.0)
    stroke_m: float = Field(..., gt=0.0)
    water_cut: float = Field(0.12, ge=0.0, le=1.0)
    position_m: List[float] = Field(..., min_length=8)
    load_n: List[float] = Field(..., min_length=8)
    casing_head_pressure_mpa: float
    wellhead_temperature_c: float
    surface_flow_m3_per_day: float = Field(..., ge=0.0)
    bottom_hole_temperature_c: Optional[float] = None
    pump_fillage: Optional[float] = Field(None, ge=0.0)
    cumulative_sor: Optional[float] = None


class TelemetryBatch(BaseModel):
    """A batch of scans to replay through the twin."""

    samples: List[TelemetrySample] = Field(..., min_length=1, max_length=2000)
    apply_setpoints: bool = Field(
        False, description="Let the twin apply its own VFD setpoints while replaying"
    )


# ---------------------------------------------------------------------------
# Metadata and health
# ---------------------------------------------------------------------------


@router.get("/health")
async def health() -> Dict[str, object]:
    """Liveness probe."""
    return {
        "status": "ok",
        "service": "thermatwin",
        "specification": "SIH26120",
        "field": "Baghewala, Oil India Limited",
    }


@router.get("/config")
async def configuration() -> Dict[str, object]:
    """Full field configuration, as JSON."""
    return get_config().as_dict()


# ---------------------------------------------------------------------------
# Physics queries
# ---------------------------------------------------------------------------


@router.get("/rheology/viscosity")
async def viscosity(
    temperature_c: float = Query(
        default=RESERVOIR.reservoir_temperature_c,
        ge=-20.0,
        le=400.0,
        description="Temperature at which to evaluate the dead-oil viscosity, degC",
    ),
) -> Dict[str, object]:
    """Dead-oil viscosity at a temperature, with the fitted Walther coefficients."""
    crude = baghewala_crude()
    temperature = float(temperature_c)
    curve_t, curve_mu = crude.curve(20.0, 260.0, points=120)
    return {
        "temperature_c": temperature,
        "viscosity_cp": crude.dynamic_viscosity_cp(temperature),
        "viscosity_pa_s": crude.dynamic_viscosity_pa_s(temperature),
        "walther_A": crude.A,
        "walther_B": crude.B,
        "curve": {"temperature_c": curve_t, "viscosity_cp": curve_mu},
    }


@router.get("/wellbore/traverse")
async def traverse(
    mass_flow_kg_s: Optional[float] = Query(
        default=None,
        gt=0.0,
        description="Mass flow; defaults to the twin's current production rate",
    ),
    segments: int = Query(40, ge=10, le=200, description="Traverse resolution"),
    water_cut: float = Query(0.12, ge=0.0, le=1.0, description="Produced water cut"),
) -> Dict[str, object]:
    """Pressure and temperature traverse from the pump to the wellhead."""
    twin = get_twin()
    if mass_flow_kg_s is None:
        result = twin.traverse(segments=segments)
    else:
        result = compute_traverse(
            segments=segments,
            mass_flow_kg_s=float(mass_flow_kg_s),
            water_cut=water_cut,
            bottom_hole_temperature_c=twin.reservoir.bottom_hole_temperature_c,
        )
    return result.as_dict()


@router.post("/srp/diagnose")
async def diagnose(request_model: DiagnoseRequest) -> Dict[str, object]:
    """Invert a surface card to the pump and classify the operating state."""
    twin = get_twin()
    try:
        card = twin.reconstruct_card(
            request_model.position_m,
            request_model.load_n,
            request_model.duration_s,
            request_model.temperature_c,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    prediction = get_classifier().predict(card.position_m, card.load_n)
    return {
        "card": card.as_dict(),
        "diagnosis": prediction.as_dict(),
        "alarms": twin.alarms,
    }


@router.get("/thermal/cycle")
async def thermal_cycle(
    production_days: float = 60.0, cycles: int = 1
) -> Dict[str, object]:
    """Simulate a CSS cycle and return the trajectory summary."""
    if not 0.0 < production_days <= 400.0:
        raise HTTPException(status_code=422, detail="production_days must lie in (0, 400]")
    if not 1 <= cycles <= 12:
        raise HTTPException(status_code=422, detail="cycles must lie in [1, 12]")
    model = ThermalReservoirModel()
    started = time.perf_counter()
    programme = model.simulate_programme(cycles=cycles, production_days=production_days)
    producing = [
        state
        for state in programme.states
        if state.phase.value == "PRODUCTION"
    ]
    return {
        "cycles_completed": programme.cycles_completed,
        "cumulative_sor": programme.cumulative_sor,
        "steam_tonnes": programme.total_steam_tonnes,
        "oil_tonnes": programme.total_oil_tonnes,
        "oil_bbl": programme.total_oil_tonnes / model.oil_specific_gravity * 6.2898,
        "final_bht_c": programme.final_bht_c,
        "economic_cutoff_cycles": programme.economic_cutoff_cycles,
        "net_revenue_usd": programme.net_revenue_usd,
        "energy_balance_residual": model.energy_balance(
            model.simulate_cycle(production_days=min(production_days, 5.0))
        ),
        "elapsed_ms": (time.perf_counter() - started) * 1e3,
        "final_state": {
            "pump_fillage": producing[-1].pump_fillage if producing else 0.0,
            "bottom_hole_temperature_c": programme.final_bht_c,
        },
    }


# ---------------------------------------------------------------------------
# SCADA ingestion
# ---------------------------------------------------------------------------


@router.post("/telemetry/ingest")
async def ingest(batch: TelemetryBatch) -> Dict[str, object]:
    """Replay a batch of SCADA scans through the twin.

    A historian replay is a bulk upload rather than a real-time push, so it is
    accepted here as a batch and reduced to the frames the dashboard would have
    drawn.  Only the first and last frames are returned in full -- they are the
    ones an operator needs -- with a per-sample summary for the rest, because a
    full cycle is thousands of scans and the card arrays alone would be tens of
    megabytes.

    The reservoir state is advanced along with the scans, so a replayed
    lifecycle actually drives the CSS model rather than replaying a frozen one.
    """
    twin = get_twin()
    if batch.apply_setpoints:
        twin.set_autonomous(True)
    elif twin.autonomous:
        twin.set_autonomous(False)

    history: List[Dict[str, object]] = []
    first_frame: Optional[Dict[str, object]] = None
    last_frame: Optional[Dict[str, object]] = None
    latencies: List[float] = []
    alarms: List[str] = []

    advanced_to_day: Optional[float] = None
    for index, sample in enumerate(batch.samples):
        # The reservoir state is advanced only when the replay crosses into a
        # new day.  A scan is one crank cycle, so a full lifecycle is thousands
        # of samples over a few dozen days; simulating the CSS model per scan
        # would repeat the same work hundreds of times.
        day = round(max(sample.day, 0.0), 1)
        if advanced_to_day is None or abs(day - advanced_to_day) > 0.05:
            try:
                programme = twin.css_model.simulate_cycle(
                    injection_days=twin.start_day, production_days=max(day, 0.25)
                )
                producing = [
                    state
                    for state in programme.states
                    if state.phase.value == "PRODUCTION"
                ]
                if producing:
                    twin.reservoir = producing[-1]
                advanced_to_day = day
            except ValueError:  # pragma: no cover - defensive
                pass

        twin.apply_setpoint(sample.spm, sample.stroke_m)
        packet = TelemetryPacket(
            timestamp_s=sample.timestamp_s,
            casing_head_pressure_pa=sample.casing_head_pressure_mpa * 1e6,
            wellhead_temperature_c=sample.wellhead_temperature_c,
            surface_flow_m3_per_day=sample.surface_flow_m3_per_day,
            card_min_load_n=float(min(sample.load_n)),
            card_max_load_n=float(max(sample.load_n)),
            spm=sample.spm,
            stroke_m=sample.stroke_m,
            water_cut=sample.water_cut,
        )
        try:
            snapshot = twin.step(
                packet,
                surface_position_m=sample.position_m,
                surface_load_n=sample.load_n,
                duration_s=60.0 / max(sample.spm, 1e-6),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        frame = snapshot.as_frame(cycle=sample.cycle, well_name="BAG-17")
        frame["alarms"] = twin.alarms
        latencies.append(snapshot.latency_ms)
        alarms = twin.alarms
        if index == 0:
            first_frame = frame
        last_frame = frame
        history.append(
            {
                "timestamp_s": sample.timestamp_s,
                "phase": sample.phase,
                "day": sample.day,
                "diagnosis": frame["diagnosis"]["label"],
                "confidence": frame["diagnosis"]["confidence"],
                "action": frame["recommendation"]["action"],
                "spm": frame["recommendation"]["spm"],
                "bottom_hole_temperature_c": frame["estimate"][
                    "bottom_hole_temperature_c"
                ],
                "cumulative_sor": frame["css"]["cumulative_sor"],
                "pump_fillage": frame["css"]["pump_fillage"],
            }
        )

    counts: Dict[str, int] = {}
    for entry in history:
        key = str(entry["diagnosis"])
        counts[key] = counts.get(key, 0) + 1
    return {
        "scans": len(batch.samples),
        "first": first_frame,
        "last": last_frame,
        "diagnosis_counts": counts,
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "alarms": alarms,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------


@router.get("/schedule")
async def schedule() -> Dict[str, object]:
    """Optimal CSS programme from the schedule optimiser."""
    optimiser = get_optimizer()
    async with _OPTIMIZER_LOCK:
        result = await asyncio.to_thread(optimiser.optimise)
    return result.as_dict()


@router.post("/schedule/evaluate")
async def evaluate_schedule(schedule_model: Dict[str, float]) -> Dict[str, object]:
    """Evaluate a caller-supplied CSS programme."""
    try:
        schedule = CSSSchedule(
            injection_days=float(schedule_model["injection_days"]),
            soak_days=float(schedule_model["soak_days"]),
            production_days=float(schedule_model["production_days"]),
            cycles=int(schedule_model["cycles"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail="injection_days, soak_days, production_days and cycles are required",
        ) from exc
    optimiser = get_optimizer()
    return (await asyncio.to_thread(optimiser.evaluate, schedule)).as_dict()


@router.post("/control/apply")
async def apply_setpoint(request_model: SetpointRequest) -> Dict[str, object]:
    """Apply an approved VFD setpoint."""
    twin = get_twin()
    if request_model.autonomous:
        twin.set_autonomous(True)
    return twin.apply_setpoint(request_model.spm, request_model.stroke_length_m)


@router.post("/control/autonomy")
async def set_autonomy(request_model: AutonomyRequest) -> Dict[str, object]:
    """Enable or disable the autonomous VFD closed loop."""
    return get_twin().set_autonomous(request_model.enabled)


@router.get("/control/state")
async def control_state() -> Dict[str, object]:
    """Current VFD setpoints, alarms and twin state."""
    twin = get_twin()
    return {
        "spm": twin.applied_spm,
        "stroke_length_m": twin.applied_stroke_m,
        "autonomous": twin.autonomous,
        "alarms": twin.alarms,
        "phase": twin.reservoir.phase.value,
        "cycle": twin.cycle,
        "bottom_hole_temperature_c": twin.reservoir.bottom_hole_temperature_c,
        "cumulative_sor": twin.reservoir.cumulative_sor,
        "pump_fillage": twin.reservoir.pump_fillage,
    }
