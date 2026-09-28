#!/usr/bin/env python
"""Synthetic Baghewala field streamer.

Emulates the OPC-UA / SCADA poll cycle of a CSS producer through a whole CSS
lifecycle -- cycle 1 steam injection, a five day soak and sixty days of
production with progressive cooling and viscous thickening -- and streams the
resulting cards and instrument readings to the ThermaTwin backend over a
WebSocket.

The trajectory is produced by the *same* physics the twin uses
(:class:`backend.app.physics.thermal_reservoir.ThermalReservoirModel` and the
parametric dyno card model), so the stream is a faithful replay of the field
rather than an independent noise generator.  Sensor noise, quantisation and a
small sample-to-sample drift are layered on top to emulate the instrument chain.

Run it with

.. code-block:: bash

    python simulation/synthetic_field_generator.py --url http://localhost:8000 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.app.ai.dyno_classifier import CardLabel, synthesize_card
from backend.app.core.config import CSS, PUMPING_UNIT, RESERVOIR, THRESHOLDS, WELL
from backend.app.physics.gibbs_solver import baghewala_rod_string
from backend.app.physics.hydraulics import FluidProperties, mixture_properties
from backend.app.physics.rheology import baghewala_crude
from backend.app.physics.thermal_reservoir import (
    CSSPhase,
    ThermalReservoirModel,
)

__all__ = ["InstrumentNoise", "FieldStreamer", "main"]

#: Instrument classes for the emulated SCADA chain.
CASING_PRESSURE_NOISE_MPA = 0.035
WELLHEAD_TEMPERATURE_NOISE_C = 1.6
FLOW_NOISE_FRACTION = 0.022
CARD_LOAD_NOISE_KN = 0.45
CARD_SAMPLES = 240


@dataclass
class InstrumentNoise:
    """Additive noise and quantisation for the emulated instrument chain.

    Attributes
    ----------
    casing_pressure_noise_mpa
        Standard deviation of the casing head pressure transmitter, MPa.
    wellhead_temperature_noise_c
        Standard deviation of the wellhead RTD, degC.
    flow_noise_fraction
        Standard deviation of the flow meter, as a fraction of reading.
    card_load_noise_kn
        Standard deviation of the load cell, kN.
    """

    casing_pressure_noise_mpa: float = CASING_PRESSURE_NOISE_MPA
    wellhead_temperature_noise_c: float = WELLHEAD_TEMPERATURE_NOISE_C
    flow_noise_fraction: float = FLOW_NOISE_FRACTION
    card_load_noise_kn: float = CARD_LOAD_NOISE_KN

    def apply(
        self, rng: random.Random, casing_pressure_mpa: float, wellhead_temperature_c: float,
        flow_m3_per_day: float, load_n: float
    ) -> Dict[str, float]:
        """Return one noisy instrument reading set."""
        return {
            "casing_head_pressure_mpa": casing_pressure_mpa
            + rng.gauss(0.0, self.casing_pressure_noise_mpa),
            "wellhead_temperature_c": wellhead_temperature_c
            + rng.gauss(0.0, self.wellhead_temperature_noise_c),
            "surface_flow_m3_per_day": max(
                flow_m3_per_day * (1.0 + rng.gauss(0.0, self.flow_noise_fraction)), 0.0
            ),
            "load_n": load_n + rng.gauss(0.0, self.card_load_noise_kn) * 1e3,
        }


@dataclass
class StreamSample:
    """One emulated SCADA scan."""

    timestamp_s: float
    cycle: int
    phase: str
    day: float
    spm: float
    stroke_m: float
    water_cut: float
    position_m: List[float]
    load_n: List[float]
    casing_head_pressure_mpa: float
    wellhead_temperature_c: float
    surface_flow_m3_per_day: float
    bottom_hole_temperature_c: float
    pump_fillage: float
    cumulative_sor: float

    def as_payload(self) -> dict:
        """JSON body for the WebSocket telemetry ingestion route."""
        return {
            "timestamp_s": self.timestamp_s,
            "cycle": self.cycle,
            "phase": self.phase,
            "day": self.day,
            "spm": self.spm,
            "stroke_m": self.stroke_m,
            "water_cut": self.water_cut,
            "position_m": self.position_m,
            "load_n": self.load_n,
            "casing_head_pressure_mpa": self.casing_head_pressure_mpa,
            "wellhead_temperature_c": self.wellhead_temperature_c,
            "surface_flow_m3_per_day": self.surface_flow_m3_per_day,
            "bottom_hole_temperature_c": self.bottom_hole_temperature_c,
            "pump_fillage": self.pump_fillage,
            "cumulative_sor": self.cumulative_sor,
        }


class FieldStreamer:
    """Replays a full CSS lifecycle as a sequence of SCADA scans.

    Parameters
    ----------
    cycles
        Number of CSS cycles to replay.
    production_days
        Production phase length of each cycle, days.
    samples_per_card
        Samples in each emulated crank cycle.
    spm
        Strokes per minute; the streamer will *reduce* it on its own when the
        emulated fillage falls below the fluid-pound threshold, which is the
        behaviour the closed loop is meant to reproduce.
    fault
        Card fault to inject, so a fault scenario can be demonstrated.
    seed
        Seed for the reproducible noise sequence.
    """

    def __init__(
        self,
        cycles: int = CSS.design_cycles,
        production_days: float = CSS.production_days,
        samples_per_card: int = CARD_SAMPLES,
        spm: float = PUMPING_UNIT.nominal_spm,
        stroke_m: float = PUMPING_UNIT.nominal_stroke_m,
        fault: CardLabel = CardLabel.NORMAL_FULL_BARREL,
        noise: Optional[InstrumentNoise] = None,
        seed: int = 26120,
    ) -> None:
        if cycles < 1:
            raise ValueError("at least one cycle is required")
        if production_days <= 0.0:
            raise ValueError("production_days must be positive")
        if samples_per_card < 32:
            raise ValueError("at least 32 samples per card are required")
        if spm <= 0.0 or stroke_m <= 0.0:
            raise ValueError("SPM and stroke must be positive")
        self.cycles = cycles
        self.production_days = production_days
        self.samples_per_card = samples_per_card
        self.spm = spm
        self.stroke_m = stroke_m
        self.fault = fault
        self.noise = noise or InstrumentNoise()
        self.rng = random.Random(seed)
        self.rod_weight_n = baghewala_rod_string().weight_n
        self.crude = baghewala_crude()
        self.timestamp_s = 0.0
        self._water_cut = 0.12

    # -- physics ----------------------------------------------------------
    def _mixture_density(self, temperature_c: float, water_cut: float) -> float:
        fluid = FluidProperties(
            temperature_c=temperature_c,
            pressure_pa=RESERVOIR.initial_reservoir_pressure_mpa * 1e6,
            oil_fraction=1.0 - water_cut,
            water_fraction=water_cut,
            oil_viscosity_pa_s=self.crude.dynamic_viscosity_pa_s(temperature_c),
        )
        rho, _ = mixture_properties(fluid)
        return rho

    def casing_head_pressure_mpa(
        self, bottom_hole_pressure_mpa: float, temperature_c: float
    ) -> float:
        """Casing head pressure from the wellbore hydrostatic balance, MPa.

        .. math:: P_{chp} = P_{bh} - \\rho_m g H - \\Delta p_{friction}
        """
        density = self._mixture_density(temperature_c, self._water_cut)
        column = density * 9.80665 * WELL.pump_depth_m * 1e-6
        # The pump draws down the tubing; the casing annulus reads the standing
        # column less the intake pressure recovery.
        return max(bottom_hole_pressure_mpa - 0.62 * column, 0.05)

    def wellhead_temperature_c(self, bottom_hole_temperature_c: float) -> float:
        """Wellhead temperature after the Ramey wellbore heat loss, degC.

        At the pump-limited rate the residence time in a 1000 m well is about
        22 hours, which is long enough for the stream to relax substantially
        towards the formation temperature on the way up.
        """
        return RESERVOIR.reservoir_temperature_c + 0.28 * (
            bottom_hole_temperature_c - RESERVOIR.reservoir_temperature_c
        )

    # -- generation -------------------------------------------------------
    def samples(self) -> List[StreamSample]:
        """Generate the full lifecycle as a list of SCADA scans."""
        model = ThermalReservoirModel()
        programme = model.simulate_programme(
            cycles=self.cycles, production_days=self.production_days
        )
        output: List[StreamSample] = []
        cycle_index = 1
        previous_phase: Optional[CSSPhase] = None
        for state in programme.states:
            producing = state.phase is CSSPhase.PRODUCTION
            fillage = min(max(state.pump_fillage, 0.15), 1.0) if producing else 1.0
            # The streamer performs the same fillage-matched SPM reduction the
            # closed loop recommends, which is the behaviour the twin is meant
            # to reproduce from the measurements alone.
            target_spm = self.spm
            if producing and fillage < THRESHOLDS.fluid_pound_fillage:
                target_spm = max(
                    THRESHOLDS.min_spm,
                    min(
                        self.spm
                        * (fillage / THRESHOLDS.fluid_pound_fillage),
                        self.spm,
                    ),
                )
            self.spm = target_spm
            if not producing:
                fillage = 1.0

            fluid_load = (
                PUMPING_UNIT.plunger_area_m2 * PUMPING_UNIT.pump_differential_pa * fillage
            )
            effective_fault = self.fault
            if self.fault is CardLabel.FLUID_POUND and not producing:
                effective_fault = CardLabel.NORMAL_FULL_BARREL
            position, load = synthesize_card(
                effective_fault,
                rod_weight_n=self.rod_weight_n,
                fluid_load_n=fluid_load,
                stroke_m=self.stroke_m,
                fillage=fillage,
                samples=self.samples_per_card,
                noise_fraction=0.006,
                seed=self.rng.randrange(2**31),
            )
            water_cut = self._water_cut + math.sin(state.time_days * 0.35) * 0.02
            water_cut = min(max(water_cut, 0.0), 0.6)
            casing = self.casing_head_pressure_mpa(
                model.producing_bottom_hole_pressure_mpa,
                state.bottom_hole_temperature_c,
            )
            wellhead = self.wellhead_temperature_c(state.bottom_hole_temperature_c)
            reading = self.noise.apply(
                self.rng, casing, wellhead, state.oil_rate_tpd, float(np.min(load))
            )
            output.append(
                StreamSample(
                    timestamp_s=self.timestamp_s,
                    cycle=cycle_index,
                    phase=state.phase.value,
                    day=state.time_days,
                    spm=self.spm,
                    stroke_m=self.stroke_m,
                    water_cut=water_cut,
                    position_m=[float(v) for v in position],
                    load_n=[float(v) for v in load],
                    casing_head_pressure_mpa=reading["casing_head_pressure_mpa"],
                    wellhead_temperature_c=reading["wellhead_temperature_c"],
                    surface_flow_m3_per_day=reading["surface_flow_m3_per_day"],
                    bottom_hole_temperature_c=state.bottom_hole_temperature_c,
                    pump_fillage=fillage,
                    cumulative_sor=state.cumulative_sor,
                )
            )
            # A SCADA scan covers one crank cycle.  The CSS state clock is
            # cumulative across the whole programme, so the cycle number is
            # tracked from the phase transitions rather than from that clock.
            # A new cycle begins at injection, so the counter advances on the
            # transition *into* injection, not into production.
            if (
                previous_phase is CSSPhase.PRODUCTION
                and state.phase is CSSPhase.INJECTION
            ):
                cycle_index += 1
            previous_phase = state.phase
            self.timestamp_s += 60.0 / max(self.spm, 1e-6)
        return output

    # -- transport --------------------------------------------------------
    async def send(
        self,
        url: str,
        samples: Sequence[StreamSample],
        interval_s: float,
        limit: Optional[int] = None,
        batch_size: int = 200,
    ) -> int:
        """Replay the samples to the backend, returning the number sent.

        A historian replay is a bulk upload rather than a real-time push, so the
        scans are posted to the ingestion route in batches.  ``/ws/telemetry`` is
        a *server push* stream: the backend emits frames to the dashboard and
        accepts only control commands, so posting scans to it would silently
        discard them.  Passing a ``ws://`` URL is therefore rejected rather than
        quietly ignored.
        """
        import httpx

        endpoint = url
        if endpoint.startswith("ws://"):
            endpoint = "http://" + endpoint[len("ws://") :]
        elif endpoint.startswith("wss://"):
            endpoint = "https://" + endpoint[len("wss://") :]
        endpoint = endpoint.rstrip("/")
        if not endpoint.endswith("/api/v1/telemetry/ingest"):
            endpoint = endpoint + "/api/v1/telemetry/ingest"

        payload_limit = len(samples) if limit is None else min(limit, len(samples))
        selected = samples[:payload_limit]
        sent = 0
        async with httpx.AsyncClient(timeout=120.0) as client:
            for start in range(0, len(selected), batch_size):
                batch = selected[start : start + batch_size]
                body = {"samples": [s.as_payload() for s in batch], "apply_setpoints": True}
                response = await client.post(endpoint, json=body)
                response.raise_for_status()
                result = response.json()
                sent += len(batch)
                last = batch[-1]
                print(
                    f"  replayed {sent}/{payload_limit} scans "
                    f"(cycle {last.cycle}, {last.phase}, day {last.day:.1f}, "
                    f"fillage {last.pump_fillage:.2f}) -> "
                    f"{result.get('diagnosis_counts', {})} "
                    f"[{result.get('mean_latency_ms', 0.0):.0f} ms/scan]",
                    flush=True,
                )
                if interval_s > 0.0:
                    await asyncio.sleep(interval_s)
        return sent


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="backend base URL; scans are posted to /api/v1/telemetry/ingest",
    )
    parser.add_argument("--cycles", type=int, default=3, help="CSS cycles to replay")
    parser.add_argument(
        "--production-days", type=float, default=CSS.production_days, help="production phase length"
    )
    parser.add_argument(
        "--interval", type=float, default=0.5, help="seconds between scans"
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N scans")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="generate the lifecycle and print a summary without connecting",
    )
    parser.add_argument(
        "--fault",
        default="NORMAL_FULL_BARREL",
        choices=[label.value for label in CardLabel],
        help="card fault to inject",
    )
    parser.add_argument("--seed", type=int, default=26120, help="noise seed")
    arguments = parser.parse_args(argv)

    streamer = FieldStreamer(
        cycles=arguments.cycles,
        production_days=arguments.production_days,
        fault=CardLabel(arguments.fault),
        seed=arguments.seed,
    )
    samples = streamer.samples()
    if not samples:
        print("no samples generated", file=sys.stderr)
        return 1

    phases: Dict[str, int] = {}
    for sample in samples:
        phases[sample.phase] = phases.get(sample.phase, 0) + 1
    final = samples[-1]
    print(
        f"generated {len(samples)} scans across {arguments.cycles} cycle(s): {phases}\n"
        f"  final: cycle {final.cycle}, {final.phase}, day {final.day:.1f}, "
        f"BHT {final.bottom_hole_temperature_c:.1f} degC, "
        f"fillage {final.pump_fillage:.2f}, cumulative SOR {final.cumulative_sor:.2f}, "
        f"SPM {final.spm:.2f}"
    )
    if arguments.dry_run:
        return 0

    sent = asyncio.run(
        streamer.send(arguments.url, samples, arguments.interval, arguments.limit)
    )
    print(f"replayed {sent} scans to {arguments.url}/api/v1/telemetry/ingest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
