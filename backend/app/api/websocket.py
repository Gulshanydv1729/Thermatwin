"""Real-time WebSocket streaming of the digital twin.

The stream emulates the OPC-UA/SNMP poll cycle of a SCADA system.  A synthetic
card generator produces a crank cycle of surface load and position at the
current SPM, the twin inverts it to the downhole pump card, and the resulting
frame -- card, diagnosis, wellbore traverse, state estimate and VFD
recommendation -- is pushed to the client every ``stream_interval_s``.

The synthesiser is deterministic given a seed, so an integration test can
assert on the contents of a frame.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.ai.dyno_classifier import CardLabel
from backend.app.api.routes import get_twin
from backend.app.core.config import PUMPING_UNIT, THRESHOLDS
from backend.app.engine.digital_twin import DigitalTwin
from backend.app.engine.state_estimator import TelemetryPacket
from backend.app.physics.gibbs_solver import baghewala_rod_string

__all__ = ["router", "StreamEngine"]

router = APIRouter(tags=["thermatwin"])


class StreamEngine:
    """Drives a twin with a synthetic SCADA cycle and yields telemetry frames.

    The engine owns the crank kinematics and the card synthesiser so that the
    WebSocket handler only has to pump it, and so that the same generator can be
    driven by a test without a socket.

    Parameters
    ----------
    twin
        Twin to drive.  The process-wide twin is used when omitted.
    fault
        Card fault the synthesiser injects, which is how a fault scenario is
        demonstrated in the dashboard.
    noise
        Additive load noise as a fraction of the card span.
    seed
        Seed for the reproducible noise sequence.
    advance
        Whether each frame advances the CSS reservoir clock, so the stream plays
        out the production decline instead of holding one operating point.
    day_step
        Days of production elapsed per frame; defaults to the twin's setting.
    """

    def __init__(
        self,
        twin: Optional[DigitalTwin] = None,
        fault: CardLabel = CardLabel.NORMAL_FULL_BARREL,
        noise: float = 0.004,
        seed: int = 26120,
        samples: int = 240,
        advance: bool = True,
        day_step: Optional[float] = None,
    ) -> None:
        if samples < 32:
            raise ValueError("at least 32 samples per crank cycle are required")
        self.twin = twin or get_twin()
        self.fault = fault
        self.noise = noise
        self.seed = seed
        self.samples = samples
        self.advance = advance
        self.day_step = day_step
        self.rod_weight_n = self.twin.rod.weight_n
        self.timestamp_s = 0.0

    # -- card synthesis ---------------------------------------------------
    def crank_cycle_s(self, spm: float) -> float:
        """Duration of one crank cycle at ``spm`` seconds, s."""
        if spm <= 0.0:
            raise ValueError("SPM must be positive")
        return 60.0 / spm

    def synthesise_surface_card(
        self,
        spm: float,
        stroke_m: float,
        fluid_load_n: float,
        fillage: float = 1.0,
    ) -> Dict[str, List[float]]:
        """Generate one crank cycle of surface position and load.

        The card is built from the twin's own rod weight, stroke and SPM, and
        the requested fault deformation is applied, so the stream shows the same
        geometry the solver is being asked to invert.
        """
        from backend.app.ai.dyno_classifier import synthesize_card

        # The card the load cell would record follows from the barrel fillage,
        # not from an independent choice: a barrel that cannot fill produces a
        # fluid-pound trace.  That keeps the card, the diagnosis and the VFD
        # recommendation causally consistent, all three being consequences of
        # the same reservoir state.
        #
        # An *explicitly injected* fault is different.  Requesting
        # ROD_FLOATING exists so a specific fault can be demonstrated, and a
        # coincidentally low fillage must not silently replace it.  So the
        # fillage only imposes fluid pound in the nominal case, where the
        # caller has asked for no particular fault.
        effective_fillage = min(max(float(fillage), 0.15), 1.0)
        fault = self.fault
        if fault is CardLabel.NORMAL_FULL_BARREL and (
            effective_fillage < THRESHOLDS.fluid_pound_fillage
        ):
            fault = CardLabel.FLUID_POUND
        if fault is CardLabel.FLUID_POUND:
            effective_fillage = min(effective_fillage, 0.55)
        position, load = synthesize_card(
            fault,
            rod_weight_n=self.rod_weight_n,
            fluid_load_n=fluid_load_n,
            stroke_m=stroke_m,
            fillage=effective_fillage,
            samples=self.samples,
            noise_fraction=self.noise,
            seed=self.seed,
        )
        return {
            "position_m": [float(v) for v in position],
            "load_n": [float(v) for v in load],
        }

    def build_packet(
        self,
        spm: float,
        stroke_m: float,
        card: Dict[str, List[float]],
        traverse_pressure_pa: float,
        traverse_temperature_c: float,
    ) -> TelemetryPacket:
        """Assemble the telemetry packet for one stream tick."""
        loads = np.asarray(card["load_n"], dtype=float)
        delivered = self.twin.reservoir.oil_rate_tpd
        mass_flow = delivered * 900.0 / 86400.0
        return TelemetryPacket(
            timestamp_s=self.timestamp_s,
            casing_head_pressure_pa=float(traverse_pressure_pa),
            wellhead_temperature_c=float(traverse_temperature_c),
            surface_flow_m3_per_day=delivered,
            card_min_load_n=float(np.min(loads)),
            card_max_load_n=float(np.max(loads)),
            spm=float(spm),
            stroke_m=float(stroke_m),
        )

    # -- frame ------------------------------------------------------------
    def tick(self) -> Dict[str, object]:
        """Produce one telemetry frame and advance the stream clock."""
        twin = self.twin
        spm = twin.applied_spm
        stroke_m = twin.applied_stroke_m
        duration = self.crank_cycle_s(spm)

        traverse = twin.traverse(segments=30)
        # Drive the synthesised card from the *estimated* fillage so that the
        # card, the diagnosis and the recommendation are causally linked.
        fillage = twin.current_fillage()
        card_surface = self.synthesise_surface_card(
            spm=spm,
            stroke_m=stroke_m,
            fluid_load_n=PUMPING_UNIT.plunger_area_m2
            * PUMPING_UNIT.pump_differential_pa
            * fillage,
            fillage=fillage,
        )
        packet = self.build_packet(
            spm=spm,
            stroke_m=stroke_m,
            card=card_surface,
            traverse_pressure_pa=float(traverse.pressure_pa[-1]),
            traverse_temperature_c=float(traverse.temperature_c[-1]),
        )
        snapshot = twin.step(
            packet,
            surface_position_m=card_surface["position_m"],
            surface_load_n=card_surface["load_n"],
            duration_s=duration,
        )
        frame = snapshot.as_frame(cycle=twin.cycle, well_name="BAG-17")
        frame["well"]["day"] = twin.reservoir_day
        frame["well"]["days_into_phase"] = twin.reservoir_day
        frame["wellbore"] = traverse.as_dict()
        frame["alarms"] = twin.alarms
        self.timestamp_s += duration
        # Advance the reservoir clock so the dashboard watches the well decline
        # through the production phase rather than sitting at one point.
        if self.advance:
            twin.advance_reservoir()
        return frame


@router.websocket("/ws/telemetry")
async def telemetry_stream(websocket: WebSocket) -> None:
    """Stream a live telemetry frame every ``stream_interval_s`` seconds.

    The client may send ``{"type":"setpoint", ...}`` to apply a VFD setpoint and
    ``{"type":"autonomy", "enabled": true}`` to toggle the autonomous loop; both
    are answered with an ``ack`` frame.
    """
    from backend.app.core.config import get_config

    await websocket.accept()
    config = get_config()
    twin = get_twin()
    engine = StreamEngine(twin=twin)
    await websocket.send_json(
        {
            "type": "hello",
            "well": "BAG-17",
            "interval_s": config.api.stream_interval_s,
            "specification": "SIH26120",
        }
    )

    receiver = asyncio.create_task(_receive_commands(websocket, twin))
    try:
        while True:
            frame = await asyncio.to_thread(engine.tick)
            await websocket.send_json(frame)
            await asyncio.sleep(config.api.stream_interval_s)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receiver


async def _receive_commands(websocket: WebSocket, twin: DigitalTwin) -> None:
    """Handle inbound control frames without blocking the broadcast loop."""
    while True:
        try:
            message = await websocket.receive_json()
        except WebSocketDisconnect:
            # A normal client disconnect is not an error; the broadcast loop
            # owns the teardown.
            return
        except Exception:
            return
        if not isinstance(message, dict):
            continue
        kind = message.get("type")
        if kind == "setpoint":
            response = twin.apply_setpoint(
                float(message.get("spm", twin.applied_spm)),
                float(message.get("stroke_length_m", twin.applied_stroke_m)),
            )
        elif kind == "autonomy":
            response = twin.set_autonomous(bool(message.get("enabled", False)))
        elif kind == "ping":
            response = {"accepted": True, "message": "pong"}
        else:
            response = {"accepted": False, "message": f"unknown command {kind!r}"}
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "ack", **response})
