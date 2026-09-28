"""Closed-loop well-to-surface digital twin for a CSS producer.

The twin couples the four layers of the model in the direction the physics
actually runs:

.. code-block:: text

    reservoir (Marx-Langenheim CSS)
        -> reservoir inflow, steam chest radius, bottom hole temperature
    wellbore (Beggs & Brill holdup + Ramey heat transfer)
        -> intake pressure, wellhead temperature, delivered rate
    rod string (Gibbs' damped wave equation)
        -> downhole pump card from the measured surface card
    AI (dyno card CNN + thermal surrogate)
        -> fault label, fillage, 14-day forecast

    state estimator (extended Kalman filter)
        -> pump intake pressure, skin, thermal radius
    controller (fillage and rod-float hydraulics)
        -> VFD setpoint

The loop closes because the recommended SPM feeds back into the hydraulics:
slowing the pump to match the inflow restores the barrel fillage, which is
exactly the fluid-pound remediation the specification asks for.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.app.ai.dyno_classifier import (
    CardLabel,
    CardPrediction,
    DynoCardClassifier,
    analytic_diagnosis,
    card_features,
)
from backend.app.ai.sor_optimizer import (
    SRPController,
    SetpointRecommendation,
)
from backend.app.ai.thermal_surrogate import ThermalSurrogate
from backend.app.core.config import (
    CSS,
    PUMPING_UNIT,
    RESERVOIR,
    THRESHOLDS,
    WELL,
    AppConfig,
    get_config,
)
from backend.app.engine.state_estimator import (
    ExtendedKalmanFilter,
    StateEstimate,
    TelemetryPacket,
)
from backend.app.physics.gibbs_solver import (
    GibbsSolver,
    PumpCard,
    RodString,
    baghewala_rod_string,
    damping_coefficient,
    default_column_weight_n,
)
from backend.app.physics.hydraulics import WellboreTraverse, compute_traverse
from backend.app.physics.rheology import baghewala_crude
from backend.app.physics.thermal_reservoir import (
    CSSPhase,
    SteamChestState,
    ThermalReservoirModel,
)

__all__ = ["TwinSnapshot", "DigitalTwin"]


def _downsample_card(card: PumpCard, output_samples: int) -> PumpCard:
    """Reduce a pump card to ``output_samples`` points for transport.

    The Gibbs solver needs a fine internal grid to resolve the wave transit, but
    a crank cycle at field SPM contains only a few hundred distinct load levels,
    so shipping the full solver grid would bloat every WebSocket frame without
    adding information.  Downstream analysis uses the coarse sampling.
    """
    if len(card.time_s) <= output_samples:
        return card
    indices = np.linspace(0, len(card.time_s) - 1, output_samples).astype(int)
    return PumpCard(
        time_s=card.time_s[indices],
        position_m=card.position_m[indices],
        load_n=card.load_n[indices],
        surface_position_m=card.surface_position_m,
        surface_load_n=card.surface_load_n,
        damping_rate_s=card.damping_rate_s,
        segments=card.segments,
        arrival_time_s=card.arrival_time_s,
        transmission_gain=card.transmission_gain,
        metadata={**card.metadata, "output_samples": output_samples},
    )


@dataclass
class TwinSnapshot:
    """One complete pass of the closed loop."""

    timestamp_s: float
    card: PumpCard
    prediction: CardPrediction
    traverse: WellboreTraverse
    estimate: StateEstimate
    recommendation: SetpointRecommendation
    reservoir: SteamChestState
    latency_ms: float

    def as_frame(self, cycle: int, well_name: str) -> dict:
        """JSON frame for the WebSocket stream, matching the frontend contract."""
        return {
            "type": "telemetry",
            "timestamp_s": self.timestamp_s,
            "well": {
                "name": well_name,
                "cycle": cycle,
                "phase": self.reservoir.phase.value,
                "day": self.reservoir.time_days,
                "days_into_phase": self.reservoir.time_days,
            },
            "card": {
                "surface": {
                    "position_m": [float(v) for v in self.card.surface_position_m],
                    "load_n": [float(v) for v in self.card.surface_load_n],
                },
                "downhole": {
                    "position_m": [float(v) for v in self.card.position_m],
                    "load_n": [float(v) for v in self.card.load_n],
                },
                "stroke_m": self.card.stroke_m,
                "min_load_n": self.card.min_load_n,
                "max_load_n": self.card.max_load_n,
                "load_span_n": self.card.load_span_n,
            },
            "diagnosis": self.prediction.as_dict(),
            "wellbore": self.traverse.as_dict(),
            "estimate": self.estimate.as_dict(),
            "css": {
                "cumulative_sor": self.reservoir.cumulative_sor,
                "instantaneous_sor": self.reservoir.instantaneous_sor,
                "steam_tonnes": self.reservoir.cumulative_steam_tonnes,
                "oil_bbl": self.reservoir.cumulative_oil_volume_m3 * 6.2898,
                "cumulative_steam_m3": self.reservoir.cumulative_steam_volume_m3,
                "cumulative_oil_m3": self.reservoir.cumulative_oil_volume_m3,
                "cutoff_reached": self.reservoir.cutoff_reached,
                "cutoff_reason": self.reservoir.cutoff_reason,
                "chest_radius_m": self.reservoir.radius_m,
                "pump_fillage": self.reservoir.pump_fillage,
            },
            "recommendation": self.recommendation.as_dict(),
            "latency_ms": self.latency_ms,
        }


class DigitalTwin:
    """Coupled reservoir / wellbore / rod-string / AI twin with a VFD actuator.

    Parameters
    ----------
    config
        Field configuration; the process-wide default is used when omitted.
    rod
        Rod string definition.
    segments
        Spatial resolution of the Gibbs solver.
    classifier
        Trained dyno card classifier.  A lazily trained default is created if
        none is supplied.
    surrogate
        Trained thermal surrogate; optional, used for the forward forecast.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        rod: Optional[RodString] = None,
        segments: int = PUMPING_UNIT.rod_segments,
        classifier: Optional[DynoCardClassifier] = None,
        surrogate: Optional[ThermalSurrogate] = None,
        state_filter: Optional[ExtendedKalmanFilter] = None,
        output_samples: int = 360,
        start_day: float = 6.0,
    ) -> None:
        if output_samples < 16:
            raise ValueError("output_samples must be at least 16")
        self.start_day = float(start_day)
        self.config = config or get_config()
        self.rod = rod or baghewala_rod_string()
        self.segments = segments
        self.output_samples = output_samples
        self.classifier = classifier
        self.surrogate = surrogate
        self.state_filter = state_filter or ExtendedKalmanFilter()
        self.controller = SRPController()
        self.css_model = ThermalReservoirModel()

        self.cycle = 1
        self.applied_spm = PUMPING_UNIT.nominal_spm
        self.applied_stroke_m = PUMPING_UNIT.nominal_stroke_m
        self.autonomous = False
        self._last_time_s = 0.0
        self._last_estimate: Optional[StateEstimate] = None
        self._trajectory_cache: Optional[List[SteamChestState]] = None
        self._alarms: List[str] = []
        self._load_thermal_state()

    # -- lazy AI ----------------------------------------------------------
    def _ensure_classifier(self) -> DynoCardClassifier:
        """Return a trained classifier, loading the checkpoint or training one.

        The instance is only published to ``self.classifier`` once it is known to
        be trained.  Assigning it first would leave a half-initialised twin
        behind if the training were interrupted -- which the on-boot warm-up in
        a ``to_thread`` call can be, when a client disconnects mid-first-frame --
        and every later ``predict`` would then raise instead of recovering.
        """
        if self.classifier is not None and self.classifier.trained:
            return self.classifier
        weights = self.config.weights_dir / "dyno_classifier.pt"
        classifier = DynoCardClassifier(train=False)
        if weights.exists():
            # The cached weights are the same model the training path produces,
            # so prefer them: a container that cannot write its weights volume
            # must not retrain on every boot.
            classifier.load(weights)
        else:
            classifier.train_and_save(weights, epochs=10, samples_per_label=250)
        self.classifier = classifier
        return self.classifier

    #: Elapsed CSS days represented by one stream frame.  At the 500 ms cadence
    #: the default puts one field day on the dashboard every 10 s, so a full
    #: 60-day production decline is visible in about ten minutes.
    day_step: float = 0.05
    #: Total days the stream is allowed to run before it restarts the cycle.
    max_day: float = CSS.production_days

    def _production_trajectory(self) -> List[SteamChestState]:
        """The production-phase trajectory, simulated once and cached.

        The whole cycle is a deterministic function of the configuration, so it
        is solved a single time and then *interpolated* on demand.  Re-solving
        the Marx-Langenheim model on every stream frame would put a full
        reservoir simulation inside the 100 ms telemetry budget for no reason,
        and it is the single largest cost in the frame.
        """
        if self._trajectory_cache is None:
            programme = self.css_model.simulate_cycle(
                injection_days=CSS.injection_days,
                soak_days=CSS.soak_days,
                production_days=self.max_day,
            )
            producing = [
                state
                for state in programme.states
                if state.phase.value == "PRODUCTION"
            ]
            if not producing:
                producing = list(programme.states)
            self._trajectory_cache = producing
        return self._trajectory_cache

    def reservoir_state_at(self, day: float) -> SteamChestState:
        """The CSS state at ``day`` of the production phase.

        The value is interpolated between the two cached trajectory samples
        that bracket it, so the state is a continuous function of the day rather
        than a staircase, and the ``time_days`` field agrees with the day
        requested.  The same rule is used when the stream starts and when it
        restarts a cycle, which is what stops the first frame of a cycle and the
        first frame of the next from disagreeing.
        """
        trajectory = self._production_trajectory()
        # ``day`` counts production days; the trajectory's clock is cumulative
        # from the start of the cycle, so the offset is the injection and soak
        # time that precede production.
        offset = CSS.injection_days + CSS.soak_days
        target = offset + max(float(day), 0.0)
        times = [state.time_days for state in trajectory]
        if target <= times[0]:
            return trajectory[0]
        if target >= times[-1]:
            return trajectory[-1]
        upper = int(np.searchsorted(times, target))
        lower = upper - 1
        span = times[upper] - times[lower]
        weight = 0.0 if span <= 0.0 else (target - times[lower]) / span
        start, end = trajectory[lower], trajectory[upper]
        interpolated = replace(
            start,
            time_days=target,
            phase=CSSPhase.PRODUCTION,
            radius_m=start.radius_m + weight * (end.radius_m - start.radius_m),
            chest_temperature_c=(
                start.chest_temperature_c
                + weight * (end.chest_temperature_c - start.chest_temperature_c)
            ),
            bottom_hole_temperature_c=(
                start.bottom_hole_temperature_c
                + weight
                * (end.bottom_hole_temperature_c - start.bottom_hole_temperature_c)
            ),
            oil_rate_tpd=(
                start.oil_rate_tpd + weight * (end.oil_rate_tpd - start.oil_rate_tpd)
            ),
            pump_fillage=(
                start.pump_fillage + weight * (end.pump_fillage - start.pump_fillage)
            ),
            instantaneous_sor=(
                start.instantaneous_sor
                + weight * (end.instantaneous_sor - start.instantaneous_sor)
            ),
            cumulative_sor=(
                start.cumulative_sor + weight * (end.cumulative_sor - start.cumulative_sor)
            ),
        )
        return interpolated

    def advance_reservoir(self, day_step: Optional[float] = None) -> None:
        """Advance the CSS reservoir state by ``day_step`` days.

        The dashboard is far more informative if the well is *seen* to decline
        rather than being frozen at one operating point, so each stream frame
        moves the reservoir clock on and the whole production phase is played
        out.  When the cycle runs out the well is returned to the start of
        production, so the stream is endless without becoming unphysical.
        """
        step = self.day_step if day_step is None else float(day_step)
        if step <= 0.0:
            return
        self.reservoir_day = min(self.reservoir_day + step, self.max_day)
        self.reservoir = self.reservoir_state_at(self.reservoir_day)
        if self.reservoir_day >= self.max_day:
            # Restart the cycle so the stream keeps showing a live decline.
            self.reservoir_day = self.start_day
            self.reservoir = self.reservoir_state_at(self.start_day)
            self.cycle += 1
            self.state_filter.reset()

    def _load_thermal_state(self) -> None:
        """Place the twin at the configured production day."""
        # Open the stream a few days into the production phase, while the barrel
        # is still full, so the dashboard *watches* the decline develop into the
        # fluid-pound regime rather than appearing already starved.  ``start_day``
        # counts production days, not elapsed cycle days: the injection and soak
        # phases precede it.  Past roughly day 30 of production the reservoir can
        # no longer fill the pump at 9 spm, so the stream opens at day 6 and
        # plays through the crossover.
        # Solving the trajectory here rather than lazily keeps the cost in the
        # warm-up, where it belongs: done on demand it lands inside the first
        # telemetry frame and makes that frame take seconds instead of
        # milliseconds.
        self._production_trajectory()
        self.reservoir = self.reservoir_state_at(self.start_day)
        self.reservoir_day = self.start_day
        self.cycle = 1

    def current_fillage(self) -> float:
        """Pump fillage for the current reservoir state, in (0, 1].

        This is the CSS solver's own fillage, because that model is the
        authoritative statement of the thermal state and the demo must show its
        physics.  The Kalman estimate is deliberately *not* used here: it is a
        noisy reconstruction of an unmeasured quantity, and letting it drive the
        synthesised card would make the card, the diagnosis and the
        recommendation incoherent with each other and with the reservoir model.

        The estimate still constrains the *control* decision, through
        :meth:`card_fillage` and the reservoir inflow in :meth:`step`, so the
        VFD is never run faster than the measurements support.
        """
        return min(max(self.reservoir.pump_fillage, 0.15), 1.0)

    # -- rod string --------------------------------------------------------
    def _damping_rate(self, temperature_c: float) -> float:
        """Viscous damping rate from the local temperature, 1/s."""
        viscosity = baghewala_crude().dynamic_viscosity_pa_s(temperature_c)
        return damping_coefficient(temperature_c, viscosity)

    #: Fillage used to shape the reconstructed pump card.  This is an
    #: *independent* measurement, not the state estimate: the load cell already
    #: carries the barrel fillage, so defaulting to a full barrel makes the
    #: reconstruction a pure transmission of the measured card.
    reconstruction_fillage: float = 1.0

    def reconstruct_card(
        self,
        surface_position_m: Sequence[float],
        surface_load_n: Sequence[float],
        duration_s: float,
        temperature_c: Optional[float] = None,
    ) -> PumpCard:
        """Invert the measured surface card to the downhole pump card.

        The damping rate is evaluated from the local tubing temperature and the
        corresponding Walther viscosity, so the reconstruction responds to the
        CSS thermal state rather than using a fixed value.
        """
        if duration_s <= 0.0:
            raise ValueError("crank cycle duration must be positive")
        damping = self._damping_rate(
            self.reservoir.bottom_hole_temperature_c
            if temperature_c is None
            else temperature_c
        )
        if self.rod is None:  # pragma: no cover - defensive
            self.rod = baghewala_rod_string()
        solver = GibbsSolver(
            rod=self.rod, segments=self.segments, damping_rate_s=damping
        )
        card = solver.solve(
            surface_position_m=surface_position_m,
            surface_load_n=surface_load_n,
            duration_s=duration_s,
            fillage=self.reconstruction_fillage,
        )
        return _downsample_card(card, output_samples=self.output_samples)

    # -- hydraulics -------------------------------------------------------
    def traverse(
        self, mass_flow_kg_s: Optional[float] = None, segments: int = 40
    ) -> WellboreTraverse:
        """Pressure and temperature traverse for the current well state."""
        if mass_flow_kg_s is None:
            crude = baghewala_crude()
            density = crude.dynamic_viscosity_pa_s  # noqa: F841  (documentation)
            rho = 900.0
            mass_flow_kg_s = (
                self.reservoir.oil_rate_tpd * rho / 86400.0
            )
        # The formation temperature along the wellbore is the *native* 46 degC:
        # the steam chest only surrounds the reservoir interval, and above it
        # the tubing is in contact with cold rock.  At the pump-limited rate the
        # residence time in a 1000 m well is about 22 hours, which is long
        # enough for the stream to relax onto the formation temperature by the
        # wellhead -- a real and important result, and the reason the twin keys
        # its cycle decisions on the reservoir BHT rather than the wellhead.
        return compute_traverse(
            segments=segments,
            mass_flow_kg_s=max(mass_flow_kg_s, 1e-4),
            bottom_hole_temperature_c=self.reservoir.bottom_hole_temperature_c,
            bottom_hole_pressure_pa=self.css_model.producing_bottom_hole_pressure_mpa * 1e6,
        )

    # -- rod float risk ----------------------------------------------------
    def card_fillage(self, card: PumpCard) -> float:
        """Pump fillage implied by the load span of the reconstructed card, (0, 1].

        The load cell measures the weight of the fluid column the plunger
        supports, so the peak-to-peak span of the card is the column weight and
        the ratio of that span to the full standing column is the fraction of
        the barrel that actually filled.  This is a direct measurement, and it
        is what keeps the control decision honest when the flow channel alone
        cannot separate the reservoir state.
        """
        column = default_column_weight_n()
        if column <= 0.0:
            return 1.0
        return min(max(card.load_span_n / column, 0.02), 1.0)

    def rod_float_risk_index(self, card: PumpCard, damping_rate_s: float) -> float:
        """Rod floating risk index in [0, 1].

        A rod string floats when the buoyant viscous drag on the downstroke
        exceeds the weight of the string, so the plunger load passes through
        zero and the rod goes slack.  The index is therefore the fraction of the
        crank cycle over which the load is *negative* -- the load the surface
        load cell would have to exert to keep the rod taut -- combined with the
        damping rate, because the drag that causes the buoyancy grows with the
        fluid viscosity that sets it.
        """
        load = np.asarray(card.load_n, dtype=float)
        if load.size < 4:
            return 0.0
        span = float(np.max(load) - np.min(load))
        if span <= 0.0:
            return 0.0
        # A tolerance of one percent of the span keeps the count from being
        # driven by load-cell noise sitting on zero.
        negative = float(np.count_nonzero(load < -0.01 * span)) / load.size
        drag = min(
            damping_rate_s / max(2.0 * THRESHOLDS.rod_float_risk_high, 1e-9), 1.0
        )
        return float(min(max(negative, 0.0) * 0.5 + drag * 0.5, 1.0))

    # -- main loop ---------------------------------------------------------
    def step(
        self,
        packet: TelemetryPacket,
        surface_position_m: Sequence[float],
        surface_load_n: Sequence[float],
        duration_s: float,
        fillage: Optional[float] = None,
    ) -> TwinSnapshot:
        """Run one full well-to-surface pass for a telemetry packet."""
        started = time.perf_counter()
        if duration_s <= 0.0:
            raise ValueError("crank cycle duration must be positive")
        # The load cell already measures the plunger load, so a starved barrel
        # appears in the card itself.  Imposing the estimated fillage on top of
        # the transmission would count the collapse twice and mask any other
        # fault, so the reconstruction is a pure transmission of the measured
        # card unless the caller has an independent fillage measurement.
        self.reconstruction_fillage = 1.0 if fillage is None else min(
            max(float(fillage), 0.05), 1.0
        )

        # 1. rod string: measured surface card -> downhole pump card
        card = self.reconstruct_card(
            surface_position_m, surface_load_n, duration_s, packet.wellhead_temperature_c
        )
        # The damping actually used is recorded on the card for the risk index.
        damping = card.damping_rate_s

        # 2. AI: classify the reconstructed card
        classifier = self._ensure_classifier()
        prediction = classifier.predict(card.position_m, card.load_n)

        # 3. state estimation: fuse the instruments with the physics
        estimate = self.state_filter.step(
            packet,
            bottom_hole_temperature_prior_c=self.reservoir.chest_temperature_c,
        )
        # The flow channel cannot separate the reservoir state while the well is
        # pump limited, so the reconstructed card -- which measures the fluid
        # column directly -- is allowed to bound the inflow.  Taking the smaller
        # of the two keeps the control action conservative: the pump is never
        # run faster than the evidence supports.
        estimated_inflow = self.state_filter.reservoir_inflow(estimate)
        capacity = self.controller.delivered_rate_m3_per_day(
            self.applied_spm, self.applied_stroke_m
        )
        inflow = min(estimated_inflow, self.card_fillage(card) * capacity)

        # 4. hydraulics: pressure and temperature traverse
        traverse = self.traverse()

        # 5. control: map the fault onto a VFD setpoint
        risk = self.rod_float_risk_index(card, damping)
        recommendation = self.controller.recommend(
            label=prediction.label,
            confidence=prediction.confidence,
            inflow_m3_per_day=inflow,
            current_spm=self.applied_spm,
            current_stroke_m=self.applied_stroke_m,
            rod_float_risk_index=risk,
            autonomous=self.autonomous,
        )
        if self.autonomous and recommendation.action.startswith("REDUCE"):
            self.applied_spm = recommendation.spm
            self.applied_stroke_m = recommendation.stroke_length_m

        self._last_estimate = estimate
        self._update_alarms(prediction, estimate, recommendation)

        elapsed_ms = (time.perf_counter() - started) * 1e3
        return TwinSnapshot(
            timestamp_s=packet.timestamp_s,
            card=card,
            prediction=prediction,
            traverse=traverse,
            estimate=estimate,
            recommendation=recommendation,
            reservoir=self.reservoir,
            latency_ms=elapsed_ms,
        )

    def _update_alarms(
        self,
        prediction: CardPrediction,
        estimate: StateEstimate,
        recommendation: SetpointRecommendation,
    ) -> None:
        """Refresh the active alarm list from the current twin state."""
        alarms: List[str] = []
        if prediction.label is CardLabel.PUMP_TAGGING:
            alarms.append("PUMP TAGGING: mechanical impact at the end of the downstroke")
        if prediction.label is CardLabel.ROD_FLOATING:
            alarms.append("ROD FLOATING: load minimum driven below rod weight")
        if prediction.label is CardLabel.FLUID_POUND:
            alarms.append("FLUID POUND: the barrel cannot fill at the current SPM")
        if self.reservoir.cutoff_reached:
            alarms.append(f"CSS CUT-OFF: {self.reservoir.cutoff_reason}")
        if (
            self.reservoir.bottom_hole_temperature_c
            < THRESHOLDS.trigger_bht_c
        ):
            alarms.append(
                f"LOW BOTTOM HOLE TEMPERATURE: {self.reservoir.bottom_hole_temperature_c:.1f} degC "
                f"is below the {THRESHOLDS.trigger_bht_c:.0f} degC trigger"
            )
        if recommendation.pump_fillage < THRESHOLDS.fluid_pound_fillage:
            alarms.append(
                f"FILLAGE {recommendation.pump_fillage:.2f}: below the "
                f"{THRESHOLDS.fluid_pound_fillage:.2f} threshold"
            )
        self._alarms = alarms

    # -- control surface ---------------------------------------------------
    @property
    def reservoir_day(self) -> float:
        """Elapsed production days currently represented, days."""
        return getattr(self, "_reservoir_day", self.start_day)

    @reservoir_day.setter
    def reservoir_day(self, value: float) -> None:
        self._reservoir_day = float(value)

    @property
    def alarms(self) -> List[str]:
        """Currently active alarms."""
        return list(self._alarms)

    def apply_setpoint(self, spm: float, stroke_length_m: float) -> Dict[str, object]:
        """Apply an operator- or autonomous-approved VFD setpoint."""
        if not THRESHOLDS.min_spm <= spm <= THRESHOLDS.max_spm:
            return {
                "accepted": False,
                "spm": self.applied_spm,
                "stroke_length_m": self.applied_stroke_m,
                "message": (
                    f"SPM {spm:.2f} is outside the permitted range "
                    f"{THRESHOLDS.min_spm:.1f}-{THRESHOLDS.max_spm:.1f}"
                ),
            }
        if not 0.5 <= stroke_length_m <= 3.5:
            return {
                "accepted": False,
                "spm": self.applied_spm,
                "stroke_length_m": self.applied_stroke_m,
                "message": (
                    f"stroke {stroke_length_m:.2f} m is outside the permitted "
                    f"range 0.50-3.50 m"
                ),
            }
        self.applied_spm = float(spm)
        self.applied_stroke_m = float(stroke_length_m)
        return {
            "accepted": True,
            "spm": self.applied_spm,
            "stroke_length_m": self.applied_stroke_m,
            "message": (
                f"VFD setpoint applied: {self.applied_spm:.2f} spm at "
                f"{self.applied_stroke_m:.2f} m stroke"
            ),
        }

    def set_autonomous(self, enabled: bool) -> Dict[str, object]:
        """Enable or disable the autonomous closed loop."""
        self.autonomous = bool(enabled)
        return {
            "accepted": True,
            "autonomous": self.autonomous,
            "message": (
                "autonomous VFD closed loop enabled"
                if self.autonomous
                else "autonomous VFD closed loop disabled; operator has control"
            ),
        }

    def forecast(self) -> Optional[dict]:
        """14-day-ahead forecast from the thermal surrogate, when available."""
        if self.surrogate is None:
            return None
        history = 21
        steam = [self.reservoir.cumulative_steam_volume_m3] * history
        soak = [CSS.soak_days] * history
        casing = [RESERVOIR.initial_reservoir_pressure_mpa * 0.85] * history
        days = list(range(history))
        rate = [self.reservoir.oil_rate_tpd] * history
        temperature = [self.reservoir.bottom_hole_temperature_c] * history
        prediction = self.surrogate.predict(
            steam_m3=steam,
            soak_days=soak,
            casing_pressure_mpa=casing,
            inflow_m3_per_day=rate,
            bottom_hole_temperature_c=temperature,
        )
        return prediction.as_dict()
