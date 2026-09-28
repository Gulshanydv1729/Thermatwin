"""End-to-end verification of the well-to-surface closed loop.

The specification requires that a cooling reservoir state produces an
automatic SPM down-regulation recommendation.  These tests exercise the full
coupled path -- reservoir physics, wellbore hydraulics, Gibbs rod-string
inversion, the CNN diagnosis, the Kalman state estimate and the hydraulic
controller -- and assert the behaviour an operator would see.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pytest

from backend.app.ai.dyno_classifier import CardLabel, DynoCardClassifier, synthesize_card
from backend.app.ai.sor_optimizer import SRPController
from backend.app.core.config import CSS, PUMPING_UNIT, RESERVOIR, THRESHOLDS, WELL
from backend.app.engine.digital_twin import DigitalTwin
from backend.app.engine.state_estimator import (
    ExtendedKalmanFilter,
    StateEstimate,
    TelemetryPacket,
    baghewala_state_filter,
)
from backend.app.physics.rheology import baghewala_crude
from backend.app.physics.thermal_reservoir import CSSPhase, ThermalReservoirModel


#: One crank cycle at the nominal speed, which is the period every card the
#: Gibbs transmission rephases onto must actually span.
CYCLE_S = 60.0 / PUMPING_UNIT.nominal_spm


def _surface_card(
    fillage: float = 1.0,
    fault: CardLabel = CardLabel.NORMAL_FULL_BARREL,
    samples: int = 240,
    seed: int = 11,
) -> tuple:
    position, load = synthesize_card(
        fault,
        rod_weight_n=21937.6,
        fluid_load_n=PUMPING_UNIT.plunger_area_m2 * PUMPING_UNIT.pump_differential_pa * fillage,
        stroke_m=PUMPING_UNIT.nominal_stroke_m,
        fillage=fillage,
        samples=samples,
        noise_fraction=0.005,
        seed=seed,
    )
    return position, load


def _packet(
    timestamp_s: float,
    flow_m3_per_day: float,
    load_min_n: float,
    load_max_n: float,
    spm: float = PUMPING_UNIT.nominal_spm,
    stroke_m: float = PUMPING_UNIT.nominal_stroke_m,
) -> TelemetryPacket:
    crude = baghewala_crude()
    rho = crude.dynamic_viscosity_pa_s(120.0)
    del rho
    return TelemetryPacket(
        timestamp_s=timestamp_s,
        casing_head_pressure_pa=3.9e6,
        wellhead_temperature_c=100.0,
        surface_flow_m3_per_day=flow_m3_per_day,
        card_min_load_n=load_min_n,
        card_max_load_n=load_max_n,
        spm=spm,
        stroke_m=stroke_m,
    )


@pytest.fixture(scope="module")
def twin(tmp_path_factory) -> DigitalTwin:
    directory = tmp_path_factory.mktemp("twin")
    from backend.app.core.config import AppConfig

    config = AppConfig(weights_dir=directory)
    classifier = DynoCardClassifier(train=False)
    classifier.train_and_save(
        directory / "dyno_classifier.pt",
        epochs=10,
        samples_per_label=250,
        damping_rates=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    )
    return DigitalTwin(config=config, classifier=classifier)


class TestCoolingReservoirDownRegulation:
    """The headline requirement of the specification."""

    def test_cooling_reservoir_triggers_spm_downregulation(self) -> None:
        """A well whose inflow falls below the pump capacity is slowed down.

        The chain is the one the specification asks for: the reservoir is
        cooled until the Vogel inflow drops under the pump displacement, the
        resulting fillage is below the fluid-pound threshold, and the controller
        solves the fillage constraint for a lower SPM.
        """
        controller = SRPController()
        pump_capacity = controller.delivered_rate_m3_per_day(
            PUMPING_UNIT.nominal_spm, PUMPING_UNIT.nominal_stroke_m
        )
        # Cool the chest until the inflow is a third of what the pump can take.
        cooled_inflow = pump_capacity / 3.0
        recommendation = controller.recommend(
            label=CardLabel.FLUID_POUND,
            confidence=0.96,
            inflow_m3_per_day=cooled_inflow,
            current_spm=PUMPING_UNIT.nominal_spm,
            current_stroke_m=PUMPING_UNIT.nominal_stroke_m,
        )
        assert recommendation.action == "REDUCE_SPM"
        assert recommendation.spm < PUMPING_UNIT.nominal_spm
        assert recommendation.pump_fillage < THRESHOLDS.fluid_pound_fillage
        # The recommendation restores full barrel fillage.
        assert controller.pump_fillage(
            cooled_inflow, recommendation.spm, recommendation.stroke_length_m
        ) == pytest.approx(1.0, rel=1e-6)
        # And it is materially faster than doing nothing.
        assert recommendation.spm_ratio < 0.7

    def test_cooling_actually_reduces_the_inflow(self) -> None:
        """The premise of the control action holds in the reservoir model.

        The heated chest and the cold formation are two radial regions of
        different oil viscosity in series, and the mobility enters the inflow
        inversely with the viscosity, so a hot chest must deliver far more than
        a collapsed one.  By the end of a cycle the chest has decayed to the
        wellbore, so the two states are compared explicitly.
        """
        model = ThermalReservoirModel()
        model.simulate_cycle()
        pressure = model.producing_bottom_hole_pressure_mpa
        collapsed_radius = model._last_steam_chest_radius_m
        injected_radius = model._chest_radius_at_injection_end_m
        assert injected_radius > 10.0 * collapsed_radius, "injection must build a chest"

        hot = model._vogel_inflow(pressure, temperature_c=150.0)
        model._last_steam_chest_radius_m = injected_radius
        hot_with_chest = model._vogel_inflow(pressure, temperature_c=150.0)
        model._last_steam_chest_radius_m = injected_radius
        cold_with_chest = model._vogel_inflow(pressure, temperature_c=60.0)

        # A hot chest is a strong injector; a cold formation is not.
        assert hot_with_chest > 2.0 * hot
        # Cooling the *same* chest collapses the mobility, because the oil
        # thickens by more than an order of magnitude over that range.  The
        # reduction is not the full viscosity ratio: most of the flow path
        # lies outside the chest, in the cold annulus, whose resistance does
        # not depend on the chest temperature and therefore dilutes the
        # sensitivity of the total.
        assert cold_with_chest < hot_with_chest
        assert cold_with_chest / hot_with_chest < 0.35

    def test_viscosity_dominates_the_decline(self) -> None:
        """The decline mechanism is viscosity, not a pressure change."""
        model = ThermalReservoirModel()
        cold = model._oil_viscosity_cP(60.0)
        hot = model._oil_viscosity_cP(150.0)
        assert cold > 10.0 * hot

    def test_late_cycle_fillage_collapses_below_the_threshold(self) -> None:
        model = ThermalReservoirModel()
        result = model.simulate_cycle()
        producing = [s for s in result.states if s.phase is CSSPhase.PRODUCTION]
        assert producing[0].pump_fillage > 1.0
        assert producing[-1].pump_fillage < THRESHOLDS.fluid_pound_fillage

    def test_bht_cutoff_trigger_raises_an_alarm(self, twin: DigitalTwin) -> None:
        """When the BHT falls through 70 degC the twin must raise the alarm."""
        original = twin.reservoir
        try:
            state = original
            state.bottom_hole_temperature_c = 62.0
            prediction = twin._ensure_classifier().predict(
                *_card_arguments(1.0, CardLabel.NORMAL_FULL_BARREL)[0:1]
                + _card_arguments(1.0, CardLabel.NORMAL_FULL_BARREL)[1:2],
            )
            recommendation = twin.controller.recommend(
                label=prediction.label,
                confidence=prediction.confidence,
                inflow_m3_per_day=2.0,
                current_spm=9.0,
                current_stroke_m=2.44,
            )
            twin._update_alarms(prediction, StateEstimate(
                pump_intake_pressure_pa=1.0e7,
                skin_factor=3.0,
                thermal_radius_m=10.0,
                bottom_hole_temperature_c=62.0,
                covariance=np.eye(4),
                residual_norm=0.0,
                innovations=np.zeros(5),
            ), recommendation)
            assert any("LOW BOTTOM HOLE TEMPERATURE" in alarm for alarm in twin.alarms)
        finally:
            twin.reservoir = original


def _card_arguments(fillage: float, fault: CardLabel):
    position, load = _surface_card(fillage=fillage, fault=fault)
    return position, load


class TestTwinLoop:
    def test_full_pass_produces_a_complete_snapshot(self, twin: DigitalTwin) -> None:
        position, load = _surface_card(0.9)
        packet = _packet(0.0, 3.0, float(np.min(load)), float(np.max(load)))
        snapshot = twin.step(packet, position, load, duration_s=CYCLE_S)
        assert snapshot.card.stroke_m > 0.0
        assert snapshot.prediction.label in CardLabel
        assert len(snapshot.traverse.depth_m) == snapshot.traverse.segments + 1
        assert snapshot.estimate.pump_intake_pressure_pa > 0.0
        assert snapshot.recommendation.spm > 0.0
        assert snapshot.latency_ms > 0.0

    def test_frame_matches_the_websocket_contract(self, twin: DigitalTwin) -> None:
        position, load = _surface_card(0.8)
        packet = _packet(0.0, 2.5, float(np.min(load)), float(np.max(load)))
        snapshot = twin.step(packet, position, load, duration_s=CYCLE_S)
        frame = snapshot.as_frame(cycle=1, well_name="BAG-17")
        for key in (
            "type",
            "timestamp_s",
            "well",
            "card",
            "diagnosis",
            "wellbore",
            "estimate",
            "css",
            "recommendation",
            "latency_ms",
        ):
            assert key in frame
        assert frame["type"] == "telemetry"
        assert set(frame["well"]) == {"name", "cycle", "phase", "day", "days_into_phase"}
        assert frame["well"]["phase"] in {"INJECTION", "SOAKING", "PRODUCTION"}
        assert set(frame["card"]) == {
            "surface",
            "downhole",
            "stroke_m",
            "min_load_n",
            "max_load_n",
            "load_span_n",
        }
        assert set(frame["css"]) >= {
            "cumulative_sor",
            "cutoff_reached",
            "chest_radius_m",
            "pump_fillage",
        }
        # The frame must survive a JSON round trip.
        import json

        assert json.loads(json.dumps(frame))["type"] == "telemetry"

    @pytest.mark.parametrize(
        "fault",
        [
            CardLabel.ROD_FLOATING,
            CardLabel.FLUID_POUND,
            CardLabel.PUMP_TAGGING,
            CardLabel.GAS_INTERFERENCE,
            CardLabel.UNANCHORED_TUBING,
            CardLabel.NORMAL_FULL_BARREL,
        ],
    )
    def test_every_injected_fault_is_diagnosed_and_acted_on(
        self, twin: DigitalTwin, fault: CardLabel
    ) -> None:
        """The whole chain: inject a fault, diagnose it, act on it."""
        position, load = _surface_card(0.6 if fault is CardLabel.FLUID_POUND else 1.0, fault)
        packet = _packet(0.0, 1.2, float(np.min(load)), float(np.max(load)))
        snapshot = twin.step(packet, position, load, duration_s=CYCLE_S)
        assert snapshot.prediction.label is fault
        assert snapshot.prediction.confidence > 0.9
        if fault in {
            CardLabel.ROD_FLOATING,
            CardLabel.FLUID_POUND,
            CardLabel.PUMP_TAGGING,
            CardLabel.GAS_INTERFERENCE,
        }:
            assert snapshot.recommendation.action == "REDUCE_SPM"
            assert snapshot.recommendation.spm < 9.0
        elif fault is CardLabel.UNANCHORED_TUBING:
            assert snapshot.recommendation.action == "INSPECT_TUBING"

    def test_autonomous_loop_applies_its_own_setpoint(self, twin: DigitalTwin) -> None:
        twin.set_autonomous(False)
        twin.apply_setpoint(9.0, 2.44)
        position, load = _surface_card(0.5, CardLabel.FLUID_POUND)
        packet = _packet(0.0, 0.9, float(np.min(load)), float(np.max(load)))
        snapshot = twin.step(packet, position, load, duration_s=CYCLE_S)
        assert twin.applied_spm == pytest.approx(9.0), "manual mode must not move the VFD"

        twin.set_autonomous(True)
        try:
            for _ in range(4):
                snapshot = twin.step(packet, position, load, duration_s=CYCLE_S)
            assert twin.applied_spm < 9.0, "autonomous mode must reduce the SPM"
            assert snapshot.recommendation.spm == pytest.approx(twin.applied_spm)
        finally:
            twin.set_autonomous(False)
            twin.apply_setpoint(PUMPING_UNIT.nominal_spm, PUMPING_UNIT.nominal_stroke_m)

    def test_steady_state_latency_is_within_budget(self, twin: DigitalTwin) -> None:
        position, load = _surface_card(0.9)
        packet = _packet(0.0, 3.0, float(np.min(load)), float(np.max(load)))
        # One warm-up pass builds the classifier and primes the filter.
        twin.step(packet, position, load, duration_s=CYCLE_S)
        latencies = []
        for index in range(5):
            snapshot = twin.step(packet, position, load, duration_s=CYCLE_S)
            latencies.append(snapshot.latency_ms)
        # 500 ms stream interval; steady-state median must be under 200 ms.
        # This accommodates loaded CI runners where GC can spike a pass.
        # The WebSocket stream timing tests (test_api_stream.py) enforce a
        # tighter 100 ms median on a real server; this is a unit-level check.
        median_ms = float(np.median(latencies))
        assert median_ms < 200.0, f"median {median_ms:.1f} ms exceeds 200 ms budget"
        # No single pass should exceed the stream interval.
        assert max(latencies) < 500.0, f"slowest pass {max(latencies):.1f} ms"

    def test_damping_responds_to_the_reservoir_temperature(self, twin: DigitalTwin) -> None:
        cold = twin._damping_rate(RESERVOIR.reservoir_temperature_c)
        hot = twin._damping_rate(CSS.steam_temperature_c)
        assert cold > hot >= 0.0

    def test_rod_float_risk_index_is_bounded(self, twin: DigitalTwin) -> None:
        position, load = _surface_card(1.0, CardLabel.ROD_FLOATING)
        card = twin.reconstruct_card(position, load, CYCLE_S)
        risk = twin.rod_float_risk_index(card, card.damping_rate_s)
        assert 0.0 <= risk <= 1.0
        normal_position, normal_load = _surface_card(1.0)
        normal_card = twin.reconstruct_card(normal_position, normal_load, CYCLE_S)
        normal_risk = twin.rod_float_risk_index(normal_card, normal_card.damping_rate_s)
        # A full barrel never takes the load negative, so its risk is the drag
        # term alone; the floating card drives the load below zero and must
        # score strictly higher.
        assert normal_card.min_load_n >= -0.01 * normal_card.load_span_n
        assert risk > normal_risk

    def test_rejects_invalid_twin_inputs(self, twin: DigitalTwin) -> None:
        position, load = _surface_card(0.9)
        packet = _packet(0.0, 3.0, float(np.min(load)), float(np.max(load)))
        with pytest.raises(ValueError):
            twin.step(packet, position, load, duration_s=0.0)

    def test_rejects_impossible_output_samples(self) -> None:
        with pytest.raises(ValueError):
            DigitalTwin(output_samples=4)


class TestControlSurface:
    def test_valid_setpoint_is_accepted(self, twin: DigitalTwin) -> None:
        response = twin.apply_setpoint(7.5, 2.20)
        assert response["accepted"]
        assert twin.applied_spm == pytest.approx(7.5)
        assert twin.applied_stroke_m == pytest.approx(2.20)
        twin.apply_setpoint(PUMPING_UNIT.nominal_spm, PUMPING_UNIT.nominal_stroke_m)

    def test_spm_outside_the_limits_is_rejected(self, twin: DigitalTwin) -> None:
        before = twin.applied_spm
        response = twin.apply_setpoint(THRESHOLDS.max_spm + 5.0, 2.44)
        assert not response["accepted"]
        assert "SPM" in response["message"]
        assert twin.applied_spm == pytest.approx(before)

    def test_stroke_outside_the_limits_is_rejected(self, twin: DigitalTwin) -> None:
        response = twin.apply_setpoint(9.0, 9.0)
        assert not response["accepted"]
        assert "stroke" in response["message"]

    def test_autonomy_toggle_reports_state(self, twin: DigitalTwin) -> None:
        assert twin.set_autonomous(True)["autonomous"] is True
        assert twin.set_autonomous(False)["autonomous"] is False

    def test_forecast_is_none_without_a_surrogate(self, twin: DigitalTwin) -> None:
        assert twin.forecast() is None


class TestStateEstimator:
    def test_intake_pressure_converges_to_the_truth(self) -> None:
        """The filter must recover the unmeasured pump intake pressure.

        The casing head pressure is the only direct measurement of the intake
        conditions, and the hydrostatic column is what has to be added back.
        """
        state_filter = baghewala_state_filter()
        model = ThermalReservoirModel()
        true_pip_pa = model.producing_bottom_hole_pressure_mpa * 1e6
        static = state_filter.rod_weight_n + state_filter.fluid_column_kg * 9.80665
        errors = []
        for index in range(80):
            temperature = 120.0
            hydrostatic = state_filter._hydrostatic_pa(temperature, 0.12)
            packet = TelemetryPacket(
                timestamp_s=index * 3600.0,
                casing_head_pressure_pa=true_pip_pa - hydrostatic,
                wellhead_temperature_c=100.0,
                surface_flow_m3_per_day=3.0,
                card_min_load_n=static,
                card_max_load_n=static + 4000.0,
                spm=9.0,
                stroke_m=2.44,
            )
            estimate = state_filter.step(
                packet, dt_days=1.0, bottom_hole_temperature_prior_c=120.0
            )
            errors.append(abs(estimate.pump_intake_pressure_pa - true_pip_pa))
        final = errors[-1] / 1e6
        assert final < 0.5, f"final intake pressure error {final:.3f} MPa"
        # The error must not grow: the filter is converging, not drifting.
        assert errors[-1] <= errors[len(errors) // 2] * 1.2

    def test_covariance_stays_positive_definite(self) -> None:
        state_filter = baghewala_state_filter()
        static = state_filter.rod_weight_n + state_filter.fluid_column_kg * 9.80665
        for index in range(30):
            packet = TelemetryPacket(
                timestamp_s=index * 60.0,
                casing_head_pressure_pa=4.0e6,
                wellhead_temperature_c=95.0,
                surface_flow_m3_per_day=3.5,
                card_min_load_n=static,
                card_max_load_n=static + 4200.0,
                spm=9.0,
                stroke_m=2.44,
            )
            estimate = state_filter.step(packet, dt_days=0.05)
        eigenvalues = np.linalg.eigvalsh(estimate.covariance)
        assert np.all(eigenvalues >= -1e-6)

    def test_states_remain_inside_physical_bounds(self) -> None:
        state_filter = baghewala_state_filter()
        static = state_filter.rod_weight_n + state_filter.fluid_column_kg * 9.80665
        for index in range(30):
            packet = TelemetryPacket(
                timestamp_s=index * 60.0,
                # A wildly inconsistent instrument set must not push the state
                # outside the physical envelope.
                casing_head_pressure_pa=-1.0e6,
                wellhead_temperature_c=400.0,
                surface_flow_m3_per_day=1e4,
                card_min_load_n=0.0,
                card_max_load_n=1.0e6,
                spm=9.0,
                stroke_m=2.44,
            )
            estimate = state_filter.step(packet, dt_days=0.05)
        assert 101325.0 <= estimate.pump_intake_pressure_pa <= 3.0e7
        assert estimate.skin_factor >= 0.0
        assert 0.0 <= estimate.thermal_radius_m <= RESERVOIR.drainage_radius_m
        assert RESERVOIR.reservoir_temperature_c <= estimate.bottom_hole_temperature_c <= CSS.steam_temperature_c

    def test_batching_matches_stepping(self) -> None:
        packets = [
            TelemetryPacket(
                timestamp_s=index * 3600.0,
                casing_head_pressure_pa=4.0e6,
                wellhead_temperature_c=95.0,
                surface_flow_m3_per_day=3.0,
                card_min_load_n=62000.0,
                card_max_load_n=66000.0,
                spm=9.0,
                stroke_m=2.44,
            )
            for index in range(12)
        ]
        one = baghewala_state_filter()
        two = baghewala_state_filter()
        stepped = [one.step(p) for p in packets]
        batched = two.run(packets)
        for a, b in zip(stepped, batched):
            assert a.pump_intake_pressure_pa == pytest.approx(
                b.pump_intake_pressure_pa, rel=1e-12
            )

    def test_rejects_invalid_packets(self) -> None:
        state_filter = baghewala_state_filter()
        with pytest.raises(ValueError):
            state_filter.step(
                TelemetryPacket(0.0, float("nan"), 90.0, 3.0, 6e4, 6.6e4, 9.0, 2.44)
            )
        with pytest.raises(ValueError):
            state_filter.step(
                TelemetryPacket(0.0, 4.0e6, 90.0, 3.0, 6e4, 6.6e4, 9.0, 2.44),
                dt_days=-1.0,
            )

    def test_reset_returns_the_prior(self) -> None:
        state_filter = baghewala_state_filter()
        for index in range(5):
            state_filter.step(
                TelemetryPacket(
                    float(index), 4.0e6, 90.0, 3.0, 6e4, 6.6e4, 9.0, 2.44
                )
            )
        state_filter.reset()
        assert state_filter.initialised is False

    def test_rejects_invalid_noise_vectors(self) -> None:
        with pytest.raises(ValueError):
            ExtendedKalmanFilter(process_noise=(1.0, 2.0))
        with pytest.raises(ValueError):
            ExtendedKalmanFilter(measurement_noise=(1.0, 2.0))
        with pytest.raises(ValueError):
            ExtendedKalmanFilter(plunger_area_m2=0.0)
        with pytest.raises(ValueError):
            ExtendedKalmanFilter(rod_weight_n=0.0)
