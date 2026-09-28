"""Accuracy verification for the AI engines.

Three models are exercised end to end:

* the residual 1D CNN dyno card classifier, on synthetic cards whose ground
  truth is known by construction;
* the analytic geometric cross-check, which must agree with the network;
* the bidirectional GRU thermal surrogate, against the physics trajectories it
  was trained on.

The specification requires recall above 95% for ``ROD_FLOATING`` and
``FLUID_POUND`` specifically; that is asserted on a held-out sample set that is
disjoint from the training draw.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from backend.app.ai.dyno_classifier import (
    CARD_LABELS,
    LABEL_INDEX,
    CardLabel,
    DynoCardClassifier,
    analytic_diagnosis,
    build_training_set,
    card_features,
    normalise_card,
    synthesize_card,
)
from backend.app.ai.sor_optimizer import (
    CSSSchedule,
    CSOScheduleOptimizer,
    SRPController,
)
from backend.app.ai.thermal_surrogate import (
    INPUT_CHANNELS,
    OUTPUT_CHANNELS,
    SURROGATE_HORIZON_DAYS,
    SURROGATE_SEQUENCE_LENGTH,
    ThermalSurrogate,
    build_supervised_windows,
    generate_training_trajectories,
)
from backend.app.core.config import PUMPING_UNIT, THRESHOLDS


@pytest.fixture(scope="module")
def classifier(tmp_path_factory) -> DynoCardClassifier:
    """A trained classifier, cached in the test's temporary directory."""
    directory = tmp_path_factory.mktemp("weights")
    model = DynoCardClassifier(train=False)
    model.train_and_save(
        directory / "dyno_classifier.pt", epochs=12, samples_per_label=300
    )
    return model


def _card(label: CardLabel, seed: int, fillage: float | None = None):
    if fillage is None:
        fillage = 0.55 if label is CardLabel.FLUID_POUND else 1.0
    return synthesize_card(
        label,
        rod_weight_n=21937.6,
        fluid_load_n=26000.0,
        stroke_m=PUMPING_UNIT.nominal_stroke_m,
        fillage=fillage,
        samples=512,
        noise_fraction=0.008,
        seed=seed,
    )


class TestCardLabels:
    def test_all_six_states_are_defined(self) -> None:
        assert len(CARD_LABELS) == 6
        expected = {
            "NORMAL_FULL_BARREL",
            "FLUID_POUND",
            "ROD_FLOATING",
            "GAS_INTERFERENCE",
            "PUMP_TAGGING",
            "UNANCHORED_TUBING",
        }
        assert {label.value for label in CARD_LABELS} == expected
        assert len(LABEL_INDEX) == 6


class TestParametricCardModel:
    def test_full_barrel_is_a_parallelogram(self) -> None:
        position, load = _card(CardLabel.NORMAL_FULL_BARREL, 1)
        features = card_features(position, load)
        # A full barrel traces a straight-sided parallelogram.
        assert features["linearity"] < 0.03
        assert features["upstroke_reversal"] < 0.05
        assert features["downstroke_asymmetry"] == pytest.approx(0.0, abs=0.05)
        assert features["skew"] == pytest.approx(0.0, abs=0.05)

    def test_fluid_pound_reverses_the_upstroke(self) -> None:
        """The defining signature: the load falls while the plunger rises."""
        position, load = _card(CardLabel.FLUID_POUND, 2, fillage=0.5)
        features = card_features(position, load)
        assert features["upstroke_reversal"] > 0.15
        assert analytic_diagnosis(features) is CardLabel.FLUID_POUND

    def test_rod_float_depresses_the_downstroke(self) -> None:
        position, load = _card(CardLabel.ROD_FLOATING, 3)
        features = card_features(position, load)
        assert features["downstroke_asymmetry"] < -0.05
        assert analytic_diagnosis(features) is CardLabel.ROD_FLOATING

    def test_pump_tagging_spikes_at_the_end_of_the_downstroke(self) -> None:
        position, load = _card(CardLabel.PUMP_TAGGING, 4)
        features = card_features(position, load)
        assert features["tail_spike"] > 0.15
        assert analytic_diagnosis(features) is CardLabel.PUMP_TAGGING

    def test_unanchored_tubing_lags_the_load(self) -> None:
        position, load = _card(CardLabel.UNANCHORED_TUBING, 5)
        features = card_features(position, load)
        assert features["downstroke_asymmetry"] > 0.1
        assert analytic_diagnosis(features) is CardLabel.UNANCHORED_TUBING

    def test_gas_interference_rounds_the_card(self) -> None:
        position, load = _card(CardLabel.GAS_INTERFERENCE, 6)
        features = card_features(position, load)
        assert features["linearity"] > 0.05
        assert analytic_diagnosis(features) is CardLabel.GAS_INTERFERENCE

    def test_every_state_is_identified_analytically(self) -> None:
        """The geometric cross-check is exact over many noise realisations."""
        correct = 0
        total = 0
        for trial in range(25):
            for label in CARD_LABELS:
                position, load = _card(label, 1000 + trial)
                total += 1
                if analytic_diagnosis(card_features(position, load)) is label:
                    correct += 1
        assert correct == total, f"analytic accuracy {correct}/{total}"

    def test_rejects_invalid_card_input(self) -> None:
        with pytest.raises(ValueError):
            card_features([0.0] * 4, [0.0] * 4)
        with pytest.raises(ValueError):
            card_features([0.0] * 8, [0.0] * 4)
        with pytest.raises(ValueError):
            card_features([1.0] * 8, [1.0] * 8)  # zero stroke
    def test_normalisation_raises_on_zero_stroke(self) -> None:
        with pytest.raises(ValueError):
            normalise_card([1.0] * 32, [2.0] * 32)


class TestNetworkAccuracy:
    def test_validation_accuracy_is_high(self, classifier: DynoCardClassifier) -> None:
        assert classifier.metrics["validation_accuracy"] > 0.95

    def test_per_class_recall_is_high(self, classifier: DynoCardClassifier) -> None:
        for label in CARD_LABELS:
            recall = classifier.metrics.get(label.value, 0.0)
            assert recall > 0.90, f"{label.value} recall {recall:.3f}"

    def test_rod_float_recall_exceeds_ninety_five_percent(
        self, classifier: DynoCardClassifier
    ) -> None:
        """Specification requirement for ROD_FLOATING."""
        assert classifier.metrics[CardLabel.ROD_FLOATING.value] > 0.95

    def test_fluid_pound_recall_exceeds_ninety_five_percent(
        self, classifier: DynoCardClassifier
    ) -> None:
        """Specification requirement for FLUID_POUND."""
        assert classifier.metrics[CardLabel.FLUID_POUND.value] > 0.95

    def test_held_out_recall_meets_the_specification(
        self, classifier: DynoCardClassifier
    ) -> None:
        """Recall on cards drawn after training, over a randomised envelope."""
        rng = np.random.default_rng(9001)
        hits: Dict[str, int] = {label.value: 0 for label in CARD_LABELS}
        trials = 40
        for label in CARD_LABELS:
            for _ in range(trials):
                stroke = float(rng.uniform(1.8, 3.0))
                if label is CardLabel.FLUID_POUND:
                    fillage = float(rng.uniform(0.35, 0.85))
                elif label is CardLabel.ROD_FLOATING:
                    fillage = float(rng.uniform(0.85, 1.0))
                else:
                    fillage = float(rng.uniform(0.92, 1.0))
                position, load = synthesize_card(
                    label,
                    rod_weight_n=float(rng.uniform(15000.0, 30000.0)),
                    fluid_load_n=float(rng.uniform(12000.0, 45000.0)),
                    stroke_m=stroke,
                    fillage=fillage,
                    samples=512,
                    noise_fraction=float(rng.uniform(0.002, 0.03)),
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
                if classifier.predict(position, load).label is label:
                    hits[label.value] += 1
        for label in CARD_LABELS:
            recall = hits[label.value] / trials
            assert recall > 0.95, f"{label.value} held-out recall {recall:.2f}"
        assert (
            hits[CardLabel.ROD_FLOATING.value] / trials > 0.95
        )
        assert hits[CardLabel.FLUID_POUND.value] / trials > 0.95

    def test_confidence_is_reported_and_bounded(
        self, classifier: DynoCardClassifier
    ) -> None:
        position, load = _card(CardLabel.ROD_FLOATING, 77)
        prediction = classifier.predict(position, load)
        assert 0.0 <= prediction.confidence <= 1.0
        assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
        assert set(prediction.probabilities) == {label.value for label in CARD_LABELS}

    def test_network_agrees_with_the_analytic_cross_check(
        self, classifier: DynoCardClassifier
    ) -> None:
        """Two independent routes to the same answer, over unseen cards."""
        agree = 0
        total = 0
        for trial in range(500, 530):
            for label in CARD_LABELS:
                position, load = _card(label, trial)
                prediction = classifier.predict(position, load)
                total += 1
                if prediction.agrees_with_analytic:
                    agree += 1
        assert agree / total > 0.95, f"agreement {agree}/{total}"

    def test_batch_prediction_matches_single_prediction(
        self, classifier: DynoCardClassifier
    ) -> None:
        positions, loads = [], []
        for label in CARD_LABELS:
            position, load = _card(label, 31)
            positions.append(position)
            loads.append(load)
        batched = classifier.predict_batch(positions, loads)
        assert len(batched) == len(CARD_LABELS)
        for index, (position, load) in enumerate(zip(positions, loads)):
            single = classifier.predict(position, load)
            assert batched[index].label is single.label
            assert batched[index].confidence == pytest.approx(single.confidence, rel=1e-5)

    def test_weights_round_trip(self, classifier: DynoCardClassifier, tmp_path: Path) -> None:
        path = tmp_path / "round_trip.pt"
        classifier.save(path)
        assert path.exists()
        restored = DynoCardClassifier(weights_path=path, train=False)
        restored.load()
        position, load = _card(CardLabel.PUMP_TAGGING, 44)
        assert restored.predict(position, load).label is CardLabel.PUMP_TAGGING

    def test_untrained_classifier_refuses_to_predict(self) -> None:
        model = DynoCardClassifier(train=False)
        position, load = _card(CardLabel.NORMAL_FULL_BARREL, 1)
        with pytest.raises(RuntimeError):
            model.predict(position, load)

    def test_empty_batch_is_handled(self, classifier: DynoCardClassifier) -> None:
        assert classifier.predict_batch([], []) == []


class TestThermalSurrogate:
    @pytest.fixture(scope="class")
    def surrogate(self, tmp_path_factory) -> ThermalSurrogate:
        directory = tmp_path_factory.mktemp("surrogate")
        model = ThermalSurrogate(train=False)
        model.train_and_save(
            directory / "thermal_surrogate.pt",
            epochs=50,
            window_stride=6,
            max_windows=2500,
        )
        return model

    def test_training_trajectories_cover_the_parameter_space(self) -> None:
        trajectories = generate_training_trajectories()
        assert len(trajectories) >= 8
        for trajectory in trajectories:
            assert len(trajectory) > SURROGATE_SEQUENCE_LENGTH + SURROGATE_HORIZON_DAYS
            assert np.all(trajectory.inflow_m3_per_day >= 0.0)
            assert np.all(trajectory.bottom_hole_temperature_c > 0.0)

    def test_windows_have_the_declared_shape(self) -> None:
        trajectories = generate_training_trajectories(cycles=1, production_days=40.0)
        xs, ys = build_supervised_windows(trajectories, stride=4)
        assert xs.shape[1:] == (SURROGATE_SEQUENCE_LENGTH, INPUT_CHANNELS)
        assert ys.shape[1:] == (OUTPUT_CHANNELS,)
        assert np.all(np.isfinite(xs))
        assert np.all(ys > 0.0)

    def test_temperature_forecast_is_accurate(self, surrogate: ThermalSurrogate) -> None:
        assert surrogate.metrics["temperature_r2"] > 0.85
        assert surrogate.metrics["temperature_mae_c"] < 2.0

    def test_inflow_forecast_is_accurate(self, surrogate: ThermalSurrogate) -> None:
        """The inflow error is scored as a fraction of the mean rate."""
        trajectories = generate_training_trajectories()
        rates = np.concatenate([t.inflow_m3_per_day for t in trajectories])
        mean_rate = float(np.mean(rates[rates > 0.0]))
        assert surrogate.metrics["inflow_mae_m3_d"] < 0.05 * mean_rate

    def test_forecast_is_physically_bounded(self, surrogate: ThermalSurrogate) -> None:
        trajectory = generate_training_trajectories()[0]
        prediction = surrogate.predict(
            trajectory.steam_m3[:SURROGATE_SEQUENCE_LENGTH],
            trajectory.soak_days[:SURROGATE_SEQUENCE_LENGTH],
            trajectory.casing_pressure_mpa[:SURROGATE_SEQUENCE_LENGTH],
            trajectory.inflow_m3_per_day[:SURROGATE_SEQUENCE_LENGTH],
            trajectory.bottom_hole_temperature_c[:SURROGATE_SEQUENCE_LENGTH],
        )
        assert prediction.inflow_m3_per_day > 0.0
        assert 40.0 < prediction.bottom_hole_temperature_c < 240.0
        assert prediction.horizon_days == SURROGATE_HORIZON_DAYS

    def test_short_history_is_resampled(self, surrogate: ThermalSurrogate) -> None:
        trajectory = generate_training_trajectories()[0]
        prediction = surrogate.predict(
            trajectory.steam_m3[:9],
            trajectory.soak_days[:9],
            trajectory.casing_pressure_mpa[:9],
            trajectory.inflow_m3_per_day[:9],
            trajectory.bottom_hole_temperature_c[:9],
        )
        assert np.isfinite(prediction.inflow_m3_per_day)
        assert np.isfinite(prediction.bottom_hole_temperature_c)

    def test_rejects_mismatched_input(self, surrogate: ThermalSurrogate) -> None:
        trajectory = generate_training_trajectories()[0]
        n = SURROGATE_SEQUENCE_LENGTH
        with pytest.raises(ValueError):
            surrogate.predict(
                trajectory.steam_m3[:n],
                trajectory.soak_days[: n - 1],
                trajectory.casing_pressure_mpa[:n],
                trajectory.inflow_m3_per_day[:n],
                trajectory.bottom_hole_temperature_c[:n],
            )
        with pytest.raises(ValueError):
            surrogate.predict([1.0], [1.0], [1.0], [1.0], [1.0])

    def test_untrained_surrogate_refuses_to_predict(self) -> None:
        model = ThermalSurrogate(train=False)
        with pytest.raises(RuntimeError):
            model.predict([1.0] * 21, [1.0] * 21, [1.0] * 21, [1.0] * 21, [1.0] * 21)

    def test_weights_round_trip(self, surrogate: ThermalSurrogate, tmp_path: Path) -> None:
        path = tmp_path / "surrogate_round_trip.pt"
        surrogate.save(path)
        restored = ThermalSurrogate(weights_path=path, train=False)
        restored.load()
        trajectory = generate_training_trajectories()[0]
        n = SURROGATE_SEQUENCE_LENGTH
        fresh = restored.predict(
            trajectory.steam_m3[:n],
            trajectory.soak_days[:n],
            trajectory.casing_pressure_mpa[:n],
            trajectory.inflow_m3_per_day[:n],
            trajectory.bottom_hole_temperature_c[:n],
        )
        original = surrogate.predict(
            trajectory.steam_m3[:n],
            trajectory.soak_days[:n],
            trajectory.casing_pressure_mpa[:n],
            trajectory.inflow_m3_per_day[:n],
            trajectory.bottom_hole_temperature_c[:n],
        )
        assert fresh.inflow_m3_per_day == pytest.approx(
            original.inflow_m3_per_day, rel=1e-6
        )


class TestSOROptimizer:
    def test_schedule_evaluation_is_internally_consistent(self) -> None:
        optimiser = CSOScheduleOptimizer(cycles_bounds=(1, 2), production_bounds=(30.0, 40.0))
        result = optimiser.evaluate(CSSSchedule(6.0, 4.0, 30.0, 1))
        assert result.steam_m3 > 0.0
        assert result.oil_bbl > 0.0
        assert result.cumulative_sor > 0.0
        assert result.net_revenue_usd == pytest.approx(
            result.oil_bbl * 68.0 - result.steam_m3 * 21.0 - result.oil_bbl * 11.0, rel=1e-9
        )

    def test_infeasible_schedules_are_penalised(self) -> None:
        optimiser = CSOScheduleOptimizer(
            sor_cutoff=0.05, cycles_bounds=(1, 2), production_bounds=(30.0, 40.0)
        )
        result = optimiser.evaluate(CSSSchedule(6.0, 4.0, 30.0, 1))
        assert not result.feasible
        assert "SOR" in result.reason
        assert optimiser.objective(result.schedule) < -1.0e6

    def test_optimiser_returns_a_feasible_schedule(self) -> None:
        optimiser = CSOScheduleOptimizer(
            cycles_bounds=(1, 3), production_bounds=(30.0, 60.0)
        )
        result = optimiser.optimise()
        assert result.feasible
        assert optimiser.injection_bounds[0] <= result.schedule.injection_days
        assert result.schedule.injection_days <= optimiser.injection_bounds[1]
        assert result.net_revenue_usd > 0.0

    def test_more_steam_earns_more_when_the_well_is_reservoir_limited(self) -> None:
        optimiser = CSOScheduleOptimizer(cycles_bounds=(1, 2), production_bounds=(40.0, 60.0))
        light = optimiser.evaluate(CSSSchedule(4.0, 4.0, 40.0, 1))
        heavy = optimiser.evaluate(CSSSchedule(14.0, 4.0, 40.0, 1))
        assert heavy.oil_bbl > light.oil_bbl
        assert heavy.steam_m3 > light.steam_m3
        assert heavy.net_revenue_usd > light.net_revenue_usd

    def test_evaluation_does_not_leak_state_between_candidates(self) -> None:
        """Two identical evaluations must agree, so no chest state carries over."""
        optimiser = CSOScheduleOptimizer(cycles_bounds=(1, 2), production_bounds=(30.0, 40.0))
        schedule = CSSSchedule(6.0, 4.0, 30.0, 1)
        first = optimiser.evaluate(schedule)
        second = optimiser.evaluate(schedule)
        assert first.oil_bbl == pytest.approx(second.oil_bbl, rel=1e-12)
        assert first.steam_m3 == pytest.approx(second.steam_m3, rel=1e-12)

    def test_rejects_invalid_bounds(self) -> None:
        with pytest.raises(ValueError):
            CSOScheduleOptimizer(injection_bounds=(10.0, 4.0))
        with pytest.raises(ValueError):
            CSOScheduleOptimizer(soak_bounds=(10.0, 4.0))
        with pytest.raises(ValueError):
            CSOScheduleOptimizer(production_bounds=(90.0, 30.0))
        with pytest.raises(ValueError):
            CSOScheduleOptimizer(cycles_bounds=(0, 3))


class TestSRPController:
    def test_pump_capacity_matches_the_kinematic_formula(self) -> None:
        controller = SRPController()
        rate = controller.delivered_rate_m3_per_day(9.0, 2.44)
        expected = (
            9.0 * 2.0 * 60.0 * PUMPING_UNIT.plunger_area_m2 * 2.44 * PUMPING_UNIT.pump_efficiency
        )
        assert rate == pytest.approx(expected, rel=1e-12)
        # 4.88 m3/day is about 30.7 bbl/d for this installation.
        assert 20.0 < rate * 6.2898 < 45.0

    def test_fluid_pound_reduces_spm_to_match_the_inflow(self) -> None:
        """The specification's control action, solved from the fillage constraint."""
        controller = SRPController()
        inflow = 1.5  # m3/day, well below the 4.88 m3/day pump capacity
        recommendation = controller.recommend(
            label=CardLabel.FLUID_POUND,
            confidence=0.97,
            inflow_m3_per_day=inflow,
            current_spm=9.0,
            current_stroke_m=2.44,
        )
        assert recommendation.action == "REDUCE_SPM"
        assert recommendation.spm < 9.0
        # At the recommended SPM the pump exactly matches the inflow.
        assert controller.delivered_rate_m3_per_day(recommendation.spm, 2.44) == pytest.approx(
            inflow, rel=1e-6
        )
        # So the fillage is restored to unity.
        assert controller.pump_fillage(inflow, recommendation.spm, 2.44) == pytest.approx(
            1.0, rel=1e-6
        )

    def test_rod_float_reduces_the_downstroke_speed(self) -> None:
        controller = SRPController()
        recommendation = controller.recommend(
            label=CardLabel.ROD_FLOATING,
            confidence=0.95,
            inflow_m3_per_day=4.8,
            current_spm=9.0,
            current_stroke_m=2.44,
            rod_float_risk_index=0.8,
        )
        assert recommendation.action == "REDUCE_SPM"
        assert recommendation.spm == pytest.approx(
            9.0 * THRESHOLDS.rod_float_spm_factor, rel=1e-9
        )
        assert "rod float risk index" in recommendation.reason

    def test_normal_pump_holds_the_setpoint(self) -> None:
        controller = SRPController()
        recommendation = controller.recommend(
            label=CardLabel.NORMAL_FULL_BARREL,
            confidence=0.99,
            inflow_m3_per_day=6.0,
            current_spm=9.0,
            current_stroke_m=2.44,
        )
        assert recommendation.action == "HOLD"
        assert recommendation.spm == pytest.approx(9.0)

    def test_pump_tagging_and_gas_are_treated(self) -> None:
        controller = SRPController()
        for label, expected in (
            (CardLabel.PUMP_TAGGING, "REDUCE_SPM"),
            (CardLabel.GAS_INTERFERENCE, "REDUCE_SPM"),
            (CardLabel.UNANCHORED_TUBING, "INSPECT_TUBING"),
        ):
            recommendation = controller.recommend(
                label=label,
                confidence=0.9,
                inflow_m3_per_day=5.0,
                current_spm=9.0,
                current_stroke_m=2.44,
            )
            assert recommendation.action == expected

    def test_recommendations_respect_the_actuator_limits(self) -> None:
        controller = SRPController()
        for label in CARD_LABELS:
            recommendation = controller.recommend(
                label=label,
                confidence=0.9,
                inflow_m3_per_day=1e-6,
                current_spm=THRESHOLDS.max_spm,
                current_stroke_m=2.44,
                rod_float_risk_index=1.0,
            )
            assert THRESHOLDS.min_spm <= recommendation.spm <= THRESHOLDS.max_spm

    def test_spm_for_inflow_is_clamped(self) -> None:
        controller = SRPController()
        assert controller.spm_for_inflow(0.0, 2.44) == THRESHOLDS.min_spm
        assert controller.spm_for_inflow(1e6, 2.44) == THRESHOLDS.max_spm

    def test_stroke_for_inflow_inverts_the_capacity(self) -> None:
        controller = SRPController()
        for spm in (4.0, 9.0, 12.0):
            for rate in (0.8, 2.0, 4.0):
                stroke = controller.stroke_for_inflow(rate, spm)
                assert controller.delivered_rate_m3_per_day(spm, stroke) == pytest.approx(
                    rate, rel=1e-9
                )

    def test_rejects_invalid_configuration(self) -> None:
        with pytest.raises(ValueError):
            SRPController(plunger_area_m2=0.0)
        with pytest.raises(ValueError):
            SRPController(volumetric_efficiency=0.0)
        with pytest.raises(ValueError):
            SRPController(volumetric_efficiency=1.5)
        with pytest.raises(ValueError):
            SRPController(min_spm=10.0, max_spm=2.0)
        with pytest.raises(ValueError):
            SRPController(rod_float_factor=1.5)
        controller = SRPController()
        with pytest.raises(ValueError):
            controller.recommend(CardLabel.NORMAL_FULL_BARREL, 1.0, 1.0, 0.0, 2.44)
        with pytest.raises(ValueError):
            controller.recommend(CardLabel.NORMAL_FULL_BARREL, 1.0, 1.0, 9.0, 0.0)
        with pytest.raises(ValueError):
            controller.delivered_rate_m3_per_day(9.0, -1.0)
        with pytest.raises(ValueError):
            controller.stroke_for_inflow(1.0, 0.0)
        with pytest.raises(ValueError):
            controller.maximum_spm(0.0)
