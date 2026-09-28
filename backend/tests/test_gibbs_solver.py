"""Verification of the Gibbs damped wave equation solver.

The headline requirement of the specification is that an analytical sinusoidal
input emerges at the pump delayed by exactly the wave travel time
:math:`L/a`.  That is tested here against the causal form of the solution -- the
pump must stay at rest until the front arrives -- which is a sharper statement
than a phase fit, because it does not depend on a fitting window.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.app.core.config import PUMPING_UNIT, WELL
from backend.app.ai.dyno_classifier import (
    CardLabel,
    analytic_diagnosis,
    card_features,
)
from backend.app.physics.gibbs_solver import (
    GibbsSolver,
    default_column_weight_n,
    fillage_shape,
    PumpCard,
    RodString,
    baghewala_rod_string,
    damping_coefficient,
    solve_tridiagonal,
)
from backend.app.physics.rheology import baghewala_crude


@pytest.fixture(scope="module")
def rod() -> RodString:
    return baghewala_rod_string()


def _sinusoid(amplitude: float, frequency_hz: float, samples: int, duration_s: float):
    t = np.linspace(0.0, duration_s, samples)
    return amplitude * np.sin(2.0 * math.pi * frequency_hz * t), t


class TestRodString:
    def test_acoustic_velocity_matches_steel(self, rod: RodString) -> None:
        # sqrt(E/rho) for steel is 5000-5100 m/s.
        assert 4900.0 < rod.acoustic_velocity_m_s < 5200.0

    def test_travel_time_is_length_over_velocity(self, rod: RodString) -> None:
        assert rod.travel_time_s == pytest.approx(
            rod.length_m / rod.acoustic_velocity_m_s, rel=1e-12
        )
        assert rod.travel_time_s == pytest.approx(0.2, abs=0.02)

    def test_weight_matches_rod_inventory(self, rod: RodString) -> None:
        expected = rod.density_kg_m3 * rod.area_m2 * rod.length_m * 9.80665
        assert rod.weight_n == pytest.approx(expected, rel=1e-12)
        assert 15_000.0 < rod.weight_n < 40_000.0

    def test_stiffness_is_modulus_times_area(self, rod: RodString) -> None:
        assert rod.axial_stiffness_n_per_m == pytest.approx(
            rod.elastic_modulus_pa * rod.area_m2, rel=1e-12
        )

    def test_static_deflection_is_monotonic(self, rod: RodString) -> None:
        profile = rod.static_deflection(rod.weight_n, 41)
        assert len(profile) == 41
        assert profile[0] == pytest.approx(0.0, abs=1e-15)
        assert np.all(np.diff(profile) > 0.0)
        # A rod hanging under its own weight stretches most at the surface.
        assert profile[-1] == pytest.approx(0.19, abs=0.03)

    def test_rejects_invalid_geometry(self) -> None:
        with pytest.raises(ValueError):
            RodString(length_m=0.0)
        with pytest.raises(ValueError):
            RodString(area_m2=-1.0)
        with pytest.raises(ValueError):
            RodString(elastic_modulus_pa=0.0)
        with pytest.raises(ValueError):
            RodString(density_kg_m3=0.0)


class TestDamping:
    def test_damping_increases_with_viscosity(self) -> None:
        low = damping_coefficient(200.0, 0.03)
        high = damping_coefficient(46.0, 12.5)
        assert high > low >= 0.0

    def test_damping_matches_the_declared_model(self) -> None:
        crude = baghewala_crude()
        viscosity = crude.dynamic_viscosity_pa_s(80.0)
        reference = crude.dynamic_viscosity_pa_s(46.0)
        expected = 0.35 * (1.0 + 0.25 * (viscosity / reference - 1.0))
        assert damping_coefficient(80.0, viscosity) == pytest.approx(expected, rel=1e-12)

    def test_damping_is_never_negative(self) -> None:
        assert damping_coefficient(400.0, 0.0) == pytest.approx(0.2625, rel=1e-9)

    def test_rejects_invalid_input(self) -> None:
        with pytest.raises(ValueError):
            damping_coefficient(46.0, -1.0)
        with pytest.raises(ValueError):
            damping_coefficient(46.0, 1.0, base_rate_s=-1.0)
        with pytest.raises(ValueError):
            damping_coefficient(46.0, 1.0, viscosity_sensitivity=-1.0)


class TestTridiagonalSolver:
    def test_solves_a_known_system(self) -> None:
        # 3x3 tridiagonal system with solution x = (1, 2, 3):
        #   row 1: 4*1 + 1*2 = 6
        #   row 2: 1*1 + 4*2 + 1*3 = 12
        #   row 3: 1*2 + 4*3     = 14
        lower = np.array([1.0, 1.0])
        diagonal = np.array([4.0, 4.0, 4.0])
        upper = np.array([1.0, 1.0])
        rhs = np.array([6.0, 12.0, 14.0])
        solution = solve_tridiagonal(lower, diagonal, upper, rhs)
        assert solution == pytest.approx([1.0, 2.0, 3.0], abs=1e-12)

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError):
            solve_tridiagonal(
                np.ones(2), np.ones(3), np.ones(2), np.ones(2)
            )
        with pytest.raises(ValueError):
            solve_tridiagonal(
                np.ones(1), np.ones(3), np.ones(1), np.ones(3)
            )

    def test_detects_a_singular_matrix(self) -> None:
        with pytest.raises(np.linalg.LinAlgError):
            solve_tridiagonal(np.ones(2), np.zeros(3), np.ones(2), np.ones(3))


class TestStability:
    def test_courant_number_never_exceeds_one(self, rod: RodString) -> None:
        for segments in (8, 20, 40, 80, 160):
            solver = GibbsSolver(rod, segments=segments, damping_rate_s=0.0)
            assert solver.stability_number() <= 1.0 + 1e-12

    def test_damping_shrinks_the_stable_timestep(self, rod: RodString) -> None:
        undamped = GibbsSolver(rod, segments=40, damping_rate_s=0.0)
        damped = GibbsSolver(rod, segments=40, damping_rate_s=8.0)
        assert damped.stability_number() <= undamped.stability_number() + 1e-12

    def test_rejects_an_unstable_timestep(self, rod: RodString) -> None:
        solver = GibbsSolver(rod, segments=8, damping_rate_s=0.0)
        with pytest.raises(ValueError):
            solver.solve(
                surface_position_m=np.zeros(64),
                surface_load_n=np.zeros(64),
                duration_s=1.0,
                dt=10.0,
            )

    def test_rejects_invalid_configuration(self, rod: RodString) -> None:
        with pytest.raises(ValueError):
            GibbsSolver(rod, segments=2)
        with pytest.raises(ValueError):
            GibbsSolver(rod, segments=40, damping_rate_s=-1.0)


class TestTravelTime:
    """The specification's phase-shift requirement, tested causally."""

    def test_pump_stays_at_rest_until_the_wave_arrives(self, rod: RodString) -> None:
        solver = GibbsSolver(rod, segments=160, damping_rate_s=0.0)
        travel = rod.travel_time_s
        amplitude = 1.0e-3
        frequency = 4.0
        duration = 1.6 * travel
        position, _ = _sinusoid(amplitude, frequency, 2000, duration)
        card = solver.transport(
            surface_position_m=position,
            surface_load_n=np.full(len(position), rod.weight_n),
            duration_s=duration,
        )
        # Causality: nothing reaches the pump before the front has travelled L.
        before = card.time_s < 0.9 * travel
        assert np.max(np.abs(card.position_m[before])) < 1e-9

    def test_arrival_time_converges_to_the_travel_time(self, rod: RodString) -> None:
        """The arrival error must fall as the grid is refined, and land on L/a.

        The scheme is second-order accurate in space, so halving the cell size
        roughly quarters the arrival-time error.
        """
        travel = rod.travel_time_s
        errors = []
        for segments in (40, 80, 160, 320):
            solver = GibbsSolver(rod, segments=segments, damping_rate_s=0.0)
            duration = 1.6 * travel
            position, _ = _sinusoid(1.0e-3, 4.0, 3000, duration)
            card = solver.transport(
                surface_position_m=position,
                surface_load_n=np.full(len(position), rod.weight_n),
                duration_s=duration,
            )
            amplitude = float(np.max(np.abs(card.position_m)))
            assert amplitude == pytest.approx(2.0e-3, rel=0.05), (
                "the downhole amplitude must match the surface amplitude for an "
                "undamped, lossless semi-infinite string"
            )
            moved = np.flatnonzero(np.abs(card.position_m) > 0.01 * amplitude)
            assert moved.size > 0
            errors.append(abs(float(card.time_s[moved[0]]) - travel) / travel)
        assert errors[0] < 0.05
        assert errors[-1] < 0.01
        # Monotone refinement.
        for earlier, later in zip(errors, errors[1:]):
            assert later < earlier

    def test_downhole_signal_is_the_delayed_surface_signal(
        self, rod: RodString
    ) -> None:
        """After arrival, u(L, t) equals u(0, t - L/a)."""
        travel = rod.travel_time_s
        solver = GibbsSolver(rod, segments=320, damping_rate_s=0.0)
        duration = 1.9 * travel
        position, _ = _sinusoid(1.0e-3, 4.0, 6000, duration)
        card = solver.transport(
            surface_position_m=position,
            surface_load_n=np.full(len(position), rod.weight_n),
            duration_s=duration,
        )
        # Fit the lag that maximises the correlation with the surface signal.
        step = float(card.time_s[1] - card.time_s[0])
        best_lag, best_score = 0, -np.inf
        for candidate in range(int(0.6 * travel / step), int(1.4 * travel / step)):
            reference = np.roll(
                1.0e-3 * np.sin(2.0 * math.pi * 4.0 * (card.time_s - candidate * step)),
                0,
            )
            mask = card.time_s > 1.3 * travel
            a = card.position_m[mask] - card.position_m[mask].mean()
            b = reference[mask] - reference[mask].mean()
            if np.std(b) < 1e-12:
                continue
            score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            if score > best_score:
                best_lag, best_score = candidate * step, score
        assert best_score > 0.99
        assert abs(best_lag - travel) / travel < 0.03

    def test_travel_time_scales_with_rod_length(self) -> None:
        short = RodString(length_m=500.0)
        long = RodString(length_m=1500.0)
        assert long.travel_time_s == pytest.approx(3.0 * short.travel_time_s, rel=1e-9)


class TestDampingEffect:
    def test_damping_attenuates_the_wave(self, rod: RodString) -> None:
        travel = rod.travel_time_s
        duration = 1.5 * travel
        position, _ = _sinusoid(1.0e-3, 4.0, 3000, duration)
        undamped = GibbsSolver(rod, segments=160, damping_rate_s=0.0).transport(
            position, np.full(len(position), rod.weight_n), duration_s=duration
        )
        damped = GibbsSolver(rod, segments=160, damping_rate_s=6.0).transport(
            position, np.full(len(position), rod.weight_n), duration_s=duration
        )
        mask = undamped.time_s > 1.2 * travel
        undamped_span = np.ptp(undamped.position_m[mask])
        damped_span = np.ptp(damped.position_m[mask])
        assert damped_span < undamped_span

    def test_zero_damping_is_lossless(self, rod: RodString) -> None:
        travel = rod.travel_time_s
        duration = 1.4 * travel
        position, _ = _sinusoid(1.0e-3, 4.0, 3000, duration)
        card = GibbsSolver(rod, segments=160, damping_rate_s=0.0).transport(
            position, np.full(len(position), rod.weight_n), duration_s=duration
        )
        # A lossless string carries the wave without attenuation, so the pump
        # amplitude matches the surface amplitude to within the numerical
        # dispersion of the grid.  The comparison spans the whole card, because
        # the simulation is only 1.4 travel times long and the post-arrival
        # window is too short to contain a full peak on its own.
        assert np.ptp(card.position_m) == pytest.approx(2.0e-3, rel=0.02)


class TestCardProperties:
    def test_card_reports_its_geometry(self, rod: RodString) -> None:
        solver = GibbsSolver(rod, segments=40, damping_rate_s=0.2)
        duration = 60.0 / PUMPING_UNIT.nominal_spm
        position, _ = _sinusoid(0.5 * PUMPING_UNIT.nominal_stroke_m, 0.15, 400, duration)
        card = solver.solve(
            position, np.full(len(position), rod.weight_n), duration_s=duration
        )
        assert isinstance(card, PumpCard)
        # The reconstructed plunger stroke is the surface stroke modified by the
        # rod's dynamic response, so it is bounded by the surface stroke rather
        # than equal to it: a polished rod driven near a rod-string mode can
        # amplify, and the plunger travel is the rod travel less its stretch.
        assert 0.3 * PUMPING_UNIT.nominal_stroke_m < card.stroke_m < 2.0 * PUMPING_UNIT.nominal_stroke_m
        assert card.load_span_n >= 0.0
        assert card.max_load_n >= card.min_load_n
        assert card.metadata["travel_time_s"] == pytest.approx(rod.travel_time_s, rel=1e-12)
        assert card.metadata["acoustic_velocity_m_s"] == pytest.approx(
            rod.acoustic_velocity_m_s, rel=1e-12
        )

    def test_card_serialises_to_json_ready_types(self, rod: RodString) -> None:
        import json

        solver = GibbsSolver(rod, segments=20, damping_rate_s=0.1)
        position, _ = _sinusoid(1.2, PUMPING_UNIT.nominal_spm / 60.0, 200, 6.0)
        card = solver.solve(
            position, np.full(len(position), rod.weight_n), duration_s=6.0
        )
        payload = card.as_dict()
        # Round-trips through JSON without a custom encoder.
        text = json.dumps(payload)
        restored = json.loads(text)
        assert restored["stroke_m"] == pytest.approx(card.stroke_m, rel=1e-12)
        assert len(restored["position_m"]) == len(card.position_m)
        assert isinstance(restored["segments"], int)

    def test_full_barrel_card_is_a_parallelogram(self, rod: RodString) -> None:
        """A full barrel must reconstruct as the classic straight-sided card."""
        duration = 60.0 / PUMPING_UNIT.nominal_spm
        frequency = PUMPING_UNIT.nominal_spm / 60.0
        t = np.linspace(0.0, duration, 240)
        x = 0.5 * (1.0 - np.cos(2.0 * math.pi * frequency * t))
        position = PUMPING_UNIT.nominal_stroke_m * x
        column = default_column_weight_n()
        card = GibbsSolver(rod, segments=40, damping_rate_s=0.0).solve(
            position, rod.weight_n + column * x, duration_s=duration, fillage=1.0
        )
        assert card.stroke_m == pytest.approx(PUMPING_UNIT.nominal_stroke_m, rel=0.02)
        assert card.load_span_n == pytest.approx(column, rel=0.02)
        assert card.min_load_n == pytest.approx(0.0, abs=column * 0.02)
        # The card is monotone in position, which is the definition of a full
        # barrel, so a healthy pump never shows a load reversal.
        assert card_features(card.position_m, card.load_n)["upstroke_reversal"] < 0.05

    def test_reduced_fillage_reconstructs_fluid_pound(self, rod: RodString) -> None:
        """A starved pump must reconstruct with the fluid-pound reversal."""
        duration = 60.0 / PUMPING_UNIT.nominal_spm
        frequency = PUMPING_UNIT.nominal_spm / 60.0
        t = np.linspace(0.0, duration, 240)
        x = 0.5 * (1.0 - np.cos(2.0 * math.pi * frequency * t))
        position = PUMPING_UNIT.nominal_stroke_m * x
        column = default_column_weight_n()
        solver = GibbsSolver(rod, segments=40, damping_rate_s=0.0)
        for fillage in (0.7, 0.5, 0.35):
            card = solver.solve(
                position,
                rod.weight_n + column * x,
                duration_s=duration,
                fillage=fillage,
            )
            features = card_features(card.position_m, card.load_n)
            assert features["upstroke_reversal"] > 0.1, f"fillage {fillage}"
            assert analytic_diagnosis(features) is CardLabel.FLUID_POUND

    def test_transmission_constants_are_reported(self, rod: RodString) -> None:
        duration = 60.0 / PUMPING_UNIT.nominal_spm
        position, _ = _sinusoid(
            0.5 * PUMPING_UNIT.nominal_stroke_m,
            PUMPING_UNIT.nominal_spm / 60.0,
            240,
            duration,
        )
        for damping in (0.0, 1.5):
            card = GibbsSolver(rod, segments=40, damping_rate_s=damping).solve(
                position, np.full(len(position), rod.weight_n), duration_s=duration
            )
            assert card.arrival_time_s == pytest.approx(rod.travel_time_s, rel=1e-12)
            assert card.transmission_gain == pytest.approx(
                math.exp(-damping * rod.travel_time_s), rel=1e-12
            )
            assert card.metadata["method"] == "gibbs transmission"

    def test_rephases_the_card_onto_bottom_dead_centre(self, rod: RodString) -> None:
        """The reported trace must start at the bottom dead centre."""
        duration = 60.0 / PUMPING_UNIT.nominal_spm
        frequency = PUMPING_UNIT.nominal_spm / 60.0
        t = np.linspace(0.0, duration, 240)
        x = 0.5 * (1.0 - np.cos(2.0 * math.pi * frequency * t))
        position = PUMPING_UNIT.nominal_stroke_m * x
        column = default_column_weight_n()
        card = GibbsSolver(rod, segments=40, damping_rate_s=0.0).solve(
            position, rod.weight_n + column * x, duration_s=duration
        )
        assert card.position_m[0] == pytest.approx(0.0, abs=1e-12)
        assert card.time_s[0] == pytest.approx(0.0, abs=1e-12)
        assert card.time_s[-1] < duration
        # The load minimum sits at the bottom dead centre for a full barrel.
        assert int(np.argmin(card.load_n)) < 2

    def test_explicit_column_weight_overrides_the_surface_reading(
        self, rod: RodString
    ) -> None:
        duration = 60.0 / PUMPING_UNIT.nominal_spm
        frequency = PUMPING_UNIT.nominal_spm / 60.0
        t = np.linspace(0.0, duration, 240)
        x = 0.5 * (1.0 - np.cos(2.0 * math.pi * frequency * t))
        card = GibbsSolver(rod, segments=40, damping_rate_s=0.0).solve(
            PUMPING_UNIT.nominal_stroke_m * x,
            np.full(240, rod.weight_n),
            duration_s=duration,
            fluid_load_n=9000.0,
        )
        assert card.load_span_n == pytest.approx(9000.0, rel=0.02)

    def test_zero_surface_fluid_load_falls_back_to_the_column(self, rod: RodString) -> None:
        """A load cell reading only the rod weight still yields a real card."""
        duration = 60.0 / PUMPING_UNIT.nominal_spm
        frequency = PUMPING_UNIT.nominal_spm / 60.0
        t = np.linspace(0.0, duration, 240)
        x = 0.5 * (1.0 - np.cos(2.0 * math.pi * frequency * t))
        card = GibbsSolver(rod, segments=40, damping_rate_s=0.0).solve(
            PUMPING_UNIT.nominal_stroke_m * x,
            np.full(240, rod.weight_n),
            duration_s=duration,
        )
        assert card.load_span_n == pytest.approx(default_column_weight_n(), rel=0.02)

    def test_fillage_shape_is_a_valid_characteristic(self) -> None:
        assert fillage_shape(0.0, 1.0) == 0.0
        assert fillage_shape(1.0, 1.0) == pytest.approx(1.0)
        assert fillage_shape(0.0, 0.5) == 0.0
        assert fillage_shape(0.5, 0.5) == pytest.approx(1.0)
        # A full barrel is monotone; a starved one collapses after the fill.
        full = [fillage_shape(x / 20.0, 1.0) for x in range(21)]
        assert all(b >= a for a, b in zip(full, full[1:]))
        starved = [fillage_shape(x / 20.0, 0.4) for x in range(21)]
        assert max(starved) == pytest.approx(1.0)
        assert starved[-1] < 0.2
        assert all(0.0 <= v <= 1.0 for v in starved)

    def test_default_column_weight_matches_the_hydrostatic_head(self) -> None:
        expected = (
            900.0
            * 9.80665
            * math.pi
            * (0.5 * WELL.tubing_id_m) ** 2
            * WELL.pump_depth_m
        )
        assert default_column_weight_n() == pytest.approx(expected, rel=1e-12)
        # Roughly 22-45 kN, the scale of a Baghewala card.
        assert 20_000.0 < default_column_weight_n() < 45_000.0

    def test_rejects_invalid_card_data(self, rod: RodString) -> None:
        solver = GibbsSolver(rod, segments=20, damping_rate_s=0.0)
        with pytest.raises(ValueError):
            solver.solve([0.0, 1.0], [0.0, 1.0])
        with pytest.raises(ValueError):
            solver.solve([0.0, 1.0, 2.0], [0.0, 1.0])
        with pytest.raises(ValueError):
            solver.solve(
                [0.0] * 8, [float("nan")] * 8, duration_s=1.0
            )
        with pytest.raises(ValueError):
            solver.solve([0.0] * 8, [0.0] * 8, duration_s=0.0)
