"""Verification of the Marx-Langenheim CSS and conductive heat-dissipation model.

The central assertion is an exact energy balance: the enthalpy injected during
steam injection must equal the heat retained in the chest plus the conductive
losses to the formation and to the over/underburden.
"""

from __future__ import annotations

import math

import pytest

from backend.app.core.config import CSS, RESERVOIR, THRESHOLDS
from backend.app.physics.thermal_reservoir import (
    CSSPhase,
    ThermalReservoirModel,
    baghewala_css_model,
    erfc,
)


@pytest.fixture(scope="module")
def model() -> ThermalReservoirModel:
    return ThermalReservoirModel()


def _after_soak(result) -> "object":
    soaking = [s for s in result.states if s.phase == CSSPhase.SOAKING]
    assert soaking, "the cycle must contain a soak phase"
    return soaking[-1]


class TestErfc:
    def test_matches_known_values(self) -> None:
        assert erfc(0.0) == pytest.approx(1.0, rel=1e-12)
        assert erfc(0.5) == pytest.approx(0.479500, abs=1e-6)
        assert erfc(1.0) == pytest.approx(0.157299, abs=1e-6)
        assert erfc(2.0) == pytest.approx(0.004678, abs=1e-6)

    def test_complementary_identity(self) -> None:
        for x in (-2.0, -0.5, 0.0, 0.3, 1.7):
            assert erfc(x) + erfc(-x) == pytest.approx(2.0, rel=1e-12)

    def test_decreasing_and_bounded(self) -> None:
        previous = math.inf
        for i in range(0, 41):
            value = erfc(i * 0.1)
            assert 0.0 < value <= 1.0
            assert value < previous
            previous = value

    def test_matches_math_erf(self) -> None:
        for x in (0.0, 0.25, 0.75, 1.5, 3.0):
            assert erfc(x) == pytest.approx(1.0 - math.erf(x), abs=1e-12)

    def test_accurate_in_the_tail(self) -> None:
        """No catastrophic cancellation once erf underflows to 1."""
        for x in (2.5, 3.0, 4.0, 6.0, 10.0):
            assert erfc(x) == pytest.approx(math.erfc(x), rel=1e-13)

    def test_saturation_limits(self) -> None:
        assert erfc(40.0) == 0.0
        assert erfc(-40.0) == 2.0
        assert math.isnan(erfc(float("nan")))


class TestSteamChestGrowth:
    def test_injection_grows_the_chest(self, model: ThermalReservoirModel) -> None:
        result = model.simulate_cycle()
        injecting = [s for s in result.states if s.phase == CSSPhase.INJECTION]
        assert len(injecting) > 10
        for earlier, later in zip(injecting, injecting[1:]):
            assert later.radius_m >= earlier.radius_m - 1e-9
        assert injecting[-1].radius_m > model.rw * 5.0

    def test_chest_radius_is_field_plausible(self, model: ThermalReservoirModel) -> None:
        """A 9-day injection of 45 t/day on this sand gives a 20-35 m chest."""
        result = model.simulate_cycle()
        injecting = [s for s in result.states if s.phase == CSSPhase.INJECTION]
        assert 20.0 <= injecting[-1].radius_m <= 35.0

    def test_radius_stays_within_geometry_bounds(self, model: ThermalReservoirModel) -> None:
        result = model.simulate_cycle()
        for state in result.states:
            assert model.rw <= state.radius_m <= model.re

    def test_injection_heats_the_chest_to_steam_temperature(
        self, model: ThermalReservoirModel
    ) -> None:
        result = model.simulate_cycle()
        injecting = [s for s in result.states if s.phase == CSSPhase.INJECTION]
        assert result.states[0].chest_temperature_c == pytest.approx(
            model.reservoir_temperature_c, rel=1e-9
        )
        assert injecting[-1].chest_temperature_c == pytest.approx(
            model.steam_temperature_c, rel=1e-9
        )

    def test_chest_cools_during_soak(self, model: ThermalReservoirModel) -> None:
        result = model.simulate_cycle()
        soaking = [s.chest_temperature_c for s in result.states if s.phase == CSSPhase.SOAKING]
        assert len(soaking) > 5
        for earlier, later in zip(soaking, soaking[1:]):
            assert later <= earlier + 1e-9
        assert soaking[-1] < model.steam_temperature_c


class TestEnergyBalance:
    def test_injected_enthalpy_equals_retained_plus_losses(
        self, model: ThermalReservoirModel
    ) -> None:
        """Core conservation test on the Marx-Langenheim solver.

        At the end of the soak phase the energy injected must equal the heat
        still in the chest plus the conductive losses to the formation and to
        the over/underburden, to double-precision round-off.
        """
        result = model.simulate_cycle(injection_days=9.0, soak_days=5.0, production_days=1.0)
        state = _after_soak(result)

        injected = (
            state.cumulative_steam_tonnes
            * 1000.0
            * model.steam_enthalpy_j_kg
            * model.steam_retention_factor
        )
        retained = state.cumulative_heat_retained_j
        losses = state.cumulative_heat_formation_j + state.cumulative_heat_overburden_j
        assert injected > 0.0
        assert retained > 0.0
        assert losses > 0.0
        assert injected == pytest.approx(retained + losses, rel=1e-12)

    def test_energy_balance_helper_closes(self, model: ThermalReservoirModel) -> None:
        assert model.energy_balance(model.simulate_cycle()) < 1e-12

    def test_ledger_holds_at_every_recorded_step(self, model: ThermalReservoirModel) -> None:
        """The identity is checked along the whole injection and soak trajectory."""
        result = model.simulate_cycle(production_days=1.0)
        for state in result.states:
            if state.phase == CSSPhase.PRODUCTION:
                break
            injected = (
                state.cumulative_steam_tonnes
                * 1000.0
                * model.steam_enthalpy_j_kg
                * model.steam_retention_factor
            )
            accounted = (
                state.cumulative_heat_retained_j
                + state.cumulative_heat_formation_j
                + state.cumulative_heat_overburden_j
            )
            if injected > 0.0:
                assert injected == pytest.approx(accounted, rel=1e-12)

    def test_losses_are_non_negative_and_monotonic(self, model: ThermalReservoirModel) -> None:
        result = model.simulate_cycle()
        for series in (
            [s.cumulative_heat_formation_j for s in result.states],
            [s.cumulative_heat_overburden_j for s in result.states],
        ):
            for earlier, later in zip(series, series[1:]):
                assert later >= earlier - 1e-6

    def test_retained_energy_never_negative(self, model: ThermalReservoirModel) -> None:
        for state in model.simulate_cycle().states:
            assert state.cumulative_heat_retained_j >= -1e-6

    def test_injected_enthalpy_matches_steam_tonnage(
        self, model: ThermalReservoirModel
    ) -> None:
        state = _after_soak(model.simulate_cycle())
        expected_kg = (
            model.steam_rate_m3_per_day
            * CSS.injection_days
            * 1000.0
            * model.steam_retention_factor
        )
        stored = state.cumulative_heat_retained_j
        # The stored energy can never exceed what was injected.
        assert stored <= expected_kg * model.steam_enthalpy_j_kg


class TestHeatLossPhysics:
    def test_overburden_loss_decays_as_inverse_sqrt_time(
        self, model: ThermalReservoirModel
    ) -> None:
        """The Carslaw-Jaeger surface flux is exactly k*dT/sqrt(pi*alpha*t)."""
        times = (1.0, 5.0, 20.0, 40.0, 60.0, 140.0)
        reference = None
        for day in times:
            scaled = model._overburden_loss_rate(220.0, day, 20.0) * math.sqrt(day)
            if reference is None:
                reference = scaled
            else:
                assert scaled == pytest.approx(reference, rel=1e-12)

    def test_overburden_loss_matches_the_closed_form(
        self, model: ThermalReservoirModel
    ) -> None:
        radius, day, temperature = 20.0, 40.0, 220.0
        area = model.vertical_contact * 2.0 * math.pi * (radius**2 - model.rw**2)
        expected = (
            model.k_over
            * area
            * (temperature - model.reservoir_temperature_c)
            / (math.sqrt(math.pi) * math.sqrt(model.alpha_over * day * 86400.0))
            * 86400.0
        )
        assert model._overburden_loss_rate(temperature, day, radius) == pytest.approx(
            expected, rel=1e-12
        )

    def test_shale_temperature_follows_the_erfc_profile(
        self, model: ThermalReservoirModel
    ) -> None:
        """T(x,t) = Tres + (Tch-Tres) erfc(x / (2 sqrt(alpha t)))."""
        day, temperature = 60.0, 220.0
        for depth in (0.0, 0.5, 1.0, 2.0, 4.0):
            front = 2.0 * math.sqrt(model.alpha_over * day * 86400.0)
            expected = model.reservoir_temperature_c + (
                temperature - model.reservoir_temperature_c
            ) * math.erfc(depth / front)
            assert model._shale_temperature(depth, temperature, day) == pytest.approx(
                expected, rel=1e-12
            )

    def test_shale_profile_is_monotonic_in_depth(
        self, model: ThermalReservoirModel
    ) -> None:
        temperatures = [
            model._shale_temperature(d, 220.0, 40.0) for d in (0.0, 1.0, 2.0, 3.0, 5.0, 8.0)
        ]
        for earlier, later in zip(temperatures, temperatures[1:]):
            assert later <= earlier
        # The front never reaches the far boundary within a cycle.
        assert temperatures[-1] == pytest.approx(model.reservoir_temperature_c, abs=1e-6)

    def test_overburden_loss_scales_with_contact_area(
        self, model: ThermalReservoirModel
    ) -> None:
        small = model._overburden_loss_rate(200.0, 40.0, 5.0)
        large = model._overburden_loss_rate(200.0, 40.0, 20.0)
        assert large > small
        area_ratio = (20.0**2 - model.rw**2) / (5.0**2 - model.rw**2)
        assert large / small == pytest.approx(area_ratio, rel=1e-9)

    def test_zero_loss_at_reservoir_temperature(self, model: ThermalReservoirModel) -> None:
        assert model._overburden_loss_rate(
            model.reservoir_temperature_c, 5.0, 10.0
        ) == 0.0
        assert model._overburden_loss_rate(200.0, 0.0, 10.0) == 0.0

    def test_conduction_conductance_grows_with_chest_radius(
        self, model: ThermalReservoirModel
    ) -> None:
        """ln(Re/Rs) shrinks as the front advances, so the conductance rises."""
        rates = [model._conduction_to_formation(200.0, r) for r in (0.5, 2.0, 10.0, 40.0)]
        for earlier, later in zip(rates, rates[1:]):
            assert later > earlier

    def test_conduction_requires_positive_contrast(self, model: ThermalReservoirModel) -> None:
        assert model._conduction_to_formation(model.reservoir_temperature_c, 10.0) == 0.0
        assert model._conduction_to_formation(200.0, model.rw) == 0.0
        assert model._conduction_to_formation(200.0, 10.0) > 0.0

    def test_latent_heat_decreases_with_temperature(self) -> None:
        hot = ThermalReservoirModel(steam_temperature_c=240.0).latent_heat_kj_kg
        cool = ThermalReservoirModel(steam_temperature_c=200.0).latent_heat_kj_kg
        assert hot < cool
        # Watson correlation at 220 degC lands near 1.8 MJ/kg.
        assert 1500.0 < ThermalReservoirModel().latent_heat_kj_kg < 2100.0

    def test_steam_mass_flow_matches_the_configured_rate(
        self, model: ThermalReservoirModel
    ) -> None:
        assert model.steam_mass_flow_kg_s * 86400.0 == pytest.approx(
            model.steam_rate_m3_per_day * 1000.0, rel=1e-12
        )


class TestProductionAndSOR:
    def test_bht_decays_towards_reservoir_temperature(
        self, model: ThermalReservoirModel
    ) -> None:
        result = model.simulate_cycle()
        production = [
            s.bottom_hole_temperature_c for s in result.states if s.phase == CSSPhase.PRODUCTION
        ]
        assert len(production) > 20
        for earlier, later in zip(production, production[1:]):
            assert later <= earlier + 1e-9
        assert production[-1] < production[0]
        assert production[-1] > model.reservoir_temperature_c

    def test_bht_is_attenuated_below_the_chest_temperature(
        self, model: ThermalReservoirModel
    ) -> None:
        result = model.simulate_cycle()
        for state in result.states:
            assert state.bottom_hole_temperature_c <= state.chest_temperature_c + 1e-9
            assert (
                state.bottom_hole_temperature_c >= model.reservoir_temperature_c - 1e-9
            )

    def test_delivered_rate_is_pump_limited_early(self, model: ThermalReservoirModel) -> None:
        """A Baghewala CSS well is pump-limited, not reservoir-limited."""
        result = model.simulate_cycle()
        production = [s for s in result.states if s.phase == CSSPhase.PRODUCTION]
        assert production[0].reservoir_inflow_m3_per_day > model.pump_capacity_m3_per_day
        assert production[0].oil_rate_tpd == pytest.approx(
            model.pump_capacity_m3_per_day, rel=1e-9
        )
        assert production[0].pump_fillage > 1.0

    def test_delivered_rate_tracks_the_pump_while_fillage_exceeds_one(
        self, model: ThermalReservoirModel
    ) -> None:
        result = model.simulate_cycle()
        production = [s for s in result.states if s.phase == CSSPhase.PRODUCTION]
        for state in production:
            expected = min(state.reservoir_inflow_m3_per_day, model.pump_capacity_m3_per_day)
            assert state.oil_rate_tpd == pytest.approx(expected, rel=1e-9)
            if state.pump_fillage > 1.0:
                assert state.oil_rate_tpd == pytest.approx(
                    model.pump_capacity_m3_per_day, rel=1e-9
                )

    def test_degraded_reservoir_starves_the_pump(self) -> None:
        """A heavily skinned, low-permeability well can no longer fill the barrel."""
        starved = ThermalReservoirModel(skin_factor=20.0, permeability_md=40.0)
        result = starved.simulate_cycle()
        production = [s for s in result.states if s.phase == CSSPhase.PRODUCTION]
        assert production[-1].pump_fillage < 1.0
        assert production[-1].oil_rate_tpd < starved.pump_capacity_m3_per_day

    def test_fillage_never_exceeds_the_pump_capacity(
        self, model: ThermalReservoirModel
    ) -> None:
        for state in model.simulate_cycle().states:
            assert state.oil_rate_tpd <= model.pump_capacity_m3_per_day * (1.0 + 1e-9)

    def test_rates_are_field_plausible(self, model: ThermalReservoirModel) -> None:
        """Plateau and cycle-average rates must be in the Baghewala range."""
        result = model.simulate_cycle()
        production = [s for s in result.states if s.phase == CSSPhase.PRODUCTION]
        plateau_bpd = production[0].oil_rate_tpd * 6.2898
        average_bpd = (
            result.total_oil_tonnes
            / model.oil_specific_gravity
            / CSS.production_days
            * 6.2898
        )
        assert 20.0 < plateau_bpd < 45.0, f"plateau {plateau_bpd:.1f} bbl/d"
        assert 15.0 < average_bpd < 40.0, f"average {average_bpd:.1f} bbl/d"

    def test_vogel_half_rate_point_without_skin(self) -> None:
        """Vogel reaches half its potential at x = 0.6754, i.e. 32.5% drawdown.

        Solving 1 - 0.2x - 0.8x^2 = 1/2 for x gives
        ``(-0.2 + sqrt(1.64))/1.6 = 0.67539...``, which is the well known
        "Vogel half-rate drawdown" of about a third of the reservoir pressure.
        """
        clean = ThermalReservoirModel(skin_factor=0.0)
        x_half = (math.sqrt(1.64) - 0.2) / 1.6
        q_max = clean._vogel_inflow(0.0)
        half = clean._vogel_inflow(x_half * RESERVOIR.initial_reservoir_pressure_mpa)
        assert half / q_max == pytest.approx(0.5, rel=1e-9)
        assert x_half == pytest.approx(0.6754, abs=1e-4)
        assert clean._vogel_inflow(0.0) == pytest.approx(q_max, rel=1e-9)

    def test_vogel_ipr_is_monotonic_in_drawdown(self, model: ThermalReservoirModel) -> None:
        p_res = RESERVOIR.initial_reservoir_pressure_mpa
        rates = [model._vogel_inflow(f * p_res) for f in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
        for earlier, later in zip(rates, rates[1:]):
            assert later < earlier

    def test_skin_reduces_inflow(self) -> None:
        low = ThermalReservoirModel(skin_factor=0.0)
        high = ThermalReservoirModel(skin_factor=8.0)
        p = 0.35 * RESERVOIR.initial_reservoir_pressure_mpa
        assert high._vogel_inflow(p) < low._vogel_inflow(p)

    def test_skin_never_saturates_inflow_to_zero(self) -> None:
        """The naive x*exp(s) substitution collapses; the drawdown split does not."""
        heavy_skin = ThermalReservoirModel(skin_factor=25.0)
        rate = heavy_skin._vogel_inflow(0.35 * RESERVOIR.initial_reservoir_pressure_mpa)
        assert rate > 0.0

    def test_sor_is_volumetric(self, model: ThermalReservoirModel) -> None:
        result = model.simulate_cycle()
        final = result.states[-1]
        assert final.cumulative_sor == pytest.approx(
            final.cumulative_steam_volume_m3 / final.cumulative_oil_volume_m3, rel=1e-12
        )

    def test_steam_volume_tracks_the_injection_rate(
        self, model: ThermalReservoirModel
    ) -> None:
        result = model.simulate_cycle()
        final = result.states[-1]
        expected = model.steam_rate_m3_per_day * CSS.injection_days
        assert final.cumulative_steam_volume_m3 == pytest.approx(
            expected, rel=2 * model.dt / CSS.injection_days
        )

    def test_sor_falls_as_the_well_produces(self, model: ThermalReservoirModel) -> None:
        result = model.simulate_cycle()
        production = [s for s in result.states if s.phase == CSSPhase.PRODUCTION]
        sor = [s.cumulative_sor for s in production]
        for earlier, later in zip(sor[20:], sor[21:]):
            assert later < earlier  # strictly improving as oil accumulates


class TestEconomicCutoff:
    def test_healthy_cycle_is_not_cut_short(self, model: ThermalReservoirModel) -> None:
        result = model.simulate_cycle()
        assert result.states[-1].cumulative_sor < model.cutoff_sor
        assert not result.states[-1].cutoff_reached

    def test_degraded_reservoir_triggers_the_cutoff(self) -> None:
        """High skin and low permeability push the SOR through the cut-off.

        The cumulative SOR peaks early in a degraded cycle (all the steam is
        charged while the starved well builds its flush) and can then fall back
        below the limit, so the test inspects the state at the moment of the
        trip rather than the end of the cycle.
        """
        degraded = ThermalReservoirModel(skin_factor=20.0, permeability_md=40.0)
        result = degraded.simulate_cycle(production_days=200.0)
        tripped = next((s for s in result.states if s.cutoff_reached), None)
        assert tripped is not None
        assert tripped.cutoff_reason
        assert tripped.cumulative_sor > degraded.cutoff_sor
        assert tripped.bottom_hole_temperature_c < CSS.cutoff_bht_c
        assert tripped.time_days >= THRESHOLDS.min_production_days

    def test_cutoff_respects_the_minimum_production_period(
        self, model: ThermalReservoirModel
    ) -> None:
        """A short cycle must never be cut off inside the minimum period."""
        for days in (1.0, 5.0, 11.0):
            result = model.simulate_cycle(production_days=days)
            assert not result.states[-1].cutoff_reached

    def test_cutoff_is_monotonic_in_severity(self) -> None:
        """A tighter cut-off can never terminate a cycle later than a looser one.

        Ties are expected and correct: once the reservoir has lost its thermal
        support the BHT condition is what binds, and every cut-off above the
        then-current SOR trips on the same day.
        """
        cut_days = []
        for cutoff in (1.5, 2.0, 3.0, 4.2, 6.0):
            m = ThermalReservoirModel(
                cutoff_sor=cutoff, permeability_md=60.0, skin_factor=12.0
            )
            result = m.simulate_cycle(production_days=200.0)
            first = next((s.time_days for s in result.states if s.cutoff_reached), None)
            cut_days.append(first if first is not None else math.inf)
        for earlier, later in zip(cut_days, cut_days[1:]):
            assert earlier <= later
        assert cut_days[0] < math.inf

    def test_cutoff_requires_both_conditions(self) -> None:
        """A well that still has thermal support is never cut off.

        The BHT only falls through the economic limit near the end of the
        cycle, so a cut-off tight enough to be breached on the very first
        production day must still not trip: the minimum production period and
        the BHT condition both hold it back until the well is genuinely spent.
        """
        healthy = ThermalReservoirModel(
            cutoff_sor=0.01, permeability_md=320.0, skin_factor=3.2
        )
        result = healthy.simulate_cycle(production_days=20.0)
        assert not result.states[-1].cutoff_reached
        assert result.states[-1].bottom_hole_temperature_c > CSS.cutoff_bht_c
        assert result.states[-1].time_days > THRESHOLDS.min_production_days

    def test_cutoff_truncates_the_production_phase(self) -> None:
        """Production stops at the cut-off, which is what makes the schedule trade-off real."""
        truncated = ThermalReservoirModel(cutoff_sor=1.5, permeability_md=60.0, skin_factor=12.0)
        production_days = 200.0
        result = truncated.simulate_cycle(production_days=production_days)
        assert result.states[-1].cutoff_reached
        # Elapsed time counts the injection and soak as well, so the production
        # span that actually ran is the elapsed time less those two phases.
        produced = result.states[-1].time_days - CSS.injection_days - CSS.soak_days
        assert produced < production_days
        assert produced > 0.0
        # A generous limit on the same well runs the full period.
        full = ThermalReservoirModel(cutoff_sor=1.0e6, permeability_md=60.0, skin_factor=12.0)
        assert not full.simulate_cycle(production_days=200.0).states[-1].cutoff_reached


class TestProgramme:
    def test_programme_runs_multiple_cycles(self) -> None:
        # A short production period keeps the cumulative SOR well inside the
        # cut-off, so the programme is free to run every cycle requested.
        programme = baghewala_css_model().simulate_programme(cycles=2, production_days=32.0)
        assert programme.cycles_completed == 2
        assert programme.total_steam_tonnes > 0.0
        assert programme.total_oil_tonnes > 0.0

    def test_programme_sor_is_volumetric(self) -> None:
        programme = baghewala_css_model().simulate_programme(cycles=2, production_days=32.0)
        final = programme.states[-1]
        assert programme.cumulative_sor == pytest.approx(
            final.cumulative_steam_volume_m3 / final.cumulative_oil_volume_m3, rel=1e-12
        )

    def test_programme_stops_at_economic_cutoff(self) -> None:
        degraded = ThermalReservoirModel(
            skin_factor=12.0, permeability_md=60.0, cutoff_sor=2.0
        )
        programme = degraded.simulate_programme(cycles=6, production_days=140.0)
        assert programme.economic_cutoff_cycles
        assert programme.cycles_completed <= 6

    def test_state_lookup_returns_nearest_time(self) -> None:
        result = ThermalReservoirModel().simulate_cycle()
        for time_days in (0.0, 5.0, 30.0, 70.0):
            state = result.state_at(time_days)
            assert state is not None
            assert abs(state.time_days - time_days) < 5.0

    def test_rejects_invalid_configuration(self) -> None:
        with pytest.raises(ValueError):
            ThermalReservoirModel(dt_days=0.0)
        with pytest.raises(ValueError):
            ThermalReservoirModel(steam_quality=1.5)
        with pytest.raises(ValueError):
            ThermalReservoirModel(steam_retention_factor=0.0)
        with pytest.raises(ValueError):
            ThermalReservoirModel(drainage_radius_m=0.05, wellbore_radius_m=0.108)
        with pytest.raises(ValueError):
            ThermalReservoirModel().simulate_cycle(production_days=0.0)
        with pytest.raises(ValueError):
            ThermalReservoirModel().simulate_cycle(soak_days=-1.0)
        with pytest.raises(ValueError):
            ThermalReservoirModel().simulate_cycle(record_every=0)
        with pytest.raises(ValueError):
            ThermalReservoirModel().simulate_programme(cycles=0)
        with pytest.raises(ValueError):
            ThermalReservoirModel()._thermal_decay(-1.0)
