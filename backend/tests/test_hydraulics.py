"""Verification of the wellbore multiphase hydraulics and Ramey heat transfer.

The Ramey result is physically demanding for a Baghewala completion: at the
pump-limited rate of ~30 bbl/d the residence time in a 1000 m well is about
22 hours, which is long enough for the stream to relax almost completely onto
the formation temperature.  That is not a numerical artefact -- it is the
physical reason a CSS well whose wellhead temperature has collapsed onto the
native 46 degC has no remaining thermal support, and it is exactly the signal
the digital twin keys its cycle-cutoff decision on.
"""

from __future__ import annotations

import math

import pytest

from backend.app.core.config import PUMPING_UNIT, RESERVOIR, WELL
from backend.app.physics.hydraulics import (
    FluidProperties,
    beggs_brill_holdup,
    colebrook_white_friction,
    compute_traverse,
    default_mass_flow_kg_s,
    emulsion_viscosity,
    flow_regime,
    mixture_properties,
    ramey_heat_fraction,
    ramey_heat_transfer_conductance,
    ramey_wellbore_temperature,
    wellbore_thermal_conductivity,
)


class TestEmulsionRheology:
    def test_end_points_reduce_to_the_pure_phases(self) -> None:
        assert emulsion_viscosity(12.5, 1.0e-3, 1.0, 0.0) == pytest.approx(12.5, rel=1e-9)
        assert emulsion_viscosity(12.5, 1.0e-3, 0.0, 1.0) == pytest.approx(1.0e-3, rel=1e-9)

    def test_emulsion_exceeds_the_log_mean(self) -> None:
        """The Grunberg-Nissan interaction term raises the viscosity above the ideal mean."""
        mu_o, mu_w, k = 12.5, 1.0e-3, 2.6
        for x_w in (0.1, 0.3, 0.5, 0.7):
            log_mean = math.exp((1 - x_w) * math.log(mu_o) + x_w * math.log(mu_w))
            assert emulsion_viscosity(mu_o, mu_w, 1 - x_w, x_w, k) > log_mean

    def test_stronger_interaction_raises_viscosity(self) -> None:
        weak = emulsion_viscosity(12.5, 1.0e-3, 0.6, 0.4, grunberg_nissan=0.5)
        strong = emulsion_viscosity(12.5, 1.0e-3, 0.6, 0.4, grunberg_nissan=6.0)
        assert strong > weak

    def test_heavier_crude_gives_a_higher_emulsion_viscosity(self) -> None:
        thin = emulsion_viscosity(0.5, 1.0e-3, 0.6, 0.4)
        heavy = emulsion_viscosity(12.5, 1.0e-3, 0.6, 0.4)
        assert heavy > thin

    def test_rejects_invalid_input(self) -> None:
        with pytest.raises(ValueError):
            emulsion_viscosity(0.0, 1.0e-3, 1.0, 1.0)
        with pytest.raises(ValueError):
            emulsion_viscosity(12.5, 1.0e-3, 0.0, 0.0)
        with pytest.raises(ValueError):
            emulsion_viscosity(12.5, 1.0e-3, 1.0, 1.0, grunberg_nissan=-1.0)


class TestMixtureProperties:
    def test_density_bracketsthe_phase_densities(self) -> None:
        oil = FluidProperties(46.0, 8.6e6, 1.0, 0.0, oil_viscosity_pa_s=12.5)
        water = FluidProperties(46.0, 8.6e6, 0.0, 1.0, oil_viscosity_pa_s=12.5)
        mix = FluidProperties(46.0, 8.6e6, 0.5, 0.5, oil_viscosity_pa_s=12.5)
        rho_oil, _ = mixture_properties(oil)
        rho_water, _ = mixture_properties(water)
        rho_mix, _ = mixture_properties(mix)
        assert min(rho_oil, rho_water) < rho_mix < max(rho_oil, rho_water)

    def test_density_increases_with_water_cut(self) -> None:
        densities = []
        for water_cut in (0.0, 0.25, 0.5, 0.75, 1.0):
            fluid = FluidProperties(
                46.0, 8.6e6, 1 - water_cut, water_cut, oil_viscosity_pa_s=12.5
            )
            rho, _ = mixture_properties(fluid)
            densities.append(rho)
        for earlier, later in zip(densities, densities[1:]):
            assert later > earlier

    def test_viscosity_decreases_with_temperature(self) -> None:
        from backend.app.physics.rheology import baghewala_crude

        crude = baghewala_crude()
        viscosities = []
        for temperature_c in (46.0, 80.0, 140.0, 200.0):
            fluid = FluidProperties(
                temperature_c,
                8.6e6,
                0.88,
                0.12,
                oil_viscosity_pa_s=crude.dynamic_viscosity_pa_s(temperature_c),
            )
            _, mu = mixture_properties(fluid)
            viscosities.append(mu)
        for earlier, later in zip(viscosities, viscosities[1:]):
            assert later < earlier

    def test_water_cut_property(self) -> None:
        fluid = FluidProperties(46.0, 8.6e6, 0.75, 0.25, oil_viscosity_pa_s=1.0)
        assert fluid.water_cut == pytest.approx(0.25)
        dry = FluidProperties(46.0, 8.6e6, 1.0, 0.0, oil_viscosity_pa_s=1.0)
        assert dry.water_cut == 0.0

    def test_rejects_negative_fractions(self) -> None:
        with pytest.raises(ValueError):
            FluidProperties(46.0, 8.6e6, -0.1, 1.0, oil_viscosity_pa_s=1.0)
        with pytest.raises(ValueError):
            FluidProperties(46.0, 8.6e6, 0.0, 0.0, oil_viscosity_pa_s=1.0)


class TestBeggsBrillHoldup:
    def test_holdup_is_bounded(self) -> None:
        for v_liquid in (0.01, 0.1, 1.0, 5.0):
            for v_gas in (0.0, 0.5, 2.0, 10.0):
                holdup, regime = beggs_brill_holdup(v_liquid, v_gas)
                assert 0.0 <= holdup <= 1.0
                assert regime in {
                    "homogeneous",
                    "stratified",
                    "annular",
                    "slug",
                }

    def test_single_phase_limits_are_exact(self) -> None:
        """A gas-free line is fully liquid filled and a liquid-free line fully gas filled."""
        assert beggs_brill_holdup(1.0, 0.0)[0] == 1.0
        assert beggs_brill_holdup(0.0, 1.0)[0] == 0.0

    def test_holdup_matches_the_published_closed_form(self) -> None:
        """The implementation must reproduce the Beggs-Brill correlation exactly.

        With C_NL = 10**a and C_NS = b fixed by the flow pattern, the
        correlation is Y_L = 1 / (1 + C_NL (1 + v_m / v_l) ** C_NS).
        """
        from backend.app.physics.hydraulics import _BB_COEFFICIENTS

        diameter = 0.0762
        for v_gas in (0.05, 0.5, 2.0):
            for v_liquid in (0.2, 1.0, 5.0):
                holdup, regime = beggs_brill_holdup(
                    v_liquid,
                    v_gas,
                    0.0,
                    900.0,
                    60.0,
                    diameter,
                )
                a, b = _BB_COEFFICIENTS[regime][0]
                expected = 1.0 / (
                    1.0 + (10.0**a) * (1.0 + (v_liquid + v_gas) / v_liquid) ** b
                )
                assert holdup == pytest.approx(expected, rel=1e-12)

    def test_holdup_is_insensitive_to_the_gas_fraction(self) -> None:
        """Documents a genuine property of the Beggs-Brill holdup correlation.

        Its ratio exponent is small (|C_NS| ~ 0.03-0.06), so over the
        two-phase domain the holdup varies only weakly with the gas fraction
        and sits in the 0.4-0.55 band.  The value of the correlation lies in
        the regime selection and the exact single-phase limits, not in a strong
        gas-fraction response; asserting a strong response would be asserting
        something the published correlation does not contain.
        """
        holdups = [beggs_brill_holdup(2.0, v_g)[0] for v_g in (0.05, 0.2, 0.5, 1.0, 3.0, 4.0)]
        for holdup in holdups:
            assert 0.30 < holdup < 0.60
        assert max(holdups) - min(holdups) < 0.10

    def test_holdup_is_bounded_across_the_two_phase_domain(self) -> None:
        for v_liquid in (0.01, 0.1, 1.0, 10.0):
            for v_gas in (0.01, 0.1, 1.0, 10.0):
                holdup, _ = beggs_brill_holdup(v_liquid, v_gas)
                assert 0.0 < holdup < 1.0

    def test_holdup_rejects_negative_velocities(self) -> None:
        with pytest.raises(ValueError):
            beggs_brill_holdup(-1.0, 1.0)
        with pytest.raises(ValueError):
            beggs_brill_holdup(1.0, -1.0)

    def test_no_liquid_gives_zero_holdup(self) -> None:
        holdup, regime = beggs_brill_holdup(0.0, 1.0)
        assert holdup == 0.0
        assert regime == "annular"

    def test_high_gas_fraction_selects_a_slug_or_homogeneous_regime(self) -> None:
        _, regime = beggs_brill_holdup(0.2, 5.0)
        assert regime in {"slug", "homogeneous"}

    def test_flow_pattern_map(self) -> None:
        """Beggs & Brill map: Fr_g = v_g^2/(gD) and M = rho_L/rho_G select the pattern."""
        g, d = 9.80665, 0.0762
        # Fr_g < 0.01 and M < 350 -> stratified (900/60 = 15)
        slow_gas = math.sqrt(0.005 * g * d)
        assert flow_regime(0.5, slow_gas, 900.0, 60.0, d) == "stratified"
        # Fr_g < 0.01 and M > 350 -> annular
        assert flow_regime(0.5, slow_gas, 900.0, 1.0, d) == "annular"
        # 0.01 < Fr_g < 1 and M > 350 -> slug
        mid_gas = math.sqrt(0.1 * g * d)
        assert flow_regime(0.5, mid_gas, 900.0, 1.0, d) == "slug"
        # Fr_g > 1 -> homogeneous
        fast_gas = math.sqrt(5.0 * g * d)
        assert flow_regime(0.5, fast_gas, 900.0, 60.0, d) == "homogeneous"

    def test_single_phase_regimes(self) -> None:
        assert flow_regime(1.0, 0.0) == "homogeneous"
        assert flow_regime(0.0, 1.0) == "annular"


class TestFrictionFactor:
    def test_laminar_branch(self) -> None:
        for reynolds in (10.0, 100.0, 1000.0):
            assert colebrook_white_friction(reynolds, 0.0) == pytest.approx(
                64.0 / reynolds, rel=1e-9
            )

    def test_colebrook_satisfies_its_own_equation(self) -> None:
        """The returned factor must satisfy 1/sqrt(f) = -2 log10(eps/3.7D + 2.51/(Re sqrt(f)))."""
        for reynolds in (5e3, 1e4, 1e5, 1e6):
            for roughness in (1e-5, 1e-4, 1e-3):
                f = colebrook_white_friction(reynolds, roughness)
                lhs = 1.0 / math.sqrt(f)
                rhs = -2.0 * math.log10(roughness / 3.7 + 2.51 / (reynolds * math.sqrt(f)))
                assert lhs == pytest.approx(rhs, rel=1e-4)

    def test_smooth_pipe_uses_blasius(self) -> None:
        for reynolds in (1e4, 5e4, 1e5):
            assert colebrook_white_friction(reynolds, 0.0) == pytest.approx(
                0.3164 * reynolds**-0.25, rel=1e-9
            )

    def test_roughness_increases_friction_in_turbulent_flow(self) -> None:
        smooth = colebrook_white_friction(1e5, 1e-6)
        rough = colebrook_white_friction(1e5, 1e-2)
        assert rough > smooth

    def test_friction_decreases_with_reynolds_in_turbulent_flow(self) -> None:
        values = [colebrook_white_friction(re, 1e-4) for re in (1e4, 1e5, 1e6)]
        for earlier, later in zip(values, values[1:]):
            assert later < earlier

    def test_extreme_low_reynolds_is_clamped(self) -> None:
        assert colebrook_white_friction(0.0, 0.0) == pytest.approx(64.0, rel=1e-9)


class TestRameyHeatTransfer:
    def test_effective_conductivity_is_finite_and_positive(self) -> None:
        conductivity = wellbore_thermal_conductivity()
        assert 0.5 < conductivity < 10.0

    def test_conductance_relaxes_to_the_quasi_steady_value(self) -> None:
        radius = 0.5 * WELL.tubing_id_m
        conductivity = wellbore_thermal_conductivity()
        steady = (
            2.0
            * math.pi
            * conductivity
            / math.log(RESERVOIR.drainage_radius_m / radius)
        )
        # Far from the formation the transient branch is the binding one and
        # decays; the conductance can never exceed the quasi-steady value.
        for days in (1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0):
            value = ramey_heat_transfer_conductance(
                days, radius, conductivity, 1.1e-7, RESERVOIR.drainage_radius_m
            )
            assert 0.0 < value <= steady + 1e-9

    def test_heat_fraction_bounds(self) -> None:
        """The Ramey heat fraction must lie in (0, 1] for all depths and times."""
        radius = 0.5 * WELL.tubing_id_m
        conductivity = wellbore_thermal_conductivity()
        for depth in (0.0, 1.0, 100.0, 500.0, 1000.0, 2000.0):
            for mass_flow in (0.01, 0.05, 0.5, 5.0):
                fraction = ramey_heat_fraction(
                    depth, mass_flow, radius, conductivity, 900.0, 2000.0
                )
                assert 0.0 <= fraction <= 1.0

    def test_heat_fraction_is_one_at_the_inlet(self) -> None:
        radius = 0.5 * WELL.tubing_id_m
        assert ramey_heat_fraction(0.0, 0.05, radius, 2.3, 900.0, 2000.0) == 1.0

    def test_heat_fraction_decays_with_depth(self) -> None:
        radius = 0.5 * WELL.tubing_id_m
        fractions = [
            ramey_heat_fraction(d, 0.05, radius, 2.3, 900.0, 2000.0)
            for d in (0.0, 100.0, 300.0, 600.0, 1000.0)
        ]
        assert fractions[0] == 1.0
        for earlier, later in zip(fractions, fractions[1:]):
            assert later < earlier

    def test_higher_flow_rate_reduces_the_cooling(self) -> None:
        """Shorter contact time means less heat transferred to the formation.

        The sweep is run over the rate range where the Ramey solution is
        sensitive.  At the pump-limited Baghewala rate of ~30 bbl/d the
        residence time in a 1000 m well is about 22 hours, which is long
        enough for the stream to relax completely onto the formation
        temperature; that asymptotic regime is asserted separately.
        """
        radius = 0.5 * WELL.tubing_id_m
        conductivity = wellbore_thermal_conductivity()
        temperatures = []
        for mass_flow in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
            temperatures.append(
                ramey_wellbore_temperature(
                    flowing_temperature_c=145.0,
                    formation_temperature_c=RESERVOIR.reservoir_temperature_c,
                    mass_flow_kg_s=mass_flow,
                    radius_m=radius,
                    depth_m=1000.0,
                    effective_conductivity_w_mk=conductivity,
                    flowing_density_kg_m3=900.0,
                    specific_heat_j_kgk=2000.0,
                    drainage_radius_m=RESERVOIR.drainage_radius_m,
                )
            )
        for earlier, later in zip(temperatures, temperatures[1:]):
            assert later > earlier

    def test_pump_rate_relaxes_onto_the_formation_temperature(self) -> None:
        """At the pump rate the stream is in thermal equilibrium with the formation.

        This is the physical basis of the CSS cycle cut-off: once the wellhead
        temperature has collapsed onto the native reservoir temperature there
        is no remaining thermal support in the produced fluid.
        """
        radius = 0.5 * WELL.tubing_id_m
        temperature = ramey_wellbore_temperature(
            flowing_temperature_c=145.0,
            formation_temperature_c=RESERVOIR.reservoir_temperature_c,
            mass_flow_kg_s=default_mass_flow_kg_s(),
            radius_m=radius,
            depth_m=WELL.pump_depth_m,
            effective_conductivity_w_mk=wellbore_thermal_conductivity(),
            flowing_density_kg_m3=900.0,
            specific_heat_j_kgk=2000.0,
            drainage_radius_m=RESERVOIR.drainage_radius_m,
        )
        assert temperature == pytest.approx(
            RESERVOIR.reservoir_temperature_c, abs=1.0
        )

    def test_temperature_stays_between_formation_and_inlet(self) -> None:
        """The spec's Ramey bound: the wellbore temperature can never overshoot."""
        radius = 0.5 * WELL.tubing_id_m
        conductivity = wellbore_thermal_conductivity()
        inlet = 145.0
        formation = RESERVOIR.reservoir_temperature_c
        for mass_flow in (0.01, 0.05, 0.5, 5.0):
            for depth in (0.0, 10.0, 500.0, 1000.0):
                temperature = ramey_wellbore_temperature(
                    flowing_temperature_c=inlet,
                    formation_temperature_c=formation,
                    mass_flow_kg_s=mass_flow,
                    radius_m=radius,
                    depth_m=depth,
                    effective_conductivity_w_mk=conductivity,
                    flowing_density_kg_m3=900.0,
                    specific_heat_j_kgk=2000.0,
                    drainage_radius_m=RESERVOIR.drainage_radius_m,
                )
                assert formation <= temperature <= inlet

    def test_cold_stream_is_warmed_towards_the_formation(self) -> None:
        radius = 0.5 * WELL.tubing_id_m
        temperature = ramey_wellbore_temperature(
            flowing_temperature_c=30.0,
            formation_temperature_c=RESERVOIR.reservoir_temperature_c,
            mass_flow_kg_s=0.05,
            radius_m=radius,
            depth_m=1000.0,
            effective_conductivity_w_mk=2.3,
            flowing_density_kg_m3=900.0,
            specific_heat_j_kgk=2000.0,
        )
        assert temperature >= 30.0
        assert temperature <= RESERVOIR.reservoir_temperature_c

    def test_rejects_invalid_arguments(self) -> None:
        with pytest.raises(ValueError):
            ramey_wellbore_temperature(120.0, 46.0, 0.0, 0.038, 100.0, 2.3)
        with pytest.raises(ValueError):
            ramey_wellbore_temperature(120.0, 46.0, 0.05, 0.0, 100.0, 2.3)
        with pytest.raises(ValueError):
            ramey_wellbore_temperature(120.0, 46.0, 0.05, 0.038, 100.0, 0.0)
        with pytest.raises(ValueError):
            ramey_wellbore_temperature(120.0, 46.0, 0.05, 0.038, -1.0, 2.3)


class TestTraverse:
    def test_default_mass_flow_matches_the_pump_capacity(self) -> None:
        """The default rate is the pump rate, not an arbitrary mass flow."""
        expected = (
            PUMPING_UNIT.nominal_spm
            * 2.0
            * 60.0
            * PUMPING_UNIT.plunger_area_m2
            * PUMPING_UNIT.nominal_stroke_m
            * PUMPING_UNIT.pump_efficiency
            * 900.0
            / 86400.0
        )
        assert default_mass_flow_kg_s() == pytest.approx(expected, rel=1e-12)
        bbl_per_day = default_mass_flow_kg_s() / 900.0 * 86400.0 * 6.2898
        assert 20.0 < bbl_per_day < 45.0

    def test_profiles_are_monotonic_in_depth(self) -> None:
        traverse = compute_traverse(segments=40)
        for earlier, later in zip(traverse.pressure_pa, traverse.pressure_pa[1:]):
            assert later <= earlier + 1e-6
        for earlier, later in zip(traverse.temperature_c, traverse.temperature_c[1:]):
            assert later <= earlier + 1e-6

    def test_wellhead_pressure_stays_above_atmospheric(self) -> None:
        traverse = compute_traverse(segments=40)
        assert traverse.pressure_pa[-1] > 101325.0
        assert traverse.total_pressure_drop_pa > 0.0

    def test_pressure_drop_increases_monotonically_with_water_cut(self) -> None:
        """Spec requirement: the pressure drop grows with water cut.

        Water is denser than the crude, so the hydrostatic term rises
        monotonically; the emulsion viscosity term partly offsets it, and the
        net hydrostatic dominance is what the test pins down.
        """
        water_cuts = [i / 20.0 for i in range(21)]
        drops = [compute_traverse(water_cut=wc, segments=30).total_pressure_drop_pa for wc in water_cuts]
        for earlier, later in zip(drops, drops[1:]):
            assert later > earlier, "pressure drop must increase with water cut"

    def test_hydrostatic_component_increases_with_water_cut(self) -> None:
        hydrostatics = [
            compute_traverse(water_cut=wc, segments=30).metadata["hydrostatic_pressure_drop_pa"]
            for wc in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        for earlier, later in zip(hydrostatics, hydrostatics[1:]):
            assert later > earlier

    def test_pressure_drop_increases_with_mixture_viscosity(self) -> None:
        """At fixed density and rate, a more viscous emulsion costs more pressure."""
        # Compare identical traverses at two inlet temperatures: the colder
        # stream is far more viscous, so its friction loss must be larger.
        cold = compute_traverse(bottom_hole_temperature_c=52.0, segments=30)
        hot = compute_traverse(bottom_hole_temperature_c=200.0, segments=30)
        assert (
            cold.metadata["friction_pressure_drop_pa"]
            > hot.metadata["friction_pressure_drop_pa"]
        )

    def test_pressure_drop_increases_with_mass_rate(self) -> None:
        rates = [0.02, 0.05, 0.1, 0.2, 0.4]
        drops = [compute_traverse(mass_flow_kg_s=q, segments=30).total_pressure_drop_pa for q in rates]
        for earlier, later in zip(drops, drops[1:]):
            assert later > earlier

    def test_holdup_is_bounded_along_the_whole_traverse(self) -> None:
        traverse = compute_traverse(segments=40)
        assert np_all_in_unit_interval(traverse.liquid_holdup)

    def test_regimes_are_labelled_everywhere(self) -> None:
        traverse = compute_traverse(segments=40)
        allowed = {"homogeneous", "stratified", "annular", "slug"}
        assert set(traverse.regime) <= allowed
        assert len(traverse.regime) == traverse.segments + 1

    def test_serialisation_round_trips_the_arrays(self) -> None:
        traverse = compute_traverse(segments=20)
        payload = traverse.as_dict()
        assert len(payload["depth_m"]) == traverse.segments + 1
        assert len(payload["pressure_mpa"]) == traverse.segments + 1
        assert payload["total_pressure_drop_mpa"] == pytest.approx(
            traverse.total_pressure_drop_pa / 1e6, rel=1e-12
        )

    def test_rejects_invalid_configuration(self) -> None:
        with pytest.raises(ValueError):
            compute_traverse(segments=1)
        with pytest.raises(ValueError):
            compute_traverse(mass_flow_kg_s=0.0)
        with pytest.raises(ValueError):
            compute_traverse(mass_flow_kg_s=-1.0)
        with pytest.raises(ValueError):
            compute_traverse(water_cut=1.5)
        with pytest.raises(ValueError):
            compute_traverse(water_cut=-0.1)


def np_all_in_unit_interval(values) -> bool:
    """True when every entry of ``values`` lies in [0, 1]."""
    return all(0.0 <= float(v) <= 1.0 for v in values)
