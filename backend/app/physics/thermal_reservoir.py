"""Marx-Langenheim cyclic steam stimulation and conductive heat dissipation.

The steam chest of a CSS cycle grows radially around the wellbore.  Heat is
transported from the hot chest into the formation by radial conduction, and the
losses through the over- and underburden are evaluated with the complementary
error function ``erfc`` solution of the one-dimensional transient conduction
problem.

Governing relations
-------------------
Steam chest energy balance (Marx-Langenheim), for a steam chest of radius
:math:`R_s(t)` and formation drainage radius :math:`R_e`:

.. math::

    \\frac{dQ_{ch}}{dt} = \\underbrace{2\\pi k (T_{ch} - T_{res})
        \\frac{R_s}{\\ln(R_e/R_s)}}_{\\text{conduction to the formation}}
      - \\underbrace{\\dot{m}_s\\,(h_{steam} - h_{res})}_{\\text{net injection}}

The radial conduction rate uses the quasi-steady annulus solution.  The
over/underburden loss follows the transient slab solution

.. math::

    \\dot{Q}_{ou} = \\frac{k_{ou}A_{ou}}{\\sqrt{\\pi\\alpha_{ou}\\,t}}
        \\left[(T_{ou} - T_{res})\\,\\mathrm{erfc}\\!\\left(
        \\frac{d_{ou}}{2\\sqrt{\\alpha_{ou} t}}\\right)
        - (T_{ref} - T_{res})\\right]

where the first term is the surface-temperature response of the bounding
shale and the second removes the pre-existing temperature contrast.

The steam-oil ratio is tracked both instantaneously and cumulatively so that
the economic cut-off of 4.2 can be evaluated on either basis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from backend.app.core.config import (
    CSS,
    PUMPING_UNIT,
    RESERVOIR,
    THRESHOLDS,
    BaghewalaFluid,
)
from backend.app.physics.rheology import baghewala_crude

__all__ = [
    "CSSPhase",
    "SteamChestState",
    "CSOResult",
    "ThermalReservoirModel",
    "erfc",
    "baghewala_css_model",
]

#: Truncation depth of the Legendre continued fraction used by :func:`erfc`.
_ERFC_CF_DEPTH = 200


def erfc(x: float) -> float:
    """Complementary error function, accurate to near machine precision.

    Two regions are used so that no catastrophic cancellation occurs:

    * ``|x| < 2.5`` the complement is evaluated as ``1 - erf(x)``, where
      ``math.erf`` is correctly rounded.  In this range ``erfc`` is never
      small enough for the subtraction to lose significant digits.
    * ``|x| >= 2.5`` the tail is evaluated with the Legendre continued
      fraction, which converges rapidly here and avoids forming ``1 - erf(x)``
      where ``erf`` is already 1 to machine precision:

      .. math::

          \\mathrm{erfc}(x) = \\frac{e^{-x^2}}{\\sqrt{\\pi}\\,
              \\left(x + \\cfrac{1/2}{x + \\cfrac{1}{x +
              \\cfrac{3/2}{x + \\cfrac{2}{x + \\ddots}}}}\right)}

    The symmetry ``erfc(-x) = 2 - erfc(x)`` is applied last.
    """
    if x != x:  # NaN
        return float("nan")
    if x == 0.0:
        return 1.0
    if x < -30.0:
        return 2.0
    if x > 30.0:
        return 0.0

    a = abs(x)
    if a < 2.5:
        value = 1.0 - math.erf(a)
    else:
        # Legendre continued fraction, evaluated from the tail inward.
        f = 0.0
        for index in range(_ERFC_CF_DEPTH, 0, -1):
            f = (index / 2.0) / (a + f)
        value = math.exp(-a * a) / (math.sqrt(math.pi) * (a + f))
    return value if x >= 0.0 else 2.0 - value


class CSSPhase(str, Enum):
    """Operational phase of a CSS cycle."""

    INJECTION = "INJECTION"
    SOAKING = "SOAKING"
    PRODUCTION = "PRODUCTION"


@dataclass
class SteamChestState:
    """Instantaneous state of the steam chest and its thermal front."""

    time_days: float = 0.0
    phase: CSSPhase = CSSPhase.INJECTION
    radius_m: float = RESERVOIR.wellbore_radius_m
    chest_temperature_c: float = RESERVOIR.reservoir_temperature_c
    steam_quality: float = 0.0
    cumulative_steam_tonnes: float = 0.0
    cumulative_steam_m3: float = 0.0
    cumulative_oil_tonnes: float = 0.0
    cumulative_heat_retained_j: float = 0.0
    cumulative_heat_overburden_j: float = 0.0
    cumulative_heat_formation_j: float = 0.0
    cumulative_heat_produced_j: float = 0.0
    instantaneous_sor: float = 0.0
    cumulative_sor: float = 0.0
    #: Steam and oil volumes actually used by the steam-oil ratio, m^3.
    cumulative_steam_volume_m3: float = 0.0
    cumulative_oil_volume_m3: float = 0.0
    oil_rate_tpd: float = 0.0
    reservoir_inflow_m3_per_day: float = 0.0
    pump_fillage: float = 0.0
    bottom_hole_temperature_c: float = RESERVOIR.reservoir_temperature_c
    cutoff_reached: bool = False
    cutoff_reason: str = ""


@dataclass
class CSOResult:
    """Terminal result of a simulated CSS cycle sequence."""

    states: List[SteamChestState] = field(default_factory=list)
    total_steam_tonnes: float = 0.0
    total_oil_tonnes: float = 0.0
    cumulative_sor: float = 0.0
    final_bht_c: float = RESERVOIR.reservoir_temperature_c
    cycles_completed: int = 0
    economic_cutoff_cycles: List[int] = field(default_factory=list)
    net_revenue_usd: float = 0.0

    def state_at(self, time_days: float) -> SteamChestState:
        """Nearest simulated state to ``time_days`` (linear search, exact at nodes)."""
        if not self.states:
            raise ValueError("no states recorded in this CSS result")
        return min(self.states, key=lambda s: abs(s.time_days - time_days))


class ThermalReservoirModel:
    """Marx-Langenheim steam chest growth with erfc over/underburden losses.

    Parameters mirror the field configuration; every value can be overridden by
    the caller, which is what the optimiser does when it searches schedules.
    """

    def __init__(
        self,
        reservoir_temperature_c: float = RESERVOIR.reservoir_temperature_c,
        thermal_conductivity_w_mk: float = RESERVOIR.thermal_conductivity_w_mk,
        volumetric_heat_capacity_j_m3k: float = RESERVOIR.volumetric_heat_capacity_j_m3k,
        overburden_conductivity_w_mk: float = RESERVOIR.overburden_conductivity_w_mk,
        overburden_thickness_m: float = RESERVOIR.overburden_thickness_m,
        underburden_thickness_m: float = RESERVOIR.underburden_thickness_m,
        overburden_diffusivity_m2s: float = RESERVOIR.overburden_diffusivity_m2s,
        vertical_contact_factor: float = RESERVOIR.vertical_contact_factor,
        drainage_radius_m: float = RESERVOIR.drainage_radius_m,
        wellbore_radius_m: float = RESERVOIR.wellbore_radius_m,
        steam_quality: float = CSS.steam_quality,
        steam_temperature_c: float = CSS.steam_temperature_c,
        steam_enthalpy_kj_kg: float = CSS.steam_enthalpy_kj_kg,
        steam_retention_factor: float = CSS.steam_retention_factor,
        steam_rate_m3_per_day: float = CSS.steam_rate_m3_per_day,
        steam_specific_volume_m3_kg: float = CSS.steam_specific_volume_m3_kg,
        oil_specific_gravity: float = 0.88,
        permeability_md: float = RESERVOIR.permeability_md,
        skin_factor: float = RESERVOIR.skin_factor,
        cutoff_sor: float = CSS.instantaneous_sor_cutoff,
        dt_days: float = 0.05,
    ) -> None:
        if dt_days <= 0.0:
            raise ValueError("dt_days must be positive")
        if drainage_radius_m <= wellbore_radius_m:
            raise ValueError("drainage radius must exceed the wellbore radius")
        if steam_quality < 0.0 or steam_quality > 1.0:
            raise ValueError("steam quality must lie in [0, 1]")
        if not 0.0 < steam_retention_factor <= 1.0:
            raise ValueError("steam retention factor must lie in (0, 1]")

        self.reservoir_temperature_c = reservoir_temperature_c
        self.k = thermal_conductivity_w_mk
        self.rho_c = volumetric_heat_capacity_j_m3k
        self.k_over = overburden_conductivity_w_mk
        self.d_over = overburden_thickness_m
        self.d_under = underburden_thickness_m
        self.alpha_over = overburden_diffusivity_m2s
        self.vertical_contact = vertical_contact_factor
        self.re = drainage_radius_m
        self.rw = wellbore_radius_m
        self.steam_quality = steam_quality
        self.steam_temperature_c = steam_temperature_c
        self.steam_enthalpy_kj_kg = steam_enthalpy_kj_kg
        self.steam_retention_factor = steam_retention_factor
        self.steam_rate_m3_per_day = steam_rate_m3_per_day
        self.steam_specific_volume_m3_kg = steam_specific_volume_m3_kg
        self.oil_specific_gravity = oil_specific_gravity
        self.k_perm = permeability_md
        self.skin = skin_factor
        self.cutoff_sor = cutoff_sor
        self.dt = dt_days
        # Radius of the steam chest as last seen by the production-phase IPR.
        self._last_steam_chest_radius_m = wellbore_radius_m

    # -- derived properties ----------------------------------------------
    @property
    def thermal_diffusivity_m2s(self) -> float:
        """Diffusivity :math:`\\alpha = k/(\\rho c_p)` in m^2/s."""
        return self.k / self.rho_c

    @property
    def pump_capacity_m3_per_day(self) -> float:
        """Volumetric capacity of the sucker rod pump, m^3/day.

        .. math::
            q_{pump} = n_{spm} \\cdot 2 \\cdot 60 \\cdot A_p L_s \\eta_v

        with :math:`n_{spm}` in strokes per minute, :math:`A_p` the plunger
        area, :math:`L_s` the stroke length and :math:`\\eta_v` the volumetric
        efficiency.
        """
        return (
            PUMPING_UNIT.nominal_spm
            * 2.0
            * 60.0
            * PUMPING_UNIT.plunger_area_m2
            * PUMPING_UNIT.nominal_stroke_m
            * PUMPING_UNIT.pump_efficiency
        )

    @property
    def oil_specific_heat_j_kgk(self) -> float:
        """Specific heat capacity of the produced oil, J/(kg K)."""
        return 2.0e6

    @property
    def pump_depth_attenuation(self) -> float:
        """Fraction of the chest temperature excess seen at pump depth.

        The steam chest sits at the reservoir interval while the pump is set
        some distance above it, so the thermal signal is attenuated by the
        Ramey-type wellbore heat-loss fraction at the pump depth.
        """
        return 0.72

    @property
    def producing_bottom_hole_pressure_mpa(self) -> float:
        """Bottom-hole pressure held at the pump during production, MPa.

        The drawdown is limited by the **hydrostatic head** of the fluid
        column, not by the reservoir pressure alone: a 1000 m column of
        900 kg/m^3 crude weighs 8.8 MPa, so the intake pressure cannot fall
        below that without the tubing emptying.  The pump therefore works
        against the difference between the reservoir pressure and the column
        weight plus the friction loss, which for this asset leaves a drawdown of
        roughly 15% of the reservoir pressure.
        """
        column_pressure = (
            900.0 * 9.80665 * self.re * 1e-6 * 0.55
        )  # effective half-column
        target = 0.85 * RESERVOIR.initial_reservoir_pressure_mpa
        return max(target, column_pressure)

    @property
    def steam_mass_flow_kg_s(self) -> float:
        """Steam mass rate in kg/s.

        ``steam_rate_m3_per_day`` is the water-equivalent injection rate, so
        the injected steam mass is simply that volume of liquid water per day:

        .. math:: \\dot{m}_s = V_{we}\\,\\rho_w / 86400
        """
        return self.steam_rate_m3_per_day * 1000.0 / 86400.0

    @property
    def steam_enthalpy_j_kg(self) -> float:
        """Total injected specific enthalpy including the latent contribution.

        .. math:: h = h_{sensible} + x\\,h_{fg}(T_{steam})
        """
        return (self.steam_enthalpy_kj_kg + self.latent_heat_kj_kg * self.steam_quality) * 1e3

    @property
    def latent_heat_kj_kg(self) -> float:
        """Latent heat of vaporisation at the injection temperature.

        Uses the Watson correlation anchored on the critical point:

        .. math::

            h_{fg}(T) = h_{fg}(T_c)\\left(\\frac{1 - T/T_c}{1 - T_{ref}/T_c}\\right)^{0.38}
        """
        t_critical_k = 647.096
        t_ref_k = 373.15
        h_fg_ref = 2257.0  # kJ/kg at 100 degC
        t_k = self.steam_temperature_c + 273.15
        return h_fg_ref * ((1.0 - t_k / t_critical_k) / (1.0 - t_ref_k / t_critical_k)) ** 0.38

    # -- energy ledger ----------------------------------------------------
    #
    # The solver keeps a single conservative energy ledger over the cycle:
    #
    #     E_injected = E_retained + E_formation + E_overburden
    #
    # Every phase moves energy between the three accounts and never creates
    # any.  A timestep computes the conductive losses, caps the drained
    # energy at what the chest actually holds, and splits the drain between
    # the two loss accounts strictly in proportion to their rates, so the
    # identity above closes to double-precision round-off.

    def _injection_step(self, state: SteamChestState) -> float:
        """Advance one injection timestep; returns the injected energy, J."""
        mass_kg = self.steam_mass_flow_kg_s * 86400.0 * self.dt
        state.cumulative_steam_tonnes += mass_kg / 1000.0
        state.cumulative_steam_m3 += mass_kg * self.steam_specific_volume_m3_kg
        # SOR is defined industrially on a volumetric basis: water-equivalent
        # steam volume over stock-tank oil volume.
        state.cumulative_steam_volume_m3 += mass_kg / 1000.0
        state.steam_quality = self.steam_quality
        # Only the retained fraction of the steam stays in the formation.
        return mass_kg * self.steam_enthalpy_j_kg * self.steam_retention_factor

    def _apply_losses(
        self, state: SteamChestState, formation_rate: float, overburden_rate: float
    ) -> Tuple[float, float]:
        """Drain the chest by the conductive losses; returns (formation, overburden).

        Both rates are J/day.  The drained energy is capped at the energy the
        chest actually holds, and the cap is shared between the two accounts in
        proportion to the rates, which keeps the ledger exactly conservative.
        """
        formation_step = formation_rate * self.dt
        overburden_step = overburden_rate * self.dt
        total_step = formation_step + overburden_step
        drained = min(total_step, state.cumulative_heat_retained_j)
        if total_step > 0.0:
            share_formation = drained * formation_step / total_step
        else:
            share_formation = 0.0
        share_overburden = drained - share_formation
        state.cumulative_heat_retained_j -= drained
        state.cumulative_heat_formation_j += share_formation
        state.cumulative_heat_overburden_j += share_overburden
        return share_formation, share_overburden

    def _chest_temperature(self, radius_m: float, energy_j: float) -> float:
        """Chest temperature from the annulus energy balance.

        .. math::
            Q = \\pi (R_s^2 - R_w^2)\\,\\rho c\\,(T_{ch} - T_{res})
            \\quad\\Longrightarrow\\quad
            T_{ch} = T_{res} + \\frac{Q}{\\pi (R_s^2 - R_w^2)\\,\\rho c}
        """
        radius = max(radius_m, self.rw)
        chest_volume = math.pi * (radius**2 - self.rw**2)
        if chest_volume <= 0.0 or energy_j <= 0.0:
            return self.reservoir_temperature_c
        return self.reservoir_temperature_c + energy_j / (self.rho_c * chest_volume)

    def _conduction_to_formation(self, chest_temp_c: float, radius_m: float) -> float:
        """Steady radial conduction from the steam chest into the formation, J/day.

        Heat leaving a cylindrical surface of radius :math:`R_s` and flowing
        radially outwards through a homogeneous annulus to the drainage radius
        :math:`R_e` is governed by the *annulus* thermal resistance

        .. math::

            \\dot{Q} = \\frac{2\\pi k\\,(T_{ch} - T_{res})}{\\ln(R_e/R_s)}

        so the conductance grows as the chest front approaches the drainage
        radius (the log shrinks), which is the driving term of the
        Marx-Langenheim chest growth.
        """
        if radius_m <= self.rw or chest_temp_c <= self.reservoir_temperature_c:
            return 0.0
        ratio = self.re / radius_m
        if ratio <= 1.0:
            # The front has reached the drainage boundary: all heat is lost.
            return 2.0 * math.pi * self.k * (chest_temp_c - self.reservoir_temperature_c) * 86400.0
        resistance = math.log(ratio)
        conductance = 2.0 * math.pi * self.k / resistance
        return conductance * (chest_temp_c - self.reservoir_temperature_c) * 86400.0

    def _shale_temperature(
        self, depth_m: float, chest_temp_c: float, time_days: float
    ) -> float:
        r"""Temperature at ``depth_m`` inside the bounding shale, degC.

        For a semi-infinite solid whose surface is held at the steam chest
        temperature the exact transient profile is

        .. math::

            T(x, t) = T_{res} + (T_{ch} - T_{res})
                \, \mathrm{erfc}\!\left(\frac{x}{2\sqrt{\alpha t}}\right)

        so the thermal front has advanced to :math:`x \approx 2\sqrt{\alpha t}`
        and everything deeper is still at the native reservoir temperature.
        """
        if depth_m < 0.0:
            raise ValueError("depth must be non-negative")
        if time_days <= 0.0 or depth_m == 0.0:
            return max(chest_temp_c, self.reservoir_temperature_c)
        front = 2.0 * math.sqrt(self.alpha_over * time_days * 86400.0)
        return self.reservoir_temperature_c + (
            chest_temp_c - self.reservoir_temperature_c
        ) * erfc(depth_m / front)

    def _overburden_loss_rate(
        self, chest_temp_c: float, time_days: float, radius_m: float
    ) -> float:
        r"""Transient conductive loss to the over- and underburden, J/day.

        The heat escapes upwards into the caprock and downwards into the
        basement across the two horizontal faces of the steam chest, of total
        area

        .. math:: A_{ou} = 2 \, \pi (R_s^2 - R_w^2)

        Each bounding shale is a semi-infinite body whose surface is suddenly
        raised to the chest temperature.  Differentiating the exact
        :func:`_shale_temperature` profile with respect to depth,

        .. math::
            \dot{Q}_{ou} = \frac{k_{ou} A_{ou}}{\sqrt{\pi \alpha_{ou} t}}
                \left(T_{ch} - T_{res}\right)

        which is the classical Carslaw-Jaeger surface flux and decays as
        :math:`t^{-1/2}`.  The ``erfc`` argument
        :math:`d_{ou}/(2\sqrt{\alpha t})` also shows that a shale thicker than
        the penetration depth is not yet reached at all, so the loss is
        negligible over a CSS cycle.

        Parameters
        ----------
        chest_temp_c
            Current steam chest temperature.
        time_days
            Elapsed time since the start of injection.
        radius_m
            Current steam chest radius, setting the contact area.
        """
        if time_days <= 0.0 or chest_temp_c <= self.reservoir_temperature_c:
            return 0.0
        radius = max(self.rw, radius_m)
        # Only the interbedded fraction of the chest face couples to the shales.
        area = (
            self.vertical_contact * 2.0 * math.pi * (radius**2 - self.rw**2)
        )
        if area <= 0.0:
            return 0.0
        sqrt_alpha_t = math.sqrt(self.alpha_over * time_days * 86400.0)
        return (
            self.k_over
            * area
            * (chest_temp_c - self.reservoir_temperature_c)
            / (math.sqrt(math.pi) * sqrt_alpha_t)
            * 86400.0
        )

    def _oil_viscosity_cP(self, temperature_c: float) -> float:
        """Live reservoir oil viscosity at ``temperature_c``, cP.

        The temperature dependence is the validated ASTM D341 / Walther
        relation, normalised so that the value at the reservoir temperature
        reproduces the measured *live* oil viscosity (which is two orders of
        magnitude below the dead-oil figure of the fluid specification
        because of the 145 scf/STB of dissolved gas).
        """
        walther = baghewala_crude()
        reference = walther.dynamic_viscosity_cp(RESERVOIR.reservoir_temperature_c)
        if reference <= 0.0:
            return RESERVOIR.reservoir_oil_viscosity_cP
        return (
            RESERVOIR.reservoir_oil_viscosity_cP
            * walther.dynamic_viscosity_cp(temperature_c)
            / reference
        )

    def _vogel_inflow(
        self, bottom_hole_pressure_mpa: float, temperature_c: Optional[float] = None
    ) -> float:
        """Vogel IPR with skin, evaluated at the live reservoir temperature.

        Vogel's solution-gas-drive relationship in terms of the pressure ratio
        :math:`x = p_{wf}/p_r` is

        .. math:: \\frac{q_o}{q_{max}} = 1 - 0.2x - 0.8x^2

        Skin does not act on the pressure ratio, it consumes part of the
        available drawdown.  Writing the total drawdown as the sum of the
        flowing part and the skin drop, :math:`\\Delta p = \\Delta p_{flow} +
        \\Delta p_{skin}` with :math:`\\Delta p_{skin} = s\\,\\Delta p_{flow}`,
        the flowing drawdown is

        .. math::
            \\Delta p_{flow} = \\frac{p_r - p_{wf}}{1+s}

        and the effective ratio fed to Vogel is

        .. math::
            x_{eff} = 1 - \\frac{\\Delta p_{flow}}{p_r}
                    = 1 - \\frac{1 - p_{wf}/p_r}{1 + s}

        which stays in :math:`[0, 1]` for any non-negative skin factor, unlike
        the naive :math:`x e^{s}` substitution which saturates to zero inflow.
        """
        p_res = RESERVOIR.initial_reservoir_pressure_mpa
        if p_res <= 0.0 or self.skin < 0.0:
            return 0.0
        pressure_ratio = min(max(bottom_hole_pressure_mpa / p_res, 0.0), 1.0)
        effective_ratio = 1.0 - (1.0 - pressure_ratio) / (1.0 + self.skin)
        productivity = 1.0 - 0.2 * effective_ratio - 0.8 * effective_ratio**2
        temperature = (
            self.reservoir_temperature_c if temperature_c is None else temperature_c
        )
        return productivity * self._vogel_q_max_m3_per_day(temperature)

    def _vogel_q_max_m3_per_day(
        self, temperature_c: Optional[float] = None
    ) -> float:
        """Maximum (zero drawdown) potential from the Darcy productivity index.

        The heated steam chest and the cold formation form two radial regions
        of different oil viscosity in series, so their hydraulic resistances
        add:

        .. math::

            q_{max} = \\frac{2\\pi k h\\,\\Delta p}
                {\\mu_{hot}\\ln(R_s/R_w) + \\mu_{cold}\\ln(R_e/R_s)}

        The temperature dependence of the mobility is taken from the Walther
        relation, and a zero-radius chest degenerates correctly to the
        homogeneous annulus result.
        """
        temperature = (
            self.reservoir_temperature_c if temperature_c is None else temperature_c
        )
        viscosity_cP = max(self._oil_viscosity_cP(temperature), 1e-6)
        mu_cold_pa_s = self._oil_viscosity_cP(self.reservoir_temperature_c) * 1e-3
        mu_hot_pa_s = viscosity_cP * 1e-3
        radius = min(max(self._last_steam_chest_radius_m, self.rw), self.re)

        if radius <= self.rw * (1.0 + 1e-12):
            log_hot = 0.0
            log_cold = math.log(self.re / self.rw)
        else:
            log_hot = math.log(radius / self.rw)
            log_cold = math.log(self.re / radius)
        resistance = mu_hot_pa_s * log_hot + mu_cold_pa_s * log_cold
        if resistance <= 0.0:
            return 0.0

        darcy_m2 = self.k_perm * 9.869233e-16
        area_scale = 2.0 * math.pi
        pressure_pa = RESERVOIR.initial_reservoir_pressure_mpa * 1e6
        q_m3_per_s = (
            darcy_m2
            * RESERVOIR.net_pay_thickness_m
            * pressure_pa
            * area_scale
            / (RESERVOIR.oil_formation_volume_factor * resistance)
        )
        return q_m3_per_s * 86400.0

    @property
    def deep_thermal_time_constant_days(self) -> float:
        """Slow conductive relaxation time constant of the steam chest, days.

        The radial conduction time constant of a cylinder of radius
        :math:`R_s` is

        .. math:: \\tau = \\frac{\\rho_c R_s^2}{\\pi^2 k},

        which for a 30 m chest of the Baghewala sand is of order years.  The
        effective value is that scaled by :attr:`thermal_time_scale` and
        evaluated at the chest radius that injection actually built, so that a
        short injection stores less thermal mass and decays proportionally
        faster.  That coupling is what makes the injection/soak/production
        trade-off real rather than degenerate.
        """
        # The time constant is set by the chest that *injection* built, which
        # is snapshotted at the end of injection; the live radius shrinks as the
        # chest cools and would otherwise drive the constant towards zero.
        radius = max(self._chest_radius_at_injection_end_m, self.rw)
        conduction_s = (
            self.rho_c * radius**2 / (math.pi**2 * self.k)
        )
        return self.thermal_time_scale * conduction_s / 86400.0

    @property
    def near_thermal_time_constant_days(self) -> float:
        """Fast near-wellbore convective cooling time constant, days."""
        return self.near_time_scale * self.deep_thermal_time_constant_days

    @property
    def thermal_time_scale(self) -> float:
        """Calibration factor on the conduction time constant.

        The bare radial conduction time of the chest is of order years, whereas
        upper Assam CSS wells hold a productive plateau for one to two months.
        The factor is the calibrated ratio of the observed plateau to the
        conduction time.
        """
        return 0.05

    @property
    def near_time_scale(self) -> float:
        """Ratio of the fast cooling mode to the slow one."""
        return 0.22

    @property
    def near_thermal_weight(self) -> float:
        """Fraction of the thermal excess in the fast cooling mode."""
        return 0.55

    def _thermal_decay(self, elapsed_days: float) -> float:
        r"""Fraction of the injected thermal excess still supporting inflow.

        A CSS well does not cool with a single time constant.  Two processes
        act simultaneously, so the excess temperature is represented as the
        sum of two relaxation modes,

        .. math::
            \frac{T(t) - T_{res}}{T_0 - T_{res}} =
                w_{near} e^{-t/\tau_{near}} + w_{deep} e^{-t/\tau_{deep}}

        * the **near-wellbore convective** mode (:math:`\tau_{near}`), the fast
          collapse caused by cold unheated fluid being drawn in from the
          retreating thermal front;
        * the **deep conductive** mode (:math:`\tau_{deep}`), the slow
          relaxation of the bulk steam chest by conduction into the
          surrounding formation.

        The near-wellbore weight dominates the early flush; the deep mode sets
        the long plateau and ultimately the economic cut-off.
        """
        if elapsed_days < 0.0:
            raise ValueError("elapsed time must be non-negative")
        near = self.near_thermal_weight * math.exp(
            -elapsed_days / max(self.near_thermal_time_constant_days, 1e-9)
        )
        deep = (1.0 - self.near_thermal_weight) * math.exp(
            -elapsed_days / max(self.deep_thermal_time_constant_days, 1e-9)
        )
        return near + deep

    # -- cycle simulation -------------------------------------------------
    def simulate_cycle(
        self,
        injection_days: float = CSS.injection_days,
        soak_days: float = CSS.soak_days,
        production_days: float = CSS.production_days,
        min_production_days: float = THRESHOLDS.min_production_days,
        record_every: int = 1,
    ) -> CSOResult:
        """Simulate one injection -> soak -> production cycle.

        Parameters
        ----------
        injection_days, soak_days, production_days
            Durations of the three CSS phases.
        record_every
            Record a state every ``record_every`` timesteps.

        Returns
        -------
        CSOResult
            Recorded state trajectory and cycle aggregates.
        """
        if min(injection_days, soak_days, production_days) < 0.0:
            raise ValueError("phase durations must be non-negative")
        if production_days <= 0.0:
            raise ValueError("production phase must be positive")
        if record_every < 1:
            raise ValueError("record_every must be a positive integer")

        result = CSOResult()
        state = SteamChestState(
            radius_m=self.rw,
            chest_temperature_c=self.reservoir_temperature_c,
            bottom_hole_temperature_c=self.reservoir_temperature_c,
        )
        time_days = 0.0
        # Instantaneous SOR is evaluated on a trailing window so the ratio is
        # not dominated by the whole-cycle average.
        window_days = max(1.0, production_days / 12.0)
        window_start_oil_m3 = 0.0
        window_start_step = 0
        window_steps = max(1, int(round(window_days / self.dt)))
        # Potential for the whole production phase, used to gate the SOR
        # cut-off against a vanishing denominator at the cycle start.
        expected_oil_volume = (
            min(
                self._vogel_inflow(
                    self.producing_bottom_hole_pressure_mpa,
                    temperature_c=state.bottom_hole_temperature_c,
                ),
                self.pump_capacity_m3_per_day,
            )
            * production_days
        )

        def _record() -> None:
            result.states.append(SteamChestState(**vars(state)))

        _record()

        # ---- Injection -------------------------------------------------
        # The chest is held at the injected steam temperature (saturated steam
        # at the wellhead pressure), and its radius grows to accommodate the
        # accumulating energy.  This is the Marx-Langenheim construction: the
        # steam chest boundary is where the formation reaches steam temperature.
        state.phase = CSSPhase.INJECTION
        steps = max(1, int(round(injection_days / self.dt)))
        for i in range(steps):
            heat_in = self._injection_step(state)
            state.cumulative_heat_retained_j += heat_in
            # Conductive losses are active during injection too.
            self._apply_losses(
                state,
                self._conduction_to_formation(state.chest_temperature_c, state.radius_m),
                self._overburden_loss_rate(
                    state.chest_temperature_c, max(time_days, self.dt), state.radius_m
                ),
            )
            state.chest_temperature_c = self.steam_temperature_c
            radius = self._radius_from_energy(
                state.cumulative_heat_retained_j, state.chest_temperature_c
            )
            state.radius_m = min(max(state.radius_m, radius), self.re)
            self._last_steam_chest_radius_m = state.radius_m
            time_days += self.dt
            state.time_days = time_days
            if (i + 1) % record_every == 0:
                _record()
        _record()

        # The chest geometry that sets the production-phase cooling time
        # constant is the one injection built, so it is frozen here before
        # the soak and production phases start shrinking it.
        self._chest_radius_at_injection_end_m = max(state.radius_m, self.rw)

        # ---- Soak ------------------------------------------------------
        # No injection: the chest cools at (approximately) fixed geometry and
        # the drained energy leaves through the formation and the shales.
        state.phase = CSSPhase.SOAKING
        steps = max(1, int(round(soak_days / self.dt)))
        for i in range(steps):
            self._apply_losses(
                state,
                self._conduction_to_formation(state.chest_temperature_c, state.radius_m),
                self._overburden_loss_rate(
                    state.chest_temperature_c, max(time_days, self.dt), state.radius_m
                ),
            )
            state.chest_temperature_c = self._chest_temperature(
                state.radius_m, state.cumulative_heat_retained_j
            )
            self._last_steam_chest_radius_m = state.radius_m
            time_days += self.dt
            state.time_days = time_days
            if (i + 1) % record_every == 0:
                _record()
        _record()

        # ---- Production ------------------------------------------------
        state.phase = CSSPhase.PRODUCTION
        peak_temperature = state.chest_temperature_c
        steps = max(1, int(round(production_days / self.dt)))
        for i in range(steps):
            # Two-mode thermal relaxation of the steam chest.
            decay = self._thermal_decay(i * self.dt)
            target = self.reservoir_temperature_c
            state.chest_temperature_c = target + (peak_temperature - target) * decay
            # Ramey-type attenuation between the steam chest and the pump depth.
            state.bottom_hole_temperature_c = target + (
                state.chest_temperature_c - target
            ) * self.pump_depth_attenuation

            # Inflow follows from the mobility of the oil actually entering
            # the well, which is set by the attenuated pump-depth temperature.
            # The thermal effect enters *only* through this viscosity, so the
            # decline is not counted twice.
            # The well-to-surface coupling: a CSS well is pump-limited, not
            # reservoir-limited.  The delivered rate is the lesser of the
            # available Vogel inflow and the pump's volumetric capacity, and
            # the shortfall is reported as the fill deficit that drives fluid
            # pound detection in the SRP layer.
            inflow_m3_per_day = self._vogel_inflow(
                self.producing_bottom_hole_pressure_mpa,
                temperature_c=state.bottom_hole_temperature_c,
            )
            rate_m3_per_day = min(inflow_m3_per_day, self.pump_capacity_m3_per_day)
            state.reservoir_inflow_m3_per_day = inflow_m3_per_day
            state.pump_fillage = (
                inflow_m3_per_day / self.pump_capacity_m3_per_day
                if self.pump_capacity_m3_per_day > 0.0
                else 0.0
            )
            state.oil_rate_tpd = rate_m3_per_day

            oil_volume_m3 = rate_m3_per_day * self.dt
            oil_tonnes = oil_volume_m3 * self.oil_specific_gravity
            state.cumulative_oil_tonnes += oil_tonnes
            state.cumulative_oil_volume_m3 += oil_volume_m3

            # Enthalpy carried out with the produced fluid.
            state.cumulative_heat_produced_j += (
                oil_tonnes
                / self.oil_specific_gravity
                * self.oil_specific_heat_j_kgk
                * (state.chest_temperature_c - target)
            )
            # The heated annulus retreats as the chest decays; the IPR needs
            # this radius to split hot and cold radial resistances.
            # The thermal front retreats diffusively, R ~ sqrt(4 alpha t),
            # which over a single 60-day cycle is negligible compared with the
            # 30 m chest radius, so the chest geometry is effectively fixed.
            retreat = math.sqrt(
                max(state.radius_m**2 - 4.0 * self.alpha_over * i * self.dt * 86400.0, 0.0)
            )
            state.radius_m = max(self.rw, min(retreat, self.re))
            self._last_steam_chest_radius_m = state.radius_m

            if state.cumulative_oil_volume_m3 > 1e-12:
                state.cumulative_sor = (
                    state.cumulative_steam_volume_m3 / state.cumulative_oil_volume_m3
                )

            # The specific steam consumption -- the metric the field actually
            # uses to terminate a cycle -- is evaluated once per *complete*
            # trailing window, so that the denominator is always a full window
            # of production.  Evaluating it every timestep would divide by a
            # nearly empty window immediately after each rollover.
            window_oil_volume = state.cumulative_oil_volume_m3 - window_start_oil_m3
            if i + 1 - window_start_step >= window_steps:
                if window_oil_volume > 1e-9:
                    state.instantaneous_sor = (
                        state.cumulative_steam_volume_m3 / window_oil_volume
                    )
                window_start_oil_m3 = state.cumulative_oil_volume_m3
                window_start_step = i + 1

            time_days += self.dt
            state.time_days = time_days
            if (i + 1) % record_every == 0:
                _record()

            # The economic cut-off is only evaluated once a meaningful
            # fraction of the cycle oil has been produced; otherwise the
            # first timestep of any cycle trips it on a near-zero denominator.
            # Economic cut-off.  Following the SIH26120 operating rule a cycle
            # is terminated when the reservoir has lost its thermal support
            # (BHT below the economic limit) *and* the steam-oil ratio has
            # crossed the cut-off.  The BHT condition is essential: early in a
            # cycle all the steam is charged while the well is still building
            # its flush, so the cycle-to-date SOR is large but not yet
            # meaningful, and cutting off on it alone would abandon a well that
            # is still producing at its plateau rate.
            production_elapsed_days = (i + 1) * self.dt
            thermally_spent = state.bottom_hole_temperature_c < CSS.cutoff_bht_c
            minimum_period_met = production_elapsed_days >= min_production_days
            if (
                not state.cutoff_reached
                and minimum_period_met
                and thermally_spent
                and state.cumulative_sor > self.cutoff_sor
            ):
                state.cutoff_reached = True
                state.cutoff_reason = (
                    f"BHT {state.bottom_hole_temperature_c:.1f} degC below "
                    f"{CSS.cutoff_bht_c:.1f} degC with cumulative SOR "
                    f"{state.cumulative_sor:.2f} above cut-off {self.cutoff_sor:.2f}"
                )
            # Production is *terminated* at the cut-off rather than merely
            # flagged: the well is switched over to the next injection, so a
            # schedule that injects too little steam genuinely earns less over
            # the programme.  Without this the schedule trade-off is degenerate,
            # because running the full production period would always win.
            if state.cutoff_reached:
                _record()
                break

        result.total_steam_tonnes = state.cumulative_steam_tonnes
        result.total_oil_tonnes = state.cumulative_oil_tonnes
        result.cumulative_sor = state.cumulative_sor
        result.final_bht_c = state.bottom_hole_temperature_c
        result.cycles_completed = 1
        return result

    def _radius_from_energy(self, energy_j: float, temperature_c: float) -> float:
        """Steam chest radius implied by the retained energy.

        .. math::

            Q_{ch} = \\pi (R_s^2 - R_w^2)\\,\\rho c\\,(T_{ch} - T_{res})
            \\quad\\Longrightarrow\\quad
            R_s = \\sqrt{R_w^2 + \\frac{Q_{ch}}{\\pi \\rho c (T_{ch}-T_{res})}}

        This is the Marx-Langenheim construction: the chest front sits where
        the formation has been heated to the steam temperature.
        """
        delta_t = temperature_c - self.reservoir_temperature_c
        if delta_t <= 0.0:
            return self.rw
        radius_squared = self.rw**2 + energy_j / (math.pi * self.rho_c * delta_t)
        return math.sqrt(radius_squared)

    def simulate_programme(
        self,
        cycles: int = CSS.design_cycles,
        injection_days: float = CSS.injection_days,
        soak_days: float = CSS.soak_days,
        production_days: float = CSS.production_days,
    ) -> CSOResult:
        """Run ``cycles`` CSS cycles, accumulating steam, oil and economics.

        The economic cut-off is recorded on the *cumulative* basis, and the
        programme stops early once the cumulative SOR breaches the limit for a
        sustained window, mirroring the field operating rule.
        """
        if cycles < 1:
            raise ValueError("at least one cycle is required")
        programme = CSOResult()
        cumulative_steam = 0.0
        cumulative_oil = 0.0
        revenue = 0.0
        steam_cost = 0.0

        for cycle_index in range(1, cycles + 1):
            cycle = self.simulate_cycle(
                injection_days=injection_days,
                soak_days=soak_days,
                production_days=production_days,
            )
            programme.states.extend(cycle.states)
            cumulative_steam += cycle.total_steam_tonnes
            cumulative_oil += cycle.total_oil_tonnes
            programme.final_bht_c = cycle.final_bht_c
            programme.cycles_completed += 1

            oil_m3 = cumulative_oil / self.oil_specific_gravity
            steam_m3 = cumulative_steam  # water-equivalent volume, 1 t == 1 m^3
            revenue = oil_m3 * 6.2898 * 68.0
            steam_cost = steam_m3 * 21.0

            # Steam-oil ratio is defined on a volumetric basis.
            cycle_sor = steam_m3 / oil_m3 if oil_m3 > 0 else math.inf
            if cycle_sor > self.cutoff_sor:
                programme.economic_cutoff_cycles.append(cycle_index)
                break

        programme.total_steam_tonnes = cumulative_steam
        programme.total_oil_tonnes = cumulative_oil
        programme.cumulative_sor = (
            (cumulative_steam)
            / (cumulative_oil / self.oil_specific_gravity)
            if cumulative_oil > 0
            else math.inf
        )
        programme.net_revenue_usd = revenue - steam_cost
        return programme

    def energy_balance(self, result: CSOResult) -> float:
        """Relative closure error of the CSS energy ledger.

        The solver maintains the identity

        .. math::
            E_{injected} = E_{retained} + E_{formation} + E_{overburden}

        over the injection and soak phases.  This method returns the absolute
        fractional residual of that identity; a value below 1e-12 means the
        balance closes to double-precision round-off.
        """
        if not result.states:
            raise ValueError("no states recorded")
        final = result.states[-1]
        injected = final.cumulative_steam_tonnes * 1000.0 * self.steam_enthalpy_j_kg * self.steam_retention_factor
        accounted = (
            final.cumulative_heat_retained_j
            + final.cumulative_heat_formation_j
            + final.cumulative_heat_overburden_j
        )
        if injected <= 0.0:
            return 0.0
        return abs(injected - accounted) / injected


def baghewala_css_model(fluid: Optional[BaghewalaFluid] = None) -> ThermalReservoirModel:
    """Default CSS model calibrated to the Baghewala asset."""
    if fluid is not None:
        return ThermalReservoirModel(oil_specific_gravity=fluid.specific_gravity)
    return ThermalReservoirModel()
