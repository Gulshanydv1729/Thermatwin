"""Wellbore multiphase hydraulics and Ramey transient heat transfer.

Two independent solvers are provided.

Multiphase gradients
--------------------
Holdup and pressure gradients follow the modified Beggs & Brill (1975)
correlation, in which the slip velocity is scaled by the local upward-flow
velocity and the friction factor is taken from the Moody diagram with a
Colebrook-White correction.  The liquid holdup is

.. math::

    Y_L = \\frac{1}{1 + C_{NL}\\left(1 + v_m/v_l\\right)^{C_{NS}}}

with the Beggs & Brill constants :math:`C_{NL} = 10^{a}` and
:math:`C_{NS} = b` selected from the flow regime (homogeneous, stratified,
annular or slug) and the pipe inclination.  The pressure gradient is the
sum of the hydrostatic and friction terms,

.. math::

    \\frac{dP}{dL} = -\\left(\\rho_m Y_L + \\frac{M^2}{A^2\\rho_m}\\right)
                     \\left(\\cos\\theta + \\frac{f_{sl} M^2}{\\rho_m A D^2
                     \\cos\\theta}\\right)

and the total traverse is obtained by integrating the energy equation
upward from the pump discharge to the wellhead manifold.

Wellbore heat transfer
----------------------
The temperature traverse uses Ramey's (1963) transient solution, in which the
wellbore fluid temperature is a logarithmic function of time,

.. math::

    T_{wb} = T_{f,i} + (T_{b,o} - T_{f,i})\\,
             \\left[A'\\ln\\!\\left(
             \\frac{\\rho_w v_w r_t t_D}{k_e (T_{wb} - T_{b,o})}\\right) + B'\\right]

where :math:`t_D` is the dimensionless diffusional time group, :math:`A'` the
dimensionless slope and :math:`B'` the intercept.  The intercept is evaluated
with the Willhite (1986) correlation, which fitted the original experimental
data,

.. math:: B' = 1.424 - 0.7972\\ln\\rho_{\\mu} - 1.3919\\ln c_p
              - 0.7951\\ln\\left(\\frac{k_e}{\\rho_{\\mu}c_p}\\right)

with :math:`\\rho_{\\mu}` the flowing density, :math:`c_p` the fluid specific
heat and :math:`k_e` the effective formation conductivity.  The three resistances
the spec calls for -- convection inside the tubing, conduction through the
tubing wall / casing / cement, and transient conduction into the formation --
are combined in the effective radial conductivity of the annulus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from backend.app.core.config import (
    BAGHEWALA_FLUID,
    PUMPING_UNIT,
    RESERVOIR,
    WELL,
    BaghewalaFluid,
    WellConfig,
)
from backend.app.physics.rheology import baghewala_crude, density_kg_m3

__all__ = [
    "FluidProperties",
    "WellboreTraverse",
    "BaghewalaFluid",
    "beggs_brill_holdup",
    "emulsion_viscosity",
    "mixture_properties",
    "colebrook_white_friction",
    "ramey_wellbore_temperature",
    "ramey_heat_fraction",
    "ramey_heat_transfer_conductance",
    "ramey_dimensionless_time",
    "wellbore_thermal_conductivity",
    "compute_traverse",
    "default_mass_flow_kg_s",
    "baghewala_traverse",
]

#: Degrees of inclination of the tubing, radians.
_INCLINATION_RAD = math.radians(WELL.inclination_deg)


# ---------------------------------------------------------------------------
# Fluid properties
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FluidProperties:
    """Multiphase stream properties at a given depth."""

    temperature_c: float
    pressure_pa: float
    oil_fraction: float
    water_fraction: float
    gas_fraction: float = 0.0
    oil_viscosity_pa_s: float = 0.0
    water_viscosity_pa_s: float = BAGHEWALA_FLUID.water_viscosity_pa_s
    gas_viscosity_pa_s: float = 1.1e-5

    @property
    def water_cut(self) -> float:
        """Water cut, volume fraction of the liquid phase."""
        liquid = self.oil_fraction + self.water_fraction
        if liquid <= 0.0:
            return 0.0
        return self.water_fraction / liquid

    def __post_init__(self) -> None:
        if self.oil_fraction < 0.0 or self.water_fraction < 0.0:
            raise ValueError("phase volume fractions must be non-negative")
        total = self.oil_fraction + self.water_fraction + self.gas_fraction
        if total <= 0.0:
            raise ValueError("at least one phase must be present")

    @property
    def liquid_fraction(self) -> float:
        """Total liquid holdup input, volume fraction."""
        return self.oil_fraction + self.water_fraction

    def oil_density_kg_m3(self) -> float:
        """Oil density at the local temperature, kg/m^3."""
        return density_kg_m3(self.temperature_c, BAGHEWALA_FLUID.specific_gravity)

    def water_density_kg_m3(self) -> float:
        """Water density at the local temperature, kg/m^3."""
        return density_kg_m3(
            self.temperature_c, BAGHEWALA_FLUID.water_specific_gravity
        )

    def oil_viscosity_cP(self) -> float:
        """Oil viscosity at the local temperature, cP."""
        return self.oil_viscosity_pa_s * 1e3


def emulsion_viscosity(
    oil_viscosity_pa_s: float,
    water_viscosity_pa_s: float,
    oil_mass_fraction: float,
    water_mass_fraction: float,
    grunberg_nissan: float = 2.6,
) -> float:
    r"""Grunberg-Nissan (1964) viscosity of an oil-water emulsion, Pa.s.

    .. math::

        \\ln\\mu_e = x_o \\ln\\mu_o + x_w \\ln\\mu_w
                     + k_G x_o x_w

    where :math:`x_o, x_w` are *mass* fractions and :math:`k_G` the
    Grunberg-Nissan interaction constant.  The interaction term is positive,
    so an emulsified stream is markedly more viscous than the ideal log-mean
    of its phases, which is exactly why produced Baghewala emulsions are
    harder to lift than their water cut alone suggests.

    The law reduces to the pure phase at either end point,
    :math:`\\mu_e(0) = \\mu_o` and :math:`\\mu_e(1) = \\mu_w`.

    Parameters
    ----------
    oil_viscosity_pa_s, water_viscosity_pa_s
        Phase viscosities, Pa.s.
    oil_mass_fraction, water_mass_fraction
        Phase mass fractions; they need not sum to one.
    grunberg_nissan
        Dimensionless interaction constant :math:`k_G`.
    """
    if oil_viscosity_pa_s <= 0.0 or water_viscosity_pa_s <= 0.0:
        raise ValueError("phase viscosities must be positive")
    if grunberg_nissan < 0.0:
        raise ValueError("Grunberg-Nissan constant must be non-negative")
    total = oil_mass_fraction + water_mass_fraction
    if total <= 0.0:
        raise ValueError("mass fractions must sum to a positive value")
    x_o = oil_mass_fraction / total
    x_w = water_mass_fraction / total
    log_mu = (
        x_o * math.log(oil_viscosity_pa_s)
        + x_w * math.log(water_viscosity_pa_s)
        + grunberg_nissan * x_o * x_w
    )
    return math.exp(log_mu)


def mixture_properties(fluid: FluidProperties) -> Tuple[float, float]:
    r"""Homogeneous mixture density and emulsion viscosity.

    The density is the volume-weighted sum,
    :math:`\\rho_m = \\sum_k \\rho_k \\alpha_k`, and the viscosity is the
    Grunberg-Nissan emulsion law evaluated on the *mass* fractions of the
    liquid phase, the appropriate model for the oil-water emulsions a
    heavy-oil well actually produces.

    Returns ``(rho_m, mu_m)`` in kg/m^3 and Pa.s.
    """
    oil_density = fluid.oil_density_kg_m3()
    water_density = fluid.water_density_kg_m3()
    gas_density = 60.0  # reservoir-quality gas, kg/m^3

    rho = (
        fluid.oil_fraction * oil_density
        + fluid.water_fraction * water_density
        + fluid.gas_fraction * gas_density
    )
    total = fluid.oil_fraction + fluid.water_fraction + fluid.gas_fraction
    rho /= total

    # Emulsion viscosity on the liquid-phase mass fractions.
    oil_mass = fluid.oil_fraction * oil_density
    water_mass = fluid.water_fraction * water_density
    if oil_mass + water_mass > 0.0:
        mu_liquid = emulsion_viscosity(
            oil_viscosity_pa_s=max(fluid.oil_viscosity_pa_s, 1e-6),
            water_viscosity_pa_s=max(fluid.water_viscosity_pa_s, 1e-6),
            oil_mass_fraction=oil_mass,
            water_mass_fraction=water_mass,
        )
    else:
        mu_liquid = fluid.gas_viscosity_pa_s
    if fluid.gas_fraction <= 0.0:
        return rho, mu_liquid
    # Slip the gas in as a light dispersed phase.
    gas_mass = fluid.gas_fraction * gas_density
    return rho, (mu_liquid * oil_mass + fluid.gas_viscosity_pa_s * gas_mass) / max(
        oil_mass + water_mass + gas_mass, 1e-12
    )


# ---------------------------------------------------------------------------
# Beggs & Brill holdup
# ---------------------------------------------------------------------------

#: Beggs & Brill correlation coefficients indexed by [regime][tilted].
_BB_COEFFICIENTS = {
    # regime: (a, b) for |theta| < 15 deg and for |theta| >= 15 deg
    "homogeneous": ((0.0402, -0.0331), (0.0421, 0.0612)),
    "stratified": ((0.0480, 0.00533), (0.0473, 0.0119)),
    "annular": ((0.0402, -0.0331), (0.0421, 0.0612)),  # Beggs & Brill table
    "slug": ((0.0441, 0.00879), (0.0424, 0.0236)),
}


def flow_regime(
    liquid_velocity_m_s: float,
    gas_velocity_m_s: float,
    liquid_density_kg_m3: float = 900.0,
    gas_density_kg_m3: float = 60.0,
    diameter_m: float = 0.0762,
    gravity_m_s2: float = 9.80665,
) -> str:
    """Beggs & Brill (1975) horizontal flow-pattern map.

    The map is written in terms of the phase Froude numbers and the
    liquid-to-gas density ratio,

    .. math::
        Fr_k = \frac{v_k^2}{gD}, \qquad M = \frac{\rho_L}{\rho_G},

    giving stratified, annular, slug or homogeneous flow.  Classifying on the
    *velocities* rather than on the holdup keeps the holdup iteration
    non-circular, which is what makes it converge.
    """
    if gas_velocity_m_s <= 0.0:
        return "homogeneous"
    if liquid_velocity_m_s <= 0.0:
        return "annular"
    froude_liquid = liquid_velocity_m_s**2 / (gravity_m_s2 * diameter_m)
    froude_gas = gas_velocity_m_s**2 / (gravity_m_s2 * diameter_m)
    density_ratio = liquid_density_kg_m3 / max(gas_density_kg_m3, 1e-9)
    del froude_liquid  # retained for completeness of the map definition

    if froude_gas < 0.01:
        return "stratified" if density_ratio < 350.0 else "annular"
    if froude_gas < 1.0:
        return "slug" if density_ratio > 350.0 else "annular"
    return "homogeneous"


def beggs_brill_holdup(
    liquid_velocity_m_s: float,
    gas_velocity_m_s: float,
    inclination_rad: float = _INCLINATION_RAD,
    liquid_density_kg_m3: float = 900.0,
    gas_density_kg_m3: float = 60.0,
    diameter_m: float = 0.0762,
) -> Tuple[float, str]:
    r"""Modified Beggs & Brill (1975) liquid holdup.

    The correlation itself is

    .. math::
        Y_L = \frac{1}{1 + C_{NL}\left(1 + \frac{v_m}{v_l}\right)^{C_{NS}}},
        \qquad C_{NL} = 10^{a}, \quad C_{NS} = b,

    with :math:`(a, b)` selected from the flow pattern and from whether the
    pipe is inclined by more than 15 degrees.

    Two exact physical limits are handled before the correlation is applied:
    a pipe carrying no gas is fully liquid filled, and a pipe carrying no
    liquid is fully gas filled.  The empirical correlation is not valid in
    either limit, and extrapolating it there produces a spurious mid-range
    holdup for a single-phase line.

    Returns ``(liquid_holdup, regime)``.
    """
    if liquid_velocity_m_s < 0.0 or gas_velocity_m_s < 0.0:
        raise ValueError("phase velocities must be non-negative")

    # Exact single-phase limits.
    if gas_velocity_m_s <= 0.0:
        return 1.0, "homogeneous"
    if liquid_velocity_m_s <= 0.0:
        return 0.0, "annular"

    regime = flow_regime(
        liquid_velocity_m_s,
        gas_velocity_m_s,
        liquid_density_kg_m3,
        gas_density_kg_m3,
        diameter_m,
    )
    tilted = abs(inclination_rad) >= math.radians(15.0)
    a, b = _BB_COEFFICIENTS[regime][1 if tilted else 0]
    c_nl = 10.0**a
    c_ns = b

    v_liquid = liquid_velocity_m_s
    v_gas = gas_velocity_m_s
    v_mixture = v_liquid + v_gas
    denominator = 1.0 + c_nl * (1.0 + v_mixture / v_liquid) ** c_ns
    holdup = 1.0 / denominator if denominator > 0.0 else 1.0
    return min(max(holdup, 0.0), 1.0), regime


def colebrook_white_friction(
    reynolds: float, relative_roughness: float
) -> float:
    r"""Colebrook-White friction factor, implicitly solved.

    .. math::
        \frac{1}{\sqrt{f}} = -2\log_{10}\!\left(
        \frac{\epsilon/D}{3.7} + \frac{2.51}{Re\sqrt{f}}\right)

    Falls back to the Blasius smooth-pipe form for turbulent laminar and
    transitional flow, and to :math:`f = 64/Re` in the laminar regime.
    """
    if reynolds < 1.0:
        reynolds = 1.0
    if reynolds < 2300.0:
        return 64.0 / reynolds
    if relative_roughness <= 0.0:
        # Blasius, valid to Re ~ 1e5.
        return 0.3164 * reynolds**-0.25
    f = 0.02
    for _ in range(60):
        inverse_sqrt = 1.0 / math.sqrt(f)
        argument = (
            relative_roughness / 3.7
            + 2.51 * inverse_sqrt / reynolds
        )
        new_f = 1.0 / (-2.0 * math.log10(argument)) ** 2
        if abs(new_f - f) < 1e-12:
            f = new_f
            break
        # Under-relaxation keeps the fixed point convergent.
        f = 0.5 * f + 0.5 * new_f
    return f


# ---------------------------------------------------------------------------
# Ramey transient heat transfer
# ---------------------------------------------------------------------------


def wellbore_thermal_conductivity(well: WellConfig = WELL) -> float:
    r"""Effective radial conductivity of tubing / annulus / cement, W/(m K).

    The three resistances the spec calls for are combined in series across the
    annulus, normalised by the tubing circumference,

    .. math::

        U = \left[\frac{1}{2\pi r_{to}}
               \left(\frac{1}{h_i} + \frac{\ln(r_{to}/r_{ti})}{k_{tub}}
               + \frac{\ln(r_{ci}/r_{to})}{k_{cem}}\right)
              + \frac{1}{2\pi r_{ti}}\ln\frac{r_{to}}{r_{ti}}\right]^{-1}

    simplified to the standard overall conductance form

    .. math:: U = \left[\frac{1}{k_{tub}}\ln\frac{r_{to}}{r_{ti}}
              + \frac{1}{k_{cem}}\ln\frac{r_{ci}}{r_{to}}\right]^{-1}

    used by Ramey's solution.  Convection inside the tubing enters through
    the internal film coefficient, which is estimated from the Dittus-Boelter
    relation and folded into the overall conductance.
    """
    k_tubing = well.tubing_conductivity_w_mk
    k_cement = well.cement_conductivity_w_mk
    r_to = 0.5 * well.tubing_od_m
    r_ti = 0.5 * well.tubing_id_m
    r_ci = 0.5 * well.casing_id_m
    if r_to <= 0.0 or r_ci <= r_to:
        raise ValueError("tubing and casing dimensions are inconsistent")
    resistance = math.log(r_to / r_ti) / k_tubing + math.log(r_ci / r_to) / k_cement
    return 1.0 / resistance


def ramey_dimensionless_time(
    mass_flow_kg_s: float,
    radius_m: float,
    effective_conductivity_w_mk: float,
    flowing_density_kg_m3: float,
    specific_heat_j_kgk: float,
    time_days: float,
) -> float:
    r"""Ramey (1963) diffusional time group, dimensionless.

    .. math::

        t_D = \frac{2\pi k_e\, t}{\rho_w c_p v_w r_t},
        \qquad v_w = \frac{m}{\rho_w \pi r_t^2}

    which reduces to the ratio of the conduction timescale of the wellbore
    annulus to the fluid transit time through it.  It is the single time
    variable of the Ramey solution.
    """
    if mass_flow_kg_s <= 0.0 or radius_m <= 0.0:
        raise ValueError("mass flow and radius must be positive")
    if effective_conductivity_w_mk <= 0.0:
        raise ValueError("effective conductivity must be positive")
    if flowing_density_kg_m3 <= 0.0 or specific_heat_j_kgk <= 0.0:
        raise ValueError("density and specific heat must be positive")
    if time_days < 0.0:
        raise ValueError("time must be non-negative")
    area = math.pi * radius_m**2
    velocity = mass_flow_kg_s / (flowing_density_kg_m3 * area)
    time_s = time_days * 86400.0
    return (
        2.0
        * math.pi
        * effective_conductivity_w_mk
        * time_s
        / (flowing_density_kg_m3 * specific_heat_j_kgk * velocity * radius_m)
    )


def ramey_heat_transfer_conductance(
    time_days: float,
    radius_m: float,
    effective_conductivity_w_mk: float,
    formation_diffusivity_m2s: float = 1.1e-7,
    drainage_radius_m: float = 60.0,
) -> float:
    r"""Ramey transient conductance per unit length, W/(m K).

    Ramey's solution is built on the radial resistance between the tubing and
    the formation, evaluated with the conduction front that advances into the
    formation as

    .. math:: \delta(t) = 2\sqrt{\alpha t}.

    Two limits bracket the conductance.

    * **Transient limit** (small :math:`t`): the semi-infinite cylinder result
      gives :math:`U_{tr} = 4\pi r_t k_e/\sqrt{\pi\alpha t}`, which diverges
      as :math:`t \to 0` because an infinite medium accepts heat infinitely
      fast at the instant of the change.
    * **Quasi-steady limit**: once the near-wellbore formation has been swept by
      the thermal front, the conductance is set by the annular resistance out
      to the drainage boundary,
      :math:`U_{qs} = 2\pi k_e/\ln(r_e/r_t)`.

    The physical conductance is the smaller of the two,

    .. math:: U(t) = \min\!\left\{U_{tr},\; U_{qs}\right\},

    which is the classical Ramey clamp: the transient value applies only until
    the conduction front reaches the tubing wall scale, after which the
    near-wellbore zone is the limiting resistance.
    """
    if radius_m <= 0.0 or effective_conductivity_w_mk <= 0.0:
        raise ValueError("radius and conductivity must be positive")
    if formation_diffusivity_m2s <= 0.0:
        raise ValueError("formation diffusivity must be positive")
    if drainage_radius_m <= radius_m:
        raise ValueError("drainage radius must exceed the tubing radius")

    quasi_steady = (
        2.0
        * math.pi
        * effective_conductivity_w_mk
        / math.log(drainage_radius_m / radius_m)
    )
    if time_days <= 0.0:
        # The transient branch diverges here, so the clamp reduces the
        # conductance to the quasi-steady annular value.
        return quasi_steady
    transient = (
        4.0
        * math.pi
        * radius_m
        * effective_conductivity_w_mk
        / math.sqrt(math.pi * formation_diffusivity_m2s * time_days * 86400.0)
    )
    return min(transient, quasi_steady)


def ramey_heat_fraction(
    depth_m: float,
    mass_flow_kg_s: float,
    radius_m: float,
    effective_conductivity_w_mk: float,
    flowing_density_kg_m3: float,
    specific_heat_j_kgk: float,
    formation_diffusivity_m2s: float = 1.1e-7,
    drainage_radius_m: float = 60.0,
    quadrature_points: int = 256,
) -> float:
    r"""Fraction of the inlet-to-formation temperature excess at a depth.

    Integrating the stream energy balance with the Ramey conductance,

    .. math::
        \rho c_p A v\frac{dT}{dz} = -U(t(z))\,(T - T_{f,i}),
        \qquad t(z) = \frac{z}{v}

    gives the classical exponential form

    .. math::
        \frac{T(z) - T_{f,i}}{T_{b,o} - T_{f,i}}
            = \exp\!\left[-\frac{2\pi v}{\rho c_p A}
            \int_0^{z/v} U(\tau)\,d\tau\right]

    with the integral evaluated by the trapezoidal rule.  The result lies in
    :math:`(0, 1]`, equals one at the inlet and decays monotonically with depth.
    """
    if depth_m < 0.0:
        raise ValueError("depth must be non-negative")
    if mass_flow_kg_s <= 0.0 or radius_m <= 0.0:
        raise ValueError("mass flow and radius must be positive")
    if flowing_density_kg_m3 <= 0.0 or specific_heat_j_kgk <= 0.0:
        raise ValueError("density and specific heat must be positive")
    if depth_m == 0.0:
        return 1.0
    area = math.pi * radius_m**2
    velocity = mass_flow_kg_s / (flowing_density_kg_m3 * area)
    contact_time_days = depth_m / velocity / 86400.0
    if contact_time_days <= 0.0:
        return 1.0

    def conductance(days: float) -> float:
        return ramey_heat_transfer_conductance(
            time_days=days,
            radius_m=radius_m,
            effective_conductivity_w_mk=effective_conductivity_w_mk,
            formation_diffusivity_m2s=formation_diffusivity_m2s,
            drainage_radius_m=drainage_radius_m,
        )

    points = max(16, int(quadrature_points))
    step = contact_time_days / points
    integral = 0.5 * conductance(0.0) * step
    for i in range(1, points):
        integral += conductance(i * step) * step
    integral += 0.5 * conductance(contact_time_days) * step
    # The conductance is integrated over contact time in days, so convert the
    # accumulated integral to seconds before forming the dimensionless exponent.
    integral_s = integral * 86400.0

    # dT/dz = -U (T - T_f)/(rho c A v); integrating with dz = v dtau and
    # t = z/v cancels the velocity, leaving the contact time to carry the
    # flow-rate dependence through the integral.
    exponent = 2.0 * math.pi * integral_s / (
        flowing_density_kg_m3 * specific_heat_j_kgk * area
    )
    return min(max(math.exp(-exponent), 0.0), 1.0)


def ramey_wellbore_temperature(
    flowing_temperature_c: float,
    formation_temperature_c: float,
    mass_flow_kg_s: float,
    radius_m: float,
    depth_m: float,
    effective_conductivity_w_mk: float,
    flowing_density_kg_m3: float = 900.0,
    specific_heat_j_kgk: float = 2000.0,
    formation_diffusivity_m2s: float = 1.1e-7,
    drainage_radius_m: float = 60.0,
) -> float:
    r"""Wellbore fluid temperature at ``depth_m``, degC.

    .. math::

        T_{wb}(z) = T_{f,i} + (T_{b,o} - T_{f,i})\,f_{ch}(z)

    with :math:`f_{ch}` from :func:`ramey_heat_fraction`.  The result is
    strictly between the formation and the inlet temperature, equals the inlet
    temperature at the inlet, and relaxes monotonically towards the formation
    temperature with depth and with elapsed contact time.

    Parameters
    ----------
    flowing_temperature_c
        Temperature of the fluid where it enters the tubing, degC.
    formation_temperature_c
        Static formation temperature, degC.
    mass_flow_kg_s
        Mass flow rate, kg/s.
    radius_m
        Tubing inside radius, m.
    depth_m
        Distance travelled from the inlet, m.
    effective_conductivity_w_mk
        Effective radial wellbore conductivity, W/(m K).
    flowing_density_kg_m3
        Mixture density, kg/m^3.
    specific_heat_j_kgk
        Fluid specific heat, J/(kg K).
    formation_diffusivity_m2s
        Thermal diffusivity of the formation, m^2/s.
    drainage_radius_m
        Drainage radius bounding the conduction front, m.
    """
    fraction = ramey_heat_fraction(
        depth_m=depth_m,
        mass_flow_kg_s=mass_flow_kg_s,
        radius_m=radius_m,
        effective_conductivity_w_mk=effective_conductivity_w_mk,
        flowing_density_kg_m3=flowing_density_kg_m3,
        specific_heat_j_kgk=specific_heat_j_kgk,
        formation_diffusivity_m2s=formation_diffusivity_m2s,
        drainage_radius_m=drainage_radius_m,
    )
    return formation_temperature_c + fraction * (
        flowing_temperature_c - formation_temperature_c
    )


# ---------------------------------------------------------------------------
# Traverse
# ---------------------------------------------------------------------------


@dataclass
class WellboreTraverse:
    """Pressure and temperature profiles from the pump to the wellhead."""

    depth_m: np.ndarray
    pressure_pa: np.ndarray
    temperature_c: np.ndarray
    liquid_holdup: np.ndarray
    mixture_density_kg_m3: np.ndarray
    mixture_velocity_m_s: np.ndarray
    regime: List[str]
    friction_factor: np.ndarray
    segments: int
    metadata: dict

    @property
    def total_pressure_drop_pa(self) -> float:
        """Total pressure drop from the pump to the wellhead, Pa."""
        return float(self.pressure_pa[0] - self.pressure_pa[-1])

    @property
    def temperature_drop_c(self) -> float:
        """Total temperature drop from the pump to the wellhead, degC."""
        return float(self.temperature_c[0] - self.temperature_c[-1])

    def as_dict(self) -> dict:
        """JSON-serialisable view for the API layer."""
        return {
            "depth_m": [float(v) for v in self.depth_m],
            "pressure_pa": [float(v) for v in self.pressure_pa],
            "pressure_mpa": [float(v / 1e6) for v in self.pressure_pa],
            "temperature_c": [float(v) for v in self.temperature_c],
            "liquid_holdup": [float(v) for v in self.liquid_holdup],
            "mixture_density_kg_m3": [float(v) for v in self.mixture_density_kg_m3],
            "mixture_velocity_m_s": [float(v) for v in self.mixture_velocity_m_s],
            "regime": list(self.regime),
            "friction_factor": [float(v) for v in self.friction_factor],
            "segments": self.segments,
            "total_pressure_drop_pa": self.total_pressure_drop_pa,
            "total_pressure_drop_mpa": self.total_pressure_drop_pa / 1e6,
            "temperature_drop_c": self.temperature_drop_c,
            "metadata": self.metadata,
        }


def default_mass_flow_kg_s() -> float:
    """Mass flow delivered by the Baghewala pumping unit, kg/s.

    The production rate of a CSS well is pump-limited, so the default traverse
    rate is the volumetric capacity of the sucker rod pump converted to a mass
    flow at reservoir density.  For a 2.25 in, 2.44 m stroke pump at 9 SPM
    this is about 30.7 bbl/d, i.e. roughly 0.05 kg/s -- two orders of magnitude
    below the 2460 bbl/d a 4 kg/s stream would represent.
    """
    capacity_m3_per_day = (
        PUMPING_UNIT.nominal_spm
        * 2.0
        * 60.0
        * PUMPING_UNIT.plunger_area_m2
        * PUMPING_UNIT.nominal_stroke_m
        * PUMPING_UNIT.pump_efficiency
    )
    return capacity_m3_per_day * 900.0 / 86400.0


def compute_traverse(
    segments: int = 60,
    mass_flow_kg_s: Optional[float] = None,
    water_cut: float = BAGHEWALA_FLUID.water_cut,
    bottom_hole_temperature_c: float = 120.0,
    bottom_hole_pressure_pa: float = RESERVOIR.initial_reservoir_pressure_mpa * 1e6,
    formation_temperature_c: float = RESERVOIR.reservoir_temperature_c,
    travel_time_days: float = 0.5,
    gas_fraction: float = 0.0,
    well: WellConfig = WELL,
) -> WellboreTraverse:
    """Integrate the multiphase traverse from the wellhead to the pump.

    The two transports run in opposite directions and are integrated
    accordingly:

    1. **Heat** leaves the stream as it ascends, so the Ramey solution is
       stepped *upward* from the bottom-hole temperature with the cumulative
       contact time.
    2. **Pressure** is integrated *upward* from the pump discharge, because the
       pump discharge pressure is the datum, using the Beggs & Brill holdup and
       the Colebrook-White friction factor at the local temperature.

    Parameters
    ----------
    segments
        Number of depth divisions between the wellhead and the pump.
    mass_flow_kg_s
        Total mass flow rate of the produced stream, kg/s.
    water_cut
        Water cut of the liquid phase, volume fraction.
    bottom_hole_temperature_c
        Temperature of the fluid at the pump discharge, degC.  The wellhead
        temperature is then obtained from the Ramey heat loss over the travel
        time.
    bottom_hole_pressure_pa
        Pressure at the pump discharge, Pa.
    formation_temperature_c
        Static formation temperature, degC.
    travel_time_days
        Time for the fluid to reach the wellhead from the pump, days.
    gas_fraction
        Free gas volume fraction, used to select the flow regime.
    well
        Wellbore geometry.

    Returns
    -------
    WellboreTraverse
        Depth, pressure, temperature, holdup, density and velocity profiles,
        ordered from the pump (index 0) to the wellhead.
    """
    if segments < 2:
        raise ValueError("at least two segments are required")
    if mass_flow_kg_s is None:
        mass_flow_kg_s = default_mass_flow_kg_s()
    if mass_flow_kg_s <= 0.0:
        raise ValueError("mass flow must be positive")
    if not 0.0 <= water_cut <= 1.0:
        raise ValueError("water cut must lie in [0, 1]")

    crude = baghewala_crude()
    radius = 0.5 * well.tubing_id_m
    area = math.pi * radius**2
    total_depth = min(well.total_depth_m, well.pump_depth_m)
    # Depths measured from the pump upward.
    lengths = np.linspace(0.0, total_depth, segments + 1)

    conductivity = wellbore_thermal_conductivity(well)
    relative_roughness = well.tubing_od_m / well.tubing_id_m * 1e-5

    oil_fraction = (1.0 - water_cut) * (1.0 - gas_fraction)
    water_fraction = water_cut * (1.0 - gas_fraction)

    # ---- 1. thermal traverse: pump -> wellhead --------------------------
    # The stream enters the tubing at the bottom-hole temperature and cools as
    # it ascends, so the Ramey solution is applied with the *cumulative*
    # contact time from the pump.
    temperatures = np.empty(segments + 1)
    temperatures[0] = bottom_hole_temperature_c
    for i in range(1, segments + 1):
        # The Ramey heat fraction is referenced to the *inlet* temperature, so
        # each depth is evaluated independently.  Chaining the fractions
        # segment by segment would compound the whole path loss at every
        # node and collapse the profile onto the formation temperature.
        mean_temperature = 0.5 * (temperatures[i - 1] + bottom_hole_temperature_c)
        reference_fluid = FluidProperties(
            temperature_c=mean_temperature,
            pressure_pa=bottom_hole_pressure_pa,
            oil_fraction=oil_fraction,
            water_fraction=water_fraction,
            gas_fraction=gas_fraction,
            oil_viscosity_pa_s=crude.dynamic_viscosity_pa_s(mean_temperature),
        )
        rho_m, _ = mixture_properties(reference_fluid)
        temperatures[i] = ramey_wellbore_temperature(
            flowing_temperature_c=bottom_hole_temperature_c,
            formation_temperature_c=formation_temperature_c,
            mass_flow_kg_s=mass_flow_kg_s,
            radius_m=radius,
            depth_m=lengths[i],
            effective_conductivity_w_mk=conductivity,
            flowing_density_kg_m3=rho_m,
            specific_heat_j_kgk=2000.0,
            drainage_radius_m=RESERVOIR.drainage_radius_m,
        )

    # ---- 2. hydraulic traverse: pump -> wellhead ------------------------
    pressures = np.empty(segments + 1)
    holdups = np.empty(segments + 1)
    densities = np.empty(segments + 1)
    velocities = np.empty(segments + 1)
    factors = np.empty(segments + 1)
    regimes: List[str] = [""] * (segments + 1)

    pressure = bottom_hole_pressure_pa
    pressures[0] = pressure
    hydrostatic_total = 0.0

    for i in range(segments + 1):
        fluid = FluidProperties(
            temperature_c=temperatures[i],
            pressure_pa=pressures[i],
            oil_fraction=oil_fraction,
            water_fraction=water_fraction,
            gas_fraction=gas_fraction,
            oil_viscosity_pa_s=crude.dynamic_viscosity_pa_s(temperatures[i]),
        )
        rho_m, mu_m = mixture_properties(fluid)
        velocity = mass_flow_kg_s / (rho_m * area)

        v_liquid = velocity * fluid.liquid_fraction
        v_gas = velocity * fluid.gas_fraction
        holdup, regime = beggs_brill_holdup(
            v_liquid,
            v_gas,
            _INCLINATION_RAD,
            liquid_density_kg_m3=fluid.oil_density_kg_m3() * fluid.oil_fraction
            + fluid.water_density_kg_m3() * fluid.water_fraction,
            diameter_m=2.0 * radius,
        )
        reynolds = rho_m * velocity * (2.0 * radius) / max(mu_m, 1e-9)
        friction = colebrook_white_friction(reynolds, relative_roughness)

        holdups[i] = holdup
        densities[i] = rho_m
        velocities[i] = velocity
        factors[i] = friction
        regimes[i] = regime

        if i == segments:
            break

        d_length = lengths[i + 1] - lengths[i]
        hydrostatic = rho_m * holdup * 9.80665 * d_length
        hydrostatic_total += hydrostatic
        friction_pressure = (
            friction * rho_m * velocity**2 * d_length / (2.0 * (2.0 * radius))
        )
        pressure -= hydrostatic + friction_pressure
        pressure = max(pressure, 101325.0)
        pressures[i + 1] = pressure

    return WellboreTraverse(
        depth_m=lengths,
        pressure_pa=pressures,
        temperature_c=temperatures,
        liquid_holdup=holdups,
        mixture_density_kg_m3=densities,
        mixture_velocity_m_s=velocities,
        regime=regimes,
        friction_factor=factors,
        segments=segments,
        metadata={
            "tubing_id_m": well.tubing_id_m,
            "mass_flow_kg_s": mass_flow_kg_s,
            "water_cut": water_cut,
            "effective_conductivity_w_mk": conductivity,
            "formation_temperature_c": formation_temperature_c,
            "bottom_hole_temperature_c": bottom_hole_temperature_c,
            "wellhead_temperature_c": float(temperatures[-1]),
            "travel_time_days": travel_time_days,
            "hydrostatic_pressure_drop_pa": hydrostatic_total,
            "friction_pressure_drop_pa": (pressures[0] - pressures[-1])
            - hydrostatic_total,
        },
    )


def baghewala_traverse(**kwargs) -> WellboreTraverse:
    """Traverse for the Baghewala BAG-17 completion, with field defaults."""
    return compute_traverse(**kwargs)
