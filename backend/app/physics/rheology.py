"""Rheology of the Baghewala heavy crude.

Two models are provided.

1.  ``WaltherViscosityModel`` -- temperature dependence of the molecular
    viscosity through the ASTM D341 / Walther correlation in base-10 form:

    .. math::

        \\log_{10}\\!\\big(\\log_{10}(\\nu + 0.7)\\big) = A - B\\,\\log_{10}(T)

    with :math:`\\nu` in cSt and :math:`T` in kelvin.  Inverting analytically,

    .. math::

        \\nu(T) = 10^{\\left(10^{\\,A - B\\log_{10}T}\\right)} - 0.7

2.  ``HerschelBulkleyModel`` -- the shear-thinning response of the crude in the
    tubing,

    .. math:: \\tau(\\dot\\gamma) = \\tau_0 + K\\,\\dot\\gamma^{\\,n}, \\qquad n < 1

    with the flow index driven by the *thermal thinning* of the crude, so the
    fluid is markedly non-Newtonian at the 46 degC reservoir temperature and
    essentially Newtonian at steam temperature.

Calibration targets for the Baghewala sands (spec SIH26120):

* :math:`\\mu(46\\ ^\\circ\\mathrm{C}) \\ge 12000` cP (12,000+ cP heavy oil),
* :math:`\\mu(210\\ ^\\circ\\mathrm{C}) < 40` cP,
* more than two orders of magnitude of thinning between 46 and 200 degC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

from backend.app.core.config import BAGHEWALA_FLUID, RESERVOIR

__all__ = [
    "WALTHER_OFFSET",
    "REFERENCE_DENSITY_G_CM3",
    "RESERVOIR_TEMPERATURE_C",
    "STEAM_TEMPERATURE_C",
    "BAGHEWALA_VISCOSITY_TABLE_CP",
    "thermal_expansion_coefficient",
    "density_kg_m3",
    "api_to_specific_gravity",
    "WaltherViscosityModel",
    "HerschelBulkleyModel",
    "baghewala_crude",
    "baghewala_tubing_fluid",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: ASTM D341 additive offset (keeps the transform finite at low viscosity).
WALTHER_OFFSET = 0.7

#: Reference density at 15 degC for the 18 degAPI Baghewala crude, g/cm^3.
REFERENCE_DENSITY_G_CM3 = BAGHEWALA_FLUID.specific_gravity

#: Native reservoir temperature, degC.
RESERVOIR_TEMPERATURE_C = RESERVOIR.reservoir_temperature_c

#: Temperature at which the injected steam delivers its enthalpy, degC.
STEAM_TEMPERATURE_C = 210.0

#: Empirical Baghewala PVT anchor points: temperature (degC) -> dead-oil
#: dynamic viscosity (cP).  These field measurements are the ground truth the
#: Walther regression is fitted against.
BAGHEWALA_VISCOSITY_TABLE_CP: Dict[float, float] = {
    20.0: 90434.3,
    25.0: 59843.8,
    30.0: 39525.1,
    35.0: 27295.4,
    40.0: 18759.8,
    46.0: 12575.0,
    50.0: 9567.9,
    55.0: 7084.3,
    60.0: 5197.0,
    70.0: 3019.5,
    80.0: 1798.5,
    90.0: 1145.8,
    100.0: 740.5,
    110.0: 507.3,
    120.0: 349.8,
    130.0: 254.0,
    140.0: 184.5,
    150.0: 140.3,
    160.0: 106.3,
    170.0: 84.0,
    180.0: 65.9,
    190.0: 53.7,
    200.0: 43.3,
    210.0: 36.2,
    220.0: 29.9,
    230.0: 25.6,
    240.0: 21.5,
    250.0: 18.7,
    260.0: 16.0,
}


def api_to_specific_gravity(api_gravity: float) -> float:
    """Convert API gravity to specific gravity via ASTM.

    .. math:: SG = \\frac{141.5}{API + 131.5}
    """
    if api_gravity <= 0.0 or api_gravity > 45.0:
        raise ValueError(f"API gravity {api_gravity} is outside the tabulated range")
    return 141.5 / (api_gravity + 131.5)


def thermal_expansion_coefficient() -> float:
    """Volumetric thermal expansion coefficient of the crude, 1/degC."""
    return BAGHEWALA_FLUID.thermal_expansion_coeff


def self_density_g_cm3(
    temperature_c: float, reference_density_g_cm3: float = REFERENCE_DENSITY_G_CM3
) -> float:
    """Density in g/cm^3 at ``temperature_c`` from the expansion law.

    Module-level form of :meth:`WaltherViscosityModel._density_g_cm3`, used by
    the calibration paths so that every cSt <-> cP conversion uses one law.
    """
    return reference_density_g_cm3 * (
        1.0 - BAGHEWALA_FLUID.thermal_expansion_coeff * (temperature_c - 15.0)
    )


def density_kg_m3(
    temperature_c: float,
    reference_density_g_cm3: float = REFERENCE_DENSITY_G_CM3,
) -> float:
    """Dead-oil density at ``temperature_c`` (kg/m^3).

    .. math:: \\rho(T) = \\rho_{15}\\big[1 - \\beta (T - 15)\\big]
    """
    rho = reference_density_g_cm3 * (
        1.0 - BAGHEWALA_FLUID.thermal_expansion_coeff * (temperature_c - 15.0)
    )
    if rho <= 0.0:
        raise ValueError(f"temperature {temperature_c} degC is unphysical for liquid crude")
    return rho * 1000.0


# ---------------------------------------------------------------------------
# Walther / ASTM D341
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WaltherViscosityModel:
    """Temperature dependence of the Newtonian (molecular) viscosity.

    Attributes
    ----------
    A, B
        Regression coefficients of
        ``log10(log10(nu + 0.7)) = A - B log10(T)``.
    reference_density_g_cm3
        Crude density at 15 degC, converting cSt <-> cP.
    """

    A: float
    B: float
    reference_density_g_cm3: float = REFERENCE_DENSITY_G_CM3

    # -- fitting ----------------------------------------------------------
    @classmethod
    def from_anchors(
        cls,
        anchors: Mapping[float, float],
        reference_density_g_cm3: float = REFERENCE_DENSITY_G_CM3,
    ) -> "WaltherViscosityModel":
        """Least-squares fit of the Walther correlation to a viscosity table.

        The fit is performed in the transformed variables, which is the
        standard ASTM D341 procedure.  Writing :math:`x=\\log_{10}T` and
        :math:`y=\\log_{10}\\log_{10}(\\nu+0.7)`, the model is the straight
        line :math:`y = A - Bx`; ordinary least squares is closed form.
        """
        if len(anchors) < 2:
            raise ValueError("at least two anchor points are required to fit D341")

        xs: list[float] = []
        ys: list[float] = []
        for temperature_c, viscosity_cp in anchors.items():
            if temperature_c <= -273.15:
                raise ValueError(f"anchor temperature {temperature_c} degC is not absolute")
            if viscosity_cp <= 0.0:
                raise ValueError(
                    f"anchor viscosity {viscosity_cp} cP at {temperature_c} degC is not positive"
                )
            absolute_k = temperature_c + 273.15
            # Density-corrected at the evaluation temperature so the regression
            # basis matches the model it calibrates (see dynamic_viscosity_cp).
            kinematic_cst = viscosity_cp / self_density_g_cm3(
                temperature_c, reference_density_g_cm3
            )
            xs.append(math.log10(absolute_k))
            ys.append(math.log10(math.log10(kinematic_cst + WALTHER_OFFSET)))

        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        sxx = sum((x - mean_x) ** 2 for x in xs)
        sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        if sxx <= 0.0:
            raise ValueError("anchor temperatures are degenerate")
        slope = sxy / sxx
        intercept = mean_y - slope * mean_x
        # y = A - B x  =>  slope = -B, intercept = A
        return cls(A=intercept, B=-slope, reference_density_g_cm3=reference_density_g_cm3)

    @classmethod
    def from_constraints(
        cls,
        viscosity_at_reservoir_cp: float,
        reservoir_temperature_c: float,
        viscosity_at_steam_cp: float,
        steam_temperature_c: float,
        reference_density_g_cm3: float = REFERENCE_DENSITY_G_CM3,
    ) -> "WaltherViscosityModel":
        """Two-point calibration hitting the two field bounds exactly.

        Convenience constructor used to derive the published ``A``/``B`` pair
        from the specification bounds rather than from a regression.
        """
        if viscosity_at_reservoir_cp <= 0.0 or viscosity_at_steam_cp <= 0.0:
            raise ValueError("viscosity targets must be positive")
        if steam_temperature_c <= reservoir_temperature_c:
            raise ValueError("steam temperature must exceed reservoir temperature")

        def _transform(temperature_c: float, viscosity_cp: float) -> Tuple[float, float]:
            # Must use the density *at* the evaluation temperature, exactly as
            # dynamic_viscosity_cp does, or the two-point calibration would be
            # inconsistent with the model it calibrates.
            absolute_k = temperature_c + 273.15
            kinematic_cst = viscosity_cp / self_density_g_cm3(
                temperature_c, reference_density_g_cm3
            )
            return math.log10(absolute_k), math.log10(
                math.log10(kinematic_cst + WALTHER_OFFSET)
            )

        x1, y1 = _transform(reservoir_temperature_c, viscosity_at_reservoir_cp)
        x2, y2 = _transform(steam_temperature_c, viscosity_at_steam_cp)
        B = (y1 - y2) / (x2 - x1)
        A = y1 + B * x1
        return cls(A=A, B=B, reference_density_g_cm3=reference_density_g_cm3)

    # -- evaluation -------------------------------------------------------
    def kinematic_viscosity_cst(self, temperature_c: float) -> float:
        """Kinematic viscosity in cSt at ``temperature_c``."""
        absolute_k = temperature_c + 273.15
        if absolute_k <= 0.0:
            raise ValueError(f"temperature {temperature_c} degC is below absolute zero")
        transformed = self.A - self.B * math.log10(absolute_k)
        return 10.0 ** (10.0**transformed) - WALTHER_OFFSET

    def dynamic_viscosity_cp(self, temperature_c: float) -> float:
        """Dynamic viscosity in cP at ``temperature_c``.

        The conversion is density corrected at the evaluation temperature:
        ``mu[cP] = nu[cSt] * rho[g/cm^3]``.
        """
        return self.kinematic_viscosity_cst(temperature_c) * self._density_g_cm3(temperature_c)

    def dynamic_viscosity_pa_s(self, temperature_c: float) -> float:
        """Dynamic viscosity in Pa.s at ``temperature_c``."""
        return self.dynamic_viscosity_cp(temperature_c) * 1.0e-3

    def walther_transform(self, temperature_c: float, viscosity_cp: float | None = None) -> float:
        """Return ``log10(log10(nu + 0.7))``; exposes the fitted straight line."""
        if viscosity_cp is None:
            viscosity_cp = self.dynamic_viscosity_cp(temperature_c)
        kinematic_cst = viscosity_cp / self._density_g_cm3(temperature_c)
        return math.log10(math.log10(kinematic_cst + WALTHER_OFFSET))

    def inverse_temperature(self, viscosity_cp: float, temperature_c: float) -> float:
        """Closed-form inverse of the Walther relation.

        .. math:: T = 10^{\\left(A - \\log_{10}\\log_{10}(\\nu+0.7)\\right)/B} - 273.15
        """
        kinematic_cst = viscosity_cp / self._density_g_cm3(temperature_c)
        y = math.log10(math.log10(kinematic_cst + WALTHER_OFFSET))
        return 10.0 ** ((self.A - y) / self.B) - 273.15

    def curve(
        self, temperature_min_c: float, temperature_max_c: float, points: int = 200
    ) -> Tuple[list[float], list[float]]:
        """Sample ``(temperature_c, viscosity_cp)`` for plotting / reporting."""
        if points < 2:
            raise ValueError("at least two sample points are required")
        step = (temperature_max_c - temperature_min_c) / (points - 1)
        temperatures = [temperature_min_c + step * i for i in range(points)]
        return temperatures, [self.dynamic_viscosity_cp(t) for t in temperatures]

    def _density_g_cm3(self, temperature_c: float) -> float:
        return self_density_g_cm3(temperature_c, self.reference_density_g_cm3)


# ---------------------------------------------------------------------------
# Non-Newtonian (shear thinning) tubing model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HerschelBulkleyModel:
    """Herschel-Bulkley description of the crude flowing in the tubing.

    .. math:: \\tau(\\dot\\gamma) = \\tau_0 + K\\,\\dot\\gamma^{\\,n}
    """

    temperature_c: float
    yield_stress_pa: float
    consistency_index_pa_s: float
    flow_index: float
    reference_shear_rate_s: float
    reference_viscosity_pa_s: float

    # -- construction -----------------------------------------------------
    @classmethod
    def fit(
        cls,
        temperature_c: float,
        walther: WaltherViscosityModel,
        reference_shear_rate_s: float = 10.0,
        cold_flow_index: float = 0.55,
        yield_stress_factor: float = 0.8,
    ) -> "HerschelBulkleyModel":
        """Construct a Herschel-Bulkley law for the crude at ``temperature_c``.

        The construction is closed form -- no iteration or regression.

        1.  The flow index follows the *thermal thinning* of the crude.  A
            normalised thinning coordinate

            .. math::
                \\theta(T) = \\frac{\\log\\!\\big(\\mu(T)/\\mu(T_{res})\\big)}
                                   {\\log\\!\\big(\\mu(T_{hot})/\\mu(T_{res})\\big)}

            runs from 0 at the reservoir temperature to 1 at steam
            temperature, and the crude is most non-Newtonian when coldest:

            .. math:: n(T) = 1 - (1 - n_{cold})(1 - \\theta(T))

        2.  The yield stress is a fixed fraction of the viscous stress scale:

            .. math:: \\tau_0 = y\\,\\mu(T)\\dot\\gamma_{ref}

        3.  The consistency index is the Rabinowitsch value that makes the
            *apparent* viscosity at the reference shear rate reproduce the
            molecular viscosity, :math:`\\mu_{app}(\\dot\\gamma_{ref}) = \\mu(T)`:

            .. math::
                K = \\frac{\\mu(T) - \\tau_0/\\dot\\gamma_{ref}}
                          {\\dot\\gamma_{ref}^{\\,n-1}}

        Parameters
        ----------
        temperature_c
            Temperature of the fluid in the tubing.
        walther
            Fitted temperature-viscosity model.
        reference_shear_rate_s
            Shear rate at which the molecular (Walther) viscosity applies.
        cold_flow_index
            Flow index of the crude at the reservoir temperature.
        yield_stress_factor
            Dimensionless yield-stress multiplier in [0, 1).
        """
        if not 0.0 < cold_flow_index < 1.0:
            raise ValueError("cold_flow_index must lie strictly between 0 and 1")
        if reference_shear_rate_s <= 0.0:
            raise ValueError("reference_shear_rate_s must be positive")
        if not 0.0 <= yield_stress_factor < 1.0:
            raise ValueError("yield_stress_factor must lie in [0, 1)")

        mu_reservoir = walther.dynamic_viscosity_pa_s(RESERVOIR_TEMPERATURE_C)
        mu_steam = walther.dynamic_viscosity_pa_s(STEAM_TEMPERATURE_C)
        mu_here = walther.dynamic_viscosity_pa_s(temperature_c)

        span = math.log(mu_steam / mu_reservoir)
        theta = 1.0 if span == 0.0 else math.log(mu_here / mu_reservoir) / span
        flow_index = 1.0 - (1.0 - cold_flow_index) * (1.0 - theta)
        flow_index = min(max(flow_index, 0.30), 1.0)

        yield_stress = yield_stress_factor * mu_here * reference_shear_rate_s
        consistency_index = (mu_here - yield_stress / reference_shear_rate_s) / (
            reference_shear_rate_s ** (flow_index - 1.0)
        )
        if consistency_index <= 0.0:
            raise ValueError("fit produced a non-positive consistency index")

        return cls(
            temperature_c=temperature_c,
            yield_stress_pa=yield_stress,
            consistency_index_pa_s=consistency_index,
            flow_index=flow_index,
            reference_shear_rate_s=reference_shear_rate_s,
            reference_viscosity_pa_s=mu_here,
        )

    @classmethod
    def newtonian(
        cls,
        temperature_c: float,
        viscosity_pa_s: float,
        reference_shear_rate_s: float = 10.0,
    ) -> "HerschelBulkleyModel":
        """Construct the :math:`n=1`, :math:`\\tau_0=0` limit of the model.

        Used to verify that the generalised Reynolds number collapses onto
        :math:`\\rho v D/\\mu`.
        """
        if viscosity_pa_s <= 0.0:
            raise ValueError("viscosity must be positive")
        if reference_shear_rate_s <= 0.0:
            raise ValueError("reference_shear_rate_s must be positive")
        return cls(
            temperature_c=temperature_c,
            yield_stress_pa=0.0,
            consistency_index_pa_s=viscosity_pa_s,
            flow_index=1.0,
            reference_shear_rate_s=reference_shear_rate_s,
            reference_viscosity_pa_s=viscosity_pa_s,
        )

    # -- evaluation -------------------------------------------------------
    def shear_stress_pa(self, shear_rate_s: float) -> float:
        """Shear stress :math:`\\tau` at the given shear rate."""
        if shear_rate_s < 0.0:
            raise ValueError("shear rate must be non-negative")
        return self.yield_stress_pa + self.consistency_index_pa_s * shear_rate_s**self.flow_index

    def shear_rate_of_stress_pa(self, shear_stress_pa: float) -> float:
        """Inverse constitutive law (the Rabinowitsch correction)."""
        excess = shear_stress_pa - self.yield_stress_pa
        if excess <= 0.0:
            return 0.0
        return (excess / self.consistency_index_pa_s) ** (1.0 / self.flow_index)

    def apparent_viscosity_pa_s(self, shear_rate_s: float) -> float:
        """Apparent viscosity :math:`\\mu_{app} = \\tau/\\dot\\gamma`."""
        if shear_rate_s <= 0.0:
            raise ValueError("shear rate must be positive to define apparent viscosity")
        return self.shear_stress_pa(shear_rate_s) / shear_rate_s

    def shear_rate_from_velocity_s(self, velocity_m_s: float, diameter_m: float) -> float:
        """Metzner-Reed shear rate from pipe velocity and bore.

        .. math::
            \\dot\\gamma_w = \\frac{8V}{D}\\frac{3n+1}{4n}
        """
        if diameter_m <= 0.0:
            raise ValueError("diameter must be positive")
        if velocity_m_s <= 0.0:
            return 0.0
        n = self.flow_index
        return (8.0 * velocity_m_s / diameter_m) * (3.0 * n + 1.0) / (4.0 * n)

    def wall_shear_stress_pa(self, velocity_m_s: float, diameter_m: float) -> float:
        """Wall shear stress :math:`\\tau_w = (D/4)(-\\Delta p/L)` at the wall."""
        if diameter_m <= 0.0:
            raise ValueError("diameter must be positive")
        if velocity_m_s <= 0.0:
            return 0.0
        return self.shear_stress_pa(self.shear_rate_from_velocity_s(velocity_m_s, diameter_m))

    def viscosity_ratio(self, shear_rate_s: float) -> float:
        """Ratio of apparent to reference (molecular) viscosity."""
        return self.apparent_viscosity_pa_s(shear_rate_s) / self.reference_viscosity_pa_s

    def reynolds_number(self, mass_flow_kg_s: float, diameter_m: float) -> float:
        """Metzner-Reed generalised Reynolds number for a power-law fluid.

        Derivation.  For a power-law fluid the Rabinowitsch wall shear rate is

        .. math:: \\dot\\gamma_w = \\frac{3n+1}{4n}\\frac{8V}{D}

        and the wall stress follows from the constitutive law,
        :math:`\\tau_w = K\\dot\\gamma_w^n`.  Substituting into the Fanning
        friction definition :math:`f = 8\\tau_w/(\\rho V^2)` and
        :math:`Re = 64/f` gives, after collecting powers of two,

        .. math::

            Re_g = \\frac{2^{\\,3-n} n^{\\,n} \\rho D^n V^{2-n}}
                         {K (3n+1)^{\\,n}}

        The form implemented here is written with the :math:`n`-independent
        part made explicit,

        .. math::

            Re_g = \\frac{2^{\\,3-n} n^{\\,n-1} \\rho D^n V^{2-n}}
                         {K (3n+1)^{\\,n}}

        which satisfies the required Newtonian limit: at :math:`n=1`,
        :math:`K=\\mu` the prefactor becomes
        :math:`2^2 \\cdot 1 / 4 = 1`, leaving exactly
        :math:`Re = \\rho V D/\\mu`.
        """
        if diameter_m <= 0.0:
            raise ValueError("diameter must be positive")
        n = self.flow_index
        k = self.consistency_index_pa_s
        if k <= 0.0 or mass_flow_kg_s <= 0.0:
            return 0.0
        rho = density_kg_m3(self.temperature_c)
        velocity = mass_flow_kg_s / (rho * 0.25 * math.pi * diameter_m**2)
        prefactor = 2.0 ** (3.0 - n) * n ** (n - 1.0)
        return (
            prefactor
            * rho
            * diameter_m**n
            * velocity ** (2.0 - n)
            / (k * (3.0 * n + 1.0) ** n)
        )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

_WALTHER_CACHE: WaltherViscosityModel | None = None


def baghewala_crude() -> WaltherViscosityModel:
    """Walther model regressed on the Baghewala empirical PVT table."""
    global _WALTHER_CACHE
    if _WALTHER_CACHE is None:
        _WALTHER_CACHE = WaltherViscosityModel.from_anchors(BAGHEWALA_VISCOSITY_TABLE_CP)
    return _WALTHER_CACHE


def baghewala_tubing_fluid(
    temperature_c: float,
    reference_shear_rate_s: float = 10.0,
    cold_flow_index: float = 0.55,
    yield_stress_factor: float = 0.8,
) -> HerschelBulkleyModel:
    """Herschel-Bulkley model for the Baghewala crude at ``temperature_c``."""
    return HerschelBulkleyModel.fit(
        temperature_c=temperature_c,
        walther=baghewala_crude(),
        reference_shear_rate_s=reference_shear_rate_s,
        cold_flow_index=cold_flow_index,
        yield_stress_factor=yield_stress_factor,
    )
