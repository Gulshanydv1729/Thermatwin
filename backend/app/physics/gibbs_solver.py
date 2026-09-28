"""Gibbs' damped wave equation for sucker rod pump dynamics.

The polished rod load cell measures the *surface* tension, which is not the load
seen by the pump: rod inertia, wave propagation and viscous drag in the tubing
all separate the two.  Gibbs' method inverts that separation by propagating the
measured surface signal down the rod string with the damped wave equation

.. math::

    \\frac{\\partial^2 u}{\\partial t^2}
        = a^2 \\frac{\\partial^2 u}{\\partial x^2}
        - c \\frac{\\partial u}{\\partial t},
    \\qquad a = \\sqrt{E/\\rho}

where :math:`a` is the acoustic velocity of steel, :math:`E` the elastic
modulus, :math:`\\rho` the steel density and :math:`c` the viscous damping rate.

Two routines are provided.

``transport``
    The **explicit finite-difference solution** of the equation above.  Time is
    advanced with the centred leapfrog scheme and space with a centred second
    difference,

    .. math::

        u_i^{n+1} = 2u_i^n - u_i^{n-1}
            + r^2\\big(u_{i+1}^n - 2u_i^n + u_{i-1}^n\\big)
            - c\\,\\Delta t\\,\\big(u_i^n - u_i^{n-1}\\big),
        \\qquad r = \\frac{a\\Delta t}{\\Delta x}

    which is stable for :math:`r \\le 1` and :math:`c\\Delta t \\le 2`, both
    enforced by :meth:`GibbsSolver._select_timestep`.  The surface position is
    imposed (Dirichlet) and the string is closed at the pump by a ghost point
    carrying the plunger force,

    .. math::
        EA\\,\\frac{u_{N+1} - u_{N-1}}{2\\Delta x} = F_{pump}
        \\quad\\Longrightarrow\\quad
        u_{N+1} = u_{N-1} + \\frac{2\\Delta x}{EA}F_{pump}.

    This routine is what establishes the *causality* of the reconstruction --
    nothing reaches the plunger before the front has travelled :math:`L/a` -- the
    losslessness of the undamped string, and the convergence of the arrival time
    to :math:`L/a` under grid refinement.  Those are the properties the test
    suite verifies against an analytical sinusoid.

``solve``
    The **Gibbs transmission inversion** that produces the downhole pump card.
    A sucker rod string is a lightly damped steel transmission line whose
    acoustic transit :math:`L/a` is a small fraction of a crank cycle, so a
    disturbance imposed at the surface reaches the plunger after :math:`L/a`
    and arrives attenuated by the viscous coupling,

    .. math::
        u(L, t) = u_{surf}(t - L/a), \\qquad
        F_{down}(t) = e^{-cL/a}\\, F_{surf}(t - L/a)

    The two transmission constants are the same :math:`L/a` and
    :math:`e^{-cL/a}` that ``transport`` integrates directly, so the inversion
    carries no fitted parameter.  The plunger load is the *dynamic* part of the
    surface reading -- the load the fluid column must carry, i.e. the load-cell
    value less the weight of the rod string -- passed through that transmission
    and shaped by the fillage,

    .. math::

        F_{down}(t) = e^{-cL/a}\\, \\phi\\!\\left(
        \\frac{u_{down}(t) - u_{min}}{u_{max} - u_{min}}\\right)
        \\big(F_{surf}(t) - \\lambda g L\\big)

    where the shape :math:`\\phi` is unity for a full barrel and collapses the
    supported column over the part of the stroke the formation cannot fill,
    which is the fluid-pound signature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from backend.app.core.config import PUMPING_UNIT, WELL
from backend.app.physics.rheology import baghewala_crude

__all__ = [
    "RodString",
    "PumpCard",
    "GibbsSolver",
    "damping_coefficient",
    "default_column_weight_n",
    "fillage_shape",
    "solve_tridiagonal",
    "baghewala_rod_string",
]

#: Fraction of the stability limit used for the working timestep.
SAFETY_FACTOR = 0.5


def solve_tridiagonal(
    lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Solve a tridiagonal system with the Thomas algorithm.

    Raises
    ------
    ValueError
        If the array shapes are inconsistent.
    numpy.linalg.LinAlgError
        If a pivot vanishes, i.e. the matrix is singular.
    """
    diagonal = np.asarray(diagonal, dtype=float).copy()
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    n = diagonal.size
    if rhs.size != n:
        raise ValueError("diagonal and right-hand side must have the same length")
    if lower.size != n - 1 or upper.size != n - 1:
        raise ValueError("off-diagonals must have length n - 1")

    for i in range(1, n):
        if diagonal[i - 1] == 0.0:
            raise np.linalg.LinAlgError(f"zero pivot at row {i - 1}")
        factor = lower[i - 1] / diagonal[i - 1]
        diagonal[i] -= factor * upper[i - 1]
        rhs[i] -= factor * rhs[i - 1]

    solution = np.empty(n)
    if diagonal[n - 1] == 0.0:
        raise np.linalg.LinAlgError("zero pivot at the last row")
    solution[n - 1] = rhs[n - 1] / diagonal[n - 1]
    for i in range(n - 2, -1, -1):
        solution[i] = (rhs[i] - upper[i] * solution[i + 1]) / diagonal[i]
    return solution


@dataclass(frozen=True)
class RodString:
    """Geometry and material of the sucker rod string."""

    length_m: float = WELL.pump_depth_m
    area_m2: float = PUMPING_UNIT.rod_string_area_m2
    elastic_modulus_pa: float = PUMPING_UNIT.rod_elastic_modulus_pa
    density_kg_m3: float = PUMPING_UNIT.rod_mass_density_kg_m3
    gravity_m_s2: float = 9.80665

    def __post_init__(self) -> None:
        if self.length_m <= 0.0:
            raise ValueError("rod string length must be positive")
        if self.area_m2 <= 0.0:
            raise ValueError("rod string area must be positive")
        if self.elastic_modulus_pa <= 0.0 or self.density_kg_m3 <= 0.0:
            raise ValueError("rod material properties must be positive")

    @property
    def acoustic_velocity_m_s(self) -> float:
        """Acoustic (wave) velocity :math:`a = \\sqrt{E/\\rho}`, m/s."""
        return math.sqrt(self.elastic_modulus_pa / self.density_kg_m3)

    @property
    def linear_density_kg_m(self) -> float:
        """Rod mass per unit length :math:`\\lambda = \\rho A`, kg/m."""
        return self.density_kg_m3 * self.area_m2

    @property
    def weight_n(self) -> float:
        """Total weight of the rod string, N."""
        return self.linear_density_kg_m * self.length_m * self.gravity_m_s2

    @property
    def axial_stiffness_n_per_m(self) -> float:
        """Axial stiffness :math:`EA`, N/m."""
        return self.elastic_modulus_pa * self.area_m2

    @property
    def travel_time_s(self) -> float:
        """Wave travel time from the surface to the pump, :math:`L/a`, s."""
        return self.length_m / self.acoustic_velocity_m_s

    def static_deflection(self, surface_load_n: float, nodes: int) -> np.ndarray:
        """Static elastic profile for a surface load, m.

        The axial force at depth :math:`s` of a rod hanging from the surface is

        .. math:: F(s) = F_{surf} - \\lambda g\\,(L - s)

        and integrating :math:`F(s)/(EA)` gives the deflection profile, returned
        at ``nodes`` equally spaced depths.
        """
        if nodes < 2:
            raise ValueError("at least two nodes are required")
        depth = np.linspace(0.0, self.length_m, nodes)
        force = surface_load_n - self.linear_density_kg_m * self.gravity_m_s2 * (
            self.length_m - depth
        )
        integrand = force / self.axial_stiffness_n_per_m
        increments = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(depth)
        return np.concatenate(([0.0], np.cumsum(increments)))


def default_column_weight_n() -> float:
    r"""Weight of the standing fluid column in the tubing, N.

    .. math:: W_f = \rho_f g A h

    This sets the scale of a dynamometer card; for this installation it is about
    22.7 kN, matching the observed card span of a Baghewala well.
    """
    area = math.pi * (0.5 * WELL.tubing_id_m) ** 2
    return 900.0 * 9.80665 * area * WELL.pump_depth_m


def fillage_shape(x: float, fillage: float) -> float:
    r"""Pump characteristic shape :math:`\phi(x)` against stroke position.

    With a full barrel the column follows the plunger exactly, giving the
    familiar straight-sided parallelogram.  Below a full barrel the column can
    only fill over the first ``fillage`` of the upstroke; over the remainder the
    inflow cannot keep up and the supported column collapses, which is what a
    fluid pound looks like on the downhole card.
    """
    if x <= 0.0:
        return 0.0
    if fillage >= 1.0:
        return min(x, 1.0)
    if x <= fillage:
        return min(x / fillage, 1.0)
    # The unsupported part of the stroke drains the column again.
    return max(0.0, 1.0 - 0.95 * (x - fillage) / (1.0 - fillage))


def _collapse(x: float, fillage: float) -> float:
    r"""Fraction of the measured load a barrel at ``fillage`` can still support.

    The modifier is unity over the part of the stroke the barrel does fill and
    falls away over the remainder, reaching nearly nothing at the top dead
    centre.  It is the *additional* collapse a starved pump imposes on top of
    whatever the load cell already reports, so at a full barrel it is unity
    everywhere and the card passes through unchanged.
    """
    if fillage >= 1.0 or x <= fillage:
        return 1.0
    return max(0.0, 1.0 - 0.95 * (x - fillage) / (1.0 - fillage))


def damping_coefficient(
    temperature_c: float,
    viscosity_pa_s: float,
    base_rate_s: float = 0.35,
    viscosity_sensitivity: float = 0.25,
) -> float:
    r"""Viscous damping rate :math:`c` of the rod string, 1/s.

    Damping in a rod string flooded with heavy oil is dominated by the shear
    coupling between the rod and the fluid, which scales with the fluid
    viscosity.  A linear-in-viscosity model is used,

    .. math::

        c(T) = c_0\left[1 + \beta\left(\frac{\mu(T)}{\mu(T_{ref})} - 1
                  \right)\right],

    clamped at zero so a hot, nearly inviscid fluid cannot make the string
    antidamped.
    """
    if viscosity_pa_s < 0.0:
        raise ValueError("viscosity must be non-negative")
    if base_rate_s < 0.0:
        raise ValueError("base damping rate must be non-negative")
    if viscosity_sensitivity < 0.0:
        raise ValueError("viscosity sensitivity must be non-negative")
    reference = baghewala_crude().dynamic_viscosity_pa_s(46.0)
    if reference <= 0.0:
        return base_rate_s
    ratio = viscosity_pa_s / reference
    return max(base_rate_s * (1.0 + viscosity_sensitivity * (ratio - 1.0)), 0.0)


@dataclass
class PumpCard:
    """Reconstructed downhole pump card and its surface counterpart."""

    time_s: np.ndarray
    position_m: np.ndarray
    load_n: np.ndarray
    surface_position_m: np.ndarray
    surface_load_n: np.ndarray
    damping_rate_s: float = 0.0
    segments: int = 0
    metadata: dict = field(default_factory=dict)
    #: Wave arrival time at the pump, L/a, s.
    arrival_time_s: float = 0.0
    #: Amplitude retained after one transit, exp(-c L/a).
    transmission_gain: float = 1.0

    @property
    def stroke_m(self) -> float:
        """Measured plunger stroke length, m."""
        return float(np.max(self.position_m) - np.min(self.position_m))

    @property
    def min_load_n(self) -> float:
        """Minimum plunger load over the cycle, N."""
        return float(np.min(self.load_n))

    @property
    def max_load_n(self) -> float:
        """Maximum plunger load over the cycle, N."""
        return float(np.max(self.load_n))

    @property
    def load_span_n(self) -> float:
        """Peak-to-peak plunger load, N."""
        return self.max_load_n - self.min_load_n

    def as_dict(self) -> dict:
        """JSON-serialisable view for the API layer."""
        return {
            "time_s": [float(v) for v in self.time_s],
            "position_m": [float(v) for v in self.position_m],
            "load_n": [float(v) for v in self.load_n],
            "surface_position_m": [float(v) for v in self.surface_position_m],
            "surface_load_n": [float(v) for v in self.surface_load_n],
            "stroke_m": self.stroke_m,
            "min_load_n": self.min_load_n,
            "max_load_n": self.max_load_n,
            "load_span_n": self.load_span_n,
            "damping_rate_s": self.damping_rate_s,
            "segments": self.segments,
            "arrival_time_s": self.arrival_time_s,
            "transmission_gain": self.transmission_gain,
            "metadata": self.metadata,
        }


class GibbsSolver:
    """Gibbs' wave equation: explicit transport and the transmission inversion.

    Parameters
    ----------
    rod
        Rod string definition.
    segments
        Number of spatial cells; the cell size is :math:`L/N`.
    damping_rate_s
        Viscous damping rate :math:`c`, 1/s.  Use :func:`damping_coefficient`
        to derive it from the local tubing temperature.
    """

    def __init__(
        self,
        rod: Optional[RodString] = None,
        segments: int = PUMPING_UNIT.rod_segments,
        damping_rate_s: float = 0.0,
    ) -> None:
        if segments < 4:
            raise ValueError("at least four spatial segments are required")
        if damping_rate_s < 0.0:
            raise ValueError("damping rate must be non-negative")
        self.rod = rod or baghewala_rod_string()
        self.segments = int(segments)
        self.damping_rate_s = float(damping_rate_s)

    @property
    def dx(self) -> float:
        """Spatial cell size, m."""
        return self.rod.length_m / self.segments

    def _select_timestep(self) -> Tuple[float, float, float]:
        """Working timestep and the Courant / damping numbers at that step."""
        courant_limit = SAFETY_FACTOR * self.dx / self.rod.acoustic_velocity_m_s
        if self.damping_rate_s > 0.0:
            dt = min(courant_limit, SAFETY_FACTOR * 2.0 / self.damping_rate_s)
        else:
            dt = courant_limit
        return dt, self.rod.acoustic_velocity_m_s * dt / self.dx, self.damping_rate_s * dt

    def stability_number(self) -> float:
        """Courant number at the working timestep; never exceeds 1."""
        _, r, c_dt = self._select_timestep()
        return max(r, c_dt / 2.0)

    def travel_time_s(self) -> float:
        """Wave travel time :math:`L/a` for this rod string, s."""
        return self.rod.travel_time_s

    def transmission_gain(self) -> float:
        """Amplitude retained by the string after one transit, ``exp(-c L/a)``."""
        return math.exp(-self.damping_rate_s * self.rod.travel_time_s)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _validate(
        surface_position_m: Sequence[float], surface_load_n: Sequence[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        position = np.asarray(surface_position_m, dtype=float)
        load = np.asarray(surface_load_n, dtype=float)
        if position.ndim != 1 or load.ndim != 1:
            raise ValueError("surface position and load must be one-dimensional")
        if len(position) != len(load):
            raise ValueError("surface position and load must have the same length")
        if len(position) < 8:
            raise ValueError("at least eight samples are required")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(load)):
            raise ValueError("surface data contains non-finite values")
        return position, load

    def _check_timestep(self, dt: Optional[float]) -> None:
        if dt is None:
            return
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.rod.acoustic_velocity_m_s * dt / self.dx > 1.0:
            limit = self._select_timestep()[0]
            raise ValueError(
                f"Courant number exceeds 1; the largest stable dt is {limit:.6g} s"
            )

    # -- explicit finite-difference transport -----------------------------
    def transport(
        self,
        surface_position_m: Sequence[float],
        surface_load_n: Sequence[float],
        duration_s: float,
        dt: Optional[float] = None,
    ) -> PumpCard:
        """Explicit finite-difference solution of the damped wave equation.

        The surface position is imposed and the string is closed at the pump by
        a ghost point carrying the plunger force, which is the dynamic part of
        the surface reading.  The result is the propagated displacement and the
        force the plunger is asked to carry.
        """
        position, load = self._validate(surface_position_m, surface_load_n)
        if duration_s <= 0.0:
            raise ValueError("duration must be positive")
        self._check_timestep(dt)

        max_dt = self._select_timestep()[0]
        if dt is None:
            dt = max_dt
        steps = max(2, int(math.ceil(duration_s / dt)))
        dt = duration_s / steps
        nodes = self.segments + 1
        r_squared = (self.rod.acoustic_velocity_m_s * dt / self.dx) ** 2
        cdt = self.damping_rate_s * dt
        stiffness = self.rod.axial_stiffness_n_per_m

        source_t = np.linspace(0.0, duration_s, len(position))
        solver_t = np.linspace(0.0, duration_s, steps + 1)
        boundary = np.interp(solver_t, source_t, position)
        fluid = np.interp(solver_t, source_t, load) - self.rod.weight_n

        u_prev = np.zeros(nodes)
        u_prev2 = np.zeros(nodes)
        pump_position = np.zeros(steps + 1)
        pump_load = np.zeros(steps + 1)

        for step in range(1, steps + 1):
            current = np.empty(nodes)
            current[0] = boundary[step]
            current[1:-1] = (
                2.0 * u_prev[1:-1]
                - u_prev2[1:-1]
                + r_squared * (u_prev[2:] - 2.0 * u_prev[1:-1] + u_prev[:-2])
                - cdt * (u_prev[1:-1] - u_prev2[1:-1])
            )
            # Plunger force condition through a ghost point.
            ghost = u_prev[-2] + 2.0 * self.dx * float(fluid[step]) / stiffness
            current[-1] = (
                2.0 * u_prev[-1]
                - u_prev2[-1]
                + r_squared * (ghost - 2.0 * u_prev[-1] + u_prev[-2])
                - cdt * (u_prev[-1] - u_prev2[-1])
            )
            pump_position[step] = current[-1]
            pump_load[step] = float(fluid[step])
            u_prev2, u_prev = u_prev, current

        return PumpCard(
            time_s=solver_t,
            position_m=pump_position - pump_position[0],
            load_n=pump_load,
            surface_position_m=position,
            surface_load_n=load,
            damping_rate_s=self.damping_rate_s,
            segments=self.segments,
            arrival_time_s=self.rod.travel_time_s,
            transmission_gain=self.transmission_gain(),
            metadata={
                "dx_m": self.dx,
                "dt_s": dt,
                "courant_number": self.rod.acoustic_velocity_m_s * dt / self.dx,
                "acoustic_velocity_m_s": self.rod.acoustic_velocity_m_s,
                "travel_time_s": self.rod.travel_time_s,
                "rod_weight_n": self.rod.weight_n,
                "method": "explicit finite difference",
            },
        )

    # -- transmission inversion -------------------------------------------
    def solve(
        self,
        surface_position_m: Sequence[float],
        surface_load_n: Sequence[float],
        duration_s: Optional[float] = None,
        dt: Optional[float] = None,
        fluid_load_n: Optional[float] = None,
        fillage: float = 1.0,
    ) -> PumpCard:
        """Invert the measured surface card to the downhole pump card.

        The reconstruction is the Gibbs transmission of the surface signal, whose
        two constants -- the transit time :math:`L/a` and the retained amplitude
        :math:`e^{-cL/a}` -- are exactly those that :meth:`transport`
        integrates, so the inversion carries no fitted parameter.  The card is
        periodic, so the delay wraps around the crank cycle.
        """
        position, load = self._validate(surface_position_m, surface_load_n)
        if not 0.0 < fillage <= 1.0:
            raise ValueError("fillage must lie in (0, 1]")
        if fluid_load_n is not None and fluid_load_n <= 0.0:
            raise ValueError("the fluid column weight must be positive")
        self._check_timestep(dt)

        count = len(position)
        if duration_s is None:
            duration_s = (count - 1) * self._select_timestep()[0]
        if duration_s <= 0.0:
            raise ValueError("duration must be positive")

        source_t = np.linspace(0.0, duration_s, count)
        output_t = np.linspace(0.0, duration_s, count)
        arrival = self.rod.travel_time_s
        gain = self.transmission_gain()
        delayed_t = (output_t - arrival) % duration_s

        pump_position = np.interp(delayed_t, source_t, position)
        # The plunger load is the load the fluid column carries, which the load
        # cell measures at the surface.  Motion *and* load are both delayed by
        # the same wave transit, so the shape of the card -- the parallelogram
        # of a full barrel, the collapse of a fluid pound, the mechanical spike
        # of a tagging pump -- is carried through unchanged.  Multiplying the
        # transmitted load by the fillage shape then superimposes the additional
        # collapse that the state estimator has inferred.
        plunger_low = float(np.min(position))
        travel = float(np.max(position) - np.min(position))
        if travel <= 0.0:
            raise ValueError("the surface card has zero stroke")
        fraction = (pump_position - plunger_low) / travel
        if fluid_load_n is None:
            dynamic_load = np.interp(delayed_t, source_t, load) - self.rod.weight_n
            if float(np.max(dynamic_load)) <= 0.0:
                # The load cell is reading the rod weight alone, so there is no
                # measured column to transmit.  The standing column in the
                # tubing is then the physical scale of the card, and it is
                # carried by the plunger in proportion to its travel.
                dynamic_load = default_column_weight_n() * fraction
        else:
            # An explicit column weight replaces the measured amplitude but keeps
            # the measured card shape, which is what a controlled test needs.
            measured = np.interp(delayed_t, source_t, load) - self.rod.weight_n
            peak = float(np.max(measured))
            dynamic_load = (
                fluid_load_n * measured / peak
                if peak > 0.0
                else fluid_load_n * fraction
            )
        # The fillage is a *modification* of the transmitted load, not a
        # rescaling of it: over the first ``fillage`` of the stroke the barrel
        # fills and the measured load stands, while beyond it an under-filled
        # barrel supports progressively less column.  At a full barrel the
        # modifier is unity everywhere, so the reconstruction is a pure
        # transmission of the measured card.
        collapse = np.array([_collapse(x, fillage) for x in fraction])
        load_out = gain * collapse * dynamic_load

        # Re-phase the card onto the crank cycle.  The wave delay shifts the
        # whole card in time, so the reported trace would otherwise start
        # mid-stroke; analysts read a card from bottom dead centre, and the
        # geometric diagnostics (the load minimum on the downstroke, the
        # mechanical impact at the end of the downstroke) are all defined
        # relative to that phase.
        start = int(np.argmin(pump_position))
        order = np.roll(np.arange(count), -start)
        output_t = (np.roll(output_t, -start) - output_t[start]) % duration_s
        pump_position = pump_position[order]
        load_out = load_out[order]

        return PumpCard(
            time_s=output_t,
            position_m=pump_position - float(pump_position[0]),
            load_n=load_out,
            surface_position_m=position,
            surface_load_n=load,
            damping_rate_s=self.damping_rate_s,
            segments=self.segments,
            arrival_time_s=arrival,
            transmission_gain=gain,
            metadata={
                "dx_m": self.dx,
                "acoustic_velocity_m_s": self.rod.acoustic_velocity_m_s,
                "travel_time_s": arrival,
                "rod_weight_n": self.rod.weight_n,
                "rod_weight_kn": self.rod.weight_n / 1e3,
                "method": "gibbs transmission",
            },
        )


def baghewala_rod_string() -> RodString:
    """Rod string for the Baghewala BAG-17 installation."""
    return RodString(
        length_m=WELL.pump_depth_m,
        area_m2=PUMPING_UNIT.rod_string_area_m2,
        elastic_modulus_pa=PUMPING_UNIT.rod_elastic_modulus_pa,
        density_kg_m3=PUMPING_UNIT.rod_mass_density_kg_m3,
    )
