"""Extended Kalman filter fusing SCADA telemetry with the physics models.

The well does not report the quantities the control policy needs.  The load
cell and the position encoder give the surface card; the casing head pressure
and the wellhead temperature give the top of the tubing.  The quantities that
actually determine whether the pump can fill -- the **pump intake pressure**,
the **reservoir skin** and the **thermal radius** -- are unmeasured.

This module closes that gap with a linearised extended Kalman filter.  The
state is

.. math::
    x = \\begin{bmatrix} P_{ip} & s & R_h & T_{bh} \\end{bmatrix}^{T}

being the pump intake pressure (Pa), the dimensionless skin factor, the steam
chest radius (m) and the bottom hole temperature (degC).  The process model
propagates the skin and the thermal radius with the CSS physics, while the
measurement model relates the state to the instrument readings,

.. math::
    \\begin{aligned}
    P_{chp} &= P_{ip} - \\rho_m g H - \\Delta p_{friction} \\\\
    T_{wh}  &= T_{bh} - \\Delta T_{Ramey}(P_{ip}) \\\\
    q_{surf} &= \\min\\!\\left(q_{res}(P_{ip}, s),\\; q_{pump}\\right) \\\\
    F_{min} &= F_{rod} + m_{fluid} g \\\\
    F_{prl} &= F_{rod} + m_{fluid} g + A_p \\Delta P_{pump}
    \\end{aligned}

so the filter interpolates between the physics prediction and the noisy
instrument readings according to their stated uncertainties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.app.core.config import (
    CSS,
    PUMPING_UNIT,
    RESERVOIR,
    THRESHOLDS,
    WELL,
    WellConfig,
)
from backend.app.physics.hydraulics import FluidProperties, mixture_properties
from backend.app.physics.rheology import baghewala_crude

__all__ = [
    "TelemetryPacket",
    "StateEstimate",
    "ExtendedKalmanFilter",
    "baghewala_state_filter",
]

#: Index layout of the state vector.
STATE_PIP = 0
STATE_SKIN = 1
STATE_RADIUS = 2
STATE_BHT = 3
STATE_SIZE = 4

#: Index layout of the measurement vector.
MEAS_CASING_PRESSURE = 0
MEAS_WELLHEAD_TEMPERATURE = 1
MEAS_SURFACE_FLOW = 2
MEAS_CARD_MIN_LOAD = 3
MEAS_CARD_MAX_LOAD = 4
MEAS_SIZE = 7
#: Weak prior channels.  Two states are only weakly identifiable from the
#: surface instruments: the skin factor, which a pump-limited well barely
#: constrains through its flow channel, and the bottom hole temperature, whose
#: surface signature has largely equilibrated with the formation by the time the
#: fluid reaches the wellhead.  Both are supplied as soft priors.
MEAS_SKIN_PRIOR = 5
MEAS_BHT_PRIOR = 6


@dataclass
class TelemetryPacket:
    """One SCADA sample of a Baghewala well."""

    timestamp_s: float
    casing_head_pressure_pa: float
    wellhead_temperature_c: float
    surface_flow_m3_per_day: float
    card_min_load_n: float
    card_max_load_n: float
    spm: float
    stroke_m: float
    water_cut: float = 0.12

    def as_dict(self) -> dict:
        """JSON-serialisable view for the API layer."""
        return {
            "timestamp_s": self.timestamp_s,
            "casing_head_pressure_pa": self.casing_head_pressure_pa,
            "casing_head_pressure_mpa": self.casing_head_pressure_pa / 1e6,
            "wellhead_temperature_c": self.wellhead_temperature_c,
            "surface_flow_m3_per_day": self.surface_flow_m3_per_day,
            "card_min_load_n": self.card_min_load_n,
            "card_max_load_n": self.card_max_load_n,
            "spm": self.spm,
            "stroke_m": self.stroke_m,
            "water_cut": self.water_cut,
        }


@dataclass
class StateEstimate:
    """Posterior state and its uncertainty."""

    pump_intake_pressure_pa: float
    skin_factor: float
    thermal_radius_m: float
    bottom_hole_temperature_c: float
    covariance: np.ndarray
    residual_norm: float
    innovations: np.ndarray

    @property
    def pump_intake_pressure_mpa(self) -> float:
        """Pump intake pressure in MPa."""
        return self.pump_intake_pressure_pa / 1e6

    @property
    def standard_error_pip_mpa(self) -> float:
        """Posterior standard error of the pump intake pressure, MPa."""
        return float(math.sqrt(max(self.covariance[STATE_PIP, STATE_PIP], 0.0)) / 1e6)

    def as_dict(self) -> dict:
        """JSON-serialisable view for the API layer."""
        return {
            "pump_intake_pressure_pa": self.pump_intake_pressure_pa,
            "pump_intake_pressure_mpa": self.pump_intake_pressure_mpa,
            "skin_factor": self.skin_factor,
            "thermal_radius_m": self.thermal_radius_m,
            "bottom_hole_temperature_c": self.bottom_hole_temperature_c,
            "pip_standard_error_mpa": self.standard_error_pip_mpa,
            "residual_norm": self.residual_norm,
            "innovations": [float(v) for v in self.innovations],
        }


class ExtendedKalmanFilter:
    """Linearised extended Kalman filter for the four-state well model.

    Parameters
    ----------
    rod_weight_n
        Rod string weight, which sets the reference of the load channel.
    fluid_column_kg
        Mass of fluid in the tubing, giving the static load offset.
    plunger_area_m2
        Plunger area, converting a pump pressure rise into a card load rise.
    process_noise
        Diagonal of :math:`Q`, the per-step state uncertainty growth.
    measurement_noise
        Diagonal of :math:`R`, the instrument variance.  The defaults reflect
        the field instrument classes: casing head pressure +-0.05 MPa, wellhead
        temperature +-2 degC, flow +-3%, card loads +-1%.
    """

    def __init__(
        self,
        rod_weight_n: float = 21937.6,
        fluid_column_kg: float = None,
        plunger_area_m2: float = PUMPING_UNIT.plunger_area_m2,
        process_noise: Optional[Sequence[float]] = None,
        measurement_noise: Optional[Sequence[float]] = None,
        well: WellConfig = WELL,
    ) -> None:
        if plunger_area_m2 <= 0.0:
            raise ValueError("plunger area must be positive")
        if rod_weight_n <= 0.0:
            raise ValueError("rod weight must be positive")
        if fluid_column_kg is None:
            # Mass of the standing fluid column in the tubing, rho A L.  This is
            # the static load offset the card rides on, and it dwarfs the
            # A_p dP differential that carries the fillage information.
            fluid_column_kg = 900.0 * math.pi * (0.5 * WELL.tubing_id_m) ** 2 * WELL.pump_depth_m
        self.rod_weight_n = rod_weight_n
        self.fluid_column_kg = fluid_column_kg
        self.plunger_area_m2 = plunger_area_m2
        self.well = well

        if process_noise is None:
            process_noise = (2.0e4, 4.0e-4, 4.0e-3, 1.2e-1)
        if measurement_noise is None:
            measurement_noise = (5.0e4, 4.0, 0.05, 800.0, 1500.0, 4.0, 400.0)
        self.Q = np.diag(np.asarray(process_noise, dtype=float))
        self.R = np.diag(np.asarray(measurement_noise, dtype=float))
        if self.Q.shape != (STATE_SIZE, STATE_SIZE):
            raise ValueError("process_noise must have four entries")
        if self.R.shape != (MEAS_SIZE, MEAS_SIZE):
            raise ValueError("measurement_noise must have six entries")

        self.skin_prior = getattr(self, "skin_prior", RESERVOIR.skin_factor)
        self.state = np.array(
            [
                RESERVOIR.initial_reservoir_pressure_mpa * 0.55e6,
                RESERVOIR.skin_factor,
                5.0,
                RESERVOIR.reservoir_temperature_c,
            ],
            dtype=float,
        )
        self.covariance = np.diag(
            np.array(
                [
                    (2.0e6) ** 2,
                    4.0,
                    25.0,
                    400.0,
                ],
                dtype=float,
            )
        )
        self.skin_prior = RESERVOIR.skin_factor
        self.bottom_hole_temperature_prior = RESERVOIR.reservoir_temperature_c
        self.initialised = False
        self._last_timestamp: Optional[float] = None

    # -- physics helpers --------------------------------------------------
    def _mixture_density(self, temperature_c: float, water_cut: float) -> float:
        crude = baghewala_crude()
        fluid = FluidProperties(
            temperature_c=temperature_c,
            pressure_pa=self.state[STATE_PIP],
            oil_fraction=1.0 - water_cut,
            water_fraction=water_cut,
            oil_viscosity_pa_s=crude.dynamic_viscosity_pa_s(temperature_c),
        )
        rho, _ = mixture_properties(fluid)
        return rho

    def _friction_pressure_pa(self, temperature_c: float, rate_m3_per_day: float) -> float:
        """Viscous friction pressure for the given rate, Pa.

        Heavy oil in small tubing is laminar, where the Hagen-Poiseuille loss
        is linear in the superficial velocity,

        .. math:: \\Delta p_f = \\frac{32\\mu L v}{D^2}.
        """
        crude = baghewala_crude()
        viscosity = crude.dynamic_viscosity_pa_s(temperature_c)
        radius = 0.5 * self.well.tubing_id_m
        area = math.pi * radius**2
        rho = self._mixture_density(temperature_c, 0.12)
        velocity = (rate_m3_per_day / 86400.0) / area
        return 32.0 * viscosity * self.well.pump_depth_m * velocity / (4.0 * radius**2)

    def _hydrostatic_pa(self, temperature_c: float, water_cut: float) -> float:
        rho = self._mixture_density(temperature_c, water_cut)
        return rho * 9.80665 * self.well.pump_depth_m

    def _reservoir_inflow(self, pip_pa: float, skin: float, temperature_c: float) -> float:
        """Vogel inflow at the estimated intake conditions, m^3/day."""
        p_res = RESERVOIR.initial_reservoir_pressure_mpa
        if p_res <= 0.0 or skin < 0.0:
            return 0.0
        ratio = min(max((pip_pa / 1e6) / p_res, 0.0), 1.0)
        effective = 1.0 - (1.0 - ratio) / (1.0 + skin)
        productivity = 1.0 - 0.2 * effective - 0.8 * effective**2
        crude = baghewala_crude()
        viscosity = crude.dynamic_viscosity_pa_s(temperature_c)
        reference = crude.dynamic_viscosity_pa_s(RESERVOIR.reservoir_temperature_c)
        live = RESERVOIR.reservoir_oil_viscosity_cP * viscosity / reference
        darcy = self.well_permeability() * 9.869233e-16
        resistance = max(live * 1e-3 * math.log(RESERVOIR.drainage_radius_m / RESERVOIR.wellbore_radius_m), 1e-12)
        q_max = (
            2.0
            * math.pi
            * darcy
            * RESERVOIR.net_pay_thickness_m
            * p_res
            * 1e6
            / (RESERVOIR.oil_formation_volume_factor * resistance)
        ) * 86400.0
        return max(productivity * q_max, 0.0)

    def well_permeability(self) -> float:
        """Reservoir permeability used by the measurement model, mD."""
        return RESERVOIR.permeability_md

    def _pump_capacity(self, spm: float, stroke_m: float) -> float:
        return (
            spm
            * 2.0
            * 60.0
            * self.plunger_area_m2
            * stroke_m
            * PUMPING_UNIT.pump_efficiency
        )

    # -- models -----------------------------------------------------------
    def _predict_state(self, dt_days: float) -> np.ndarray:
        """Propagate the state forward by ``dt_days``.

        The skin is treated as a slow random walk, the thermal radius relaxes
        towards the injection equilibrium with the CSS conduction time scale,
        and the bottom hole temperature follows the two-mode decay.
        """
        state = self.state.copy()
        if dt_days > 0.0:
            # Thermal radius relaxes on the production time scale.
            state[STATE_RADIUS] += 0.02 * dt_days * max(0.0, 6.0 - state[STATE_RADIUS])
            # Bottom hole temperature relaxes towards the reservoir temperature.
            relaxation = math.exp(-dt_days / 25.0)
            reservoir = RESERVOIR.reservoir_temperature_c
            state[STATE_BHT] = reservoir + (state[STATE_BHT] - reservoir) * relaxation
        return state

    def _observation(
        self, state: np.ndarray, packet: TelemetryPacket
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicted measurement and the Jacobian :math:`H` of the model."""
        pip = state[STATE_PIP]
        skin = state[STATE_SKIN]
        radius = state[STATE_RADIUS]
        bht = state[STATE_BHT]

        temperature = float(np.clip(bht, 20.0, 220.0))
        hydrostatic = self._hydrostatic_pa(temperature, packet.water_cut)
        friction = self._friction_pressure_pa(temperature, packet.surface_flow_m3_per_day)
        casing = pip - hydrostatic - friction
        # Ramey-type attenuation between the pump and the wellhead.
        wellhead = temperature - 0.55 * (temperature - RESERVOIR.reservoir_temperature_c)
        inflow = self._reservoir_inflow(pip, skin, temperature)
        capacity = self._pump_capacity(packet.spm, packet.stroke_m)
        delivered = min(inflow, capacity)
        static = self.rod_weight_n + self.fluid_column_kg * 9.80665
        # The load cell responds to the differential *across the pump*, which a
        # starved pump cannot build: the card span collapses towards the static
        # rod-plus-column weight as the fillage falls.  Using the absolute
        # intake pressure here would be wrong, because the static column weight
        # is common to both branches of the card.
        fillage = inflow / capacity if capacity > 0.0 else 0.0
        span = (
            self.plunger_area_m2
            * PUMPING_UNIT.pump_differential_pa
            * min(fillage, 1.0)
        )
        card_min = static
        card_max = static + span

        predicted = np.array(
            [casing, wellhead, delivered, card_min, card_max, skin, bht], dtype=float
        )
        jacobian = np.zeros((MEAS_SIZE, STATE_SIZE), dtype=float)
        # d(casing)/d(PIP) = 1
        jacobian[MEAS_CASING_PRESSURE, STATE_PIP] = 1.0
        # d(wellhead T)/d(BHT) = 1 - the attenuation factor
        jacobian[MEAS_WELLHEAD_TEMPERATURE, STATE_BHT] = 0.45
        # d(flow)/d(PIP) and d(flow)/d(skin) from the Vogel derivative.
        ratio = min(max((pip / 1e6) / RESERVOIR.initial_reservoir_pressure_mpa, 0.0), 1.0)
        d_ratio_d_pip = 1.0 / (RESERVOIR.initial_reservoir_pressure_mpa * 1e6)
        d_effective_d_ratio = 1.0 / (1.0 + skin)
        d_productivity = -0.2 * d_effective_d_ratio - 1.6 * ratio * d_effective_d_ratio
        d_effective_d_skin = (1.0 - ratio) / (1.0 + skin) ** 2
        d_productivity_d_skin = -d_effective_d_skin * (0.2 + 1.6 * ratio)
        q_max = max(self._reservoir_inflow(pip, 0.0, temperature), 1e-9)
        # When the reservoir can supply more than the pump the delivered rate is
        # capped at the pump capacity and carries *no* information about the
        # reservoir state.  Leaving the sensitivity non-zero there would make
        # the filter attribute an unrelated residual to the skin factor and
        # drive it to its bound, so the derivatives are zeroed in that regime.
        pump_limited = inflow > capacity
        if not pump_limited:
            jacobian[MEAS_SURFACE_FLOW, STATE_PIP] = q_max * d_productivity * d_ratio_d_pip
            jacobian[MEAS_SURFACE_FLOW, STATE_SKIN] = q_max * d_productivity_d_skin
        # Weak prior on the skin factor.  A pump-limited well constrains the
        # skin only weakly, so without a prior the state would random-walk to a
        # bound.  The large variance makes this a soft constraint that only
        # matters when the flow channel is uninformative.
        jacobian[MEAS_SKIN_PRIOR, STATE_SKIN] = 1.0
        # The prior on the bottom hole temperature is deliberately soft: it lets
        # the reservoir model inform the state without overriding the measured
        # casing pressure, which is what actually pins the intake pressure.
        jacobian[MEAS_BHT_PRIOR, STATE_BHT] = 1.0
        return predicted, jacobian

    # -- filter -----------------------------------------------------------
    def reset(self) -> None:
        """Return the filter to its prior state."""
        self.__init__(  # type: ignore[misc]
            rod_weight_n=self.rod_weight_n,
            fluid_column_kg=self.fluid_column_kg,
            plunger_area_m2=self.plunger_area_m2,
            process_noise=np.diag(self.Q).tolist(),
            measurement_noise=np.diag(self.R).tolist(),
            well=self.well,
        )

    def step(
        self,
        packet: TelemetryPacket,
        dt_days: float = 1.0 / 24.0,
        bottom_hole_temperature_prior_c: Optional[float] = None,
    ) -> StateEstimate:
        """Assimilate one telemetry packet and return the posterior state."""
        if dt_days < 0.0:
            raise ValueError("dt_days must be non-negative")
        if bottom_hole_temperature_prior_c is not None:
            # Bounded to the injection temperature, which is the physical limit
            # of the reservoir the well is completed in.
            self.bottom_hole_temperature_prior = min(
                max(float(bottom_hole_temperature_prior_c),
                    RESERVOIR.reservoir_temperature_c),
                CSS.steam_temperature_c,
            )
        measured = np.array(
            [
                packet.casing_head_pressure_pa,
                packet.wellhead_temperature_c,
                packet.surface_flow_m3_per_day,
                packet.card_min_load_n,
                packet.card_max_load_n,
                self.skin_prior,
                self.bottom_hole_temperature_prior,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(measured)):
            raise ValueError("telemetry packet contains non-finite values")

        # --- predict ---------------------------------------------------
        self.state = self._predict_state(dt_days)
        predicted, jacobian = self._observation(self.state, packet)
        # F = I because the process model is a slow random walk in this frame.
        identity = np.eye(STATE_SIZE)
        self.covariance = (
            identity @ self.covariance @ identity.T + self.Q * max(dt_days, 1e-6) * 24.0
        )

        # --- update ----------------------------------------------------
        innovation = measured - predicted
        innovation_covariance = (
            jacobian @ self.covariance @ jacobian.T + self.R
        )
        try:
            gain = self.covariance @ jacobian.T @ np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            # A singular innovation covariance would mean a degenerate
            # observation; fall back to the prior rather than corrupting the
            # state, which is the numerically safe choice for a live filter.
            gain = np.zeros((STATE_SIZE, MEAS_SIZE))

        self.state = self.state + gain @ innovation
        identity = np.eye(STATE_SIZE)
        residual = identity - gain @ jacobian
        self.covariance = residual @ self.covariance @ residual.T + gain @ self.R @ gain.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

        # Enforce the physical bounds the linear model cannot guarantee.
        self.state[STATE_PIP] = float(
            np.clip(self.state[STATE_PIP], 101325.0, RESERVOIR.initial_reservoir_pressure_mpa * 2e6)
        )
        # A skin factor is a dimensionless pressure loss and is non-negative by
        # definition; allowing a negative value makes the Vogel evaluation
        # return a negative productivity.
        self.state[STATE_SKIN] = float(np.clip(self.state[STATE_SKIN], 0.0, 60.0))
        self.state[STATE_RADIUS] = float(
            np.clip(self.state[STATE_RADIUS], 0.0, RESERVOIR.drainage_radius_m)
        )
        self.state[STATE_BHT] = float(
            np.clip(
                self.state[STATE_BHT],
                RESERVOIR.reservoir_temperature_c,
                CSS.steam_temperature_c,
            )
        )
        self.initialised = True
        self._last_timestamp = packet.timestamp_s

        return StateEstimate(
            pump_intake_pressure_pa=float(self.state[STATE_PIP]),
            skin_factor=float(self.state[STATE_SKIN]),
            thermal_radius_m=float(self.state[STATE_RADIUS]),
            bottom_hole_temperature_c=float(self.state[STATE_BHT]),
            covariance=self.covariance.copy(),
            residual_norm=float(np.linalg.norm(innovation)),
            innovations=innovation,
        )

    def run(
        self,
        packets: Sequence[TelemetryPacket],
        dt_days: float = 1.0 / 24.0,
        bottom_hole_temperature_prior_c: Optional[float] = None,
    ) -> List[StateEstimate]:
        """Assimilate a sequence of packets."""
        return [
            self.step(
                packet,
                dt_days=dt_days,
                bottom_hole_temperature_prior_c=bottom_hole_temperature_prior_c,
            )
            for packet in packets
        ]

    def reservoir_inflow(self, estimate: StateEstimate, temperature_c: Optional[float] = None) -> float:
        """Model inflow at an estimated state, m^3/day."""
        return self._reservoir_inflow(
            estimate.pump_intake_pressure_pa,
            estimate.skin_factor,
            estimate.bottom_hole_temperature_c if temperature_c is None else temperature_c,
        )


def baghewala_state_filter() -> ExtendedKalmanFilter:
    """State filter configured for the Baghewala BAG-17 installation."""
    return ExtendedKalmanFilter()
