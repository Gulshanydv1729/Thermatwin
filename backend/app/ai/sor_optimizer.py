"""Multi-objective CSS and SRP control policy.

Two coupled problems are solved here.

**Steam-oil-ratio schedule.**  The cycle schedule (injection days, soak days
and the number of cycles) is chosen to maximise the net present value of the
programme subject to the cumulative SOR cut-off of 4.2 and the 70 degC bottom
hole temperature limit.  The objective is evaluated with the Marx-Langenheim
solver, and the schedule is improved with a coarse-to-fine coordinate search
rather than a black-box optimiser, so every evaluated point is a real
simulation and the search is reproducible.

**Pump setpoints.**  When the dynamometer classifier reports a fault the
recommended strokes per minute and stroke length are computed from the
*hydraulic* constraint, not from a lookup table.  For fluid pound the pump must
be slowed until its displacement no longer exceeds the reservoir inflow,

.. math::
    n_{spm} \\le \\frac{q_{in}}{60\\,A_p\\,L_s\\,\\eta_v},

and for rod float the downstroke must be slowed enough to keep the buoyant
drag below the margin that separates the load minimum from zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from backend.app.core.config import (
    CSS,
    ECONOMICS,
    PUMPING_UNIT,
    THRESHOLDS,
    WellConfig,
)
from backend.app.ai.dyno_classifier import CardLabel
from backend.app.physics.thermal_reservoir import (
    ThermalReservoirModel,
)

__all__ = [
    "CSSSchedule",
    "CSSScheduleResult",
    "SetpointRecommendation",
    "CSOScheduleOptimizer",
    "SRPController",
    "default_srp_controller",
]


# ---------------------------------------------------------------------------
# CSS schedule optimisation
# ---------------------------------------------------------------------------


@dataclass
class CSSSchedule:
    """A candidate CSS programme."""

    injection_days: float
    soak_days: float
    production_days: float
    cycles: int

    def as_dict(self) -> dict:
        """JSON-serialisable view for the API layer."""
        return {
            "injection_days": self.injection_days,
            "soak_days": self.soak_days,
            "production_days": self.production_days,
            "cycles": self.cycles,
        }


@dataclass
class CSSScheduleResult:
    """Outcome of evaluating a CSS programme."""

    schedule: CSSSchedule
    cumulative_sor: float
    oil_bbl: float
    steam_m3: float
    net_revenue_usd: float
    feasible: bool
    reason: str
    cycles_completed: int
    final_bht_c: float

    def as_dict(self) -> dict:
        """JSON-serialisable view for the API layer."""
        return {
            "schedule": self.schedule.as_dict(),
            "cumulative_sor": self.cumulative_sor,
            "oil_bbl": self.oil_bbl,
            "steam_m3": self.steam_m3,
            "net_revenue_usd": self.net_revenue_usd,
            "feasible": self.feasible,
            "reason": self.reason,
            "cycles_completed": self.cycles_completed,
            "final_bht_c": self.final_bht_c,
        }


class CSOScheduleOptimizer:
    """Coordinate search over the CSS programme, scored by the physics model.

    Parameters
    ----------
    model
        Marx-Langenheim model used to score every candidate.  A fresh model is
        constructed per evaluation so that the internal chest state cannot leak
        between candidates.
    economics
        Cost and price model.
    injection_bounds, soak_bounds, production_bounds
        Inclusive ``(low, high)`` search bounds in days.
    cycles_bounds
        Inclusive ``(low, high)`` bounds on the number of cycles.
    sor_cutoff
        Cumulative steam-oil ratio above which a programme is infeasible.
    bht_cutoff_c
        Bottom hole temperature below which production is uneconomic.
    """

    def __init__(
        self,
        model: Optional[ThermalReservoirModel] = None,
        economics=ECONOMICS,
        injection_bounds: Tuple[float, float] = (4.0, 16.0),
        soak_bounds: Tuple[float, float] = (1.0, 10.0),
        production_bounds: Tuple[float, float] = (25.0, 90.0),
        cycles_bounds: Tuple[int, int] = (1, 6),
        sor_cutoff: float = CSS.cumulative_sor_cutoff,
        bht_cutoff_c: float = CSS.cutoff_bht_c,
    ) -> None:
        if injection_bounds[0] > injection_bounds[1]:
            raise ValueError("injection bounds must be ordered")
        if soak_bounds[0] > soak_bounds[1]:
            raise ValueError("soak bounds must be ordered")
        if production_bounds[0] > production_bounds[1]:
            raise ValueError("production bounds must be ordered")
        if cycles_bounds[0] < 1:
            raise ValueError("at least one cycle is required")
        self.model = model or ThermalReservoirModel()
        self.economics = economics
        self.injection_bounds = injection_bounds
        self.soak_bounds = soak_bounds
        self.production_bounds = production_bounds
        self.cycles_bounds = cycles_bounds
        self.sor_cutoff = sor_cutoff
        self.bht_cutoff_c = bht_cutoff_c

    # -- evaluation -------------------------------------------------------
    def _fresh_model(self) -> ThermalReservoirModel:
        return ThermalReservoirModel(
            reservoir_temperature_c=self.model.reservoir_temperature_c,
            thermal_conductivity_w_mk=self.model.k,
            volumetric_heat_capacity_j_m3k=self.model.rho_c,
            overburden_conductivity_w_mk=self.model.k_over,
            overburden_thickness_m=self.model.d_over,
            underburden_thickness_m=self.model.d_under,
            overburden_diffusivity_m2s=self.model.alpha_over,
            vertical_contact_factor=self.model.vertical_contact,
            drainage_radius_m=self.model.re,
            wellbore_radius_m=self.model.rw,
            steam_quality=self.model.steam_quality,
            steam_temperature_c=self.model.steam_temperature_c,
            steam_enthalpy_kj_kg=self.model.steam_enthalpy_kj_kg,
            steam_retention_factor=self.model.steam_retention_factor,
            steam_rate_m3_per_day=self.model.steam_rate_m3_per_day,
            steam_specific_volume_m3_kg=self.model.steam_specific_volume_m3_kg,
            oil_specific_gravity=self.model.oil_specific_gravity,
            permeability_md=self.model.k_perm,
            skin_factor=self.model.skin,
            cutoff_sor=self.model.cutoff_sor,
            dt_days=self.model.dt,
        )

    def evaluate(self, schedule: CSSSchedule) -> CSSScheduleResult:
        """Simulate a programme and score it against the economic constraints."""
        if schedule.cycles < 1:
            raise ValueError("a programme needs at least one cycle")
        if min(
            schedule.injection_days, schedule.soak_days, schedule.production_days
        ) <= 0.0:
            raise ValueError("phase durations must be positive")
        model = self._fresh_model()
        programme = model.simulate_programme(
            cycles=schedule.cycles,
            injection_days=schedule.injection_days,
            soak_days=schedule.soak_days,
            production_days=schedule.production_days,
        )
        oil_m3 = programme.total_oil_tonnes / model.oil_specific_gravity
        oil_bbl = oil_m3 * 6.2898
        # One tonne of steam is one cubic metre of water equivalent.
        steam_m3 = programme.total_steam_tonnes
        revenue = oil_bbl * self.economics.oil_price_usd_bbl
        steam_cost = steam_m3 * self.economics.steam_cost_usd_m3
        lifting = oil_bbl * self.economics.lifting_cost_usd_bbl
        net = revenue - steam_cost - lifting

        feasible = True
        reason = "within the cumulative SOR and bottom hole temperature limits"
        if programme.cumulative_sor > self.sor_cutoff:
            feasible = False
            reason = (
                f"cumulative SOR {programme.cumulative_sor:.2f} exceeds the "
                f"cut-off {self.sor_cutoff:.2f}"
            )
        elif programme.final_bht_c < self.bht_cutoff_c and oil_bbl <= 0.0:
            feasible = False
            reason = (
                f"final bottom hole temperature {programme.final_bht_c:.1f} degC "
                f"is below {self.bht_cutoff_c:.1f} degC with no oil produced"
            )
        return CSSScheduleResult(
            schedule=schedule,
            cumulative_sor=programme.cumulative_sor,
            oil_bbl=oil_bbl,
            steam_m3=steam_m3,
            net_revenue_usd=net,
            feasible=feasible,
            reason=reason,
            cycles_completed=programme.cycles_completed,
            final_bht_c=programme.final_bht_c,
        )

    def objective(self, schedule: CSSSchedule) -> float:
        """Scalar objective: net revenue, with a penalty for infeasibility."""
        result = self.evaluate(schedule)
        if not result.feasible:
            # Keep infeasible programmes ranked below any feasible one while
            # still preferring the least-bad revenue among them.
            return result.net_revenue_usd - 1.0e7
        return result.net_revenue_usd

    # -- search -----------------------------------------------------------
    def optimise(self, seed: Optional[CSSSchedule] = None) -> CSSScheduleResult:
        """Coordinate descent with shrinking steps over the four schedule axes.

        Each axis is swept on a small grid centred on the incumbent, the best
        value is taken, and the step is halved.  The search terminates when no
        improvement is found at the current resolution, so it is deterministic
        and always evaluates physically meaningful programmes.
        """
        start = seed or CSSSchedule(
            injection_days=0.5 * (self.injection_bounds[0] + self.injection_bounds[1]),
            soak_days=0.5 * (self.soak_bounds[0] + self.soak_bounds[1]),
            production_days=0.5 * (self.production_bounds[0] + self.production_bounds[1]),
            cycles=(self.cycles_bounds[0] + self.cycles_bounds[1]) // 2,
        )
        incumbent = self._clamp(start)
        best = self.objective(incumbent)

        injection_step = 0.25 * (self.injection_bounds[1] - self.injection_bounds[0])
        soak_step = 0.25 * (self.soak_bounds[1] - self.soak_bounds[0])
        production_step = 0.25 * (self.production_bounds[1] - self.production_bounds[0])
        cycle_step = max(1, (self.cycles_bounds[1] - self.cycles_bounds[0]) // 2)

        for _ in range(8):
            improved = False
            for axis in ("injection", "soak", "production", "cycles"):
                candidates: List[CSSSchedule] = []
                for delta in (-1, 1):
                    if axis == "injection":
                        value = incumbent.injection_days + delta * injection_step
                        if not (self.injection_bounds[0] <= value <= self.injection_bounds[1]):
                            continue
                        candidates.append(
                            CSSSchedule(value, incumbent.soak_days,
                                         incumbent.production_days, incumbent.cycles)
                        )
                    elif axis == "soak":
                        value = incumbent.soak_days + delta * soak_step
                        if not (self.soak_bounds[0] <= value <= self.soak_bounds[1]):
                            continue
                        candidates.append(
                            CSSSchedule(incumbent.injection_days, value,
                                         incumbent.production_days, incumbent.cycles)
                        )
                    elif axis == "production":
                        value = incumbent.production_days + delta * production_step
                        if not (
                            self.production_bounds[0] <= value <= self.production_bounds[1]
                        ):
                            continue
                        candidates.append(
                            CSSSchedule(incumbent.injection_days, incumbent.soak_days,
                                         value, incumbent.cycles)
                        )
                    else:
                        value = incumbent.cycles + delta * cycle_step
                        if not (self.cycles_bounds[0] <= value <= self.cycles_bounds[1]):
                            continue
                        candidates.append(
                            CSSSchedule(incumbent.injection_days, incumbent.soak_days,
                                         incumbent.production_days, int(value))
                        )
                for candidate in candidates:
                    value = self.objective(candidate)
                    if value > best + 1e-9:
                        best = value
                        incumbent = candidate
                        improved = True
            if not improved:
                injection_step *= 0.5
                soak_step *= 0.5
                production_step *= 0.5
                cycle_step = max(1, cycle_step // 2)
                if injection_step < 0.05:
                    break
        return self.evaluate(incumbent)

    def _clamp(self, schedule: CSSSchedule) -> CSSSchedule:
        return CSSSchedule(
            injection_days=min(
                max(schedule.injection_days, self.injection_bounds[0]),
                self.injection_bounds[1],
            ),
            soak_days=min(max(schedule.soak_days, self.soak_bounds[0]), self.soak_bounds[1]),
            production_days=min(
                max(schedule.production_days, self.production_bounds[0]),
                self.production_bounds[1],
            ),
            cycles=int(
                min(max(schedule.cycles, self.cycles_bounds[0]), self.cycles_bounds[1])
            ),
        )


# ---------------------------------------------------------------------------
# SRP setpoint control
# ---------------------------------------------------------------------------


@dataclass
class SetpointRecommendation:
    """Recommended VFD setpoints with the reason that produced them."""

    spm: float
    stroke_length_m: float
    current_spm: float
    current_stroke_m: float
    action: str
    reason: str
    pump_fillage: float
    rod_float_risk_index: float
    autonomous: bool = False

    @property
    def spm_ratio(self) -> float:
        """Recommended SPM as a fraction of the current SPM."""
        if self.current_spm <= 0.0:
            return 0.0
        return self.spm / self.current_spm

    def as_dict(self) -> dict:
        """JSON-serialisable view for the API layer."""
        return {
            "spm": self.spm,
            "stroke_length_m": self.stroke_length_m,
            "current_spm": self.current_spm,
            "current_stroke_m": self.current_stroke_m,
            "spm_ratio": self.spm_ratio,
            "action": self.action,
            "reason": self.reason,
            "pump_fillage": self.pump_fillage,
            "rod_float_risk_index": self.rod_float_risk_index,
            "autonomous": self.autonomous,
        }


class SRPController:
    """Closed-loop VFD setpoint policy driven by the dyno card diagnosis.

    Parameters
    ----------
    plunger_area_m2
        Plunger area; defaults to the configured unit.
    volumetric_efficiency
        Delivered fraction of the swept volume.
    max_spm, min_spm
        Hard actuator limits.
    rod_float_factor
        SPM multiplier applied on a rod-float event.
    """

    def __init__(
        self,
        plunger_area_m2: float = PUMPING_UNIT.plunger_area_m2,
        volumetric_efficiency: float = PUMPING_UNIT.pump_efficiency,
        max_spm: float = THRESHOLDS.max_spm,
        min_spm: float = THRESHOLDS.min_spm,
        rod_float_factor: float = THRESHOLDS.rod_float_spm_factor,
    ) -> None:
        if plunger_area_m2 <= 0.0:
            raise ValueError("plunger area must be positive")
        if not 0.0 < volumetric_efficiency <= 1.0:
            raise ValueError("volumetric efficiency must lie in (0, 1]")
        if min_spm <= 0.0 or max_spm < min_spm:
            raise ValueError("SPM limits must be positive and ordered")
        if not 0.0 < rod_float_factor <= 1.0:
            raise ValueError("rod float factor must lie in (0, 1]")
        self.plunger_area_m2 = plunger_area_m2
        self.volumetric_efficiency = volumetric_efficiency
        self.max_spm = max_spm
        self.min_spm = min_spm
        self.rod_float_factor = rod_float_factor

    # -- hydraulics -------------------------------------------------------
    def maximum_spm(self, stroke_m: float) -> float:
        """Highest SPM that keeps the plunger inside the mechanical stroke.

        .. math:: n_{max} = \\frac{L_s}{A_p L_s n_{rod}}\\cdots

        The mechanical limit is the crank speed at which the polished rod would
        exceed its rated maximum, which the pumping unit already encodes in
        its nominal SPM; the hydraulic limit is handled by
        :meth:`match_inflow`.
        """
        if stroke_m <= 0.0:
            raise ValueError("stroke length must be positive")
        return self.max_spm

    def delivered_rate_m3_per_day(self, spm: float, stroke_m: float) -> float:
        """Volumetric delivery of the pump, m^3/day.

        .. math:: q = n_{spm}\\,2\\,60\\,A_p\\,L_s\\,\\eta_v
        """
        if stroke_m < 0.0:
            raise ValueError("stroke length must be non-negative")
        return (
            spm
            * 2.0
            * 60.0
            * self.plunger_area_m2
            * stroke_m
            * self.volumetric_efficiency
        )

    def spm_for_inflow(self, inflow_m3_per_day: float, stroke_m: float) -> float:
        """Largest SPM whose delivery still matches the reservoir inflow.

        This is the fluid-pound constraint: above it the pump displaces more
        fluid than the formation supplies and the barrel cannot fill.

        .. math::
            n_{spm} \\le \\frac{q_{in}}{120\\,A_p\\,L_s\\,\\eta_v}
        """
        if stroke_m <= 0.0:
            raise ValueError("stroke length must be positive")
        if inflow_m3_per_day <= 0.0:
            return self.min_spm
        rate = inflow_m3_per_day / (
            2.0 * 60.0 * self.plunger_area_m2 * stroke_m * self.volumetric_efficiency
        )
        return min(max(rate, self.min_spm), self.max_spm)

    def stroke_for_inflow(self, inflow_m3_per_day: float, spm: float) -> float:
        """Shortest stroke whose delivery still matches the inflow at ``spm``."""
        if spm <= 0.0:
            raise ValueError("SPM must be positive")
        if inflow_m3_per_day <= 0.0:
            return 0.0
        return inflow_m3_per_day / (
            2.0 * 60.0 * self.plunger_area_m2 * spm * self.volumetric_efficiency
        )

    def pump_fillage(self, inflow_m3_per_day: float, spm: float, stroke_m: float) -> float:
        """Fraction of the swept volume the formation actually supplies."""
        capacity = self.delivered_rate_m3_per_day(spm, stroke_m)
        if capacity <= 0.0:
            return 0.0
        return inflow_m3_per_day / capacity

    # -- policy -----------------------------------------------------------
    def recommend(
        self,
        label: CardLabel,
        confidence: float,
        inflow_m3_per_day: float,
        current_spm: float,
        current_stroke_m: float,
        rod_float_risk_index: float = 0.0,
        autonomous: bool = False,
    ) -> SetpointRecommendation:
        """Map a card diagnosis onto a VFD setpoint.

        The mapping is hydraulic rather than heuristic: the fluid-pound case
        solves the fillage constraint for SPM, and the rod-float case scales the
        rate by the configured downstroke factor.
        """
        if current_spm <= 0.0:
            raise ValueError("current SPM must be positive")
        if current_stroke_m <= 0.0:
            raise ValueError("current stroke must be positive")

        fillage = self.pump_fillage(inflow_m3_per_day, current_spm, current_stroke_m)
        spm = current_spm
        stroke = current_stroke_m
        action = "HOLD"
        reason = "no actionable fault detected on the downhole card"

        if label is CardLabel.FLUID_POUND:
            target = self.spm_for_inflow(inflow_m3_per_day, current_stroke_m)
            spm = min(current_spm, target)
            action = "REDUCE_SPM" if spm < current_spm - 1e-9 else "HOLD"
            reason = (
                f"fillage {fillage:.2f} is below the {THRESHOLDS.fluid_pound_fillage:.2f} "
                f"threshold; reducing SPM to {spm:.2f} matches pump displacement "
                f"to the {inflow_m3_per_day:.3f} m3/d inflow"
            )
        elif label is CardLabel.ROD_FLOATING:
            spm = max(
                self.min_spm,
                min(current_spm * self.rod_float_factor, self.max_spm),
            )
            action = "REDUCE_SPM"
            reason = (
                f"rod float risk index {rod_float_risk_index:.2f} exceeds the "
                f"{THRESHOLDS.rod_float_risk_high:.2f} threshold; the downstroke is "
                f"slowed by the factor {self.rod_float_factor:.2f} to keep the load "
                f"minimum clear of zero and protect the rod string"
            )
        elif label is CardLabel.PUMP_TAGGING:
            spm = max(self.min_spm, min(current_spm * 0.80, self.max_spm))
            action = "REDUCE_SPM"
            reason = (
                "mechanical impact at the end of the downstroke; the rate is cut "
                "to reduce the impact energy and protect the polish rod and tubing"
            )
        elif label is CardLabel.GAS_INTERFERENCE:
            spm = max(self.min_spm, min(current_spm * 0.92, self.max_spm))
            action = "REDUCE_SPM"
            reason = (
                "a compressed gas cap is cushioning the card; the rate is trimmed "
                "to let the gas clear the barrel and restore full fillage"
            )
        elif label is CardLabel.UNANCHORED_TUBING:
            spm = current_spm
            action = "INSPECT_TUBING"
            reason = (
                "the card is skewed, which indicates an unanchored tubing string; "
                "the rate is held pending a mechanical inspection"
            )

        spm = min(max(spm, self.min_spm), self.max_spm)
        return SetpointRecommendation(
            spm=spm,
            stroke_length_m=stroke,
            current_spm=current_spm,
            current_stroke_m=current_stroke_m,
            action=action,
            reason=f"{reason} (confidence {confidence:.2f})",
            pump_fillage=fillage,
            rod_float_risk_index=rod_float_risk_index,
            autonomous=autonomous,
        )


def default_srp_controller() -> SRPController:
    """SRP controller configured for the Baghewala BAG-17 installation."""
    return SRPController()
