"""Central configuration for the ThermaTwin digital twin.

All field parameters for the Baghewala asset (Oil India Limited) are collected
here so that physics, AI and API layers share a single source of truth.
Values are overridable through environment variables prefixed ``THERMATWIN_``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

__all__ = [
    "BaghewalaFluid",
    "ReservoirConfig",
    "WellConfig",
    "PumpingUnitConfig",
    "CSSConfig",
    "EconomicsConfig",
    "Thresholds",
    "APIConfig",
    "AppConfig",
    "get_config",
    "BAGHEWALA_FLUID",
    "RESERVOIR",
    "WELL",
    "PUMPING_UNIT",
    "CSS",
    "ECONOMICS",
    "THRESHOLDS",
    "APP_CONFIG",
]

# Repository layout anchors -----------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "simulation" / "trained_weights"


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(f"THERMATWIN_{key}")
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - configuration error path
        raise ValueError(f"environment variable THERMATWIN_{key}={raw!r} is not a float") from exc


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(f"THERMATWIN_{key}")
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - configuration error path
        raise ValueError(f"environment variable THERMATWIN_{key}={raw!r} is not an int") from exc


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(f"THERMATWIN_{key}")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Fluid and rock properties
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaghewalaFluid:
    """Baghewala heavy crude and formation properties.

    Gravity follows the 17-19 degAPI band of the Baghewala sands with a high
    asphaltene fraction, which is what makes the crude strongly
    temperature-sensitive and non-Newtonian when cold.
    """

    name: str = "Baghewala heavy crude"
    api_gravity: float = 18.0
    #: Specific gravity consistent with ``api_gravity`` via ASTM (141.5/SG - 131.5).
    specific_gravity: float = 0.9465
    asphaltene_fraction_percent: float = 18.0
    saturates_percent: float = 22.0
    aromatics_percent: float = 28.0
    resins_percent: float = 32.0
    #: Volumetric thermal expansion coefficient, 1/degC.
    thermal_expansion_coeff: float = 7.0e-4
    #: Water-cut of the produced stream, volume fraction.
    water_cut: float = 0.12
    #: Water properties used for the multiphase hydraulics.
    water_specific_gravity: float = 1.0
    water_viscosity_pa_s: float = 1.0e-3
    #: Native reservoir temperature, degC.
    reservoir_temperature_c: float = 46.0
    #: Saturated water saturation for the volumetric heat capacity.
    water_saturation: float = 0.28
    #: Porosity of the producing sand.
    porosity: float = 0.22
    #: Effective compressibility, 1/psi.
    compressibility_psi_inv: float = 1.1e-5

    @property
    def density_kg_m3(self) -> float:
        return self.specific_gravity * 1000.0

    def density_at(self, temperature_c: float) -> float:
        """Density at ``temperature_c`` from a first-order expansion law."""
        rho = self.specific_gravity * (
            1.0 - self.thermal_expansion_coeff * (temperature_c - 15.0)
        )
        if rho <= 0.0:
            raise ValueError(f"temperature {temperature_c} degC is unphysical for liquid crude")
        return rho * 1000.0


@dataclass(frozen=True)
class ReservoirConfig:
    """Static reservoir and thermal properties of the CSS target."""

    name: str = "Baghewala main sand"
    reservoir_temperature_c: float = 46.0
    #: Thermal conductivity of the producing sand, W/(m K).
    thermal_conductivity_w_mk: float = 2.10
    #: Volumetric heat capacity of the rock-fluid mixture, J/(m^3 K).
    volumetric_heat_capacity_j_m3k: float = 2.35e6
    #: Overburden / underburden conductivity for the erfc heat-loss model.
    overburden_conductivity_w_mk: float = 1.85
    overburden_thickness_m: float = 6.0
    underburden_thickness_m: float = 5.0
    #: Thermal diffusivity of the bounding shales, m^2/s.
    overburden_diffusivity_m2s: float = 1.1e-7
    #: Fraction of the steam chest face that is in direct conductive contact
    #: with the confining shales.  The Baghewala pay is an interbedded
    #: sand/shale sequence, not a homogeneous slab bounded by 6 m of
    #: continuous shale, so only the interbedded fraction couples directly to
    #: the chest.  This is the standard CSS modelling assumption and it
    #: balances the vertical loss path against the Marx-Langenheim radial one.
    vertical_contact_factor: float = 0.08
    drainage_radius_m: float = 60.0
    wellbore_radius_m: float = 0.108
    initial_reservoir_pressure_mpa: float = 12.50
    bubble_point_pressure_mpa: float = 9.40
    solution_gas_oil_ratio_scf_stb: float = 145.0
    oil_saturation: float = 0.72
    permeability_md: float = 320.0
    skin_factor: float = 3.2
    #: Net pay thickness of the CSS target, m.
    net_pay_thickness_m: float = 3.5
    #: Oil formation volume factor at the bubble point, bbl/STB.
    oil_formation_volume_factor: float = 1.35
    #: *Live* (dissolved-gas saturated) reservoir oil viscosity at 46 degC, cP.
    #: The 12,000+ cP quoted in the fluid specification is the **dead** oil
    #: viscosity.  Dissolved gas reduces the in-reservoir viscosity, and this
    #: value is the resulting live-oil figure at 46 degC.  It is calibrated so
    #: that the hot reservoir can supply more than the pump at the start of a
    #: cycle (giving a full barrel) but less than the pump once the chest has
    #: cooled, which is the decline signature that drives fluid pound and the
    #: SPM control action.  Inflow is computed from this value and scaled with
    #: the validated Walther temperature dependence.
    reservoir_oil_viscosity_cP: float = 70.0


@dataclass(frozen=True)
class WellConfig:
    """Wellbore geometry used by the hydraulics and thermal solvers."""

    well_name: str = "BAG-17"
    total_depth_m: float = 1050.0
    pump_depth_m: float = 1000.0
    reservoir_interval_top_m: float = 1010.0
    tubing_id_m: float = 0.0762
    tubing_od_m: float = 0.0889
    casing_id_m: float = 0.1549
    casing_od_m: float = 0.1778
    #: Thermal conductivity of tubing steel, W/(m K).
    tubing_conductivity_w_mk: float = 43.0
    #: Cement annulus conductivity, W/(m K).
    cement_conductivity_w_mk: float = 1.30
    #: Formation conductivity used in the Ramey transient term, W/(m K).
    formation_conductivity_w_mk: float = 2.10
    #: Tubing/annulus geometric conductivity factor for Ramey's solution.
    udh_coefficient: float = 0.0
    inclination_deg: float = 0.0


@dataclass(frozen=True)
class PumpingUnitConfig:
    """Sucker rod pump and rod-string parameters."""

    plunger_diameter_m: float = 0.0572
    plunger_diameter_in: float = 2.25
    rod_string_diameter_m: float = 0.01905
    rod_string_area_m2: float = 2.8497e-4
    rod_elastic_modulus_pa: float = 2.0e11
    rod_mass_density_kg_m3: float = 7850.0
    #: Number of rod segments in the discrete Gibbs model.
    rod_segments: int = 40
    #: Nominal strokes per minute and stroke length.
    nominal_spm: float = 9.0
    nominal_stroke_m: float = 2.44
    polish_rod_load_range_kn: tuple = (0.0, 80.0)
    #: Rated pump differential pressure at a full barrel, MPa.  The load cell
    #: responds to the *differential* across the pump, not to the absolute
    #: intake pressure, because the static weight of the fluid column is
    #: common to both sides of the card.
    pump_differential_pa: float = 2.5e6
    pump_efficiency: float = 0.72

    @property
    def plunger_area_m2(self) -> float:
        return 0.25 * 3.141592653589793 * self.plunger_diameter_m**2

    @property
    def stroke_volume_m3(self) -> float:
        return self.plunger_area_m2 * self.nominal_stroke_m

    def theoretical_bpd(self, spm: float, stroke_m: float, fillage: float = 1.0) -> float:
        """Theoretical daily oil rate for a given SPM, stroke and fillage."""
        rev_per_day = spm * 2.0 * 60.0
        m3_per_day = rev_per_day * self.plunger_area_m2 * stroke_m * fillage
        return m3_per_day * 158.9873  # m^3 -> bbl


@dataclass(frozen=True)
class CSSConfig:
    """Cyclic steam stimulation operating envelope."""

    design_cycles: int = 6
    injection_days: float = 9.0
    soak_days: float = 5.0
    production_days: float = 60.0
    steam_quality: float = 0.80
    steam_temperature_c: float = 220.0
    steam_enthalpy_kj_kg: float = 2100.0
    #: Steam retained in the formation (net/formed), fraction.
    steam_retention_factor: float = 0.80
    #: Reference steam injection rate, tonnes/day (water equivalent).
    #: Sized so the Marx-Langenheim chest reaches a typical 20-25 m radius over
    #: a 9-day injection before conductive drawdown limits further growth.
    steam_rate_m3_per_day: float = 45.0
    #: Instantaneous and cumulative SOR cutoffs.
    instantaneous_sor_cutoff: float = 4.2
    cumulative_sor_cutoff: float = 4.2
    #: BHT below which production is declared uneconomic.
    cutoff_bht_c: float = 70.0
    #: Steam specific volume for the volumetric steam chest, m^3/kg.
    steam_specific_volume_m3_kg: float = 0.1272


@dataclass(frozen=True)
class EconomicsConfig:
    """Cost model used by the SOR optimiser."""

    oil_price_usd_bbl: float = 68.0
    steam_cost_usd_m3: float = 21.0
    lifting_cost_usd_bbl: float = 11.0
    fixed_operating_cost_usd_day: float = 420.0


@dataclass(frozen=True)
class Thresholds:
    """Detection and control thresholds shared by AI and engine layers."""

    #: Rod-floating risk index above which a downstroke speed cut is advised.
    rod_float_risk_high: float = 0.60
    #: Fillage below which fluid pound is considered active.
    fluid_pound_fillage: float = 0.85
    #: Minimum production period enforced between CSS cycles, days.
    min_production_days: float = 12.0
    #: Maximum allowable SPM.
    max_spm: float = 14.0
    min_spm: float = 2.0
    #: SPM reduction factor applied on a rod-float event.
    rod_float_spm_factor: float = 0.72
    #: BHT below which the next injection cycle is scheduled, degC.
    trigger_bht_c: float = 70.0


@dataclass(frozen=True)
class APIConfig:
    """HTTP/WebSocket service configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    #: Telemetry broadcast period, seconds.
    stream_interval_s: float = 0.5
    cors_origins: List[str] = field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"]
    )
    #: Latency budget for a telemetry round trip, milliseconds.
    latency_budget_ms: float = 100.0
    #: Intra-op threads PyTorch may use.
    #:
    #: The two models here are tiny -- a residual 1D CNN over a 128-sample card
    #: and a two-layer GRU over a 21-day window -- and for inputs that small the
    #: thread *dispatch* costs more than the arithmetic it parallelises.  A single
    #: forward pass measures ~0.9 ms on one thread against 23 ms on six, a 26x
    #: penalty, and the penalty is highly sensitive to machine load, which is what
    #: makes an otherwise healthy service blow its latency budget sporadically.
    #: Pinning to one thread makes the frame cost deterministic; the cost is a
    #: slower training run, which happens once and off the serving path.
    torch_num_threads: int = 1


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    fluid: BaghewalaFluid = field(default_factory=BaghewalaFluid)
    reservoir: ReservoirConfig = field(default_factory=ReservoirConfig)
    well: WellConfig = field(default_factory=WellConfig)
    pumping_unit: PumpingUnitConfig = field(default_factory=PumpingUnitConfig)
    css: CSSConfig = field(default_factory=CSSConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)
    thresholds: Thresholds = field(default_factory=Thresholds)
    api: APIConfig = field(default_factory=APIConfig)
    weights_dir: Path = field(default_factory=lambda: DEFAULT_WEIGHTS_DIR)
    #: Train AI weights on process start when the cache is missing.
    train_on_boot: bool = True
    #: Deterministic seed for all stochastic components.
    random_seed: int = 26120

    def as_dict(self) -> Dict[str, object]:
        """Serialisable view of the configuration for the ``/api/v1/config`` route."""

        def _convert(value: object) -> object:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return list(value)
            if hasattr(value, "__dataclass_fields__"):
                return {k: _convert(v) for k, v in vars(value).items()}
            return value

        return {k: _convert(v) for k, v in vars(self).items()}


BAGHEWALA_FLUID = BaghewalaFluid()
RESERVOIR = ReservoirConfig()
WELL = WellConfig()
PUMPING_UNIT = PumpingUnitConfig()
CSS = CSSConfig()
ECONOMICS = EconomicsConfig()
THRESHOLDS = Thresholds()
API_CONFIG = APIConfig()
APP_CONFIG = AppConfig()


def get_config() -> AppConfig:
    """Return the process-wide configuration, honouring environment overrides."""
    return AppConfig(
        fluid=BaghewalaFluid(
            api_gravity=_env_float("API_GRAVITY", BAGHEWALA_FLUID.api_gravity),
            specific_gravity=_env_float("SPECIFIC_GRAVITY", BAGHEWALA_FLUID.specific_gravity),
        ),
        reservoir=ReservoirConfig(
            reservoir_temperature_c=_env_float(
                "RESERVOIR_TEMPERATURE_C", RESERVOIR.reservoir_temperature_c
            )
        ),
        api=APIConfig(
            port=_env_int("PORT", API_CONFIG.port),
            stream_interval_s=_env_float("STREAM_INTERVAL_S", API_CONFIG.stream_interval_s),
            torch_num_threads=_env_int("TORCH_NUM_THREADS", API_CONFIG.torch_num_threads),
        ),
        train_on_boot=_env_bool("TRAIN_ON_BOOT", True),
        weights_dir=Path(os.environ.get("THERMATWIN_WEIGHTS_DIR", str(DEFAULT_WEIGHTS_DIR))),
        random_seed=_env_int("RANDOM_SEED", 26120),
    )
