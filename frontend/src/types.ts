/**
 * Wire types for the ThermaTwin telemetry backend.
 *
 * These mirror the FastAPI/WebSocket contract exactly. The stream pushes a
 * `TelemetryFrame` every 500 ms; anything else on the socket is a control
 * message and must be tolerated.
 */

export type Phase = "INJECTION" | "SOAKING" | "PRODUCTION";

export type CardLabel =
  | "NORMAL_FULL_BARREL"
  | "FLUID_POUND"
  | "ROD_FLOATING"
  | "GAS_INTERFERENCE"
  | "PUMP_TAGGING"
  | "UNANCHORED_TUBING";

/** A single half-cycle of the polished-rod load trace. */
export interface CardTrace {
  position_m: number[];
  load_n: number[];
}

export interface TelemetryFrame {
  type: "telemetry";
  timestamp_s: number;
  well: {
    name: string;
    cycle: number;
    phase: Phase;
    day: number;
    days_into_phase: number;
  };
  card: {
    surface: CardTrace;
    downhole: CardTrace;
    stroke_m: number;
    min_load_n: number;
    max_load_n: number;
    load_span_n: number;
  };
  diagnosis: {
    label: CardLabel;
    confidence: number;
    probabilities: Record<string, number>;
    features: Record<string, number>;
    analytic_label: string;
    agrees_with_analytic: boolean;
  };
  wellbore: {
    depth_m: number[];
    pressure_mpa: number[];
    temperature_c: number[];
    liquid_holdup: number[];
    mixture_density_kg_m3: number[];
    regime: string[];
  };
  estimate: {
    pump_intake_pressure_mpa: number;
    skin_factor: number;
    thermal_radius_m: number;
    bottom_hole_temperature_c: number;
    pip_standard_error_mpa: number;
    residual_norm: number;
  };
  css: {
    cumulative_sor: number;
    instantaneous_sor: number;
    steam_tonnes: number;
    oil_bbl: number;
    cumulative_steam_m3: number;
    cumulative_oil_m3: number;
    cutoff_reached: boolean;
    cutoff_reason: string;
    chest_radius_m: number;
    pump_fillage: number;
  };
  recommendation: {
    spm: number;
    stroke_length_m: number;
    current_spm: number;
    current_stroke_m: number;
    spm_ratio: number;
    action: string;
    reason: string;
    pump_fillage: number;
    rod_float_risk_index: number;
    autonomous: boolean;
  };
  latency_ms: number;
}

/** Lifecycle state of the telemetry socket, surfaced in the status banner. */
export type ConnectionState = "connecting" | "live" | "reconnecting" | "offline";

export interface UseTelemetryResult {
  frame: TelemetryFrame | null;
  connection: ConnectionState;
  /** Consecutive failed attempts; 0 while live. Drives backoff display. */
  attempt: number;
  /** Round-trip latency of the most recent frame in ms, or null if unknown. */
  latencyMs: number | null;
  lastError: string | null;
}

// ---------------------------------------------------------------------------
// REST contract
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: string;
  [key: string]: unknown;
}

export interface ControlApplyRequest {
  spm: number;
  stroke_length_m: number;
  autonomous: boolean;
}

export interface ControlApplyResponse {
  accepted: boolean;
  spm: number;
  stroke_length_m: number;
  message: string;
}

export interface ScheduleResponse {
  schedule: {
    injection_days: number;
    soak_days: number;
    production_days: number;
    cycles: number;
  };
  cumulative_sor: number;
  oil_bbl: number;
  steam_m3: number;
  net_revenue_usd: number;
  feasible: boolean;
  reason: string;
  final_bht_c: number;
}

/** Only the fields the header needs; the config document is deep and dynamic. */
export interface WellConfig {
  well_name?: string;
  name?: string;
  wellbore_depth_m?: number;
  total_depth_m?: number;
  depth_m?: number;
  pump_depth_m?: number;
  [key: string]: unknown;
}
