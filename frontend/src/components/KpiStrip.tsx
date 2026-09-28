import { DASH, fmt, fmtInt, fmtPct } from "../api";
import type { TelemetryFrame } from "../types";

type Tone =
  | "text-slate-100"
  | "text-slate-500"
  | "text-signal-green"
  | "text-signal-amber"
  | "text-signal-red"
  | "text-signal-blue"
  | "text-signal-cyan";

interface TileSpec {
  key: string;
  label: string;
  value: string;
  unit?: string;
  tone: Tone;
  hint?: string;
}

interface Props {
  frame: TelemetryFrame | null;
  /** Server-reported round-trip latency; falls back to the socket's own measure. */
  latencyMs: number | null;
  loading: boolean;
}

function latencyTone(ms: number | null): Tone {
  if (ms === null || !Number.isFinite(ms)) return "text-slate-500";
  if (ms < 120) return "text-signal-green";
  if (ms < 400) return "text-signal-amber";
  return "text-signal-red";
}

function sorToneValue(sor: number | null | undefined): Tone {
  if (sor === null || sor === undefined || !Number.isFinite(sor)) return "text-slate-500";
  if (sor < 4.2) return "text-signal-green";
  if (sor <= 5) return "text-signal-amber";
  return "text-signal-red";
}

function fillageTone(fillage: number | null | undefined): Tone {
  if (fillage === null || fillage === undefined || !Number.isFinite(fillage)) return "text-slate-500";
  if (fillage >= 0.8) return "text-signal-green";
  if (fillage >= 0.5) return "text-signal-amber";
  return "text-signal-red";
}

function radiusTone(radius: number | null | undefined): Tone {
  if (radius === null || radius === undefined || !Number.isFinite(radius)) return "text-slate-500";
  if (radius < 12) return "text-signal-amber";
  return "text-signal-green";
}

function buildTiles(frame: TelemetryFrame | null, latencyMs: number | null): TileSpec[] {
  const est = frame?.estimate;
  const css = frame?.css;
  const rec = frame?.recommendation;
  return [
    {
      key: "bht",
      label: "Bottom-Hole Temp",
      value: est ? fmt(est.bottom_hole_temperature_c, 1) : DASH,
      unit: "°C",
      tone: "text-signal-amber",
      hint: est ? `σ ${fmt(est.pip_standard_error_mpa, 3)} MPa · residual ${fmt(est.residual_norm, 3)}` : undefined,
    },
    {
      key: "pip",
      label: "Pump Intake Press",
      value: est ? fmt(est.pump_intake_pressure_mpa, 2) : DASH,
      unit: "MPa",
      tone: "text-signal-blue",
      hint: est ? `skin ${fmt(est.skin_factor, 1)}` : undefined,
    },
    {
      key: "sor",
      label: "Cumulative SOR",
      value: css && Number.isFinite(css.cumulative_sor) ? fmt(css.cumulative_sor, 2) : DASH,
      unit: "bbl/bbl",
      tone: sorToneValue(css?.cumulative_sor),
      hint: css && Number.isFinite(css.instantaneous_sor) ? `now ${fmt(css.instantaneous_sor, 2)}` : undefined,
    },
    {
      key: "fillage",
      label: "Pump Fillage",
      value: css && Number.isFinite(css.pump_fillage) ? fmtPct(css.pump_fillage, 0) : DASH,
      unit: "%",
      tone: fillageTone(css?.pump_fillage),
      hint: rec && Number.isFinite(rec.rod_float_risk_index) ? `rod float risk ${fmtPct(rec.rod_float_risk_index, 0)}%` : undefined,
    },
    {
      key: "spm",
      label: "Pump Speed",
      value: rec && Number.isFinite(rec.current_spm) ? fmt(rec.current_spm, 1) : DASH,
      unit: "spm",
      tone: "text-slate-100",
      hint: rec && Number.isFinite(rec.spm) ? `AI target ${fmt(rec.spm, 1)}` : undefined,
    },
    {
      key: "chest",
      label: "Steam Chest",
      value: css && Number.isFinite(css.chest_radius_m) ? fmt(css.chest_radius_m, 1) : DASH,
      unit: "m",
      tone: radiusTone(css?.chest_radius_m),
      hint: est && Number.isFinite(est.thermal_radius_m) ? `thermal r ${fmt(est.thermal_radius_m, 1)} m` : undefined,
    },
    {
      key: "flow",
      label: "Wellhead Flow",
      value: css && Number.isFinite(css.instantaneous_sor) ? fmt(css.instantaneous_sor, 1) : DASH,
      unit: "bbl/d",
      tone: "text-signal-green",
      hint: css ? `${fmtInt(css.oil_bbl)} bbl cum.` : undefined,
    },
    {
      key: "latency",
      label: "Round-Trip Latency",
      value: latencyMs !== null ? fmt(latencyMs, 0) : DASH,
      unit: "ms",
      tone: latencyTone(latencyMs),
      hint: frame && Number.isFinite(frame.timestamp_s) ? `t+${fmt(frame.timestamp_s, 1)} s` : undefined,
    },
  ];
}

export default function KpiStrip({ frame, latencyMs, loading }: Props) {
  const tiles = buildTiles(frame, latencyMs);

  return (
    <section aria-label="Key operating indicators" className="rounded-md border border-scada-700 bg-scada-900">
      <h2 className="sr-only">Key operating indicators</h2>
      <ul className="grid grid-cols-2 gap-px bg-scada-700 sm:grid-cols-4 xl:grid-cols-8">
        {tiles.map((tile) => (
          <li key={tile.key} className="flex min-w-0 flex-col justify-center bg-scada-900 px-2 py-1.5">
            <span className="truncate text-2xs uppercase tracking-wide text-slate-400" title={tile.label}>
              {tile.label}
            </span>
            {loading && !frame ? (
              // Placeholder occupies the same space as a value, so the strip
              // never reflows when the first frame lands.
              <span
                aria-hidden="true"
                className="mt-0.5 h-4 w-14 animate-pulse rounded bg-scada-700"
              />
            ) : (
              <span className="flex items-baseline gap-1">
                <span className={`font-mono text-base font-semibold leading-tight tabular-nums ${tile.tone}`}>
                  {tile.value}
                </span>
                {tile.unit ? <span className="text-2xs text-slate-500">{tile.unit}</span> : null}
              </span>
            )}
            <span className="truncate text-2xs text-slate-500" title={tile.hint}>
              {tile.hint ?? (loading && !frame ? "connecting…" : " ")}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
