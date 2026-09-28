import { DASH, fmt, fmtPct } from "../api";
import type { CardLabel, ConnectionState, TelemetryFrame } from "../types";

const CONNECTION_META: Record<
  ConnectionState,
  { label: string; dot: string; text: string; bg: string; pulse: boolean }
> = {
  connecting: { label: "Connecting", dot: "bg-slate-400", text: "text-slate-300", bg: "bg-scada-700", pulse: true },
  live: { label: "Live", dot: "bg-signal-green", text: "text-signal-green", bg: "bg-signal-green/15", pulse: true },
  reconnecting: { label: "Reconnecting", dot: "bg-signal-amber", text: "text-signal-amber", bg: "bg-signal-amber/15", pulse: true },
  offline: { label: "Offline", dot: "bg-signal-red", text: "text-signal-red", bg: "bg-signal-red/15", pulse: false },
};

type Severity = "critical" | "warning" | "info" | "ok";

/**
 * Severity ranking for a diagnosed card fault. PUMP_TAGGING is a hard
 * mechanical impact, so it is the only condition that trips the critical alarm.
 */
const FAULT_SEVERITY: Record<CardLabel, { severity: Severity; text: string }> = {
  NORMAL_FULL_BARREL: { severity: "ok", text: "No active card alarm" },
  FLUID_POUND: { severity: "critical", text: "Fluid pound — upstroke load collapse" },
  ROD_FLOATING: { severity: "warning", text: "Rod floating — downstroke load depression" },
  GAS_INTERFERENCE: { severity: "warning", text: "Gas interference — rounded card" },
  PUMP_TAGGING: { severity: "critical", text: "Pump tagging — mechanical impact at BDC" },
  UNANCHORED_TUBING: { severity: "warning", text: "Unanchored tubing — load peak lag" },
};

const SEVERITY_STYLES: Record<Severity, { text: string; bg: string; border: string }> = {
  critical: { text: "text-signal-red", bg: "bg-red-950/50", border: "border-signal-red" },
  warning: { text: "text-signal-amber", bg: "bg-amber-950/30", border: "border-signal-amber" },
  info: { text: "text-signal-blue", bg: "bg-signal-blue/10", border: "border-signal-blue" },
  ok: { text: "text-signal-green", bg: "bg-signal-green/10", border: "border-signal-green" },
};

function latencyTone(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) return "text-slate-500";
  if (ms < 120) return "text-signal-green";
  if (ms < 400) return "text-signal-amber";
  return "text-signal-red";
}

interface Props {
  connection: ConnectionState;
  frame: TelemetryFrame | null;
  latencyMs: number | null;
  /** Consecutive reconnect attempts, shown while backing off. */
  attempt: number;
  lastError: string | null;
  wellName: string | null;
  /** Health probe result; null until known. */
  backendHealthy: boolean | null;
}

export default function StatusBanner({
  connection,
  frame,
  latencyMs,
  attempt,
  lastError,
  wellName,
  backendHealthy,
}: Props) {
  const meta = CONNECTION_META[connection];
  const label = frame?.diagnosis.label ?? "NORMAL_FULL_BARREL";
  const fault = FAULT_SEVERITY[label];
  const severity: Severity = fault?.severity ?? "info";
  const style = SEVERITY_STYLES[severity];

  const confidence = frame?.diagnosis.confidence ?? null;
  const agrees = frame?.diagnosis.agrees_with_analytic;
  const analyticLabel = frame?.diagnosis.analytic_label ?? "";

  return (
    <header
      aria-label="System status"
      className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-md border border-scada-700 bg-scada-900 px-3 py-1.5"
    >
      {/* Identity */}
      <div className="flex min-w-0 items-baseline gap-2">
        <span className="text-sm font-bold uppercase tracking-widest text-slate-100">ThermaTwin</span>
        <span className="hidden text-2xs uppercase tracking-wide text-slate-500 sm:inline">SCADA</span>
        <span className="truncate font-mono text-xs text-amber-scada" title={wellName ?? undefined}>
          {wellName ?? DASH}
        </span>
      </div>

      {/* Connection state */}
      <div
        role="status"
        aria-live="polite"
        className={`flex items-center gap-1.5 rounded px-1.5 py-0.5 ${meta.bg}`}
      >
        <span className="relative flex h-2 w-2" aria-hidden="true">
          {meta.pulse ? (
            <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${meta.dot}`} />
          ) : null}
          <span className={`relative inline-flex h-2 w-2 rounded-full ${meta.dot}`} />
        </span>
        <span className={`text-2xs font-semibold uppercase tracking-wide ${meta.text}`}>{meta.label}</span>
        {connection === "reconnecting" && attempt > 0 ? (
          <span className="font-mono text-2xs tabular-nums text-slate-400">
            attempt {attempt} · backoff {Math.min(30000, 1000 * 2 ** (attempt - 1)) / 1000}s
          </span>
        ) : null}
      </div>

      {/* Latency */}
      <div className="flex items-baseline gap-1 text-2xs text-slate-400">
        <span className="uppercase tracking-wide">WS latency</span>
        <span className={`font-mono tabular-nums ${latencyTone(latencyMs)}`}>
          {latencyMs !== null ? `${fmt(latencyMs, 0)} ms` : DASH}
        </span>
      </div>

      {/* REST backend health */}
      <div className="flex items-baseline gap-1 text-2xs text-slate-400">
        <span className="uppercase tracking-wide">API</span>
        <span
          className={`font-mono uppercase ${
            backendHealthy === null
              ? "text-slate-500"
              : backendHealthy
                ? "text-signal-green"
                : "text-signal-red"
          }`}
        >
          {backendHealthy === null ? "unknown" : backendHealthy ? "ok" : "down"}
        </span>
      </div>

      <div className="ml-auto" />

      {/* Diagnosis alarm */}
      <div
        role="alert"
        className={`flex min-w-0 items-center gap-2 rounded border px-2 py-0.5 ${style.border} ${style.bg}`}
      >
        <span
          aria-hidden="true"
          className={`h-2 w-2 shrink-0 rounded-full ${
            severity === "critical"
              ? "animate-pulse bg-signal-red"
              : severity === "warning"
                ? "bg-signal-amber"
                : severity === "ok"
                  ? "bg-signal-green"
                  : "bg-signal-blue"
          }`}
        />
        <div className="min-w-0">
          <p className={`truncate text-2xs font-semibold uppercase tracking-wide ${style.text}`}>
            {label.replace(/_/g, " ")}
          </p>
          <p className="truncate text-2xs text-slate-400" title={fault?.text}>
            {frame ? `${fault?.text ?? "Unrecognised diagnosis"} · confidence ${fmtPct(confidence, 0)}%` : "Awaiting first telemetry frame…"}
          </p>
        </div>
      </div>

      {/* Analytic cross-check and stream errors are secondary but must be visible. */}
      {frame && agrees === false && analyticLabel ? (
        <div className="flex items-baseline gap-1 rounded border border-signal-amber/60 px-1.5 py-0.5 text-2xs">
          <span className="uppercase tracking-wide text-slate-400">Rule check</span>
          <span className="font-mono uppercase text-signal-amber">{analyticLabel.replace(/_/g, " ")}</span>
        </div>
      ) : null}

      {lastError && connection !== "live" ? (
        <p className="w-full truncate text-2xs text-slate-500" title={lastError}>
          {lastError}
        </p>
      ) : null}
    </header>
  );
}
