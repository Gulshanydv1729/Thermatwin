import { useCallback, useEffect, useId, useState } from "react";
import { ApiError, DASH, applySetpoint, clamp, fmt, fmtPct } from "../api";
import type { TelemetryFrame } from "../types";

/**
 * Stroke envelope the VFD accepts, as a fallback when the config document is
 * unavailable. `THRESHOLDS.min_spm`/`max_spm` are published under
 * `config.thresholds`, so the SPM bounds come from the backend at runtime; the
 * stroke bound is not in the config document, so this mirrors the check in
 * `DigitalTwin.apply_setpoint`.
 */
const STROKE_MIN = 0.5;
const STROKE_MAX = 3.5;

type ApplyState =
  | { kind: "idle" }
  | { kind: "pending" }
  | { kind: "success"; message: string; spm: number; stroke: number }
  | { kind: "error"; message: string };

function riskTone(risk: number | null | undefined): { text: string; bar: string; track: string } {
  if (risk === null || risk === undefined || !Number.isFinite(risk)) {
    return { text: "text-slate-500", bar: "bg-scada-500", track: "bg-scada-700" };
  }
  if (risk < 0.35) return { text: "text-signal-green", bar: "bg-signal-green", track: "bg-scada-700" };
  if (risk < 0.6) return { text: "text-signal-amber", bar: "bg-signal-amber", track: "bg-scada-700" };
  return { text: "text-signal-red", bar: "bg-signal-red", track: "bg-scada-700" };
}

function fillageTone(fillage: number | null | undefined): string {
  if (fillage === null || fillage === undefined || !Number.isFinite(fillage)) return "text-slate-500";
  if (fillage >= 0.8) return "text-signal-green";
  if (fillage >= 0.5) return "text-signal-amber";
  return "text-signal-red";
}

/** A labelled horizontal meter used for the risk index and fillage. */
function Meter({
  label,
  value,
  display,
  tone,
  ariaLabel,
}: {
  label: string;
  value: number;
  display: string;
  tone: { text: string; bar: string; track: string };
  ariaLabel: string;
}) {
  const pct = clamp(Number.isFinite(value) ? value * 100 : 0, 0, 100);
  return (
    <div>
      <div className="mb-0.5 flex items-baseline justify-between gap-2">
        <span className="text-2xs uppercase tracking-wide text-slate-400">{label}</span>
        <span className={`font-mono text-xs font-semibold tabular-nums ${tone.text}`}>{display}</span>
      </div>
      <div
        role="meter"
        aria-label={ariaLabel}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(pct)}
        className={`h-1.5 w-full overflow-hidden rounded-full ${tone.track}`}
      >
        <div
          className={`h-full rounded-full ${tone.bar} transition-[width] duration-300`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

/** SPM envelope, as published by the backend under `config.thresholds`. */
export interface SpmEnvelope {
  min: number;
  max: number;
}

interface Props {
  frame: TelemetryFrame | null;
  /** Backend-declared SPM limits; falls back to a conservative local default. */
  spmRange?: SpmEnvelope | null;
}

const DEFAULT_SPM_RANGE: SpmEnvelope = { min: 2, max: 14 };

export default function VFDControlPanel({ frame, spmRange }: Props) {
  const rec = frame?.recommendation;
  const spmId = useId();
  const strokeId = useId();

  const [spmInput, setSpmInput] = useState<string>("");
  const [strokeInput, setStrokeInput] = useState<string>("");
  const [autonomous, setAutonomous] = useState<boolean>(true);
  const [applyState, setApplyState] = useState<ApplyState>({ kind: "idle" });

  // Adopt the AI recommendation whenever the stream changes it, until the
  // operator edits the field. Editing re-arms tracking on the next change only
  // if the operator has not typed.
  const recommendedSpm = rec?.spm ?? null;
  const recommendedStroke = rec?.stroke_length_m ?? null;
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    if (dirty) return;
    setSpmInput(recommendedSpm === null ? "" : String(recommendedSpm));
  }, [recommendedSpm, dirty]);

  useEffect(() => {
    if (dirty) return;
    setStrokeInput(recommendedStroke === null ? "" : String(recommendedStroke));
  }, [recommendedStroke, dirty]);

  useEffect(() => {
    if (rec?.autonomous !== undefined) setAutonomous(rec.autonomous);
  }, [rec?.autonomous]);

  const SPM_MIN = spmRange?.min ?? DEFAULT_SPM_RANGE.min;
  const SPM_MAX = spmRange?.max ?? DEFAULT_SPM_RANGE.max;

  const parsedSpm = Number.parseFloat(spmInput);
  const parsedStroke = Number.parseFloat(strokeInput);
  const spmValid = Number.isFinite(parsedSpm) && parsedSpm >= SPM_MIN && parsedSpm <= SPM_MAX;
  const strokeValid =
    Number.isFinite(parsedStroke) && parsedStroke >= STROKE_MIN && parsedStroke <= STROKE_MAX;

  const onApply = useCallback(async () => {
    if (!spmValid || !strokeValid) {
      setApplyState({
        kind: "error",
        message: `Setpoint out of range. SPM must be ${SPM_MIN}–${SPM_MAX} and stroke ${STROKE_MIN}–${STROKE_MAX} m.`,
      });
      return;
    }
    setApplyState({ kind: "pending" });
    try {
      const response = await applySetpoint({
        spm: parsedSpm,
        stroke_length_m: parsedStroke,
        autonomous,
      });
      if (response.accepted) {
        setApplyState({
          kind: "success",
          message: response.message || "Setpoint accepted by the VFD controller.",
          spm: response.spm,
          stroke: response.stroke_length_m,
        });
      } else {
        setApplyState({
          kind: "error",
          message: response.message || "The VFD controller rejected the setpoint.",
        });
      }
    } catch (cause) {
      setApplyState({
        kind: "error",
        message:
          cause instanceof ApiError
            ? cause.message
            : cause instanceof Error
              ? cause.message
              : "Unexpected error applying the setpoint.",
      });
    }
  }, [SPM_MAX, SPM_MIN, autonomous, parsedSpm, parsedStroke, spmValid, strokeValid]);

  const risk = rec?.rod_float_risk_index ?? null;
  const fillage = rec?.pump_fillage ?? null;
  const currentSpm = rec?.current_spm ?? null;
  const spmDelta =
    recommendedSpm !== null && currentSpm !== null && Number.isFinite(recommendedSpm) && Number.isFinite(currentSpm)
      ? recommendedSpm - currentSpm
      : null;

  const actionTone =
    rec?.action === "HOLD" || rec?.action === "MAINTAIN"
      ? "text-signal-green"
      : rec?.action === "REDUCE_SPM" || rec?.action === "SLOW_DOWN"
        ? "text-signal-amber"
        : "text-signal-blue";

  return (
    <section
      aria-label="VFD speed control and AI setpoint recommendation"
      className="flex h-full min-h-0 flex-col rounded-md border border-scada-700 bg-scada-900"
    >
      <header className="flex items-center justify-between gap-2 border-b border-scada-700 px-3 py-1.5">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-300">VFD Speed Control</h2>
        <span
          className={`rounded px-1.5 py-0.5 text-2xs font-semibold uppercase tracking-wide ${
            autonomous ? "bg-signal-green/15 text-signal-green" : "bg-scada-700 text-slate-400"
          }`}
        >
          {autonomous ? "Autonomous" : "Manual"}
        </span>
      </header>

      <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto p-2">
        {/* Current vs recommended. */}
        <div className="grid grid-cols-3 gap-1">
          <div className="rounded border border-scada-700 bg-scada-850 px-2 py-1">
            <span className="block text-2xs uppercase tracking-wide text-slate-400">Current SPM</span>
            <span className="font-mono text-base font-semibold tabular-nums text-slate-100">
              {currentSpm !== null ? fmt(currentSpm, 1) : DASH}
            </span>
          </div>
          <div className="rounded border border-signal-blue/40 bg-signal-blue/10 px-2 py-1">
            <span className="block text-2xs uppercase tracking-wide text-signal-blue">AI Target SPM</span>
            <span className="font-mono text-base font-semibold tabular-nums text-signal-blue">
              {recommendedSpm !== null ? fmt(recommendedSpm, 1) : DASH}
            </span>
          </div>
          <div className="rounded border border-scada-700 bg-scada-850 px-2 py-1">
            <span className="block text-2xs uppercase tracking-wide text-slate-400">Δ SPM</span>
            <span
              className={`font-mono text-base font-semibold tabular-nums ${
                spmDelta === null ? "text-slate-500" : spmDelta > 0 ? "text-signal-amber" : "text-signal-cyan"
              }`}
            >
              {spmDelta === null ? DASH : `${spmDelta > 0 ? "+" : ""}${fmt(spmDelta, 1)}`}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <Meter
            label="Rod-float risk"
            ariaLabel="Rod floating risk index, 0 to 100 percent"
            value={risk ?? 0}
            display={risk !== null ? `${fmtPct(risk, 0)}%` : DASH}
            tone={riskTone(risk)}
          />
          <Meter
            label="Pump fillage"
            ariaLabel="Pump fillage, 0 to 100 percent"
            value={fillage ?? 0}
            display={fillage !== null ? `${fmtPct(fillage, 0)}%` : DASH}
            tone={{ ...riskTone(fillage === null ? null : 1 - fillage), text: fillageTone(fillage) }}
          />
        </div>

        {/* Action + reason. */}
        <div className="rounded border border-scada-700 bg-scada-850 px-2 py-1.5">
          <div className="mb-0.5 flex items-center gap-2">
            <span className="text-2xs uppercase tracking-wide text-slate-400">Action</span>
            <span className={`rounded px-1.5 py-0.5 text-2xs font-bold uppercase ${actionTone} bg-scada-700`}>
              {rec?.action ?? "HOLD"}
            </span>
            {rec && Number.isFinite(rec.spm_ratio) ? (
              <span className="ml-auto font-mono text-2xs tabular-nums text-slate-400">
                ratio {fmt(rec.spm_ratio, 2)}× · stroke {fmt(rec.current_stroke_m, 2)} m
              </span>
            ) : null}
          </div>
          <p className="whitespace-pre-wrap break-words text-xs leading-snug text-slate-300">
            {rec?.reason?.trim() ? rec.reason : "Awaiting optimiser recommendation from the telemetry stream…"}
          </p>
        </div>

        {/* Manual setpoint entry. */}
        <form
          className="rounded border border-scada-700 bg-scada-850 px-2 py-1.5"
          // Native constraint validation would abort the submit event for an
          // out-of-range value, so `onApply` never ran and the rejection was
          // reported only as a transient browser bubble that disappears before
          // it can be read on a plant-floor screen. Suppressing it keeps the
          // persistent, aria-live in-panel alert as the single source of truth.
          noValidate
          onSubmit={(event) => {
            event.preventDefault();
            void onApply();
          }}
        >
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label htmlFor={spmId} className="block text-2xs uppercase tracking-wide text-slate-400">
                Strokes / min
              </label>
              <input
                id={spmId}
                type="number"
                inputMode="decimal"
                step={0.1}
                min={SPM_MIN}
                max={SPM_MAX}
                value={spmInput}
                aria-invalid={!spmValid}
                onChange={(event) => {
                  setSpmInput(event.target.value);
                  setDirty(true);
                  setApplyState({ kind: "idle" });
                }}
                className={`mt-0.5 w-full rounded border bg-scada-900 px-1.5 py-1 font-mono text-sm tabular-nums text-slate-100 outline-none focus:ring-1 ${
                  spmValid
                    ? "border-scada-600 focus:border-signal-blue focus:ring-signal-blue"
                    : "border-signal-red focus:border-signal-red focus:ring-signal-red"
                }`}
              />
            </div>
            <div>
              <label htmlFor={strokeId} className="block text-2xs uppercase tracking-wide text-slate-400">
                Stroke length (m)
              </label>
              <input
                id={strokeId}
                type="number"
                inputMode="decimal"
                step={0.1}
                min={STROKE_MIN}
                max={STROKE_MAX}
                value={strokeInput}
                aria-invalid={!strokeValid}
                onChange={(event) => {
                  setStrokeInput(event.target.value);
                  setDirty(true);
                  setApplyState({ kind: "idle" });
                }}
                className={`mt-0.5 w-full rounded border bg-scada-900 px-1.5 py-1 font-mono text-sm tabular-nums text-slate-100 outline-none focus:ring-1 ${
                  strokeValid
                    ? "border-scada-600 focus:border-signal-blue focus:ring-signal-blue"
                    : "border-signal-red focus:border-signal-red focus:ring-signal-red"
                }`}
              />
            </div>
          </div>

          <label className="mt-2 flex cursor-pointer items-center gap-2 text-xs text-slate-300">
            <input
              type="checkbox"
              checked={autonomous}
              onChange={(event) => {
                setAutonomous(event.target.checked);
                setApplyState({ kind: "idle" });
              }}
              className="h-3.5 w-3.5 cursor-pointer accent-amber-500"
              aria-label="Autonomous VFD closed-loop control"
            />
            <span>Autonomous VFD closed-loop</span>
            <span className="ml-auto text-2xs text-slate-500">
              {autonomous ? "AI drives the setpoint" : "Operator holds the setpoint"}
            </span>
          </label>

          <div className="mt-2 flex items-center gap-2">
            <button
              type="submit"
              disabled={applyState.kind === "pending"}
              aria-label="Apply setpoint to VFD"
              className="flex-1 rounded bg-amber-scada px-3 py-1.5 text-xs font-bold uppercase tracking-wide text-scada-950 transition-colors hover:bg-amber-400 focus:outline-none focus:ring-2 focus:ring-amber-300 focus:ring-offset-1 focus:ring-offset-scada-900 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {applyState.kind === "pending" ? "Applying…" : "Apply Setpoint to VFD"}
            </button>
            {dirty ? (
              <button
                type="button"
                aria-label="Reset inputs to the AI recommendation"
                onClick={() => {
                  setDirty(false);
                  setSpmInput(recommendedSpm === null ? "" : String(recommendedSpm));
                  setStrokeInput(recommendedStroke === null ? "" : String(recommendedStroke));
                  setApplyState({ kind: "idle" });
                }}
                className="rounded border border-scada-600 px-2 py-1.5 text-2xs uppercase text-slate-300 hover:bg-scada-700 focus:outline-none focus:ring-1 focus:ring-slate-400"
              >
                Reset
              </button>
            ) : null}
          </div>
        </form>

        {/* Apply result. */}
        <div aria-live="polite">
          {applyState.kind === "success" ? (
            <div className="rounded border border-signal-green bg-green-950/40 px-2 py-1.5">
              <p className="text-2xs font-semibold uppercase tracking-wide text-signal-green">Setpoint applied</p>
              <p className="break-words text-xs text-green-200">{applyState.message}</p>
              <p className="mt-0.5 font-mono text-2xs tabular-nums text-green-300/80">
                {fmt(applyState.spm, 1)} spm · {fmt(applyState.stroke, 2)} m
              </p>
            </div>
          ) : applyState.kind === "error" ? (
            <div role="alert" className="rounded border border-signal-red bg-red-950/40 px-2 py-1.5">
              <p className="text-2xs font-semibold uppercase tracking-wide text-signal-red">Setpoint rejected</p>
              <p className="break-words text-xs text-red-200">{applyState.message}</p>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}
