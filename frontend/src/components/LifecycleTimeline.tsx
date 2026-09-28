import { DASH, clamp, fmt, fmtInt, num } from "../api";
import type { Phase, TelemetryFrame } from "../types";

const PHASES: Array<{ key: Phase; label: string; accent: string }> = [
  { key: "INJECTION", label: "Injection", accent: "#f97316" },
  { key: "SOAKING", label: "Soaking", accent: "#a78bfa" },
  { key: "PRODUCTION", label: "Production", accent: "#22c55e" },
];

/** Duration of each phase in days, used only when the schedule is unavailable. */
const FALLBACK_DAYS: Record<Phase, number> = {
  INJECTION: 21,
  SOAKING: 7,
  PRODUCTION: 30,
};

/** Steam-oil ratio thresholds for the colour-coded performance readout. */
const SOR_GOOD = 4.2;
const SOR_WARN = 5.0;

export function sorTone(sor: number | null | undefined): { text: string; bg: string; label: string } {
  if (sor === null || sor === undefined || !Number.isFinite(sor)) {
    return { text: "text-slate-500", bg: "bg-scada-700", label: "no data" };
  }
  if (sor < SOR_GOOD) return { text: "text-signal-green", bg: "bg-signal-green", label: "efficient" };
  if (sor <= SOR_WARN) return { text: "text-signal-amber", bg: "bg-signal-amber", label: "watch" };
  return { text: "text-signal-red", bg: "bg-signal-red", label: "inefficient" };
}

interface TileProps {
  label: string;
  value: string;
  unit?: string;
  tone?: string;
  title?: string;
}

function Tile({ label, value, unit, tone = "text-slate-100", title }: TileProps) {
  return (
    <div className="flex min-w-0 flex-col justify-center rounded border border-scada-700 bg-scada-850 px-2 py-1" title={title}>
      <span className="truncate text-2xs uppercase tracking-wide text-slate-400">{label}</span>
      <span className="flex items-baseline gap-1">
        <span className={`font-mono text-sm font-semibold tabular-nums ${tone}`}>{value}</span>
        {unit ? <span className="text-2xs text-slate-500">{unit}</span> : null}
      </span>
    </div>
  );
}

interface Props {
  frame: TelemetryFrame | null;
  /** Phase lengths in days; the stream's own counters are the primary source. */
  phaseDays?: Record<Phase, number> | null;
  /** Number of planned cycles, used to scale the day total across the schedule. */
  cycles?: number | null;
}

export default function LifecycleTimeline({ frame, phaseDays, cycles }: Props) {
  const well = frame?.well;
  const css = frame?.css;
  const phase: Phase = well?.phase ?? "INJECTION";
  const activeIndex = Math.max(
    0,
    PHASES.findIndex((p) => p.key === phase),
  );

  const daysFor = (key: Phase): number => {
    const configured = phaseDays?.[key];
    return typeof configured === "number" && Number.isFinite(configured) && configured > 0
      ? configured
      : FALLBACK_DAYS[key];
  };

  // The phase lengths describe one cycle, so the schedule total spans all of them.
  const cycleCount = cycles !== null && cycles !== undefined && Number.isFinite(cycles) && cycles > 0 ? cycles : 1;
  const cycleDays = PHASES.reduce((sum, p) => sum + daysFor(p.key), 0);
  const totalDays = cycleDays * cycleCount;
  const daysIntoPhase = num(well?.days_into_phase, 0);
  const phaseLength = Math.max(daysFor(phase), 1e-6);
  const progress = clamp(daysIntoPhase / phaseLength, 0, 1);
  const day = num(well?.day, 0);
  const overallProgress = totalDays > 0 ? clamp(day / totalDays, 0, 1) : 0;

  const sor = css?.cumulative_sor ?? null;
  const tone = sorTone(sor);
  const cutoff = css?.cutoff_reached === true;
  const cutoffReason = css?.cutoff_reason ?? "";

  return (
    <section
      aria-label="CSS lifecycle phase and performance summary"
      className="flex h-full min-h-0 flex-col rounded-md border border-scada-700 bg-scada-900"
    >
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-scada-700 px-3 py-1.5">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          CSS Lifecycle
        </h2>
        <div className="flex items-center gap-3 text-2xs font-mono text-slate-400">
          <span>
            CYCLE <span className="tabular-nums text-slate-200">{well ? fmtInt(well.cycle) : DASH}</span>
          </span>
          <span>
            DAY <span className="tabular-nums text-slate-200">{well ? fmtInt(well.day) : DASH}</span>
            <span className="text-slate-500">/{fmtInt(totalDays)}</span>
          </span>
        </div>
      </header>

      <div className="flex min-h-0 flex-1 flex-col gap-2 p-2">
        {/* Phase stepper. */}
        <ol className="flex items-stretch gap-1" aria-label="Cyclic steam stimulation phases">
          {PHASES.map((p, i) => {
            const isActive = i === activeIndex;
            const isDone = i < activeIndex;
            return (
              <li key={p.key} className="flex min-w-0 flex-1 items-center gap-1">
                <div
                  aria-current={isActive ? "step" : undefined}
                  className={`min-w-0 flex-1 rounded border px-2 py-1 transition-colors ${
                    isActive
                      ? "border-transparent bg-scada-700 text-slate-50"
                      : isDone
                        ? "border-scada-600 bg-scada-850 text-slate-400"
                        : "border-scada-700 bg-scada-900 text-slate-500"
                  }`}
                  style={isActive ? { borderColor: p.accent, boxShadow: `inset 0 0 0 1px ${p.accent}` } : undefined}
                >
                  <div className="flex items-center gap-1.5">
                    <span
                      aria-hidden="true"
                      className="h-2 w-2 shrink-0 rounded-full"
                      style={{ backgroundColor: isActive || isDone ? p.accent : "#334155" }}
                    />
                    <span className="truncate text-2xs font-semibold uppercase tracking-wide">{p.label}</span>
                  </div>
                  <div className="mt-0.5 flex items-baseline justify-between gap-1">
                    <span className="truncate text-2xs text-slate-500">
                      {isActive ? `${fmt(daysIntoPhase, 1)} d in` : `${fmtInt(daysFor(p.key))} d`}
                    </span>
                    {isDone ? <span className="text-2xs text-signal-green">done</span> : null}
                  </div>
                </div>
                {i < PHASES.length - 1 ? (
                  <span aria-hidden="true" className={isDone ? "text-signal-green" : "text-scada-600"}>
                    ›
                  </span>
                ) : null}
              </li>
            );
          })}
        </ol>

        {/* Progress bars for the current phase and the whole cycle. */}
        <div className="space-y-1">
          <div>
            <div className="mb-0.5 flex items-center justify-between text-2xs text-slate-400">
              <span>Phase progress · {PHASES[activeIndex]?.label ?? phase}</span>
              <span className="font-mono tabular-nums">
                {fmt(daysIntoPhase, 1)} / {fmtInt(phaseLength)} d
              </span>
            </div>
            <div
              role="progressbar"
              aria-label="Current phase progress"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(progress * 100)}
              className="h-1.5 w-full overflow-hidden rounded-full bg-scada-700"
            >
              <div
                className="h-full rounded-full transition-[width] duration-300"
                style={{
                  width: `${Math.max(progress * 100, well ? 2 : 0)}%`,
                  backgroundColor: PHASES[activeIndex]?.accent ?? "#f5a524",
                }}
              />
            </div>
          </div>
          <div>
            <div className="mb-0.5 flex items-center justify-between text-2xs text-slate-400">
              <span>Cycle progress</span>
              <span className="font-mono tabular-nums">{fmt(overallProgress * 100, 0)}%</span>
            </div>
            <div className="h-1 w-full overflow-hidden rounded-full bg-scada-700">
              <div
                className="h-full rounded-full bg-slate-500 transition-[width] duration-300"
                style={{ width: `${overallProgress * 100}%` }}
              />
            </div>
          </div>
        </div>

        {/* KPI tiles.

            The tile count per row has to suit the panel, not the viewport: this
            panel is one of three columns, so `xl` (1280px of *window*) still
            leaves it under 500px. Five tiles across that width truncated every
            label, so the row only widens at 2xl. */}
        <div className="grid grid-cols-2 gap-1 sm:grid-cols-3 2xl:grid-cols-5">
          <div className="relative overflow-hidden rounded border border-scada-700 bg-scada-850 px-2 py-1">
            <span className="absolute inset-y-0 left-0 w-1" style={{ backgroundColor: tone.bg }} aria-hidden="true" />
            <span className="block pl-1.5 text-2xs uppercase tracking-wide text-slate-400">Cumulative SOR</span>
            <span className="flex items-baseline gap-1 pl-1.5">
              <span className={`font-mono text-sm font-semibold tabular-nums ${tone.text}`}>
                {sor === null || sor === undefined || !Number.isFinite(sor) ? DASH : fmt(sor, 2)}
              </span>
              <span className="text-2xs text-slate-500">bbl/bbl</span>
            </span>
            <span className={`block pl-1.5 text-2xs ${tone.text}`}>{tone.label}</span>
          </div>

          <Tile
            label="Steam Injected"
            value={css ? fmtInt(css.cumulative_steam_m3) : DASH}
            unit="m³"
            title={`${fmt(css?.steam_tonnes, 0)} t steam`}
          />
          <Tile
            label="Oil Produced"
            value={css ? fmtInt(css.oil_bbl) : DASH}
            unit="bbl"
            tone="text-signal-green"
            title={`${fmt(css?.cumulative_oil_m3, 0)} m³`}
          />
          <Tile
            label="Steam Chest"
            value={css ? fmt(css.chest_radius_m, 1) : DASH}
            unit="m"
            tone="text-amber-scada"
          />
          <Tile
            label="Instant. SOR"
            value={css && Number.isFinite(css.instantaneous_sor) ? fmt(css.instantaneous_sor, 2) : DASH}
            unit="bbl/bbl"
            tone={sorTone(css?.instantaneous_sor).text}
          />
        </div>

        {/* Cutoff banner. */}
        {cutoff ? (
          <div
            role="alert"
            className="flex items-start gap-2 rounded border border-signal-red bg-red-950/40 px-2 py-1.5"
          >
            <span aria-hidden="true" className="mt-0.5 text-signal-red">
              ⚠
            </span>
            <div className="min-w-0">
              <p className="text-2xs font-semibold uppercase tracking-wide text-signal-red">
                Economic cutoff reached
              </p>
              <p className="break-words text-xs text-red-200">
                {cutoffReason.trim() !== "" ? cutoffReason : "The optimiser halted the cycle; see schedule details."}
              </p>
            </div>
          </div>
        ) : null}
      </div>
    </section>
  );
}
