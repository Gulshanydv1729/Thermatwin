import { useMemo } from "react";
import { DASH, clamp, fmt, fmtKn, num } from "../api";
import type { CardLabel, CardTrace, TelemetryFrame } from "../types";

/** Plot geometry in viewBox units; the SVG scales to its container. */
const VB_W = 720;
const VB_H = 380;
const M = { top: 18, right: 18, bottom: 44, left: 60 } as const;
const PLOT_W = VB_W - M.left - M.right;
const PLOT_H = VB_H - M.top - M.bottom;

/** A trace reduced to finite points, safe to plot even from a partial frame. */
interface CleanTrace {
  points: Array<{ position: number; load: number }>;
}

function cleanTrace(trace: CardTrace | undefined): CleanTrace {
  if (!trace || !Array.isArray(trace.position_m) || !Array.isArray(trace.load_n)) {
    return { points: [] };
  }
  const points: Array<{ position: number; load: number }> = [];
  const n = Math.min(trace.position_m.length, trace.load_n.length);
  for (let i = 0; i < n; i += 1) {
    const position = trace.position_m[i];
    const load = trace.load_n[i];
    if (typeof position === "number" && Number.isFinite(position) && typeof load === "number" && Number.isFinite(load)) {
      points.push({ position, load });
    }
  }
  return { points };
}

function extent(values: number[]): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1 };
  if (min === max) return { min: min - 0.5, max: max + 0.5 };
  return { min, max };
}

interface Scale {
  x: (position: number) => number;
  y: (load: number) => number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

function makeScale(positions: number[], loads: number[]): Scale {
  const px = extent(positions);
  const pl = extent(loads);
  // A dynamometer card is read against a load axis that starts at zero, so the
  // window is anchored there; the ceiling follows the data (extent() already
  // widens a flat or single-sample series so the span can never be zero).
  const yMin = Math.min(pl.min, 0);
  const yMax = pl.max;
  const xSpan = px.max - px.min;
  const ySpan = yMax - yMin;
  return {
    x: (p) => M.left + ((p - px.min) / xSpan) * PLOT_W,
    y: (l) => M.top + (1 - (l - yMin) / ySpan) * PLOT_H,
    xMin: px.min,
    xMax: px.max,
    yMin,
    yMax,
  };
}

function pathFrom(points: Array<{ position: number; load: number }>, scale: Scale): string {
  if (points.length === 0) return "";
  return points
    .map((p, i) => `${i === 0 ? "M" : "L"}${scale.x(p.position).toFixed(2)},${scale.y(p.load).toFixed(2)}`)
    .join(" ");
}

/** Chooses ~6 evenly spaced, human-friendly tick values across a range. */
function ticks(min: number, max: number, target = 6): number[] {
  const span = max - min;
  if (!Number.isFinite(span) || span <= 0) return [min];
  const rawStep = span / Math.max(1, target);
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalised = rawStep / magnitude;
  const step = (normalised >= 5 ? 5 : normalised >= 2 ? 2 : 1) * magnitude;
  const start = Math.ceil(min / step) * step;
  const out: number[] = [];
  for (let v = start; v <= max + step * 1e-6; v += step) {
    out.push(Number(v.toFixed(10)));
  }
  return out;
}

/** Short human label for each diagnosed card fault. */
const FAULT_TEXT: Record<CardLabel, string> = {
  NORMAL_FULL_BARREL: "Full barrel — no fault signature",
  FLUID_POUND: "Fluid pound: load collapse on the upstroke",
  ROD_FLOATING: "Rod float: downstroke load depression",
  GAS_INTERFERENCE: "Gas interference: rounded, non-parallelogram card",
  PUMP_TAGGING: "Pump tagging: impact at end of downstroke",
  UNANCHORED_TUBING: "Unanchored tubing: load peak lags top dead centre",
};

interface Props {
  frame: TelemetryFrame | null;
}

export default function DynoCardCanvas({ frame }: Props) {
  const surface = useMemo(() => cleanTrace(frame?.card.surface), [frame]);
  const downhole = useMemo(() => cleanTrace(frame?.card.downhole), [frame]);

  const view = useMemo(() => {
    const positions = [...surface.points.map((p) => p.position), ...downhole.points.map((p) => p.position)];
    const loads = [...surface.points.map((p) => p.load), ...downhole.points.map((p) => p.load)];
    const stroke = num(frame?.card.stroke_m, 0);
    const scale = makeScale(positions, loads);
    const label = frame?.diagnosis.label ?? "NORMAL_FULL_BARREL";
    const features = frame?.diagnosis.features ?? {};
    return { positions, loads, scale, stroke, label, features };
  }, [surface.points, downhole.points, frame]);

  const { scale, stroke, label, features } = view;
  const hasData = surface.points.length > 0 || downhole.points.length > 0;

  const xTicks = ticks(scale.xMin, scale.xMax, 6);
  const yTicks = ticks(scale.yMin, scale.yMax, 5);

  /**
   * Fault region overlays.
   *
   * The region is located from the card samples themselves -- the part of the
   * loop where the fault signature actually shows up -- and the geometric
   * features supply the colour, label and severity. This keeps the box on the
   * data for any stroke length or load range instead of a fixed plot fraction.
   */
  const faultRegion = useMemo(() => {
    if (!hasData || label === "NORMAL_FULL_BARREL") return null;

    // The reconstructed downhole card is preferred; the surface trace is the
    // fallback when the reconstruction is too short to localise anything.
    const trace = downhole.points.length >= 8 ? downhole.points : surface.points;
    if (trace.length < 4) return null;

    let cardMin = Infinity;
    let cardMax = -Infinity;
    for (const p of trace) {
      if (p.load < cardMin) cardMin = p.load;
      if (p.load > cardMax) cardMax = p.load;
    }
    const cardSpan = Math.max(cardMax - cardMin, 1);

    // Padding is in screen units; the spans above are data units (newtons).
    const padX = PLOT_W * 0.012;
    const padY = PLOT_H * 0.03;

    /** Screen-space bounding box of a point subset, clipped to the plot. */
    const box = (points: Array<{ position: number; load: number }>) => {
      if (points.length === 0) return null;
      let pMin = Infinity;
      let pMax = -Infinity;
      let lMin = Infinity;
      let lMax = -Infinity;
      for (const p of points) {
        if (p.position < pMin) pMin = p.position;
        if (p.position > pMax) pMax = p.position;
        if (p.load < lMin) lMin = p.load;
        if (p.load > lMax) lMax = p.load;
      }
      if (!Number.isFinite(pMin) || !Number.isFinite(lMin)) return null;
      const x = clamp(scale.x(pMin) - padX, M.left, M.left + PLOT_W);
      const y = clamp(scale.y(lMax) - padY, M.top, M.top + PLOT_H);
      const w = clamp(scale.x(pMax) + padX - x, 0, M.left + PLOT_W - x);
      const h = clamp(scale.y(lMin) + padY - y, 0, M.top + PLOT_H - y);
      if (w <= 0 || h <= 0) return null;
      return { x, y, w, h };
    };

    /** Samples taken while the plunger rises, i.e. the upstroke. */
    const upstroke = trace.filter((p, i) => {
      const prev = trace[i - 1];
      return prev !== undefined && p.position > prev.position;
    });

    const reversal = num(features["upstroke_reversal"], 0);
    const asymmetry = num(features["downstroke_asymmetry"], 0);
    const tailSpike = num(features["tail_spike"], 0);
    const linearity = num(features["linearity"], 0);
    const skew = num(features["skew"], 0);

    switch (label) {
      case "GAS_INTERFERENCE": {
        // A cushioning gas cap rounds the whole card, so the card is the region.
        const rect = box(trace);
        return rect === null ? null : { rect, color: "#a78bfa", caption: `GAS CAP · linearity ${fmt(linearity, 3)}` };
      }
      case "UNANCHORED_TUBING": {
        // The load peak lags top dead centre; the envelope it drags is the
        // upper shoulder of the card, which is what the skew displaces.
        const peak = trace.filter((p) => p.load >= cardMax - 0.3 * cardSpan);
        const rect = box(peak.length > 1 ? peak : trace);
        return rect === null ? null : { rect, color: "#f5a524", caption: `SKEW ENVELOPE · skew ${fmt(skew, 3)}` };
      }
      case "PUMP_TAGGING": {
        // Excess load in the final samples of the downstroke.
        const keep = Math.max(2, Math.round(trace.length * 0.06));
        const rect = box(trace.slice(Math.max(0, trace.length - keep)));
        return rect === null ? null : { rect, color: "#ef4444", caption: `IMPACT TAIL · spike ${fmt(tailSpike, 3)}` };
      }
      case "ROD_FLOATING": {
        // Buoyant rod drag depresses the loop: box the load minimum.
        const low = trace.filter((p) => p.load <= cardMin + 0.25 * cardSpan);
        const rect = box(low.length > 1 ? low : trace);
        return rect === null ? null : { rect, color: "#22d3ee", caption: `LOAD MINIMUM · asym ${fmt(asymmetry, 3)}` };
      }
      case "FLUID_POUND": {
        // Box the part of the upstroke where the load is anomalously low: the
        // physically impossible load fall that defines a pound.
        const base = upstroke.length >= 4 ? upstroke : trace;
        let baseMin = Infinity;
        for (const p of base) if (p.load < baseMin) baseMin = p.load;
        const collapsed = base.filter((p) => p.load <= baseMin + 0.3 * cardSpan);
        const rect = box(collapsed.length >= 2 ? collapsed : base);
        return rect === null ? null : { rect, color: "#ef4444", caption: `UPSTROKE COLLAPSE · reversal ${fmt(reversal, 3)}` };
      }
      default:
        return null;
    }
  }, [hasData, label, features, scale, downhole.points, surface.points]);

  const surfacePath = pathFrom(surface.points, scale);
  const downholePath = pathFrom(downhole.points, scale);

  // Dead centres sit at the extremes of the observed stroke envelope.
  const bdcX = scale.x(scale.xMin);
  const tdcX = scale.x(scale.xMax);
  const strokeLabel = stroke > 0 ? `${fmt(stroke, 2)} m` : DASH;

  return (
    <section
      aria-label="Dyno-card load versus position"
      className="flex h-full min-h-0 flex-col rounded-md border border-scada-700 bg-scada-900"
    >
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-scada-700 px-3 py-1.5">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Dyno-Card · Load vs Position
        </h2>
        <div className="flex items-center gap-3 text-2xs font-mono text-slate-400">
          <span>STROKE {strokeLabel}</span>
          <span>SPAN {frame ? fmtKn(num(frame.card.load_span_n, 0), 1) : DASH} kN</span>
        </div>
      </header>

      <div className="min-h-0 flex-1 p-1">
        {!hasData ? (
          <div
            role="status"
            className="flex h-full min-h-[220px] items-center justify-center text-xs text-slate-500"
          >
            Awaiting dyno-card samples from the telemetry stream…
          </div>
        ) : (
          <svg
            viewBox={`0 0 ${VB_W} ${VB_H}`}
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label={`Dyno-card plot. Diagnosis ${label.replace(/_/g, " ")}.`}
            className="h-full w-full"
          >
            <rect
              x={M.left}
              y={M.top}
              width={PLOT_W}
              height={PLOT_H}
              fill="#0e141d"
              stroke="#1c2634"
              strokeWidth={1}
            />

            {/* Horizontal gridlines with left-hand load axis labels (kN). */}
            {yTicks.map((t) => (
              <g key={`y${t}`}>
                <line
                  x1={M.left}
                  x2={M.left + PLOT_W}
                  y1={scale.y(t)}
                  y2={scale.y(t)}
                  stroke="#1c2634"
                  strokeWidth={1}
                />
                <text
                  x={M.left - 8}
                  y={scale.y(t) + 3}
                  textAnchor="end"
                  fontSize={10}
                  fill="#94a3b8"
                  fontFamily="ui-monospace, monospace"
                >
                  {fmt(t / 1000, 0)}
                </text>
              </g>
            ))}

            {/* Vertical gridlines with bottom position axis labels (m). */}
            {xTicks.map((t) => (
              <g key={`x${t}`}>
                <line
                  x1={scale.x(t)}
                  x2={scale.x(t)}
                  y1={M.top}
                  y2={M.top + PLOT_H}
                  stroke="#1c2634"
                  strokeWidth={1}
                />
                <text
                  x={scale.x(t)}
                  y={M.top + PLOT_H + 14}
                  textAnchor="middle"
                  fontSize={10}
                  fill="#94a3b8"
                  fontFamily="ui-monospace, monospace"
                >
                  {fmt(t, 1)}
                </text>
              </g>
            ))}

            {/* Dead-centre guides: the stroke envelope ends. */}
            <line
              x1={bdcX}
              x2={bdcX}
              y1={M.top}
              y2={M.top + PLOT_H}
              stroke="#475569"
              strokeWidth={1}
              strokeDasharray="2 4"
            />
            <line
              x1={tdcX}
              x2={tdcX}
              y1={M.top}
              y2={M.top + PLOT_H}
              stroke="#475569"
              strokeWidth={1}
              strokeDasharray="2 4"
            />
            <text x={bdcX + 3} y={M.top + PLOT_H - 6} fontSize={9} fill="#64748b" fontFamily="ui-monospace, monospace">
              BDC
            </text>
            <text
              x={tdcX - 3}
              y={M.top + PLOT_H - 6}
              textAnchor="end"
              fontSize={9}
              fill="#64748b"
              fontFamily="ui-monospace, monospace"
            >
              TDC
            </text>

            {/* Diagnosed fault region, behind the traces. */}
            {faultRegion ? (
              <g>
                <rect
                  x={faultRegion.rect.x}
                  y={faultRegion.rect.y}
                  width={faultRegion.rect.w}
                  height={faultRegion.rect.h}
                  fill={faultRegion.color}
                  fillOpacity={0.18}
                  stroke={faultRegion.color}
                  strokeWidth={1}
                  strokeDasharray="4 3"
                />
                <text
                  x={
                    // Flip the caption to the right edge when the box sits far
                    // right, so the label is never clipped by the plot border.
                    faultRegion.rect.x + 4 + faultRegion.caption.length * 5.2 >
                    M.left + PLOT_W
                      ? faultRegion.rect.x + faultRegion.rect.w - 4
                      : faultRegion.rect.x + 4
                  }
                  y={faultRegion.rect.y + 12}
                  textAnchor={
                    faultRegion.rect.x + 4 + faultRegion.caption.length * 5.2 > M.left + PLOT_W
                      ? "end"
                      : "start"
                  }
                  fontSize={9}
                  fill={faultRegion.color}
                  fontFamily="ui-monospace, monospace"
                >
                  {faultRegion.caption}
                </text>
              </g>
            ) : null}

            {/* Surface card: dashed amber. */}
            {surfacePath ? (
              <path
                d={surfacePath}
                fill="none"
                stroke="#f5a524"
                strokeWidth={1.6}
                strokeDasharray="5 3"
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            ) : null}

            {/* Downhole card: solid blue, Gibbs reconstruction. */}
            {downholePath ? (
              <path
                d={downholePath}
                fill="none"
                stroke="#3b9dff"
                strokeWidth={1.8}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            ) : null}

            {/* Single-sample data has no path; mark the point so it is visible. */}
            {surface.points.length === 1 && surface.points[0] ? (
              <circle
                cx={scale.x(surface.points[0].position)}
                cy={scale.y(surface.points[0].load)}
                r={3}
                fill="#f5a524"
              />
            ) : null}
            {downhole.points.length === 1 && downhole.points[0] ? (
              <circle
                cx={scale.x(downhole.points[0].position)}
                cy={scale.y(downhole.points[0].load)}
                r={3}
                fill="#3b9dff"
              />
            ) : null}

            {/* Axis titles. */}
            <text
              x={M.left + PLOT_W / 2}
              y={VB_H - 6}
              textAnchor="middle"
              fontSize={10}
              fill="#cbd5e1"
            >
              Plunger Position (m)
            </text>
            <text
              transform={`translate(14, ${M.top + PLOT_H / 2}) rotate(-90)`}
              textAnchor="middle"
              fontSize={10}
              fill="#cbd5e1"
            >
              Load (kN)
            </text>
          </svg>
        )}
      </div>

      {/* Legend lives outside the plot so it can never occlude the card, the
          fault annotation or the dead-centre labels at any data range. */}
      <footer className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-scada-700 px-3 py-1">
        <span className="inline-flex items-center gap-1.5 text-2xs text-slate-400">
          <svg width="18" height="6" aria-hidden="true">
            <line x1={0} x2={18} y1={3} y2={3} stroke="#f5a524" strokeWidth={1.6} strokeDasharray="5 3" />
          </svg>
          Surface card
        </span>
        <span className="inline-flex items-center gap-1.5 text-2xs text-slate-400">
          <svg width="18" height="6" aria-hidden="true">
            <line x1={0} x2={18} y1={3} y2={3} stroke="#3b9dff" strokeWidth={1.8} />
          </svg>
          Downhole (Gibbs)
        </span>
        <p className="min-w-0 flex-1 truncate text-2xs text-slate-400" title={FAULT_TEXT[label] ?? label}>
          <span className="font-semibold text-slate-200">{label.replace(/_/g, " ")}</span>
          {" · "}
          {FAULT_TEXT[label] ?? "Unrecognised diagnosis label"}
        </p>
      </footer>
    </section>
  );
}
