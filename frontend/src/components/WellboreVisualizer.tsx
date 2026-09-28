import { useMemo } from "react";
import { DASH, clamp, fmt, num } from "../api";
import type { TelemetryFrame } from "../types";

const VB_W = 460;
const VB_H = 620;
const M = { top: 20, right: 74, bottom: 46, left: 52 } as const;
const TRACK_W = VB_W - M.left - M.right;
const TRACK_H = VB_H - M.top - M.bottom;

/** Full wellbore depth assumed by the grid, per the project's well definition. */
const DEPTH_MAX_M = 1050;

/** Twin-profile overlap: temp and pressure each own a 62% share of the track. */
const TEMP_SPAN = 0.62;
const PRESS_SPAN = 0.62;

interface Series {
  values: Array<{ depth: number; value: number }>;
  min: number;
  max: number;
}

function buildSeries(depths: unknown, values: unknown): Series {
  const out: Array<{ depth: number; value: number }> = [];
  if (!Array.isArray(depths) || !Array.isArray(values)) return { values: out, min: 0, max: 1 };
  const n = Math.min(depths.length, values.length);
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < n; i += 1) {
    const d = depths[i];
    const v = values[i];
    if (typeof d !== "number" || !Number.isFinite(d) || typeof v !== "number" || !Number.isFinite(v)) {
      continue;
    }
    out.push({ depth: d, value: v });
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (out.length === 0) return { values: out, min: 0, max: 1 };
  if (min === max) {
    // Flat profile still needs a drawable band.
    const pad = Math.max(Math.abs(min) * 0.05, 1);
    return { values: out, min: min - pad, max: max + pad };
  }
  return { values: out, min, max };
}

/** Colour-scale stops, emitted as SVG linear gradients so each profile carries
 *  its own scale: cool-to-hot for temperature, deep-to-light for pressure. */
/** Cool-to-hot ramp for temperature. */
const TEMP_STOPS: Array<[number, number, number]> = [
  [59, 130, 246],
  [34, 197, 94],
  [234, 179, 8],
  [249, 115, 22],
  [239, 68, 68],
];

/** Deep-to-light ramp for pressure. */
const PRESS_STOPS: Array<[number, number, number]> = [
  [30, 58, 138],
  [59, 130, 246],
  [34, 211, 238],
  [167, 139, 250],
];

const REGIME_COLORS: Record<string, string> = {
  SLIP: "#22c55e",
  BUBBLE: "#eab308",
  MIST: "#f97316",
  ANNULAR: "#3b82f6",
  TRANSITION: "#94a3b8",
  SINGLE: "#64748b",
};

function depthToY(depth: number): number {
  return M.top + (clamp(depth, 0, DEPTH_MAX_M) / DEPTH_MAX_M) * TRACK_H;
}

interface Props {
  frame: TelemetryFrame | null;
  /** Pump setting depth; falls back to a mid-wellbore position when unknown. */
  pumpDepthM?: number | null;
}

export default function WellboreVisualizer({ frame, pumpDepthM }: Props) {
  const wellbore = frame?.wellbore;

  const model = useMemo(() => {
    const temp = buildSeries(wellbore?.depth_m, wellbore?.temperature_c);
    const press = buildSeries(wellbore?.depth_m, wellbore?.pressure_mpa);
    const density = buildSeries(wellbore?.depth_m, wellbore?.mixture_density_kg_m3);

    // Holdup is sampled on the same depth grid as the profiles.
    const holdup = (() => {
      const out: Array<{ depth: number; value: number }> = [];
      const depths = wellbore?.depth_m;
      const values = wellbore?.liquid_holdup;
      if (!Array.isArray(depths) || !Array.isArray(values)) return out;
      const n = Math.min(depths.length, values.length);
      for (let i = 0; i < n; i += 1) {
        const d = depths[i];
        const v = values[i];
        if (typeof d !== "number" || !Number.isFinite(d) || typeof v !== "number" || !Number.isFinite(v)) {
          continue;
        }
        out.push({ depth: d, value: clamp(v, 0, 1) });
      }
      return out;
    })();

    const regimes: Array<{ depth: number; label: string }> = (() => {
      const out: Array<{ depth: number; label: string }> = [];
      const depths = wellbore?.depth_m;
      const labels = wellbore?.regime;
      if (!Array.isArray(depths) || !Array.isArray(labels)) return out;
      const n = Math.min(depths.length, labels.length);
      for (let i = 0; i < n; i += 1) {
        const d = depths[i];
        const l = labels[i];
        if (typeof d === "number" && Number.isFinite(d) && typeof l === "string") {
          out.push({ depth: d, label: l });
        }
      }
      return out;
    })();

    return { temp, press, density, holdup, regimes };
  }, [wellbore]);

  const { temp, press, holdup, regimes } = model;

  const hasProfiles = temp.values.length > 1 || press.values.length > 1;

  const tempX = (value: number) => M.left + ((value - temp.min) / (temp.max - temp.min)) * TRACK_W * TEMP_SPAN;
  const pressX = (value: number) =>
    M.left + TRACK_W * (1 - PRESS_SPAN) + ((value - press.min) / (press.max - press.min)) * TRACK_W * PRESS_SPAN;

  const tempPath = temp.values
    .map((p, i) => `${i === 0 ? "M" : "L"}${tempX(p.value).toFixed(2)},${depthToY(p.depth).toFixed(2)}`)
    .join(" ");
  const pressPath = press.values
    .map((p, i) => `${i === 0 ? "M" : "L"}${pressX(p.value).toFixed(2)},${depthToY(p.depth).toFixed(2)}`)
    .join(" ");

  const depthTicks = [0, 200, 400, 600, 800, 1000];
  const wellDepth = Math.max(
    ...model.temp.values.map((p) => p.depth),
    ...model.press.values.map((p) => p.depth),
    0,
  );
  const pumpDepth = num(pumpDepthM, 0);
  const hasPump = pumpDepth > 0 && pumpDepth <= DEPTH_MAX_M;

  // Regime colour strip on the leading edge of the track.
  const regimeStops = regimes
    .map((r) => ({
      ...r,
      y: depthToY(r.depth),
      color: REGIME_COLORS[r.label.toUpperCase()] ?? "#475569",
    }))
    .sort((a, b) => a.y - b.y);
  const regimeBands = regimeStops.slice(0, -1).map((r, i) => {
    const next = regimeStops[i + 1];
    if (!next) return null;
    return { key: `${r.depth}-${i}`, y: r.y, height: Math.max(1, next.y - r.y), color: r.color };
  });

  return (
    <section
      aria-label="Wellbore temperature, pressure and liquid holdup profiles"
      className="flex h-full min-h-0 flex-col rounded-md border border-scada-700 bg-scada-900"
    >
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-scada-700 px-3 py-1.5">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-300">Wellbore Profiles</h2>
        <div className="flex items-center gap-3 text-2xs font-mono text-slate-400">
          <span className="text-orange-400">T {fmt(temp.max, 0)} °C</span>
          <span className="text-cyan-300">P {fmt(press.max, 1)} MPa</span>
        </div>
      </header>

      <div className="min-h-0 flex-1 p-1">
        <svg
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label="Vertical wellbore plot from surface to 1050 metres showing temperature and pressure profiles."
          className="h-full w-full"
        >
          <defs>
            {/* userSpaceOnUse so each ramp spans exactly its profile's track
                segment, making the two colour scales directly comparable. */}
            <linearGradient
              id="wellbore-temp-scale"
              gradientUnits="userSpaceOnUse"
              x1={M.left}
              y1={0}
              x2={M.left + TRACK_W * TEMP_SPAN}
              y2={0}
            >
              {TEMP_STOPS.map((s, i) => (
                <stop
                  key={`t${i}`}
                  offset={`${(i / (TEMP_STOPS.length - 1)) * 100}%`}
                  stopColor={`rgb(${s[0]},${s[1]},${s[2]})`}
                />
              ))}
            </linearGradient>
            <linearGradient
              id="wellbore-press-scale"
              gradientUnits="userSpaceOnUse"
              x1={M.left + TRACK_W * (1 - PRESS_SPAN)}
              y1={0}
              x2={M.left + TRACK_W}
              y2={0}
            >
              {PRESS_STOPS.map((s, i) => (
                <stop
                  key={`p${i}`}
                  offset={`${(i / (PRESS_STOPS.length - 1)) * 100}%`}
                  stopColor={`rgb(${s[0]},${s[1]},${s[2]})`}
                />
              ))}
            </linearGradient>
          </defs>

          {/* Track background and depth gridlines. */}
          <rect x={M.left} y={M.top} width={TRACK_W} height={TRACK_H} fill="#0e141d" stroke="#1c2634" />
          {depthTicks.map((d) => (
            <g key={d}>
              <line
                x1={M.left}
                x2={M.left + TRACK_W}
                y1={depthToY(d)}
                y2={depthToY(d)}
                stroke="#1c2634"
                strokeWidth={1}
              />
              <text
                x={M.left - 8}
                y={depthToY(d) + 3}
                textAnchor="end"
                fontSize={10}
                fill="#94a3b8"
                fontFamily="ui-monospace, monospace"
              >
                {d}
              </text>
            </g>
          ))}
          {/* Completion depth, always labelled so the axis terminates clearly. */}
          <text
            x={M.left - 8}
            y={M.top + TRACK_H + 3}
            textAnchor="end"
            fontSize={10}
            fill="#94a3b8"
            fontFamily="ui-monospace, monospace"
          >
            {fmt(wellDepth, 0)}
          </text>

          {/* Liquid-holdup band: a closed ribbon whose width encodes the
              liquid fraction at each depth. The polygon runs down the liquid
              interface, then back up the wellbore wall as a vertical edge. */}
          {holdup.length > 1 ? (
            <path
              d={
                holdup
                  .map((p, i) => {
                    const y = depthToY(p.depth).toFixed(2);
                    const x = (M.left + p.value * TRACK_W).toFixed(2);
                    return `${i === 0 ? "M" : "L"}${x},${y}`;
                  })
                  .join(" ") +
                ` L${M.left},${depthToY(holdup[holdup.length - 1]?.depth ?? 0).toFixed(2)}` +
                ` L${M.left},${depthToY(holdup[0]?.depth ?? 0).toFixed(2)} Z`
              }
              fill="#38bdf8"
              fillOpacity={0.14}
              stroke="#38bdf8"
              strokeOpacity={0.5}
              strokeWidth={1}
            />
          ) : null}

          {/* Flow-regime strip along the track edge. */}
          <g>
            {regimeBands.map((b) =>
              b ? (
                <rect
                  key={b.key}
                  x={M.left + 1}
                  y={b.y}
                  width={3}
                  height={b.height}
                  fill={b.color}
                />
              ) : null,
            )}
          </g>

          {/* Temperature profile, drawn on its own left-weighted colour scale. */}
          {tempPath ? (
            <path
              d={tempPath}
              fill="none"
              stroke="url(#wellbore-temp-scale)"
              strokeWidth={2.4}
              strokeLinejoin="round"
            />
          ) : null}

          {/* Pressure profile, drawn on its own right-weighted colour scale. */}
          {pressPath ? (
            <path
              d={pressPath}
              fill="none"
              stroke="url(#wellbore-press-scale)"
              strokeWidth={1.8}
              strokeDasharray="6 3"
              strokeLinejoin="round"
            />
          ) : null}

          {/* Empty / insufficient-sample state. */}
          {!hasProfiles ? (
            <text
              x={M.left + TRACK_W / 2}
              y={M.top + TRACK_H / 2}
              textAnchor="middle"
              fontSize={11}
              fill="#64748b"
            >
              Awaiting wellbore profile samples…
            </text>
          ) : null}

          {/* Pump depth marker. */}
          {hasPump ? (
            <g>
              <line
                x1={M.left}
                x2={M.left + TRACK_W}
                y1={depthToY(pumpDepth)}
                y2={depthToY(pumpDepth)}
                stroke="#22c55e"
                strokeWidth={1.4}
                strokeDasharray="5 3"
              />
              <rect
                x={M.left}
                y={depthToY(pumpDepth) - 8}
                width={62}
                height={14}
                rx={2}
                fill="#052e16"
                stroke="#22c55e"
                strokeWidth={0.8}
              />
              <text
                x={M.left + 4}
                y={depthToY(pumpDepth) + 2}
                fontSize={9}
                fill="#4ade80"
                fontFamily="ui-monospace, monospace"
              >
                PUMP {fmt(pumpDepth, 0)} m
              </text>
            </g>
          ) : null}

          {/* Right-hand twin axes. */}
          <g>
            {/* Temperature axis (inner). */}
            <line
              x1={M.left + TRACK_W * TEMP_SPAN}
              x2={M.left + TRACK_W * TEMP_SPAN}
              y1={M.top}
              y2={M.top + TRACK_H}
              stroke="#fb923c"
              strokeOpacity={0.4}
              strokeWidth={1}
            />
            <text
              x={M.left + TRACK_W * TEMP_SPAN + 4}
              y={M.top - 8}
              fontSize={9}
              fill="#fb923c"
              fontFamily="ui-monospace, monospace"
            >
              °C
            </text>
            <text
              x={M.left + TRACK_W * TEMP_SPAN + 4}
              y={M.top + 8}
              fontSize={9}
              fill="#fb923c"
              fontFamily="ui-monospace, monospace"
            >
              {fmt(temp.max, 0)}
            </text>
            <text
              x={M.left + TRACK_W * TEMP_SPAN + 4}
              y={M.top + TRACK_H}
              fontSize={9}
              fill="#fb923c"
              fontFamily="ui-monospace, monospace"
            >
              {fmt(temp.min, 0)}
            </text>

            {/* Pressure axis (outer). */}
            <line
              x1={M.left + TRACK_W}
              x2={M.left + TRACK_W}
              y1={M.top}
              y2={M.top + TRACK_H}
              stroke="#22d3ee"
              strokeOpacity={0.4}
              strokeWidth={1}
            />
            <text
              x={M.left + TRACK_W + 6}
              y={M.top - 8}
              fontSize={9}
              fill="#22d3ee"
              fontFamily="ui-monospace, monospace"
            >
              MPa
            </text>
            <text
              x={M.left + TRACK_W + 6}
              y={M.top + 8}
              fontSize={9}
              fill="#22d3ee"
              fontFamily="ui-monospace, monospace"
            >
              {fmt(press.max, 1)}
            </text>
            <text
              x={M.left + TRACK_W + 6}
              y={M.top + TRACK_H}
              fontSize={9}
              fill="#22d3ee"
              fontFamily="ui-monospace, monospace"
            >
              {fmt(press.min, 1)}
            </text>
          </g>

          {/* Axis title. */}
          <text
            transform={`translate(14, ${M.top + TRACK_H / 2}) rotate(-90)`}
            textAnchor="middle"
            fontSize={10}
            fill="#cbd5e1"
          >
            Depth (m)
          </text>
          <text
            x={M.left + TRACK_W / 2}
            y={VB_H - 8}
            textAnchor="middle"
            fontSize={10}
            fill="#cbd5e1"
          >
            0 – {DEPTH_MAX_M} m completion interval · holdup shading 0–100%
          </text>
        </svg>
      </div>

      <footer className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-scada-700 px-3 py-1 text-2xs text-slate-400">
        <LegendSwatch color="#fb923c" label="Temperature" />
        <LegendSwatch color="#22d3ee" label="Pressure" dashed />
        <LegendSwatch color="#38bdf8" label="Liquid holdup" />
        <span className="ml-auto font-mono text-slate-500">
          ρ̄ {model.density.values.length > 0 ? fmt(model.density.values[model.density.values.length - 1]?.value, 0) : DASH} kg/m³
        </span>
      </footer>
    </section>
  );
}

function LegendSwatch({ color, label, dashed = false }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <svg width="18" height="6" aria-hidden="true">
        <line
          x1={0}
          x2={18}
          y1={3}
          y2={3}
          stroke={color}
          strokeWidth={2}
          strokeDasharray={dashed ? "5 2" : undefined}
        />
      </svg>
      {label}
    </span>
  );
}
