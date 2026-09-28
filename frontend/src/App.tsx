import { useEffect, useMemo, useState } from "react";
import DynoCardCanvas from "./components/DynoCardCanvas";
import KpiStrip from "./components/KpiStrip";
import LifecycleTimeline from "./components/LifecycleTimeline";
import StatusBanner from "./components/StatusBanner";
import VFDControlPanel from "./components/VFDControlPanel";
import WellboreVisualizer from "./components/WellboreVisualizer";
import {
  DASH,
  getConfig,
  getHealth,
  getSchedule,
  pickNumber,
  pickObject,
  pickString,
} from "./api";
import { useTelemetry } from "./hooks/useTelemetry";
import type { Phase, ScheduleResponse, TelemetryFrame, WellConfig } from "./types";

/** Refreshes the slow-moving REST resources on this cadence. */
const REST_REFRESH_MS = 30000;

export default function App() {
  const { frame, connection, attempt, latencyMs, lastError } = useTelemetry();

  const [config, setConfig] = useState<WellConfig | null>(null);
  const [schedule, setSchedule] = useState<ScheduleResponse | null>(null);
  const [backendHealthy, setBackendHealthy] = useState<boolean | null>(null);

  // Initial load, then a slow poll. Failures are non-fatal: the dashboard stays
  // useful on telemetry alone, with REST-backed fields left blank.
  useEffect(() => {
    let cancelled = false;

    const loadRest = async () => {
      const [healthResult, configResult, scheduleResult] = await Promise.allSettled([
        getHealth(),
        getConfig(),
        getSchedule(),
      ]);
      if (cancelled) return;

      if (healthResult.status === "fulfilled") {
        const status = healthResult.value?.status;
        setBackendHealthy(typeof status === "string" ? status.toLowerCase() === "ok" : true);
      } else {
        setBackendHealthy(false);
      }
      if (configResult.status === "fulfilled") setConfig(configResult.value);
      if (scheduleResult.status === "fulfilled") setSchedule(scheduleResult.value);
    };

    void loadRest();
    const timer = setInterval(() => void loadRest(), REST_REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  // The config document is deep and schema-versioned; only the identity and
  // geometry fields the header and wellbore view need are picked out of it.
  const meta = useMemo(() => {
    const wellNode = pickObject(config, ["well", "well_config", "wellbore"]) ?? config;
    const wellName =
      pickString(config, ["well_name", "name", "well"]) ??
      pickString(wellNode, ["well_name", "name", "id", "well_id"]);
    const pumpDepthM = pickNumber(config, [
      "pump_depth_m",
      "pump_depth",
      "pump_setting_depth_m",
    ]) ?? pickNumber(wellNode, ["pump_depth_m", "pump_depth", "total_depth_m", "depth_m"]);

    // The VFD's permitted SPM range is published by the backend, so the
    // control panel validates against the real envelope rather than a guess.
    const thresholds = pickObject(config, ["thresholds"]);
    const spmMin = pickNumber(thresholds, ["min_spm"]);
    const spmMax = pickNumber(thresholds, ["max_spm"]);
    const spmRange =
      spmMin !== null && spmMax !== null && spmMin < spmMax ? { min: spmMin, max: spmMax } : null;
    return { wellName, pumpDepthM, spmRange };
  }, [config]);

  const phaseDays = useMemo(() => {
    const s = schedule?.schedule;
    if (!s) return null;
    const map: Record<Phase, number> = {
      INJECTION: s.injection_days,
      SOAKING: s.soak_days,
      PRODUCTION: s.production_days,
    };
    const valid = (Object.values(map) as number[]).every((v) => Number.isFinite(v) && v > 0);
    return valid ? map : null;
  }, [schedule]);

  const liveName = frame?.well.name ?? null;
  const wellName = liveName ?? meta.wellName;

  return (
    <div className="flex min-h-screen flex-col bg-scada-950 text-slate-200">
      <StatusBanner
        connection={connection}
        frame={frame}
        latencyMs={latencyMs}
        attempt={attempt}
        lastError={lastError}
        wellName={wellName}
        backendHealthy={backendHealthy}
      />

      <main className="flex min-h-0 flex-1 flex-col gap-2 p-2">
        <KpiStrip frame={frame} latencyMs={latencyMs} loading={!frame} />

        {/* Primary row: dyno-card and wellbore charts.

            The two viewBoxes have opposite aspect ratios (720x380 against
            460x620), so the row is given an explicit height at every
            breakpoint. Without one the SVG's intrinsic ratio drives the panel
            height, which made the tall wellbore chart dictate a column far
            wider than it could fill and left the dyno-card letterboxed inside
            a much taller box than it needed. The two-up split starts at `lg`
            (1024px) so a 1024-wide console still gets both plots side by side
            rather than one chart marooned in a full-width panel. */}
        <div className="grid min-h-0 grid-cols-1 gap-2 lg:h-[440px] lg:grid-cols-[1.35fr_1fr] xl:h-[470px]">
          <DynoCardCanvas frame={frame} />
          <WellboreVisualizer frame={frame} pumpDepthM={meta.pumpDepthM} />
        </div>

        {/* Secondary row: lifecycle, VFD control and schedule economics. */}
        <div className="grid min-h-0 grid-cols-1 gap-2 lg:grid-cols-[1.15fr_1fr_0.85fr]">
          <LifecycleTimeline
            frame={frame}
            phaseDays={phaseDays}
            cycles={schedule?.schedule.cycles ?? null}
          />
          <VFDControlPanel frame={frame} spmRange={meta.spmRange} />
          <ScheduleSummary schedule={schedule} frame={frame} />
        </div>
      </main>
    </div>
  );
}

/** Compact economic read-out from the schedule endpoint. */
function ScheduleSummary({ schedule, frame }: { schedule: ScheduleResponse | null; frame: TelemetryFrame | null }) {
  const rows: Array<{ label: string; value: string; tone?: string }> = [
    {
      label: "Cycles Planned",
      value: schedule && Number.isFinite(schedule.schedule.cycles) ? String(schedule.schedule.cycles) : DASH,
    },
    {
      label: "Cumulative SOR",
      value: schedule && Number.isFinite(schedule.cumulative_sor) ? schedule.cumulative_sor.toFixed(2) : DASH,
    },
    {
      label: "Oil (bbl)",
      value: schedule && Number.isFinite(schedule.oil_bbl) ? Math.round(schedule.oil_bbl).toLocaleString("en-US") : DASH,
      tone: "text-signal-green",
    },
    {
      label: "Steam (m³)",
      value: schedule && Number.isFinite(schedule.steam_m3) ? Math.round(schedule.steam_m3).toLocaleString("en-US") : DASH,
      tone: "text-amber-scada",
    },
    {
      label: "Net Revenue",
      value:
        schedule && Number.isFinite(schedule.net_revenue_usd)
          ? `$${Math.round(schedule.net_revenue_usd).toLocaleString("en-US")}`
          : DASH,
      tone: "text-signal-blue",
    },
    {
      label: "Final BHT",
      value: schedule && Number.isFinite(schedule.final_bht_c) ? `${schedule.final_bht_c.toFixed(1)} °C` : DASH,
    },
    {
      label: "Live Oil (bbl)",
      value: frame && Number.isFinite(frame.css.oil_bbl) ? Math.round(frame.css.oil_bbl).toLocaleString("en-US") : DASH,
    },
  ];

  const feasible = schedule?.feasible;

  return (
    <section
      aria-label="Cycle schedule and economics"
      className="flex h-full min-h-0 flex-col rounded-md border border-scada-700 bg-scada-900"
    >
      <header className="flex items-center justify-between gap-2 border-b border-scada-700 px-3 py-1.5">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-300">Schedule &amp; Economics</h2>
        {feasible === undefined ? null : (
          <span
            className={`rounded px-1.5 py-0.5 text-2xs font-semibold uppercase ${
              feasible ? "bg-signal-green/15 text-signal-green" : "bg-signal-red/15 text-signal-red"
            }`}
          >
            {feasible ? "Feasible" : "Not feasible"}
          </span>
        )}
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        <dl className="grid grid-cols-2 gap-1">
          {rows.map((row) => (
            <div key={row.label} className="flex items-baseline justify-between gap-2 rounded border border-scada-700 bg-scada-850 px-2 py-1">
              {/* The label wraps rather than truncating: in a third-width column
                  an elided "CUMULATIVE…" is less use to an operator than two
                  short lines. */}
              <dt className="text-2xs uppercase leading-tight tracking-wide text-slate-400">{row.label}</dt>
              <dd className={`font-mono text-xs font-semibold tabular-nums ${row.tone ?? "text-slate-100"}`}>
                {row.value}
              </dd>
            </div>
          ))}
        </dl>

        {schedule && schedule.reason && schedule.reason.trim() !== "" ? (
          <p className="mt-2 rounded border border-scada-700 bg-scada-850 px-2 py-1.5 text-2xs leading-snug text-slate-300">
            <span className="font-semibold uppercase tracking-wide text-slate-400">Optimiser: </span>
            {schedule.reason}
          </p>
        ) : null}
      </div>
    </section>
  );
}
