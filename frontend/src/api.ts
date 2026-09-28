/**
 * HTTP client for the ThermaTwin control plane, plus the shared numeric
 * formatting helpers used by every readout in the dashboard.
 *
 * Base URL resolution:
 *  - `VITE_API_BASE` unset or "" -> site-relative, i.e. same-origin
 *  - `VITE_API_BASE=http://host:8000` -> absolute
 *
 * Site-relative is the default on purpose. Both deployment targets proxy the
 * API and the socket on the same origin -- `vite.config.ts` does it in
 * development and `nginx.conf` does it in the container -- so an absolute
 * default would turn every REST call into a cross-origin request that the
 * backend's CORS policy rejects, and would point the container at its own
 * loopback where no backend is listening.
 */

import type {
  ControlApplyRequest,
  ControlApplyResponse,
  HealthResponse,
  ScheduleResponse,
  WellConfig,
} from "./types";

const envApiBase = import.meta.env.VITE_API_BASE as string | undefined;
const envWsUrl = import.meta.env.VITE_WS_URL as string | undefined;

export const API_BASE: string = (envApiBase === undefined ? "" : envApiBase).replace(
  /\/+$/,
  "");

export const WS_PATH = "/ws/telemetry";

/** Error carrying the HTTP status so callers can distinguish a rejection from a fault. */
export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Builds the absolute websocket URL. Defaults to a site-relative path so the
 * same bundle works behind the dev-server proxy and behind nginx; an explicit
 * `VITE_WS_URL` may still be an absolute ws(s):// endpoint.
 */
export function resolveWebSocketUrl(): string {
  const raw = (envWsUrl === undefined ? WS_PATH : envWsUrl).trim();
  if (/^wss?:\/\//i.test(raw)) return raw;
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const path = raw.startsWith("/") ? raw : `/${raw}`;
  return `${proto}//${window.location.host}${path}`;
}

function endpoint(path: string): string {
  return `${API_BASE}${path}`;
}

/** Pulls a human-readable message out of a FastAPI error body when present. */
function messageFromBody(body: unknown, fallback: string): string {
  if (typeof body === "string" && body.trim() !== "") return body;
  if (body && typeof body === "object") {
    const detail = (body as { detail?: unknown }).detail;
    if (typeof detail === "string" && detail.trim() !== "") return detail;
    if (Array.isArray(detail)) {
      const first = detail[0] as { msg?: unknown } | undefined;
      if (first && typeof first.msg === "string") return first.msg;
    }
    const message = (body as { message?: unknown }).message;
    if (typeof message === "string" && message.trim() !== "") return message;
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(endpoint(path), {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
  } catch (cause) {
    throw new ApiError(
      cause instanceof Error && cause.message
        ? `Network unreachable: ${cause.message}`
        : "Network unreachable",
      0,
    );
  }

  const text = await response.text();
  let parsed: unknown = null;
  if (text !== "") {
    try {
      parsed = JSON.parse(text) as unknown;
    } catch {
      parsed = text;
    }
  }

  if (!response.ok) {
    throw new ApiError(
      messageFromBody(parsed, `HTTP ${response.status} ${response.statusText}`.trim()),
      response.status,
    );
  }
  return parsed as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/v1/health");
}

export function getConfig(): Promise<WellConfig> {
  return request<WellConfig>("/api/v1/config");
}

export function getSchedule(): Promise<ScheduleResponse> {
  return request<ScheduleResponse>("/api/v1/schedule");
}

export function applySetpoint(payload: ControlApplyRequest): Promise<ControlApplyResponse> {
  return request<ControlApplyResponse>("/api/v1/control/apply", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/** Sentinel rendered wherever a value is absent, rather than NaN. */
export const DASH = "—";

/**
 * Core numeric formatter. Anything non-finite (NaN, Infinity, undefined) is
 * rendered as the em dash so a malformed frame can never leak NaN into the UI.
 */
export function fmt(value: number | null | undefined, digits = 1, fallback: string = DASH): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return fallback;
  return value.toFixed(digits);
}

export function fmtInt(value: number | null | undefined, fallback: string = DASH): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return fallback;
  return Math.round(value).toLocaleString("en-US");
}

/** Newtons -> kilonewtons, the unit dyno-cards are conventionally read in. */
export function fmtKn(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return (value / 1000).toFixed(digits);
}

export function fmtPct(fraction: number | null | undefined, digits = 0): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return DASH;
  return (fraction * 100).toFixed(digits);
}

export function fmtUsd(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return `$${Math.round(value).toLocaleString("en-US")}`;
}

/** Clamps to a finite number, substituting `fallback` for anything unusable. */
export function num(value: number | null | undefined, fallback: number): number {
  return value === null || value === undefined || !Number.isFinite(value) ? fallback : value;
}

export function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

/** Extracts the first finite number from an unknown config node. */
export function pickNumber(source: Record<string, unknown> | null | undefined, keys: string[]): number | null {
  if (!source) return null;
  for (const key of keys) {
    const value = source[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

export function pickString(source: Record<string, unknown> | null | undefined, keys: string[]): string | null {
  if (!source) return null;
  for (const key of keys) {
    const value = source[key];
    if (typeof value === "string" && value.trim() !== "") return value;
  }
  return null;
}

/** Digs a nested object out of a config document without walking it recursively. */
export function pickObject(
  source: Record<string, unknown> | null | undefined,
  keys: string[],
): Record<string, unknown> | undefined {
  if (!source) return undefined;
  for (const key of keys) {
    const value = source[key];
    if (value && typeof value === "object" && !Array.isArray(value)) {
      return value as Record<string, unknown>;
    }
  }
  return undefined;
}
