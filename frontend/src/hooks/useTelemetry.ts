import { useCallback, useEffect, useRef, useState } from "react";
import { resolveWebSocketUrl } from "../api";
import type { ConnectionState, TelemetryFrame, UseTelemetryResult } from "../types";

/** Backoff ladder in ms: 1s doubling to a 30s ceiling, with jitter. */
const BASE_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 30000;

/** A stream older than this is treated as stale for the "live" indicator. */
const STALE_AFTER_MS = 2500;

function backoffDelay(attempt: number): number {
  const exponential = Math.min(MAX_BACKOFF_MS, BASE_BACKOFF_MS * 2 ** Math.max(0, attempt - 1));
  // Jitter avoids a thundering herd of reconnecting dashboards on backend restart.
  return Math.round(exponential * (0.7 + Math.random() * 0.6));
}

function isTelemetryFrame(value: unknown): value is TelemetryFrame {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<TelemetryFrame>;
  return (
    candidate.type === "telemetry" &&
    typeof candidate.timestamp_s === "number" &&
    typeof candidate.well === "object" &&
    candidate.well !== null &&
    typeof candidate.card === "object" &&
    candidate.card !== null
  );
}

/**
 * Subscribes to the telemetry stream and owns reconnection.
 *
 * Unknown message types (`hello`, `pong`, future additions) are ignored rather
 * than treated as errors, and a malformed frame is dropped without tearing the
 * socket down.
 */
export function useTelemetry(): UseTelemetryResult {
  const [frame, setFrame] = useState<TelemetryFrame | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [attempt, setAttempt] = useState(0);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [lastMessageAt, setLastMessageAt] = useState<number | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);

  const [isStale, setIsStale] = useState(false);

  const socketRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const attemptRef = useRef(0);
  /** Guards against the effect cleanup scheduling a reconnect after unmount. */
  const disposedRef = useRef(false);

  const connect = useCallback(() => {
    if (disposedRef.current) return;

    let url: string;
    try {
      url = resolveWebSocketUrl();
    } catch {
      setConnection("offline");
      setLastError("WebSocket URL could not be resolved");
      return;
    }

    setConnection(attemptRef.current === 0 ? "connecting" : "reconnecting");

    let socket: WebSocket;
    try {
      socket = new WebSocket(url);
    } catch (cause) {
      setLastError(cause instanceof Error ? cause.message : "WebSocket construction failed");
      attemptRef.current += 1;
      setAttempt(attemptRef.current);
      setConnection("reconnecting");
      retryRef.current = setTimeout(connect, backoffDelay(attemptRef.current));
      return;
    }

    socketRef.current = socket;

    socket.onopen = () => {
      if (disposedRef.current) return;
      attemptRef.current = 0;
      setAttempt(0);
      setConnection("live");
      setLastError(null);
    };

    socket.onmessage = (event: MessageEvent<unknown>) => {
      if (disposedRef.current) return;
      let payload: unknown;
      try {
        payload = JSON.parse(typeof event.data === "string" ? event.data : String(event.data)) as unknown;
      } catch {
        return; // Malformed payload: drop it, keep the stream alive.
      }
      if (!payload || typeof payload !== "object") return;
      const type = (payload as { type?: unknown }).type;
      if (type === "hello" || type === "pong") return; // Control frames, no data.
      if (!isTelemetryFrame(payload)) return; // Unknown/future types ignored.

      setFrame(payload);
      setLastMessageAt(Date.now());
      setLatencyMs(Number.isFinite(payload.latency_ms) ? payload.latency_ms : null);
    };

    socket.onerror = () => {
      if (disposedRef.current) return;
      // `onclose` always follows; the backoff is scheduled there so it runs once.
      setLastError("Telemetry stream error");
    };

    socket.onclose = (event: CloseEvent) => {
      if (disposedRef.current) return;
      socketRef.current = null;
      attemptRef.current += 1;
      setAttempt(attemptRef.current);
      setConnection("reconnecting");
      setLastError(
        event.code === 1000
          ? "Stream closed by server"
          : `Stream closed (code ${event.code || "unknown"})`,
      );
      const delay = backoffDelay(attemptRef.current);
      retryRef.current = setTimeout(connect, delay);
    };
  }, []);

  useEffect(() => {
    disposedRef.current = false;
    connect();
    return () => {
      disposedRef.current = true;
    };
    // `connect` is stable (no reactive deps); this effect runs once on mount.
  }, [connect]);

  // Staleness is a genuine time-driven condition, so it needs a real clock
  // rather than being inferred during render. Only ticks while connected.
  useEffect(() => {
    if (connection !== "live") {
      setIsStale(false);
      return;
    }
    const tick = () => {
      setIsStale(lastMessageAt !== null && Date.now() - lastMessageAt > STALE_AFTER_MS);
    };
    tick();
    const timer = setInterval(tick, 1000);
    return () => clearInterval(timer);
  }, [connection, lastMessageAt]);

  useEffect(() => {
    return () => {
      if (retryRef.current !== null) clearTimeout(retryRef.current);
      retryRef.current = null;
      const socket = socketRef.current;
      socketRef.current = null;
      if (socket) {
        // Detach handlers first so closing cannot schedule a retry post-unmount.
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        if (socket.readyState === WebSocket.OPEN) {
          socket.close();
        } else if (socket.readyState === WebSocket.CONNECTING) {
          // Closing a socket that has not finished its handshake makes the
          // browser log "closed before the connection is established". Under
          // StrictMode the mount effect is torn down immediately after the
          // first connect, so this is the common path. Defer the close to
          // onopen instead and let the handshake finish quietly.
          socket.onopen = () => socket.close();
        }
      }
    };
  }, []);

  return {
    frame,
    // A silent socket is indistinguishable from a dead one to an operator, so
    // it is surfaced as a reconnecting state rather than a false "live".
    connection: isStale && connection === "live" ? "reconnecting" : connection,
    attempt,
    latencyMs,
    lastError,
  };
}
