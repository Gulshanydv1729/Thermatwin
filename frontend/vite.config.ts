import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

/**
 * The dev server proxies /api and /ws to the FastAPI backend so the app runs
 * with identical relative URLs in development (via this proxy) and in
 * production (via nginx.conf). `VITE_API_BASE` is honoured for both the proxy
 * target and the client-side default.
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiBase = env.VITE_API_BASE || "http://localhost:8000";
  // The proxy `target` must be a bare origin: http-proxy appends the incoming
  // request path to it. Pointing it at a full `.../ws/telemetry` URL therefore
  // rewrote /ws/telemetry into /ws/telemetry/ws/telemetry and the backend
  // answered the handshake with 403, so the dashboard never went live.
  const wsTarget = env.VITE_WS_URL || apiBase.replace(/^http/, "ws");

  return {
    plugins: [react()],
    server: {
      host: true,
      port: 5173,
      proxy: {
        "/api": {
          target: apiBase,
          changeOrigin: true,
        },
        "/ws": {
          target: wsTarget,
          changeOrigin: true,
          ws: true,
        },
      },
    },
    preview: {
      host: true,
      port: 4173,
    },
    build: {
      outDir: "dist",
      sourcemap: false,
      chunkSizeWarningLimit: 900,
    },
  };
});
