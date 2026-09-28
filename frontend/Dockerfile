# syntax=docker/dockerfile:1

# ---- Stage 1: build the static bundle -------------------------------------
FROM node:20-alpine AS build

WORKDIR /app

# Install dependencies from the lockfile when present so image layers cache.
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund 2>/dev/null || npm install --no-audit --no-fund

COPY tsconfig.json tsconfig.node.json vite.config.ts tailwind.config.js postcss.config.js index.html ./
COPY src ./src

# Empty VITE_API_BASE makes the client use site-relative URLs, so the bundle
# works behind the nginx proxy without a rebuild per environment.
ENV VITE_API_BASE=""
ENV VITE_WS_URL="/ws/telemetry"

RUN npm run build

# ---- Stage 2: serve the bundle and proxy the API ---------------------------
FROM nginx:1.27-alpine AS runtime

RUN rm -f /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/scada.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://127.0.0.1/healthz || exit 1

CMD ["nginx", "-g", "daemon off;"]
