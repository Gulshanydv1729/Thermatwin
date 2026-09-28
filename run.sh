#!/usr/bin/env bash
#
# ThermaTwin — Well-to-Surface Digital Twin (SIH26120, Baghewala)
#
# Orchestrates every service in the system.  The digital twin is
#
#   backend   FastAPI  :8000   physics, AI, EKF, telemetry stream at /ws/telemetry
#   frontend  Vite/nginx :5173 (dev) / :8080 (docker)   SCADA dashboard
#   streamer  one-shot  replays a synthetic Baghewala lifecycle into the twin
#
# and the repository additionally *preserves* a pre-existing Streamlit prototype
# and an "enterprise" data shim, reachable through the `legacy-*` commands.
#
#   ./run.sh help        full command reference
#   ./run.sh doctor      check the environment without changing anything
#   ./run.sh up          start the twin locally (backend + frontend)
#   ./run.sh docker      start the whole stack in containers
#
set -euo pipefail

cd "$(dirname "$0")"

# ── configuration ──────────────────────────────────────────────────────────
# Override any of these from the environment, e.g. PORT=9000 ./run.sh backend
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"

# torch is CPU-only here: the models are a few hundred kilobytes of weights, and
# the default index would pull ~2 GB of CUDA runtime for nothing.
TORCH_INDEX="https://download.pytorch.org/whl/cpu"

VENV="${VENV:-venv}"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

# ── output ─────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    BOLD=$'\033[1m'; RED=$'\033[0;31m'; GREEN=$'\033[0;32m'
    YELLOW=$'\033[1;33m'; BLUE=$'\033[0;34m'; DIM=$'\033[2m'; NC=$'\033[0m'
else
    BOLD=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; DIM=""; NC=""
fi
log()  { printf '%s[run.sh]%s %s\n' "$GREEN" "$NC" "$*"; }
info() { printf '%s[run.sh]%s %s\n' "$BLUE" "$NC" "$*"; }
warn() { printf '%s[run.sh]%s %s\n' "$YELLOW" "$NC" "$*" >&2; }
err()  { printf '%s[run.sh]%s %s\n' "$RED" "$NC" "$*" >&2; }
die()  { err "$*"; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# ── environment ────────────────────────────────────────────────────────────
ensure_venv() {
    if [[ ! -x "$PY" ]]; then
        log "Creating the virtual environment in ./$VENV"
        python3 -m venv "$VENV" || die "could not create a virtualenv (needs python3-venv)"
    fi
}

# Install the digital twin's Python dependencies.
install_api_deps() {
    ensure_venv
    [[ -f backend/requirements.txt ]] || die "backend/requirements.txt is missing"
    log "Installing the digital twin's Python dependencies"
    "$PIP" install --quiet --upgrade pip
    # torch comes from the CPU index; everything else from PyPI.
    "$PIP" install --quiet -r backend/requirements.txt --extra-index-url "$TORCH_INDEX"
}

# The legacy prototype has its own, larger dependency set (Streamlit, pandas).
install_legacy_deps() {
    ensure_venv
    [[ -f requirements.txt ]] || die "the root requirements.txt is missing"
    log "Installing the legacy prototype's Python dependencies"
    "$PIP" install --quiet -r requirements.txt
}

install_node_deps() {
    have npm || die "npm is not installed (needed for the dashboard)"
    if [[ ! -d frontend/node_modules ]]; then
        log "Installing the dashboard's Node dependencies (this takes a minute)"
        (cd frontend && npm install --no-fund --no-audit)
    fi
}

# ── preflight ──────────────────────────────────────────────────────────────
# A port clash is the single most common way this goes wrong, and uvicorn's error
# for it is easy to misread, so check before starting rather than after.
# Is anything listening on this port?  This must not depend on being able to
# *name* the holder: `ss -p` omits the users:(...) field for a process owned by
# another user, so a port can be busy while the name lookup returns nothing.  A
# preflight that trusts the name alone silently passes and then uvicorn dies with
# a much harder-to-read "address already in use".
port_busy() {
    local port="$1"
    if have ss; then
        [[ -n "$(ss -ltnH "sport = :$port" 2>/dev/null | head -1)" ]]
    elif have lsof; then
        lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
    elif have netstat; then
        netstat -ltn 2>/dev/null | grep -qE "[:.]$port[[:space:]]"
    else
        return 1   # cannot tell; let the service try and report for itself
    fi
}

# Best-effort "<name> (pid)" for whatever holds the port.  Empty when the holder
# is not ours to inspect, which is normal and not an error.
port_owner() {
    local port="$1" owner=""
    if have ss; then
        # `ss -p` reports users:(("name",pid=123,fd=4)); reduce that to
        # "name (123)" so a port clash reads as something actionable.  The
        # substitution is one pass because the fd field follows the pid.
        owner="$(ss -ltnpH "sport = :$port" 2>/dev/null | head -1 \
            | sed -nE 's/.*users:\(\("?([^",]+)"?,pid=([0-9]+).*/\1 (\2)/p' || true)"
    elif have lsof; then
        owner="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | awk 'NR>1 {print $1" ("$2")"}' | head -1 || true)"
    fi
    [[ -n "$owner" ]] && { echo "$owner"; return 0; }

    # A container published on the host port is held by docker-proxy, which runs
    # as root and is therefore not attributable to this user.  `ss` shows the
    # socket with no users: field at all, so without this the clash reports
    # "a process this user cannot identify" and gives the operator nothing to
    # act on.  Name the container instead.
    if have docker; then
        local container
        container="$(docker ps --filter "publish=$port" --format '{{.Names}}' 2>/dev/null | head -1)"
        if [[ -n "$container" ]]; then
            echo "docker container '$container'"
            return 0
        fi
    fi
    return 0
}

require_free_port() {
    local port="$1" what="$2" var="$3" cmd="$4"
    if ! port_busy "$port"; then
        return 0
    fi
    # Offer a port that is actually free, rather than echoing the busy one back.
    local alt="$port" candidate
    for candidate in $((port + 1)) $((port + 2)) $((port + 10)) $((port + 20)); do
        if ! port_busy "$candidate"; then
            alt="$candidate"
            break
        fi
    done
    local owner
    owner="$(port_owner "$port" || true)"
    err "port $port is already in use — $what cannot start"
    if [[ -n "$owner" ]]; then
        err "  held by $owner"
        case "$owner" in
            docker\ container*)
                err "  stop it with:  ./run.sh stop"
                ;;
        esac
    else
        err "  held by a process this user cannot identify"
    fi
    if [[ "$alt" != "$port" ]]; then
        err "  stop it, or move to a free port:  $var=$alt ./run.sh $cmd"
    else
        err "  stop it first, then re-run:  ./run.sh $cmd"
    fi
    exit 1
}

wait_for_health() {
    local url="$1" tries="${2:-90}" i=0
    while (( i < tries )); do
        if have curl && curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        (( i++ ))
    done
    return 1
}

# ── services ───────────────────────────────────────────────────────────────
# The twin.  A single worker on purpose: the digital twin keeps process-wide
# state (the Kalman filter and the VFD setpoints), so a second worker would serve
# a divergent twin against the same well.
serve_api() {
    local mode="${1:-dev}"
    install_api_deps
    require_free_port "$API_PORT" "the API" "API_PORT" "backend"
    log "Starting the digital twin API on http://localhost:$API_PORT"
    info "  docs      http://localhost:$API_PORT/docs"
    info "  telemetry ws://localhost:$API_PORT/ws/telemetry"
    info "  the first start solves the CSS trajectory and primes the AI; allow ~30 s"
    local -a args=(--host "$API_HOST" --port "$API_PORT" --ws websockets --timeout-keep-alive 75)
    if [[ "$mode" == "dev" ]]; then
        args+=(--reload)
    else
        args+=(--workers 1)
    fi
    exec "$VENV/bin/uvicorn" backend.app.main:app "${args[@]}"
}

serve_frontend() {
    install_node_deps
    log "Starting the dashboard dev server on http://localhost:$WEB_PORT"
    info "  /api and /ws are proxied to http://localhost:$API_PORT"
    # Only VITE_API_BASE is set, and only when the port is non-default.  The
    # WebSocket URL is deliberately left unset: the client then resolves a
    # site-relative /ws/telemetry and the dev server proxies it, which is the
    # arrangement the frontend is built for.  Setting VITE_WS_URL to a full
    # ws://host/path would (a) make the client bypass the proxy and (b) be used
    # by vite as the *proxy target*, where a full path is appended to the
    # incoming request -- rewriting /ws/telemetry into
    # /ws/telemetry/ws/telemetry and answering the handshake with 403.  The
    # proxy target has to be a bare origin.
    if [[ "$API_PORT" != "8000" ]]; then
        export VITE_API_BASE="http://localhost:$API_PORT"
        info "  VITE_API_BASE set to $VITE_API_BASE"
    fi
    cd frontend
    exec npm run dev -- --port "$WEB_PORT" --strictPort
}

# One-shot: replay a synthetic Baghewala lifecycle into a running twin.
# Useful for exercising the ingestion route and for demonstrating a fault.
replay_stream() {
    install_api_deps
    local base="http://localhost:${API_PORT}"
    if ! wait_for_health "$base/api/v1/health" 2; then
        die "no twin at $base — start it first with  ./run.sh backend"
    fi
    local cycles="${CYCLES:-2}" days="${PRODUCTION_DAYS:-60}" \
          fault="${FAULT:-NORMAL_FULL_BARREL}" limit="${LIMIT:-400}"
    log "Replaying $cycles CSS cycle(s) into the twin at $base"
    info "  fault=$fault  production-days=$days  limit=$limit"
    # An injected fault is the interesting demonstration: the twin must diagnose
    # it from the card alone and act on it.
    exec "$PY" simulation/synthetic_field_generator.py \
        --url "$base" --cycles "$cycles" --production-days "$days" \
        --fault "$fault" --limit "$limit" --interval 0.05
}

# ── the whole stack, locally, in one terminal ──────────────────────────────
# The API is started in the background, the dashboard in the foreground, and the
# API is torn down on Ctrl-C so no orphan is left listening on the port.
up() {
    install_api_deps
    install_node_deps
    require_free_port "$API_PORT" "the API" "API_PORT" "up"
    require_free_port "$WEB_PORT" "the dashboard" "WEB_PORT" "up"

    log "Starting the digital twin: API + dashboard"
    (
        "$VENV/bin/uvicorn" backend.app.main:app --host "$API_HOST" --port "$API_PORT" \
            --ws websockets --timeout-keep-alive 75
    ) &
    local api_pid=$!

    # Wait for the API before the dashboard opens, so the first frame has data.
    if wait_for_health "http://localhost:$API_PORT/api/v1/health" 90; then
        log "API healthy on http://localhost:$API_PORT"
    else
        err "the API did not become healthy; its output follows"
        kill "$api_pid" 2>/dev/null || true
        wait "$api_pid" 2>/dev/null || true
        exit 1
    fi

    cleanup() {
        trap - INT TERM EXIT
        info "shutting down"
        kill "$api_pid" 2>/dev/null || true
        wait "$api_pid" 2>/dev/null || true
    }
    trap cleanup INT TERM EXIT

    info ""
    info "  dashboard  http://localhost:$WEB_PORT"
    info "  API docs   http://localhost:$API_PORT/docs"
    info "  telemetry  ws://localhost:$API_PORT/ws/telemetry"
    info ""
    info "Ctrl-C stops both."

    # VITE_WS_URL is deliberately not set here: see serve_frontend for why the
    # proxy target must be a bare origin.
    (cd frontend && VITE_API_BASE="http://localhost:$API_PORT" \
        npm run dev -- --port "$WEB_PORT" --strictPort)
}

# ── docker ─────────────────────────────────────────────────────────────────
require_docker() {
    have docker || die "docker is not installed"
    docker compose version >/dev/null 2>&1 \
        || die "the docker compose plugin is missing (docker compose, not docker-compose)"
}

# docker compose with the port variables exported, so run.sh's API_PORT and
# DASHBOARD_PORT really do move the published ports (docker-compose.yml
# parameterises them).  Without this the preflight would check a port the
# containers then ignore, and `docker compose up` would fail on a bind clash
# the user was told was free.
compose() {
    API_PORT="$API_PORT" DASHBOARD_PORT="$DASHBOARD_PORT" docker compose "$@"
}

docker_up() {
    require_docker
    require_free_port "$API_PORT" "the containerised API" "API_PORT" "docker"
    require_free_port "$DASHBOARD_PORT" "the containerised dashboard" "DASHBOARD_PORT" "docker"
    log "Building and starting the stack (this compiles torch, so allow several minutes)"
    compose up --build "$@"
    # Not reached in the foreground case; used by docker-detach.
}

docker_detach() {
    require_docker
    require_free_port "$API_PORT" "the containerised API" "API_PORT" "docker-detach"
    log "Starting the stack in the background (the first build takes several minutes)"
    compose up --build -d
    log "Waiting for the API to become healthy..."
    if wait_for_health "http://localhost:$API_PORT/api/v1/health" 120; then
        log "stack is up"
    else
        warn "the API is not healthy yet; check  ./run.sh logs backend"
    fi
    cat <<EOF

  dashboard  http://localhost:$DASHBOARD_PORT
  API docs   http://localhost:$API_PORT/docs
  telemetry  ws://localhost:$API_PORT/ws/telemetry

  logs       ./run.sh logs [service]
  stop       ./run.sh stop

EOF
}

# ── checks ─────────────────────────────────────────────────────────────────
doctor() {
    local problems=0
    printf '%sThermaTwin environment check%s\n' "$BOLD" "$NC"
    echo

    printf '  %-26s' "python3"
    if have python3; then
        printf '%sok%s  %s\n' "$GREEN" "$NC" "$(python3 --version 2>&1)"
    else
        printf '%sMISSING%s\n' "$RED" "$NC"; problems=1
    fi

    printf '  %-26s' "virtualenv"
    if [[ -x "$PY" ]]; then
        printf '%sok%s  %s (%s)\n' "$GREEN" "$NC" "$VENV" "$("$PY" --version 2>&1)"
    else
        printf '%snot created%s  ./run.sh backend creates it\n' "$YELLOW" "$NC"
    fi

    printf '  %-26s' "torch (CPU)"
    if [[ -x "$PY" ]] && "$PY" -c "import torch" >/dev/null 2>&1; then
        printf '%sok%s  %s\n' "$GREEN" "$NC" "$("$PY" -c 'import torch;print(torch.__version__)')"
    else
        printf '%snot installed%s\n' "$YELLOW" "$NC"
    fi

    printf '  %-26s' "fastapi / uvicorn"
    if [[ -x "$PY" ]] && "$PY" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
        printf '%sok%s  fastapi %s\n' "$GREEN" "$NC" "$("$PY" -c 'import fastapi;print(fastapi.__version__)')"
    else
        printf '%snot installed%s\n' "$YELLOW" "$NC"
    fi

    printf '  %-26s' "npm"
    if have npm; then
        printf '%sok%s  %s\n' "$GREEN" "$NC" "$(npm --version)"
    else
        printf '%sMISSING%s  needed for the dashboard\n' "$RED" "$NC"; problems=1
    fi

    printf '  %-26s' "frontend/node_modules"
    if [[ -d frontend/node_modules ]]; then
        printf '%sok%s\n' "$GREEN" "$NC"
    else
        printf '%snot installed%s  ./run.sh frontend installs it\n' "$YELLOW" "$NC"
    fi

    printf '  %-26s' "docker + compose"
    if have docker && docker compose version >/dev/null 2>&1; then
        printf '%sok%s  %s\n' "$GREEN" "$NC" "$(docker compose version --short 2>/dev/null)"
    else
        printf '%sunavailable%s  the container path is optional\n' "$YELLOW" "$NC"
    fi

    echo
    # A busy API port is a hard stop; a busy dashboard port only matters if you
    # were about to start the dashboard, so it is a warning.
    local owner
    report_port() {
        local port="$1" label="$2" severity="$3"
        printf '  %-26s' "port $port ($label)"
        if port_busy "$port"; then
            owner="$(port_owner "$port" || true)"
            if [[ -n "$owner" ]]; then
                printf '%sBUSY%s  held by %s\n' "$severity" "$NC" "$owner"
            else
                printf '%sBUSY%s  held by an unidentifiable process\n' "$severity" "$NC"
            fi
            [[ "$severity" == "$RED" ]] && problems=1
        else
            printf '%sfree%s\n' "$GREEN" "$NC"
        fi
    }
    report_port "$API_PORT"        "API"        "$RED"
    report_port "$WEB_PORT"        "dashboard"  "$YELLOW"
    report_port "$DASHBOARD_PORT"  "docker ui"  "$YELLOW"

    echo
    if compgen -G "simulation/trained_weights/*.pt" >/dev/null; then
        printf '  %-26s%scached%s  the twin loads it rather than retraining\n' \
            "classifier checkpoint" "$GREEN" "$NC"
    else
        printf '  %-26s%sabsent%s  the first start will train one\n' \
            "classifier checkpoint" "$YELLOW" "$NC"
    fi

    echo
    if (( problems == 0 )); then
        log "ready —  ./run.sh up   (or ./run.sh docker)"
    else
        warn "something above needs attention"
        exit 1
    fi
}

status() {
    require_docker
    compose ps
}

# ── tests ──────────────────────────────────────────────────────────────────
# pytest.ini already collects both suites, so no paths are passed: that keeps the
# two from drifting apart.
run_tests() {
    local what="${1:-all}"
    install_api_deps
    case "$what" in
        all)     log "Running both suites (this trains the AI models; ~2-3 min)"; exec "$PY" -m pytest ;;
        api)     log "Running the digital twin suite";                      exec "$PY" -m pytest backend/tests -q ;;
        legacy)  install_legacy_deps; log "Running the legacy suite";      exec "$PY" -m pytest tests -q ;;
        *)       die "unknown test target: $what" ;;
    esac
}

# ── help ───────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
${BOLD}ThermaTwin — Well-to-Surface Digital Twin${NC}
SIH26120 · Oil India Limited · Baghewala heavy oil asset

${BOLD}SERVICES${NC}
  backend    :8000   FastAPI — physics, AI, Kalman filter, /ws/telemetry
  frontend   :5173   React SCADA dashboard (Vite dev server)
  streamer   —       replays a synthetic Baghewala lifecycle into a running twin
  docker     :8000/:8080   the above in containers (API + nginx-served dashboard)

${BOLD}STARTING${NC}
  up                  API + dashboard locally, in one terminal (Ctrl-C stops both)
  backend             the API alone, with hot reload
  backend-prod        the API alone, no reload, one worker
  frontend            the dashboard alone
  replay              replay a synthetic lifecycle into a running twin
  docker              the whole stack in containers, in the foreground
  docker-detach       the whole stack in containers, in the background

  replay options (environment variables):
    FAULT=NORMAL_FULL_BARREL|FLUID_POUND|ROD_FLOATING|GAS_INTERFERENCE
          |PUMP_TAGGING|UNANCHORED_TUBING   inject a card fault
    CYCLES=2   PRODUCTION_DAYS=60   LIMIT=400

${BOLD}MANAGING${NC}
  status              container status
  logs [service]      follow container logs
  stop                stop the containers
  clean               delete the volumes and rebuild (retrains the AI)

${BOLD}CHECKING${NC}
  doctor              verify the environment, report port clashes
  test                both suites (~2-3 min; trains the AI models)
  test-api            the digital twin suite only
  test-legacy         the legacy prototype suite only (~1 s)

${BOLD}BUILDING${NC}
  frontend-build      produce the production bundle
  lint-frontend       type-check the dashboard

${BOLD}PRESERVED LEGACY${NC}
  legacy-streamlit    the original Streamlit prototype (:8501)
  legacy-backend      the original "enterprise" FastAPI shim (:8000)

${BOLD}ENVIRONMENT${NC}
  API_PORT=$API_PORT  WEB_PORT=$WEB_PORT  DASHBOARD_PORT=$DASHBOARD_PORT  VENV=$VENV

${BOLD}EXAMPLES${NC}
  ./run.sh doctor
  ./run.sh up
  ./run.sh docker-detach
  FAULT=ROD_FLOATING ./run.sh replay
EOF
}

# ── dispatch ───────────────────────────────────────────────────────────────
case "${1:-help}" in
    # ── the twin ──
    up)             shift; up "$@" ;;
    backend)        serve_api dev ;;
    backend-prod)   serve_api prod ;;
    frontend)       serve_frontend ;;
    replay|stream)  shift || true; replay_stream "$@" ;;

    # ── containers ──
    docker|compose) shift || true; docker_up "$@" ;;
    docker-detach)  docker_detach ;;
    stop)           require_docker; log "stopping the stack"; compose down ;;
    status)         status ;;
    logs)           require_docker; compose logs -f "${2:-}" ;;
    clean)          require_docker
                    warn "deleting volumes — this discards the trained weights"
                    compose down -v
                    log "rebuilding"; compose up --build -d ;;

    # ── checks ──
    doctor|check)   doctor ;;
    test)           run_tests all ;;
    test-api)       run_tests api ;;
    test-legacy)    run_tests legacy ;;

    # ── building ──
    frontend-build) install_node_deps; (cd frontend && npm run build) ;;
    lint-frontend)  install_node_deps; (cd frontend && npm run typecheck) ;;

    # ── preserved legacy ──
    legacy-streamlit)
        install_legacy_deps
        log "starting the preserved Streamlit prototype on http://localhost:8501"
        exec "$VENV/bin/streamlit" run app.py --server.port 8501
        ;;
    legacy-backend)
        install_legacy_deps
        log "starting the preserved enterprise shim on http://localhost:8000"
        exec "$VENV/bin/uvicorn" backend.main:app --host "$API_HOST" --port "$API_PORT"
        ;;

    help|-h|--help|"")  usage ;;
    *)  err "unknown command: $1"; echo; usage; exit 1 ;;
esac
