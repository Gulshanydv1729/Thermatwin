# ThermaTwin — Well-to-Surface Digital Twin

**SIH26120 · Oil India Limited · Baghewala heavy-oil asset**

A physics-informed digital twin of a cyclic steam stimulation (CSS) well coupled
to its sucker rod pump. It carries the well from the steam chest in the
reservoir up to the polish rod load cell, diagnoses the downhole pump card in
real time, reconciles the result with surface instruments through an extended
Kalman filter, and closes the loop on the VFD.

It is not a rule-based dashboard. Every number on screen comes out of a solver,
and every solver is covered by a test.

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
- [Docker](#docker)
- [API reference](#api-reference)
- [WebSocket frame contract](#websocket-frame-contract)
- [Tests](#tests)
- [Configuration](#configuration)
- [Known limitations](#known-limitations)
- [The legacy prototype](#the-legacy-prototype)

---

## What it does

For one CSS cycle the twin answers five questions continuously:

1. **Is the reservoir still hot enough to produce?** A Marx–Langenheim steam
   chest is grown during injection, drawn down through Ramey transient heat
   transfer into the formation and the bounding shales, and the resulting
   bottom-hole temperature drives live-oil viscosity through a Walther-type
   correlation. Low viscosity means the formation can supply more than the pump;
   high viscosity means it cannot, and the pump starts pounding.
2. **What is the downhole card really doing?** The surface card is pushed
   through a Gibbs damped-wave-equation inversion of the rod string. Motion and
   load are both delayed by `L/a` and scaled by `exp(−cL/a)`, so the *shape* of
   the card — parallelogram, fluid-pound collapse, tagging spike — survives the
   trip downhole unchanged and is re-phased onto bottom dead centre.
3. **What fault is that?** A residual 1D CNN (dilated conv blocks, GELU) reads
   the normalised downhole card directly as two channels — position and load —
   and classifies the pump state into one of six labels. A separate analytic
   rule set over ten geometric `card_features()` cross-checks it, and the frame
   reports both plus `agrees_with_analytic`.
4. **What is actually happening, given the instruments?** An EKF fuses
   surface load, casing head pressure and wellhead temperature with the
   downhole model to estimate pump intake pressure, skin factor, thermal radius
   and bottom-hole temperature, and reports a residual norm so you can see
   whether the model and the plant still agree.
5. **What should the VFD do?** A hydraulic controller picks SPM and stroke
   length from the fillage and the rod-float risk index, either as a
   recommendation or applied automatically.

When the cumulative or instantaneous steam–oil ratio passes the 4.2 cut-off, or
the bottom-hole temperature falls below 70 °C, the cycle **terminates**. That
matters: without it the schedule optimiser degenerates, because running the
full production period always wins and it always picks minimum steam.

---

## Architecture

```
            ┌────────────────┬─────────────────────────────────────────────┐
            │ CSS RESERVOIR  ·  physics/thermal_reservoir.py               │
            │ Marx–Langenheim chest growth · Ramey U(t) heat loss          │
            │ erfc over/underburden · vertical_contact_factor = 0.08       │
            └────────────────┬─────────────────────────────────────────────┘
                             │ BHT, thermal radius, SOR, fillage
            ┌────────────────┬─────────────────────────────────────────────┐
            │ WELLBORE  ·  physics/hydraulics.py + rheology.py             │
            │ Beggs–Brill multiphase traverse · live-oil viscosity         │
            │ Grunberg–Nissan emulsion · Walther temperature law           │
            └────────────────┬─────────────────────────────────────────────┘
                             │ pressure / temperature / holdup
            ┌────────────────┬─────────────────────────────────────────────┐
            │ ROD STRING  ·  physics/gibbs_solver.py                       │
            │ 40-segment Gibbs wave equation · pump differential           │
            │ shape-preserving transmission · re-phased to BDC             │
            └────────────────┬─────────────────────────────────────────────┘
                             │ downhole card (position, load)
            ┌────────────────┬─────────────────────────────────────────────┐
            │ AI  ·  ai/dyno_classifier.py · thermal_surrogate.py          │
            │ residual 1D CNN over the card (6 states) · bidirectional GRU │
            │ SOR optimiser over the injection/soak/production grid        │
            └────────────────┬─────────────────────────────────────────────┘
                             │ label, confidence, features
            ┌────────────────┬─────────────────────────────────────────────┐
            │ STATE ESTIMATION  ·  engine/state_estimator.py               │
            │ EKF: PIP, skin, thermal radius, BHT                          │
            │ residual norm: does the twin still agree with the plant?     │
            └────────────────┬─────────────────────────────────────────────┘
                             │
            ┌────────────────┬─────────────────────────────────────────────┐
            │ CONTROLLER  →  VFD setpoint (SPM, stroke length)             │
            └────────────────┬─────────────────────────────────────────────┘
                             │
            ┌────────────────┬─────────────────────────────────────────────┐
            │ FRAME  ·  engine/digital_twin.py                             │
            │ TwinSnapshot.as_frame() — the wire contract                  │
            └────────────────┬─────────────────────────────────────────────┘
                             │ REST /api/v1   ·   WebSocket /ws/telemetry @ 500 ms
```

`TwinSnapshot.as_frame()` in `backend/app/engine/digital_twin.py` is the single
source of truth for the wire format. `frontend/src/types.ts` mirrors it. If the
two disagree, the backend is right and the frontend is wrong.

---

## Repository layout

```
thermatwin/
├── backend/
│   ├── app/                    ← THE PRODUCT (v3)
│   │   ├── main.py             FastAPI app: lifespan warm-up, CORS, routers
│   │   ├── core/config.py      every field constant, env-overridable
│   │   ├── physics/
│   │   │   ├── thermal_reservoir.py   CSS chest + Ramey heat loss
│   │   │   ├── hydraulics.py          Beggs–Brill wellbore traverse
│   │   │   ├── rheology.py            live oil, emulsion, Walther
│   │   │   └── gibbs_solver.py        rod string wave equation
│   │   ├── ai/
│   │   │   ├── dyno_classifier.py     1D residual CNN, 6 pump states
│   │   │   ├── thermal_surrogate.py   bidirectional GRU
│   │   │   └── sor_optimizer.py       injection/soak/production search
│   │   ├── engine/
│   │   │   ├── digital_twin.py        coupling + as_frame() contract
│   │   │   └── state_estimator.py     EKF
│   │   ├── api/
│   │   │   ├── routes.py              /api/v1 REST
│   │   │   └── websocket.py           /ws/telemetry
│   │   └── requirements.txt
│   ├── tests/                  297 tests for the digital twin
│   ├── main.py                 ← v2 "enterprise" shim (Postgres/Kafka). Not served.
│   ├── api/ db/ models/ schemas/ services/ ingestion/   ← v2 layer, not served
│   └── config.py
├── frontend/                   React + Vite + Tailwind dashboard
│   ├── src/types.ts            mirrors the WebSocket frame
│   ├── nginx.conf              proxies /api/ and /ws/ → backend:8000
│   └── Dockerfile              node build → nginx runtime
├── simulation/
│   ├── synthetic_field_generator.py   SCADA replay streamer
│   └── trained_weights/               dyno_classifier.pt (build artifact)
├── tests/                      legacy suite — 42 passed, 1 skipped
├── core/  components/  app.py  ← LEGACY Streamlit prototype
├── Dockerfile.backend          the image; serves backend.app.main:app
├── docker-compose.yml          backend + frontend
├── docker-compose.yml.bak      the previous v2 data stack, verbatim
├── pytest.ini
└── talk.md                     the shared agent coordination log and contract
```

> `backend/main.py` and `backend/api/`, `backend/db/`, `backend/services/`,
> `backend/models/`, `backend/schemas/`, `backend/ingestion/` are the older
> "v2 enterprise" Postgres/Kafka/MQTT code. They are still in the tree and
> `tests/` still exercises parts of them, but **nothing in the container serves
> them** and no dependency they need is installed in the backend image. If you
> see `uvicorn backend.main:app` anywhere, it is wrong — that app exposes only
> `/health` and `/`.

---

## Quickstart

`run.sh` starts any service on its own, or the whole thing at once. It creates
the virtualenv, installs the right dependency set for what you asked for, and
refuses to start on a port that is already taken — naming the holder where the
OS will say who it is.

```bash
./run.sh doctor        # verify the environment; report port clashes
./run.sh up            # API :8000 + dashboard :5173 in one terminal
./run.sh docker-detach # the same, containerised (dashboard :8080)
```

| Command | What it starts |
|---|---|
| `up` | API **and** dashboard together; Ctrl-C stops both |
| `backend` / `backend-prod` | the API alone (`:8000`), with or without reload |
| `frontend` | the dashboard alone (`:5173`) |
| `replay` | replays a synthetic Baghewala lifecycle into a running twin |
| `docker` / `docker-detach` | the whole stack in containers |
| `status` / `logs [service]` / `stop` / `clean` | container management |
| `test` / `test-api` / `test-legacy` | the test suites |
| `frontend-build` / `lint-frontend` | bundle the dashboard, type-check it |
| `legacy-streamlit` / `legacy-backend` | the preserved prototype and shim |

`./run.sh help` is the full reference. Ports are overridable:

```bash
API_PORT=8010 WEB_PORT=5180 ./run.sh up
```

The first API start trains the dyno card classifier (a few seconds) and caches
it in `simulation/trained_weights/dyno_classifier.pt`. Every later start loads
it. Set `THERMATWIN_SKIP_WARMUP=1` to skip it entirely.

### Doing it by hand

```bash
# 1. Backend
python3 -m venv venv
./venv/bin/pip install -r requirements.txt          # legacy prototype deps
./venv/bin/pip install -r backend/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu   # CPU-only torch

# 2. Tests — both suites (pytest.ini collects both)
./venv/bin/python -m pytest -q

# 3. API
./venv/bin/python -m uvicorn backend.app.main:app --reload --port 8000
#   http://localhost:8000/docs

# 4. Dashboard
cd frontend && npm install && npm run dev
#   http://localhost:5173
```

The dashboard's dev server proxies `/api` and `/ws` to the API, so the client
uses site-relative URLs in development and the same paths in production behind
nginx. If you move the API off `:8000`, set `VITE_API_BASE` — but **do not** set
`VITE_WS_URL` to a full `ws://host/path`, because vite uses it as the *proxy
target* and appends the incoming path to it, rewriting `/ws/telemetry` into
`/ws/telemetry/ws/telemetry` and answering the handshake with 403.

### Replay a synthetic field

Needs the API running. `run.sh replay` checks first and says so if it is not:

```bash
FAULT=ROD_FLOATING CYCLES=1 ./run.sh replay
```

`FAULT` accepts any of `NORMAL_FULL_BARREL`, `FLUID_POUND`, `ROD_FLOATING`,
`GAS_INTERFERENCE`, `PUMP_TAGGING`, `UNANCHORED_TUBING`; the injected fault is
carried through the Gibbs inversion and must come back out as the same
diagnosis. Or call the generator directly:

```bash
./venv/bin/python simulation/synthetic_field_generator.py \
    --url http://localhost:8000 --cycles 3 --production-days 60
```

It POSTs SCADA scans to `/api/v1/telemetry/ingest`, which replays them through
the twin and advances the reservoir clock as the replay crosses into new days.

---

## Docker

```bash
docker compose up --build
```

| Service | Host | Notes |
|---|---|---|
| `frontend` | <http://localhost:8080> | nginx; proxies `/api/` and `/ws/` to the backend |
| `backend` | <http://localhost:8000> | FastAPI; `/docs`, `/api/v1`, `/ws/telemetry` |

The frontend waits on `backend` being **healthy**, not merely started — the
health check is a real request to `/api/v1/health`. The 90 s start period covers
the case where the build context carried no checkpoint and the first boot has
to train the classifier.

The AI checkpoint lives in a named volume (`weights`) mounted at
`/data/trained_weights`. The image seeds that path from
`simulation/trained_weights` at build time and Docker copies image content into
a fresh volume on first use, so **the container does not retrain on every
restart**.

Other useful invocations:

```bash
docker compose --profile simulation run --rm streamer   # synthetic field replay
docker compose --profile legacy up -d                  # the old v2 data stack
docker compose logs -f backend
```

Notes:

- Build context for the backend is the **repository root**, not `backend/`,
  because the app is imported as `backend.app.*` and resolves its checkpoint
  against `simulation/`. A `.dockerignore` keeps the 1.8 GB local `venv/` out
  of the context.
- The dashboard is published on **8080**. The legacy InfluxDB keeps 8086, so the
  two profiles can run at once.
- The backend runs a single uvicorn worker on purpose: the twin holds
  process-wide state (the EKF and the VFD setpoints), so a second worker would
  serve a divergent twin against the same well. Scaling out needs an external
  store first.
- The legacy v2 services (postgres, timescaledb, influxdb, zookeeper, kafka,
  mosquitto) are **not** in the default profile — nothing in the digital twin
  reads a database or publishes a topic. They are preserved under
  `docker compose --profile legacy up -d` and verbatim in `docker-compose.yml.bak`.

---

## API reference

Base path `/api/v1`. Interactive docs at `/docs` (Swagger) and `/redoc`.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness; also the container health check |
| `GET` | `/api/v1/config` | The full effective configuration, env overrides applied |
| `GET` | `/api/v1/rheology/viscosity?temperature_c=` | Live-oil and emulsion viscosity at a temperature |
| `GET` | `/api/v1/wellbore/traverse?mass_flow_kg_s=&segments=&water_cut=` | Beggs–Brill pressure/temperature traverse down the well |
| `GET` | `/api/v1/thermal/cycle?production_days=&cycles=` | Run a CSS cycle and return the thermal + production history |
| `POST` | `/api/v1/srp/diagnose` | Classify a raw surface card; body `{position_m[], load_n[], duration_s, temperature_c}` |
| `POST` | `/api/v1/telemetry/ingest` | Replay a batch of SCADA scans (`{samples[], apply_setpoints}`) through the twin |
| `GET` | `/api/v1/schedule` | The current recommended injection/soak/production schedule |
| `POST` | `/api/v1/schedule/evaluate` | Score an alternative schedule |
| `GET` | `/api/v1/control/state` | Current VFD setpoints and whether autonomy is on |
| `POST` | `/api/v1/control/apply` | Body `{spm, stroke_length_m, autonomous}`; applies the setpoint when `autonomous` |
| `POST` | `/api/v1/control/autonomy` | Body `{enabled}`; turns the autonomous VFD loop on or off |

### Examples

```bash
curl -s localhost:8000/api/v1/health

curl -s 'localhost:8000/api/v1/rheology/viscosity?temperature_c=46'
#   -> live-oil viscosity at reservoir temperature

curl -s 'localhost:8000/api/v1/wellbore/traverse?mass_flow_kg_s=4.2&segments=80'
#   -> depth_m[], pressure_mpa[], temperature_c[], liquid_holdup[], regime[]

curl -s -X POST localhost:8000/api/v1/control/apply \
     -H 'content-type: application/json' \
     -d '{"spm": 9.0, "stroke_length_m": 2.44, "autonomous": false}'

curl -s -X POST localhost:8000/api/v1/control/autonomy \
     -H 'content-type: application/json' -d '{"enabled": true}'
```

`POST /control/apply` checks the setpoint against `THRESHOLDS.min_spm` /
`max_spm` (2.0–14.0) and a 0.50–3.50 m stroke range. An out-of-range setpoint
comes back as **HTTP 200 with `"accepted": false`** and a `message` explaining
which bound was violated — the current applied setpoints are left untouched, so
a caller must check the `accepted` field rather than the status code.
Passing `"autonomous": true` also switches the closed loop on.

---

## WebSocket frame contract

`ws://<host>:8000/ws/telemetry` — one frame every **500 ms**
(`THERMATWIN_STREAM_INTERVAL_S`). The first frame is a `hello`, then `telemetry`
frames until disconnect.

| Key | Contents |
|---|---|
| `type` | `"telemetry"` (also `"hello"`, `"ack"`, `"error"`) |
| `timestamp_s` | Twin clock, seconds |
| `well` | `name`, `cycle`, `phase` (`INJECTION`/`SOAKING`/`PRODUCTION`), `day`, `days_into_phase` |
| `card` | `surface` + `downhole` (`position_m[]`, `load_n[]`), `stroke_m`, `min_load_n`, `max_load_n`, `load_span_n` |
| `diagnosis` | `label`, `confidence`, `probabilities`, `features`, `analytic_label`, `agrees_with_analytic` |
| `wellbore` | `depth_m[]`, `pressure_mpa[]`, `temperature_c[]`, `liquid_holdup[]`, `mixture_density_kg_m3[]`, `regime[]` |
| `estimate` | `pump_intake_pressure_mpa`, `skin_factor`, `thermal_radius_m`, `bottom_hole_temperature_c`, `pip_standard_error_mpa`, `residual_norm` |
| `css` | `cumulative_sor`, `instantaneous_sor`, `steam_tonnes`, `oil_bbl`, `cumulative_steam_m3`, `cumulative_oil_m3`, `cutoff_reached`, `cutoff_reason`, `chest_radius_m`, `pump_fillage` |
| `recommendation` | `spm`, `stroke_length_m`, `current_spm`, `current_stroke_m`, `spm_ratio`, `action`, `reason`, `pump_fillage`, `rod_float_risk_index`, `autonomous` |
| `alarms` | array of strings — added by the stream, not by `as_frame` |
| `latency_ms` | wall time of one full twin pass |

**Feature keys are contract, not implementation detail.**
`diagnosis.features` emits exactly what `card_features()` returns — ten
geometric descriptors of the card — and the frontend reads them by name:

`stroke_m`, `load_span_n`, `upstroke_reversal`, `downstroke_asymmetry`,
`tail_spike`, `ripple`, `coherence`, `linearity`, `skew`,
`top_position_fraction`

Do not rename them. Adding is fine; removing or renaming is a breaking change
for `frontend/src/types.ts`.

Six pump states are classified (`CardLabel` in
`backend/app/ai/dyno_classifier.py`):

`NORMAL_FULL_BARREL` · `FLUID_POUND` · `ROD_FLOATING` · `GAS_INTERFERENCE` ·
`PUMP_TAGGING` · `UNANCHORED_TUBING`

All six injected faults round-trip through the Gibbs inversion and are
diagnosed at ≥ 0.96 confidence with the correct control action.

---

## Tests

```bash
./venv/bin/python -m pytest -q                  # both suites, from the root
./venv/bin/python -m pytest backend/tests -q    # digital twin
./venv/bin/python -m pytest tests -q            # legacy

cd frontend && npx tsc --noEmit && npx vite build
```

`pytest.ini` sets `testpaths = backend/tests tests` and `pythonpath = .` so a
bare `pytest` at the root runs both. The root `conftest.py` puts the repository
root on `sys.path`, which is all `backend/tests` needs — it imports
`backend.app.*` and needs no fixtures of its own.

### Digital twin — `backend/tests/` (297 collected)

| File | Tests | Covers |
|---|---:|---|
| `test_thermal_reservoir.py` | 52 | chest growth, Ramey loss, phases, SOR cut-off termination |
| `test_api_stream.py` | 60 | REST + WebSocket contract: frame shape, 500 ms cadence, latency budget, setpoint validation |
| `test_hydraulics.py` | 47 | Beggs–Brill traverse, holdup, regimes, wellbore heat transfer |
| `test_ai_models.py` | 45 | classifier accuracy and per-class recall, surrogate, optimiser |
| `test_gibbs_solver.py` | 34 | causality, losslessness, arrival-time convergence, card shape preservation |
| `test_closed_loop.py` | 31 | all six injected faults round-trip and trigger the right action |
| `test_rheology.py` | 28 | live vs dead oil, Walther dependence, Grunberg–Nissan emulsion |

Result: **296 passed, 1 xfailed**.

### Legacy — `tests/` (43 collected)

The pre-existing Streamlit prototype suite and the v2 model/schema/ingestion
tests. Kept green on purpose; it is not replaced by the digital twin.

Result: **42 passed, 1 skipped**.

### Both together

`340 collected — 338 passed, 1 skipped, 1 xfailed`.

> `talk.md` §5 records 237 backend tests. That was the state before
> `test_api_stream.py` landed (60 tests); the suite is now 297.

### A note on timing assertions

`test_closed_loop.py::test_steady_state_latency_is_within_budget` and the
stream-latency tests in `test_api_stream.py` assert wall-clock budgets, and the
AI models train inside the test session. On a loaded machine (load average > 10,
which several concurrent test runners will produce) these can fail on
scheduling, not on physics — the test takes a median of five passes for exactly
this reason. If one of them fails, re-run it in isolation before believing it:

```bash
./venv/bin/python -m pytest backend/tests/test_closed_loop.py -q
```

### Verified state

Measured on 2026-09-28 (see `talk.md` §5):

- Classifier: 100 % validation accuracy, 100 % per-class recall on all six
  states, 100 % agreement with the independent geometric cross-check.
- Steady-state twin pass **9–11 ms** on an idle machine, against a 100 ms
  budget. The first call includes one-off classifier training.
- WebSocket cadence measured at 0.522 s, `latency_ms` 9–11 ms, label
  `NORMAL_FULL_BARREL` at 0.999 confidence.
- In the container: warm-up reports `dyno_classifier: loaded` (checkpoint
  restored from the `weights` volume, not retrained), `/api/v1/health` 200
  direct and through the nginx proxy, first telemetry frame 104 ms, steady
  cadence 549 ms, `latency_ms` 13–105 ms.
- CSS cut-off verified firing: SOR 18.98 > 4.2 with BHT < 70 °C.
- Frontend: `tsc --noEmit` clean, `vite build` **192 kB** (59.5 kB gzip).
- Streamer end to end: 4449 scans generated over 3 cycles, 12/12 replayed,
  ~365 ms/scan.

---

## Configuration

All physics constants live in `backend/app/core/config.py`. A few are
env-overridable through `get_config()`:

| Variable | Default | Effect |
|---|---|---|
| `THERMATWIN_LOG_LEVEL` | `INFO` | Log level |
| `THERMATWIN_WEIGHTS_DIR` | `<repo>/simulation/trained_weights` | Where the classifier checkpoint is cached |
| `THERMATWIN_SKIP_WARMUP` | `0` | `1` skips the on-boot classifier load/train |
| `THERMATWIN_TRAIN_ON_BOOT` | `1` | `0` disables on-boot training |
| `THERMATWIN_STREAM_INTERVAL_S` | `0.5` | Telemetry broadcast period, s |
| `THERMATWIN_PORT` | `8000` | Service port |
| `THERMATWIN_API_GRAVITY` | `18.0` | Crude API gravity |
| `THERMATWIN_SPECIFIC_GRAVITY` | `0.9465` | Crude specific gravity |
| `THERMATWIN_RESERVOIR_TEMPERATURE_C` | `46.0` | Reservoir temperature |
| `THERMATWIN_RANDOM_SEED` | `26120` | Seed for all stochastic components |

Changing a physics constant to make a test pass is not a fix. Several of these
numbers exist because the model was *wrong* before them — 70 cP live oil rather
than 12,000 cP dead oil, 0.85 × reservoir pressure as the producing BHP, a
`vertical_contact_factor` of 0.08, time constants that scale with the injected
chest radius. `talk.md` §3 lists them and §4 explains how each was found.

---

## Known limitations

Honest list.

- **Single well, single rod string.** `PumpingUnitConfig` describes one pump.
  There is no well pad, no interference and no lift optimisation across wells.
- **The EKF is process-local.** State lives in the process. Restarting the
  service resets the filter and the VFD setpoints, and two workers behind a load
  balancer would diverge. This is why the container runs one worker.
- **The classifier checkpoint is a build artifact, not source.** It is produced
  by `warm_up()` on first boot and is not committed. A clean checkout therefore
  trains on first start, which takes a few seconds. It is fully deterministic
  (`THERMATWIN_RANDOM_SEED`), so this is reproducible, but it does mean the
  first boot is not instant.
- **The wellbore model is 1D and steady per pass.** No transient slugging, no
  gas lift through the casing, no thermal history along the annulus between
  passes — the traverse is recomputed from the current bottom-hole temperature
  each time.
- **Injection schedule search is coarse.** The optimiser evaluates a finite grid
  of injection/soak/production combinations; it is not a continuous optimum and
  the "optimal" schedule is optimal with respect to that grid and the current
  cost model.
- **Cut-off is a hard stop at a threshold.** Real CSS turn-around involves
  operational judgement — cyclic thresholds, operator override, ramp-down. Here
  it terminates the cycle outright.
- **No persistence.** The twin does not store history; the thermal cycle and
  schedule endpoints recompute from the configuration each call. There is no
  database, no event log and no replay from disk in v3.
- **The Gibbs solver is a 40-segment discretisation.** Verified to 0.51 % on
  arrival time at 160 segments, but the 40-segment default trades accuracy for
  the latency budget.
- **No field calibration loop.** Skin factor and thermal radius are estimated,
  but there is no history matching against real production data, because there is
  no SCADA history in this repository.
- **`hydraulics.py` emits a `SyntaxWarning` on an invalid `\q` escape sequence**
  in a docstring under Python 3.12+. It is cosmetic and does not affect
  behaviour, but it will become an error in a future CPython and should be
  changed to a raw string. It is visible in `docker compose build backend`
  output.

### Packaging limitations

- **The backend image is 1.57 GB, almost all of it torch.** See the image-size
  trade-off note in the report; a slimmed variant that keeps the weights on a
  mount and drops torch from the runtime is possible but not done, because the
  AI diagnosis is a core feature rather than an optional one.
- **Host ports 8000 and 8080 must be free.** If either is taken, the container
  fails to start with `address already in use`; nothing in the compose file
  shifts them automatically.
- **The image is single-architecture as built.** `python:3.12-slim` plus a pip
  resolve means whatever host you build on is what you get. There is no
  `buildx`/manifest list in this repo, so the image is not portable across
  amd64 and arm64 without a rebuild.
- **No build cache tuning.** The dependency install is one layer, so any change
  to `backend/requirements.txt` reinstalls torch from scratch (~2 min on a cold
  cache). Splitting the heavy pins into a layer of their own would be the first
  thing to do about it.

---

## The legacy prototype

`app.py`, `core/`, `components/`, `requirements.txt` (root), `data/` and
`tests/` are the original Streamlit prototype. It is a simplified 1D
lumped-parameter model with a Savitzky–Golay filter and a viscosity-threshold
optimizer. It is **preserved, not replaced** — `tests/` stays green and CI runs
it — and it is not the product. Do not add features to it; add them to
`backend/app`.

The v2 "enterprise" layer (`backend/main.py`, `backend/api/`, `backend/db/`,
`backend/services/`, `backend/ingestion/`, `backend/models/`,
`backend/schemas/`) is in the same category: real code with real tests, wired to
Postgres, Kafka, MQTT and InfluxDB, and not on the serving path. The compose
file keeps its data stack available under the `legacy` profile.
