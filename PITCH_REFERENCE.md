# ThermaTwin — Pitch Deck Reference & Knowledge Base

> **Purpose:** Single source of truth for building a pitch deck (PPT) for ThermaTwin.
> Everything below is derived from the actual repository. Sections marked **[VERIFY]**
> are claims you should confirm by running the demo before putting them on a slide.
> Sections marked **[SUGGESTION]** are strategic advice, not project facts.
>
> Last updated: 2026-09-28 · Repo: `/home/gux/thermatwin` · Spec ID: **SIH26120** · Field: **Baghewala, Oil India Limited**

---

## 1. Elevator Pitch (30 seconds)

> Heavy oil wells die quietly. As a CSS (Cyclic Steam Stimulation) well cools down between
> steam cycles, crude viscosity climbs by **orders of magnitude**, the sucker-rod pump starts
> to *float* on viscous fluid, the dynamometer card distorts, and rod/pump failures follow —
> usually after production has already collapsed.
>
> **ThermaTwin is a closed-loop digital twin of the reservoir → wellbore → rod string → pump
> chain.** It predicts the thermal state 14 days ahead, reads the dynamometer card with a
> neural network, and automatically retunes the VFD (pump speed) to keep the well inside its
> safe and economic operating envelope — all streaming live at 2 Hz into an operator console.

**One-liner alternative:** *"ThermaTwin: the physics-and-AI digital twin that tells a heavy-oil
sucker-rod pump how fast it should run tomorrow — and can change it itself."*

---

## 2. The Problem

| # | Problem statement | Supporting detail from the project |
|---|-------------------|------------------------------------|
| P1 | **Viscosity explosion between steam cycles.** Baghewala crude is 18 °API heavy oil (SG 0.9465, 18 % asphaltenes). Live-oil viscosity is ~70 cP at reservoir temperature but ≥12,000 cP at 46 °C and 90,434 cP at 20 °C per the field's empirical PVT table. | `backend/app/core/config.py`, `backend/app/physics/rheology.py` |
| P2 | **Rod float / distorted dynamometer cards.** As viscosity rises, downhole load transfer lags the surface → the card stretches horizontally ("rod float"), impact loading and fluid pound appear, and rod string/pump failures follow. | `core/diagnostics.py`, `backend/app/physics/gibbs_solver.py`, `backend/app/ai/dyno_classifier.py` |
| P3 | **Operators react late.** Card interpretation is manual and retrospective; SPM/VFD setpoints are changed by experience, not by a forward-looking model. | Project premise (see [SUGGESTION] §10 for how to source external evidence) |
| P4 | **Steam economics are unforgiving.** CSS stops paying when steam-oil ratio (SOR) exceeds **4.2** or bottom-hole temperature falls below **70 °C** — hard cutoffs encoded in the model. Nobody knows *when* the next cycle will stop being profitable until it already has. | `config.py: instantaneous_sor_cutoff=4.2, cutoff_bht_c=70.0` |

**Visual for the slide:** a CSS lifecycle curve — BHT decaying, viscosity rising on a log axis,
and a red band where the well leaves its safe SPM window.

---

## 3. The Solution — What ThermaTwin Actually Is

A **closed-loop digital twin** that couples seven models in one running loop:

```
   ┌──────────────────────── EDGE / SCADA ────────────────────────┐
   │  PLC/RTU → MQTT (Mosquitto) → stream processor (units,       │
   │  outlier rejection) → TimescaleDB / InfluxDB                 │
   └──────────────────────────────┬────────────────────────────────┘
                                  │ telemetry + synthetic field generator
                                  ▼
 ┌──────────────────────────── DIGITAL TWIN (backend/app) ───────────────────────────┐
 │                                                                                    │
 │  RESERVOIR        WELLBORE           ROD STRING           AI                      │
 │  Marx-Langenheim  Beggs & Brill      Gibbs damped         1D-CNN dyno classifier   │
 │  CSS steam chest  + Ramey/Willhite   wave equation        (6 fault classes)        │
 │  + Vogel IPR      transient heat     surface → pump       BiGRU thermal forecast   │
 │  + overburden     + emulsion         card reconstruction  NPV CSS scheduler       │
 │                                                                                    │
 │                    ▼                    ▼                   ▼                      │
 │              Extended Kalman Filter  ──►  DigitalTwin.step() ──►  SRP Controller    │
 │              (P_ip, skin, R_h, T_bh)                          (SPM / stroke / VFD) │
 └────────────────────────────────────────────┬───────────────────────────────────────┘
                                              │ 2 Hz WebSocket frames
                                              ▼
                       REACT CONSOLE (frontend/) — live dyno card, wellbore
                       profile, KPI strip, VFD panel, CSS lifecycle timeline
                                              │  POST /api/v1/control/apply
                                              ▼
                                    VFD SETPOINT (closed loop)
```

### The loop in one sentence
**Measure → forecast → diagnose → decide → act → verify**, every 500 ms, autonomously or
with an operator in the loop.

### Capability map

| Layer | What it does | Module | Headline method |
|---|---|---|---|
| Rheology | μ(T) for the actual Baghewala crude | `physics/rheology.py` (558 L) | Walther (ASTM D341) fitted to field PVT table + Herschel-Bulkley yield-stress model |
| Reservoir | Steam chest growth, heat loss, cycle economics | `physics/thermal_reservoir.py` (1020 L) | Marx-Langenheim CSS, Vogel IPR w/ skin, SOR tracking |
| Wellbore | Pressure/temperature/holdup traverse pump→wellhead | `physics/hydraulics.py` (914 L) | Beggs & Brill (1975) + Ramey (1963) + Willhite (1986) |
| Rod string | Surface card ↔ downhole card, damping | `physics/gibbs_solver.py` (622 L) | Gibbs damped wave eq. (FD leapfrog + transmission inversion) |
| State estimation | Unmeasured downhole state from surface sensors | `engine/state_estimator.py` (516 L) | Extended Kalman filter on `[P_ip, s, R_h, T_bh]` |
| Fault diagnosis | Reads the dyno card | `ai/dyno_classifier.py` (774 L) | Residual 1D CNN, 128-point cards, 6 classes |
| Forecasting | 14-day ahead BHT + oil rate | `ai/thermal_surrogate.py` (573 L) | Bidirectional GRU |
| Scheduling | Best CSS programme (inject/soak/produce days) | `ai/sor_optimizer.py` (566 L) | NPV-max coarse-to-fine coordinate search under SOR/BHT limits |
| Control | Sets pump speed | `ai/sor_optimizer.py: SRPController` | Hydraulic setpoint rule + rod-float downstroke slowdown |
| Orchestration | Runs it all as one twin | `engine/digital_twin.py` (553 L) | `step()` loop, alarms, `TwinSnapshot` |
| Operator UI | Live console | `frontend/src` (12 files, ~2.8k L) | React 18 + Vite + Tailwind, hand-built SVG charts |
| Data simulator | Full lifecycle field data on demand | `simulation/synthetic_field_generator.py` (400 L) | Physics-faithful generator + instrument noise + injected faults |

**The 6 card classes the CNN can name:**
`NORMAL_FULL_BARREL`, `FLUID_POUND`, `ROD_FLOATING`, `GAS_INTERFERENCE`, `PUMP_TAGGING`, `UNANCHORED_TUBING`

---

## 4. Product Screens (what to screenshot for the deck)

| Screen | File / how to get it | Shows |
|---|---|---|
| **Live SCADA console** | `frontend/` — `npm run dev` | Dyno card, wellbore P/T/holdup profile, KPI strip (SOR, fillage, latency, radius), VFD control panel, CSS lifecycle timeline, status banner |
| **Dyno card overlay** | `DynoCardCanvas.tsx` (489 L) | Surface vs downhole card, SVG, fault-labelled |
| **Wellbore visualizer** | `WellboreVisualizer.tsx` (503 L) | Depth-indexed pressure / temperature / holdup / flow-regime |
| **VFD control panel** | `VFDControlPanel.tsx` (371 L) | SPM + stroke inputs → `POST /api/v1/control/apply`, autonomous/manual badge, risk & fillage meters |
| **Streamlit prototype** | `streamlit run app.py` | Simpler demo: thermal decay chart + dyno overlay + KPI cards, synthetic/live mode toggle, MQTT write-back |

**Demo data source:** `python simulation/synthetic_field_generator.py --url ws://localhost:8000/ws/telemetry`
— replays a whole CSS lifecycle (injection → 5-day soak → 60-day production) with realistic
instrument noise. Add `--fault ROD_FLOATING` (or any card label) to force a fault for a live "watch it diagnose" moment.

---

## 5. Architecture & Tech Stack (for the "How we built it" slide)

### Backend
- **FastAPI + Uvicorn**, async SQLAlchemy 2.0 + asyncpg, Pydantic v2 / pydantic-settings
- **PostgreSQL 16** (well metadata, CSS cycles, equipment, maintenance) + **TimescaleDB**
  hypertables `scada_surface` / `scada_downhole` (SCADA time series) + **InfluxDB** (`vfd_telemetry`)
- **MQTT (Eclipse Mosquitto)** ingestion → unit-conversion stream processor (lbf→N, °F→°C, psi→kPa,
  outlier rejection) → TS writers; Kafka/Influx connectors present in the topology
- **Python physics/ML stack:** NumPy, SciPy, pandas, scikit-learn, PyTorch (CNN + GRU)
- Entrypoint: `backend/app/main.py` (`uvicorn backend.app.main:app`), spec banner `SIH26120`

### Frontend
- **React 18.3 + Vite 5.4 + TypeScript 5.6 + Tailwind 3.4**
- No chart library — all visuals are hand-written SVG (small bundle, full control)
- `useTelemetry.ts`: WebSocket with exponential backoff (1 s→30 s + jitter), staleness detection
  at 2500 ms, frame validation; REST poll of `/health`, `/config`, `/schedule` every 30 s
- Multi-stage Docker build → nginx reverse proxy with `/api/` + `/ws/` upgrade proxying

### Infrastructure
- `docker-compose.yml`: postgres, timescaledb, influxdb, kafka, zookeeper, mosquitto, backend, ingestion
- `run.sh`: 11 modes (`streamlit`, `backend`, `frontend`, `ingestion`, `docker`, `test`, `migrate`, `seed`, …)

### API surface (v3 twin)
`GET /api/v1/health` · `/config` · `/rheology/viscosity` · `/wellbore/traverse` · `/thermal/cycle` · `/schedule` · `/control/state`
`POST /api/v1/srp/diagnose` · `/schedule/evaluate` · `/control/apply` · `/control/autonomy`
`WS /ws/telemetry` — frames every **0.5 s** with card, diagnosis, wellbore, estimate, CSS state, recommendation, alarms, latency.

---

## 6. Proof Points & Numbers (candidate "traction/engineering" slide)

**[VERIFY] each number by running the commands shown before the pitch.**

| Metric | Value | Source / how to verify |
|---|---|---|
| Test functions in repo | **275** (43 prototype + 232 physics/AI/closed-loop) | `pytest tests/ -v` and `pytest backend/tests/ -v` |
| Physics/AI test lines | ~2,980 lines in 6 files | `backend/tests/` |
| Largest physics modules | 1020 / 914 / 774 / 622 / 573 / 566 / 558 lines | `wc -l backend/app/**/*.py` |
| Real-time latency budget | **100 ms** target, 0.5 s stream interval | `config.APIConfig`; measured `latency_ms` is in every WS frame |
| Fault classes detected | 6 | `ai/dyno_classifier.py` |
| Per-class CNN recall | ≥95 % asserted in tests | `backend/tests/test_ai_models.py` |
| Training data | synthetic parametric cards, 10 epochs × 250 samples/label at warmup | `backend/app/main.py:warm_up()` |
| Closed-loop response | cooling → automatic SPM downregulation; fillage collapse → alarm; BHT cutoff → alarm | `backend/tests/test_closed_loop.py` (26 tests) |
| Reservoir validity checks | energy-balance/ledger tests, erfc accuracy, Vogel IPR, SOR cutoffs | `backend/tests/test_thermal_reservoir.py` (52 tests) |
| Committed model weights | `simulation/trained_weights/dyno_classifier.pt` | file exists in repo |
| Economics encoded | $68/bbl oil, $21/m³ steam, $11/bbl lifting, $420/d fixed | `config.EconomicsConfig` |

### Known-good story beats from the test suite (great for "it works" slides)
1. *"We inject a fault into the twin and the system both diagnoses it and acts on it"* —
   `test_every_injected_fault_is_diagnosed_and_acted_on` (parameterised over fault types).
2. *"The wave solver obeys causality — the downhole signal arrives exactly L/a after the surface one."*
3. *"The classifier now trains on damped downhole cards, not pristine surface cards, so it matches
   inference"* — see `talk.md` for the root-cause story (a great "engineering rigour" anecdote).

---

## 7. Slide-by-Slide Deck Outline (12 slides, ~10 min)

Each slide: **Title → bullets → speaker notes → visual**.

---

**S1 — Title**
- *ThermaTwin — Closed-Loop Digital Twin for Heavy-Oil Sucker-Rod Pumps*
- Sub: Baghewala field · Oil India Limited · Spec SIH26120 · Team + date
- **Notes:** One-sentence hook: "We don't just visualise the well — we operate it."
- **Visual:** Console screenshot, darkened, with the dyno card glowing.

**S2 — The problem**
- Heavy oil + cyclic steam = viscosity swings of 100×+ between cycles
- Rod float, distorted cards, failed rods, lost production
- Steam economics have hard cutoffs (SOR 4.2 / BHT 70 °C) that arrive unannounced
- **Notes:** Walk the causal chain: *cools → thickens → card distorts → pump underperforms →
  failure → deferred oil*. Stress that today's response is manual and reactive.
- **Visual:** Cooling/viscosity curve with the failure zone shaded red.

**S3 — Why now / Why this is hard**
- Downhole state is not measured — you only see the surface card
- Physics spans three domains (reservoir, wellbore, rod dynamics) that are usually modelled separately
- Operators need answers in seconds, not after post-well analysis
- **Notes:** Frame the gap: "everyone has dashboards; nobody has a twin that closes the loop."

**S4 — Solution overview**
- Diagram from §3 (the closed loop)
- Three pillars: **Physics-faithful · AI-augmented · Actuating**
- **Notes:** Emphasise it is *one* twin, not seven notebooks: reservoir → wellbore → rod → AI → EKF → controller → VFD.

**S5 — How it works (physics)**
- Marx-Langenheim CSS steam chest + Vogel IPR w/ skin + overburden heat loss
- Beggs & Brill multiphase + Ramey/Willhite transient wellbore heat transfer
- Gibbs damped wave equation: surface card → pump card, causally correct
- Walther rheology calibrated on the actual Baghewala PVT table
- **Notes:** This is the credibility slide. Name the classic methods — a domain judge will nod.
- **Visual:** Small formula callouts + the wellbore traverse screenshot.

**S6 — How it works (AI)**
- 1D-CNN reads the card → 6 fault classes (fluid pound, rod float, gas interference, pump tagging, unanchored tubing, normal)
- Bidirectional GRU forecasts 14 days of BHT + oil rate
- Extended Kalman filter fuses surface measurements into unmeasured downhole state
- NPV-max CSS schedule optimiser under SOR/BHT constraints
- **Notes:** Key line: *"the AI never replaces the physics — the CNN classifies the physics solver's
  output, and the EKF keeps the state honest."*
- **Visual:** Card thumbnail strip showing one card per fault class.

**S7 — Closed loop in action (LIVE DEMO)**
1. `docker compose up` / `uvicorn backend.app.main:app` → `npm run dev`
2. `python simulation/synthetic_field_generator.py` — watch frames arrive at 2 Hz
3. Show live dyno card + KPI strip + wellbore profile
4. Inject a fault (`--fault ROD_FLOATING`) → watch the classifier flag it and the VFD panel react
5. Toggle **autonomous** → the twin changes the SPM setpoint itself
- **Notes:** Have a pre-recorded backup video/screenshot in case of network issues.
- **Visual:** The running console.

**S8 — Architecture**
- Diagram from §5, one row per layer: edge → ingest → twin → API/WS → console
- Stack badges: FastAPI · TimescaleDB · MQTT · Kafka · React · Docker
- **Notes:** Mention scale story: hypertables for time series, 2 Hz streaming, WS with backoff/reconnect.

**S9 — Validation & engineering rigour**
- **275 tests** across physics, AI and the closed loop
- Energy-balance checks on the reservoir model, causality/Courant checks on the wave solver,
  published-closed-form checks on Beggs & Brill, per-class recall ≥95 % on the classifier
- Fault-injection tests: every injected fault is diagnosed *and* acted on
- **Notes:** Use the `talk.md` anecdote about matching train/inference distributions.

**S10 — Impact / value**
- [SUGGESTION — quantify before presenting] Candidate value levers:
  - Fewer rod/pump failures → less deferred oil, fewer workovers
  - Automatic SPM tuning → operation at the safe/economic optimum instead of a conservative default
  - Cycle scheduling optimised on NPV with SOR cutoffs → steam cost avoided
  - Sub-second decision latency → act during the stroke, not after the failure
- **Notes:** If you can't source real field numbers, present these as *model outputs* from
  `/api/v1/schedule` and `/thermal/cycle` (they return NPV, cumulative SOR, net revenue) rather than as claimed savings.

**S11 — Roadmap**
- [SUGGESTION] e.g.: ① harden ingestion (wire Kafka writers, schema registry) ② auth/RBAC +
  audit trail for actuation ③ multi-well fleet view ④ FEM/high-fidelity rod model upgrade
  ⑤ on-prem/edge deployment at the wellsite ⑥ historical back-testing against real card archives

**S12 — The ask**
- [SUGGESTION] State precisely: pilot wells, field data access, mentorship/compute, or funding amount.
- Contact + repo + demo link.

**Backup slides (not shown, answer questions):**
- B1: full API endpoint table (§5)
- B2: model parameters table for BAG-17 (TD 1050 m, pump 1000 m, 40 rod segments, 9 SPM,
  2.44 m stroke, η 0.72, unit 57.2 mm plunger, k 320 mD, skin 3.2, net pay 3.5 m, R_e 60 m)
- B3: test-suite breakdown by file
- B4: honest known-gaps slide (§9)

---

## 8. Differentiation (for "why not X" questions)

| Alternative | How ThermaTwin differs |
|---|---|
| Conventional SCADA / dashboards | Visualise only; ThermaTwin forecasts, diagnoses **and actuates** |
| Standalone dyno-card analytics tools | Card is one sensor input in a twin that also models the reservoir and wellbore; card interpretation is physically conditioned on the current thermal state |
| Pure ML "black box" anomaly detector | Physics (Gibbs/Marx-Langenheim/Beggs-Brill) generates and constrains the data; the EKF keeps the state observable; every recommendation is explainable from model state |
| Reservoir-only simulators | They stop at BHT/NPV; ThermaTwin goes all the way to a VFD setpoint on the rod pump |
| [SUGGESTION] Named commercial competitors | Research before adding — only include a row if you can defend it |

---

## 9. Honest Disclosure — Known Gaps (put on a backup slide, not the main deck)

Use these if asked; do **not** claim these are solved:

1. **No authentication/authorization** anywhere (API, WebSocket, MQTT, frontend). CORS only.
2. **Two FastAPI apps exist**: `backend/main.py` (v2, Postgres CRUD, launched by Docker/run.sh)
   and `backend/app/main.py` (v3, the physics/AI twin the frontend targets). v3 has no
   launcher script yet — start it manually with `uvicorn backend.app.main:app`.
3. **Kafka / Timescale / Influx writers are provisioned but not fully wired**: no `aiokafka`
   usage in code, `tsdb_writer.py` and `influx_writer.py` have no callers, and the MQTT→Kafka
   forward is a log-only stub.
4. `frontend` is not a service in `docker-compose.yml` (it has its own Dockerfile + nginx conf).
5. `torch` is not declared in `backend/requirements.txt` although the AI modules need it.
6. `run.sh migrate` / `seed` reference files (`alembic.ini`, `backend.db.seed`) that don't exist.
7. `pytest.ini` `testpaths = tests` means a bare `pytest` runs only the 43 prototype tests —
   run `pytest backend/tests/` explicitly for the full 232.
8. Data is **synthetic-but-physics-faithful** (field-calibrated parameters, instrument noise
   model); it is not yet a live connection to OIL's SCADA. Say "field-calibrated simulation",
   never "running on real wells today".

---

## 10. Evidence You Should Add Before Pitching **[SUGGESTION]**

- Run and paste: `pytest tests/ -q`, `pytest backend/tests/ -q` → final counts.
- Capture 3 screenshots: console, dyno card with a fault flag, VFD panel changing setpoint.
- Record a 45-second demo video as a fallback.
- If pitching to OIL/hackathon judges: pull 1–2 external stats on rod-pumping failures /
  workover cost per rod string / heavy-oil share of production to size the market (label sources).
- Decide the ask (data, pilot, funding) — S12 is empty by design.

---

## 11. Glossary (put in appendix or use to keep jargon consistent)

| Term | Meaning |
|---|---|
| **SRP** | Sucker-Rod Pump — beam/pump jack lifting oil via a rod string |
| **Dynamometer (dyno) card** | Load vs. position plot of one pump stroke; its shape diagnoses downhole faults |
| **Rod float** | Downhole load transfer lags the surface because of viscous drag; card stretches on the downstroke |
| **Fluid pound** | Pump barrel under-fills; rod slams onto the fluid level on downstroke |
| **SPM** | Strokes per minute — pump speed |
| **VFD** | Variable Frequency Drive — electronic motor speed controller |
| **CSS** | Cyclic Steam Stimulation ("huff and puff"): inject steam → soak → produce |
| **SOR** | Steam-Oil Ratio — m³ steam per barrel oil; the economics kill-switch |
| **BHT / BHP** | Bottom-hole temperature / pressure |
| **Fillage** | Fraction of the pump barrel actually filled with liquid |
| **Vogel IPR** | Classic inflow-performance relationship for solution-gas drive |
| **Marx-Langenheim** | Classic steam-flood / steam-chest heated-area model |
| **Beggs & Brill** | Multiphase vertical-flow holdup & pressure-gradient correlation |
| **Ramey / Willhite** | Transient wellbore heat-transfer / overall heat-transfer correlation |
| **Gibbs wave equation** | 1-D damped wave model propagating load/position down the rod string |
| **EKF** | Extended Kalman Filter — recursive state estimator for non-linear systems |
| **Walther model** | ASTM D341 viscosity–temperature relation |
| **Herschel-Bulkley** | Yield-stress shear-thinning fluid model (heavy oil emulsions) |
| **Baghewala / BAG-17** | The heavy-oil field / reference well modelled in this twin |

---

## 12. Repository Map (for yourself — where to look up any claim)

```
thermatwin/
├── README.md, PLAN.md, PLAN_FILES.md, PLAN_ENTERPRISE.md   # docs & plans
├── PITCH_REFERENCE.md                                      # this file
├── talk.md                                                 # engineering log (train/inference fix)
├── app.py + core/ + components/ + data/ + tests/           # v1 Streamlit prototype (43 tests)
├── backend/
│   ├── main.py, api/, services/, models/, db/, schemas/,   # v2 CRUD/enterprise API
│   ├── ingestion/                                          # MQTT consumer, stream processor, writers
│   ├── app/                                                # v3 physics+AI twin  ← the pitch core
│   │   ├── main.py           FastAPI + WS + warmup
│   │   ├── core/config.py    all Baghewala field parameters
│   │   ├── physics/          rheology, hydraulics, thermal_reservoir, gibbs_solver
│   │   ├── engine/           digital_twin, state_estimator (EKF)
│   │   ├── ai/               dyno_classifier, thermal_surrogate, sor_optimizer
│   │   └── api/              routes, websocket
│   └── tests/                 232 physics/AI/closed-loop tests
├── frontend/               React 18 + Vite console (src/: App, 6 components, hook, api, types)
├── simulation/             synthetic_field_generator.py + trained_weights/dyno_classifier.pt
├── docker-compose.yml      postgres, timescaledb, influxdb, kafka, zookeeper, mosquitto, backend, ingestion
└── run.sh                  11 run modes
```
