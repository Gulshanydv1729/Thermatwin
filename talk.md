# ThermaTwin — agent coordination log

Shared working document for the agents building ThermaTwin (SIH26120, Oil India
Limited, Baghewala). **Backend/physics is the contract owner; frontend consumes
it.** Append to this file rather than editing other agents' sections.

Last updated: 2026-09-28

---

## 1. Division of labour

| Area | Owner | Files (do not cross) |
|---|---|---|
| Physics | backend agent | `backend/app/physics/**`, `backend/app/core/config.py` |
| AI models | backend agent | `backend/app/ai/**` |
| Engine | backend agent | `backend/app/engine/**` |
| API | backend agent | `backend/app/api/**`, `backend/app/main.py` |
| Simulator | backend agent | `simulation/**` |
| Backend tests | backend agent | `backend/tests/**` |
| Frontend | frontend agent | `frontend/**` |
| Deployment | backend agent | `docker-compose.yml`, `backend/Dockerfile`, `backend/requirements.txt`, `README.md`, `pytest.ini` |

**Legacy code is off-limits to both agents.** The pre-existing Streamlit
prototype (`app.py`, `core/`, `components/`) and its suite (`tests/`) are being
preserved, not replaced. `pytest.ini` must keep `tests/` green.

---

## 2. The contract (the thing we both depend on)

The single source of truth is `backend/app/engine/digital_twin.py` →
`TwinSnapshot.as_frame()`, which emits the WebSocket frame. The frontend must
mirror the shape in `frontend/src/types.ts`.

**WebSocket** `ws://<host>:8000/ws/telemetry`, a frame every **500 ms**:

| Key | Contents |
|---|---|
| `type` | `"telemetry"` (also `"hello"`, `"ack"`, `"error"`) |
| `well` | `name`, `cycle`, `phase` (`INJECTION`/`SOAKING`/`PRODUCTION`), `day`, `days_into_phase` |
| `card` | `surface` + `downhole` (`position_m[]`, `load_n[]`), `stroke_m`, `min_load_n`, `max_load_n`, `load_span_n` |
| `diagnosis` | `label`, `confidence`, `probabilities`, `features`, `analytic_label`, `agrees_with_analytic` |
| `wellbore` | `depth_m[]`, `pressure_mpa[]`, `temperature_c[]`, `liquid_holdup[]`, `mixture_density_kg_m3[]`, `regime[]` |
| `estimate` | `pump_intake_pressure_mpa`, `skin_factor`, `thermal_radius_m`, `bottom_hole_temperature_c`, `pip_standard_error_mpa`, `residual_norm` |
| `css` | `cumulative_sor`, `instantaneous_sor`, `steam_tonnes`, `oil_bbl`, `cumulative_steam_m3`, `cumulative_oil_m3`, `cutoff_reached`, `cutoff_reason`, `chest_radius_m`, `pump_fillage` |
| `recommendation` | `spm`, `stroke_length_m`, `current_spm`, `current_stroke_m`, `spm_ratio`, `action`, `reason`, `pump_fillage`, `rod_float_risk_index`, `autonomous` |
| `alarms` | array of strings (added by the stream, not `as_frame`) |
| `latency_ms` | wall time of one full twin pass |

**Feature keys are real, not invented.** `diagnosis.features` emits exactly the
keys `card_features()` returns: `stroke_m`, `load_span_n`, `upstroke_reversal`,
`downstroke_asymmetry`, `tail_spike`, `ripple`, `coherence`, `linearity`, `skew`,
`top_position_fraction`. The frontend already reads these — do not rename them.

**REST** under `/api/v1`: `GET /health`, `/config`, `/rheology/viscosity`,
`/wellbore/traverse`, `/thermal/cycle`, `/schedule`, `/control/state`;
`POST /control/apply` (`{spm, stroke_length_m, autonomous}`),
`/control/autonomy`, `/srp/diagnose`, `/schedule/evaluate`.

---

## 3. Physics decisions that are NOT negotiable

These were each arrived at by finding and fixing a bug. Reverting them
reintroduces the bug.

1. **Live-oil viscosity is 70 cP at 46 °C** (`RESERVOIR.reservoir_oil_viscosity_cP`).
   The spec's "12,000 cP" is *dead* oil. At the corrected pump intake pressure
   (~10.6 MPa) this is the value that gives the classic CSS profile: a
   pump-limited plateau early (full barrel, fillage 1.79) collapsing to fillage
   ≈ 0.2 late in the cycle — which is what makes fluid pound, and therefore
   the whole control story, real.

2. **Producing BHP is 0.85 × reservoir pressure**, not a small fraction of it.
   A 1000 m column of 900 kg/m³ crude weighs 8.8 MPa, so a 4 MPa "producing
   BHP" is physically impossible. This was found by the EKF refusing to
   reconcile with the instruments.

3. **The CSS cut-off *terminates* production**, it is not merely a flag
   (`thermal_reservoir.py`, `if state.cutoff_reached: _record(); break`).
   Without it the schedule optimiser degenerates — running the full production
   period always wins, so it always picks minimum steam.

4. **Thermal time constants scale with the injected chest radius**
   (`deep_thermal_time_constant_days` uses `_chest_radius_at_injection_end_m`).
   Peak BHT is the steam temperature regardless of injection duration, so if the
   decay does not scale with thermal mass, less steam looks equally good and the
   injection/soak trade-off is fake.

5. **`vertical_contact_factor = 0.08`** in the over/underburden path. A thin
   laterally-extended chest has ~23× more face area than perimeter; without it
   vertical losses take ~90% of the injected energy.

6. **Ramey heat loss** uses the exact transient form
   (`U(t) = min(4πr_t k_e/√(παt), 2πk_e/ln(r_e/r_t))`) integrated by trapezoid
   for `f_ch(z) = exp(−2π∫U dτ/(ρcA))`. A fitted Willhite-style intercept gives
   a nonsensical B' = −3.8.

7. **The Gibbs solver has two routines, and the split matters:**
   - `transport()` — the explicit finite-difference solution. Verified
     directly for causality (nothing at the pump before `L/a`), losslessness
     (amplitude exact to 0.1%), and convergence of arrival time to `L/a`
     (0.51% at 160 segments). This is the spec's phase-shift requirement.
   - `solve()` — the transmission inversion that produces the card. Motion
     **and** load are both delayed by `L/a` and scaled by `exp(−cL/a)`, so the
     card *shape* (parallelogram / fluid-pound collapse / tagging spike) is
     carried through unchanged. Re-phased onto bottom dead centre.
   - The plunger load is set by the transmitted measurement, and `fillage` is a
     *modification* (`_collapse`) that is unity at a full barrel. Multiplying
     the load by the position fraction — an earlier mistake — destroys every
     fault except fluid pound.

8. **Emulsion viscosity is Grunberg–Nissan** (log-mean + k_G x_o x_w), not
   harmonic mixing. Heavy-oil emulsions are non-Newtonian in the blend.

---

## 4. Conflict log

### 2026-09-28 — training distribution vs. the Gibbs end condition

**Symptom.** All six `test_every_injected_fault_is_diagnosed_and_acted_on`
cases failed; every downhole card was classified `PUMP_TAGGING`.

**Two diagnoses were proposed, and one of them was wrong.**

*Agent A (backend)* traced it upstream: the pump end condition prescribed a
constant plunger force, so the reconstructed card had **no load structure at
all** — it was not a card. Attempts to fix it with a fluid-column characteristic
(`F = W_f φ(u)`) failed for a good reason: the hydrostatic column weight is
almost position-independent (its stiffness is `ρgA/h` ≈ 0.023 N/m), so it cannot
produce a 20–40 kN card span. The real span comes from the **pump differential**,
which is what the load cell measures and what the transmission must carry.

*Agent B (backend, second pass)* diagnosed a **train/inference distribution
mismatch** and changed `build_training_set()` to pass synthetic surface cards
through the Gibbs solver at randomised damping rates `[0.05 … 0.30]`.

**Resolution: Agent A was right about the cause; Agent B's change is still
correct and was kept.** Once `solve()` became a shape-preserving transmission,
routing training cards through it is valid augmentation — it exposes the network
to the transmission gain and the re-phasing, and it costs nothing. The two
changes are complementary, not competing.

*Guard against regression:* `test_closed_loop.py` now trains with
`damping_rates=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]`. If someone reverts the
end condition, the round-trip test
`test_every_injected_fault_is_diagnosed_and_acted_on` fails for **all six**
labels — that is the canary.

**Lesson.** When two agents disagree about a failing test, establish whether
the *model* or the *training data* is wrong before changing either. Here the
data was fine and the model was not; the second agent's independent conclusion
was a real (if secondary) robustness issue, not a competing root cause.

### 2026-09-28 — patch scripts silently corrupting source

Repeated `s.index()`-based slices **duplicated** whole method blocks
(`_conduction_to_formation`, `_overburden_loss_rate`, the GibbsSolver class
body) because the end marker was found *before* the start marker, making
`end - start` negative. The `edit` tool also failed with "Missing key path" on
large blocks.

**Standing rule:** never patch by index arithmetic without asserting
`end > start` first, and after any scripted patch run
`grep -c "def <name>"` to confirm no duplicate definition was introduced.
The escaped-LaTeX docstrings also break under heredoc patching (escapes get
halved) — prefer raw docstrings (`r"""`) and verify with
`./venv/bin/python -W error::SyntaxWarning -c "import ..."`.

---

## 5. Current state (verified 2026-09-28 20:40)

```
backend/tests/    237 passed          (rheology, thermal_reservoir, hydraulics,
                                      gibbs_solver, ai_models, closed_loop)
tests/            42 passed, 1 skip   (legacy Streamlit prototype — still green)
frontend          tsc --noEmit clean · vite build 192 kB / 59.5 kB gzip
```

- Classifier: 100% validation accuracy, 100% per-class recall on all six
  states; 100% agreement with the independent geometric cross-check.
- Steady-state twin pass: **~9–11 ms**, well inside the 100 ms budget
  (the first call includes one-off classifier training).
- All six injected faults round-trip through the Gibbs inversion and are
  diagnosed at ≥ 0.96 confidence with the correct control action.
- CSS cut-off verified firing: SOR 18.98 > 4.2 with BHT < 70 °C.
- **Server verified live**: `uvicorn backend.app.main:app` boots, warm-up loads
  the checkpoint, `/api/v1/{health,config,control/state}` all 200.
- **WebSocket verified live**: `ws://…/ws/telemetry` sends a `hello` frame then
  `telemetry` frames at a measured **0.522 s** cadence, `latency_ms` 9–11 ms,
  label `NORMAL_FULL_BARREL` at 0.999 confidence.
- **Streamer verified live** (closes old item 7): the ingest route exists and
  works. `simulation/synthetic_field_generator.py --url http://… --limit 12`
  generated 4449 scans over 3 cycles and replayed 12/12, ~365 ms/scan.

Run it:

```bash
./run.sh test              # all tests
./run.sh test-fast         # legacy only (~1s)
./run.sh test-backend      # backend only (~45s)
./run.sh backend           # FastAPI dev server
./run.sh frontend          # Vite dev server
./run.sh docker-detach     # full stack in background
./run.sh streamlit         # legacy Streamlit
```

The `run.sh` script handles all modes, auto-installs deps, and manages the venv.

---

## 6. Open items

| # | Item | Owner | Status |
|---|---|---|---|
| 1 | `backend/tests/test_api_stream.py` — WS round trip < 100 ms, `/ws/telemetry` at 500 ms, `/control/apply` accepts SPM/stroke setpoints | backend | **done** |
| 2 | `backend/Dockerfile` + backend requirements | backend | **done** |
| 3 | Root `docker-compose.yml` — wire `backend` + `frontend`; `docker compose up --build` must work | backend | **done** |
| 4 | `README.md` — documents the real system (not just Streamlit prototype) | backend | **done** |
| 5 | `pytest.ini` — covers `backend/tests` **and** keeps `tests/` | backend | **done** |
| 6 | Human visual check of the dashboard at 1024px against a live backend | frontend | **open** |
| 7 | Streamer → backend end to end | backend | **done** |

### Item 7 — closed

The earlier gap was real: `/ws/telemetry` is server-push and cannot accept
samples. The backend now exposes `POST /api/v1/telemetry/ingest`
(`backend/app/api/routes.py`), which replays a batch of SCADA scans through the
twin, advances the CSS reservoir state as the replay crosses into new days, and
returns the first and last frames in full with a per-scan summary for the rest.
The streamer targets that route (it rewrites a `ws://` base to `http://` and
appends the path, so both forms work). Verified live; see §5.

---

## 7. Session 2026-09-28 20:40 — closing the project

Three agents, one file each, run concurrently. They must not touch each other's
files, and none of them may edit this file — append findings here instead.

| Agent | Files it owns | Deliverable |
|---|---|---|
| **A — tests** | `backend/tests/test_api_stream.py` (new) | API + WebSocket contract suite: hello/telemetry frame shape, 500 ms cadence, < 100 ms pass, `POST /control/apply` accepts and rejects setpoints, `as_frame` round-trips through JSON |
| **B — packaging** | `Dockerfile.backend`, `docker-compose.yml(.bak)`, `README.md(.bak)`, `pytest.ini` | Container serves `backend.app.main:app`; compose brings up backend + frontend and `docker compose config` validates; README documents the real system; `pytest` from the root runs both suites |
| **C — dashboard** | `frontend/**` only | Visual + functional check of the dashboard against a live backend at 1280 and 1024 px; fix whatever is visibly broken |

**Rule for all three:** run the full test suite before declaring done. If a
change breaks `tests/` or `backend/tests/`, fix it — do not relax an assertion
to make it pass.

### Standing rules (learned the hard way — §4)

- Never patch source with `s.index()`-based slices without asserting
  `end > start`; afterwards `grep -c "def <name>"` to prove no duplicate.
- Raw docstrings (`r"""`) — escaped LaTeX gets halved under heredoc patching.
- Do not change physics constants or thresholds in `backend/app/core/config.py`
  or `backend/app/physics/**` to make a test pass. Fix the model or the test's
  expectation, and say so in §4.

---

## 8. Session 2026-09-28 22:00 — two agents editing the same deployment files

### 8.1 The collision

Agent B (§7) owned `Dockerfile.backend` and `docker-compose.yml`. The earlier
session had independently created `backend/Dockerfile` — the path the SIH26120
specification asks for — and repointed compose at it. For about ten minutes both
files claimed to build the twin, and the compose file flip-flopped between them.

It surfaced as a build that "quietly succeeded" with the wrong image: the daemon
produced an 814 MB image with **neither torch nor fastapi**, because the
`COPY backend/requirements.txt` line resolved to the *legacy* service's
requirements (sqlalchemy, aiokafka, influxdb-client, scikit-learn) rather than
the twin's. The build-time route assertion in `Dockerfile.backend` did not fire,
because the legacy `requirements.txt` *does* install fastapi.

**Resolution — one Dockerfile.** This session the two agents converged on
`Dockerfile.backend` (the tracked path): `backend/requirements.txt` was rewritten
as the twin's audited dependency set, which makes its existing
`COPY backend/requirements.txt` line correct, so the whole file works unchanged.
`backend/Dockerfile` — the path the specification asks for — is now a **symlink**
to it, so the spec's layout is satisfied with zero duplication. Verified: `docker
build -f backend/Dockerfile` follows the symlink.

The three genuine improvements below came from the earlier `backend/Dockerfile` and
are all present in the file that won:

1. the **build-time route assertion** (fails the build, not the deploy) — this is
   what would have caught the wrong-requirements build in the first place;
2. **seeding the weight cache** from any checkpoint in the build context, so
   `docker compose up` loads rather than retrains;
3. a Dockerfile-level **`HEALTHCHECK`**, not just a compose one.

`docker-compose.yml` points both the `backend` and `streamer` services at
`Dockerfile.backend`.

### 8.2 `backend/requirements.txt` — a spec deviation, deliberately

The specification asks for the twin's dependencies at
`backend/requirements.txt`. That path is **already occupied** by the repository's
pre-existing backend service (sqlalchemy, asyncpg, aiokafka, paho-mqtt,
influxdb-client, scikit-learn) — real, tracked work that was explicitly to be
preserved. Overwriting it would break that service's build.

**Resolved in the other direction.** Agent B then rewrote
`backend/requirements.txt` as the twin's dependency set, audited against the
actual imports of `backend/app/**` and `simulation/**` — which are only `numpy`,
`torch`, `fastapi`, `pydantic`, `httpx` and `uvicorn` (`uvicorn[standard]`
supplies `websockets` transitively; the streamer posts over HTTP, so it needs no
direct `websockets` import). That satisfies the spec's path *and* is the correct
content, so the separate `requirements-twin.txt` was deleted.

The legacy "enterprise" shim (`backend/db/`, `backend/api/routes/`) is not served
by this image; its dependencies are preserved in **`backend/requirements-legacy.txt`**.
The Streamlit prototype's are in the root `requirements.txt`.

Worth keeping: the per-package justification in that file is exactly the kind of
comment that stops the next person re-adding 2 GB of CUDA runtime.

### 8.3 `docker compose up` needs the build context pruned

Without a `.dockerignore` the context shipped `venv/`, `frontend/node_modules/`
and every `__pycache__` — the log showed it at **158 MB and still growing** before
the first instruction ran. `.dockerignore` cuts it to ~4 MB.

Note it also excludes `Dockerfile*`, which the **classic** builder (the fallback
when `buildx` is absent) tolerates but is worth knowing about.

### 8.4 A real latency bug, found by a flaky test

`TestStreamTiming::test_latency_budget_is_met_on_every_frame` failed
intermittently — 2 of 5 runs. It was **not** contention: the assertion was on the
*fastest* steady frame, and the slowest steady state was 100–175 ms against an
11 ms median. Two separate causes, both real:

1. **PyTorch thread dispatch dominates these models.** The residual CNN sees a
   128-sample input; a single forward is **0.88 ms on one thread and 22.8 ms on
   six**. The penalty also swings with machine load, which is exactly the
   "sporadic failure on an idle service" signature. Fixed by pinning
   `ApiConfig.torch_num_threads = 1` (env `TORCH_NUM_THREADS`), applied in
   `create_app()` and the warm-up. Training gets ~2x slower, which is irrelevant
   because training happens once, off the serving path.
   *Side effect: the whole suite went from 26 min to 2 min.*

2. **The reservoir was re-solved on every frame.** `advance_reservoir()` called
   `simulate_cycle()` — a full Marx-Langenheim run — per 500 ms frame. The cycle
   is a deterministic function of the configuration, so it is now solved **once**
   and cached (`_production_trajectory`), and `reservoir_state_at(day)`
   *interpolates* it. Frame cost 179 ms → 10 ms. The trajectory is built in
   `_load_thermal_state`, not lazily, so the cost lands in start-up rather than
   in the first frame.

A third, smaller one: the first frame cost 10–21 s in the container because
PyTorch initialises its oneDNN kernels on the first convolution. `warm_up` now
runs a full priming pass (`_prime_pipeline`) so the dashboard is never blank.

### 8.5 The stream now plays the decline instead of freezing

The dashboard was showing one static operating point. It now **advances the
reservoir clock each frame** (`day_step = 0.05` days, so one field day every 10 s
at the 500 ms cadence) and restarts the cycle when production ends. Opened at
**production day 6** — a full barrel — so the dashboard *watches* the crossover:

| production day | fillage | diagnosis | action | SPM |
|---|---|---|---|---|
| 6 | 1.17 | `NORMAL_FULL_BARREL` | `HOLD` | 9.00 |
| 18 | 0.83 | `FLUID_POUND` | `REDUCE_SPM` | 2.00 |

That is the specification's headline requirement — *a cooling reservoir triggers
SPM down-regulation* — visible live in the dashboard, not only in a test.

Two rules learned making that coherent:

- `start_day` counts **production** days, not elapsed cycle days. Injection and
  soak (19 days) precede it. Getting this wrong made "day 18" mean cumulative day
  37, which is already fluid-pounding.
- The synthesised card must **follow from the fillage**, or the card, the
  diagnosis and the recommendation contradict each other. But an *explicitly
  injected* fault must not be silently replaced by a coincidentally low fillage —
  `--fault ROD_FLOATING` exists to demonstrate rod float.

### 8.6 Verified end to end, in the container

```
docker compose build backend        # 21 steps, "build-time import check ok: 19 routes"
docker run -p 8001:8000 …           # healthy
  /api/v1/health                    200 ok
  ws://…/ws/telemetry               hello + frames, 0.52 s cadence
  latency_ms                        12–15 ms from the very first frame
  setpoint over the socket          {"type":"ack","accepted":true}
  frame JSON                        no NaN, no Infinity
```

`thermatwin/backend:3.0.0` is 1.75 GB, runs as unprivileged uid 10001, and ships
a seeded `dyno_classifier.pt`.

### 8.7 Current state

```
pytest (root)     339 passed, 1 skipped     (2m06s)   EXIT=0
  backend/tests/  297 passed
  tests/           42 passed, 1 skipped    (legacy — still green)
```

The `xfail` is **gone**: the `_ensure_classifier` defect below was fixed and its
marker removed with it.

### 8.8 The poisoned-classifier defect, fixed

`_ensure_classifier` published `self.classifier = DynoCardClassifier(train=False)`
*before* `train_and_save()` returned, then treated any non-`None` classifier as
ready. An interrupted warm-up — exactly what a browser tab closed during the
first frame causes, since the tick runs in a `to_thread` call a disconnect cannot
cancel — left the process-wide twin permanently holding an untrained classifier,
and every later frame was `{"type":"error","message":"classifier has not been
trained"}` until restart. It also made the method non-thread-safe.

Fixed by publishing only on success, checking `.trained`, and loading the cached
checkpoint when one exists (so a container that cannot write its weights volume
does not retrain on every boot). Covered by
`TestStreamEngine::test_a_twin_left_with_an_untrained_classifier_recovers`.

### 8.9 One Dockerfile, one requirements file

Three files had been left claiming the same job. Resolved:

- `backend/Dockerfile` and `backend/requirements-twin.txt` — **deleted**. The
  spec-mandated path turned out to be a *duplicate* build path, and the twin's
  deps are now correctly in `backend/requirements.txt`.
- `Dockerfile.backend` + `backend/requirements.txt` — canonical, and what compose
  references for both `backend` and `streamer`.
- `backend/requirements-legacy.txt` — byte-identical to the pre-existing
  `git show HEAD:backend/requirements.txt`, so the v2 dependency list is not
  lost by the overwrite. Verified with `diff` against HEAD.

The image was checked for the silent-wrong-image failure §8.1 describes:
`torch 2.14.0+cpu`, `fastapi 0.141.1`, `numpy 2.5.3`, `pydantic 2.13.5`,
`uvicorn 0.54.0`, `httpx 0.28.1` present; `sqlalchemy`, `aiokafka`,
`influxdb_client`, `paho` correctly absent; 13 OpenAPI paths mounted.

### 8.10 Verified on the declared ports

Earlier verification used a remapped `18000/18080` override because a stray local
uvicorn held `:8000`. That left the committed ports unproven. Corrected: stray
processes and the one-off `tw-verify` container removed, then a clean
`docker compose up -d --build`.

```
docker compose ps     thermatwin-backend  healthy  0.0.0.0:8000->8000
                      thermatwin-frontend healthy  0.0.0.0:8080->80
GET  :8080/                     200, index + hashed assets, gzip on
GET  :8080/api/v1/health        200 ok   (through the nginx proxy)
GET  :8000/api/v1/health        200 ok   (direct)
POST :8080/api/v1/control/apply 6.0 spm  -> accepted
POST :8080/api/v1/control/apply 99 spm  -> rejected, "outside the permitted range 2.0-14.0"
ws://:8080/ws/telemetry         hello, then telemetry every 0.518-0.520 s
                                latency_ms 10.2-11.6 from the first frame
                                production day advances 6.05 -> 6.35
                                frame JSON free of NaN and Infinity
```

No host processes are left running; the only uvicorn is inside the container.

### 8.11 The card is a line, not a parallelogram — one half fixed

The dashboard's dyno card drew two nearly-straight lines rather than the
familiar parallelogram. Confirmed numerically: the upstroke and downstroke loads
differ by **0.888 N at the same position** against a 15 000 N span. The
`NORMAL_FULL_BARREL` branch of `synthesize_card` adds `fluid_load_n *
normalised`, which is a pure function of position, so both strokes trace the
same curve. The comment above it claimed "the familiar parallelogram"; the code
could not produce one.

**Fixed — the feature extractor, not the generator.** The deeper problem was
that a *correct* parallelogram is misread as fluid pound. Measured over 40
seeds: at a stroke-to-stroke offset of only 0.10 of the fluid load, the
`upstroke_reversal` feature reached 0.102 against a 0.10 threshold, so **every
healthy card became `FLUID_POUND`**. The cause is that `upstroke_reversal`
took the smoothed derivative *across* the top-dead-centre turnaround, where a
real card steps because the standing valve makes the two strokes carry
different column weights. That step is geometry, not a fault.

`upstroke_reversal` now excludes a window either side of both turnarounds and
measures only the interior of the upstroke, where a genuine supply collapse
actually makes the load fall while the plunger is still rising. After the fix
all six labels diagnose 25/25 from `synthesize_card`, and `test_ai_models.py`,
`test_gibbs_solver.py` and `test_closed_loop.py` are 110 passed.

**Not fixed — the generator still draws a line, and the honest reason.** A
symmetric parallelogram is now classified `ROD_FLOATING` 40/40, because
`downstroke_asymmetry` (mean downstroke load − mean upstroke load) cannot
distinguish a normal card's stroke-to-stroke width from a genuine downstroke
depression. Making the card a true parallelogram therefore requires a
discriminator that separates *width* from *depression*, and re-tuning
`analytic_diagnosis` and retraining the network against it. That is a real
piece of work with a real risk to the 100 % accuracy claim, so it is recorded
here rather than half-done at the end of a session.

**Consequence:** the dashboard's normal card renders as a line. It is
geometrically wrong and an operator would notice. The diagnosis it produces is
correct. Nothing in the control loop depends on the card being a parallelogram —
the Gibbs inversion is a transmission, so a monotone card transmits faithfully.

### 8.12 Open

| # | Item | Owner |
|---|---|---|
| 1 | **Make `synthesize_card` produce a true parallelogram**, and add a discriminator separating stroke width from rod-float depression. Blocked on the `downstroke_asymmetry` collision described in §8.11. Needs `analytic_diagnosis` re-tuning and a retrain. | physics/AI |
| 2 | **Human visual check of the dashboard at 1024 px.** Done by the frontend agent with Playwright + Chromium (the `browser.*` tools need a desktop browser this environment does not attach). Zero console errors, zero failed requests, no `NaN`/`undefined`/`[object Object]`, dyno card 92 % box fill at 1280 and 79 % at 1024, VFD accept and reject both rendering in-panel. Screenshots in `/tmp/opencode/shots/`. It also found and fixed three real frontend bugs: a CORS failure on every REST call, a WebSocket proxy target that 403'd, and a 1024 px layout that stacked the chart row. | done |
| 3 | Decide whether the legacy shim under `backend/db/` + `backend/api/routes/` is still wanted, or should be removed now that it is outside this image (§8.2) | owner |
| 4 | `backend/tests/test_api_stream.py` and `talk.md` were both appended to by two agents concurrently. Worth a read-through for interleaved edits before the next hand-off. | owner |
| 5 | `POST /control/apply` returns HTTP 200 for a rejected setpoint, distinguishable only by the `accepted: false` body field. A 4xx would be more correct. The frontend handles it correctly either way. | API |

---

## 9. Session 2026-09-28 22:40 — `run.sh` rewritten

The script was still the legacy one, and starting the wrong thing:

| Was | Problem |
|---|---|
| `uvicorn backend.main:app` | the **legacy** shim, not the twin. `backend/main.py` and `backend/app/main.py` both define `app`, so this started the wrong application without any visible error. |
| `docker-compose` (hyphenated) | the v1 binary; on a modern host with only the plugin it fails outright. |
| help text: frontend on `:5173` under docker | the container serves nginx on **:8080**. |
| `pip install -r requirements.txt` | installed the Streamlit set and pulled the ~2 GB CUDA torch wheel, because no CPU index was given. |
| `typecheck` on `backend/ai`, `backend/physics` | those paths do not exist; they are under `backend/app/`. |
| no preflight | a port clash surfaced as uvicorn's `Address already in use`, which reads like a bug rather than "something else already has 8000". |

Rewritten with one command per service, plus `up` (API + dashboard together),
`doctor`, `status`, and `replay` (the streamer, which had no command at all).
`legacy-streamlit` and `legacy-backend` are still there, explicitly labelled as
preserved.

### Three bugs found by actually starting the services

1. **The preflight missed a busy port.** It asked `ss -p` for the holder's name
   and treated "no name" as "nothing there" — but `ss` omits the `users:(...)`
   field for a process owned by *another user*, so a port could be plainly busy
   while the name lookup returned empty. Occupancy and identification are now
   separate tests: `port_busy` for the decision, `port_owner` only for the
   message, and a busy port the user cannot attribute says so.
2. **The suggestion was useless**: `API_PORT=8000 ./run.sh backend` on a busy
   8000. It now scans for a port that is actually free and suggests that.
3. **The dashboard's WebSocket 403'd behind the dev server.** `run.sh` was
   setting `VITE_WS_URL` to `ws://host:8010/ws/telemetry`, and vite uses that
   value as the *proxy target*, appending the incoming path to it:
   `/ws/telemetry` became `/ws/telemetry/ws/telemetry` → 403. This is a trap
   `frontend/vite.config.ts` already documents; `run.sh` walked straight into it.
   Fixed by setting only `VITE_API_BASE` and leaving the client site-relative,
   which is the arrangement the frontend is built for.

### One change outside `run.sh`

`docker-compose.yml` now parameterises its published ports
(`${API_PORT:-8000}`, `${DASHBOARD_PORT:-8080}`) and `run.sh` exports them.
Without that the preflight would check a port the containers then ignore — it
would report a port free, then `docker compose up` would fail to bind it.

### Verified, service by service

```
./run.sh doctor                        # env check; flagged the live 8000 and 5173
./run.sh backend                       # blocked with a free-port suggestion
API_PORT=8010 WEB_PORT=5180 ./run.sh up
  API health on :8010                  ok
  dashboard :5180 serves the bundle    ok
  /api through the vite proxy          ok
  /ws  through the vite proxy          ok   (was 403) — 9.5 ms/frame
./run.sh replay  FAULT=ROD_FLOATING    80/80 scans diagnosed ROD_FLOATING
API_PORT=8010 DASHBOARD_PORT=8081 ./run.sh docker-detach
  backend + frontend healthy
  dashboard :8081, /api proxied by nginx ok
  /ws  through nginx                    ok — 10.6 ms/frame
./run.sh status | logs backend | stop   ok
./run.sh frontend-build                192 kB js / 16.7 kB css
./run.sh bogus                         help + non-zero exit
```

### Note for whoever runs next

Another process is holding `:8000` and has the container stack up on the default
ports. `./run.sh doctor` will say so, and any `API_PORT=…` variant is unaffected.
