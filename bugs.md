# ThermaTwin — Known Bugs & Code Issues

*Generated 2026-09-28 from code review. Not fixed — just documented.*

---

## 1. Thread Safety — Critical

| File | Location | Issue |
|------|----------|-------|
| `backend/app/engine/digital_twin.py` | `_ensure_classifier()` lines 218-239 | **Race condition documented in comments (224-225)**. Classifier assigned to `self.classifier` *before* training completes. If interrupted (client disconnects mid-first-frame), twin left with untrained classifier. XFAIL test `test_a_twin_left_with_an_untrained_classifier_recovers` confirms. |
| `backend/app/engine/digital_twin.py` | `reconstruction_fillage` line 394 / `step()` lines 514-516 | Class attribute mutated per-request. Concurrent requests clobber each other's fillage value. |
| `backend/app/engine/digital_twin.py` | `advance_reservoir()` lines 329-348 | Uses `self.reservoir_day` property (lines 611-617). Not thread-safe; concurrent WebSocket + REST calls race. |
| `backend/app/api/routes.py` | Global singletons lines 48-50 | `_TWIN`, `_OPTIMIZER`, `_CLASSIFIER` shared across all requests. No locking for `DigitalTwin` state mutations. |

---

## 2. Hardcoded Magic Numbers (Should Be Configurable)

| File | Line | Value | Suggested Fix |
|------|------|-------|---------------|
| `gibbs_solver.py` | 100 | `SAFETY_FACTOR = 0.5` | Move to `AppConfig` |
| `digital_twin.py` | 244 | `day_step: float = 0.05` | Env var `THERMATWIN_DAY_STEP` |
| `digital_twin.py` | 246 | `max_day: float = CSS.production_days` | Configurable per deployment |
| `digital_twin.py` | 490 | `-0.01 * span` tolerance | `THRESHOLDS.rod_float_zero_tolerance` |
| `digital_twin.py` | 492 | `2.0 * THRESHOLDS.rod_float_risk_high` | Named constant |
| `digital_twin.py` | 318 | `0.02` minimum fillage | `THRESHOLDS.min_card_fillage` |
| `digital_twin.py` | 689 | `history = 21` | Configurable forecast window |
| `websocket.py` | 65 | `noise: float = 0.004` | Env var `THERMATWIN_SYNTH_NOISE` |
| `websocket.py` | 66 | `seed: int = 26120` | Already `THERMATWIN_RANDOM_SEED` but not used here |

---

## 3. Missing Input Validation

| File | Location | Missing Check |
|------|----------|---------------|
| `digital_twin.py` | `step()` lines 519-521 | `surface_position_m` and `surface_load_n` length match before `reconstruct_card` |
| `digital_twin.py` | `reconstruct_card()` lines 409-410 | Only checks `duration_s <= 0.0`; no array length validation |
| `routes.py` | `ingest()` line 350 | Ignores `twin.apply_setpoint()` return value (`accepted: false` silently dropped) |
| `routes.py` | `ingest()` lines 356-360 | `TelemetryPacket` created without validating `min/max` of `load_n` (empty arrays possible) |
| `routes.py` | `ingest()` line 367 | `duration_s=60.0 / max(sample.spm, 1e-6)` — no upper bound on SPM |
| `websocket.py` | `synthesise_surface_card()` line 109 | No validation `fluid_load_n > 0` |

---

## 4. Error Handling Gaps

| File | Location | Issue |
|------|----------|-------|
| `routes.py` | `ingest()` lines 346-348 | `except ValueError: pass` swallows errors when advancing reservoir day |
| `routes.py` | `ingest()` lines 369-370 | Only catches `ValueError`; other exceptions → 500 |
| `digital_twin.py` | `_ensure_classifier()` line 237 | `classifier.train_and_save()` exceptions not caught → half-trained state |
| `websocket.py` | `synthesise_surface_card()` | No validation of inputs before synthesis |

---

## 5. Type Hint Issues

| File | Location | Issue |
|------|----------|-------|
| `digital_twin.py` | `__init__` lines 183-193 | `Optional[AppConfig] = None` but `config or get_config()` — no validation if passed config invalid |
| `digital_twin.py` | Line 394 | `reconstruction_fillage: float = 1.0` class attr but used as instance state |
| `routes.py` | `ingest()` line 301 | Returns `Dict[str, object]` — too loose; should use `TypedDict` |
| `digital_twin.py` | `forecast()` line 671 | Returns `Optional[dict]` — should be `Optional[ForecastDict]` |
| `routes.py` | Multiple endpoints | Return `Dict[str, object]` instead of Pydantic models |

---

## 6. Performance Issues

| File | Location | Issue |
|------|----------|-------|
| `digital_twin.py` | `_production_trajectory()` lines 258-271 | Re-simulates entire CSS cycle on first call; called multiple times during warm-up |
| `routes.py` | `ingest()` lines 336-348 | Calls `css_model.simulate_cycle()` *per sample* when day changes — O(n) expensive for batch replay |
| `digital_twin.py` | `traverse()` lines 430-453 | New `compute_traverse` call every time; no caching for same parameters |

---

## 7. Logic Bugs / Inconsistencies

| File | Location | Bug |
|------|----------|-----|
| `digital_twin.py` | `step()` lines 539-543 | `inflow = min(estimated_inflow, self.card_fillage(card) * capacity)` — mixes `card_fillage` from *reconstructed* card with `capacity` from *applied* SPM (measurement vs command) |
| `digital_twin.py` | `step()` lines 514-516 | `reconstruction_fillage = 1.0 if fillage is None else fillage` — overrides to 1.0 even when caller provides `fillage=0.0` (edge case) |
| `digital_twin.py` | `reservoir_state_at()` lines 284-293 | Uses `CSS.injection_days + CSS.soak_days` as offset but `time_days` in trajectory is already cumulative from cycle start — double counting |
| `routes.py` | `ingest()` lines 336-337 | `simulate_cycle(injection_days=twin.start_day, production_days=...)` — `start_day` is **production** days, not injection days |
| `digital_twin.py` | `forecast()` lines 688-689 | Uses `casing_pressure_mpa=casing` with hardcoded `0.85 * initial_pressure` — not current reservoir pressure |

---

## 8. Documentation vs Implementation Mismatches

| Documented (talk.md / README) | Actual Code | Discrepancy |
|-------------------------------|-------------|-------------|
| `well.days_into_phase` (talk.md §2) | `digital_twin.py:129` | Set equal to `day`, not phase-specific |
| `diagnosis.features` keys (talk.md §2) | `card_features()` | Returns extra key `coherence` not in `FEATURE_KEYS` |
| `oil_bbl` conversion | `digital_twin.py:152` | Uses `6.2898` (m³→bbl) without documentation |
| `diagnosis.agrees_with_analytic` | `CardPrediction.as_dict()` | Documented but not in `DIAGNOSIS_KEYS` test contract |

---

## 9. Syntax / Style Issues (SyntaxWarning)

| File | Issue |
|------|-------|
| `hydraulics.py` | LaTeX escapes `\q`, `\p` in docstrings trigger `SyntaxWarning` in Python 3.12+ |
| `gibbs_solver.py` | Same — `\q`, `\p`, `\quad`, `\qquad` in docstrings |
| `thermal_reservoir.py` | Same — `\q`, `\p`, `\frac`, `\quad` in docstrings |
| Fix: Use raw docstrings `r\"\"\"...\"\"\"` or double backslashes |

---

## 10. Testing Gaps

| Missing Test | Why It Matters |
|--------------|----------------|
| Concurrent `DigitalTwin` access | Thread safety (#1) untested |
| `ingest()` with `apply_setpoints=True` | Autonomous replay path untested |
| `forecast()` with surrogate | Surrogate integration untested |
| `test_steady_state_latency_is_within_budget` | Uses 200ms median budget vs 100ms spec |
| `advance_reservoir()` with concurrent calls | Race condition untested |
| `reconstruction_fillage` override edge case | `fillage=0.0` ignored |

---

## 11. Architecture Issues

| Issue | Impact |
|-------|--------|
| Global `_TWIN` singleton in `routes.py` | All API requests share same twin state — cannot support multi-well |
| `advance_reservoir()` fixed `day_step` vs configurable stream interval | Clock drift possible if `THERMATWIN_STREAM_INTERVAL_S` changed |
| `ingest()` endpoint does too much | Reservoir simulation, setpoints, frame generation, statistics all in one handler |
| No authentication/authorization | Any client can control VFD, inject telemetry |
| No rate limiting | DoS vulnerable |
| No graceful WebSocket shutdown | Clients can hang on server restart |
| No circuit breaker for external deps | Not applicable yet but planned |

---

## 12. Deprecation Warnings

| Source | Warning |
|--------|---------|
| `pytest.ini` | `asyncio_mode = auto` deprecated |
| `pytest_asyncio` | `asyncio.get_event_loop_policy` deprecated (multiple test files) |
| Fix: Update pytest-asyncio, use explicit `asyncio_mode = auto` |

---

## 13. Legacy Code Confusion (Not Bugs But Confusing)

| Path | Description |
|------|-------------|
| `backend/main.py` | v2 enterprise app (Postgres/Kafka) — **served by root `Dockerfile.backend` incorrectly** |
| `backend/api/`, `backend/db/`, `backend/services/`, `backend/ingestion/` | v2 layer — not served but in tree |
| `requirements.txt` (root) | Includes `streamlit`, `paho-mqtt`, `influxdb-client` — not needed for v3 |
| `run.sh` | Still references `backend.main:app` in some modes |

---

## Priority Ranking (Fix Order)

1. **Thread safety** (#1) — data corruption, crashes under load
2. **Logic bugs in reservoir/SPM** (#7) — wrong control decisions
3. **Hardcoded magic numbers** (#2) — unconfigurable behavior
4. **Input validation** (#3) — 422 errors instead of 500s
5. **Error handling** (#4) — silent failures
6. **Performance** (#6) — batch replay unusably slow
7. **Architecture** (#11) — blocks multi-well, auth
8. **Type hints / docs** (#5, #8) — maintainability
9. **Syntax warnings** (#9) — future Python errors
10. **Legacy cleanup** (#13) — confusion