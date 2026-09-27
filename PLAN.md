# ThermaTwin SRP Prototype — Implementation Plan

## Objective

Build **thermatwin-srp-prototype**, a Streamlit-based rapid prototype digital twin for a Sucker-Rod Pump (SRP) system in the Baghewala field. The application simulates reservoir thermal decay and its cascading effects — exponential heavy crude viscosity increase, distorted bottom-hole dynamometer cards (rod float), and adaptive VFD/SPM optimization — surfacing them on an interactive dashboard.

## Current State

The workspace `/home/gux/thermatwin` is **completely empty**. There are no existing files, dependencies, conventions, or tests. This is a greenfield build.

## Proposed Architecture

The system follows a **dataflow pipeline** pattern, where synthetic field data feeds through physics simulation, diagnostic classification, and optimization logic before reaching the visualization layer.

```
[core/data_gen.py]  ──(DataFrame)──►  [core/physics_engine.py]  ──(downhole card array)──►  [core/diagnostics.py]  ──(anomaly flags)──►
       │                                        │                                      │
       │                                        │                                      │
       ▼                                        ▼                                      ▼
  [core/optimizer.py] ◄──(viscosity)── [app.py]  ──►  [components/charts.py]  ──►  Streamlit UI
                                          │
                                          └──►  [components/metrics.py]
```

- **`core/`** contains pure computation modules with no Streamlit dependency (testable in isolation).
- **`components/`** holds Plotly figure builders and metric card renderers (thin wrappers around Streamlit primitives).
- **`app.py`** is the orchestrator — it reads slider state, calls core functions, and passes results to component renderers.

This separation keeps physics logic reusable and testable outside the web framework.

## Implementation Steps

### 1. `requirements.txt` — Pin dependencies

- **What:** Create the dependency manifest with the exact versions specified.
- **Details:**
  ```
  streamlit>=1.32.0
  pandas>=2.0.0
  numpy>=1.24.0
  scipy>=1.10.0
  plotly>=5.14.0
  scikit-learn>=1.2.0
  ```
- **Dependencies:** None.

### 2. `core/__init__.py` — Package initializer

- **What:** Empty file to make `core` a Python package.
- **Dependencies:** None.

### 3. `core/data_gen.py` — Synthetic data generator

- **What:** Contains `generate_css_cycle(days=30, initial_temp=200, final_temp=50, initial_viscosity=500, final_viscosity=50000)` returning a `pd.DataFrame`.
- **Key logic:**
  - **Temperature:** Exponential decay curve from 200 °C → 50 °C over 30 days using `T(t) = T_final + (T_initial - T_final) * exp(-k * t)` with `k ≈ 0.08`.
  - **Viscosity:** Exponential spike from ~500 cP → ~50 000 cP using an Arrhenius-like relation: `μ(t) = μ_initial * exp(alpha * (1/T(t) - 1/T_initial))`.
  - **Surface load:** Baseline polynomial curve representing polished rod load over a stroke cycle (upstroke peak ~80 000 N, downstroke trough ~20 000 N).
  - Output columns: `day`, `temperature_c`, `viscosity_cp`, `surface_load_n`, `surface_pos_m`.
- **Dependencies:** Step 1.

### 4. `core/physics_engine.py` — Downhole card calculator

- **What:** Contains `calculate_downhole_card(surface_load, surface_pos, viscosity, rod_length=1500, wave_speed=5000, damping_coeff=None)` returning a NumPy array of shape `(N, 2)` (load, position).
- **Key logic:**
  - If `viscosity < 5000`: downhole card is a slightly phase-shifted, damped version of the surface card (normal operation).
  - If `viscosity >= 5000`: introduce a **delayed weight transfer** on the downstroke — the downhole card stretches horizontally (fluid drag slows load transfer), producing the classic "rod float" distortion.
  - Model the wave equation with a damping term: `∂²u/∂t² = c² * ∂²u/∂x² - damping * ∂u/∂t`, where `damping` scales with `viscosity / 5000`.
  - Use `scipy.signal` for filtering the propagated wave.
- **Dependencies:** Step 1.

### 5. `core/diagnostics.py` — Anomaly classifier

- **What:** Contains `detect_rod_float(downhole_card, surface_card, viscosity_threshold=5000)` returning a `dict` of diagnostic flags.
- **Key logic:**
  - Compute the **downstroke load transfer delay** — the horizontal distance between the surface card's minimum-load point and the downhole card's minimum-load point.
  - If delay > threshold (e.g., 5% of stroke period) and `viscosity > 5000 cP`, flag `rod_float_detected = True`.
  - If the downhole card shows a sharp inflection on the downstroke (second derivative spike), flag `mechanical_impact_loading = True`.
  - Return `{"rod_float": bool, "impact_loading": bool, "delay_pct": float, "severity": "low|medium|high"}`.
- **Dependencies:** Step 4 (consumes its output).

### 6. `core/optimizer.py` — VFD/SPM recommendation

- **What:** Contains `calculate_safe_spm(viscosity, current_spm=6.0, max_spm=12.0, rod_grade="D", rod_diameter_mm=22)` returning a `dict`.
- **Key logic:**
  - Maximum safe SPM scales inversely with fluid viscosity: `safe_spm = max_spm * (viscosity_ref / viscosity)^0.5`, clamped to `[1.5, max_spm]`.
  - Convert SPM to VFD frequency: `freq_hz = spm / 60 * motor_pole_pairs` (assume 4-pole motor → 2 pole pairs → `freq = spm / 30`).
  - Include a safety margin: `recommended_freq = 0.9 * safe_freq`.
  - Return `{"max_safe_spm": float, "recommended_vfd_hz": float, "production_bpd_estimate": float}`.
- **Dependencies:** Step 1.

### 7. `components/__init__.py` — Package initializer

- **What:** Empty file.
- **Dependencies:** None.

### 8. `components/charts.py` — Plotly figure builders

- **What:** Contains `plot_dyno_overlay(surface_card, downhole_card, title="Dynamometer Cards")` and `plot_thermal_decay(df, highlight_day=None)` returning `plotly.graph_objects.Figure` objects.
- **Key logic:**
  - `plot_dyno_overlay`: Two traces (surface card as dashed line, downhole card as solid) on a load (N) vs. position (m) axes. Color the downhole trace red if rod float is detected.
  - `plot_thermal_decay`: Dual-axis figure — temperature (line, left y-axis) and viscosity (log-scale line, right y-axis) vs. day. Add a vertical marker at `highlight_day`.
  - Keep all Plotly imports and layout code here — `app.py` should only receive `Figure` objects.
- **Dependencies:** Steps 3–6.

### 9. `components/metrics.py` — Metric card renderers

- **What:** Contains `render_metric(label, value, unit, delta=None, alert=None)` and `render_warning_banner(diagnostics_dict)` functions.
- **Key logic:**
  - `render_metric`: Uses `st.columns` + `st.markdown` with custom CSS to render a KPI card.
  - `render_warning_banner`: If `diagnostics["severity"]` is "high", render a red `st.warning` banner with details.
- **Dependencies:** Step 5.

### 10. `app.py` — Streamlit dashboard (entry point)

- **What:** The main orchestrator that wires everything together.
- **Layout:**
  - **Sidebar:** Slider for `day` (0–30), with display of current temperature/viscosity at that day.
  - **Main area:**
    - Row 1: Three metric cards (SPM, VFD frequency, severity).
    - Row 2: Thermal decay chart (left) + Dyno card overlay (right).
    - Row 3: Diagnostics summary table + warning banner.
- **Key logic:**
  1. Call `generate_css_cycle()` once, cache with `@st.cache_data`.
  2. Read slider → slice DataFrame for that day.
  3. Pass row to `calculate_downhole_card()` → diagnostics → optimizer.
  4. Pass outputs to component functions → render with `st.plotly_chart()`.
- **Dependencies:** Steps 1–9.

### 11. `data/synthetic_well_data.csv` — Generated dataset

- **What:** Static CSV output generated by running `data_gen.py` once, so `app.py` can optionally load from disk for faster startup.
- **Dependencies:** Step 3.

### 12. `README.md` — Run instructions

- **What:** Project overview, folder structure diagram, and quickstart:
  ```bash
  python -m venv venv && source venv/bin/activate
  pip install -r requirements.txt
  streamlit run app.py
  ```
- **Dependencies:** None.

## Testing

- **Unit tests** (pytest):
  - `test_data_gen.py`: Verify DataFrame shape (30 rows), temperature range (50–200 °C), viscosity monotonic increase, no NaN values.
  - `test_physics_engine.py`: Verify output shape, verify that high-viscosity cards have larger phase delay than low-viscosity cards.
  - `test_diagnostics.py`: Verify `rod_float_detected=True` when viscosity > 5000 and delay exceeds threshold.
  - `test_optimizer.py`: Verify `safe_spm` decreases as viscosity increases; verify frequency conversion correctness.
- **Integration test:**
  - `test_app_pipeline.py`: Mock slider state → run full pipeline → verify no exceptions and expected dict keys present.
- **Edge cases:**
  - Day 0 (minimum temperature, minimum viscosity) — should show normal card.
  - Day 30 (maximum viscosity) — should trigger rod float and high-severity warning.
  - Viscosity exactly at 5000 cP threshold — verify boundary condition.
- **Manual smoke test:**
  - Run `streamlit run app.py`, move slider to day 25, confirm red banner appears and dyno card distorts.
- **Existing tests:** None (greenfield). The plan includes creating the test suite from scratch.

## Risks

| Risk | Mitigation |
|------|------------|
| **Streamlit caching issues** — slider state not refreshing correctly | Use `@st.cache_data` on the DataFrame generation only, not on derived computations. |
| **Plotly version incompatibility** — `st.plotly_chart` API changes | Pin `plotly>=5.14.0` and test against the installed version. |
| **Physics model oversimplification** — wave equation may not produce realistic card shapes | Start with a lumped-parameter approximation; iterate visually against real dyno card references. |
| **Viscosity threshold (5000 cP) is arbitrary** | Make it a configurable parameter in `diagnostics.py` so it can be tuned without code changes. |
| **Performance** — regenerating physics on every slider move | Cache the full 30-day computation once; slider just slices the result. |
| **scikit-learn unused** — listed in requirements but not used in current design | Either use it for anomaly classification (e.g., isolation forest on card features) or remove from requirements. **Ambiguity to resolve.** |

## Files To Change

```
thermatwin-srp-prototype/
├── requirements.txt
├── README.md
├── app.py
├── core/
│   ├── __init__.py
│   ├── data_gen.py
│   ├── physics_engine.py
│   ├── diagnostics.py
│   └── optimizer.py
├── components/
│   ├── __init__.py
│   ├── charts.py
│   └── metrics.py
├── data/
│   └── synthetic_well_data.csv
└── tests/
    ├── test_data_gen.py
    ├── test_physics_engine.py
    ├── test_diagnostics.py
    ├── test_optimizer.py
    └── test_app_pipeline.py
```

## Execution Order

1. `requirements.txt` — set up dependencies first.
2. `core/__init__.py` + `core/data_gen.py` — data foundation.
3. `core/physics_engine.py` — physics layer.
4. `core/diagnostics.py` — anomaly detection.
5. `core/optimizer.py` — optimization logic.
6. `components/__init__.py` + `components/charts.py` — visualization builders.
7. `components/metrics.py` — UI cards.
8. `app.py` — wire everything together.
9. `data/synthetic_well_data.csv` — generate and persist dataset.
10. `tests/` — write and run test suite.
11. `README.md` — documentation.

---

**Ambiguity to resolve before implementation:**

The `scikit-learn` dependency is listed but not referenced in any of the five logical components. Should it be:
- **(a)** Used for a more sophisticated anomaly classifier (e.g., training an isolation forest on dyno card shape features), or
- **(b)** Removed from `requirements.txt` to keep the prototype lean?
