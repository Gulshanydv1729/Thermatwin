# ThermaTwin SRP Prototype — Per-File Plan & Dependencies

## Dependency Graph (Build Order)

```
requirements.txt
       │
       ▼
core/__init__.py
       │
       ▼
core/data_gen.py ──────────────────────────────────────────────┐
       │                                                        │
       ▼                                                        │
core/physics_engine.py                                          │
       │                                                        │
       ▼                                                        │
core/diagnostics.py                                             │
       │                                                        │
       ▼                                                        │
core/optimizer.py                                               │
       │                                                        │
       ▼                                                        ▼
components/__init__.py                              data/synthetic_well_data.csv
       │
       ▼
components/charts.py
       │
       ▼
components/metrics.py
       │
       ▼
app.py
       │
       ▼
tests/
       │
       ▼
README.md
```

---

## 1. `requirements.txt`

| Attribute | Value |
|-----------|-------|
| **Purpose** | Pin all Python package dependencies |
| **Depends on** | Nothing |
| **Depended on by** | All Python files |

**Contents:**
```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
plotly>=5.14.0
scikit-learn>=1.2.0
pytest>=7.0.0
```

**Notes:**
- `pytest` added for the test suite.
- `scikit-learn` flagged as potentially removable (see Risk section in PLAN.md).

---

## 2. `core/__init__.py`

| Attribute | Value |
|-----------|-------|
| **Purpose** | Make `core/` a Python package |
| **Depends on** | Nothing |
| **Depended on by** | `app.py`, all test files |

**Contents:** Empty file (or a docstring).

---

## 3. `core/data_gen.py`

| Attribute | Value |
|-----------|-------|
| **Purpose** | Generate synthetic 30-day thermal/viscosity dataset |
| **Depends on** | `requirements.txt` (pandas, numpy) |
| **Depended on by** | `app.py`, `data/synthetic_well_data.csv`, `test_data_gen.py` |

**Public API:**
```python
def generate_css_cycle(
    days: int = 30,
    initial_temp: float = 200.0,
    final_temp: float = 50.0,
    initial_viscosity: float = 500.0,
    final_viscosity: float = 50000.0,
    samples_per_day: int = 100
) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
        - day: int (0..29)
        - temperature_c: float (200 → 50, exponential decay)
        - viscosity_cp: float (500 → 50000, exponential spike)
        - surface_load_n: float (polished rod load, N)
        - surface_pos_m: float (polished rod position, m)
    """
```

**Internal logic:**
1. `t = np.linspace(0, days, days * samples_per_day)`
2. Temperature: `T(t) = T_final + (T_initial - T_final) * exp(-0.08 * t)`
3. Viscosity: `mu(t) = mu_initial * exp(alpha * (1/T(t) - 1/T_initial))` where `alpha` calibrated to hit `final_viscosity` at `t=days`
4. Surface load: sinusoidal with upstroke peak ~80 kN, downstroke trough ~20 kN
5. Surface position: triangular wave 0 → 3 m stroke length

**Output shape:** `(days * samples_per_day, 5)`

---

## 4. `core/physics_engine.py`

| Attribute | Value |
|-----------|-------|
| **Purpose** | Convert surface card to bottom-hole dyno card with viscosity-dependent distortion |
| **Depends on** | `requirements.txt` (numpy, scipy) |
| **Depended on by** | `app.py`, `test_physics_engine.py` |

**Public API:**
```python
def calculate_downhole_card(
    surface_load: np.ndarray,
    surface_pos: np.ndarray,
    viscosity: float,
    rod_length: float = 1500.0,
    wave_speed: float = 5000.0,
    damping_coeff: float | None = None
) -> np.ndarray:
    """
    Returns array of shape (N, 2) where column 0 = load (N), column 1 = position (m).
    
    If viscosity < 5000: normal card (slight phase shift, minimal distortion).
    If viscosity >= 5000: rod-float distortion (delayed weight transfer on downstroke).
    """
```

**Internal logic:**
1. Compute damping: `damping = damping_coeff or (viscosity / 5000.0) * 0.1`
2. Apply wave propagation: `downhole_load = surface_load * exp(-damping * rod_length / wave_speed)`
3. Apply phase delay: `delay_samples = int(damping * len(surface_load) * 0.05)`
4. If `viscosity >= 5000`: stretch downstroke horizontally by interpolating with a time-warp factor `1 + (viscosity - 5000) / 50000`
5. Use `scipy.signal.savgol_filter` to smooth the result

**Key threshold:** `VISCOSITY_THRESHOLD = 5000` (cP) — module-level constant.

---

## 5. `core/diagnostics.py`

| Attribute | Value |
|-----------|-------|
| **Purpose** | Detect rod float and mechanical impact loading from dyno card shapes |
| **Depends on** | `requirements.txt` (numpy) |
| **Depended on by** | `app.py`, `components/metrics.py`, `test_diagnostics.py` |

**Public API:**
```python
def detect_rod_float(
    downhole_card: np.ndarray,
    surface_card: np.ndarray,
    viscosity: float,
    viscosity_threshold: float = 5000.0,
    delay_threshold_pct: float = 5.0
) -> dict:
    """
    Returns:
        {
            "rod_float": bool,
            "impact_loading": bool,
            "delay_pct": float,
            "severity": "low" | "medium" | "high"
        }
    """
```

**Internal logic:**
1. Find index of minimum load in surface card → `surface_min_idx`
2. Find index of minimum load in downhole card → `downhole_min_idx`
3. Compute `delay_pct = abs(downhole_min_idx - surface_min_idx) / len(surface_card) * 100`
4. `rod_float = (delay_pct > delay_threshold_pct) and (viscosity > viscosity_threshold)`
5. Compute second derivative of downhole load; if max exceeds threshold → `impact_loading = True`
6. Severity: `"high"` if both flags, `"medium"` if one flag, `"low"` if none

---

## 6. `core/optimizer.py`

| Attribute | Value |
|-----------|-------|
| **Purpose** | Calculate safe SPM and VFD frequency based on fluid viscosity |
| **Depends on** | `requirements.txt` (numpy) |
| **Depended on by** | `app.py`, `test_optimizer.py` |

**Public API:**
```python
def calculate_safe_spm(
    viscosity: float,
    current_spm: float = 6.0,
    max_spm: float = 12.0,
    min_spm: float = 1.5,
    viscosity_ref: float = 1000.0,
    motor_pole_pairs: int = 2,
    safety_factor: float = 0.9
) -> dict:
    """
    Returns:
        {
            "max_safe_spm": float,
            "recommended_vfd_hz": float,
            "production_bpd_estimate": float
        }
    """
```

**Internal logic:**
1. `safe_spm = max_spm * (viscosity_ref / viscosity) ** 0.5`
2. Clamp: `safe_spm = max(min_spm, min(max_spm, safe_spm))`
3. `safe_freq = safe_spm / 60 * motor_pole_pairs * 60` → simplifies to `safe_spm * motor_pole_pairs`
4. `recommended_vfd_hz = safety_factor * safe_freq`
5. `production_bpd = safe_spm * pump_displacement * 1440` (pump_displacement ≈ 0.5 bbl/stroke)

---

## 7. `components/__init__.py`

| Attribute | Value |
|-----------|-------|
| **Purpose** | Make `components/` a Python package |
| **Depends on** | Nothing |
| **Depended on by** | `app.py` |

**Contents:** Empty file.

---

## 8. `components/charts.py`

| Attribute | Value |
|-----------|-------|
| **Purpose** | Build Plotly figures for dyno cards and thermal decay curves |
| **Depends on** | `requirements.txt` (plotly), `core/data_gen.py` (type hints only) |
| **Depended on by** | `app.py` |

**Public API:**
```python
def plot_dyno_overlay(
    surface_card: np.ndarray,
    downhole_card: np.ndarray,
    title: str = "Dynamometer Cards",
    rod_float_detected: bool = False
) -> go.Figure:
    """
    Returns a Plotly Figure with:
        - Surface card: dashed blue line
        - Downhole card: solid line (red if rod_float_detected, green otherwise)
        - X axis: Position (m)
        - Y axis: Load (N)
    """

def plot_thermal_decay(
    df: pd.DataFrame,
    highlight_day: int | None = None
) -> go.Figure:
    """
    Returns a dual-axis Plotly Figure:
        - Left Y: Temperature (°C), line plot
        - Right Y: Viscosity (cP), log-scale line plot
        - X axis: Day
        - Vertical marker at highlight_day if provided
    """
```

**Internal logic:**
- All `import plotly.graph_objects as go` and layout code lives here.
- `app.py` receives `Figure` objects and passes them directly to `st.plotly_chart()`.

---

## 9. `components/metrics.py`

| Attribute | Value |
|-----------|-------|
| **Purpose** | Render KPI metric cards and warning banners in Streamlit |
| **Depends on** | `requirements.txt` (streamlit) |
| **Depended on by** | `app.py` |

**Public API:**
```python
def render_metric(
    label: str,
    value: str | float,
    unit: str,
    delta: str | None = None,
    alert: bool = False
) -> None:
    """
    Renders a single KPI card using st.columns + st.markdown.
    If alert=True, card border turns red.
    """

def render_warning_banner(diagnostics: dict) -> None:
    """
    Renders a st.warning banner if diagnostics["severity"] is "medium" or "high".
    Includes details: delay_pct, rod_float status, impact_loading status.
    """
```

**Internal logic:**
- Uses `st.columns(3)` layout for metric cards.
- Custom CSS via `st.markdown(..., unsafe_allow_html=True)` for card styling.
- Warning banner uses native `st.warning()` with formatted string.

---

## 10. `app.py`

| Attribute | Value |
|-----------|-------|
| **Purpose** | Main Streamlit entry point — orchestrates all modules |
| **Depends on** | All `core/` and `components/` modules |
| **Depended on by** | Nothing (entry point) |

**Public API:** None (Streamlit script, not imported).

**Internal logic:**
```python
import streamlit as st
from core.data_gen import generate_css_cycle
from core.physics_engine import calculate_downhole_card
from core.diagnostics import detect_rod_float
from core.optimizer import calculate_safe_spm
from components.charts import plot_dyno_overlay, plot_thermal_decay
from components.metrics import render_metric, render_warning_banner

@st.cache_data
def load_data():
    return generate_css_cycle()

def main():
    st.set_page_config(page_title="ThermaTwin SRP", layout="wide")
    df = load_data()
    
    # Sidebar
    day = st.sidebar.slider("Day since CSS cycle start", 0, 29, 0)
    row = df[df["day"] == day].iloc[0]
    
    # Compute
    surface_card = np.column_stack([row["surface_load_n"], row["surface_pos_m"]])
    downhole_card = calculate_downhole_card(
        row["surface_load_n"].values,
        row["surface_pos_m"].values,
        row["viscosity_cp"]
    )
    diag = detect_rod_float(downhole_card, surface_card, row["viscosity_cp"])
    opt = calculate_safe_spm(row["viscosity_cp"])
    
    # Render
    col1, col2, col3 = st.columns(3)
    with col1: render_metric("Max Safe SPM", f"{opt['max_safe_spm']:.1f}", "SPM")
    with col2: render_metric("VFD Frequency", f"{opt['recommended_vfd_hz']:.1f}", "Hz")
    with col3: render_metric("Severity", diag["severity"].upper(), alert=diag["severity"]=="high")
    
    render_warning_banner(diag)
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(plot_thermal_decay(df, highlight_day=day))
    with col_right:
        st.plotly_chart(plot_dyno_overlay(surface_card, downhole_card, rod_float_detected=diag["rod_float"]))

if __name__ == "__main__":
    main()
```

---

## 11. `data/synthetic_well_data.csv`

| Attribute | Value |
|-----------|-------|
| **Purpose** | Static CSV cache of generated dataset for faster startup |
| **Depends on** | `core/data_gen.py` |
| **Depended on by** | `app.py` (optional fallback) |

**Generation:**
```bash
python -c "from core.data_gen import generate_css_cycle; generate_css_cycle().to_csv('data/synthetic_well_data.csv', index=False)"
```

**Columns:** `day`, `temperature_c`, `viscosity_cp`, `surface_load_n`, `surface_pos_m`

---

## 12. `tests/test_data_gen.py`

| Attribute | Value |
|-----------|-------|
| **Purpose** | Unit tests for `core/data_gen.py` |
| **Depends on** | `core/data_gen.py`, `pytest` |
| **Depended on by** | Nothing |

**Test cases:**
- `test_dataframe_shape`: Assert `(3000, 5)` for default params
- `test_temperature_range`: Assert min ≈ 50, max ≈ 200
- `test_viscosity_monotonic`: Assert viscosity increases monotonically
- `test_no_nan_values`: Assert no NaN in any column
- `test_custom_days`: Assert `generate_css_cycle(days=10)` returns 1000 rows

---

## 13. `tests/test_physics_engine.py`

| Attribute | Value |
|-----------|-------|
| **Purpose** | Unit tests for `core/physics_engine.py` |
| **Depends on** | `core/physics_engine.py`, `pytest` |
| **Depended on by** | Nothing |

**Test cases:**
- `test_output_shape`: Assert output is `(N, 2)`
- `test_low_viscosity_normal_card`: Assert minimal distortion when viscosity < 5000
- `test_high_viscosity_rod_float`: Assert larger phase delay when viscosity >= 5000
- `test_damping_increases_with_viscosity`: Assert damping coefficient scales correctly

---

## 14. `tests/test_diagnostics.py`

| Attribute | Value |
|-----------|-------|
| **Purpose** | Unit tests for `core/diagnostics.py` |
| **Depends on** | `core/diagnostics.py`, `core/physics_engine.py`, `pytest` |
| **Depended on by** | Nothing |

**Test cases:**
- `test_no_rod_float_low_viscosity`: Assert `rod_float=False` when viscosity < 5000
- `test_rod_float_high_viscosity`: Assert `rod_float=True` when viscosity > 5000 and delay > threshold
- `test_severity_low`: Assert `"low"` when no flags
- `test_severity_high`: Assert `"high"` when both flags present
- `test_boundary_viscosity`: Assert behavior at exactly 5000 cP

---

## 15. `tests/test_optimizer.py`

| Attribute | Value |
|-----------|-------|
| **Purpose** | Unit tests for `core/optimizer.py` |
| **Depends on** | `core/optimizer.py`, `pytest` |
| **Depended on by** | Nothing |

**Test cases:**
- `test_safe_spm_decreases_with_viscosity`: Assert inverse relationship
- `test_safe_spm_clamped_max`: Assert never exceeds `max_spm`
- `test_safe_spm_clamped_min`: Assert never below `min_spm`
- `test_frequency_conversion`: Assert `freq = spm * motor_pole_pairs`
- `test_safety_factor_applied`: Assert `recommended < max`

---

## 16. `tests/test_app_pipeline.py`

| Attribute | Value |
|-----------|-------|
| **Purpose** | Integration test — full pipeline from data to diagnostics |
| **Depends on** | All `core/` modules, `pytest` |
| **Depended on by** | Nothing |

**Test cases:**
- `test_full_pipeline_day_0`: Assert no exceptions, expected dict keys
- `test_full_pipeline_day_30`: Assert rod_float detected, severity high
- `test_full_pipeline_all_days`: Loop all 30 days, assert no exceptions

---

## 17. `README.md`

| Attribute | Value |
|-----------|-------|
| **Purpose** | Project documentation and run instructions |
| **Depends on** | Nothing |
| **Depended on by** | Nothing |

**Contents:**
- Project title and description
- Folder structure diagram
- Quickstart (venv setup, pip install, streamlit run)
- Module descriptions
- Testing instructions (`pytest tests/`)
- Known limitations

---

## Summary: File Dependency Matrix

| File | Depends On | Depended On By |
|------|------------|----------------|
| `requirements.txt` | — | All `.py` files |
| `core/__init__.py` | — | `app.py`, tests |
| `core/data_gen.py` | `requirements.txt` | `app.py`, `data/synthetic_well_data.csv`, `test_data_gen.py` |
| `core/physics_engine.py` | `requirements.txt` | `app.py`, `test_physics_engine.py`, `test_diagnostics.py` |
| `core/diagnostics.py` | `requirements.txt` | `app.py`, `components/metrics.py`, `test_diagnostics.py` |
| `core/optimizer.py` | `requirements.txt` | `app.py`, `test_optimizer.py` |
| `components/__init__.py` | — | `app.py` |
| `components/charts.py` | `requirements.txt` | `app.py` |
| `components/metrics.py` | `requirements.txt` | `app.py` |
| `app.py` | All `core/`, `components/` | — |
| `data/synthetic_well_data.csv` | `core/data_gen.py` | `app.py` (optional) |
| `tests/test_data_gen.py` | `core/data_gen.py` | — |
| `tests/test_physics_engine.py` | `core/physics_engine.py` | — |
| `tests/test_diagnostics.py` | `core/diagnostics.py`, `core/physics_engine.py` | — |
| `tests/test_optimizer.py` | `core/optimizer.py` | — |
| `tests/test_app_pipeline.py` | All `core/` | — |
| `README.md` | — | — |
