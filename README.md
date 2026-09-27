# ThermaTwin — SRP Digital Twin Prototype

A Streamlit-based rapid prototype digital twin for a Sucker-Rod Pump (SRP) system in the Baghewala field. Simulates reservoir thermal decay and its cascading effects on heavy crude viscosity, dynamometer card distortion (rod float), and VFD/SPM optimization.

## Quickstart

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

## Folder Structure

```
thermatwin-srp-prototype/
├── requirements.txt            # Package dependencies
├── README.md                   # This file
├── app.py                      # Main Streamlit dashboard (entry point)
├── PLAN.md                     # Implementation plan
├── PLAN_FILES.md               # Per-file plan & dependencies
│
├── data/
│   └── synthetic_well_data.csv # Generated dataset
│
├── core/                       # Digital twin logic modules
│   ├── __init__.py
│   ├── data_gen.py             # Thermal/viscosity decay dataset generator
│   ├── physics_engine.py       # Downhole card generation (wave equation)
│   ├── diagnostics.py          # Rod float & impact loading detection
│   └── optimizer.py            # VFD/SPM recommendation algorithms
│
├── components/                 # UI layout modules
│   ├── __init__.py
│   ├── charts.py               # Plotly figure builders
│   └── metrics.py              # KPI cards & warning banners
│
└── tests/                      # Test suite
    ├── test_data_gen.py
    ├── test_physics_engine.py
    ├── test_diagnostics.py
    ├── test_optimizer.py
    └── test_app_pipeline.py
```

## Module Descriptions

| Module | Purpose |
|--------|---------|
| `core/data_gen.py` | Generates 30-day synthetic dataset: temperature decay (200→50 °C), viscosity spike (500→50,000 cP), surface load/position |
| `core/physics_engine.py` | Converts surface card to bottom-hole dyno card using damped wave equation; introduces rod-float distortion when viscosity > 5000 cP |
| `core/diagnostics.py` | Detects delayed weight transfer (rod float) and mechanical impact loading from card shapes |
| `core/optimizer.py` | Calculates max safe SPM based on viscosity; outputs recommended VFD frequency |
| `components/charts.py` | Plotly figure builders for dyno card overlay and thermal decay curves |
| `components/metrics.py` | Streamlit KPI cards and warning banner renderers |
| `app.py` | Main orchestrator — wires all modules into the dashboard |

## Testing

```bash
pytest tests/ -v
```

## Known Limitations

- Physics model is a simplified 1D lumped-parameter approximation, not a full FEM simulation.
- Viscosity threshold (5000 cP) is configurable but not yet exposed in the UI.
- scikit-learn is included in dependencies but not yet used (reserved for future ML-based anomaly detection).
- Surface load/position data is synthetic and simplified (sinusoidal approximation).
