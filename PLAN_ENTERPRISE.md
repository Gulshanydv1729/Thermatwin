# ThermaTwin — Enterprise Architecture Upgrade Plan

## Objective

Transition from the local Streamlit prototype to a production-grade enterprise system with a React/Next.js frontend, FastAPI backend, TimescaleDB/InfluxDB time-series storage, Kafka/MQTT real-time ingestion, and Docker/Kubernetes deployment — starting with the backend data models, ingestion pipeline, and API layer.

## Current State

| Layer | Technology | Key Files |
|-------|-----------|-----------|
| Frontend | Streamlit + Plotly | `app.py`, `components/charts.py`, `components/metrics.py` |
| Backend | None (monolithic) | `app.py` |
| Data | Synthetic CSV | `core/data_gen.py`, `data/synthetic_well_data.csv` |
| Physics | NumPy/SciPy | `core/physics_engine.py`, `core/diagnostics.py`, `core/optimizer.py` |
| Storage | None | — |
| Messaging | None | — |
| Deployment | None | `run.sh` |

The existing `core/` modules are pure-Python and framework-agnostic — they can be reused directly in the FastAPI backend.

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        REACT / NEXT.JS FRONTEND                         │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │ Dyno Card   │  │ Thermal      │  │ KPI         │  │ Alert        │  │
│  │ Overlay     │  │ Decay Chart  │  │ Dashboard   │  │ Panel        │  │
│  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘  │
│         │                │                 │                │          │
│         └────────────────┴────────┬────────┴────────────────┘          │
│                                   │ HTTP / WebSocket                    │
└───────────────────────────────────┼─────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼─────────────────────────────────────┐
│                        FASTAPI BACKEND                                  │
│                                   │                                     │
│  ┌────────────────────────────────┼────────────────────────────────┐   │
│  │                         API Routes                              │   │
│  │  /api/v1/wells  /api/v1/css-cycles  /api/v1/diagnostics          │   │
│  │  /api/v1/optimization  /ws/realtime                             │   │
│  └────────────────────────────────┼────────────────────────────────┘   │
│                                   │                                     │
│  ┌────────────────────────────────┼────────────────────────────────┐   │
│  │                    Service Layer                                │   │
│  │  WellService  CssCycleService  DiagnosticsService  OptimService │   │
│  └────────────────────────────────┼────────────────────────────────┘   │
│                                   │                                     │
│  ┌────────────────────────────────┼────────────────────────────────┐   │
│  │              Physics & ML Engine (reused from core/)            │   │
│  │  data_gen.py  physics_engine.py  diagnostics.py  optimizer.py    │   │
│  └────────────────────────────────┼────────────────────────────────┘   │
│                                   │                                     │
│  ┌────────────────────────────────┼────────────────────────────────┐   │
│  │              Ingestion Pipeline                                 │   │
│  │  MQTT Consumer → Kafka Producer → Stream Processor              │   │
│  └────────────────────────────────┼────────────────────────────────┘   │
│                                   │                                     │
└───────────────────────────────────┼─────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼─────────────────────────────────────┐
│                         DATA LAYER                                      │
│                                   │                                     │
│  ┌─────────────────────┐   ┌──────┴──────┐   ┌─────────────────────┐   │
│  │   TimescaleDB       │   │  InfluxDB   │   │   PostgreSQL        │   │
│  │  (SCADA telemetry)  │   │  (VFD freq) │   │  (well metadata)    │   │
│  └─────────────────────┘   └─────────────┘   └─────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Apache Kafka                                │   │
│  │  Topics: scada.raw  scada.processed  diagnostics  alerts        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼─────────────────────────────────────┐
│                      EDGE LAYER (Wellsite)                              │
│                                   │                                     │
│  ┌─────────────────────┐   ┌──────┴──────┐   ┌─────────────────────┐   │
│  │  PLC / RTU          │   │ Edge Node   │   │  MQTT Broker        │   │
│  │  (Surface Load,     │──▶│ (Filter,    │──▶│  (Mosquitto)        │   │
│  │   Position, VFD)    │   │  Compress)  │   │                     │   │
│  └─────────────────────┘   └─────────────┘   └─────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Backend Data Models

### PostgreSQL (Relational Metadata)

#### `wells` table
```sql
CREATE TABLE wells (
    well_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    well_name           VARCHAR(50) NOT NULL UNIQUE,
    field_name          VARCHAR(100) NOT NULL DEFAULT 'Baghewala',
    api_number          VARCHAR(20) UNIQUE,
    latitude            DECIMAL(10, 8),
    longitude           DECIMAL(11, 8),
    total_depth_m       DECIMAL(8, 2),
    perforation_top_m   DECIMAL(8, 2),
    perforation_bottom_m DECIMAL(8, 2),
    pump_type           VARCHAR(50) DEFAULT 'SRP',
    rod_string_config   JSONB,           -- rod grades, diameters, lengths
    pump_displacement   DECIMAL(6, 4),   -- bbl/stroke
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_wells_field ON wells(field_name);
```

#### `css_cycles` table
```sql
CREATE TABLE css_cycles (
    cycle_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    well_id             UUID NOT NULL REFERENCES wells(well_id) ON DELETE CASCADE,
    cycle_number        INT NOT NULL,    -- 1st, 2nd, 3rd CSS cycle
    injection_start     TIMESTAMPTZ NOT NULL,
    injection_end       TIMESTAMPTZ,
    soak_start          TIMESTAMPTZ,
    soak_end            TIMESTAMPTZ,
    production_start    TIMESTAMPTZ,
    production_end      TIMESTAMPTZ,
    steam_injected_bbl  DECIMAL(10, 2),
    steam_quality       DECIMAL(4, 3),   -- 0.0 to 1.0
    injection_pressure_kpa DECIMAL(8, 2),
    reservoir_pressure_kpa  DECIMAL(8, 2),
    initial_temp_c      DECIMAL(6, 2),
    target_temp_c       DECIMAL(6, 2),
    status              VARCHAR(20) DEFAULT 'active',  -- planned/injection/soak/production/completed
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(well_id, cycle_number)
);

CREATE INDEX idx_css_well ON css_cycles(well_id);
CREATE INDEX idx_css_status ON css_cycles(status);
CREATE INDEX idx_css_dates ON css_cycles(production_start, production_end);
```

#### `equipment_specs` table
```sql
CREATE TABLE equipment_specs (
    spec_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    well_id             UUID NOT NULL REFERENCES wells(well_id) ON DELETE CASCADE,
    equipment_type      VARCHAR(50) NOT NULL,  -- 'rod_string', 'pump', 'vfd', 'motor'
    component_name      VARCHAR(100),
    manufacturer        VARCHAR(100),
    model               VARCHAR(100),
    specifications      JSONB,                -- flexible schema per equipment type
    installed_date      DATE,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_equip_well ON equipment_specs(well_id);
```

#### `maintenance_logs` table
```sql
CREATE TABLE maintenance_logs (
    log_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    well_id             UUID NOT NULL REFERENCES wells(well_id) ON DELETE CASCADE,
    cycle_id            UUID REFERENCES css_cycles(cycle_id),
    maintenance_type    VARCHAR(50) NOT NULL,  -- 'rod_failure', 'pump_overhaul', 'preventive'
    description         TEXT,
    technician_name     VARCHAR(100),
    start_time          TIMESTAMPTZ NOT NULL,
    end_time            TIMESTAMPTZ,
    cost_usd            DECIMAL(10, 2),
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_maint_well ON maintenance_logs(well_id);
```

### TimescaleDB (Time-Series Telemetry)

#### `scada_surface` hypertable
```sql
CREATE TABLE scada_surface (
    time                TIMESTAMPTZ NOT NULL,
    well_id             UUID NOT NULL,
    cycle_id            UUID,
    surface_load_n      DECIMAL(10, 2),       -- polished rod load (N)
    surface_pos_m       DECIMAL(6, 3),        -- polished rod position (m)
    spm                 DECIMAL(4, 2),        -- strokes per minute
    vfd_frequency_hz    DECIMAL(5, 2),        -- VFD output frequency
    motor_current_a     DECIMAL(6, 2),        -- motor current draw
    motor_voltage_v     DECIMAL(6, 2),
    wellhead_temp_c     DECIMAL(6, 2),
    wellhead_pressure_kpa DECIMAL(8, 2)
);

SELECT create_hypertable('scada_surface', 'time');

CREATE INDEX idx_scada_well_time ON scada_surface(well_id, time DESC);
```

#### `scada_downhole` hypertable
```sql
CREATE TABLE scada_downhole (
    time                TIMESTAMPTZ NOT NULL,
    well_id             UUID NOT NULL,
    cycle_id            UUID,
    downhole_load_n     DECIMAL(10, 2),
    downhole_pos_m      DECIMAL(6, 3),
    temperature_c       DECIMAL(6, 2),        -- bottom-hole temperature
    viscosity_cp        DECIMAL(10, 2),        -- estimated viscosity
    rod_float_flag      BOOLEAN DEFAULT FALSE,
    impact_loading_flag BOOLEAN DEFAULT FALSE
);

SELECT create_hypertable('scada_downhole', 'time');

CREATE INDEX idx_downhole_well_time ON scada_downhole(well_id, time DESC);
```

### InfluxDB (High-Frequency VFD Metrics)

```
Measurement: vfd_telemetry
Tags: well_id, cycle_id
Fields: frequency_hz, current_a, voltage_v, power_kw, torque_nm
Timestamp: nanosecond precision
```

### Kafka Topics

| Topic | Partitions | Retention | Schema |
|-------|-----------|-----------|--------|
| `scada.raw` | 12 | 7 days | Avro |
| `scada.processed` | 12 | 7 days | Avro |
| `diagnostics.events` | 6 | 30 days | Avro |
| `alerts.critical` | 3 | 90 days | JSON |
| `optimization.commands` | 3 | 1 day | JSON |

## Implementation Steps

### Phase 1: Foundation — Database & Models

#### Step 1.1: Database schema files
- **Files:** `backend/db/migrations/001_create_wells.sql`, `002_create_css_cycles.sql`, `003_create_scada_surface.sql`, `004_create_scada_downhole.sql`
- **What:** Write all DDL for PostgreSQL and TimescaleDB
- **Dependencies:** None

#### Step 1.2: SQLAlchemy models
- **Files:** `backend/models/well.py`, `backend/models/css_cycle.py`, `backend/models/scada.py`, `backend/models/equipment.py`, `backend/models/maintenance.py`
- **What:** Define ORM models matching the schema
- **Dependencies:** Step 1.1

#### Step 1.3: Pydantic schemas
- **Files:** `backend/schemas/well.py`, `backend/schemas/css_cycle.py`, `backend/schemas/scada.py`, `backend/schema/diagnostics.py`, `backend/schemas/optimization.py`
- **What:** Request/response validation schemas for FastAPI
- **Dependencies:** Step 1.2

#### Step 1.4: Database connection & config
- **Files:** `backend/db/session.py`, `backend/config.py`, `backend/db/__init__.py`
- **What:** SQLAlchemy async session factory, environment-based config (pydantic-settings)
- **Dependencies:** Step 1.2

### Phase 2: Ingestion Pipeline

#### Step 2.1: MQTT consumer
- **File:** `backend/ingestion/mqtt_consumer.py`
- **What:** Async MQTT client connecting to Mosquitto broker, subscribing to `scada/+/+` topics, forwarding raw payloads to Kafka
- **Dependencies:** Step 1.4

#### Step 2.2: Kafka stream processor
- **File:** `backend/ingestion/stream_processor.py`
- **What:** Faust/Bytewax stream processor consuming from `scada.raw`, applying calibration, unit conversion, and edge filtering → produces to `scada.processed`
- **Dependencies:** Step 2.1

#### Step 2.3: TimescaleDB writer
- **File:** `backend/ingestion/tsdb_writer.py`
- **What:** Batch consumer from `scada.processed`, bulk-inserts into TimescaleDB hypertables using COPY
- **Dependencies:** Step 2.2

#### Step 2.4: InfluxDB writer
- **File:** `backend/ingestion/influx_writer.py`
- **What:** Consumes VFD high-frequency metrics, writes to InfluxDB via async client
- **Dependencies:** Step 2.2

### Phase 3: API Layer

#### Step 3.1: FastAPI app factory
- **Files:** `backend/main.py`, `backend/api/__init__.py`, `backend/api/deps.py`
- **What:** App factory with CORS, middleware, exception handlers, dependency injection for DB sessions
- **Dependencies:** Step 1.4

#### Step 3.2: Well endpoints
- **File:** `backend/api/routes/wells.py`
- **What:** CRUD for `/api/v1/wells` — list, get, create, update, delete
- **Dependencies:** Step 3.1

#### Step 3.3: CSS cycle endpoints
- **File:** `backend/api/routes/css_cycles.py`
- **What:** CRUD for `/api/v1/css-cycles`, plus `/api/v1/css-cycles/{id}/thermal-profile` for decay curve
- **Dependencies:** Step 3.1

#### Step 3.4: Telemetry endpoints
- **File:** `backend/api/routes/telemetry.py`
- **What:** `/api/v1/telemetry/surface` and `/api/v1/telemetry/downhole` — time-range queries with downsampling
- **Dependencies:** Step 3.1

#### Step 3.5: Diagnostics endpoints
- **File:** `backend/api/routes/diagnostics.py`
- **What:** `/api/v1/diagnostics/latest` (latest diag result), `/api/v1/diagnostics/history` (time range)
- **Dependencies:** Step 3.1

#### Step 3.6: Optimization endpoints
- **File:** `backend/api/routes/optimization.py`
- **What:** `/api/v1/optimization/recommend` — returns SPM/VFD recommendation for given well+cycle
- **Dependencies:** Step 3.1

#### Step 3.7: WebSocket endpoint
- **File:** `backend/api/routes/websocket.py`
- **What:** `/ws/realtime/{well_id}` — pushes live diagnostics and optimization updates every N seconds
- **Dependencies:** Step 3.1

### Phase 4: Service Layer (Reuse Core Modules)

#### Step 4.1: Service wrappers
- **Files:** `backend/services/well_service.py`, `backend/services/diagnostics_service.py`, `backend/services/optimization_service.py`
- **What:** Thin async wrappers around existing `core/physics_engine.py`, `core/diagnostics.py`, `core/optimizer.py` — no physics code changes, just async I/O
- **Dependencies:** Phase 3

### Phase 5: Containerization

#### Step 5.1: Dockerfiles
- **Files:** `Dockerfile.backend`, `Dockerfile.frontend`, `Dockerfile.ingestion`
- **What:** Multi-stage builds for each service
- **Dependencies:** Phases 1–4

#### Step 5.2: Docker Compose
- **File:** `docker-compose.yml`
- **What:** Full stack: PostgreSQL, TimescaleDB, InfluxDB, Kafka, Mosquitto, backend, ingestion, frontend
- **Dependencies:** Step 5.1

### Phase 6: Frontend (React/Next.js)

#### Step 6.1: Next.js scaffold
- **Files:** `frontend/` (new directory)
- **What:** Next.js app with TypeScript, Tailwind CSS, React Query
- **Dependencies:** Phase 3

#### Step 6.2: Dashboard pages
- **Files:** `frontend/app/page.tsx`, `frontend/app/wells/[id]/page.tsx`
- **What:** Main dashboard with dyno card overlay, thermal decay, KPI cards
- **Dependencies:** Step 6.1

#### Step 6.3: WebSocket hook
- **File:** `frontend/hooks/useRealtime.ts`
- **What:** React hook connecting to `/ws/realtime/{well_id}` for live updates
- **Dependencies:** Step 6.1

## Testing

### Unit tests
- `tests/backend/test_models.py` — SQLAlchemy model validation
- `tests/backend/test_schemas.py` — Pydantic schema validation
- `tests/ingestion/test_stream_processor.py` — stream processing logic
- Existing `tests/test_*.py` — all 27 tests must continue passing (core modules unchanged)

### Integration tests
- `tests/integration/test_api.py` — FastAPI TestClient against full API
- `tests/integration/test_ingestion.py` — end-to-end MQTT → Kafka → TimescaleDB
- `tests/integration/test_websocket.py` — WebSocket realtime push

### Edge cases
- Database connection loss during ingestion
- Malformed MQTT payloads
- Time-range queries returning empty results
- Concurrent WebSocket connections

## Risks

| Risk | Mitigation |
|------|------------|
| **Kafka/MQTT unavailable in dev** | Provide Docker Compose with all services; support "mock mode" using synthetic data generator |
| **TimescaleDB extension not available** | Fall back to regular PostgreSQL with partitioning |
| **Core physics modules are synchronous** | Wrap in `run_in_threadpool` or `asyncio.to_thread` — do NOT rewrite physics code |
| **WebSocket scaling** | Use Redis pub/sub as backend for horizontal WebSocket scaling |
| **Schema migration failures** | Use Alembic for versioned migrations with rollback support |
| **scikit-learn unused** | Reserved for future ML models; keep dependency |

## Files To Change

```
thermatwin/
├── backend/
│   ├── main.py                          # FastAPI app factory
│   ├── config.py                        # pydantic-settings config
│   ├── api/
│   │   ├── deps.py                      # dependency injection
│   │   └── routes/
│   │       ├── wells.py
│   │       ├── css_cycles.py
│   │       ├── telemetry.py
│   │       ├── diagnostics.py
│   │       ├── optimization.py
│   │       └── websocket.py
│   ├── db/
│   │   ├── session.py                   # async SQLAlchemy session
│   │   └── migrations/
│   │       ├── 001_create_wells.sql
│   │       ├── 002_create_css_cycles.sql
│   │       ├── 003_create_scada_surface.sql
│   │       └── 004_create_scada_downhole.sql
│   ├── models/
│   │   ├── well.py
│   │   ├── css_cycle.py
│   │   ├── scada.py
│   │   ├── equipment.py
│   │   └── maintenance.py
│   ├── schemas/
│   │   ├── well.py
│   │   ├── css_cycle.py
│   │   ├── scada.py
│   │   ├── diagnostics.py
│   │   └── optimization.py
│   ├── services/
│   │   ├── well_service.py
│   │   ├── diagnostics_service.py
│   │   └── optimization_service.py
│   └── ingestion/
│       ├── mqtt_consumer.py
│       ├── stream_processor.py
│       ├── tsdb_writer.py
│       └── influx_writer.py
├── frontend/
│   ├── package.json
│   ├── tailwind.config.ts
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── wells/[id]/page.tsx
│   ├── components/
│   │   ├── DynoCardOverlay.tsx
│   │   ├── ThermalDecayChart.tsx
│   │   ├── KpiCard.tsx
│   │   └── AlertBanner.tsx
│   └── hooks/
│       └── useRealtime.ts
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── Dockerfile.ingestion
└── core/                                # UNCHANGED — reused as-is
```

## Execution Order

1. **Phase 1** — Database schema + SQLAlchemy models + Pydantic schemas
2. **Phase 4** — Service layer wrappers (reuse `core/` immediately)
3. **Phase 3** — FastAPI routes + WebSocket
4. **Phase 5** — Docker Compose (full stack runnable locally)
5. **Phase 2** — Ingestion pipeline (MQTT → Kafka → TSDB)
6. **Phase 6** — React/Next.js frontend
7. **Tests** — Integration + E2E after each phase

---

**Ambiguity to resolve before implementation:**

1. **Kafka vs MQTT-only** — Is Kafka required for the initial phase, or can we start with direct MQTT → TimescaleDB and add Kafka later?
2. **TimescaleDB vs InfluxDB** — Should both be used, or is TimescaleDB sufficient for all time-series?
3. **Authentication** — Does the enterprise system need OAuth2/JWT auth, or is internal network trust sufficient for now?
4. **Frontend framework** — Is Next.js preferred, or would a simpler React SPA (Vite) suffice?
