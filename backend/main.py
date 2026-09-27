"""ThermaTwin Enterprise Backend — FastAPI Application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.api.routes import wells, css_cycles, telemetry, diagnostics, optimization

app = FastAPI(
    title="ThermaTwin SRP Digital Twin API",
    version="2.0.0",
    description="Enterprise digital twin for Baghewala field SRP operations",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(wells.router)
app.include_router(css_cycles.router)
app.include_router(telemetry.router)
app.include_router(diagnostics.router)
app.include_router(optimization.router)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "thermatwin-enterprise"}


@app.get("/")
async def root():
    return {
        "name": "ThermaTwin SRP Digital Twin",
        "version": "2.0.0",
        "docs": "/docs",
    }
