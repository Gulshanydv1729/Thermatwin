"""FastAPI application entry point for the ThermaTwin digital twin.

Run locally with

.. code-block:: bash

    uvicorn backend.app.main:app --reload --port 8000

and in the container with the command in ``backend/Dockerfile``.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router as rest_router
from backend.app.api.websocket import router as websocket_router
from backend.app.core.config import get_config

__all__ = ["app", "create_app", "warm_up", "configure_torch_threads"]

logger = logging.getLogger("thermatwin")
logging.basicConfig(
    level=os.environ.get("THERMATWIN_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)


def configure_torch_threads(num_threads: Optional[int] = None) -> int:
    """Pin PyTorch to a fixed intra-op thread count and report what was set.

    The models are small enough that PyTorch's own thread pool costs far more in
    dispatch than it recovers in arithmetic, and the penalty swings wildly with
    machine load -- which showed up as a telemetry frame occasionally taking
    100-175 ms against a 12 ms median, sporadically failing the latency budget
    on an otherwise idle service.  A fixed count makes the frame cost
    deterministic.  See ``ApiConfig.torch_num_threads``.
    """
    config = get_config()
    count = int(config.api.torch_num_threads if num_threads is None else num_threads)
    if count < 1:
        raise ValueError("torch_num_threads must be positive")
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return 0
    torch.set_num_threads(count)
    # The inter-op pool can only be set before any parallel work starts, so a
    # failure here is expected on a re-configure and is not an error.
    try:
        torch.set_num_interop_threads(count)
    except RuntimeError:
        pass
    return count


def warm_up(weights_dir: Optional[Path] = None) -> Dict[str, object]:
    """Train or load the AI weights and prime the twin on process start.

    This is the on-boot routine: with no cached checkpoint it trains the dyno
    card classifier from the parametric card model, so no binary weights have to
    be committed to the repository.  It is deliberately best-effort -- a failure
    here must not stop the API from serving the physics endpoints.
    """
    config = get_config()
    target = Path(weights_dir) if weights_dir is not None else config.weights_dir
    target.mkdir(parents=True, exist_ok=True)
    result: Dict[str, object] = {"weights_dir": str(target)}
    result["torch_threads"] = configure_torch_threads()
    try:
        from backend.app.ai.dyno_classifier import DynoCardClassifier

        weights = target / "dyno_classifier.pt"
        classifier = DynoCardClassifier(train=False)
        if weights.exists():
            classifier.load(weights)
            result["dyno_classifier"] = "loaded"
        else:
            metrics = classifier.train_and_save(
                weights, epochs=10, samples_per_label=250
            )
            result["dyno_classifier"] = "trained"
            result["validation_accuracy"] = metrics.get("validation_accuracy")
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("dyno classifier warm-up failed: %s", exc)
        result["dyno_classifier"] = f"failed: {exc}"
    result["first_frame_ms"] = _prime_pipeline()
    return result


def _prime_pipeline() -> float:
    """Run one full telemetry pass so the first real frame is fast.

    The first PyTorch convolution on a CPU build initialises the oneDNN kernels
    and can take several seconds, and the first CSS trajectory has to be solved.
    Both are one-off costs, and paying them during start-up means the dashboard
    receives a frame immediately instead of sitting blank.  The result is
    best-effort: if priming fails the service still serves.
    """
    started = time.perf_counter()
    try:
        from backend.app.api.websocket import StreamEngine
        from backend.app.api.routes import get_twin

        frame = StreamEngine(twin=get_twin(), samples=64).tick()
        return float(frame["latency_ms"])
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("pipeline priming failed: %s", exc)
        return -1.0


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application lifespan: warm the AI weights up on start."""
    if os.environ.get("THERMATWIN_SKIP_WARMUP", "0") not in {"1", "true", "yes"}:
        result = await __import__("asyncio").to_thread(warm_up)
        logger.info("warm-up complete: %s", result)
    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    config = get_config()
    # Applied here as well as in the warm-up, so the setting takes effect even
    # when the warm-up is skipped.
    configure_torch_threads()
    application = FastAPI(
        title="ThermaTwin Digital Twin",
        version="3.0.0",
        summary=(
            "Physics-informed well-to-surface digital twin for cyclic steam "
            "stimulation and sucker rod pump operations in the Baghewala field"
        ),
        description=(
            "SIH26120 - Oil India Limited, Baghewala heavy oil asset.  Couples a "
            "Marx-Langenheim CSS reservoir model, Beggs-Brill multiphase "
            "wellbore hydraulics with Ramey transient heat transfer, a Gibbs "
            "damped wave equation rod-string solver, a residual 1D CNN dyno card "
            "classifier and a bidirectional GRU thermal surrogate, closed "
            "through an extended Kalman filter and a hydraulic VFD controller."
        ),
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(rest_router)
    application.include_router(websocket_router)

    @application.get("/", tags=["meta"])
    async def root() -> Dict[str, object]:
        """Service banner."""
        return {
            "service": "thermatwin",
            "version": "3.0.0",
            "specification": "SIH26120",
            "field": "Baghewala, Oil India Limited",
            "docs": "/docs",
            "websocket": "/ws/telemetry",
        }

    return application


app = create_app()
