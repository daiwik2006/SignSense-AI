"""FastAPI application entry point.

Run locally:
    python -m app.main

Then open http://localhost:8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import init_service, router
from app.core.config import AppSettings
from app.core.logging_conf import configure_logging

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings or AppSettings.from_env()
    settings.validate()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        service = init_service(settings)
        service.ensure_loaded()  # load model once at startup; report if missing
        logger.info(
            "Starting app (env=%s, port=%s, model_ready=%s)",
            settings.app_env,
            settings.app_port,
            service.model_ready,
        )
        yield

    app = FastAPI(
        title="Real-Time Sign Language Detection & Translation",
        version="1.0.0",
        description=(
            "Local-only sign language detection. Webcam frames are processed in "
            "memory and are never uploaded or persisted."
        ),
        lifespan=lifespan,
    )

    origins = settings.cors_origins or (
        [f"http://localhost:{settings.app_port}", f"http://127.0.0.1:{settings.app_port}"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    app.include_router(router)

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = AppSettings.from_env()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )