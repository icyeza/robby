"""FastAPI application factory. Run: uv run uvicorn robby.main:create_app --factory"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from robby.audit.national import router as national_router
from robby.audit.routes import router as audit_router
from robby.auth.routes import router as auth_router
from robby.clinical.routes import router as clinical_router
from robby.config import Settings, get_settings
from robby.db import get_sessionmaker
from robby.facilities.routes import router as facilities_router
from robby.monitoring.readiness import router as readiness_router
from robby.monitoring.reconcile import reconcile_loop
from robby.monitoring.routes import router as monitoring_router
from robby.registry.approval_log import router as approval_log_router
from robby.registry.routes import router as models_router
from robby.users.routes import router as users_router
from robby.web.errors import install_error_handlers
from robby.web.placeholders import build_placeholder_router
from robby.web.security import install_security_headers
from robby.web.templating import STATIC_DIR, build_templates


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task = None
        if settings.reconcile_interval_minutes > 0:
            task = asyncio.create_task(
                reconcile_loop(app.state.sessionmaker, settings.reconcile_interval_minutes)
            )
        try:
            yield
        finally:
            if task is not None:
                task.cancel()

    app = FastAPI(
        title="Robson Readiness",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.templates = build_templates(settings)
    app.state.sessionmaker = get_sessionmaker(settings.database_url)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    install_error_handlers(app)
    install_security_headers(app)
    app.include_router(auth_router)
    app.include_router(facilities_router)
    app.include_router(users_router)
    app.include_router(clinical_router)
    app.include_router(audit_router)
    app.include_router(national_router)
    app.include_router(monitoring_router)
    app.include_router(readiness_router)
    app.include_router(models_router)
    app.include_router(approval_log_router)
    app.include_router(build_placeholder_router())
    return app
