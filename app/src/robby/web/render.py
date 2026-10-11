"""render(): every page goes through here so the layout always gets its context."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from robby.auth.dependencies import CurrentUser
from robby.config import Settings
from robby.web.csrf import session_csrf_token
from robby.web.nav import nav_for, who_for


def render(
    request: Request,
    template: str,
    *,
    current: CurrentUser | None = None,
    status_code: int = 200,
    **context: Any,
) -> HTMLResponse:
    settings: Settings = request.app.state.settings
    templates: Jinja2Templates = request.app.state.templates
    base: dict[str, Any] = {
        "current": current,
        "user": None,
        "nav": [],
        "who": None,
        "csrf_token": "",
        "demo": settings.demo,
    }
    if current is not None:
        user = current.user
        base["user"] = user
        base["who"] = who_for(user)
        base["csrf_token"] = session_csrf_token(settings, current.session.id)
        if not user.must_change_password:
            base["nav"] = nav_for(user.role, request.url.path)
    return templates.TemplateResponse(request, template, {**base, **context}, status_code=status_code)
