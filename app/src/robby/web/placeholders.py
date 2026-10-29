"""Pages listed in the navigation before they are built. Each leads to a plain "Not available yet"
page: role-checked like the real page will be, and never filled with fake data. None are listed
now."""

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from robby.auth.dependencies import CurrentUser, require
from robby.models import Role
from robby.web.render import render

PLACEHOLDER_PAGES: tuple[tuple[str, str, Role], ...] = ()


def _page(title: str, role: Role) -> Callable[..., Response]:
    allowed = require(role)

    def endpoint(request: Request, current: Annotated[CurrentUser, Depends(allowed)]) -> Response:
        return render(request, "pages/not_available.html", current=current, title=title)

    return endpoint


def build_placeholder_router() -> APIRouter:
    router = APIRouter()
    for path, title, role in PLACEHOLDER_PAGES:
        router.add_api_route(path, _page(title, role), methods=["GET"], name=f"later:{path}")
    return router
