"""Security headers on every response, and a styled 500 page instead of a stack trace."""

import logging
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from starlette.requests import ClientDisconnect

from robby.web.errors import error_page

logger = logging.getLogger("robby")
CLIENT_CLOSED_REQUEST = 499  # nginx's status for a request the client abandoned

CSP = (
    "default-src 'self'; base-uri 'self'; form-action 'self'; "
    "frame-ancestors 'none'; object-src 'none'"
)
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    "X-Frame-Options": "DENY",
}


def install_security_headers(app: FastAPI) -> None:
    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        try:
            response = await call_next(request)
        except ClientDisconnect:
            # The browser gave up on the request (htmx drops a live-panel preview when the next
            # one is sent): nobody is waiting for an answer, and nothing went wrong here.
            logger.info("Client left during %s %s", request.method, request.url.path)
            response = Response(status_code=CLIENT_CLOSED_REQUEST)
        except Exception:
            logger.exception("Unhandled error on %s %s", request.method, request.url.path)
            response = error_page(request, 500)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        if not request.url.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response
