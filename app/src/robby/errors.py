"""Exceptions services raise for callers (routes, CLI) to turn into responses."""

from __future__ import annotations


class FormErrors(Exception):
    """Input was not accepted. `errors` maps a form field name to a message for that field."""

    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__("; ".join(errors.values()))
        self.errors = errors


class NotFound(LookupError):
    """The record does not exist, or is outside the actor's scope (shown as 404 either way)."""


class NotAllowed(PermissionError):
    """The actor may not do this; the message is shown to them."""
