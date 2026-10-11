"""Session tokens: 32 random bytes in the cookie; only their SHA-256 is stored."""

from __future__ import annotations

import hashlib
import secrets

TOKEN_BYTES = 32


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_session_token() -> tuple[str, str]:
    """A fresh token and its hash."""
    token = secrets.token_urlsafe(TOKEN_BYTES)
    return token, hash_token(token)
