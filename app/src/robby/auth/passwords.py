"""Password hashing (argon2id), temporary passwords and the password rule."""

from __future__ import annotations

import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

MIN_PASSWORD_LENGTH = 10
# Lowercase letters and digits without i, l, o, 0 and 1, which are easy to misread.
TEMPORARY_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"

# argon2-cffi's PasswordHasher is argon2id with the RFC 9106 low-memory profile.
_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    return _hasher.check_needs_rehash(password_hash)


def generate_temporary_password() -> str:
    """A one-time password an administrator reads out, e.g. ``k7mp-3xqa-9rtw``."""
    groups = ("".join(secrets.choice(TEMPORARY_ALPHABET) for _ in range(4)) for _ in range(3))
    return "-".join(groups)


def password_problem(password: str, *, username: str) -> str | None:
    """Why a chosen password is not acceptable, or None."""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Use at least {MIN_PASSWORD_LENGTH} characters."
    if password.strip().lower() == username.strip().lower():
        return "Do not use your username as your password."
    return None
