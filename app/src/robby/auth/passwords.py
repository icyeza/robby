"""Password hashing (argon2id), temporary passwords and the password rule."""

from __future__ import annotations

import secrets
import threading

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

MIN_PASSWORD_LENGTH = 10
# Lowercase letters and digits without i, l, o, 0 and 1, which are easy to misread.
TEMPORARY_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"

# argon2-cffi's PasswordHasher is argon2id with the RFC 9106 low-memory profile: 64 MiB and four
# lanes per hash. That cost is what makes guessing slow, so it stays; instead, at most
# CONCURRENT_HASHES hashes run at once per process, and further sign-ins wait their turn. A burst
# of sign-ins (a shift change) then needs 2 x 64 MiB per worker instead of 64 MiB per request.
_hasher = PasswordHasher()
CONCURRENT_HASHES = 2
_hashing = threading.BoundedSemaphore(CONCURRENT_HASHES)


def hash_password(password: str) -> str:
    with _hashing:
        return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        with _hashing:
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
