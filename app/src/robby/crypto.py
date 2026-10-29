"""Encryption at rest for patient identifiers (patient number and name on `admission`).

One master key (ROBBY_FIELD_KEY, 32 random bytes) yields two subkeys through HKDF-SHA256: one for
AES-256-GCM, one for the blind index. A stored value is version byte 0x01, a 12-byte random
nonce, then ciphertext and tag; the associated data is the column's name, so a value copied into
another column does not decrypt. The blind index is HMAC-SHA256 over the facility id and the
normalised patient number: equal numbers at one facility give equal indexes, the same number at
two facilities does not."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import secrets
import uuid
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

KEY_BYTES = 32
NONCE_BYTES = 12
VERSION = b"\x01"
ENCRYPTION_INFO = b"robby field encryption v1"
INDEX_INFO = b"robby blind index v1"
KEY_VARIABLE = "ROBBY_FIELD_KEY"


class FieldKeyError(ValueError):
    """The configured key is missing or malformed."""


class DecryptionError(ValueError):
    """A stored value was altered, belongs to another column, or was written under another key."""


def generate_key() -> str:
    """A new master key: 32 random bytes, base64url without padding."""
    return base64.urlsafe_b64encode(secrets.token_bytes(KEY_BYTES)).rstrip(b"=").decode("ascii")


def decode_key(text: str) -> bytes:
    """The 32 key bytes of a base64url key (padding optional)."""
    value = text.strip()
    try:
        raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except (binascii.Error, ValueError):
        raise FieldKeyError("must be base64url (robby-admin keys generate)") from None
    if len(raw) != KEY_BYTES:
        raise FieldKeyError(f"must decode to {KEY_BYTES} bytes, not {len(raw)}")
    return raw


def normalise_patient_number(number: str) -> str:
    """Upper case, no outer whitespace, inner whitespace runs collapsed to one space."""
    return " ".join(number.split()).upper()


def _subkey(master: bytes, info: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=KEY_BYTES, salt=None, info=info).derive(master)


@dataclass(frozen=True)
class Keyring:
    """The two subkeys of one master key. Build with Keyring.from_master."""

    encryption_key: bytes = field(repr=False)
    index_key: bytes = field(repr=False)

    @classmethod
    def from_master(cls, master: bytes) -> Keyring:
        if len(master) != KEY_BYTES:
            raise FieldKeyError(f"master key must be {KEY_BYTES} bytes")
        return cls(_subkey(master, ENCRYPTION_INFO), _subkey(master, INDEX_INFO))

    def encrypt(self, plaintext: str, column: str) -> bytes:
        nonce = os.urandom(NONCE_BYTES)
        sealed = AESGCM(self.encryption_key).encrypt(
            nonce, plaintext.encode("utf-8"), column.encode("utf-8")
        )
        return VERSION + nonce + sealed

    def decrypt(self, stored: bytes, column: str) -> str:
        if stored[:1] != VERSION or len(stored) < 1 + NONCE_BYTES + 16:
            raise DecryptionError(f"{column}: unknown ciphertext format")
        nonce, sealed = stored[1 : 1 + NONCE_BYTES], stored[1 + NONCE_BYTES :]
        try:
            plain = AESGCM(self.encryption_key).decrypt(nonce, sealed, column.encode("utf-8"))
        except InvalidTag:
            raise DecryptionError(f"{column}: value does not authenticate") from None
        return plain.decode("utf-8")

    def blind_index(self, facility_id: uuid.UUID, patient_number: str) -> str:
        """64 hex characters; equal for equal normalised numbers at the same facility only."""
        message = facility_id.bytes + normalise_patient_number(patient_number).encode("utf-8")
        return hmac.new(self.index_key, message, hashlib.sha256).hexdigest()


_active: Keyring | None = None


def configure(master: bytes) -> Keyring:
    """Make the keyring of `master` the one the database columns use (once at start-up)."""
    global _active
    _active = Keyring.from_master(master)
    return _active


def key_from_environment() -> bytes:
    """ROBBY_FIELD_KEY from the environment or app/.env, without reading any other setting."""
    from dotenv import dotenv_values

    from robby.config import ENV_FILE

    text = os.environ.get(KEY_VARIABLE) or dotenv_values(ENV_FILE).get(KEY_VARIABLE)
    if not text:
        raise FieldKeyError(
            f"{KEY_VARIABLE} is not set; patient identifiers cannot be encrypted or read "
            "without it (app/README.md, 'Encryption at rest')"
        )
    try:
        return decode_key(text)
    except FieldKeyError as exc:
        raise FieldKeyError(f"{KEY_VARIABLE} {exc}") from None


def active() -> Keyring:
    """The configured keyring; configured from ROBBY_FIELD_KEY on first use if nobody did."""
    return _active or configure(key_from_environment())
