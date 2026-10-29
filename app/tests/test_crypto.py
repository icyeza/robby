"""robby.crypto: AES-256-GCM sealing of identifiers and the blind index on patient numbers."""

import uuid
from pathlib import Path

import pytest

from robby import crypto
from robby.crypto import DecryptionError, FieldKeyError, Keyring, normalise_patient_number

KEYRING = Keyring.from_master(bytes(range(32)))
COLUMN = "admission.patient_name"


def test_a_value_round_trips() -> None:
    for text in ("Uwase Marie", "", "Ingabire 100% A_lice", "Mukamana Grâce"):
        assert KEYRING.decrypt(KEYRING.encrypt(text, COLUMN), COLUMN) == text


def test_the_stored_form_is_version_nonce_ciphertext_and_tag() -> None:
    stored = KEYRING.encrypt("Uwase Marie", COLUMN)
    assert stored[:1] == b"\x01"
    assert len(stored) == 1 + 12 + len("Uwase Marie") + 16
    assert b"Uwase" not in stored


def test_every_encryption_uses_a_fresh_nonce() -> None:
    sealed = {KEYRING.encrypt("Uwase Marie", COLUMN) for _ in range(50)}
    assert len(sealed) == 50
    assert len({s[1:13] for s in sealed}) == 50


@pytest.mark.parametrize("position", [0, 5, 20, -1])
def test_a_changed_byte_is_detected(position: int) -> None:
    stored = bytearray(KEYRING.encrypt("Uwase Marie", COLUMN))
    stored[position] ^= 0x01
    with pytest.raises(DecryptionError):
        KEYRING.decrypt(bytes(stored), COLUMN)


def test_a_value_moved_to_another_column_does_not_decrypt() -> None:
    stored = KEYRING.encrypt("RW-1042", "admission.patient_number")
    with pytest.raises(DecryptionError, match="does not authenticate"):
        KEYRING.decrypt(stored, "admission.patient_name")


def test_another_key_cannot_read_a_value() -> None:
    stored = KEYRING.encrypt("Uwase Marie", COLUMN)
    with pytest.raises(DecryptionError):
        Keyring.from_master(bytes(32)).decrypt(stored, COLUMN)


def test_truncated_values_are_refused() -> None:
    with pytest.raises(DecryptionError, match="format"):
        KEYRING.decrypt(b"\x01" + bytes(20), COLUMN)


def test_the_subkeys_differ_from_each_other_and_from_the_master() -> None:
    master = bytes(range(32))
    assert len({master, KEYRING.encryption_key, KEYRING.index_key}) == 3


def test_the_index_is_deterministic_within_a_facility_and_differs_across_facilities() -> None:
    here, there = uuid.uuid4(), uuid.uuid4()
    first = KEYRING.blind_index(here, "RW-1042")
    assert first == KEYRING.blind_index(here, "RW-1042")
    assert len(first) == 64 and int(first, 16) >= 0
    assert first != KEYRING.blind_index(there, "RW-1042")
    assert first != KEYRING.blind_index(here, "RW-1043")
    assert first != Keyring.from_master(bytes(32)).blind_index(here, "RW-1042")


def test_the_index_ignores_case_and_spacing() -> None:
    facility = uuid.uuid4()
    expected = KEYRING.blind_index(facility, "RW 1042-A")
    for variant in (" rw 1042-a ", "RW  1042-A", "rw\t1042-a"):
        assert KEYRING.blind_index(facility, variant) == expected
    assert KEYRING.blind_index(facility, "RW1042-A") != expected  # spaces inside still count


def test_normalisation() -> None:
    assert normalise_patient_number("  rw   12\t3 ") == "RW 12 3"


def test_generated_keys_decode_to_32_bytes_and_are_random() -> None:
    keys = {crypto.generate_key() for _ in range(10)}
    assert len(keys) == 10
    assert all(len(crypto.decode_key(k)) == 32 for k in keys)


@pytest.mark.parametrize("text", ["", "short", "A" * 44, "!!!!" * 11])
def test_malformed_keys_are_refused(text: str) -> None:
    with pytest.raises(FieldKeyError):
        crypto.decode_key(text)


def test_a_missing_key_is_reported_by_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(crypto.KEY_VARIABLE, raising=False)
    monkeypatch.setattr("robby.config.ENV_FILE", tmp_path / "missing.env")
    with pytest.raises(FieldKeyError, match="ROBBY_FIELD_KEY is not set"):
        crypto.key_from_environment()


def test_the_key_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    key = crypto.generate_key()
    monkeypatch.setenv(crypto.KEY_VARIABLE, key)
    assert crypto.key_from_environment() == crypto.decode_key(key)
