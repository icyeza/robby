"""The event log: `events.append(...)` is its only writer; `events.verify(...)` checks the chain."""

from robby.events.log import GENESIS_HASH, VerifyResult, append, verify

__all__ = ["GENESIS_HASH", "VerifyResult", "append", "verify"]
