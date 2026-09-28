"""Shared utilities.

Infrastructure used across layers, with no dependency on any intelligence layer:

* :mod:`focus_engine.utils.clock` — injectable time source, so temporal logic is
  testable without sleeping.
* :mod:`focus_engine.utils.determinism` — seed derivation, stable seeding, and content
  hashing for reproducibility.
* :mod:`focus_engine.utils.logging` — logging with mandatory credential redaction.
"""

from __future__ import annotations

from focus_engine.utils.clock import Clock, FixedClock, SystemClock
from focus_engine.utils.determinism import (
    DIGEST_SIZE_BYTES,
    canonical_json,
    content_digest,
    derive_seed,
    digest_iterable,
    seed_everything,
    stable_hash_int,
)
from focus_engine.utils.logging import (
    REDACTED,
    RedactingFilter,
    configure_logging,
    get_logger,
    pseudonymous_ref,
)

__all__ = [
    "DIGEST_SIZE_BYTES",
    "REDACTED",
    "Clock",
    "FixedClock",
    "RedactingFilter",
    "SystemClock",
    "canonical_json",
    "configure_logging",
    "content_digest",
    "derive_seed",
    "digest_iterable",
    "get_logger",
    "pseudonymous_ref",
    "seed_everything",
    "stable_hash_int",
]
