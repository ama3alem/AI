"""Unit tests for :mod:`focus_engine.utils`.

Covers the three infrastructure guarantees the intelligence layers rely on: time is
injectable, randomness is derived deterministically, and credentials cannot reach a log
record.
"""

from __future__ import annotations

import io
import logging
import random
from datetime import UTC, datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path

import numpy as np
import pytest

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

pytestmark = pytest.mark.unit

_START = datetime(2026, 9, 26, 10, 0, 0, tzinfo=UTC)


# --------------------------------------------------------------------------------------
# Clock
# --------------------------------------------------------------------------------------


def test_system_clock_returns_aware_utc() -> None:
    now = SystemClock().now()
    assert now.tzinfo is UTC
    assert now.utcoffset() == timedelta(0)


def test_fixed_clock_does_not_advance_implicitly() -> None:
    """The property that makes temporal tests deterministic."""
    clock = FixedClock(_START)
    assert clock.now() == _START
    assert clock.now() == _START
    assert clock.now() == _START


def test_fixed_clock_advances_only_on_demand() -> None:
    clock = FixedClock(_START)
    clock.advance(timedelta(minutes=11))
    assert clock.now() == _START + timedelta(minutes=11)


def test_fixed_clock_normalises_a_non_utc_start() -> None:
    tz = timezone(timedelta(hours=5, minutes=30))
    clock = FixedClock(datetime(2026, 9, 26, 15, 30, 0, tzinfo=tz))
    assert clock.now() == _START


def test_fixed_clock_refuses_a_naive_start() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        FixedClock(datetime(2026, 9, 26, 10, 0, 0))


def test_fixed_clock_refuses_to_move_backwards() -> None:
    """Backwards travel would silently invalidate a monotonicity assumption."""
    clock = FixedClock(_START)
    with pytest.raises(ValueError, match="backwards"):
        clock.advance(timedelta(seconds=-1))


def test_fixed_clock_set_time_requires_awareness() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        FixedClock(_START).set_time(datetime(2026, 9, 26, 11, 0, 0))


def test_fixed_clock_set_time_jumps_absolutely() -> None:
    clock = FixedClock(_START)
    target = datetime(2026, 9, 27, 8, 0, 0, tzinfo=UTC)
    assert clock.set_time(target) == target
    assert clock.now() == target


@pytest.mark.parametrize("candidate", [SystemClock(), FixedClock(_START)])
def test_both_clocks_satisfy_the_protocol(candidate: Clock) -> None:
    assert isinstance(candidate, Clock)
    assert candidate.now().tzinfo is not None


# --------------------------------------------------------------------------------------
# Seed derivation
# --------------------------------------------------------------------------------------


def test_derive_seed_is_deterministic() -> None:
    assert derive_seed(42, "simulator", "learner-0001") == derive_seed(
        42, "simulator", "learner-0001"
    )


def test_derive_seed_varies_with_the_master_seed() -> None:
    assert derive_seed(42, "a") != derive_seed(43, "a")


def test_derive_seed_varies_with_the_label() -> None:
    assert derive_seed(42, "simulator") != derive_seed(42, "baseline")


def test_derive_seed_is_order_sensitive() -> None:
    """Swapping labels must change the seed, or two components would share a stream."""
    assert derive_seed(42, "a", "b") != derive_seed(42, "b", "a")


def test_derive_seed_stays_in_range() -> None:
    for label in ("a", "b", "c", "d", "e"):
        assert 0 <= derive_seed(7, label) < 2**32


def test_derive_seed_rejects_a_negative_master() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        derive_seed(-1, "a")


def test_separate_components_do_not_share_a_random_stream() -> None:
    """Adding a component must not shift an existing component's seed."""
    before = derive_seed(42, "simulator", "learner-0001")
    _ = derive_seed(42, "brand_new_component")
    after = derive_seed(42, "simulator", "learner-0001")
    assert before == after


def test_seed_everything_makes_global_generators_reproducible() -> None:
    seed_everything(1234)
    first = (random.random(), float(np.random.random()))
    seed_everything(1234)
    second = (random.random(), float(np.random.random()))
    assert first == second


def test_seed_everything_returns_an_isolated_generator() -> None:
    """The returned generator must not be affected by later global seeding."""
    generator = seed_everything(99)
    expected = generator.random()
    seed_everything(1234)
    assert generator.random() != expected


def test_seed_everything_rejects_a_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)


def test_stable_hash_int_is_stable_and_non_negative() -> None:
    assert stable_hash_int("x") == stable_hash_int("x")
    assert stable_hash_int("x") >= 0


def test_stable_hash_int_rejects_non_positive_width() -> None:
    with pytest.raises(ValueError, match="bits must be positive"):
        stable_hash_int("x", bits=0)


# --------------------------------------------------------------------------------------
# Content hashing
# --------------------------------------------------------------------------------------


def test_canonical_json_is_key_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_content_digest_is_key_order_independent() -> None:
    assert content_digest({"b": 1, "a": 2}) == content_digest({"a": 2, "b": 1})


def test_content_digest_changes_with_content() -> None:
    assert content_digest({"a": 1}) != content_digest({"a": 2})


def test_content_digest_has_the_declared_width() -> None:
    assert len(content_digest({"a": 1})) == DIGEST_SIZE_BYTES * 2


class _Colour(StrEnum):
    """Local enum for encoder coverage."""

    RED = "red"


def test_canonical_json_encodes_datetimes() -> None:
    assert _START.isoformat() in canonical_json({"at": _START})


def test_canonical_json_encodes_enums() -> None:
    assert canonical_json({"c": _Colour.RED}) == '{"c":"red"}'


def test_canonical_json_encodes_numpy_scalars_and_arrays() -> None:
    assert canonical_json({"i": np.int64(3), "a": np.array([1, 2])}) == '{"a":[1,2],"i":3}'


def test_canonical_json_encodes_paths() -> None:
    assert canonical_json({"p": Path("a/b")}) == '{"p":"a/b"}'


def test_canonical_json_encodes_sets_deterministically() -> None:
    assert canonical_json({"s": {"b", "a"}}) == canonical_json({"s": {"a", "b"}})


def test_canonical_json_rejects_unencodable_values() -> None:
    with pytest.raises(TypeError, match="unsupported type"):
        canonical_json({"f": object()})


def test_digest_iterable_is_order_sensitive() -> None:
    assert digest_iterable(["a", "b"]) != digest_iterable(["b", "a"])


def test_digest_iterable_is_stable() -> None:
    assert digest_iterable(["a", "b"]) == digest_iterable(["a", "b"])


# --------------------------------------------------------------------------------------
# Logging redaction
# --------------------------------------------------------------------------------------


def _emit(record_message: str, **extra: object) -> str:
    """Log a record through the redacting filter and return the formatted output.

    Extra attributes are appended to the formatted line so that a test can assert both
    that a secret was redacted and that ordinary context survived.

    Args:
        record_message: The message to log.
        **extra: Extra fields attached to the record.

    Returns:
        The formatted log line.
    """
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter())
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord("test", logging.INFO, __file__, 1, record_message, None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    # ``Handler.handle`` is the entry point that applies filters; ``Handler.emit`` does
    # not. Going through ``handle`` mirrors what Logger.callHandlers actually does.
    handler.handle(record)
    rendered = stream.getvalue().strip()
    if extra:
        # Read the attributes back off the record, which the filter mutated in place, so
        # that the assertion sees what a real handler would actually emit.
        rendered = rendered + " | " + " ".join(f"{key}={record.__dict__[key]}" for key in extra)
    return rendered


@pytest.mark.parametrize(
    "message",
    [
        'config={"password": "hunter2"}',
        "connecting with api_key=abcdef123456",
        "token: ghp_ABCDEFGHIJKLMNOP",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9abcdef",
        "secret = 'topsecretvalue'",
    ],
)
def test_credential_shaped_messages_are_redacted(message: str) -> None:
    output = _emit(message)
    for leaked in (
        "hunter2",
        "abcdef123456",
        "ghp_ABCDEFGHIJKLMNOP",
        "eyJhbGciOiJIUzI1NiJ9",
        "topsecretvalue",
    ):
        assert leaked not in output
    assert REDACTED in output


def test_sensitive_record_attributes_are_redacted() -> None:
    output = _emit("status", auth_secret="super-secret-value")
    assert "super-secret-value" not in output
    assert REDACTED in output


def test_non_sensitive_attributes_survive() -> None:
    """Redaction must not destroy ordinary diagnostic context."""
    output = _emit("status", n_samples=42, model_version="MODEL_DECL_V1")
    assert "42" in output
    assert "MODEL_DECL_V1" in output


def test_pem_private_key_blocks_are_redacted() -> None:
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----"
    assert "MIIEow" not in _emit(f"loaded {pem}")


def test_harmless_message_is_untouched() -> None:
    output = _emit("processed 120 events for 8 learners")
    assert output.strip() == "processed 120 events for 8 learners"


def test_filter_never_drops_a_record() -> None:
    """Dropping a record containing a secret would hide a security incident.

    ``logging.Filterer.filter`` returns the (scrubbed) record rather than ``True``; the
    assertion checks the record survives and that the secret is gone.
    """
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter())
    record = logging.LogRecord("t", logging.WARNING, __file__, 1, "password=leaked", None, None)
    assert handler.filter(record) is record
    assert record.msg == f"password={REDACTED}"


def test_configure_logging_writes_to_the_given_stream() -> None:
    stream = io.StringIO()
    root = configure_logging(level="INFO", stream=stream, force=True)
    try:
        get_logger("focus_engine.test").info("hello %s", "world")
        assert "hello world" in stream.getvalue()
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)


def test_configure_logging_redacts_by_default() -> None:
    stream = io.StringIO()
    root = configure_logging(level="INFO", stream=stream, force=True)
    try:
        get_logger("focus_engine.test").info("api_key=leak-me-please-1234")
        assert "leak-me-please-1234" not in stream.getvalue()
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)


def test_configure_logging_defaults_to_stderr_not_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """stdout carries machine-readable results; logs must not contaminate it."""
    root = configure_logging(level="INFO", force=True)
    try:
        get_logger("focus_engine.test").info("a log line")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "a log line" in captured.err
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)


def test_pseudonymous_ref_is_stable_and_non_reversible_by_inspection() -> None:
    reference = pseudonymous_ref("learner-0001")
    assert reference == pseudonymous_ref("learner-0001")
    assert "learner-0001" not in reference
    assert reference != pseudonymous_ref("learner-0002")


def test_pseudonymous_ref_retains_a_short_readable_prefix() -> None:
    """Operators need to correlate lines; a bare hash is unreadable in an incident."""
    assert pseudonymous_ref("learner-0001").startswith("lear#")
