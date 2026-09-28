"""Shared schema primitives.

Every identifier, timestamp, provenance label, and confidence level used anywhere in
the engine is defined here exactly once. Layers must import these types rather than
re-declaring them, so that a malformed value is rejected at the boundary regardless of
which entry point produced it.

Design rules enforced by this module:

* **Pseudonymous identifiers only.** No name, email, phone, device fingerprint, or
  other directly identifying value is representable.
* **Timezone-aware timestamps only.** Naive datetimes are rejected; the engine never
  has to guess an offset.
* **Explicit provenance.** Every derived value declares whether it is observed,
  synthetic, a proxy, a ground truth, or a model prediction. A synthetic label is
  never representable as a ground truth.
* **Immutability.** These models are frozen. A value that entered the audit trail
  cannot be edited in place afterwards.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Final

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field

__all__ = [
    "BEHAVIORAL_ENGAGEMENT_STATES",
    "ConfidenceLevel",
    "DataOrigin",
    "EventId",
    "InferenceBasis",
    "InterventionId",
    "LearnerId",
    "Probability",
    "Provenance",
    "SessionId",
    "SyntheticDataStamp",
    "Timestamp",
    "UnitInterval",
    "UtcTimestamp",
    "non_empty_text",
]

# --------------------------------------------------------------------------------------
# Identifiers
# --------------------------------------------------------------------------------------

#: Identifiers are opaque, pseudonymous tokens. The character class deliberately
#: excludes ``@``, ``.``, ``/`` and whitespace so that an email address, a domain name,
#: or a filesystem path cannot be smuggled in as a learner identifier.
_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]+$")

_MIN_ID_LENGTH: Final[int] = 8
_MAX_ID_LENGTH: Final[int] = 128


def _validate_pseudonymous_id(value: str) -> str:
    """Validate an opaque pseudonymous identifier.

    Args:
        value: Candidate identifier.

    Returns:
        The validated identifier.

    Raises:
        ValueError: If the value is too short, too long, or contains any character
            outside the opaque-token character class.
    """
    if not isinstance(value, str):  # pragma: no cover - pydantic coerces before this
        raise ValueError("identifier must be a string")
    if len(value) < _MIN_ID_LENGTH:
        raise ValueError(
            f"identifier must be at least {_MIN_ID_LENGTH} characters; "
            f"got {len(value)}. Short identifiers risk re-identification by enumeration."
        )
    if len(value) > _MAX_ID_LENGTH:
        raise ValueError(
            f"identifier must be at most {_MAX_ID_LENGTH} characters; got {len(value)}"
        )
    if not _ID_PATTERN.match(value):
        raise ValueError(
            "identifier must contain only letters, digits, hyphen, and underscore. "
            "Directly identifying values (emails, names, paths) are not permitted."
        )
    return value


PseudonymousId = Annotated[
    str,
    BeforeValidator(lambda v: v.strip() if isinstance(v, str) else v),
    AfterValidator(_validate_pseudonymous_id),
    Field(min_length=_MIN_ID_LENGTH, max_length=_MAX_ID_LENGTH),
]

#: A learner, identified pseudonymously. Never a name, email, or device identifier.
LearnerId = Annotated[PseudonymousId, Field(description="Pseudonymous learner identifier")]
#: A learning session.
SessionId = Annotated[PseudonymousId, Field(description="Pseudonymous session identifier")]
#: A single behavioural event.
EventId = Annotated[PseudonymousId, Field(description="Unique behavioural event identifier")]
#: A delivered intervention.
InterventionId = Annotated[PseudonymousId, Field(description="Intervention identifier")]


# --------------------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------------------


def _require_aware_utc(value: datetime) -> datetime:
    """Reject naive datetimes and normalise aware datetimes to UTC.

    Args:
        value: Candidate datetime.

    Returns:
        The datetime expressed in UTC.

    Raises:
        ValueError: If the datetime carries no timezone information.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(
            "timestamp must be timezone-aware. Naive datetimes are rejected because the "
            "engine must never infer an offset; supply an explicit UTC or offset-aware value."
        )
    return value.astimezone(UTC)


#: A timezone-aware instant, normalised to UTC on construction.
Timestamp = Annotated[datetime, AfterValidator(_require_aware_utc)]

#: Alias kept for call sites where the UTC normalisation is the salient property.
UtcTimestamp = Timestamp


def utc_now() -> datetime:
    """Return the current instant as a timezone-aware UTC datetime.

    Returns:
        The current UTC instant.
    """
    return datetime.now(UTC)


# --------------------------------------------------------------------------------------
# Numeric ranges
# --------------------------------------------------------------------------------------


def _validate_unit_interval(value: float) -> float:
    """Validate that a float lies within the closed unit interval.

    Args:
        value: Candidate probability.

    Returns:
        The validated probability.

    Raises:
        ValueError: If the value is outside ``[0.0, 1.0]`` or is not finite.
    """
    if value != value:  # NaN check without a numpy dependency in the schema layer
        raise ValueError("probability must not be NaN")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"probability must lie within [0.0, 1.0]; got {value}")
    return value


#: A probability in ``[0.0, 1.0]``.
Probability = Annotated[float, AfterValidator(_validate_unit_interval)]
#: Alias for the same closed unit interval.
UnitInterval = Probability


# --------------------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------------------


def _validate_non_empty_text(value: str) -> str:
    """Validate that a string is non-empty after stripping whitespace.

    Args:
        value: Candidate text.

    Returns:
        The stripped text.

    Raises:
        ValueError: If the text is empty or contains only whitespace.
    """
    if not value.strip():
        raise ValueError("value must not be empty or whitespace-only")
    return value.strip()


#: A non-empty, whitespace-stripped string.
non_empty_text = Annotated[str, AfterValidator(_validate_non_empty_text)]


# --------------------------------------------------------------------------------------
# Provenance and epistemic status
# --------------------------------------------------------------------------------------


class DataOrigin(StrEnum):
    """Whether a value originated from real observation or from simulation.

    A synthetic origin is a permanent, non-revisable property of the value. It travels
    with the value so that a synthetic artifact can never be reported as a real
    measurement, no matter how many transformations it passes through.
    """

    REAL = "real"
    """Captured from a real deployment."""

    SYNTHETIC = "synthetic"
    """Produced by the synthetic learner simulator. NOT REAL STUDENT DATA."""


class Provenance(StrEnum):
    """How a value came to exist.

    This is the mechanism that prevents the central anti-hallucination failure mode of
    behavioural analytics: presenting an inferred or simulated quantity as an observed
    fact. Every derived record carries one of these.
    """

    OBSERVED = "observed"
    """Measured directly from a behavioural event stream."""

    SYNTHETIC_LABEL = "synthetic_label"
    """Generated by the simulator from its own generative parameters.

    A synthetic label is self-referential: it encodes the simulator's assumption, not a
    measurement. It is never ground truth and never evidence about a real learner.
    """

    PROXY_LABEL = "proxy_label"
    """Constructed from observable proxies because the construct of interest is not
    directly measurable. Valid only to the extent the proxy assumption holds."""

    GROUND_TRUTH = "ground_truth"
    """Independently established by a measurement or design external to the model.

    Requires an identified external source. Absent such a source, this value must not
    be used. In this repository no ground truth exists yet."""

    MODEL_PREDICTION = "model_prediction"
    """Output of a trained model. Carries uncertainty and is never a measurement."""


class ConfidenceLevel(StrEnum):
    """Epistemic status of a prediction or estimate.

    ``UNKNOWN`` and ``INSUFFICIENT_DATA`` are first-class outcomes, not error states.
    The engine is required to return them rather than emit a fabricated number.
    """

    HIGH = "high"
    """Sufficient evidence and a well-supported estimate."""

    MEDIUM = "medium"
    """Usable evidence with material caveats."""

    LOW = "low"
    """Some evidence; the estimate should not drive an action on its own."""

    INSUFFICIENT_DATA = "insufficient_data"
    """Too little evidence to estimate. Distinct from LOW: more data would help."""

    UNKNOWN = "unknown"
    """The engine cannot characterise this case. Must not be substituted with a number."""


class InferenceBasis(StrEnum):
    """Whether an inference rests on population priors or on personal history.

    This exists so that a cold-start population estimate can never be silently
    presented as a personalised one. Both the basis and the baseline maturity travel
    with every prediction.
    """

    POPULATION_PRIOR = "population_prior"
    """No usable personal history. Rests on population-level reference statistics."""

    PARTIAL_PERSONAL = "partial_personal"
    """Some personal history exists but is not yet sufficient for a stable personal model."""

    PERSONAL = "personal"
    """Rests on an established personal baseline."""


class BehavioralEngagementState(StrEnum):
    """Observable behavioural state of a session at a point in time.

    These are descriptions of *interaction behaviour*, not of attention, cognition, or
    mental state. The naming is deliberate: no member of this enum asserts anything
    about what a learner is experiencing internally.
    """

    STABLE = "stable"
    """Behaviour is consistent with the learner's own recent history."""

    SLIGHT_DEVIATION = "slight_deviation"
    """Behaviour departs modestly from the personal baseline, without a trend."""

    INCREASING_DEVIATION = "increasing_deviation"
    """Deviation is growing across consecutive observations."""

    DECLINING = "declining"
    """Sustained multi-signal deterioration relative to the personal baseline."""

    RECOVERING = "recovering"
    """Behaviour is returning toward the personal baseline after a deviation."""

    HIGH_DEVIATION = "high_deviation"
    """Deviation is large and sustained."""

    INSUFFICIENT_DATA = "insufficient_data"
    """Not enough history to characterise the state at all."""


#: The complete, closed set of behavioural engagement states. Nothing outside this tuple
#: may appear in a state field, which keeps downstream switches exhaustive.
BEHAVIORAL_ENGAGEMENT_STATES: Final[tuple[str, ...]] = tuple(
    s.value for s in BehavioralEngagementState
)


# --------------------------------------------------------------------------------------
# Synthetic-data stamping
# --------------------------------------------------------------------------------------


class SyntheticDataStamp(BaseModel):
    """Machine-readable marker attached to every synthetic artifact.

    The stamp travels inside the data, not only in documentation, so that a synthetic
    dataset remains self-identifying after it is copied, concatenated, or loaded from
    disk by a tool that never read the README.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    origin: DataOrigin = DataOrigin.SYNTHETIC
    """Always ``SYNTHETIC`` for this stamp's intended use."""

    warning: str = "SYNTHETIC DATA — NOT REAL STUDENT DATA"
    """The literal warning string, carried in-band."""

    generator: str
    """Identifier of the simulator component and version that produced the data."""

    seed: int
    """Random seed used, so the dataset can be regenerated exactly."""

    def assert_not_real(self) -> None:
        """Raise if this stamp does not declare a synthetic origin.

        Raises:
            ValueError: If ``origin`` is not ``SYNTHETIC``.
        """
        if self.origin is not DataOrigin.SYNTHETIC:
            raise ValueError(f"expected a synthetic origin, got {self.origin!r}")
