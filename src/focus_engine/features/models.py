"""Feature data model.

A feature is a named, versioned, deterministic function from (events, context) to a
typed value or an explicit ``INSUFFICIENT_DATA`` marker. The registry is the single
source of truth for what a feature name means; changing a calculation mints a new
version rather than editing history.

Three constraints shape this module:

* **Determinism is total.** The same events, context, and feature-set version must
  produce identical output. No wall-clock reads, no randomness, no silent fallbacks.
* **Missing data is not fabricated.** A feature that cannot be computed from the
  available inputs returns ``INSUFFICIENT_DATA`` with a reason, never an invented value
  or a misleading default. Downstream layers are required to handle the marker rather
  than guessing. This includes the subtler cases: a value that is *absent* is not
  replaced by a plausible constant, and a value that falls outside its documented valid
  range is reported as a defect rather than quietly clamped into range.
* **No future information.** A feature computed at time ``t`` must not depend on events
  with timestamps greater than ``t``. This holds even when a caller supplies a longer
  event sequence than the reference time allows, so the invariant is enforced in the
  computation rather than trusted to the caller.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from focus_engine.context.models import ContextKey
from focus_engine.schemas.primitives import DataOrigin, non_empty_text
from focus_engine.schemas.versioning import FeatureSetVersion

__all__ = [
    "FEATURE_SET_V1",
    "INSUFFICIENT_DATA_MARKER",
    "FeatureAvailability",
    "FeatureCategory",
    "FeatureName",
    "FeatureSpec",
    "FeatureValue",
    "FeatureValueType",
    "TypedFeatureValue",
]


INSUFFICIENT_DATA_MARKER: Final[str] = "INSUFFICIENT_DATA"
"""The literal string returned when a feature cannot be computed from available inputs.

Downstream code must compare against this value explicitly. A feature that returns it
is not a bug; it is a statement that the evidence required for the computation is not
present, together with a reason saying which evidence was missing.
"""


_SPEC_VERSION_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d+\.\d+\.\d+$")


class FeatureName(StrEnum):
    """Closed vocabulary of feature names.

    A feature name is a contract. Adding a name here is a commitment that the
    corresponding calculation exists and is versioned; removing one is a commitment that
    no code path depends on it. Open-ended string names would permit a typo to silently
    produce ``INSUFFICIENT_DATA`` everywhere, which is exactly the failure mode this
    taxonomy exists to prevent.
    """

    SESSION_ELAPSED_SECONDS = "session_elapsed_seconds"
    SESSION_POSITION = "session_position"
    SESSION_EVENT_COUNT = "session_event_count"

    CONTENT_DIFFICULTY = "content_difficulty"
    CONTENT_TYPE_ENCODED = "content_type_encoded"

    PERFORMANCE_RECENT_ACCURACY = "performance_recent_accuracy"
    PERFORMANCE_RECENT_RESPONSE_SECONDS = "performance_recent_response_seconds"
    PERFORMANCE_ACCURACY_TREND = "performance_accuracy_trend"
    PERFORMANCE_RESPONSE_TREND = "performance_response_trend"

    INTERVENTION_COUNT = "intervention_count"
    INTERVENTION_COOLDOWN_ACTIVE = "intervention_cooldown_active"
    INTERVENTION_SECONDS_SINCE_LAST = "intervention_seconds_since_last"

    TRAJECTORY_ENCODED = "trajectory_encoded"
    BASELINE_MATURITY_ENCODED = "baseline_maturity_encoded"


class FeatureCategory(StrEnum):
    """Grouping for documentation and for consumers that operate on whole categories."""

    SESSION = "session"
    CONTENT = "content"
    PERFORMANCE = "performance"
    INTERVENTION = "intervention"
    STATE = "state"


class FeatureValueType(StrEnum):
    """The type a feature produces.

    Declared per feature so that a consumer can reject an unexpected value shape before
    it silently trains on a string.
    """

    FLOAT = "float"
    INT = "int"
    STR = "str"


class FeatureAvailability(StrEnum):
    """Whether a feature was computed, and from what.

    ``SYNTHETIC_ONLY`` is distinct from ``AVAILABLE`` rather than a flag on it, so that a
    feature vector derived entirely from simulator output cannot be read as an
    observation. A vector built from a mix of real and synthetic events is ``AVAILABLE``:
    the mixed provenance is recorded separately and losslessly in ``origins`` rather than
    being flattened into a two-valued flag here.
    """

    AVAILABLE = "available"
    """Computed from at least one real event."""

    INSUFFICIENT_DATA = "insufficient_data"
    """Not computed. The required evidence was absent or the inputs were invalid."""

    SYNTHETIC_ONLY = "synthetic_only"
    """Computed, but every contributing event was synthetic."""


FEATURE_SET_V1: Final[FeatureSetVersion] = "FEATURE_SET_V1"
"""The initial feature set.

When a calculation changes, the new version becomes ``FEATURE_SET_V2`` and the old
remains on record. Results computed under ``V1`` stay interpretable; they are not
silently reinterpreted by the new code.

The value is a plain string because :data:`FeatureSetVersion` is a validated string
alias rather than a class. The grammar is enforced at the model boundary, where an
externally supplied version is checked, not when a known-good constant is defined.
"""


class FeatureSpec(BaseModel):
    """Definition of one feature for the registry.

    The spec is the machine-readable documentation of what a feature is, where its data
    comes from, and which context keys it requires. It is not the calculation itself; the
    engine owns the implementation. Because the spec records the definition, a change to
    a calculation is visible as a version change rather than as a silent reinterpretation
    of historical vectors.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: FeatureName
    """The stable identifier used in downstream code."""

    version: str
    """Feature-specific version, incremented when the calculation or the required
    context changes. The registry refuses to re-register a version under different
    definitions, so this is a history, not a label."""

    category: FeatureCategory
    """Grouping for documentation and bulk operations."""

    data_source: non_empty_text
    """Where the data comes from, e.g. "session.started_at and context.reference_time"."""

    definition: non_empty_text
    """What the feature measures, stated so that an analyst can audit the calculation
    without reading its code."""

    required_context_keys: frozenset[ContextKey]
    """Context keys that must be present and non-null for computation. If any are
    missing, the engine returns ``INSUFFICIENT_DATA`` rather than proceeding."""

    value_type: FeatureValueType
    """The type of the computed value."""

    valid_range: tuple[float, float] | None = None
    """For numeric features, the inclusive range of valid outputs. A computed value
    outside this range is reported as ``INSUFFICIENT_DATA`` with a reason, because it
    indicates a defect in the calculation or in the input data. Clamping it into range
    would hide that defect behind a plausible-looking number."""

    @field_validator("version")
    @classmethod
    def _validate_version_format(cls, value: str) -> str:
        """Require a three-part numeric spec version.

        Args:
            value: Candidate spec version.

        Returns:
            The validated version.

        Raises:
            ValueError: If the version is not of the form ``MAJOR.MINOR.PATCH``.
        """
        if not _SPEC_VERSION_PATTERN.match(value):
            raise ValueError(f"feature spec version must look like '1.0.0'; got {value!r}")
        return value

    @model_validator(mode="after")
    def _validate_range_order(self) -> FeatureSpec:
        """Check that a declared range is ordered and applied to a numeric feature.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If the lower bound exceeds the upper bound, or if a range is
                declared for a string-valued feature.
        """
        if self.valid_range is not None:
            low, high = self.valid_range
            if low > high:
                raise ValueError(
                    f"valid_range lower bound {low} exceeds upper bound {high} "
                    f"for feature {self.name.value}"
                )
            if self.value_type is FeatureValueType.STR:
                raise ValueError(
                    f"feature {self.name.value} is string-valued and must not declare "
                    "a numeric valid_range"
                )
        return self


class TypedFeatureValue(BaseModel):
    """A single feature value with full provenance.

    Every field is filled by the engine. The consumer never has to infer the value's
    meaning or its reliability from context, and cannot accidentally treat a marker
    string as a measurement because the model refuses an inconsistent pair.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: FeatureName
    """Which feature this is."""

    value: int | float | str
    """The computed value. ``INSUFFICIENT_DATA_MARKER`` appears only when
    ``availability`` is ``INSUFFICIENT_DATA``."""

    availability: FeatureAvailability
    """Whether the value was computed, and from what."""

    reason: str | None = None
    """Required when the value is not available, explaining what was missing. ``None``
    when the value was computed."""

    computed_at: datetime
    """The reference time of the computation: the timestamp of the last event used, not
    the wall clock."""

    feature_set_version: FeatureSetVersion
    """Which feature-set version computed this value."""

    spec_version: str
    """Which feature-spec version computed this value."""

    @field_validator("computed_at")
    @classmethod
    def _require_aware_timestamp(cls, value: datetime) -> datetime:
        """Reject a naive reference time.

        Args:
            value: Candidate timestamp.

        Returns:
            The validated timestamp.

        Raises:
            ValueError: If the timestamp carries no timezone.
        """
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("computed_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_availability_matches_value(self) -> TypedFeatureValue:
        """Keep the marker, the availability, and the reason mutually consistent.

        Without this, a computed value could be published as ``INSUFFICIENT_DATA``, or a
        marker string could be consumed downstream as if it were a number. Both are the
        kind of error that is invisible until it corrupts a result.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If the marker, the availability, and the reason disagree.
        """
        marked = self.value == INSUFFICIENT_DATA_MARKER
        if self.availability is FeatureAvailability.INSUFFICIENT_DATA:
            if not marked:
                raise ValueError(
                    f"feature {self.name.value} is marked INSUFFICIENT_DATA but carries "
                    f"the value {self.value!r}"
                )
            if not self.reason:
                raise ValueError(
                    f"feature {self.name.value} is INSUFFICIENT_DATA and must state a reason"
                )
        else:
            if marked:
                raise ValueError(
                    f"feature {self.name.value} carries the {INSUFFICIENT_DATA_MARKER} "
                    f"marker but its availability is {self.availability.value!r}"
                )
            if self.reason is not None:
                raise ValueError(
                    f"feature {self.name.value} is {self.availability.value!r} and must "
                    "not carry an insufficiency reason"
                )
        return self


class FeatureValue(BaseModel):
    """A feature vector for one point in time.

    The vector is frozen, versioned, and reproducible. Downstream layers receive this
    structure rather than a dict, so the presence of a value is a checked statement
    rather than a ``None`` hidden in a field. The identifiers are optional because a
    feature vector can legitimately be computed for a stream whose session start was
    never observed; substituting a placeholder identity would invent a fact about who
    the vector describes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_set_version: FeatureSetVersion
    """The version of the feature set that produced this vector."""

    computed_at: datetime
    """The reference time. Matches the context's ``reference_time``."""

    values: tuple[TypedFeatureValue, ...]
    """One entry per feature, in the order defined by the registry."""

    learner_id: str | None = None
    """Pseudonymous learner identifier, or ``None`` when no session was observed."""

    session_id: str | None = None
    """Pseudonymous session identifier, or ``None`` when no session was observed."""

    origins: frozenset[DataOrigin] = Field(default_factory=frozenset)
    """Data origins of the contributing events, recorded losslessly. A vector carrying
    both ``REAL`` and ``SYNTHETIC`` is reported as ``MIXED`` by :attr:`origin_label`
    rather than being forced into a real-or-synthetic choice."""

    @field_validator("computed_at")
    @classmethod
    def _require_aware_timestamp(cls, value: datetime) -> datetime:
        """Reject a naive reference time.

        Args:
            value: Candidate timestamp.

        Returns:
            The validated timestamp.

        Raises:
            ValueError: If the timestamp carries no timezone.
        """
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("computed_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_vector(self) -> FeatureValue:
        """Check structural invariants of the vector.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If a feature name is repeated, or if a feature claims to be
                available while the vector records purely synthetic origins.
        """
        names = [entry.name for entry in self.values]
        duplicates = sorted({name.value for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"feature vector repeats feature names: {duplicates}")

        if self.origins == frozenset({DataOrigin.SYNTHETIC}):
            mislabelled = sorted(
                entry.name.value
                for entry in self.values
                if entry.availability is FeatureAvailability.AVAILABLE
            )
            if mislabelled:
                raise ValueError(
                    "vector origins are purely synthetic, so no feature may be reported "
                    f"as available: {mislabelled}"
                )
        return self

    @property
    def as_dict(self) -> dict[str, int | float | str]:
        """Render the feature vector as a name-to-value mapping.

        Returns:
            A dictionary suitable for logging, serialization, or model input. Missing
            features appear as ``INSUFFICIENT_DATA_MARKER``, so a consumer cannot
            mistake an absent feature for a computed zero.
        """
        return {entry.name.value: entry.value for entry in self.values}

    @property
    def available_count(self) -> int:
        """Count of features that were successfully computed.

        Returns:
            The number of features with availability ``AVAILABLE`` or ``SYNTHETIC_ONLY``.
        """
        return sum(
            1
            for entry in self.values
            if entry.availability
            in (FeatureAvailability.AVAILABLE, FeatureAvailability.SYNTHETIC_ONLY)
        )

    @property
    def insufficient_count(self) -> int:
        """Count of features that could not be computed.

        Returns:
            The number of features with availability ``INSUFFICIENT_DATA``.
        """
        return sum(
            1
            for entry in self.values
            if entry.availability is FeatureAvailability.INSUFFICIENT_DATA
        )

    @property
    def is_synthetic_only(self) -> bool:
        """Whether every contributing event was synthetic.

        Returns:
            ``True`` only when the recorded origins are exactly synthetic.
        """
        return self.origins == frozenset({DataOrigin.SYNTHETIC})

    @property
    def origin_label(self) -> str:
        """A single-word provenance label for logs and reports.

        Returns:
            ``"synthetic"``, ``"real"``, ``"mixed"``, or ``"unknown"`` for an empty
            origin set. A mixed vector is labelled ``"mixed"`` rather than ``"real"``,
            because a vector that is partly derived from the simulator has not been
            established as an observation.
        """
        if self.origins == frozenset({DataOrigin.SYNTHETIC}):
            return DataOrigin.SYNTHETIC.value
        if self.origins == frozenset({DataOrigin.REAL}):
            return DataOrigin.REAL.value
        if DataOrigin.REAL in self.origins and DataOrigin.SYNTHETIC in self.origins:
            return "mixed"
        return "unknown"

    def to_summary_dict(self) -> dict[str, object]:
        """Render a JSON-serialisable summary for logs and debugging.

        Returns:
            A dictionary with the version coordinates, the provenance label, the
            availability counts, and one entry per feature holding its value or the
            reason it is unavailable.
        """
        return {
            "feature_set_version": self.feature_set_version,
            "computed_at": self.computed_at.isoformat(),
            "learner_id": self.learner_id,
            "session_id": self.session_id,
            "origins": sorted(origin.value for origin in self.origins),
            "origin_label": self.origin_label,
            "available_count": self.available_count,
            "insufficient_count": self.insufficient_count,
            "features": {
                entry.name.value: (
                    INSUFFICIENT_DATA_MARKER
                    if entry.availability is FeatureAvailability.INSUFFICIENT_DATA
                    else entry.value
                )
                for entry in self.values
            },
            "reasons": {
                entry.name.value: entry.reason for entry in self.values if entry.reason is not None
            },
        }
