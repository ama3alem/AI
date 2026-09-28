"""Feature registry and computation engine.

The registry is the single source of truth for what a feature name means. Registering a
new feature set is an append: once a version has been registered, its definitions can
never be replaced. That is what makes ``FEATURE_SET_V1`` a historical fact rather than a
label that the next edit silently moves, so a vector computed under one version remains
interpretable after the code has changed.

The engine is deterministic. The same events, context, and feature-set version produce
identical output. It never reads the wall clock, never draws from an unseeded RNG, and
never fabricates missing data. When a feature cannot be computed, it returns
``INSUFFICIENT_DATA`` with a reason the caller can use, which covers three distinct
situations that are easy to conflate:

* the required context key is absent, recorded in ``ContextModel.missing``;
* the context key is present but its value is ``None``;
* a value was computed but falls outside the range the spec documents.

The third case is deliberately not clamped. Clamping would turn a defect into a
plausible number, and the defect is the information worth keeping.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from typing import Final, TypeAlias

from focus_engine.configuration.thresholds import FeatureSettings
from focus_engine.context.models import ContextKey, ContextModel
from focus_engine.events.types import EventEnvelope, QuestionAnsweredPayload
from focus_engine.features.models import (
    FEATURE_SET_V1,
    INSUFFICIENT_DATA_MARKER,
    FeatureAvailability,
    FeatureCategory,
    FeatureName,
    FeatureSpec,
    FeatureValue,
    FeatureValueType,
    TypedFeatureValue,
)
from focus_engine.schemas.versioning import FeatureSetVersion, checked_version

__all__ = [
    "CURRENT_FEATURE_SET",
    "DEFAULT_REGISTRY",
    "FEATURE_SPECS_V1",
    "FeatureEngine",
    "FeatureRegistry",
    "UnsupportedFeatureSetError",
    "compute_features",
]


class UnsupportedFeatureSetError(LookupError):
    """Raised when a requested feature-set version is not registered.

    A ``LookupError`` rather than an empty result. Silently returning an empty vector
    would let a typo in a version string propagate as a model trained on no features,
    which is the sort of defect that is discovered long after it has done damage.
    """


#: A computation outcome: the value, or the marker paired with a reason.
_Outcome: TypeAlias = tuple[int | float | str, str | None]

#: Signature every feature calculation implements. A calculation reads the context, may
#: read the event stream, and may read the window configuration; the spec is passed so a
#: calculation can consult its own declared range without a second lookup.
_Computer: TypeAlias = Callable[
    [FeatureSpec, Sequence[EventEnvelope], ContextModel, FeatureSettings], _Outcome
]


def _computed(value: int | float | str) -> _Outcome:
    """Build a successful outcome.

    Args:
        value: The computed value.

    Returns:
        An outcome with no reason attached.
    """
    return (value, None)


def _unavailable(reason: str) -> _Outcome:
    """Build an unavailable outcome.

    Args:
        reason: Why the feature could not be computed.

    Returns:
        An outcome carrying the marker and the reason.
    """
    return (INSUFFICIENT_DATA_MARKER, reason)


def _encode_content_type(content_type: str) -> str:
    """Map a content type to a closed encoding.

    Args:
        content_type: The content type recorded by the context layer.

    Returns:
        One of ``"question"``, ``"video"``, ``"reading"``, ``"other"``.
    """
    normalized = content_type.lower().strip()
    if normalized == "question":
        return "question"
    if normalized == "video":
        return "video"
    if normalized in ("reading", "article", "text"):
        return "reading"
    return "other"


def _encode_trajectory(trajectory: str) -> str:
    """Map a trajectory to a closed encoding.

    Args:
        trajectory: The trajectory recorded by the temporal engine.

    Returns:
        One of ``"improving"``, ``"stable"``, ``"declining"``, ``"other"``.
    """
    normalized = trajectory.lower().strip()
    if normalized in ("improving", "recovering"):
        return "improving"
    if normalized in ("stable", "flat"):
        return "stable"
    if normalized in ("declining", "deteriorating"):
        return "declining"
    return "other"


def _encode_baseline_maturity(maturity: str) -> str:
    """Map a baseline maturity to a closed encoding.

    Args:
        maturity: The maturity recorded by the baseline engine.

    Returns:
        One of ``"new"``, ``"early"``, ``"developing"``, ``"established"``,
        ``"other"``.
    """
    normalized = maturity.lower().strip()
    if normalized in ("new", "cold_start"):
        return "new"
    if normalized == "early":
        return "early"
    if normalized == "developing":
        return "developing"
    if normalized == "established":
        return "established"
    return "other"


def _answer_windows(
    events: Sequence[EventEnvelope],
    context: ContextModel,
    settings: FeatureSettings,
) -> tuple[tuple[QuestionAnsweredPayload, ...], tuple[QuestionAnsweredPayload, ...]]:
    """Split answered questions into the short window and the comparison window.

    Events later than the reference time are dropped here rather than trusted to the
    caller. The phase requirement is that no window may use future information, and a
    property that holds only because callers behave correctly is not a property.

    Args:
        events: The event sequence supplied by the caller.
        context: The context at the reference time.
        settings: Feature configuration supplying the short window width.

    Returns:
        A pair of ``(recent, earlier)`` answer payloads. ``recent`` covers the short
        window ending at the reference time; ``earlier`` covers everything before it.
    """
    window_start = context.reference_time - timedelta(minutes=settings.short_window_minutes)
    recent: list[QuestionAnsweredPayload] = []
    earlier: list[QuestionAnsweredPayload] = []
    for event in events:
        if event.timestamp > context.reference_time:
            continue
        if not isinstance(event.payload, QuestionAnsweredPayload):
            continue
        if event.timestamp >= window_start:
            recent.append(event.payload)
        else:
            earlier.append(event.payload)
    return (tuple(recent), tuple(earlier))


def _session_elapsed(
    _spec: FeatureSpec,
    _events: Sequence[EventEnvelope],
    context: ContextModel,
    _settings: FeatureSettings,
) -> _Outcome:
    """Seconds between session start and the reference time."""
    session = context.session
    if session is None:
        return _unavailable("no session has been observed")
    return _computed(session.elapsed_seconds)


def _session_position(
    _spec: FeatureSpec,
    _events: Sequence[EventEnvelope],
    context: ContextModel,
    _settings: FeatureSettings,
) -> _Outcome:
    """Normalised position within the expected session length."""
    session = context.session
    if session is None:
        return _unavailable("no session has been observed")
    return _computed(session.position_in_session)


def _session_event_count(
    _spec: FeatureSpec,
    _events: Sequence[EventEnvelope],
    context: ContextModel,
    _settings: FeatureSettings,
) -> _Outcome:
    """Number of events observed in this session."""
    session = context.session
    if session is None:
        return _unavailable("no session has been observed")
    return _computed(session.event_count)


def _content_difficulty(
    _spec: FeatureSpec,
    _events: Sequence[EventEnvelope],
    context: ContextModel,
    _settings: FeatureSettings,
) -> _Outcome:
    """Difficulty reported for the current content."""
    difficulty = context.content.difficulty
    if difficulty is None:
        return _unavailable("no difficulty metadata has been reported for the current content")
    return _computed(difficulty)


def _content_type_encoded(
    _spec: FeatureSpec,
    _events: Sequence[EventEnvelope],
    context: ContextModel,
    _settings: FeatureSettings,
) -> _Outcome:
    """Closed encoding of the current content type."""
    content_type = context.content.content_type
    if content_type is None:
        return _unavailable("no content type has been reported")
    return _computed(_encode_content_type(content_type))


def _recent_accuracy(
    _spec: FeatureSpec,
    _events: Sequence[EventEnvelope],
    context: ContextModel,
    _settings: FeatureSettings,
) -> _Outcome:
    """Accuracy over the recent performance window."""
    accuracy = context.performance.recent_accuracy
    if accuracy is None:
        return _unavailable("no question has been answered, so accuracy is undefined")
    return _computed(accuracy)


def _recent_response_seconds(
    _spec: FeatureSpec,
    _events: Sequence[EventEnvelope],
    context: ContextModel,
    _settings: FeatureSettings,
) -> _Outcome:
    """Mean response latency over the recent performance window."""
    seconds = context.performance.average_response_seconds
    if seconds is None:
        return _unavailable("no question has been answered, so response latency is undefined")
    return _computed(seconds)


def _accuracy_trend(
    _spec: FeatureSpec,
    events: Sequence[EventEnvelope],
    context: ContextModel,
    settings: FeatureSettings,
) -> _Outcome:
    """Change in accuracy between the comparison window and the short window."""
    recent, earlier = _answer_windows(events, context, settings)
    minimum = settings.min_events_for_window
    if len(recent) < minimum or len(earlier) < minimum:
        return _unavailable(
            f"accuracy trend needs at least {minimum} answers in each window; "
            f"short window has {len(recent)}, comparison window has {len(earlier)}"
        )
    recent_accuracy = sum(1 for answer in recent if answer.correct) / len(recent)
    earlier_accuracy = sum(1 for answer in earlier if answer.correct) / len(earlier)
    return _computed(recent_accuracy - earlier_accuracy)


def _response_trend(
    _spec: FeatureSpec,
    events: Sequence[EventEnvelope],
    context: ContextModel,
    settings: FeatureSettings,
) -> _Outcome:
    """Change in mean response latency between the comparison window and the short window."""
    recent, earlier = _answer_windows(events, context, settings)
    minimum = settings.min_events_for_window
    if len(recent) < minimum or len(earlier) < minimum:
        return _unavailable(
            f"response trend needs at least {minimum} answers in each window; "
            f"short window has {len(recent)}, comparison window has {len(earlier)}"
        )
    recent_mean = sum(answer.response_seconds for answer in recent) / len(recent)
    earlier_mean = sum(answer.response_seconds for answer in earlier) / len(earlier)
    return _computed(recent_mean - earlier_mean)


def _intervention_count(
    _spec: FeatureSpec,
    _events: Sequence[EventEnvelope],
    context: ContextModel,
    _settings: FeatureSettings,
) -> _Outcome:
    """Number of interventions started in this session."""
    return _computed(context.intervention.total_interventions)


def _intervention_cooldown_active(
    _spec: FeatureSpec,
    _events: Sequence[EventEnvelope],
    context: ContextModel,
    _settings: FeatureSettings,
) -> _Outcome:
    """Whether an intervention cooldown is still running."""
    return _computed(int(context.intervention.in_cooldown))


def _intervention_seconds_since_last(
    _spec: FeatureSpec,
    _events: Sequence[EventEnvelope],
    context: ContextModel,
    _settings: FeatureSettings,
) -> _Outcome:
    """Seconds since the most recent intervention.

    Returns ``INSUFFICIENT_DATA`` rather than ``0.0`` when no intervention has occurred.
    A learner who has never been intervened with is not an intervention that happened
    zero seconds ago, and the two would be indistinguishable downstream.
    """
    seconds = context.intervention.seconds_since_last_intervention
    if seconds is None:
        return _unavailable("no intervention has occurred in this session")
    return _computed(seconds)


def _trajectory_encoded(
    _spec: FeatureSpec,
    _events: Sequence[EventEnvelope],
    context: ContextModel,
    _settings: FeatureSettings,
) -> _Outcome:
    """Closed encoding of the reported trajectory."""
    trajectory = context.trajectory
    if trajectory is None:
        return _unavailable("the temporal engine has not reported a trajectory yet")
    return _computed(_encode_trajectory(trajectory))


def _baseline_maturity_encoded(
    _spec: FeatureSpec,
    _events: Sequence[EventEnvelope],
    context: ContextModel,
    _settings: FeatureSettings,
) -> _Outcome:
    """Closed encoding of the reported baseline maturity."""
    maturity = context.baseline_maturity
    if maturity is None:
        return _unavailable("the baseline engine has not reported a maturity yet")
    return _computed(_encode_baseline_maturity(maturity))


_COMPUTERS: Final[Mapping[FeatureName, _Computer]] = {
    FeatureName.SESSION_ELAPSED_SECONDS: _session_elapsed,
    FeatureName.SESSION_POSITION: _session_position,
    FeatureName.SESSION_EVENT_COUNT: _session_event_count,
    FeatureName.CONTENT_DIFFICULTY: _content_difficulty,
    FeatureName.CONTENT_TYPE_ENCODED: _content_type_encoded,
    FeatureName.PERFORMANCE_RECENT_ACCURACY: _recent_accuracy,
    FeatureName.PERFORMANCE_RECENT_RESPONSE_SECONDS: _recent_response_seconds,
    FeatureName.PERFORMANCE_ACCURACY_TREND: _accuracy_trend,
    FeatureName.PERFORMANCE_RESPONSE_TREND: _response_trend,
    FeatureName.INTERVENTION_COUNT: _intervention_count,
    FeatureName.INTERVENTION_COOLDOWN_ACTIVE: _intervention_cooldown_active,
    FeatureName.INTERVENTION_SECONDS_SINCE_LAST: _intervention_seconds_since_last,
    FeatureName.TRAJECTORY_ENCODED: _trajectory_encoded,
    FeatureName.BASELINE_MATURITY_ENCODED: _baseline_maturity_encoded,
}

FEATURE_SPECS_V1: Final[tuple[FeatureSpec, ...]] = (
    FeatureSpec(
        name=FeatureName.SESSION_ELAPSED_SECONDS,
        version="1.0.0",
        category=FeatureCategory.SESSION,
        data_source="context.session.elapsed_seconds",
        definition=(
            "Seconds between the session start and the reference time, measured against the "
            "last observed event rather than the wall clock."
        ),
        required_context_keys=frozenset({ContextKey.SESSION, ContextKey.ELAPSED_SECONDS}),
        value_type=FeatureValueType.FLOAT,
        valid_range=(0.0, 86400.0),
    ),
    FeatureSpec(
        name=FeatureName.SESSION_POSITION,
        version="1.0.0",
        category=FeatureCategory.SESSION,
        data_source="context.session.position_in_session",
        definition=(
            "Elapsed time divided by the configured expected session length, clamped by the "
            "context layer to [0, 1]. A value of 1.0 means the expected length was met, not "
            "that the session is over."
        ),
        required_context_keys=frozenset({ContextKey.SESSION, ContextKey.POSITION_IN_SESSION}),
        value_type=FeatureValueType.FLOAT,
        valid_range=(0.0, 1.0),
    ),
    FeatureSpec(
        name=FeatureName.SESSION_EVENT_COUNT,
        version="1.0.0",
        category=FeatureCategory.SESSION,
        data_source="context.session.event_count",
        definition="Number of events applied to this session so far.",
        required_context_keys=frozenset({ContextKey.SESSION}),
        value_type=FeatureValueType.INT,
        valid_range=(0.0, 100000.0),
    ),
    FeatureSpec(
        name=FeatureName.CONTENT_DIFFICULTY,
        version="1.0.0",
        category=FeatureCategory.CONTENT,
        data_source="context.content.difficulty",
        definition=(
            "Difficulty metadata reported on the current content event. Absent metadata yields "
            "INSUFFICIENT_DATA; it is never replaced by a midpoint default, because a "
            "substituted 0.5 would be indistinguishable from a genuinely average item."
        ),
        required_context_keys=frozenset({ContextKey.DIFFICULTY}),
        value_type=FeatureValueType.FLOAT,
        valid_range=(0.0, 1.0),
    ),
    FeatureSpec(
        name=FeatureName.CONTENT_TYPE_ENCODED,
        version="1.0.0",
        category=FeatureCategory.CONTENT,
        data_source="context.content.content_type",
        definition=(
            "Content type mapped onto a closed vocabulary: question, video, reading, other. "
            "The mapping is total, so an unrecognised type becomes 'other' rather than "
            "propagating an unbounded string into a model input."
        ),
        required_context_keys=frozenset({ContextKey.CONTENT_TYPE}),
        value_type=FeatureValueType.STR,
    ),
    FeatureSpec(
        name=FeatureName.PERFORMANCE_RECENT_ACCURACY,
        version="1.0.0",
        category=FeatureCategory.PERFORMANCE,
        data_source="context.performance.recent_accuracy",
        definition=(
            "Correct answers divided by answers in the configured performance window. This is "
            "an immediate-window summary, not a personal baseline."
        ),
        required_context_keys=frozenset({ContextKey.RECENT_ACCURACY}),
        value_type=FeatureValueType.FLOAT,
        valid_range=(0.0, 1.0),
    ),
    FeatureSpec(
        name=FeatureName.PERFORMANCE_RECENT_RESPONSE_SECONDS,
        version="1.0.0",
        category=FeatureCategory.PERFORMANCE,
        data_source="context.performance.average_response_seconds",
        definition="Mean response latency over the configured performance window.",
        required_context_keys=frozenset({ContextKey.AVERAGE_RESPONSE_SECONDS}),
        value_type=FeatureValueType.FLOAT,
        valid_range=(0.0, 86400.0),
    ),
    FeatureSpec(
        name=FeatureName.PERFORMANCE_ACCURACY_TREND,
        version="1.0.0",
        category=FeatureCategory.PERFORMANCE,
        data_source="answered question events within and before the short window",
        definition=(
            "Mean accuracy over answers inside the short window minus mean accuracy over "
            "answers before it. Positive means accuracy was higher recently. Both windows "
            "must hold at least min_events_for_window answers; events after the reference time "
            "are excluded."
        ),
        required_context_keys=frozenset({ContextKey.SESSION, ContextKey.RECENT_ACCURACY}),
        value_type=FeatureValueType.FLOAT,
        valid_range=(-1.0, 1.0),
    ),
    FeatureSpec(
        name=FeatureName.PERFORMANCE_RESPONSE_TREND,
        version="1.0.0",
        category=FeatureCategory.PERFORMANCE,
        data_source="answered question events within and before the short window",
        definition=(
            "Mean response latency inside the short window minus the mean before it, in "
            "seconds. Positive means responses were slower recently. Both windows must hold at "
            "least min_events_for_window answers; events after the reference time are excluded."
        ),
        required_context_keys=frozenset({ContextKey.SESSION, ContextKey.AVERAGE_RESPONSE_SECONDS}),
        value_type=FeatureValueType.FLOAT,
        valid_range=(-86400.0, 86400.0),
    ),
    FeatureSpec(
        name=FeatureName.INTERVENTION_COUNT,
        version="1.0.0",
        category=FeatureCategory.INTERVENTION,
        data_source="context.intervention.total_interventions",
        definition="Number of interventions started in this session.",
        required_context_keys=frozenset({ContextKey.INTERVENTION_HISTORY}),
        value_type=FeatureValueType.INT,
        valid_range=(0.0, 1000.0),
    ),
    FeatureSpec(
        name=FeatureName.INTERVENTION_COOLDOWN_ACTIVE,
        version="1.0.0",
        category=FeatureCategory.INTERVENTION,
        data_source="context.intervention.in_cooldown",
        definition=(
            "1 while an intervention cooldown is running, 0 otherwise. Reported as 0 when no "
            "intervention has occurred, because a learner who has not been intervened with is "
            "not restrained."
        ),
        required_context_keys=frozenset({ContextKey.COOLDOWN_STATUS}),
        value_type=FeatureValueType.INT,
        valid_range=(0.0, 1.0),
    ),
    FeatureSpec(
        name=FeatureName.INTERVENTION_SECONDS_SINCE_LAST,
        version="1.0.0",
        category=FeatureCategory.INTERVENTION,
        data_source="context.intervention.seconds_since_last_intervention",
        definition=(
            "Seconds between the reference time and the most recent intervention start. Yields "
            "INSUFFICIENT_DATA when no intervention has occurred, rather than 0.0, so that "
            "'never intervened with' stays distinguishable from 'just intervened with'."
        ),
        required_context_keys=frozenset({ContextKey.COOLDOWN_STATUS}),
        value_type=FeatureValueType.FLOAT,
        valid_range=(0.0, 86400.0),
    ),
    FeatureSpec(
        name=FeatureName.TRAJECTORY_ENCODED,
        version="1.0.0",
        category=FeatureCategory.STATE,
        data_source="context.trajectory",
        definition=(
            "Trajectory reported by the temporal engine mapped onto a closed vocabulary: "
            "improving, stable, declining, other. Inadequate until the temporal engine exists, "
            "which is the reason it reports INSUFFICIENT_DATA rather than 'unknown' now."
        ),
        required_context_keys=frozenset({ContextKey.TRAJECTORY}),
        value_type=FeatureValueType.STR,
    ),
    FeatureSpec(
        name=FeatureName.BASELINE_MATURITY_ENCODED,
        version="1.0.0",
        category=FeatureCategory.STATE,
        data_source="context.baseline_maturity",
        definition=(
            "Baseline maturity reported by the baseline engine mapped onto a closed vocabulary: "
            "new, early, developing, established, other. Inadequate until the baseline engine "
            "exists."
        ),
        required_context_keys=frozenset({ContextKey.BASELINE_MATURITY}),
        value_type=FeatureValueType.STR,
    ),
)

CURRENT_FEATURE_SET: Final[FeatureSetVersion] = FEATURE_SET_V1
"""The feature set new code should use unless it is reproducing a historical result."""


class FeatureRegistry:
    """Append-only map from feature-set version to its feature definitions.

    Registration is the only way to add definitions, and a version can be registered
    exactly once. Re-registering an existing version is refused rather than allowed to
    overwrite, because the definitions behind a version are what make an already-computed
    vector interpretable. Editing them after the fact would leave that vector describing a
    calculation that no longer exists anywhere.
    """

    __slots__ = ("_versions",)

    def __init__(
        self, initial: Mapping[FeatureSetVersion, Sequence[FeatureSpec]] | None = None
    ) -> None:
        """Create a registry, optionally seeded with existing versions.

        Args:
            initial: Version-to-specifications mapping to register up front.
        """
        self._versions: dict[FeatureSetVersion, tuple[FeatureSpec, ...]] = {}
        for version, specs in (initial or {}).items():
            self.register(version, specs)

    def register(self, version: FeatureSetVersion, specs: Sequence[FeatureSpec]) -> None:
        """Append a new feature-set version.

        Args:
            version: Version identifier, which must match the version grammar.
            specs: The definitions for this version, in vector order.

        Raises:
            ValueError: If the version is already registered, if no specs are given, or
                if a feature name appears twice.
        """
        checked = checked_version(version)
        if checked in self._versions:
            raise ValueError(
                f"feature set {checked} is already registered with "
                f"{len(self._versions[checked])} features; versioned definitions are append-only. "
                "Mint a new version instead of editing an existing one."
            )
        if not specs:
            raise ValueError(f"feature set {checked} must define at least one feature")
        ordered = tuple(specs)
        names = [spec.name for spec in ordered]
        duplicates = sorted({name.value for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"feature set {checked} defines these features more than once: {duplicates}"
            )
        self._versions[checked] = ordered

    def versions(self) -> tuple[FeatureSetVersion, ...]:
        """List every registered version in registration order.

        Returns:
            The registered version identifiers.
        """
        return tuple(self._versions)

    def list_features(self, version: FeatureSetVersion) -> tuple[FeatureSpec, ...]:
        """Return the definitions for a version.

        Args:
            version: The version identifier.

        Returns:
            The specifications in vector order, or an empty tuple if the version is not
            registered. Use :meth:`require` when an empty result would be a defect.
        """
        return self._versions.get(version, ())

    def require(self, version: FeatureSetVersion) -> tuple[FeatureSpec, ...]:
        """Return the definitions for a version, refusing an unknown version.

        Args:
            version: The version identifier.

        Returns:
            The specifications in vector order.

        Raises:
            UnsupportedFeatureSetError: If the version is not registered.
        """
        specs = self._versions.get(version)
        if specs is None:
            raise UnsupportedFeatureSetError(
                f"feature set {version!r} is not registered; known versions: {list(self._versions)}"
            )
        return specs

    def get_spec(self, name: FeatureName, version: FeatureSetVersion) -> FeatureSpec | None:
        """Look up one feature definition.

        Args:
            name: The feature name.
            version: The version identifier.

        Returns:
            The specification, or ``None`` if that version does not define the feature.
        """
        for spec in self.list_features(version):
            if spec.name is name:
                return spec
        return None

    def __contains__(self, version: object) -> bool:
        """Whether a version is registered.

        Args:
            version: The version identifier to test.

        Returns:
            ``True`` when the version is registered.
        """
        return version in self._versions

    def __len__(self) -> int:
        """Number of registered versions.

        Returns:
            The count of registered feature-set versions.
        """
        return len(self._versions)

    def __repr__(self) -> str:
        """Return a debugging representation.

        Returns:
            A short representation listing the registered versions.
        """
        return f"FeatureRegistry(versions={list(self._versions)})"


DEFAULT_REGISTRY: Final[FeatureRegistry] = FeatureRegistry({FEATURE_SET_V1: FEATURE_SPECS_V1})
"""The registry used when a caller does not supply one."""


class FeatureEngine:
    """Deterministic feature computation from events and context.

    The engine is stateless. All inputs are passed explicitly; nothing is cached between
    calls, so the same inputs always produce the same outputs regardless of what was
    computed before.

    Attributes:
        feature_set_version: The version this engine computes.
        settings: Window configuration used by the trend features.
    """

    __slots__ = ("_feature_set_version", "_registry", "_settings")

    def __init__(
        self,
        registry: FeatureRegistry | None = None,
        feature_set_version: FeatureSetVersion = CURRENT_FEATURE_SET,
        settings: FeatureSettings | None = None,
    ) -> None:
        """Create an engine bound to one feature-set version.

        Args:
            registry: Registry to resolve definitions from. Defaults to
                :data:`DEFAULT_REGISTRY`.
            feature_set_version: Version to compute.
            settings: Feature configuration. Defaults to :class:`FeatureSettings`.

        Raises:
            UnsupportedFeatureSetError: If the version is not in the registry. Failing
                here rather than at computation time means a misconfigured deployment is
                reported at construction, not on the first request.
        """
        self._registry = registry if registry is not None else DEFAULT_REGISTRY
        self._feature_set_version = checked_version(feature_set_version)
        self._settings = settings if settings is not None else FeatureSettings()
        self._registry.require(self._feature_set_version)

    @property
    def feature_set_version(self) -> FeatureSetVersion:
        """The feature-set version this engine computes.

        Returns:
            The version identifier.
        """
        return self._feature_set_version

    @property
    def settings(self) -> FeatureSettings:
        """The window configuration in force.

        Returns:
            The feature settings.
        """
        return self._settings

    def compute(self, events: Sequence[EventEnvelope], context: ContextModel) -> FeatureValue:
        """Compute the full feature vector for a point in time.

        Args:
            events: The events observed. Events after ``context.reference_time`` are
                excluded from every computation.
            context: The context at the reference time.

        Returns:
            A frozen, versioned feature vector with one entry per registered feature.
        """
        specs = self._registry.require(self._feature_set_version)
        values = tuple(self.compute_one(spec, events, context) for spec in specs)
        session = context.session
        return FeatureValue(
            feature_set_version=self._feature_set_version,
            computed_at=context.reference_time,
            values=values,
            learner_id=session.learner_id if session else None,
            session_id=session.session_id if session else None,
            origins=context.origins,
        )

    def compute_one(
        self, spec: FeatureSpec, events: Sequence[EventEnvelope], context: ContextModel
    ) -> TypedFeatureValue:
        """Compute a single feature.

        Args:
            spec: The definition to compute.
            events: The events observed.
            context: The context at the reference time.

        Returns:
            A typed value, or ``INSUFFICIENT_DATA`` with a reason stating which
            evidence was missing or which check failed.
        """
        missing = spec.required_context_keys & context.missing
        if missing:
            return self._insufficient(
                spec,
                context,
                f"missing context: {', '.join(sorted(key.value for key in missing))}",
            )

        computer = _COMPUTERS.get(spec.name)
        if computer is None:
            return self._insufficient(
                spec, context, f"no calculation is registered for {spec.name.value}"
            )

        value, reason = computer(spec, events, context, self._settings)
        if reason is None:
            reason = self._validate_result(spec, value)
        if reason is not None:
            return self._insufficient(spec, context, reason)

        availability = (
            FeatureAvailability.SYNTHETIC_ONLY
            if context.is_synthetic_only()
            else FeatureAvailability.AVAILABLE
        )
        return TypedFeatureValue(
            name=spec.name,
            value=value,
            availability=availability,
            computed_at=context.reference_time,
            feature_set_version=self._feature_set_version,
            spec_version=spec.version,
        )

    def _insufficient(
        self, spec: FeatureSpec, context: ContextModel, reason: str
    ) -> TypedFeatureValue:
        """Build an ``INSUFFICIENT_DATA`` result.

        Args:
            spec: The definition that could not be computed.
            context: The context at the reference time.
            reason: Why it could not be computed.

        Returns:
            A typed value carrying the marker and the reason.
        """
        return TypedFeatureValue(
            name=spec.name,
            value=INSUFFICIENT_DATA_MARKER,
            availability=FeatureAvailability.INSUFFICIENT_DATA,
            reason=reason,
            computed_at=context.reference_time,
            feature_set_version=self._feature_set_version,
            spec_version=spec.version,
        )

    @staticmethod
    def _validate_result(spec: FeatureSpec, value: int | float | str) -> str | None:
        """Check a computed value against its declared type and range.

        Args:
            spec: The definition the value was computed for.
            value: The computed value.

        Returns:
            A reason describing the defect, or ``None`` when the value conforms.
        """
        if spec.value_type is FeatureValueType.STR and not isinstance(value, str):
            return (
                f"computation for {spec.name.value} produced a {type(value).__name__} but the "
                "spec declares a string value"
            )
        if spec.value_type is not FeatureValueType.STR and isinstance(value, str):
            return (
                f"computation for {spec.name.value} produced the string {value!r} but the spec "
                f"declares a {spec.value_type.value} value"
            )
        if spec.value_type is FeatureValueType.INT and not isinstance(value, int):
            return (
                f"computation for {spec.name.value} produced a {type(value).__name__} but the "
                "spec declares an integer value"
            )
        if spec.valid_range is None or isinstance(value, str):
            return None
        low, high = spec.valid_range
        if value < low or value > high:
            return (
                f"computation for {spec.name.value} produced {value}, outside the valid range "
                f"[{low}, {high}] declared in its spec. The value has not been clamped: an "
                "out-of-range result indicates a defect in the calculation or the input data."
            )
        return None


def compute_features(
    events: Sequence[EventEnvelope],
    context: ContextModel,
    feature_set_version: FeatureSetVersion = CURRENT_FEATURE_SET,
    settings: FeatureSettings | None = None,
    registry: FeatureRegistry | None = None,
) -> FeatureValue:
    """Compute a feature vector in one call.

    Args:
        events: The events observed.
        context: The context at the reference time.
        feature_set_version: Version to compute.
        settings: Feature configuration. Defaults to :class:`FeatureSettings`.
        registry: Registry to resolve definitions from.

    Returns:
        A frozen, versioned feature vector.

    Raises:
        UnsupportedFeatureSetError: If the requested version is not registered.
    """
    engine = FeatureEngine(
        registry=registry, feature_set_version=feature_set_version, settings=settings
    )
    return engine.compute(events, context)
