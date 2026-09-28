"""Temporal state data model.

The baseline layer answers *how far is this measurement from normal*. The temporal layer
answers the question that a distance alone cannot: *is this a moment, or is this a
change?* Those are different claims with different consequences, and collapsing them is
the most common way a behavioural system starts generating confident nonsense.

The vocabulary is built around that distinction. A
:class:`TrendDirection` says which way the deviation is moving. A :class:`TrendCharacter`
says whether the movement is isolated, persistent, or a flat line. A
:class:`ChangeKind` says what kind of episode is occurring. And a
:class:`StateEvidence` records *why* a state was reachable at all, because a state
asserted without a traceable basis is a claim rather than a measurement.

Four properties are enforced in types rather than in review comments.

**Absence is a first-class answer, not an error.** A learner with no deviation history has
no state. The model reports :attr:`BehavioralEngagementState.INSUFFICIENT_DATA` with an
empty evidence tuple, and a validator refuses to let any other state be paired with it.
Substituting a plausible default here — reporting ``STABLE`` because nothing has gone
wrong — is precisely the failure the layer exists to prevent, and it is easier to prevent
by making it unconstructible than by asking reviewers to notice it.

**The label vocabulary is inherited, not invented.** The states this layer emits are
:class:`~focus_engine.schemas.primitives.BehavioralEngagementState` values, defined in
Phase 1. This layer does not mint a parallel state vocabulary, because a second set of
labels for the same construct guarantees that a downstream consumer will eventually read
one where it expects the other.

**A single anomalous observation can never be a trajectory.** :attr:`TrajectoryChange`
requires a run of consecutive same-sign standardised deviations, and
:attr:`StateEvidence.consecutive` records how many were actually present. A validator
refuses to describe a run shorter than
:attr:`~focus_engine.configuration.thresholds.TemporalSettings.sustained_change_min_observations`
as sustained. This is the Phase 7 exit criterion expressed as a construction rule: if the
object cannot be built for a one-off spike, the spike cannot be reported as a trajectory.

**The standing of the inference travels with the state.** Every :class:`StateEvidence`
carries the baseline ``maturity`` and ``basis`` in force when the state was computed. A
state derived from a population prior is not the same claim as one derived from a
developing personal baseline, and downstream code that reads the state without reading its
basis is reading half a record. The weakest maturity among the dimensions that contributed
is the one recorded, because a state is only as personal as its least personal input.

The module holds vocabulary and structure only. Derivation lives in
:mod:`focus_engine.temporal.engine`.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from focus_engine.baseline.models import (
    BaselineDimension,
    BaselineMaturity,
    DeviationResult,
    basis_for,
    maturity_rank,
)
from focus_engine.schemas.primitives import (
    BehavioralEngagementState,
    DataOrigin,
    InferenceBasis,
    Timestamp,
    non_empty_text,
)
from focus_engine.schemas.versioning import TemporalVersion

__all__ = [
    "TEMPORAL_STATE_V1",
    "TREND_DIRECTIONS",
    "BehavioralEngagementState",
    "ChangeKind",
    "DeviationPoint",
    "StateEvidence",
    "StateTransition",
    "TemporalObservation",
    "TemporalState",
    "TemporalTrack",
    "TrajectoryChange",
    "TrendCharacter",
    "TrendDirection",
    "weakest_maturity",
]


TEMPORAL_STATE_V1: Final[TemporalVersion] = "TEMPORAL_STATE_V1"
"""The initial temporal definition set.

Freeze the state ladder, the persistence rule, and the stability measure together, so
that a change to any of them mints ``TEMPORAL_STATE_V2`` rather than silently
reinterpreting every state already recorded. A stored state must remain explainable in
terms of the definitions that produced it.
"""


class TrendDirection(StrEnum):
    """Which way a deviation is moving relative to the personal baseline.

    Defined on the *deviation*, not on the raw measurement, and that distinction is the
    whole reason this enum exists. Response time rising is a deterioration; accuracy
    falling is a deterioration; but both are a falling deviation from that dimension's
    own centre. A direction computed from raw values would classify those two cases as
    opposite trends for the same underlying behaviour, and every consumer of the label
    would then need to know which dimensions it was looking at.
    """

    STABLE = "stable"
    """No directional evidence. The deviation is not systematically moving either way."""

    IMPROVING = "improving"
    """The standardised deviation is falling toward zero, from either side."""

    WORSENING = "worsening"
    """The standardised deviation is growing away from zero, in absolute terms."""

    INSUFFICIENT_DATA = "insufficient_data"
    """Too few observations to establish a direction. Not a synonym for ``STABLE``."""


#: Canonical direction order, for exhaustive consumer switches and stable comparison.
TREND_DIRECTIONS: Final[tuple[TrendDirection, ...]] = (
    TrendDirection.WORSENING,
    TrendDirection.STABLE,
    TrendDirection.IMPROVING,
    TrendDirection.INSUFFICIENT_DATA,
)


def weakest_maturity(levels: Sequence[BaselineMaturity]) -> BaselineMaturity:
    """Return the least-established maturity in a set.

    A temporal state aggregates several dimensions, each measured against its own
    baseline. The state's standing is the *weakest* of them, because a state is only as
    personal as its least personal input: one dimension still resting on a population
    prior means the state as a whole rests partly on a prior, and reporting it as
    ``PERSONAL`` would let a single cold-start dimension launder the other three.

    Args:
        levels: The maturities to compare. May be empty.

    Returns:
        The lowest maturity by rank, or :attr:`BaselineMaturity.NEW` for an empty
        sequence. ``NEW`` is the conservative answer: no evidence supports any better
        claim, and it makes the empty case agree with the insufficient-data case.
    """
    if not levels:
        return BaselineMaturity.NEW
    return min(levels, key=maturity_rank)


class TrendCharacter(StrEnum):
    """The *shape* of a deviation run, independent of its direction.

    Separating character from direction is what lets a run be recognised as an anomaly
    without it also being described as an improvement. A single 300-second response time
    is a large, worsening, isolated excursion: two of those three words come from
    different axes, and merging them into one enum would force the anomaly test to
    re-derive the direction it already had.
    """

    FLAT = "flat"
    """Deviations indistinguishable from zero. Behaviour matches the personal baseline."""

    ISOLATED = "isolated"
    """A single large deviation surrounded by flat ones. A momentary excursion, on the
    evidence available so far — which is not the same as a claim that it was harmless."""

    PERSISTENT = "persistent"
    """Consecutive same-sign deviations. The pattern that distinguishes a trajectory from
    an anomaly."""


class ChangeKind(StrEnum):
    """What sort of episode the deviation history currently describes.

    Each value names an observable pattern in the recorded deviations. None of them is a
    claim about the learner's internal state, and none asserts that the behaviour matters:
    that judgement belongs to the uncertainty engine and the intervention policy, and
    making it here would collapse Separation 1 at the point where the evidence is
    thinnest.
    """

    NONE = "none"
    """No deviation has been recorded yet."""

    ANOMALY = "anomaly"
    """One large deviation that has not persisted."""

    TRAJECTORY = "trajectory"
    """A sustained same-sign run. The change the system exists to notice."""

    RECOVERY = "recovery"
    """Deviation magnitude is falling consistently. A return toward the personal baseline,
    which is a distinct pattern from never having left it."""

    STABLE = "stable"
    """Repeatedly near-zero deviations. Recorded as its own value rather than as
    ``NONE`` so that a consumer can distinguish "nothing has happened yet" from
    "nothing is happening" — the two have different histories and different meanings."""


class DeviationPoint(BaseModel):
    """One deviation from a personal baseline, placed on a timeline.

    The standardised value is the quantity the temporal logic operates on, because it is
    the only form of a deviation that is comparable across dimensions. A 0.2 accuracy
    difference and a 4-second latency difference are not comparable in raw units; a
    deviation of 1.5 dispersions is comparable to a deviation of 1.5 dispersions whatever
    the dimension.

    A point with no standardised value is retained with a reason rather than discarded.
    Dropping it would silently shorten the run that the persistence rule counts, so a
    period of unmeasurable behaviour would look like a period of no behaviour.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: BaselineDimension
    value: float
    standardised: float | None = None
    """Deviation in dispersion units. ``None`` when no dispersion could be estimated."""

    reason: str | None = None
    """Why no standardised value exists. Required exactly when ``standardised`` is
    ``None``, mirroring the invariant on the baseline layer's own deviation record."""

    maturity: BaselineMaturity
    basis: InferenceBasis
    computed_at: Timestamp
    observation_index: int = Field(ge=0)
    """Position in this dimension's deviation sequence, starting at zero.

    Recorded on the point rather than inferred from position, so a run that has been
    windowed or decimated still says which observations it was built from.
    """

    @model_validator(mode="after")
    def _validate_point(self) -> DeviationPoint:
        """Check the standardised value and its reason are mutually consistent.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If exactly one of ``standardised`` and ``reason`` is present.
        """
        if (self.standardised is None) == (self.reason is None):
            raise ValueError(
                f"deviation for {self.dimension.value} at observation "
                f"{self.observation_index} must state a reason exactly when no standardised "
                "value is available, and must not state one when it is available"
            )
        return self

    @property
    def is_measurable(self) -> bool:
        """Whether this point carries a standardised deviation.

        Returns:
            ``True`` when the point can take part in direction and persistence logic.
        """
        return self.standardised is not None

    def is_material(self, minimum: float) -> bool:
        """Whether the deviation is large enough to count as a departure.

        A deviation of a hundredth of a dispersion is real and measurable but carries no
        information about a change, and counting it as material would let a learner who is
        merely stable-but-noisy satisfy a persistence requirement. A learner hovering at
        zero deviation is not stable in any useful sense, and is not changing either.

        The threshold is passed in rather than baked in because it is a decision about
        what counts as a departure, and that decision belongs to
        :class:`~focus_engine.configuration.thresholds.TemporalSettings`. Hard-coding it
        here would create a second, invisible place where the bar for "changed" is set,
        which is precisely the kind of threshold drift a configuration object exists to
        prevent.

        Args:
            minimum: The smallest absolute standardised deviation that counts.

        Returns:
            ``True`` when a standardised value exists and its absolute magnitude is at
            least ``minimum``.
        """
        return self.standardised is not None and abs(self.standardised) >= minimum


class TemporalObservation(BaseModel):
    """One learner's baseline deviations, all measured at a single instant.

    The baseline layer reports a deviation per dimension. Grouping them into one
    observation gives the temporal engine an atomic tick: it can reason about how many
    observations have occurred, and a state can never be half-updated from a partial set.

    Grouping also makes attribution explicit. A
    :class:`~focus_engine.baseline.models.DeviationResult` deliberately carries no learner
    identifier — the baseline layer that produced it was already scoped to one learner —
    so without this wrapper a deviation could be attached to the wrong learner's state,
    and a state computed from two learners' deviations would be arithmetically valid and
    entirely fictitious.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    learner_id: str
    computed_at: Timestamp
    deviations: tuple[DeviationResult, ...]
    origins: frozenset[DataOrigin]
    """Provenance of the events behind these deviations. Required, not defaulted.

    A :class:`~focus_engine.baseline.models.DeviationResult` records the *basis* of its
    comparison but not the origin of the observations that produced it, so this layer
    cannot recover the answer for itself. Defaulting the field would mean a caller who
    forgot it produced a state with no synthetic marking — exactly the erasure of
    provenance this project treats as its central anti-hallucination control, reached by
    omission rather than by intent. Requiring it makes the omission impossible.
    """

    @model_validator(mode="after")
    def _validate_observation(self) -> TemporalObservation:
        """Check the deviations are complete, unique, and simultaneous.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If no deviations are supplied, no origin is recorded, a dimension
                appears twice, or any deviation was measured at a different instant.
        """
        if not self.deviations:
            raise ValueError(
                "a temporal observation must carry at least one deviation; an observation "
                "with nothing in it is not evidence"
            )
        if not self.origins:
            raise ValueError(
                "a temporal observation must record the origin of its data. Deviation "
                "records do not carry provenance, so an unstated origin cannot be "
                "recovered later and would silently erase the synthetic marking."
            )
        dimensions = [deviation.dimension for deviation in self.deviations]
        duplicates = sorted({item.value for item in dimensions if dimensions.count(item) > 1})
        if duplicates:
            raise ValueError(f"a temporal observation repeats dimensions: {duplicates}")
        for deviation in self.deviations:
            if deviation.computed_at != self.computed_at:
                raise ValueError(
                    f"deviation for {deviation.dimension.value} was computed at "
                    f"{deviation.computed_at.isoformat()}, not at this observation's "
                    f"{self.computed_at.isoformat()}. Deviations grouped into one observation "
                    "must be simultaneous; combining measurements from different instants "
                    "would compare a learner against two different baselines at once."
                )
        return self


class StateEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    state: BehavioralEngagementState
    """The asserted state."""

    maturity: BaselineMaturity
    """The weakest baseline maturity among the dimensions that contributed."""

    basis: InferenceBasis
    """The inference basis the weakest maturity admits."""

    observations: int = Field(ge=0)
    """How many *measurable* deviations were available across all dimensions.

    Measurable, not merely recorded. A learner whose baseline was still a population prior
    can have deviations observed and retained, yet none of them standardisable against
    their own history, and such a learner genuinely has no state to report. Counting those
    points here would force a state to be asserted from evidence that does not exist.
    :attr:`~focus_engine.temporal.models.TemporalState.observations` retains the larger
    count of everything recorded."""

    consecutive: int = Field(default=0, ge=0)
    """Length of the current same-sign run. The quantity that separates an anomaly from a
    trajectory."""

    duration_seconds: float = Field(default=0.0, ge=0.0)
    """Wall-clock span the current state has persisted. Distinct from ``consecutive``:
    ten observations in ten minutes and ten observations across two hours are different
    evidence, and collapsing them would make sampling frequency look like persistence."""

    mean_standardised: float | None = None
    """Mean standardised deviation over the current state. ``None`` when nothing was
    measurable."""

    dispersion: float | None = Field(default=None, ge=0.0)
    """Robust dispersion of the recent standardised deviations, as an inverse measure of
    stability. ``None`` when fewer than two deviations were available."""

    reason: str | None = None
    """Why the state is what it is. Required for ``INSUFFICIENT_DATA`` and for any state
    that could not be reached for want of evidence."""

    @model_validator(mode="after")
    def _validate_evidence(self) -> StateEvidence:
        """Check the counts, quantities, and reasons are consistent.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If ``INSUFFICIENT_DATA`` carries evidence, if another state claims
                no observations, if dispersion is reported from fewer than two values, or
                if the basis is stronger than the maturity admits.
        """
        if self.state is BehavioralEngagementState.INSUFFICIENT_DATA:
            if self.consecutive > 0:
                raise ValueError(
                    "state insufficient_data cannot be paired with a run of "
                    f"{self.consecutive}; a persistent departure is a state, not an absence "
                    "of one"
                )
            if not self.reason:
                raise ValueError(
                    "state insufficient_data must state why there is insufficient data; an "
                    "unexplained absence is indistinguishable from a bug"
                )
        else:
            if self.observations == 0:
                raise ValueError(
                    f"state {self.state.value!r} requires at least one observation, but none "
                    "were available"
                )
        if self.mean_standardised is not None and self.observations < 1:
            raise ValueError("mean_standardised was reported but no observation was available")
        if self.dispersion is not None and self.observations < 2:
            raise ValueError(
                f"state {self.state.value!r} reports a dispersion from a single observation"
            )
        if self.basis is not basis_for(self.maturity):
            raise ValueError(
                f"maturity {self.maturity.value!r} does not admit basis {self.basis.value!r}; "
                f"it requires {basis_for(self.maturity).value!r}"
            )
        return self


class StateTransition(BaseModel):
    """A change of asserted state between two reference times.

    Transitions are recorded rather than overwritten. Whether a system is flip-flopping
    between two states is a property of the transition *sequence*, and a state record
    that keeps only the current value destroys exactly the evidence needed to detect it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_state: BehavioralEngagementState
    to_state: BehavioralEngagementState
    at: Timestamp
    consecutive: int = Field(default=0, ge=0)
    """Run length at the moment of transition, carried forward so the destination
    transition can be attributed to the evidence that produced it."""

    reason: non_empty_text
    """Why the state changed, in the engine's own words. Recorded so a transition can be
    read without re-running the engine.

    Required rather than defaulted. A transition is the record that a state *moved*, and a
    moved state with no stated cause leaves the audit trail asserting a change it cannot
    justify. Defaulting this field would let a caller construct exactly that."""


class TrajectoryChange(BaseModel):
    """A sustained same-sign run, described.

    This is the object the Phase 7 exit criteria call for: something that cannot be
    constructed for a single anomalous observation. It reports *what* moved, in which
    dimension, over how long, and to what magnitude — and it refuses to exist unless the
    run is long enough to be a change rather than a moment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: BaselineDimension
    direction: TrendDirection
    start: Timestamp
    end: Timestamp
    observations: int = Field(ge=1)
    """How many observations the run comprises. This is also the run's length: a trajectory
    describes one same-sign run and nothing else, so the count of observations *is* the
    count of consecutive ones, and a second field for it would be a second number that
    could disagree with the first."""

    first_standardised: float
    last_standardised: float
    mean_standardised: float
    peak_absolute: float = Field(ge=0.0)
    rate_per_observation: float
    """Change in standardised deviation per observation across the run, signed by
    direction. A positive value on a ``WORSENING`` run means the worsening accelerated."""

    duration_seconds: float = Field(default=0.0, ge=0.0)
    maturity: BaselineMaturity
    basis: InferenceBasis

    @model_validator(mode="after")
    def _validate_trajectory(self) -> TrajectoryChange:
        """Check the run is coherent and that its own figures agree with each other.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If the run is not a real run, if the timestamps are out of order,
                if the run changes sign, or if the recorded magnitudes do not match
                ``first_standardised`` and ``last_standardised``.
        """
        if self.observations < 1:
            raise ValueError("a trajectory must contain at least one observation")
        if self.end < self.start:
            raise ValueError(
                f"trajectory ends at {self.end.isoformat()}, before it starts at "
                f"{self.start.isoformat()}"
            )
        if self.duration_seconds < 0.0:
            raise ValueError(f"duration_seconds must be non-negative; got {self.duration_seconds}")
        if self.direction is TrendDirection.INSUFFICIENT_DATA:
            raise ValueError(
                "a trajectory has a direction by construction; insufficient_data describes "
                "an absence of direction and cannot be one"
            )
        if self.first_standardised * self.last_standardised < 0.0:
            raise ValueError(
                f"a trajectory run cannot change sign: it starts at "
                f"{self.first_standardised:+.2f} and ends at {self.last_standardised:+.2f}. "
                "A sign change is the start of a new run, not a continuation of this one."
            )
        if self.peak_absolute < max(abs(self.first_standardised), abs(self.last_standardised)):
            raise ValueError(
                f"peak magnitude {self.peak_absolute:.2f} is smaller than an endpoint of the "
                "run, so it does not describe the run it is recorded for"
            )
        if self.basis is not basis_for(self.maturity):
            raise ValueError(
                f"maturity {self.maturity.value!r} does not admit basis {self.basis.value!r}"
            )
        return self

    @property
    def is_monotonic(self) -> bool:
        """Whether the run moved consistently in magnitude, not merely in sign.

        A run of same-sign deviations is a run in *sign*, not necessarily in *magnitude*.
        A learner whose deviation stays positive but oscillates between 1.0 and 4.0
        dispersions has a persistent deviation and no trend in it, and reporting a rate
        for that run as though it were a steady climb would overstate the evidence.

        The signed slope is compared against the direction's *own* sign, because
        :attr:`direction` describes movement in absolute distance from the baseline while
        the slope is computed on signed values. For a negative run, a decreasing slope
        means a growing distance, so a naive sign comparison would call a run that is
        getting steadily worse "improving".

        Returns:
            ``True`` for a single-observation run, and otherwise whether the run's net
            change in signed standardised deviation agrees with its direction.
        """
        if self.observations < 2:
            return True
        delta = self.last_standardised - self.first_standardised
        slope = delta / (self.observations - 1)
        if self.direction is TrendDirection.STABLE:
            return slope == 0.0
        worsening = self.direction is TrendDirection.WORSENING
        growing_distance = slope > 0.0 if self.last_standardised > 0.0 else slope < 0.0
        return growing_distance if worsening else not growing_distance


class TemporalState(BaseModel):
    """The current temporal characterisation of one learner's behaviour.

    Carries the asserted state, its evidence, the current run, and every trajectory
    detected so far. The trajectory list is retained rather than replaced, because
    "this learner had two sustained changes and is currently recovering from the second"
    is a materially different record from "this learner is recovering".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    learner_id: str
    temporal_version: TemporalVersion
    reference_time: Timestamp
    created_at: Timestamp
    state: BehavioralEngagementState
    evidence: StateEvidence
    direction: TrendDirection
    character: TrendCharacter
    change: ChangeKind
    run: int = Field(default=0, ge=0)
    """Length of the current same-sign material run."""

    run_started_at: Timestamp | None = None
    origins: frozenset[DataOrigin] = Field(default_factory=frozenset)
    settings_fingerprint: str
    tracks: tuple[TemporalTrack, ...]
    """One accumulation per tracked dimension. The state is derived from these, and they
    are retained so a rate, a stability figure, or a run length can be re-derived and
    audited rather than trusted."""

    transitions: tuple[StateTransition, ...] = ()
    trajectories: tuple[TrajectoryChange, ...] = ()

    @model_validator(mode="after")
    def _validate_state(self) -> TemporalState:
        """Check the state, evidence, and supporting record agree.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If the evidence names a different state, if the tracked dimensions
                are incomplete or duplicated, if a run contradicts its timestamps, or if a
                trajectory change is claimed with no trajectory recorded.
        """
        if self.evidence.state is not self.state:
            raise ValueError(
                f"state is {self.state.value!r} but the evidence describes "
                f"{self.evidence.state.value!r}; a state and its evidence must agree"
            )
        dimensions = [track.dimension for track in self.tracks]
        if len(set(dimensions)) != len(dimensions):
            duplicates = sorted({item.value for item in dimensions if dimensions.count(item) > 1})
            raise ValueError(f"state tracks repeated dimensions: {duplicates}")
        if self.evidence.observations != self.measurable_observations:
            raise ValueError(
                f"evidence claims {self.evidence.observations} measurable observation(s) but "
                f"{self.measurable_observations} retained point(s) could be standardised; "
                "the evidence must be countable from the tracks it describes"
            )
        if self.run > 0 and self.run_started_at is None:
            raise ValueError("a non-empty run must record when it started")
        if self.run == 0 and self.run_started_at is not None:
            raise ValueError("a run of zero cannot have a start time")
        if self.run_started_at is not None and self.run_started_at > self.reference_time:
            raise ValueError(
                f"run started at {self.run_started_at.isoformat()}, after the state's "
                f"reference time at {self.reference_time.isoformat()}"
            )
        if self.change is ChangeKind.TRAJECTORY and not self.trajectories:
            raise ValueError(
                "change is reported as trajectory but no trajectory is recorded; a "
                "trajectory change must be evidenced by the trajectory that caused it"
            )
        for trajectory in self.trajectories:
            try:
                track = self.track_for(trajectory.dimension)
            except KeyError:
                raise ValueError(
                    f"a trajectory is recorded for {trajectory.dimension.value!r}, which this "
                    "state does not track"
                ) from None
            self._require_recorded_trajectory(track, trajectory)
        return self

    def _require_recorded_trajectory(
        self, track: TemporalTrack, trajectory: TrajectoryChange
    ) -> None:
        """Check a trajectory was actually emitted for this dimension's track.

        Without this, a trajectory is an assertion about observations the state does not
        contain. Every figure on it — the direction, the rate, the duration, the peak —
        would be self-consistent and unverifiable, which is the shape a fabricated record
        takes.

        The check is made against the track's own ledger of emitted trajectories rather
        than against its retained points, and that distinction matters. Retention is
        bounded, so a run long enough to outlast the window eventually has its points
        discarded while the trajectory describing it remains on the state as history.
        Checking against points would either crash the engine on a long-lived learner's
        correct behaviour, or force the trajectory to be dropped once its points aged out
        — erasing the record of a real departure so that a state could never report
        having seen one. The ledger is the record of what was emitted; a trajectory absent
        from it was never observed, whether or not the points behind it are still retained.

        Args:
            track: The trajectory's dimension.
            trajectory: The trajectory to check.

        Raises:
            ValueError: If the track never recorded this trajectory, or if the trajectory
                is current and its retained points contradict it.
        """
        if trajectory not in track.trajectories:
            raise ValueError(
                f"a trajectory for {track.dimension.value!r} spanning "
                f"{trajectory.start.isoformat()} to {trajectory.end.isoformat()} is recorded "
                "on this state but was never emitted for its track; a trajectory cannot "
                "describe an observation that was never recorded"
            )
        self._require_backing_points(track, trajectory)

    def _require_backing_points(self, track: TemporalTrack, trajectory: TrajectoryChange) -> None:
        """Check a current trajectory's endpoints are points the track still holds.

        The ledger proves a trajectory was emitted. This is the second, stronger check,
        applied while the whole run survives in the window: the standardised values at the
        recorded start and end instants must still be present in the retained points with
        the values the trajectory reports. Once the start has aged out, the run is only
        partly retained and can no longer be checked in full, so the ledger entry is
        trusted instead.

        Args:
            track: The trajectory's dimension.
            trajectory: The trajectory to check.

        Raises:
            ValueError: If a fully retained trajectory's endpoints or their values are
                absent from the track.
        """
        measurable = [point for point in track.points if point.standardised is not None]
        if not measurable:
            return
        if trajectory.start < measurable[0].computed_at:
            return
        by_instant = {(point.computed_at, point.standardised) for point in measurable}
        for label, instant, value in (
            ("start", trajectory.start, trajectory.first_standardised),
            ("end", trajectory.end, trajectory.last_standardised),
        ):
            if (instant, value) not in by_instant:
                raise ValueError(
                    f"a trajectory for {track.dimension.value!r} claims a {label} at "
                    f"{instant.isoformat()} of {value:+.2f} dispersions, which is not a point "
                    "this state retains; a trajectory cannot describe an observation that "
                    "was never recorded"
                )

    @property
    def observations(self) -> int:
        """How many deviation points are retained across every dimension.

        Includes points that could not be standardised. They are kept deliberately: a
        deviation that arrives while the baseline is too immature to compare against is
        still a recorded measurement of real behaviour, and discarding it would leave a
        hole in the learner's history that no later reader could distinguish from never
        having looked.

        Returns:
            The total retained point count.
        """
        return sum(len(track.points) for track in self.tracks)

    @property
    def measurable_observations(self) -> int:
        """How many retained points could be compared against the learner's own baseline.

        Returns:
            The count of points carrying a standardised value. This is the figure the
            evidence's ``observations`` field must equal, and the figure a state must be
            justified by.
        """
        return sum(
            1 for track in self.tracks for point in track.points if point.standardised is not None
        )

    def track_for(self, dimension: BaselineDimension) -> TemporalTrack:
        """Return the accumulation for one dimension.

        Args:
            dimension: The dimension to look up.

        Returns:
            Its track.

        Raises:
            KeyError: If the state does not track the dimension.
        """
        for track in self.tracks:
            if track.dimension is dimension:
                return track
        raise KeyError(dimension)

    @property
    def run_seconds(self) -> float:
        """How long the current run has lasted.

        Derived rather than stored, so it cannot disagree with the timestamps that define
        it.

        Returns:
            The span in seconds, or ``0.0`` when there is no run.
        """
        if self.run_started_at is None:
            return 0.0
        return max(0.0, (self.reference_time - self.run_started_at).total_seconds())

    @property
    def is_synthetic_only(self) -> bool:
        """Whether every contributing observation was synthetic.

        Returns:
            ``True`` only when the recorded origins are exactly synthetic.
        """
        return self.origins == frozenset({DataOrigin.SYNTHETIC})

    @property
    def is_insufficient(self) -> bool:
        """Whether this state declines to characterise the learner at all.

        The property a consumer should check before acting on anything in this record.

        Returns:
            ``True`` when the state is ``INSUFFICIENT_DATA``.
        """
        return self.state is BehavioralEngagementState.INSUFFICIENT_DATA

    def to_summary_dict(self) -> dict[str, object]:
        """Render a JSON-serialisable summary for logs and debugging.

        Returns:
            The state, its standing, and the counts a reviewer needs to judge it without
            re-reading the full record.
        """
        return {
            "learner_id": self.learner_id,
            "temporal_version": self.temporal_version,
            "reference_time": self.reference_time.isoformat(),
            "state": self.state.value,
            "direction": self.direction.value,
            "character": self.character.value,
            "change": self.change.value,
            "maturity": self.evidence.maturity.value,
            "basis": self.evidence.basis.value,
            "observations": self.observations,
            "consecutive": self.evidence.consecutive,
            "run": self.run,
            "run_seconds": self.run_seconds,
            "mean_standardised": self.evidence.mean_standardised,
            "dispersion": self.evidence.dispersion,
            "transitions": len(self.transitions),
            "trajectories": len(self.trajectories),
            "reason": self.evidence.reason,
        }


class TemporalTrack(BaseModel):
    """The per-dimension accumulation behind a :class:`TemporalState`.

    A :class:`TemporalState` is the summary a consumer reads; a track is the per-dimension
    evidence it was derived from. Splitting them means a state can be passed to a policy
    without also handing over every retained deviation, while an evaluation can walk the
    tracks without re-deriving the summary.

    The latest reference per dimension is retained alongside the point list, because the
    staleness of a dimension relative to the others is itself a fact: a state built from
    one dimension updated a week ago and another updated a minute ago has mixed evidence
    age, and hiding that would let a stale dimension look current.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: BaselineDimension
    points: tuple[DeviationPoint, ...] = ()
    trajectories: tuple[TrajectoryChange, ...] = ()
    latest_at: Timestamp | None = None
    maturity: BaselineMaturity = BaselineMaturity.NEW
    basis: InferenceBasis = InferenceBasis.POPULATION_PRIOR
    max_points_retained: int = Field(ge=1)
    recorded: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_track(self) -> TemporalTrack:
        """Check the point window is ordered, bounded, and consistent with its timestamp.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If points are unordered, exceed the retention bound, or disagree
                with ``latest_at`` or the recorded maturity.
        """
        if len(self.points) > self.max_points_retained:
            raise ValueError(
                f"{self.dimension.value} retains {len(self.points)} points but the bound is "
                f"{self.max_points_retained}"
            )
        if self.recorded < len(self.points):
            raise ValueError(
                f"{self.dimension.value} has recorded {self.recorded} observation(s) but "
                f"retains {len(self.points)} point(s); retention cannot outrun what was seen"
            )
        indices = [point.observation_index for point in self.points]
        if indices != sorted(indices) or len(set(indices)) != len(indices):
            raise ValueError(
                f"{self.dimension.value} points must be in strictly increasing observation order"
            )
        if self.points and self.latest_at is None:
            raise ValueError(
                f"{self.dimension.value} retains points but records no latest observation time"
            )
        return self

    @property
    def is_current(self) -> bool:
        """Whether this track has contributed an observation at the state reference time.

        Returns:
            ``True`` when the newest point sits exactly at ``latest_at``.
        """
        return bool(self.points) and self.points[-1].computed_at == self.latest_at

    @property
    def observations(self) -> int:
        """How many observations this dimension has ever contributed.

        Retained points are a window, not a tally: once the bound is reached the oldest
        points are dropped, so ``len(points)`` answers "how recent is this?" and
        ``recorded`` answers "how much has been seen?". A track that has recorded a
        hundred observations and retains five is not a track with five observations, and
        reporting it as one would let a long-lived learner look newly arrived.
        """
        return self.recorded

    @property
    def dropped(self) -> int:
        """How many observations the retention bound has discarded.

        Returns:
            The difference between everything recorded and everything retained.
        """
        return max(self.recorded - len(self.points), 0)
