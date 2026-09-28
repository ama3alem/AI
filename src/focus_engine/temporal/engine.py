"""Temporal state engine: is this a moment, or is this a change?

The baseline layer answers how far a measurement sits from a learner's own normal. This
layer answers the question a distance alone cannot: whether that distance is *happening*
or *happened*, and whether it is heading anywhere. Every state it emits is a statement
about the pattern of recorded deviations, never about a person.

Four mechanisms do the work, and each exists to prevent a specific failure.

**A single observation can never be a trajectory.** The engine keeps a bounded window of
deviations per dimension and reads the current run backwards from the newest. A run only
becomes a
:class:`~focus_engine.temporal.models.TrajectoryChange` once it reaches
``sustained_change_min_observations`` consecutive same-sign values, and the
:class:`~focus_engine.temporal.models.TrajectoryChange` model is itself the Phase 7 exit
criterion expressed as a construction rule. The gap between an
:class:`~focus_engine.temporal.models.ChangeKind.ANOMALY` and a ``TRAJECTORY`` is
therefore a count, not a judgement call made at read time.

**Absence is reported as absence.** A learner with no measurable deviations gets
``INSUFFICIENT_DATA`` with a reason, and a validator refuses to pair that state with any
evidence. The failure this prevents is the most seductive one in behavioural analytics:
reporting ``STABLE`` because nothing has been observed to go wrong. A deviation that could
not be standardised — a cold start with no dispersion — is counted and reported as
unmeasurable, never dropped, because dropping it would silently shorten the run that the
persistence rule counts.

**The standing of the inference is the weakest of its inputs.** A state aggregates several
dimensions, each measured against its own baseline, and the state is labelled with the
lowest maturity among them. A state supported partly by a population prior is not a
personal inference, and the layer cannot present it as one.

**Time only moves forward, from an injected clock.** The engine takes a
:class:`~focus_engine.utils.clock.Clock` rather than reading the host clock, so the
transition behaviour is deterministic and testable, and it refuses an observation at or
before the state's reference time. A state built from out-of-order inputs would be a
function of arrival order rather than of behaviour.

The engine is stateless. A :class:`~focus_engine.temporal.models.TemporalState` is an
immutable value passed in and returned, exactly as in the baseline layer, so a
characterisation is reproducible by replay and concurrent callers cannot interleave into a
shared accumulator.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import NamedTuple

from focus_engine.baseline.models import BaselineDimension, BaselineMaturity, basis_for
from focus_engine.configuration.thresholds import TemporalSettings
from focus_engine.schemas.primitives import BehavioralEngagementState
from focus_engine.schemas.versioning import TemporalVersion
from focus_engine.temporal.models import (
    TEMPORAL_STATE_V1,
    ChangeKind,
    DeviationPoint,
    StateEvidence,
    StateTransition,
    TemporalObservation,
    TemporalState,
    TemporalTrack,
    TrajectoryChange,
    TrendCharacter,
    TrendDirection,
    weakest_maturity,
)
from focus_engine.temporal.statistics import (
    change_per_observation,
    current_sign_run,
    deviation_dispersion,
    is_sustained,
    mean_standardised,
    signed_direction,
)
from focus_engine.utils.clock import Clock, SystemClock
from focus_engine.utils.determinism import content_digest

__all__ = [
    "TRACKED_DIMENSIONS",
    "TemporalEngine",
    "TemporalStateError",
]

#: The dimensions the temporal engine tracks.
#:
#: Every baseline dimension is tracked, and the tuple exists so that a state always has a
#: track for each one. A state built from a partial set of dimensions would be comparable
#: with another state only by luck, and two states that summarised different dimensions
#: would be silently averaged together by any consumer that did not check.
TRACKED_DIMENSIONS: tuple[BaselineDimension, ...] = tuple(BaselineDimension)


class TemporalStateError(ValueError):
    """Raised when an observation cannot legitimately advance a temporal state.

    A ``ValueError`` because every case is a defective argument rather than a missing
    record: an observation for the wrong learner, or one that would move the state
    backwards in time.
    """


class _DimensionAnalysis(NamedTuple):
    """What one dimension contributes to the aggregate state.

    Carried as a named tuple rather than as a model because it is an intermediate value
    that exists only inside :meth:`TemporalEngine.ingest`. It is never stored, so
    validating and versioning it would be cost without benefit.
    """

    track: TemporalTrack
    run: int
    run_mean: float
    direction: TrendDirection
    dispersion: float | None
    standardised: tuple[float, ...]
    trajectory: TrajectoryChange | None


class TemporalEngine:
    """Maintains and characterises one learner's temporal behavioural state.

    Args:
        settings: Temporal configuration. Defaults to
            :class:`~focus_engine.configuration.thresholds.TemporalSettings`.
        clock: Time source used when a caller does not supply the reference time. The
            clock is injected so that state transitions are reproducible in tests; nothing
            in this module reads the host clock directly.
        temporal_version: The temporal definition set to compute.

    The engine holds no per-learner state. A
    :class:`~focus_engine.temporal.models.TemporalState` is passed in and returned, so
    the same engine serves any number of learners and a characterisation can be
    reproduced by replaying its observations.
    """

    __slots__ = ("_clock", "_settings", "_temporal_version")

    def __init__(
        self,
        settings: TemporalSettings | None = None,
        clock: Clock | None = None,
        temporal_version: TemporalVersion = TEMPORAL_STATE_V1,
    ) -> None:
        """Create an engine.

        Args:
            settings: Temporal configuration. Defaults to
                :class:`~focus_engine.configuration.thresholds.TemporalSettings`.
            clock: Time source. Defaults to :class:`~focus_engine.utils.clock.SystemClock`.
            temporal_version: The temporal definition set to compute.
        """
        self._settings = settings if settings is not None else TemporalSettings()
        self._clock = clock if clock is not None else SystemClock()
        self._temporal_version = temporal_version

    @property
    def settings(self) -> TemporalSettings:
        """The configuration in force.

        Returns:
            The temporal settings.
        """
        return self._settings

    @property
    def clock(self) -> Clock:
        """The time source in force.

        Returns:
            The injected clock.
        """
        return self._clock

    @property
    def temporal_version(self) -> TemporalVersion:
        """The temporal definition set this engine computes.

        Returns:
            The version identifier.
        """
        return self._temporal_version

    def settings_fingerprint(self) -> str:
        """Fingerprint the configuration that shapes a state.

        Stored on every state, so a state cannot later be read as though it had been
        produced under thresholds it never saw. The same argument as the baseline
        layer's fingerprint: a recorded characterisation is only interpretable relative
        to the definitions that made it.

        Returns:
            A stable hex digest.
        """
        return content_digest(
            {
                "temporal_version": self._temporal_version,
                "dimensions": [d.value for d in TRACKED_DIMENSIONS],
                "settings": repr(self._settings),
            }
        )

    def create_state(self, learner_id: str, created_at: datetime | None = None) -> TemporalState:
        """Start an empty temporal state for a learner.

        The initial state is ``INSUFFICIENT_DATA`` with a reason, not ``STABLE``. A
        learner about whom nothing has been observed is not a learner who is doing well,
        and the difference is the whole reason the absence is spelled out.

        Args:
            learner_id: Pseudonymous learner identifier.
            created_at: The instant the state starts. Defaults to the clock's current
                instant.

        Returns:
            A state carrying no deviations and one explanatory transition.
        """
        moment = created_at if created_at is not None else self._clock.now()
        evidence = StateEvidence(
            state=BehavioralEngagementState.INSUFFICIENT_DATA,
            maturity=BaselineMaturity.NEW,
            basis=basis_for(BaselineMaturity.NEW),
            observations=0,
            reason=("no deviation has been observed yet, so there is no pattern to characterise"),
        )
        return TemporalState(
            learner_id=learner_id,
            temporal_version=self._temporal_version,
            reference_time=moment,
            created_at=moment,
            state=BehavioralEngagementState.INSUFFICIENT_DATA,
            evidence=evidence,
            direction=TrendDirection.INSUFFICIENT_DATA,
            character=TrendCharacter.FLAT,
            change=ChangeKind.NONE,
            settings_fingerprint=self.settings_fingerprint(),
            tracks=self._empty_tracks(),
        )

    def ingest(self, state: TemporalState, observation: TemporalObservation) -> TemporalState:
        """Fold one observation into a state.

        Args:
            state: The state to advance.
            observation: The deviations measured at a single instant.

        Returns:
            A new state. The input is unchanged.

        Raises:
            TemporalStateError: If the observation belongs to another learner, or is not
                strictly later than the state's reference time.
        """
        self._require_acceptable(state, observation)

        analyses = tuple(
            self._analyse(state, observation, dimension) for dimension in TRACKED_DIMENSIONS
        )
        dominant = self._dominant(analyses)

        state_value, character, change, reason = self._classify(dominant)
        evidence = self._evidence(analyses, dominant, state_value, reason)
        tracks = tuple(item.track for item in analyses)
        # The state's trajectory list is rebuilt from the tracks' ledgers rather than
        # accumulated. Accumulating it would let the two drift: the ledgers are bounded, so
        # an accumulated list would eventually hold trajectories no ledger could vouch for,
        # and the state's own validator — rightly — refuses a trajectory its track never
        # emitted.
        trajectories = self._recorded_trajectories(tracks)
        transition = self._transition(state, state_value, evidence, observation.computed_at)

        return TemporalState(
            learner_id=state.learner_id,
            temporal_version=self._temporal_version,
            reference_time=observation.computed_at,
            created_at=state.created_at,
            state=state_value,
            evidence=evidence,
            direction=dominant.direction if dominant is not None else state.direction,
            character=character,
            change=change,
            run=evidence.consecutive,
            # A run is only asserted alongside a state that stands behind it. During
            # warm-up a track may hold a run while the state is still INSUFFICIENT_DATA,
            # and reporting a start time for a run of zero would claim a run the state
            # has just declined to assert.
            run_started_at=(
                None
                if state_value is BehavioralEngagementState.INSUFFICIENT_DATA
                else self._run_started_at(dominant)
            ),
            origins=state.origins | observation.origins,
            settings_fingerprint=self.settings_fingerprint(),
            tracks=tracks,
            transitions=(*state.transitions, transition),
            trajectories=trajectories,
        )

    def ingest_many(
        self, state: TemporalState, observations: Sequence[TemporalObservation]
    ) -> TemporalState:
        """Fold a sequence of observations in order.

        Args:
            state: The starting state.
            observations: Observations in ascending reference-time order.

        Returns:
            The advanced state.

        Raises:
            TemporalStateError: If any observation is unattributable or out of order.
        """
        current = state
        for observation in observations:
            current = self.ingest(current, observation)
        return current

    def trajectory_for(
        self, state: TemporalState, dimension: BaselineDimension
    ) -> TrajectoryChange | None:
        """Return the most recent trajectory recorded for a dimension.

        Args:
            state: The state to search.
            dimension: The dimension of interest.

        Returns:
            The most recent trajectory for that dimension, or ``None`` when the dimension
            has never produced one.
        """
        matches = [item for item in state.trajectories if item.dimension is dimension]
        return matches[-1] if matches else None

    def _require_acceptable(self, state: TemporalState, observation: TemporalObservation) -> None:
        """Refuse an observation that cannot legitimately advance a state.

        Args:
            state: The state being advanced.
            observation: The candidate observation.

        Raises:
            TemporalStateError: If the observation belongs to another learner, or is not
                strictly later than the state's reference time.
        """
        if observation.learner_id != state.learner_id:
            raise TemporalStateError(
                f"observation belongs to learner {observation.learner_id!r} but the state "
                f"is for {state.learner_id!r}; a temporal characterisation is never built "
                "from another learner's behaviour"
            )
        if observation.computed_at <= state.reference_time:
            raise TemporalStateError(
                f"observation at {observation.computed_at.isoformat()} is not later than the "
                f"state's reference time at {state.reference_time.isoformat()}. A temporal "
                "state is a function of observed time; accepting an out-of-order "
                "observation would make it a function of arrival order instead."
            )

    def _analyse(
        self,
        state: TemporalState,
        observation: TemporalObservation,
        dimension: BaselineDimension,
    ) -> _DimensionAnalysis:
        """Fold one dimension's deviation in and derive its current run.

        The window is bounded and decimated by age, so a long-lived learner's state
        reflects recent behaviour rather than an unbounded accumulation whose earliest
        observation is from months ago.

        Args:
            state: The state being advanced.
            observation: The observation carrying this dimension's deviation.
            dimension: The dimension to analyse.

        Returns:
            The dimension's updated track and its derived run, direction, and dispersion.
        """
        deviation = next(
            (item for item in observation.deviations if item.dimension is dimension), None
        )
        track = self._track_for(state, dimension)
        if deviation is None:
            return self._derive(track, record_trajectory=False)

        point = DeviationPoint(
            dimension=dimension,
            value=deviation.value,
            standardised=deviation.standardised,
            reason=deviation.reason,
            maturity=deviation.maturity,
            basis=deviation.basis,
            computed_at=deviation.computed_at,
            observation_index=self._next_index(track),
        )
        updated = TemporalTrack.model_validate(
            {
                **track.model_dump(),
                "points": self._bounded(track.points, point),
                "recorded": track.recorded + 1,
                "latest_at": deviation.computed_at,
                "maturity": deviation.maturity,
                "basis": deviation.basis,
            }
        )
        analysis = self._derive(updated)
        if analysis.trajectory is None:
            return analysis
        # The trajectory is added to the track's own ledger as well as to the state. The
        # ledger is what makes the record verifiable: retention is bounded, so a long run
        # eventually loses the points behind it, and a state that could only be checked
        # against its retained points would either reject a correct historical trajectory
        # or lose the only evidence that the departure ever happened.
        recorded = analysis.track.model_validate(
            {
                **analysis.track.model_dump(),
                "trajectories": (
                    *analysis.track.trajectories,
                    analysis.trajectory,
                )[-self._settings.max_trajectories_retained :],
            }
        )
        return _DimensionAnalysis(
            track=recorded,
            run=analysis.run,
            run_mean=analysis.run_mean,
            direction=analysis.direction,
            dispersion=analysis.dispersion,
            standardised=analysis.standardised,
            trajectory=analysis.trajectory,
        )

    def _next_index(self, track: TemporalTrack) -> int:
        """Return the observation index for the next point in a dimension's window.

        Taken from the last retained index rather than from the window's length, so that
        indices stay unique and increasing after older points have been discarded. A
        re-used index would let two distinct observations look like the same one to
        anything reading the index, and would break the ordering invariant the track
        validates.

        Args:
            track: The dimension's current track.

        Returns:
            The next index, or ``0`` for an empty window.
        """
        return track.points[-1].observation_index + 1 if track.points else 0

    def _bounded(
        self, points: tuple[DeviationPoint, ...], point: DeviationPoint
    ) -> tuple[DeviationPoint, ...]:
        """Append a point, dropping the oldest once the window is full.

        Args:
            points: The retained window, oldest first.
            point: The point to append, already carrying its final index.

        Returns:
            The new window, bounded by the configured retention.
        """
        return (*points, point)[-self._settings.max_points_retained :]

    def _derive(
        self, track: TemporalTrack, *, record_trajectory: bool = True
    ) -> _DimensionAnalysis:
        """Derive the current run, direction, and dispersion for one dimension.

        Args:
            track: The dimension's track.
            record_trajectory: Whether this ingest may emit a trajectory. A dimension
                absent from the current observation has an unchanged track, so its run is
                unchanged too. Re-reading that run would re-emit a trajectory for a run
                that was already recorded when it first became sustained, and one
                sustained change would be counted once per observation that happened not
                to mention the dimension.

        Returns:
            The dimension's analysis, including a trajectory when the run has just become
            sustained and this ingest is the one that completed it.
        """
        standardised = tuple(
            point.standardised for point in track.points if point.standardised is not None
        )
        rate_window = standardised[-self._settings.rate_of_change_window :]
        # The run is measured over the whole retained window, not over the rate window.
        # Rate and persistence are different questions with different natural spans: a
        # slope is a local property and wants a short window, while "how many
        # consecutive observations has this learner been off their baseline" is a
        # property of the retained history. Reading the run from the rate window would
        # cap persistence at the window length, so a twelve-observation departure would
        # be reported as a five-observation one.
        run, run_mean = current_sign_run(standardised, self._settings.material_deviation_min)
        # Direction describes the run, not the approach to it. Reading the slope across a
        # window that reaches back to on-baseline values would describe the *onset* of the
        # departure rather than its present course: a run of -3.0, -2.0, -1.0 is shrinking,
        # but a window including the 0.0 it departed from has a rising signed slope and
        # would report the learner as moving further away. A run of one has no slope of its
        # own, so it falls back to the window, where "growing away from the baseline" is
        # the correct and only available reading.
        direction = signed_direction(standardised[-run:] if run >= 2 else rate_window)
        dispersion = deviation_dispersion(standardised[-self._settings.stability_window :])
        return _DimensionAnalysis(
            track=track,
            run=run,
            run_mean=run_mean,
            direction=direction,
            dispersion=dispersion,
            standardised=standardised,
            trajectory=(
                self._trajectory(track, standardised, run, direction, run_mean)
                if record_trajectory
                else None
            ),
        )

    def _trajectory(
        self,
        track: TemporalTrack,
        standardised: tuple[float, ...],
        run: int,
        direction: TrendDirection,
        run_mean: float,
    ) -> TrajectoryChange | None:
        """Build a trajectory record when a run has just become sustained.

        The record is emitted exactly once per run, at the observation where the run first
        reaches ``sustained_change_min_observations``. Emitting on every subsequent
        observation would produce a new trajectory for every point of a single sustained
        change, inflating the count and destroying the ability to tell one change from
        many. Requiring ``run == minimum`` rather than ``run >= minimum`` is what makes
        that once-only guarantee: a run grows by exactly one observation at a time, so it
        passes through the threshold value once.

        Args:
            track: The dimension's track, used for maturity and basis.
            standardised: The dimension's standardised deviations in order.
            run: The current run length.
            direction: The current direction.
            run_mean: The mean of the current run.

        Returns:
            A trajectory when the run has just crossed the threshold, otherwise ``None``.
        """
        minimum = self._settings.sustained_change_min_observations
        if run != minimum or not is_sustained(run, minimum):
            return None
        if direction is TrendDirection.INSUFFICIENT_DATA or not standardised:
            return None
        run_values = standardised[-run:]
        run_points = [point for point in track.points if point.standardised is not None][-run:]
        if len(run_points) < run:
            return None
        slope = change_per_observation(run_values)
        if slope is None:
            # A run of one observation has no slope. Reaching here would mean reporting a
            # rate of zero for a run that never moved relative to anything, which asserts
            # stability the data does not show. The configuration forbids a sustained
            # minimum below two precisely so this cannot arise; refusing beats defaulting.
            return None
        return TrajectoryChange(
            dimension=track.dimension,
            direction=direction,
            start=run_points[0].computed_at,
            end=run_points[-1].computed_at,
            observations=run,
            first_standardised=run_values[0],
            last_standardised=run_values[-1],
            mean_standardised=run_mean,
            peak_absolute=max(abs(value) for value in run_values),
            rate_per_observation=slope,
            duration_seconds=max(
                0.0,
                (run_points[-1].computed_at - run_points[0].computed_at).total_seconds(),
            ),
            maturity=track.maturity,
            basis=track.basis,
        )

    def _recorded_trajectories(
        self, tracks: Sequence[TemporalTrack]
    ) -> tuple[TrajectoryChange, ...]:
        """Collect every trajectory the tracks have recorded, newest last.

        Args:
            tracks: The dimension tracks making up the state.

        Returns:
            The trajectories across all dimensions, in the order they were recorded.
        """
        collected: list[TrajectoryChange] = []
        for track in tracks:
            collected.extend(track.trajectories)
        collected.sort(key=lambda item: (item.end, item.dimension.value))
        return tuple(collected)

    def _dominant(self, analyses: Sequence[_DimensionAnalysis]) -> _DimensionAnalysis | None:
        """Choose the dimension the aggregate state is read off.

        One dimension has to be chosen, and it has to be chosen once. Reporting a run
        length from one dimension, a direction from another, and a dispersion from a third
        would produce a state describing no learner in particular: the numbers would each
        be real, and together they would be fiction. The dominant dimension is the one
        the state's ``direction``, ``run``, ``run_started_at``, and ``mean_standardised``
        all refer to.

        Dimensions are ranked by how much they can carry a claim, in this order:

        1. The longest current run, because persistence is the difference between an
           excursion and a change.
        2. The largest ``abs(run_mean)``, because among equally persistent departures the
           larger one is the one worth describing.
        3. The largest dispersion, which is the only tie-break that matters when every run
           is zero. A learner can be "stable" only in the sense that nothing is currently
           deviating; the dimension that is furthest from rock-steady is the informative one
           to name, and reporting an arbitrary tied dimension would understate the
           instability.
        4. Tracked-dimension order, so the choice is fully deterministic. A state that
           could be described two different ways is a state that cannot be audited.

        Args:
            analyses: Every dimension's analysis.

        Returns:
            The dominant analysis, or ``None`` when no dimension has a measurable
            deviation at all.
        """
        measurable = [item for item in analyses if item.standardised]
        if not measurable:
            return None
        return max(
            measurable,
            key=lambda item: (
                item.run,
                abs(item.run_mean),
                item.dispersion if item.dispersion is not None else 0.0,
            ),
        )

    def _classify(
        self,
        dominant: _DimensionAnalysis | None,
    ) -> tuple[BehavioralEngagementState, TrendCharacter, ChangeKind, str]:
        """Choose the state, character, and change kind for the aggregate.

        The state ladder is read off the current run, in the same order a reader would: a
        run too short to interpret, a large but isolated excursion, a sustained run, a
        sustained and large run, and finally a run that is shrinking back toward the
        baseline.

        Args:
            dominant: The dimension the state is read off, or ``None`` when nothing is
                measurable.

        Returns:
            The ``(state, character, change, reason)`` tuple.
        """
        if dominant is None:
            return (
                BehavioralEngagementState.INSUFFICIENT_DATA,
                TrendCharacter.FLAT,
                ChangeKind.NONE,
                "no deviation could be standardised, so no state can be characterised",
            )

        run = dominant.run
        mean = dominant.run_mean
        minimum = self._settings.sustained_change_min_observations
        required = self._settings.state_min_observations

        if len(dominant.standardised) < required:
            return (
                BehavioralEngagementState.INSUFFICIENT_DATA,
                TrendCharacter.FLAT,
                ChangeKind.NONE,
                f"only {len(dominant.standardised)} measurable deviation(s) are available "
                f"and {required} are required before a state can be asserted; a single "
                "measurement cannot distinguish a learner at their baseline from one "
                "who happens to agree with it once",
            )

        if run == 0:
            if dominant.standardised and all(
                abs(value) < self._settings.material_deviation_min
                for value in dominant.standardised
            ):
                return (
                    BehavioralEngagementState.STABLE,
                    TrendCharacter.FLAT,
                    ChangeKind.STABLE,
                    "every recorded deviation is within noise of the personal baseline",
                )
            return (
                BehavioralEngagementState.STABLE,
                TrendCharacter.FLAT,
                ChangeKind.STABLE,
                "the most recent deviation is within noise of the personal baseline",
            )

        if not is_sustained(run, minimum):
            return (
                BehavioralEngagementState.SLIGHT_DEVIATION,
                TrendCharacter.ISOLATED,
                ChangeKind.ANOMALY,
                f"a deviation of {mean:+.2f} dispersions has not persisted across "
                f"{run} consecutive observation(s), so it is an excursion rather than a "
                "change",
            )

        sustained = (
            f"the deviation has persisted across {run} consecutive observations at "
            f"{mean:+.2f} dispersions"
        )
        if abs(mean) >= self._settings.high_deviation_standardised:
            return (
                BehavioralEngagementState.HIGH_DEVIATION,
                TrendCharacter.PERSISTENT,
                ChangeKind.TRAJECTORY,
                f"{sustained}, which is a large and sustained departure",
            )
        if dominant.direction is TrendDirection.WORSENING:
            return (
                BehavioralEngagementState.DECLINING,
                TrendCharacter.PERSISTENT,
                ChangeKind.TRAJECTORY,
                f"{sustained} and the distance from the baseline is growing",
            )
        if dominant.direction is TrendDirection.IMPROVING:
            return (
                BehavioralEngagementState.RECOVERING,
                TrendCharacter.PERSISTENT,
                ChangeKind.RECOVERY,
                f"{sustained} and the distance from the baseline is shrinking",
            )
        return (
            BehavioralEngagementState.INCREASING_DEVIATION,
            TrendCharacter.PERSISTENT,
            ChangeKind.TRAJECTORY,
            f"{sustained} at a steady distance from the baseline",
        )

    def _evidence(
        self,
        analyses: Sequence[_DimensionAnalysis],
        dominant: _DimensionAnalysis | None,
        state_value: BehavioralEngagementState,
        reason: str,
    ) -> StateEvidence:
        """Build the evidence trace for the chosen state.

        The dispersion is reported from the dominant dimension's window rather than from
        the pooled one. Pooling dispersions across dimensions mixes units — a
        standardised deviation is comparable in magnitude, but the variance of accuracy and
        latency deviations reflects different measurement noise, and averaging them would
        produce a number that describes neither.

        Args:
            analyses: Every dimension's analysis.
            dominant: The dimension the state is read off, or ``None`` when nothing is
                measurable.
            state_value: The state being asserted.
            reason: Why the state is what it is.

        Returns:
            The evidence trace.
        """
        maturity = weakest_maturity([item.track.maturity for item in analyses])
        observations = sum(len(item.standardised) for item in analyses)

        if dominant is None:
            return StateEvidence(
                state=state_value,
                maturity=maturity,
                basis=basis_for(maturity),
                observations=0,
                reason=reason,
            )

        if state_value is BehavioralEngagementState.INSUFFICIENT_DATA:
            return StateEvidence(
                state=state_value,
                maturity=maturity,
                basis=basis_for(maturity),
                observations=observations,
                consecutive=0,
                mean_standardised=mean_standardised(dominant.standardised),
                dispersion=dominant.dispersion,
                reason=reason,
            )

        run_points = [point for point in dominant.track.points if point.standardised is not None][
            -dominant.run :
        ]
        duration = 0.0
        if len(run_points) >= 2:
            duration = max(
                0.0,
                (run_points[-1].computed_at - run_points[0].computed_at).total_seconds(),
            )
        return StateEvidence(
            state=state_value,
            maturity=maturity,
            basis=basis_for(maturity),
            observations=observations,
            consecutive=dominant.run,
            duration_seconds=duration,
            mean_standardised=mean_standardised(dominant.standardised),
            dispersion=dominant.dispersion,
            reason=reason,
        )

    def _transition(
        self,
        state: TemporalState,
        state_value: BehavioralEngagementState,
        evidence: StateEvidence,
        at: datetime,
    ) -> StateTransition:
        """Record the transition into the new state.

        Every ingest produces a transition, including a self-transition. A run of
        ``STABLE`` observations is a sequence of no-change transitions, and that sequence is
        the evidence that the state was stable rather than merely currently stable. A
        system that recorded transitions only on change would make the two indistinguishable
        in its own audit trail.

        Args:
            state: The state being advanced.
            state_value: The new state value.
            evidence: The evidence supporting the new state.
            at: The instant of the transition.

        Returns:
            The transition record.
        """
        return StateTransition(
            from_state=state.state,
            to_state=state_value,
            at=at,
            consecutive=evidence.consecutive,
            reason=(f"characterised from {state.observations + 1} observation(s)"),
        )

    def _run_started_at(self, dominant: _DimensionAnalysis | None) -> datetime | None:
        """Return when the dominant dimension's current run began.

        The run's start is read from the same dimension as the run's length. Deriving the
        start from a different dimension's window would let the state assert a run that
        began before the observations supporting it, and the model's timestamp validation
        would then reject a state the engine had just built.

        Args:
            dominant: The dimension the state is read off, or ``None`` when nothing is
                measurable.

        Returns:
            The timestamp of the run's first observation, or ``None`` when there is no run
            or the retained window is shorter than the run.
        """
        if dominant is None or dominant.run == 0:
            return None
        points = [point for point in dominant.track.points if point.standardised is not None]
        if len(points) < dominant.run:
            return None
        return points[-dominant.run].computed_at

    def _track_for(self, state: TemporalState, dimension: BaselineDimension) -> TemporalTrack:
        """Return a dimension's track, creating an empty one if absent.

        Args:
            state: The state to read.
            dimension: The dimension of interest.

        Returns:
            The dimension's track.
        """
        for track in state.tracks:
            if track.dimension is dimension:
                return track
        return TemporalTrack(
            dimension=dimension,
            max_points_retained=self._settings.max_points_retained,
        )

    def _empty_tracks(self) -> tuple[TemporalTrack, ...]:
        """Build a complete set of empty tracks.

        Every tracked dimension is present from the start, so a consumer can read any
        dimension without a presence check and a state is comparable with another by
        construction.

        Returns:
            One empty track per tracked dimension.
        """
        return tuple(
            TemporalTrack(
                dimension=dimension,
                max_points_retained=self._settings.max_points_retained,
            )
            for dimension in TRACKED_DIMENSIONS
        )
