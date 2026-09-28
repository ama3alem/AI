"""Temporal state layer (Phase 7).

Answers the question the baseline layer cannot: is this a moment, or is this a change?
The baseline reports how far a measurement sits from normal; the temporal layer reports
whether that distance is heading anywhere, and whether a trajectory has become sustained.

Three properties make the distinction trustworthy rather than merely convenient.

**Absence is reported, never inferred.** A learner with no deviation history has no state.
The engine reports ``INSUFFICIENT_DATA`` with a reason, and a validator refuses to pair
that state with any evidence of movement. Substituting ``STABLE`` because nothing has gone
wrong is the single most seductive failure in behavioural analytics, and it is prevented
by making the state unconstructible rather than by asking reviewers to notice it.

**A single anomalous observation can never be a trajectory.** The engine keeps a bounded
per-dimension window of standardised deviations and reads the current run backwards from
the newest. The run must reach a configured length before a
:class:`~focus_engine.temporal.models.TrajectoryChange` is recorded, and the trajectory
model itself refuses to exist for a run shorter than its own ``consecutive`` count. This
is the Phase 7 exit criterion expressed as a construction rule rather than a review
convention.

**The standing of the inference travels with the state.** Every
:class:`~focus_engine.temporal.models.StateEvidence` carries the baseline maturity and
basis in force at the time the state was computed. A state derived partly from a population
prior is not a personal inference, and the weakest maturity among the contributing
dimensions is the one recorded.

Importing this package pulls in the temporal layer only. It does not import the feature,
prediction, or any later layer, so the dependency direction described in
``ARCHITECTURE.md`` holds at the package level.
"""

from __future__ import annotations

from focus_engine.temporal.engine import (
    TRACKED_DIMENSIONS,
    TemporalEngine,
    TemporalStateError,
)
from focus_engine.temporal.models import (
    TEMPORAL_STATE_V1,
    TREND_DIRECTIONS,
    BehavioralEngagementState,
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

__all__ = [
    "TEMPORAL_STATE_V1",
    "TRACKED_DIMENSIONS",
    "TREND_DIRECTIONS",
    "BehavioralEngagementState",
    "ChangeKind",
    "DeviationPoint",
    "StateEvidence",
    "StateTransition",
    "TemporalEngine",
    "TemporalObservation",
    "TemporalState",
    "TemporalStateError",
    "TemporalTrack",
    "TrajectoryChange",
    "TrendCharacter",
    "TrendDirection",
    "change_per_observation",
    "current_sign_run",
    "deviation_dispersion",
    "is_sustained",
    "mean_standardised",
    "signed_direction",
    "weakest_maturity",
]
