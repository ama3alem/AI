"""Temporal derivations: direction, rate, stability, and persistence.

These are the four quantities that separate a moment from a change, and each is computed
here rather than inside the engine so that the arithmetic can be tested on its own and
each definition appears exactly once.

**Direction is defined on the standardised deviation, in units of the dimension's own
sign.** The sign of a standardised deviation is meaningful — negative means the
measurement sits below the personal centre, positive above — but for accuracy a positive
deviation is good and for response time it is bad. Rather than encode a per-dimension
goodness table, the direction here is expressed as *distance from the baseline*: a
shrinking absolute deviation is improving whatever the sign. That is a statement about
distance, not about desirability, and the intervention policy is the layer that decides
whether shrinking distance is good.

**Persistence counts sign runs, not magnitude runs.** A deviation oscillating between +1
and +4 dispersions stays positive throughout, so it is one persistent run with a large
spread, not a sequence of excursions. Collapsing that into per-observation steps would
report a learner whose behaviour is volatile and consistent as one whose behaviour is
steadily changing, and the two demand different responses.

**Stability is a dispersion, reported as ``None`` rather than as a high number when
there is nothing to disperse.** Two identical values are perfectly stable, but the
robust dispersion of a single observation is undefined rather than zero, and reporting
zero would claim to have measured stability from one measurement. This module returns
``None`` in that case and lets the caller decide what an unknown means.

**Anomaly versus trajectory is decided by count, and the count is a configuration
input.** :func:`is_sustained` takes the run length and the configured minimum as
arguments rather than reading either from a module global, so a test can drive the
boundary without editing configuration and so the same function serves the engine and a
reviewer reproducing a decision by hand.
"""

from __future__ import annotations

from collections.abc import Sequence

from focus_engine.baseline.models import MAD_NORMAL_SCALE
from focus_engine.temporal.models import TrendDirection

__all__ = [
    "change_per_observation",
    "current_sign_run",
    "deviation_dispersion",
    "is_sustained",
    "mean_standardised",
    "signed_direction",
]


def current_sign_run(values: Sequence[float], material_min: float) -> tuple[int, float]:
    """Measure the current run of same-sign material deviations.

    The run is read backwards from the most recent value, so it describes the run that
    ended at the newest observation rather than the longest run anywhere in the window.
    Those are different questions: a learner who deviated for a week and has since been
    stable for a day has a long history of deviation and a current run of zero, and
    reporting the week's run as current would keep asserting a trajectory that has
    already ended.

    A value smaller than ``material_min`` terminates the run, because a deviation this
    close to zero is arithmetic rather than behaviour. That also means a run cannot pass
    through zero while remaining same-signed, which is correct: nothing happened at that
    observation.

    Args:
        values: Standardised deviations in chronological order.
        material_min: Smallest absolute magnitude treated as a behavioural fact.

    Returns:
        A ``(run_length, mean_of_run)`` pair. Both are zero for an empty sequence or when
        the newest value is immaterial.
    """
    if not values:
        return 0, 0.0
    newest = values[-1]
    if abs(newest) < material_min:
        return 0, 0.0
    sign = 1.0 if newest > 0.0 else -1.0
    run: list[float] = []
    for value in reversed(values):
        if abs(value) < material_min:
            break
        if (value > 0.0 and sign < 0.0) or (value < 0.0 and sign > 0.0):
            break
        run.append(value)
    return len(run), sum(run) / len(run)


def is_sustained(run_length: int, minimum: int) -> bool:
    """Whether a run is long enough to call a change rather than a moment.

    Args:
        run_length: Length of the current same-sign run.
        minimum: The configured number of consecutive observations required. One is the
            smallest sensible value: it asserts that any run at all is sustained, which is
            a legitimate configuration for a caller who has decided their sampling makes
            consecutive observations genuinely independent.

    Returns:
        ``True`` when the run reaches the minimum.
    """
    return run_length >= minimum


def mean_standardised(values: Sequence[float]) -> float | None:
    """Average a set of standardised deviations.

    Args:
        values: The deviations to average.

    Returns:
        The arithmetic mean, or ``None`` for an empty sequence. ``None`` rather than
        ``0.0``, because the mean of nothing is unknown and zero would read as "the
        learner is exactly on their baseline", which is a claim.
    """
    if not values:
        return None
    return sum(values) / len(values)


def deviation_dispersion(values: Sequence[float]) -> float | None:
    """Robust dispersion of a set of standardised deviations.

    A median absolute deviation scaled to be comparable with a standard deviation, for
    the same reason the baseline layer uses one: a single large excursion inflates a
    standard deviation, and an inflated dispersion is exactly what would let a genuine
    sustained change look like ordinary variability.

    Args:
        values: The deviations to summarise. Order is irrelevant.

    Returns:
        The scaled dispersion, or ``None`` when fewer than two values were supplied.
        Identical values yield ``0.0``, which is a real measurement of perfect stability
        rather than an absence, and is kept distinct from the ``None`` returned for a
        single observation.
    """
    if len(values) < 2:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        centre = (ordered[middle - 1] + ordered[middle]) / 2.0
    else:
        centre = ordered[middle]
    deviations = sorted(abs(value - centre) for value in ordered)
    if len(deviations) % 2 == 0:
        mad = (deviations[len(deviations) // 2 - 1] + deviations[len(deviations) // 2]) / 2.0
    else:
        mad = deviations[len(deviations) // 2]
    return MAD_NORMAL_SCALE * mad


def change_per_observation(values: Sequence[float]) -> float | None:
    """Estimate the slope of a run, in units per observation.

    The estimate is the total change across the run divided by the number of *steps*, not
    the number of observations. A run of three points has two steps, and dividing by three
    would understate the rate by half for the shortest runs the persistence rule permits —
    a systematic bias that would grow as the run-length minimum is tuned down.

    The slope is expressed as change in the *signed* standardised deviation, so a run
    becoming more negative yields a negative slope. Callers convert to a direction using
    :func:`signed_direction`.

    Args:
        values: The run's deviations in chronological order.

    Returns:
        The slope, or ``None`` for an empty sequence or a single observation, which
        defines no slope.
    """
    if len(values) < 2:
        return None
    return (values[-1] - values[0]) / (len(values) - 1)


def signed_direction(values: Sequence[float], minimum: int = 2) -> TrendDirection:
    """Classify the direction a run of deviations is moving.

    Args:
        values: Deviations in chronological order, most recent last.
        minimum: How many values are required before a direction is asserted.

    Returns:
        :attr:`~focus_engine.temporal.models.TrendDirection.INSUFFICIENT_DATA` when there
        are fewer than ``minimum`` values or the run is flat, and otherwise the direction
        of the change in absolute distance from the baseline.

    A run is ``IMPROVING`` when its distance from zero is shrinking and ``WORSENING`` when
    it is growing. A run that holds a constant distance is ``STABLE``, which is a
    different claim from ``INSUFFICIENT_DATA``: the evidence was sufficient and showed no
    movement.
    """
    if len(values) < minimum:
        return TrendDirection.INSUFFICIENT_DATA
    slope = change_per_observation(values)
    if slope is None:
        return TrendDirection.STABLE
    if slope > 0.0:
        # A growing signed value means growing distance when positive, and shrinking
        # distance when negative, so the sign of the value has to be read too.
        return TrendDirection.WORSENING if values[-1] > 0.0 else TrendDirection.IMPROVING
    if slope < 0.0:
        return TrendDirection.IMPROVING if values[-1] > 0.0 else TrendDirection.WORSENING
    return TrendDirection.STABLE
