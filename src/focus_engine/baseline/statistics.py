"""Statistical primitives for the personal baseline.

Kept separate from :mod:`focus_engine.baseline.engine` so the estimators can be read,
tested, and reasoned about without also reading the profile-update loop. Nothing here
knows about profiles, learners, or time; these are pure functions over numbers and
counts, which is what makes the choice of estimator a reviewable decision rather than an
emergent property of the update code.

Two rules govern this module.

**A missing estimate is absent, not zero.** Every dispersion function returns ``None``
when its input cannot support an estimate. A single retained observation has no
dispersion, and reporting ``0.0`` would make "no information" indistinguishable from
"perfectly consistent", which are opposites.

**The method is a parameter, never an assumption.** Robust and non-robust estimators are
both available, the choice is made by configuration, and the choice is recorded on every
result. Swapping the estimator changes which deviations are visible, so it must be a
visible change.

Note that this module owns the *spread*. The matching location estimate is a winsorised
exponential mean and lives in
:mod:`focus_engine.baseline.engine`, because bounding the centre's step needs the settings
object and the previous spread, and putting it here would make a pure function depend on
engine state. The two halves of an estimator are documented together even though they are
implemented apart.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from math import sqrt

from focus_engine.baseline.models import MAD_NORMAL_SCALE, BaselineMaturity
from focus_engine.configuration.thresholds import BaselineSettings

__all__ = [
    "decimate",
    "maturity_for",
    "robust_dispersion",
    "sample_dispersion",
]


def maturity_for(observation_count: int, settings: BaselineSettings) -> BaselineMaturity:
    """Map an observation count to a maturity level.

    The thresholds live in :class:`~focus_engine.configuration.thresholds.BaselineSettings`
    rather than here, so the rung that produced a level is configuration that can be
    recorded alongside a result.

    ``min_samples_new`` is not a separate branch. With a strictly increasing ladder it can
    only ever be a lower bound inside the ``NEW`` region, so the three upper thresholds
    determine every boundary. The field is kept because it is validated with the others
    and because a deployment may want a non-zero floor visible in its configuration.

    Args:
        observation_count: Accepted observations behind one dimension.
        settings: The maturity ladder in force.

    Returns:
        The maturity level for that count.

    Raises:
        ValueError: If ``observation_count`` is negative.
    """
    if observation_count < 0:
        raise ValueError(f"observation_count must be non-negative; got {observation_count}")
    if observation_count < settings.min_samples_early:
        return BaselineMaturity.NEW
    if observation_count < settings.min_samples_developing:
        return BaselineMaturity.EARLY
    if observation_count < settings.min_samples_established:
        return BaselineMaturity.DEVELOPING
    return BaselineMaturity.ESTABLISHED


def decimate(retained: Sequence[float], limit: int) -> tuple[float, ...]:
    """Bound a retained window by halving it, keeping the newer of each adjacent pair.

    A baseline that keeps every observation forever has no memory bound, and a baseline
    that discards old observations at a fixed count cannot reach
    :attr:`~focus_engine.baseline.models.BaselineMaturity.ESTABLISHED` for a long-lived
    learner. Halving resolves both: memory stays bounded, the retained sample keeps
    spanning a growing interval, and the most recent values are always present.

    The window is thinned rather than truncated, so its *time span* keeps growing while
    its *resolution* falls. That is the intended trade: a spread estimated from a
    long, coarse window is more stable, and stability is what a baseline is for. A
    truncation would instead freeze the span at ``limit`` and silently stop representing
    anything the learner did after that point.

    Halving is repeated until the window fits, because one halving of a window twice the
    limit still exceeds it, and a function that returns more than it was asked for is
    worse than one that returns less.

    An odd-length window is paired with the newest value held back rather than dropped.
    Slicing a flat odd window would silently discard the most recent observation, which
    is the one value a baseline is least able to lose.

    Args:
        retained: The current window, oldest first.
        limit: The maximum number of values to keep.

    Returns:
        A window no longer than ``limit``, always ending with the newest value.

    Raises:
        ValueError: If ``limit`` is below one.
    """
    if limit < 1:
        raise ValueError(f"retention limit must be at least 1; got {limit}")
    window = tuple(retained)
    if limit == 1:
        return (window[-1],) if window else window
    while len(window) > limit:
        window = (*window[1::2], *window[-1:]) if len(window) % 2 else window[1::2]
    return window


def robust_dispersion(centre: float, retained: Sequence[float]) -> float | None:
    """Estimate dispersion as a scaled median absolute deviation about ``centre``.

    Robust to the outliers that motivate it: a single extreme response time moves this
    estimate by at most a small step, where it would move a standard deviation
    substantially. The 1.4826 scale factor from
    :data:`~focus_engine.baseline.models.MAD_NORMAL_SCALE` makes the result comparable to
    a standard deviation on ordinary data, so a threshold written against one holds
    against the other.

    The median is taken of distances from the *supplied* centre rather than from the
    sample median. That is a deliberate departure from the textbook definition, and it
    matters here: the published centre is a time-weighted mean, so measuring dispersion
    about the unweighted median would pair a centre with a spread derived from a
    different centre, and a standardised deviation would then be measured against a
    reference that was never reported.

    Args:
        centre: The centre the spread should be measured about.
        retained: The retained observations.

    Returns:
        The scaled median absolute deviation, or ``None`` when fewer than two values are
        retained and no dispersion is defined.
    """
    if len(retained) < 2:
        return None
    deviations = sorted(abs(value - centre) for value in retained)
    return float(statistics.median(deviations)) * MAD_NORMAL_SCALE


def sample_dispersion(centre: float, retained: Sequence[float]) -> float | None:
    """Estimate dispersion as a root-mean-square distance about ``centre``.

    The classic outlier-sensitive alternative, retained as configuration rather than
    deleted, so a deployment that has reason to prefer it can record that choice instead of
    inheriting one. The Bessel-corrected divisor keeps the estimate unbiased for a
    two-point sample; with two values it equals half the distance between them.

    Args:
        centre: The centre the spread should be measured about.
        retained: The retained observations.

    Returns:
        The dispersion, or ``None`` when fewer than two values are retained.
    """
    if len(retained) < 2:
        return None
    total = sum((value - centre) ** 2 for value in retained)
    return sqrt(total / (len(retained) - 1))
