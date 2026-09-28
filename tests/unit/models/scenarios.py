"""Shared builders for the model-layer tests (Phase 8).

Building a feature vector by hand is verbose enough that a typo in it would produce a test
failure attributed to the model layer rather than to the test data. These helpers keep the
construction in one place so that a change to the feature contract shows up as one failing
builder rather than as forty confusing ones.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from focus_engine.features import (
    FEATURE_SET_V1,
    INSUFFICIENT_DATA_MARKER,
    FeatureAvailability,
    FeatureName,
    FeatureValue,
    TypedFeatureValue,
)

#: The reference time every synthetic vector is anchored to, so that two vectors differ
#: only in the values a test deliberately varies.
BASE_TIME = datetime(2025, 9, 27, 10, 0, 0, tzinfo=UTC)

#: Seeds the scenario generator. Fixed so that a dataset does not change when an unrelated
#: test happens to draw from the global NumPy generator first.
SEED = 20_260_101

#: A complete, plausible value for every feature. Tests override the entries they care
#: about, which keeps an unrelated feature from silently becoming absent.
DEFAULT_VALUES: dict[FeatureName, float | int | str] = {
    FeatureName.SESSION_ELAPSED_SECONDS: 600.0,
    FeatureName.SESSION_POSITION: 0.5,
    FeatureName.SESSION_EVENT_COUNT: 12,
    FeatureName.CONTENT_DIFFICULTY: 0.6,
    FeatureName.CONTENT_TYPE_ENCODED: "multiple_choice",
    FeatureName.PERFORMANCE_RECENT_ACCURACY: 0.72,
    FeatureName.PERFORMANCE_RECENT_RESPONSE_SECONDS: 21.4,
    FeatureName.PERFORMANCE_ACCURACY_TREND: 0.08,
    FeatureName.PERFORMANCE_RESPONSE_TREND: -0.03,
    FeatureName.INTERVENTION_COUNT: 2,
    FeatureName.INTERVENTION_COOLDOWN_ACTIVE: "false",
    FeatureName.INTERVENTION_SECONDS_SINCE_LAST: 90.0,
    FeatureName.TRAJECTORY_ENCODED: "improving",
    FeatureName.BASELINE_MATURITY_ENCODED: "developing",
}


def make_vector(
    *,
    learner_id: str | None = "learner-0001",
    session_id: str | None = "session-0001",
    offset_seconds: float = 0.0,
    overrides: dict[FeatureName, float | int | str] | None = None,
    absent: tuple[FeatureName, ...] = (),
) -> FeatureValue:
    """Build a feature vector, defaulting every feature to a plausible measured value.

    Args:
        learner_id: The learner identifier, or ``None`` for an unattributed vector.
        session_id: The session identifier, or ``None`` when none was observed.
        offset_seconds: Shift the reference time, for building an ordered series.
        overrides: Replace the default value of specific features.
        absent: Mark these features as ``INSUFFICIENT_DATA`` instead of measured.

    Returns:
        A valid, fully specified feature vector.
    """
    computed_at = BASE_TIME + timedelta(seconds=offset_seconds)
    values = dict(DEFAULT_VALUES)
    if overrides:
        values.update(overrides)

    entries: list[TypedFeatureValue] = []
    for name in FeatureName:
        if name in absent:
            entries.append(
                TypedFeatureValue(
                    name=name,
                    value=INSUFFICIENT_DATA_MARKER,
                    availability=FeatureAvailability.INSUFFICIENT_DATA,
                    reason="the test declared this feature unavailable",
                    computed_at=computed_at,
                    feature_set_version=FEATURE_SET_V1,
                    spec_version="spec-v1",
                )
            )
            continue
        entries.append(
            TypedFeatureValue(
                name=name,
                value=values[name],
                availability=FeatureAvailability.AVAILABLE,
                computed_at=computed_at,
                feature_set_version=FEATURE_SET_V1,
                spec_version="spec-v1",
            )
        )

    return FeatureValue(
        feature_set_version=FEATURE_SET_V1,
        computed_at=computed_at,
        values=tuple(entries),
        learner_id=learner_id,
        session_id=session_id,
    )


def make_labelled_rows(
    count: int,
    *,
    positive: int,
    start_index: int = 0,
    overlap: bool = False,
) -> list[tuple[FeatureValue, int]]:
    """Build a labelled dataset with a controlled class balance.

    The positive rows get features that actually vary with the label, so a model fitted on
    this data has something real to learn. A dataset where the label is noise would let a
    test pass for the wrong reason: a broken model would score as well as a working one.

    Args:
        count: The total number of rows.
        positive: How many of them belong to the positive class.
        start_index: Offset applied to learner and session identifiers, for keeping
            several independently generated datasets distinct.
        overlap: When ``True``, the two classes are drawn from overlapping
            distributions. Separable data cannot distinguish one fitted model from
            another - every fit predicts perfectly and every comparison of two models
            passes trivially - so any test about what a seed or a hyperparameter actually
            changed has to use data where the answer is not forced. Real learners are not
            linearly separable, and tests that assume they are would not be testing the
            model.

    Returns:
        Labelled rows ready for :func:`focus_engine.models.encoding.encode_rows`.

    Raises:
        ValueError: If ``positive`` is not within ``[0, count]``.
    """
    if not 0 <= positive <= count:
        raise ValueError(f"positive={positive} must be within [0, {count}]")

    # A fixed, reproducible jitter. The generator is seeded from the row index rather than
    # from the global RNG, so the dataset does not depend on test execution order.
    rng = np.random.default_rng(SEED)

    rows: list[tuple[FeatureValue, int]] = []
    for index in range(count):
        is_positive = index < positive
        jitter = float(rng.normal(0.0, 0.06)) if overlap else 0.0
        if overlap:
            accuracy = (0.46 if is_positive else 0.54) + jitter
            response = (30.1 if is_positive else 26.4) + 3.0 * jitter
            trend = (0.02 if is_positive else 0.01) + jitter
            elapsed = (900.0 if is_positive else 780.0) + 40.0 * jitter
        else:
            accuracy = 0.31 if is_positive else 0.88
            response = 47.2 if is_positive else 9.4
            trend = -0.21 if is_positive else 0.14
            elapsed = 1800.0 if is_positive else 300.0
        rows.append(
            (
                make_vector(
                    learner_id=f"learner-{start_index + index:04d}",
                    session_id=f"session-{start_index + index:04d}",
                    offset_seconds=float(index),
                    overrides={
                        FeatureName.PERFORMANCE_RECENT_ACCURACY: accuracy,
                        FeatureName.PERFORMANCE_RECENT_RESPONSE_SECONDS: response,
                        FeatureName.PERFORMANCE_ACCURACY_TREND: trend,
                        FeatureName.SESSION_ELAPSED_SECONDS: elapsed,
                    },
                ),
                1 if is_positive else 0,
            )
        )
    return rows
