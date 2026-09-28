"""Personal baseline engine: what does normal look like for this learner?

The engine maintains one :class:`~focus_engine.baseline.models.BaselineProfile` per
learner and answers two questions about it: what the learner's normal value for a
dimension is, and how far a fresh measurement sits from that normal.

Four decisions define the behaviour, and each exists to prevent a specific failure.

**Cold start is a labelled fallback, never a silent default.** Below
``min_samples_early`` the personal centre is not used for inference at all: the engine
returns the population prior, labelled as such. If no prior was supplied the reference is
reported unavailable with a reason. A hard-coded ``typical`` value would make a guess
indistinguishable from an estimate, and this layer is the last place before that guess
would be presented as a measurement.

**Dispersion is robust by default.** A single 300-second outlier raises a mean and
inflates a standard deviation, and the wider baseline that results is precisely what
would mask the real change the system exists to notice. The default estimator is a median
absolute deviation, scaled to be comparable with a standard deviation. The non-robust
estimator remains available as configuration rather than being removed, so the choice is
recorded and reversible instead of assumed.

**The centre is bounded, the spread is robust.** The centre is an exponentially weighted
mean — recent behaviour carries more of it — but the per-observation step is capped at a
configured number of current dispersions, so a single extreme value cannot relocate the
baseline. The spread is recomputed from a bounded window of retained observations, because
a weighted median absolute deviation has no useful closed form and an unbounded window has
no bound on memory. Both the centre and the window are stored, because a centre that cannot
be re-derived from the evidence behind it cannot be audited.

**Time only moves forward.** An observation older than, or simultaneous with, the
profile's last update is refused. A baseline that could absorb an out-of-order event
would be a function of arrival order rather than of behaviour, and the resulting profile
would not be reproducible from the event stream.

Nothing here classifies a deviation, attaches severity, or judges whether a change
matters. This layer reports distance; interpretation belongs to the temporal and
uncertainty engines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final, NamedTuple

from focus_engine.baseline.models import (
    BASELINE_DIMENSIONS,
    PERSONAL_BASELINE_V1,
    BaselineDimension,
    BaselineMaturity,
    BaselineProfile,
    BaselineReference,
    DeviationResult,
    DimensionStatistics,
    PopulationPriorSet,
    ReferenceSource,
    StatisticsMethod,
    basis_for,
    dimension_range,
)
from focus_engine.baseline.statistics import (
    decimate,
    maturity_for,
    robust_dispersion,
    sample_dispersion,
)
from focus_engine.configuration.thresholds import BaselineSettings
from focus_engine.features.models import (
    FeatureAvailability,
    FeatureName,
    FeatureValue,
)
from focus_engine.schemas.primitives import InferenceBasis
from focus_engine.schemas.versioning import BaselineVersion
from focus_engine.utils.determinism import content_digest

__all__ = [
    "DIMENSION_SOURCE_FEATURES",
    "BaselineEngine",
    "BaselineUpdateError",
    "maturity_for",
]


DIMENSION_SOURCE_FEATURES: Final[Mapping[BaselineDimension, FeatureName]] = {
    BaselineDimension.ACCURACY: FeatureName.PERFORMANCE_RECENT_ACCURACY,
    BaselineDimension.RESPONSE_SECONDS: FeatureName.PERFORMANCE_RECENT_RESPONSE_SECONDS,
    BaselineDimension.SESSION_SECONDS: FeatureName.SESSION_ELAPSED_SECONDS,
    BaselineDimension.CONTENT_DIFFICULTY: FeatureName.CONTENT_DIFFICULTY,
}
"""Which feature supplies each baseline dimension.

One source per dimension, deliberately. Two features could both be argued to measure
``accuracy`` — the rolling window and the accuracy trend — and choosing between them is a
definitional decision that belongs in one reviewable table rather than in a per-dimension
branch inside the update loop.
"""


class BaselineUpdateError(ValueError):
    """Raised when an observation cannot be attributed to a profile.

    A ``ValueError`` because every case is a defective argument rather than a missing
    record: an unattributable vector, a vector for the wrong learner, or one that would
    move the profile backwards in time.
    """


class _Extracted(NamedTuple):
    """The outcome of reading one dimension out of a feature vector.

    ``value`` and ``rejection`` are mutually exclusive, and the three states are
    distinct on purpose: a feature that was never computed is not a defect and must not
    be counted as a rejected observation, while a feature that was computed as the wrong
    type is a defect and must be visible.
    """

    value: float | None
    rejection: str | None


def _extract(vector: FeatureValue, dimension: BaselineDimension) -> _Extracted:
    """Read one dimension's observation out of a feature vector.

    Args:
        vector: The feature vector to read.
        dimension: The dimension to read.

    Returns:
        The observation, a rejection reason, or an absent value when the source feature
        reported ``INSUFFICIENT_DATA`` or is not in the vector.
    """
    name = DIMENSION_SOURCE_FEATURES[dimension]
    for entry in vector.values:
        if entry.name is not name:
            continue
        if entry.availability is FeatureAvailability.INSUFFICIENT_DATA:
            return _Extracted(None, None)
        if isinstance(entry.value, bool) or not isinstance(entry.value, int | float):
            return _Extracted(
                None,
                f"source feature {name.value} produced a {type(entry.value).__name__}, "
                f"which cannot serve as a {dimension.value} observation",
            )
        return _Extracted(float(entry.value), None)
    return _Extracted(None, None)


def _with(item: DimensionStatistics, **changes: object) -> DimensionStatistics:
    """Return a copy of a statistics record with fields replaced and revalidated.

    ``model_copy`` would skip validation, which is the one thing these models exist to
    do, so the updated record is rebuilt through the constructor instead.

    Args:
        item: The record to copy.
        **changes: Fields to replace.

    Returns:
        The revalidated record.
    """
    return DimensionStatistics.model_validate(item.model_dump() | changes)


def _with_profile(profile: BaselineProfile, **changes: object) -> BaselineProfile:
    """Return a copy of a profile with fields replaced and revalidated.

    Args:
        profile: The profile to copy.
        **changes: Fields to replace.

    Returns:
        The revalidated profile.
    """
    return BaselineProfile.model_validate(profile.model_dump() | changes)


class BaselineEngine:
    """Maintains and queries one learner's personal baseline.

    The engine is stateless. A profile is an immutable value that is passed in and
    returned, so a baseline is reproducible from its event stream by replay and two
    concurrent callers cannot interleave into a shared accumulator.
    """

    __slots__ = ("_baseline_version", "_prior", "_settings")

    def __init__(
        self,
        settings: BaselineSettings | None = None,
        prior: PopulationPriorSet | None = None,
        baseline_version: BaselineVersion = PERSONAL_BASELINE_V1,
    ) -> None:
        """Create an engine.

        Args:
            settings: Baseline configuration. Defaults to
                :class:`~focus_engine.configuration.thresholds.BaselineSettings`.
            prior: Population priors for cold start. ``None`` means a cold start is
                reported unavailable rather than guessed.
            baseline_version: The baseline definition set to compute.
        """
        self._settings = settings if settings is not None else BaselineSettings()
        self._prior = prior
        self._baseline_version = baseline_version

    @property
    def settings(self) -> BaselineSettings:
        """The configuration in force.

        Returns:
            The baseline settings.
        """
        return self._settings

    @property
    def prior(self) -> PopulationPriorSet | None:
        """The population priors available for cold start.

        Returns:
            The prior set, or ``None`` when none was supplied.
        """
        return self._prior

    @property
    def baseline_version(self) -> BaselineVersion:
        """The baseline definition set this engine computes.

        Returns:
            The version identifier.
        """
        return self._baseline_version

    @property
    def method(self) -> StatisticsMethod:
        """The estimator selected by configuration.

        Returns:
            The method name.
        """
        return (
            StatisticsMethod.ROBUST_WINSORISED_MAD
            if self._settings.use_robust_statistics
            else StatisticsMethod.MEAN_STDDEV
        )

    def _dispersion(self, centre: float, retained: Sequence[float]) -> float | None:
        """Route to the configured dispersion estimator.

        Args:
            centre: The reported centre.
            retained: The bounded observation window.

        Returns:
            The dispersion, or ``None`` when fewer than two values are retained.
        """
        if self.method is StatisticsMethod.ROBUST_WINSORISED_MAD:
            return robust_dispersion(centre, retained)
        return sample_dispersion(centre, retained)

    def settings_fingerprint(self) -> str:
        """Fingerprint the configuration that shapes a baseline.

        Stored on every profile, so a profile cannot later be read as though it had been
        built under thresholds it never saw.

        Returns:
            A stable hex digest.
        """
        return content_digest(
            {
                "baseline_version": self._baseline_version,
                "dimensions": [d.value for d in BASELINE_DIMENSIONS],
                "method": self.method.value,
                "settings": repr(self._settings),
            }
        )

    def create_profile(self, learner_id: str, created_at: datetime) -> BaselineProfile:
        """Start an empty baseline for a learner.

        Args:
            learner_id: Pseudonymous learner identifier.
            created_at: The instant the profile starts. Must be timezone-aware.

        Returns:
            A profile with every dimension present and no observations.
        """
        return BaselineProfile(
            learner_id=learner_id,
            baseline_version=self._baseline_version,
            created_at=created_at,
            updated_at=None,
            statistics=tuple(
                DimensionStatistics(dimension=dimension, method=self.method)
                for dimension in BASELINE_DIMENSIONS
            ),
            origins=frozenset(),
            settings_fingerprint=self.settings_fingerprint(),
            prior_source=self._prior.source if self._prior is not None else None,
        )

    def update(self, profile: BaselineProfile, vector: FeatureValue) -> BaselineProfile:
        """Fold one feature vector into a profile.

        A dimension whose source feature reported ``INSUFFICIENT_DATA`` is left
        untouched: it is an absence, not a defect, and counting it as a rejection would
        misreport data quality. A source feature that produced a value of the wrong type
        or outside the dimension's range is counted as a rejection with a reason, and the
        value is discarded rather than clamped.

        The profile's ``updated_at`` advances even when nothing was accepted, because the
        profile has genuinely been processed up to that instant.

        Args:
            profile: The profile to advance.
            vector: The observation to fold in.

        Returns:
            A new profile. The input is unchanged.

        Raises:
            BaselineUpdateError: If the vector carries no learner, belongs to a different
                learner, or is not strictly later than the profile's last update.
        """
        self._require_acceptable(profile, vector)
        return _with_profile(
            profile,
            statistics=tuple(self._apply(item, vector) for item in profile.statistics),
            updated_at=vector.computed_at,
            origins=profile.origins | vector.origins,
        )

    def update_many(
        self, profile: BaselineProfile, vectors: Sequence[FeatureValue]
    ) -> BaselineProfile:
        """Fold a sequence of feature vectors in order.

        Args:
            profile: The starting profile.
            vectors: Observations in ascending reference-time order.

        Returns:
            The advanced profile.

        Raises:
            BaselineUpdateError: If any vector is unattributable or out of order.
        """
        current = profile
        for vector in vectors:
            current = self.update(current, vector)
        return current

    def reference(
        self, profile: BaselineProfile, dimension: BaselineDimension
    ) -> BaselineReference:
        """Resolve what normal means for one dimension, right now.

        While the personal record is too thin to interpret, the personal centre is not
        used at all and the population prior is returned instead, labelled as such. This
        is the point at which Separation 2 is enforced in code rather than in prose.

        Args:
            profile: The profile to read.
            dimension: The dimension to resolve.

        Returns:
            A reference carrying its maturity, basis, and source. Its centre is ``None``
            only when no prior covers the dimension.
        """
        item = profile.statistics_for(dimension)
        obs_maturity = maturity_for(item.observations, self._settings)

        if obs_maturity is BaselineMaturity.NEW:
            return self._cold_start_reference(profile, dimension, item, obs_maturity)

        if item.centre is None:
            return self._unavailable_reference(
                profile,
                dimension,
                obs_maturity,
                f"maturity {obs_maturity.value!r} was reached with no recorded centre",
            )

        return BaselineReference(
            dimension=dimension,
            centre=item.centre,
            spread=item.spread,
            method=item.method,
            observations=item.observations,
            maturity=obs_maturity,
            basis=basis_for(obs_maturity),
            source=ReferenceSource.PERSONAL_HISTORY,
            computed_at=profile.reference_time,
        )

    def deviation(
        self, profile: BaselineProfile, dimension: BaselineDimension, value: float
    ) -> DeviationResult:
        """Measure how far a value sits from the reference.

        The result carries the distance and the standing of the comparison, and nothing
        more. It does not classify the distance, because a deviation that arrived already
        labelled would make the temporal engine a consumer of this layer's opinion rather
        than its evidence.

        Args:
            profile: The profile providing the reference.
            dimension: The dimension to compare against.
            value: The measurement to compare. Must lie in the dimension's range.

        Returns:
            A deviation result. ``standardised`` is absent, with a reason, whenever a
            dispersion could not be estimated.

        Raises:
            ValueError: If ``value`` is outside the dimension's range.
        """
        low, high = dimension_range(dimension)
        if not low <= value <= high:
            raise ValueError(
                f"{dimension.value} measurement {value} is outside the valid range "
                f"[{low}, {high}]. The value has not been clamped."
            )

        reference = self.reference(profile, dimension)
        if reference.centre is None:
            return DeviationResult(
                dimension=dimension,
                value=value,
                maturity=reference.maturity,
                basis=reference.basis,
                source=ReferenceSource.UNAVAILABLE,
                computed_at=profile.reference_time,
                reason=reference.reason,
            )

        absolute = value - reference.centre
        standardised, reason = _standardise(absolute, reference)
        return DeviationResult(
            dimension=dimension,
            value=value,
            centre=reference.centre,
            spread=reference.spread,
            absolute=absolute,
            standardised=standardised,
            method=reference.method,
            maturity=reference.maturity,
            basis=reference.basis,
            source=reference.source,
            computed_at=profile.reference_time,
            reason=reason,
        )

    def _require_acceptable(self, profile: BaselineProfile, vector: FeatureValue) -> None:
        """Refuse an observation that cannot legitimately advance a profile.

        Args:
            profile: The profile being advanced.
            vector: The candidate observation.

        Raises:
            BaselineUpdateError: If the vector is unattributable, belongs to another
                learner, or is not strictly later than the last accepted observation.
        """
        if vector.learner_id is None:
            raise BaselineUpdateError(
                f"feature vector computed at {vector.computed_at.isoformat()} carries no "
                "learner_id, so it cannot be attributed to a baseline. Deriving a baseline "
                "from an unowned stream would invent an identity for the observations."
            )
        if vector.learner_id != profile.learner_id:
            raise BaselineUpdateError(
                f"feature vector belongs to learner {vector.learner_id!r} but the profile "
                f"is for {profile.learner_id!r}; a baseline is never populated from "
                "another learner's behaviour"
            )
        if profile.updated_at is not None and vector.computed_at <= profile.updated_at:
            raise BaselineUpdateError(
                f"observation at {vector.computed_at.isoformat()} is not later than the "
                f"profile's last update at {profile.updated_at.isoformat()}. A baseline "
                "is a function of observed time; accepting an out-of-order observation "
                "would make it a function of arrival order instead."
            )

    def _apply(self, item: DimensionStatistics, vector: FeatureValue) -> DimensionStatistics:
        """Apply one observation to one dimension's statistics.

        Args:
            item: The current statistics.
            vector: The observation to apply.

        Returns:
            Updated statistics, or the input unchanged when the dimension is absent.
        """
        extracted = _extract(vector, item.dimension)
        if extracted.rejection is not None:
            return _with(
                item,
                rejected=item.rejected + 1,
                last_rejection_reason=extracted.rejection,
            )
        if extracted.value is None:
            return item
        low, high = dimension_range(item.dimension)
        if not low <= extracted.value <= high:
            return _with(
                item,
                rejected=item.rejected + 1,
                last_rejection_reason=(
                    f"observed {item.dimension.value} {extracted.value} lies outside the valid "
                    f"range [{low}, {high}]. The value was discarded, not clamped."
                ),
            )
        return self._accept(item, extracted.value)

    def _accept(self, item: DimensionStatistics, value: float) -> DimensionStatistics:
        """Fold one validated observation into a dimension's statistics.

        Args:
            item: The current statistics.
            value: The observation, already validated against the dimension's range.

        Returns:
            Updated statistics with an advanced centre and a recomputed spread.
        """
        retained = decimate((*item.retained, value), self._settings.max_observations_retained)
        centre = self._advance_centre(item.centre, value, item.spread)
        return _with(
            item,
            centre=centre,
            spread=self._dispersion(centre, retained),
            observations=item.observations + 1,
            retained=retained,
        )

    def _advance_centre(self, centre: float | None, value: float, spread: float | None) -> float:
        """Move the centre towards one observation, by a bounded amount.

        The bound is the point of the method. An exponential mean is not robust: a single
        observation 300 seconds above the baseline moves the centre by
        ``ewma_alpha * 300`` whatever the spread, and the spread is then recomputed
        *around the displaced centre*, so the widening and the masking arrive together and
        neither is visible on its own. Bounding the step in units of the current spread
        decouples them — an extreme value can no longer relocate the baseline far enough
        for the surrounding ordinary observations to stop looking ordinary.

        This bounds the estimator's *influence*, not the data. A value outside a
        dimension's valid range is still rejected outright, never truncated into range.

        Args:
            centre: The current centre, or ``None`` for the first observation.
            value: The incoming observation.
            spread: The current spread, or ``None`` while fewer than two values are
                retained. An undefined spread leaves the step unbounded, because there is
                nothing yet to bound it against.

        Returns:
            The updated centre.
        """
        if centre is None:
            return value
        if self.method is not StatisticsMethod.ROBUST_WINSORISED_MAD:
            return centre + self._settings.ewma_alpha * (value - centre)
        step = value - centre
        if spread is not None and spread > 0.0:
            limit = self._settings.winsorisation_limit_spreads * spread
            step = min(max(step, -limit), limit)
        return centre + self._settings.ewma_alpha * step

    def _cold_start_reference(
        self,
        profile: BaselineProfile,
        dimension: BaselineDimension,
        item: DimensionStatistics,
        maturity: BaselineMaturity,
    ) -> BaselineReference:
        """Return a reference backed by the population prior.

        When no prior covers the dimension the reference is reported unavailable rather
        than guessed, and when it does cover the dimension the centre and spread belong
        to the prior, not to this learner's history.

        Args:
            profile: The profile being queried.
            dimension: The dimension to resolve.
            item: The dimension's accumulated statistics.
            maturity: The maturity level that triggered the cold start.

        Returns:
            A reference sourced from the population prior, or unavailable.
        """
        prior = self._prior.for_dimension(dimension) if self._prior is not None else None
        if prior is None:
            return self._unavailable_reference(
                profile,
                dimension,
                maturity,
                (
                    f"no population prior was supplied for {dimension.value}, so a cold "
                    "start cannot fall back to population reference statistics"
                ),
            )
        return BaselineReference(
            dimension=dimension,
            centre=prior.centre,
            spread=prior.spread,
            method=prior.method,
            observations=item.observations,
            maturity=maturity,
            basis=InferenceBasis.POPULATION_PRIOR,
            source=ReferenceSource.POPULATION_PRIOR,
            computed_at=profile.reference_time,
            prior_source=self._prior.source if self._prior is not None else None,
        )

    def _unavailable_reference(
        self,
        profile: BaselineProfile,
        dimension: BaselineDimension,
        maturity: BaselineMaturity,
        reason: str,
    ) -> BaselineReference:
        """Return a reference whose centre is absent, with a reason.

        Args:
            profile: The profile being queried.
            dimension: The dimension to resolve.
            maturity: The maturity level at the time of the query.
            reason: Why a centre cannot be formed.

        Returns:
            An unavailable reference.
        """
        return BaselineReference(
            dimension=dimension,
            maturity=maturity,
            basis=basis_for(maturity),
            source=ReferenceSource.UNAVAILABLE,
            reason=reason,
            computed_at=profile.reference_time,
        )


def _standardise(absolute: float, reference: BaselineReference) -> tuple[float | None, str | None]:
    """Compute the standardised deviation, when a dispersion exists.

    A dispersion of zero means every retained observation was identical, and reporting
    an infinite z-score in that case would be the one place the engine fabricated a
    number. Instead, the reason is recorded so a consumer can decide what ``identical``
    means for that dimension.

    Args:
        absolute: The absolute deviation from the centre.
        reference: The reference carrying the spread.

    Returns:
        A ``(standardised, reason)`` pair. At most one is non-``None``.
    """
    if reference.spread is None:
        return (
            None,
            f"no dispersion is defined for {reference.dimension.value}: fewer than two "
            "observations have been retained",
        )
    if reference.spread <= 0.0:
        return (
            None,
            f"every retained observation for {reference.dimension.value} is identical, so "
            "a deviation in dispersion units is undefined",
        )
    return absolute / reference.spread, None
