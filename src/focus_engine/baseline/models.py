"""Personal baseline data model.

The baseline engine answers one question: *what does normal look like for this learner?*
Everything else in the system that wants to say "that looks different" is measured against
this. Three properties make the answer trustworthy rather than merely plausible.

**The basis travels with the value.** A cold-start number derived from population
reference statistics and a number derived from six months of one person's behaviour are
both floats, and nothing in the value itself distinguishes them. :class:`BaselineReference`
and :class:`DeviationResult` therefore carry a :class:`BaselineMaturity` and an
:class:`InferenceBasis`, and the models refuse an inconsistent pair. This is Separation 2
from ``ARCHITECTURE.md`` expressed as a type constraint rather than as a review
convention: a population estimate cannot be published as a personal one because there is
no way to construct the object that says so.

**Absence is reported, never substituted.** A dimension with no observations has no
centre. A reference with no centre reports :attr:`ReferenceSource.UNAVAILABLE` and a
reason; it does not carry a plausible default. A cold start with no population prior
supplied is therefore *unavailable*, not *zero* — inventing a prior would be the single
easiest way for this layer to launder a guess into an inference.

**Dispersion is estimated robustly, and its method is recorded.** A single 300-second
outlier raises a mean and inflates a standard deviation, and the resulting wider baseline
is exactly what would mask the real change the system exists to notice. The default
estimator is therefore a median absolute deviation. Which estimator produced a number is
carried on the result, so a consumer is never left inferring it.

The module holds vocabulary and structure only. Derivation lives in
:mod:`focus_engine.baseline.engine`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from focus_engine.schemas.primitives import DataOrigin, InferenceBasis, Timestamp, non_empty_text
from focus_engine.schemas.versioning import BaselineVersion

__all__ = [
    "BASELINE_DIMENSIONS",
    "MAD_NORMAL_SCALE",
    "PERSONAL_BASELINE_V1",
    "BaselineDimension",
    "BaselineMaturity",
    "BaselineProfile",
    "BaselineReference",
    "DeviationResult",
    "DimensionStatistics",
    "PopulationPrior",
    "PopulationPriorSet",
    "ReferenceSource",
    "StatisticsMethod",
    "basis_for",
    "dimension_range",
    "maturity_rank",
]


PERSONAL_BASELINE_V1: Final[BaselineVersion] = "PERSONAL_BASELINE_V1"
"""The initial baseline definition set.

Four dimensions, a median-absolute-deviation dispersion estimator, and a
frozen-population-prior cold start. Bump to ``PERSONAL_BASELINE_V2`` if any of those
change, so stored baselines keep their original meaning.
"""


MAD_NORMAL_SCALE: Final[float] = 1.4826
"""Scale factor making a median absolute deviation comparable to a standard deviation.

The MAD of a normal sample converges to ``sigma / sqrt(2 * pi / 5)``, about ``0.6745 *
sigma``, so the raw MAD understates the spread of ordinary data by roughly a third.
Dividing by that constant rather than 0.6745 is the same operation and keeps the
convention visible. This is a mathematical constant of the estimator, not a tunable
threshold, which is why it lives here rather than in
:class:`~focus_engine.configuration.thresholds.BaselineSettings`.
"""


class BaselineMaturity(StrEnum):
    """How much personal history a baseline rests on.

    The ladder exists so that a learner observed for thirty seconds cannot receive an
    inference of the same standing as one observed for three weeks. Each rung is defined
    by an observation count in
    :class:`~focus_engine.configuration.thresholds.BaselineSettings`, so the threshold that
    produced a level is configuration, not a constant buried in the code.
    """

    NEW = "new"
    """Cold start. Personal statistics are recorded but too thin to interpret; inference
    falls back to a population prior and says so."""

    EARLY = "early"
    """Some personal history exists. Statistics are computed and used, and are labelled
    provisional."""

    DEVELOPING = "developing"
    """A substantial personal record. Statistics are used without the cold-start
    fallback."""

    ESTABLISHED = "established"
    """A long personal record. The evidence-count cap no longer applies; confidence is
    then a question for the uncertainty engine, not for this layer."""


class BaselineDimension(StrEnum):
    """The closed set of quantities a personal baseline tracks.

    Deliberately narrow. A baseline is only meaningful for a measure that describes a
    learner's *own* recurring behaviour, and adding dimensions is a claim that the
    quantity is both measurable and worth modelling. Four qualify:

    * accuracy and response latency — the two primary interaction measures, and the two a
      learner can be compared against themselves on directly;
    * session length — how long this learner normally works before stopping;
    * content difficulty — how hard material this learner normally attempts.

    Excluded on purpose, because each would make the baseline worse rather than richer:

    * ``intervention_count`` — this is an output of the intervention policy, not a
      personal trait. Baselining it creates a feedback loop in which the policy's own
      actions raise the bar it is later measured against, so sustained over-intervention
      would progressively hide itself.
    * ``session_position`` and both trend features — session-relative or already
      derivative quantities. A trend is a difference of windows; baselining a difference
      is a difference of differences, which buys nothing and is harder to explain.
    * ``content_type_encoded`` — categorical. It has no centre or spread, and inventing
      an ordinal encoding for it would impose an ordering the source data does not have.
    """

    ACCURACY = "accuracy"
    RESPONSE_SECONDS = "response_seconds"
    SESSION_SECONDS = "session_seconds"
    CONTENT_DIFFICULTY = "content_difficulty"


#: Canonical dimension order. Statistics are stored in this order so that two profiles
#: built by different code paths are comparable by position, and so a reader can tell at a
#: glance that a dimension is missing from the record rather than merely zero.
BASELINE_DIMENSIONS: Final[tuple[BaselineDimension, ...]] = (
    BaselineDimension.ACCURACY,
    BaselineDimension.RESPONSE_SECONDS,
    BaselineDimension.SESSION_SECONDS,
    BaselineDimension.CONTENT_DIFFICULTY,
)


_DIMENSION_RANGES: Final[dict[BaselineDimension, tuple[float, float]]] = {
    BaselineDimension.ACCURACY: (0.0, 1.0),
    BaselineDimension.RESPONSE_SECONDS: (0.0, 86400.0),
    BaselineDimension.SESSION_SECONDS: (0.0, 86400.0),
    BaselineDimension.CONTENT_DIFFICULTY: (0.0, 1.0),
}
"""Valid ranges, matching the feature-spec ranges the observations are drawn from.

Used to refuse an impossible value rather than to clamp one. A negative response time or
an accuracy above one is a defect in the input, and clamping it to the boundary would
replace a visible error with an invisible plausible number.
"""


_MATURITY_RANKS: Final[dict[BaselineMaturity, int]] = {
    BaselineMaturity.NEW: 0,
    BaselineMaturity.EARLY: 1,
    BaselineMaturity.DEVELOPING: 2,
    BaselineMaturity.ESTABLISHED: 3,
}


_MATURITY_BASIS: Final[dict[BaselineMaturity, InferenceBasis]] = {
    BaselineMaturity.NEW: InferenceBasis.POPULATION_PRIOR,
    BaselineMaturity.EARLY: InferenceBasis.PARTIAL_PERSONAL,
    BaselineMaturity.DEVELOPING: InferenceBasis.PERSONAL,
    BaselineMaturity.ESTABLISHED: InferenceBasis.PERSONAL,
}
"""The only admissible maturity-to-basis mapping.

Written once, as a table, and used both to build references and to validate them. A
second place that decides this mapping is a second place Separation 2 can leak.
"""


class StatisticsMethod(StrEnum):
    """How a centre and a spread were estimated."""

    ROBUST_WINSORISED_MAD = "robust_winsorised_mad"
    """A winsorised exponential centre with a scaled median-absolute-deviation spread.

    Both halves have to be robust for the pair to be. An exponential mean is not, so its
    per-observation step is bounded by
    :attr:`~focus_engine.configuration.thresholds.BaselineSettings.winsorisation_limit_spreads`
    current dispersions; without that bound a single extreme value drags the centre and
    then re-centres the spread around the dragged value, so the widening and the masking
    happen together and neither is individually visible.
    """

    MEAN_STDDEV = "mean_stddev"
    """An unbounded exponential mean with a root-mean-square dispersion.

    Outlier-sensitive, and retained only so the choice is a recorded configuration rather
    than an assumption. It is the honest comparison case: the robustness claims made for
    the estimator above are only meaningful against this one.
    """


class ReferenceSource(StrEnum):
    """What a reference was derived from."""

    POPULATION_PRIOR = "population_prior"
    """Population reference statistics, because personal history is insufficient."""

    PERSONAL_HISTORY = "personal_history"
    """This learner's own recorded observations."""

    UNAVAILABLE = "unavailable"
    """Neither is available. The result carries a reason and no centre."""


def dimension_range(dimension: BaselineDimension) -> tuple[float, float]:
    """Return the valid range for a dimension.

    Args:
        dimension: The dimension to look up.

    Returns:
        The inclusive ``(low, high)`` bound.
    """
    return _DIMENSION_RANGES[dimension]


def maturity_rank(maturity: BaselineMaturity) -> int:
    """Return the ordinal position of a maturity level.

    Exposed so a consumer can compare levels without re-deriving the order, and so the
    order is a single definition rather than a convention repeated per call site.

    Args:
        maturity: The level to rank.

    Returns:
        ``0`` for ``NEW`` through ``3`` for ``ESTABLISHED``.
    """
    return _MATURITY_RANKS[maturity]


def basis_for(maturity: BaselineMaturity) -> InferenceBasis:
    """Return the inference basis a maturity level permits.

    Args:
        maturity: The maturity level.

    Returns:
        The admissible :class:`InferenceBasis`.
    """
    return _MATURITY_BASIS[maturity]


class PopulationPrior(BaseModel):
    """Reference statistics for one dimension, for a population of learners.

    Supplied by the caller. This layer deliberately ships no built-in priors: a hard-coded
    "typical" response time is a claim about real learners that nothing in this repository
    can support, and freezing one into the codebase would make it indistinguishable from
    a value that had actually been estimated from a cohort.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: BaselineDimension
    centre: float
    spread: float
    method: StatisticsMethod
    notes: str = ""

    @model_validator(mode="after")
    def _validate_prior(self) -> PopulationPrior:
        """Check the centre is in range and the spread is strictly positive.

        A prior with zero spread would make every deviation infinitely large, and the
        honest response to that is not to accept the prior.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If the centre is outside the dimension's range, or the spread is
                not strictly positive.
        """
        low, high = dimension_range(self.dimension)
        if not low <= self.centre <= high:
            raise ValueError(
                f"population prior centre {self.centre} for {self.dimension.value} is outside "
                f"the valid range [{low}, {high}]"
            )
        if self.spread <= 0:
            raise ValueError(
                f"population prior spread for {self.dimension.value} must be strictly "
                f"positive; got {self.spread}"
            )
        return self


class PopulationPriorSet(BaseModel):
    """A named, versioned collection of population priors.

    The whole set carries one source and one origin, so a cold-start inference can state
    which cohort it rested on and whether that cohort was synthetic. A prior built from
    simulator output is a software artifact, and the record has to say so at the point the
    number is consumed rather than at the point it was generated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: non_empty_text
    """What the reference statistics describe, e.g. a named cohort or a simulator version."""

    version: str
    """Version of the prior set, so a stored cold-start inference is reproducible."""

    n_reference: int = Field(ge=1)
    """How many learners the reference statistics were computed from."""

    data_origin: DataOrigin
    """Whether the reference cohort was observed or synthetic."""

    priors: tuple[PopulationPrior, ...]
    """The per-dimension priors. May be a subset of :data:`BASELINE_DIMENSIONS`; a
    dimension with no prior simply has no cold-start fallback."""

    @model_validator(mode="after")
    def _validate_priors(self) -> PopulationPriorSet:
        """Reject an empty or ambiguous prior set.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If no priors are supplied, or a dimension appears twice.
        """
        if not self.priors:
            raise ValueError("a population prior set must supply at least one dimension")
        dimensions = [prior.dimension for prior in self.priors]
        duplicates = sorted({item.value for item in dimensions if dimensions.count(item) > 1})
        if duplicates:
            raise ValueError(f"population prior set repeats dimensions: {duplicates}")
        return self

    def for_dimension(self, dimension: BaselineDimension) -> PopulationPrior | None:
        """Look up the prior for one dimension.

        Args:
            dimension: The dimension to look up.

        Returns:
            The prior, or ``None`` when this set does not cover the dimension.
        """
        for prior in self.priors:
            if prior.dimension is dimension:
                return prior
        return None


class DimensionStatistics(BaseModel):
    """Accumulated evidence for one dimension.

    The retained window is stored rather than summarised away. A baseline whose spread
    cannot be re-derived from the observations that produced it cannot be audited, and an
    unauditable centre is exactly the kind of number that ends up trusted too far.

    The centre is not reconstructible from the window: it is an exponentially weighted
    mean updated in observation order, and the window is decimated as it fills. Both are
    stored so the state is complete rather than half-derivable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: BaselineDimension
    centre: float | None = None
    spread: float | None = None
    method: StatisticsMethod
    observations: int = Field(default=0, ge=0)
    rejected: int = Field(default=0, ge=0)
    last_rejection_reason: str | None = None
    retained: tuple[float, ...] = ()

    @model_validator(mode="after")
    def _validate_statistics(self) -> DimensionStatistics:
        """Check the centre, spread, and window agree with the observation count.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If the counts and the presence of a centre or spread disagree, or
                if the centre is outside the dimension's range.
        """
        if self.observations == 0 and self.centre is not None:
            raise ValueError(
                f"{self.dimension.value} reports no observations but carries the centre "
                f"{self.centre}"
            )
        if self.observations > 0 and self.centre is None:
            raise ValueError(
                f"{self.dimension.value} reports {self.observations} observations but carries "
                "no centre"
            )
        if len(self.retained) > self.observations:
            raise ValueError(
                f"{self.dimension.value} retains {len(self.retained)} values from only "
                f"{self.observations} observations"
            )
        if self.spread is not None and len(self.retained) < 2:
            raise ValueError(
                f"{self.dimension.value} reports a spread from fewer than two retained values"
            )
        if self.rejected > 0 and self.last_rejection_reason is None:
            raise ValueError(
                f"{self.dimension.value} rejected {self.rejected} values without stating a reason"
            )
        if self.centre is not None:
            low, high = dimension_range(self.dimension)
            if not low <= self.centre <= high:
                raise ValueError(
                    f"{self.dimension.value} centre {self.centre} is outside the valid range "
                    f"[{low}, {high}]. The value has not been clamped."
                )
        return self


class BaselineProfile(BaseModel):
    """Everything recorded about one learner's normal behaviour.

    The profile is the accumulated state; the interpretation of it — which maturity a
    dimension has reached, and therefore whether a personal value or a population prior is
    admissible — is derived on demand by the engine rather than cached here. Deriving it
    means it cannot go stale against the settings it was derived from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    learner_id: str
    baseline_version: BaselineVersion
    created_at: Timestamp
    updated_at: Timestamp | None = None
    statistics: tuple[DimensionStatistics, ...]
    origins: frozenset[DataOrigin] = Field(default_factory=frozenset)
    settings_fingerprint: str
    prior_source: str | None = None

    @model_validator(mode="after")
    def _validate_profile(self) -> BaselineProfile:
        """Check the dimension coverage is complete and ordered.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If a dimension is missing, duplicated, or out of canonical order.
        """
        dimensions = tuple(item.dimension for item in self.statistics)
        if dimensions != BASELINE_DIMENSIONS:
            raise ValueError(
                "profile statistics must cover every dimension in canonical order; expected "
                f"{[item.value for item in BASELINE_DIMENSIONS]}, got "
                f"{[item.value for item in dimensions]}"
            )
        return self

    def statistics_for(self, dimension: BaselineDimension) -> DimensionStatistics:
        """Return the accumulated evidence for one dimension.

        Args:
            dimension: The dimension to look up.

        Returns:
            Its statistics record.

        Raises:
            KeyError: If the profile does not cover the dimension, which the model
                validator already prevents.
        """
        for item in self.statistics:
            if item.dimension is dimension:
                return item
        raise KeyError(dimension)

    @property
    def observation_count(self) -> int:
        """Total accepted observations across every dimension.

        Returns:
            The sum of per-dimension observation counts.
        """
        return sum(item.observations for item in self.statistics)

    @property
    def reference_time(self) -> Timestamp:
        """The instant this profile describes.

        Returns:
            The last update time, or the creation time if the profile has never been
            updated.
        """
        return self.updated_at if self.updated_at is not None else self.created_at

    @property
    def is_synthetic_only(self) -> bool:
        """Whether every contributing observation was synthetic.

        Returns:
            ``True`` only when the recorded origins are exactly synthetic.
        """
        return self.origins == frozenset({DataOrigin.SYNTHETIC})


class BaselineReference(BaseModel):
    """What normal looks like for one dimension, with the standing of that answer.

    A reference is the comparison point a deviation is measured from, so it carries
    everything needed to judge how much weight the comparison deserves: the maturity, the
    basis, the source, and the estimator. When no reference can be formed the centre is
    absent and a reason is given.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: BaselineDimension
    centre: float | None = None
    spread: float | None = None
    method: StatisticsMethod | None = None
    observations: int = Field(default=0, ge=0)
    maturity: BaselineMaturity
    basis: InferenceBasis
    source: ReferenceSource
    reason: str | None = None
    computed_at: Timestamp
    prior_source: str | None = None

    @model_validator(mode="after")
    def _validate_reference(self) -> BaselineReference:
        """Check availability, maturity, and basis are mutually consistent.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If an unavailable reference carries a centre, if an available one
                omits a reason, or if the maturity and basis disagree.
        """
        if self.source is ReferenceSource.UNAVAILABLE:
            if self.centre is not None or self.spread is not None or self.method is not None:
                raise ValueError(
                    f"{self.dimension.value} has no usable reference but carries a centre, "
                    "spread, or method"
                )
            if not self.reason:
                raise ValueError(
                    f"{self.dimension.value} has no usable reference and must state why"
                )
        else:
            if self.centre is None:
                raise ValueError(
                    f"{self.dimension.value} is sourced from {self.source.value} and must carry "
                    "a centre"
                )
            if self.method is None:
                raise ValueError(f"{self.dimension.value} must record the method that produced it")
            if self.reason is not None:
                raise ValueError(
                    f"{self.dimension.value} is sourced from {self.source.value} and must not "
                    "carry an unavailability reason"
                )
        if self.basis is not basis_for(self.maturity):
            raise ValueError(
                f"maturity {self.maturity.value!r} does not admit basis {self.basis.value!r}; "
                f"it requires {basis_for(self.maturity).value!r}"
            )
        return self

    @property
    def is_available(self) -> bool:
        """Whether this reference carries a centre.

        Returns:
            ``True`` when a comparison is possible.
        """
        return self.centre is not None


class DeviationResult(BaseModel):
    """How far a measurement sits from the reference, without interpreting it.

    This layer reports distance and nothing else. It does not classify the distance as a
    state, attach a severity, or decide whether a change matters — that belongs to the
    temporal and uncertainty engines, and a deviation that arrived pre-classified would
    collapse Separation 1 at the first use.

    A standardised deviation is absent whenever a dispersion could not be estimated,
    which happens for a dimension with a single observation and for a dimension whose
    retained values are all identical. Both cases report a reason rather than dividing by
    zero or substituting a floor value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: BaselineDimension
    value: float
    centre: float | None = None
    spread: float | None = None
    absolute: float | None = None
    standardised: float | None = None
    method: StatisticsMethod | None = None
    maturity: BaselineMaturity
    basis: InferenceBasis
    source: ReferenceSource
    computed_at: Timestamp
    reason: str | None = None

    @model_validator(mode="after")
    def _validate_deviation(self) -> DeviationResult:
        """Check the derived quantities agree with the reference availability.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If an unavailable reference still produced a distance, or if a
                standardised deviation is present without a reference, or if a reason is
                present alongside a usable standardised deviation.
        """
        if self.source is ReferenceSource.UNAVAILABLE:
            if self.absolute is not None or self.standardised is not None:
                raise ValueError(
                    f"{self.dimension.value} has no usable reference, so no deviation can be "
                    "reported against it"
                )
        else:
            if self.absolute is None or self.centre is None:
                raise ValueError(
                    f"{self.dimension.value} is sourced from {self.source.value} and must carry "
                    "a centre and an absolute deviation"
                )
        if self.standardised is not None and self.spread is None:
            raise ValueError(
                f"{self.dimension.value} reports a standardised deviation without a spread"
            )
        if (self.standardised is None) == (self.reason is None):
            raise ValueError(
                f"{self.dimension.value} must state a reason exactly when no standardised "
                "deviation is available, and must not state one when it is available"
            )
        if self.basis is not basis_for(self.maturity):
            raise ValueError(
                f"maturity {self.maturity.value!r} does not admit basis {self.basis.value!r}"
            )
        return self

    def to_summary_dict(self) -> dict[str, object]:
        """Render a JSON-serialisable summary for logs and debugging.

        Returns:
            A dictionary of the measurement, the reference it was compared against, and the
            standing of that comparison.
        """
        return {
            "dimension": self.dimension.value,
            "value": self.value,
            "centre": self.centre,
            "spread": self.spread,
            "absolute": self.absolute,
            "standardised": self.standardised,
            "method": self.method.value if self.method is not None else None,
            "maturity": self.maturity.value,
            "basis": self.basis.value,
            "source": self.source.value,
            "computed_at": self.computed_at.isoformat(),
            "reason": self.reason,
        }
