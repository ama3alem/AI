"""Personal baseline layer (Phase 6).

Answers one question — what does normal look like for this learner? — and answers it with
its standing attached. Every value this layer emits carries a
:class:`~focus_engine.baseline.models.BaselineMaturity` and an
:class:`~focus_engine.schemas.primitives.InferenceBasis`, so a population cold-start
estimate can never be mistaken for a personal one.

The layer reports distance from the baseline and nothing further. It does not classify a
deviation, attach severity, or decide whether a change matters: that is the temporal
engine's job, and keeping the boundary sharp is what stops a distance measurement from
quietly becoming a judgement.

Importing this package pulls in the baseline only. It does not import the feature,
temporal, or any later layer, so the dependency direction described in
``ARCHITECTURE.md`` holds at the package level.
"""

from focus_engine.baseline.engine import (
    DIMENSION_SOURCE_FEATURES,
    BaselineEngine,
    BaselineUpdateError,
)
from focus_engine.baseline.models import (
    BASELINE_DIMENSIONS,
    MAD_NORMAL_SCALE,
    PERSONAL_BASELINE_V1,
    BaselineDimension,
    BaselineMaturity,
    BaselineProfile,
    BaselineReference,
    DeviationResult,
    DimensionStatistics,
    PopulationPrior,
    PopulationPriorSet,
    ReferenceSource,
    StatisticsMethod,
    basis_for,
    dimension_range,
    maturity_rank,
)
from focus_engine.baseline.statistics import (
    decimate,
    maturity_for,
    robust_dispersion,
    sample_dispersion,
)

__all__ = [
    "BASELINE_DIMENSIONS",
    "DIMENSION_SOURCE_FEATURES",
    "MAD_NORMAL_SCALE",
    "PERSONAL_BASELINE_V1",
    "BaselineDimension",
    "BaselineEngine",
    "BaselineMaturity",
    "BaselineProfile",
    "BaselineReference",
    "BaselineUpdateError",
    "DeviationResult",
    "DimensionStatistics",
    "PopulationPrior",
    "PopulationPriorSet",
    "ReferenceSource",
    "StatisticsMethod",
    "basis_for",
    "decimate",
    "dimension_range",
    "maturity_for",
    "maturity_rank",
    "robust_dispersion",
    "sample_dispersion",
]
