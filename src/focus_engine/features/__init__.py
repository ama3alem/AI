"""Feature engine package.

Computes named, versioned, deterministic features from events and context. Missing data
yields ``INSUFFICIENT_DATA`` rather than an invented value, and no computation uses
information later than the reference time.
"""

from __future__ import annotations

from focus_engine.features.engine import (
    CURRENT_FEATURE_SET,
    DEFAULT_REGISTRY,
    FEATURE_SPECS_V1,
    FeatureEngine,
    FeatureRegistry,
    UnsupportedFeatureSetError,
    compute_features,
)
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

__all__ = [
    "CURRENT_FEATURE_SET",
    "DEFAULT_REGISTRY",
    "FEATURE_SET_V1",
    "FEATURE_SPECS_V1",
    "INSUFFICIENT_DATA_MARKER",
    "FeatureAvailability",
    "FeatureCategory",
    "FeatureEngine",
    "FeatureName",
    "FeatureRegistry",
    "FeatureSpec",
    "FeatureValue",
    "FeatureValueType",
    "TypedFeatureValue",
    "UnsupportedFeatureSetError",
    "compute_features",
]
