"""Schema layer.

Holds the vocabulary shared by every other layer: identifiers, timestamps, provenance
labels, epistemic status, versioning coordinates, and the record types used by the
model and experiment registries.

This layer has no dependencies on any intelligence layer. Every other layer may import
from it; it imports from none of them.
"""

from __future__ import annotations

from focus_engine.schemas.primitives import (
    BEHAVIORAL_ENGAGEMENT_STATES,
    BehavioralEngagementState,
    ConfidenceLevel,
    DataOrigin,
    EventId,
    InferenceBasis,
    InterventionId,
    LearnerId,
    Probability,
    Provenance,
    SessionId,
    SyntheticDataStamp,
    Timestamp,
    UnitInterval,
    UtcTimestamp,
    non_empty_text,
    utc_now,
)
from focus_engine.schemas.versioning import (
    CodeVersion,
    DatasetVersion,
    ExperimentConclusion,
    ExperimentRecord,
    FeatureSetVersion,
    ModelMetrics,
    ModelRecord,
    ModelStatus,
    ModelVersion,
    ReproductionStamp,
    VersionStamp,
)

__all__ = [
    "BEHAVIORAL_ENGAGEMENT_STATES",
    "BehavioralEngagementState",
    "CodeVersion",
    "ConfidenceLevel",
    "DataOrigin",
    "DatasetVersion",
    "EventId",
    "ExperimentConclusion",
    "ExperimentRecord",
    "FeatureSetVersion",
    "InferenceBasis",
    "InterventionId",
    "LearnerId",
    "ModelMetrics",
    "ModelRecord",
    "ModelStatus",
    "ModelVersion",
    "Probability",
    "Provenance",
    "ReproductionStamp",
    "SessionId",
    "SyntheticDataStamp",
    "Timestamp",
    "UnitInterval",
    "UtcTimestamp",
    "VersionStamp",
    "non_empty_text",
    "utc_now",
]
