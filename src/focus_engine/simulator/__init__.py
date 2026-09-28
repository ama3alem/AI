"""Synthetic learner simulator.

Generates deterministic, provenance-stamped behavioural event sequences for testing
downstream layers. Everything produced here is **synthetic** and carries the in-band
warning ``SYNTHETIC DATA — NOT REAL STUDENT DATA``.

The simulator exists so that later phases have data with *known* structure to test
against. It does not model, and cannot establish, anything about real learners. Any model
fitted on simulator output is fitted to a simulator's assumptions.
"""

from __future__ import annotations

from focus_engine.simulator.archetypes import (
    ARCHETYPE_PROFILES,
    ArchetypeProfile,
    LatencyShape,
    LearnerArchetype,
)
from focus_engine.simulator.config import (
    MAX_QUESTIONS_PER_SESSION,
    MIN_QUESTIONS_PER_SESSION,
    SIMULATOR_VERSION,
    SYNTHETIC_WARNING,
    SimulationConfig,
    SimulatorProvenance,
)
from focus_engine.simulator.generator import (
    MIN_RESPONSE_SECONDS,
    SimulatorResult,
    generate_session,
    generate_session_events,
    session_seed,
)

__all__ = [
    "ARCHETYPE_PROFILES",
    "ArchetypeProfile",
    "LatencyShape",
    "LearnerArchetype",
    "MAX_QUESTIONS_PER_SESSION",
    "MIN_QUESTIONS_PER_SESSION",
    "MIN_RESPONSE_SECONDS",
    "SIMULATOR_VERSION",
    "SYNTHETIC_WARNING",
    "SimulationConfig",
    "SimulatorProvenance",
    "SimulatorResult",
    "generate_session",
    "generate_session_events",
    "session_seed",
]
