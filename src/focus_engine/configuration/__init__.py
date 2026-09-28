"""Configuration layer.

Two distinct concerns live here:

* :mod:`focus_engine.configuration.thresholds` — every tunable number that can influence
  a feature, a prediction, an uncertainty verdict, or an intervention decision, each with
  a documented rationale.
* :mod:`focus_engine.configuration.settings` — filesystem paths, runtime mode, and
  secret sourcing.

This layer imports from the schema layer and from the standard library. It imports no
intelligence layer, and no intelligence layer may redefine a threshold locally.
"""

from __future__ import annotations

from focus_engine.configuration.settings import (
    AUTH_SECRET_ENV_VAR,
    ENV_PREFIX,
    AppSettings,
    RuntimeEnvironment,
    RuntimeSettings,
    SecuritySettings,
    SettingsPaths,
    default_project_root,
    load_settings,
)
from focus_engine.configuration.thresholds import (
    BaselineSettings,
    EngineSettings,
    EvaluationSettings,
    FeatureSettings,
    InterventionSettings,
    OutcomeSettings,
    SimulationSettings,
    TemporalSettings,
    UncertaintySettings,
)

__all__ = [
    "AUTH_SECRET_ENV_VAR",
    "ENV_PREFIX",
    "AppSettings",
    "BaselineSettings",
    "EngineSettings",
    "EvaluationSettings",
    "FeatureSettings",
    "InterventionSettings",
    "OutcomeSettings",
    "RuntimeEnvironment",
    "RuntimeSettings",
    "SecuritySettings",
    "SettingsPaths",
    "SimulationSettings",
    "TemporalSettings",
    "UncertaintySettings",
    "default_project_root",
    "load_settings",
]
