"""Focus Intelligence Engine.

A behavioral intelligence system for digital learning.

The engine is organised as independent layers (events, context, features, baseline,
temporal, prediction, uncertainty, interventions, outcomes, feedback) with narrow
typed interfaces. This package root intentionally exposes no layer symbols: importing
``focus_engine`` must not pull in any layer, so that a layer can never be used without
its declared dependencies.

See ``ARCHITECTURE.md`` for layer contracts and ``RESEARCH_PRINCIPLES.md`` for the
claim and language constraints enforced throughout the codebase.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
