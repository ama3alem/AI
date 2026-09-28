"""Context engine: what situation did a signal occur in?

Context is derived from the behavioural event stream alone. It records session position,
current content, recent performance, and intervention history, and it names the gaps in
its own knowledge so a downstream layer can decline to interpret rather than guess.

This layer states no behavioural conclusion. It answers "under what conditions did this
happen?", never "what state is the learner in?".
"""

from __future__ import annotations

from focus_engine.context.engine import ContextEngine
from focus_engine.context.models import (
    BLOCKING_CONTEXT_KEYS,
    ContentContext,
    ContextKey,
    ContextModel,
    ContextVersion,
    InterventionContext,
    PerformanceContext,
    SessionContext,
)

__all__ = [
    "BLOCKING_CONTEXT_KEYS",
    "ContentContext",
    "ContextEngine",
    "ContextKey",
    "ContextModel",
    "ContextVersion",
    "InterventionContext",
    "PerformanceContext",
    "SessionContext",
]
