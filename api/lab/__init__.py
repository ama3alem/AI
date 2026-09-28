"""FOCUS ENGINE LAB — a development-only interactive test interface.

**This is not the production EduFlow UI.** It exists so a human can feed learner-shaped
data into the real engine, watch it travel through the real pipeline, and inspect exactly
why the engine reached the decision it reached.

Three properties this package holds to, because a lab that breaks them is worse than no
lab at all:

**No algorithm lives here.** Every number this package displays is produced by
:mod:`focus_engine`. :mod:`api.lab.adapter` is the only module that imports the engine,
and it imports public surfaces only. There is no JavaScript reimplementation of the
baseline, the temporal state, the model, or the policy, and adding one would create a
second engine that could disagree with the first.

**No output is hardcoded.** The five scenarios in :mod:`api.lab.scenarios` fix *inputs* —
which archetype, how many questions, which seed. They never fix an expected verdict, a
probability, a state, or a decision. What the UI shows is whatever the engine returned,
including "insufficient data" and including a refusal.

**No performance is claimed.** Every corpus available to this lab is
:mod:`focus_engine.simulator` output. A probability computed here describes a simulator
archetype and a set of thresholds, not a human being. The UI labels this on every screen.

The package is outside ``src/focus_engine`` so the engine's layer boundaries and its
"importing the root pulls in no layer" rule are untouched.
"""

from __future__ import annotations

__all__: list[str] = []
