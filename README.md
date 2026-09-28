# Focus Intelligence Engine

A **behavioral intelligence system for digital learning**.

> **Status: ENGINEERING FOUNDATION IN PROGRESS — NOT SCIENTIFICALLY VALIDATED.**
> No model in this repository has been validated on real learner data. No scientific
> or efficacy claim is made. All behavioural data produced by this repository at
> present is synthetic. See `FOUNDATION_STATUS.md`.

---

## What this system is

The long-term objective is a continuous, personalised behavioural learning loop:

```
OBSERVE → REPRESENT → ESTABLISH BASELINE → DETECT CHANGE → UNDERSTAND CONTEXT
→ PREDICT TRAJECTORY → ESTIMATE UNCERTAINTY → SELECT INTERVENTION
→ OBSERVE OUTCOME → LEARN FROM OUTCOME → IMPROVE FUTURE DECISIONS
```

The unit of analysis is **observable interaction behaviour** — response latency,
accuracy, navigation, inactivity, content-consumption patterns — measured per learner,
over time, relative to that learner's own history.

## What this system is not

This system does **not** and will not:

- read attention, detect consciousness, or infer mental state;
- perform psychological, cognitive, or medical diagnosis;
- claim to know what a learner is thinking or feeling;
- claim novelty, priority, or scientific validation;
- integrate with, import from, or depend on EduFlow.

Language used throughout the codebase is restricted to measurable, defensible terms:
`behavioural engagement`, `interaction behaviour`, `engagement trajectory`,
`behavioural deviation`, `session state`, `intervention response`, `prediction`,
`uncertainty`. See `RESEARCH_PRINCIPLES.md`.

---

## Current implementation state

| Layer | Status |
|---|---|
| Repository organisation, configuration, schemas | `COMPLETE (Phase 1)` |
| Event intelligence | `COMPLETE (Phase 2)` |
| Context engine | `COMPLETE (Phase 3)` |
| Synthetic simulator | `COMPLETE (Phase 4)` |
| Feature engine | `COMPLETE (Phase 5)` |
| Personal baseline engine | `COMPLETE (Phase 6)` |
| Temporal state engine | `COMPLETE (Phase 7)` |
| Baseline ML models | `COMPLETE (Phase 8)` |
| Prediction engine | `COMPLETE (Phase 8 — `models/predictors.py`)` |
| Uncertainty engine | `COMPLETE (Phase 9)` |
| Intervention policy | `COMPLETE (Phase 10 — `policy/`)` |
| Intervention engine | `PLANNED — NOT IMPLEMENTED` |
| Outcome engine | `COMPLETE (Phase 11 — `outcomes/`)` |
| Feedback architecture | `PLANNED — NOT IMPLEMENTED` |
| Evaluation framework | `PARTIAL (Phase 13 — `evaluation/`: core framework complete; split construction and advanced metrics planned)` |
| Development lab (`api/lab/`) | `COMPLETE (development-only — deterministic pipeline + delivery boundary + FastAPI UI)` |
| Student Intelligence layer (`api/lab/intelligence/`) | `COMPLETE (Phase 15 — synthetic-only — profile builder, demo store, RBAC-before-retrieval, 15-intent grounded answers, deep-evidence reasoning + `/api/students` & `/api/intelligence-demo` endpoints + Lab panel)` |
| API boundary | `PLANNED — NOT IMPLEMENTED (production)` |

This table describes the **actual** repository contents. It is not a plan. The development
lab runs the real engine end to end — features → baseline → temporal state → prediction →
uncertainty → policy → delivery → outcome → evaluation — under a fixed clock and a dev-only
delivery boundary that emits lifecycle events but sends nothing. See `api/lab/` and
`ARCHITECTURE.md` §1.1 for the separation between deciding and delivering.

---

## Architecture

```
                FOCUS INTELLIGENCE ENGINE
                          │
      ┌───────────────────┴──────────────────┐
      │                                      │
      EVENT INTELLIGENCE                    CONTEXT ENGINE
      │                                      │
      └───────────────────┬──────────────────┘
                          ↓
                 FEATURE ENGINE
                          ↓
              PERSONAL BASELINE ENGINE
                          ↓
             TEMPORAL STATE ENGINE
                          ↓
                PREDICTION ENGINE
                          ↓
              UNCERTAINTY ENGINE
                          ↓
              INTERVENTION POLICY
                          ↓
             INTERVENTION ENGINE
                          ↓
                 OUTCOME ENGINE
                           ↓
                 FEEDBACK ENGINE
                           │
                           ↓
                  MODEL IMPROVEMENT
                           │
                           ↓
                EVALUATION FRAMEWORK

  ┌──────────────────────────────────────────────┐
  │  SYNTHETIC SIMULATOR (Phase 4)               │
  │  produces event streams for testing above.    │
  │  Emits only; never reads engine state.        │
  └───────────────────┬──────────────────────────┘
                      │ (events in, stamped SYNTHETIC)
                      └──→ EVENT INTELLIGENCE
```

Every layer is an independent package with a narrow, typed interface. See
`ARCHITECTURE.md` for layer responsibilities, interface contracts, and the dependency
direction rules.

The simulator is deliberately **outside** the pipeline. It is a data producer for the
layers above it, not a stage they pass through, and it imports the event schema without
importing any intelligence layer.

The **evaluation framework** is deliberately **terminal**. It scores recorded predictions
against the evidence that actually followed them, under a half-open window that cannot
admit the prediction's own input, and nothing in the repository imports it. A layer that
could be corrected by its own evaluation would be able to grade its homework. See
`ARCHITECTURE.md` §4.13.

---

## Requirements

- Python **3.11+** (developed and tested on 3.12)
- Runs **fully locally**. No cloud account, no API key, no hosted-LLM dependency.

## Setup

```powershell
uv venv .venv --python 3.12
uv pip install -e ".[dev]" --python .venv
```

Or with standard `pip`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --no-header
.\.venv\Scripts\python.exe -m pytest tests\integration -q --no-header
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy src
```

`mypy src` is the configured target (`files = ["src"]` in `pyproject.toml`). Type-checking
the `tests/` tree is not part of the gate: because the package is untracked, mypy resolves
`focus_engine` as an installed package without a `py.typed` marker and reports
`import-untyped` noise across the whole suite, which says nothing about the code.

## Repository layout

```
focus-intelligence-engine/
├── README.md                  # this file
├── PROJECT_AUDIT.md           # Phase 0 audit — environment, risks, dependency plan
├── ARCHITECTURE.md            # layer contracts and dependency rules
├── ROADMAP.md                 # phased plan with per-phase exit criteria
├── RESEARCH_PRINCIPLES.md    # claim discipline and language constraints
├── .env.example               # environment variable names, no values
├── pyproject.toml
├── requirements.txt           # pinned runtime + dev set
├── src/focus_engine/          # the engine, one package per layer
├── api/lab/                   # development-only lab: server, adapter, delivery boundary, UI
├── tests/                     # unit | integration | simulation | evaluation
└── data/                      # raw | processed | synthetic  (all git-ignored)
```

Root-level `training/` and `evaluation/` exist on disk as reserved placeholders but are
empty, and git does not track empty directories, so they will not exist in a fresh clone
until the phases that fill them land. `api/` is **not** a placeholder: `api/lab/` is a
working development-only lab (server, adapter, delivery boundary, and static UI) described
above. Note that these are *not* the layer packages: the implemented prediction/uncertainty
layer is `src/focus_engine/models/` and `src/focus_engine/uncertainty/` (Phases 8–9), and
the implemented evaluation framework is `src/focus_engine/evaluation/` (Phase 13). The
empty root directories remain placeholders for later work, not the location of the
delivered layers.

Documents named in the project specification but **not yet written** are tracked in
`ROADMAP.md` and are deliberately absent here rather than stubbed: `DATA_DICTIONARY.md`,
`MODEL_CARD.md`, `PRIVACY.md`, `TECHNICAL_INVENTORY.md`, `EXPERIMENTS.md`, and
`FOUNDATION_STATUS.md`. None of them has content to report until later phases produce
the material they are supposed to describe.

## Data notice

Everything under `data/` is currently produced by the synthetic simulator and is
stamped `SYNTHETIC DATA — NOT REAL STUDENT DATA` at the data level. Synthetic
results validate **software and architecture only**. They are never evidence that
the system works on real students.

## Scope boundary

EduFlow is **not** integrated, imported, or depended upon in this phase. Any EduFlow
path or module in this repository would be a defect.

## Licence

Proprietary. No legal conclusions, patentability claims, or exclusivity claims are
made anywhere in this project.
