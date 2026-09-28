# PROJECT AUDIT — FOCUS INTELLIGENCE ENGINE

**Audit phase:** PHASE 0 (Repository Audit)
**Audit date:** 2026-09-26
**Audited path:** `C:\Users\g_kpc\OneDrive\Desktop\AIII LMS`
**Auditor role:** Lead AI/ML Architect (this session)

---

## 1. REPOSITORY STATE

### 1.1 Verdict

```
GREENFIELD REPOSITORY
```

The audited working directory contains **zero files** (including hidden files and
subdirectories). It is **not** a Git repository.

### 1.2 Evidence

| Check | Command | Result |
|---|---|---|
| Top-level entries | `Get-ChildItem -Force` | empty |
| Recursive entry count (incl. hidden) | `Get-ChildItem -Force -Recurse \| Measure-Object` | `0` |
| Git repository | `git rev-parse --is-inside-work-tree` | `fatal: not a git repository` |

### 1.3 Explicitly NOT FOUND IN CURRENT REPOSITORY

The following were searched for and **do not exist** anywhere in the audited path:

- Source code — `NOT FOUND IN CURRENT REPOSITORY`
- Tests — `NOT FOUND IN CURRENT REPOSITORY`
- Documentation — `NOT FOUND IN CURRENT REPOSITORY`
- Datasets (raw or processed) — `NOT FOUND IN CURRENT REPOSITORY`
- Trained models / model artifacts — `NOT FOUND IN CURRENT REPOSITORY`
- API code — `NOT FOUND IN CURRENT REPOSITORY`
- Configuration files (`pyproject.toml`, `requirements.txt`, `setup.cfg`, CI config) — `NOT FOUND IN CURRENT REPOSITORY`
- Dependency lockfiles — `NOT FOUND IN CURRENT REPOSITORY`
- License file — `NOT FOUND IN CURRENT REPOSITORY`
- `CONTRIBUTING` / `AGENTS.md` — `NOT FOUND IN CURRENT REPOSITORY`

### 1.4 Adjacent artifacts (OUT OF SCOPE — deliberately not inspected or used)

A separate, unrelated project directory exists on the same parent folder:

```
C:\Users\g_kpc\OneDrive\Desktop\eduflow-lms-v3-final (1) - Copy
```

Per the project constraint **"DO NOT modify, import, depend on, or integrate with
EduFlow"**, this directory was **not opened, not read, not imported, and will not be
modified**. Its existence is recorded here only to make the boundary explicit.
No file in the audited repository references it, and none will.

Status: `NOT USED — EXPLICITLY OUT OF SCOPE`

---

## 2. ENVIRONMENT

### 2.1 Operating system

| Property | Value |
|---|---|
| OS | Windows (win32) |
| Shell available to tooling | Windows PowerShell 5.1 |
| Logical processors | 16 |
| Physical RAM | ~15.8 GB |
| Filesystem note | Working directory is under `OneDrive\Desktop` — see Risk R-08 |

### 2.2 Runtimes detected

| Runtime | Version | Path | Status |
|---|---|---|---|
| Python (default) | **3.12.0** | `C:\Users\g_kpc\AppData\Local\Programs\Python\Python312\python.exe` | Active interpreter |
| Python 3.11 | 3.11.9 | `C:\Users\g_kpc\AppData\Local\Microsoft\WindowsApps\python3.11.exe` | Available (secondary) |
| Node.js | present | `C:\Program Files\nodejs\node.exe` | Present, **not required** |
| .NET SDK | — | — | `NOT FOUND` |

### 2.3 Package managers & tooling detected

| Tool | Version | Status | Intended use here |
|---|---|---|---|
| `pip` | 26.1 | Available | Bootstrap only |
| `uv` | 0.11.8 | **Available** | **Chosen** — venv + dependency resolution |
| `poetry` | — | `NOT FOUND` | Not used |
| `conda` | — | `NOT FOUND` | Not used |
| `git` | 2.54.0.windows.1 | Available | Version control |
| `docker` | present (client) | Available, **not required** | Portability target only |
| VS Code | present | Available | Editor |

**Decision:** `uv` is used to create a project-local virtual environment. The global
interpreter's site-packages is **not** mutated.

---

## 3. DETECTED TECHNOLOGIES (global interpreter inventory)

This section reports the *pre-existing global Python environment*, not the project
environment. It matters because it reveals **one blocking defect**.

### 3.1 Present and importable

| Package | Version | Relevance |
|---|---|---|
| `numpy` | 2.5.2 | Required — core numeric layer |
| `scipy` | 1.15.3 | Available (indirect) |
| `pydantic` | 2.7.1 | Required — schema validation |
| `pydantic-settings` | 2.2.1 | Available — configuration |
| `fastapi` | 0.111.0 | Required — API boundary (Phase 14) |
| `starlette` | 0.37.2 | Available (FastAPI dependency) |
| `uvicorn` | 0.30.1 | Available — ASGI server |
| `pytest` | 9.1.1 | Required — test runner |
| `pytest-asyncio` | 1.4.0 | Available |
| `python-dotenv` | 1.0.1 | Available — env handling |
| `structlog` | 24.3.2 | Available — structured logging |
| `mypy` | 1.10.0 | Available — static typing |
| `ruff` | 0.4.9 | Available — lint/format |
| `joblib` | — | **`NOT INSTALLED`** (listed in pip output? no — absent) |
| `google` cloud SDKs, `boto3`, `azure` clients | various | Present globally, **deliberately unused** |

### 3.2 Defective / missing — BLOCKING

| Package | Global state | Impact |
|---|---|---|
| `pandas` | **BROKEN** — installed as 2.2.1 but fails to import: `ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject` | ABI mismatch: pandas 2.2.1 was compiled against an older NumPy C-ABI than the installed NumPy 2.5.2. Any `import pandas` fails. |
| `scikit-learn` | `NOT INSTALLED` | Required for Phase 8 classical ML. |
| `joblib` | `NOT INSTALLED` | Required for model serialization. |
| `xgboost` / `lightgbm` | `NOT INSTALLED` | Optional; deferred (see §6). |
| `torch` | `NOT INSTALLED` | Intentionally not required (§6). |
| `mlflow` | `NOT INSTALLED` | Optional; deferred (see §6). |

### 3.3 Cloud/hosting coupling detected in global env (must NOT be inherited)

The global interpreter already carries `boto3`, `google-cloud-*`, `azure`-adjacent,
`firebase-admin`, `stripe`, `twilio`, `supabase`, and hosted-LLM SDKs (`openai`,
`anthropic`). None of these are required by this project.

**Constraint applied:** the project environment will be created with an explicit,
minimal dependency set so that the Focus Intelligence Engine contains **no** cloud
provider SDK and **no** hosted-LLM SDK dependency.

### 3.4 Network reachability

`https://pypi.org/simple/scikit-learn/` → **HTTP 200**. Package installation is
possible.

---

## 4. EXISTING COMPONENTS

```
NONE
```

> Historical snapshot as of Phase 0 (2026-09-26). This section records what was absent
> at audit time and is deliberately not updated as phases land. Current implementation
> status is tracked in `ROADMAP.md`.

The repository was empty. There was no event architecture, no context engine, no
feature engine, no baseline engine, no temporal engine, no models, no intervention
policy, no outcome model, no feedback architecture, no simulator, no evaluation
framework, and no API.

---

## 5. MISSING COMPONENTS

Every component in the target architecture is missing. Grouped by the acceptance
criteria in the specification:

> Historical snapshot as of Phase 0 (2026-09-26). This table records what was absent
> at audit time and is deliberately not updated as phases land. Current implementation
> status is tracked in `ROADMAP.md`.

| # | Required component | Current status |
|---|---|---|
| 1 | Repository organization | `NOT FOUND IN CURRENT REPOSITORY` |
| 2 | Event system | `NOT IMPLEMENTED` |
| 3 | Context system | `NOT IMPLEMENTED` |
| 4 | Synthetic learner simulator | `NOT IMPLEMENTED` |
| 5 | Feature engine | `NOT IMPLEMENTED` |
| 6 | Personal baseline | `NOT IMPLEMENTED` |
| 7 | Baseline maturity | `NOT IMPLEMENTED` |
| 8 | Cold start | `NOT IMPLEMENTED` |
| 9 | Temporal state engine | `NOT IMPLEMENTED` |
| 10 | Initial ML models | `NOT IMPLEMENTED` |
| 11 | Prediction interface | `NOT IMPLEMENTED` |
| 12 | Uncertainty | `NOT IMPLEMENTED` |
| 13 | Intervention policy | `NOT IMPLEMENTED` |
| 14 | Intervention outcome system | `NOT IMPLEMENTED` |
| 15 | Feedback architecture | `NOT IMPLEMENTED` |
| 16 | Evaluation pipeline | `NOT IMPLEMENTED` |
| 17 | Model / version tracking | `NOT IMPLEMENTED` |
| 18 | Reproducibility | `NOT IMPLEMENTED` |
| 19 | Tests | `NOT IMPLEMENTED` |
| 20 | API boundary | `NOT IMPLEMENTED` |
| 21 | Privacy architecture | `NOT IMPLEMENTED` |
| 22 | Security foundations | `NOT IMPLEMENTED` |
| 23 | Documentation | `NOT IMPLEMENTED` |
| 24 | Technical inventory | `NOT IMPLEMENTED` |

---

## 6. RECOMMENDED ARCHITECTURE

### 6.1 Layer model (as specified, with enforced separation)

```
                FOCUS INTELLIGENCE ENGINE
                          |
      +-------------------+------------------+
      |                                      |
EVENT INTELLIGENCE                    CONTEXT ENGINE
      |                                      |
      +-------------------+------------------+
                          v
                 FEATURE ENGINE
                          v
              PERSONAL BASELINE ENGINE
                          v
             TEMPORAL STATE ENGINE
                          v
                PREDICTION ENGINE
                          v
              UNCERTAINTY ENGINE
                          v
              INTERVENTION POLICY
                          v
             INTERVENTION ENGINE
                          v
                OUTCOME ENGINE
                          v
                FEEDBACK ENGINE
                          |
                          v
                 MODEL IMPROVEMENT
```

Each layer is a separate Python package with a narrow, typed interface. No layer
imports a layer *above* it in the data-flow direction. Cross-cutting concerns
(versioning, evaluation, privacy, auditability) are separate packages, not mixed
into the intelligence layers.

### 6.2 The three separations that must never be collapsed

1. **Prediction ≠ Decision ≠ Intervention.**
   `prediction` answers *what is the estimated behavioural state*; `policy` answers
   *is intervention justified*; `intervention` answers *what is delivered*. Three
   distinct interfaces, three distinct outputs, three distinct audit records.

2. **Population inference ≠ Personalized inference.**
   Every prediction and every feature vector carries an explicit
   `inference_basis` and `baseline_maturity` so a cold-start population estimate is
   never silently presented as a personal one.

3. **Engineering complete ≠ Scientifically validated.**
   A green test suite means the machinery runs. It says nothing about validity on
   real learners. These are tracked as separate statuses.

### 6.3 Language constraints encoded into the codebase

- Target/state names use **measurable** terms: `BEHAVIORAL_ENGAGEMENT_STATE`,
  `DECLINING_ENGAGEMENT_PROBABILITY`. Not "focus", not "attention", not "mind".
- No component infers emotion, consciousness, or mental state. There is no such
  field in any schema.
- Synthetic artifacts are stamped `SYNTHETIC — NOT REAL STUDENT DATA` at the data
  level, not only in prose.

### 6.4 Storage posture for the foundation phase

A **file-backed, append-only, versioned artifact store** (JSONL + manifest hashes) is
sufficient and chosen deliberately:

- it satisfies auditability and reproducibility requirements;
- it has zero infrastructure cost and runs fully locally;
- it does not lock the project into a database vendor;
- it is replaceable behind a narrow `EventStore` / `ArtifactStore` interface if a
  real deployment later needs Postgres.

A learner-overlap-safe, temporal-split evaluation harness is designed in from the
start so that swapping the store later cannot silently introduce leakage.

---

## 7. DEPENDENCY PLAN

### 7.1 Install now (each with a concrete reason)

| Package | Constraint | Concrete reason |
|---|---|---|
| `numpy` | `>=1.26` | Feature/baseline math, simulation arrays |
| `pandas` | `>=2.2` | Tabular feature frames, evaluation matrices |
| `scikit-learn` | `>=1.4` | Logistic Regression, Random Forest, Gradient Boosting; calibration + metrics |
| `pydantic` | `>=2.7` | Runtime validation of every event and every API boundary |
| `fastapi` | `>=0.111` | API boundary that wraps the real engine (Phase 14) |
| `uvicorn` | `>=0.30` | Local ASGI runner |
| `joblib` | `>=1.3` | Model/artifact serialization (model registry) |
| `pytest` | `>=8` | Test runner |
| `httpx` | `>=0.27` | FastAPI `TestClient` transport for API tests |
| `ruff` | dev | Lint + format, deterministic style |
| `mypy` | dev | Static typing enforcement |

### 7.2 Deliberately NOT installed now

| Package | Why deferred |
|---|---|
| `xgboost`, `lightgbm` | Gradient Boosting from scikit-learn establishes the signal baseline first. Adding a second GBM implementation before knowing whether signal exists is unjustified. Revisit only if Phase 8 experiments show tree-ensemble capacity limits. |
| `torch`, `transformers` | Specification §41 forbids premature deep learning. V0–V2 must prove signal before V5 is even considered. |
| `mlflow` | The specification's own requirement (experiment ID, hypothesis, seed, metrics, conclusion) is satisfiable with a versioned local registry + `EXPERIMENTS.md`. MLflow is a large dependency for a local-first system. Revisit if multi-user tracking becomes real. |
| `darts`, `sktime`, `prophet` | Temporal work at this scale is handled by explicit, inspectable code in the temporal engine. A time-series library would hide the state-transition logic the project needs to reason about. |
| `boto3`, `google-cloud-*`, `azure-*`, `openai`, `anthropic`, `firebase-admin` | Cloud/LLM coupling is prohibited. |
| `redis`, `celery` | No async/distributed requirement in the foundation phase. |

### 7.3 Environment isolation decision

A project-local virtual environment (`.venv`, created with `uv`) is mandatory, for
three reasons:

1. It repairs the broken global `pandas`/NumPy ABI conflict **without** mutating an
   environment that other projects on this machine depend on.
2. It makes the dependency set auditable — the project's true requirements are
   visible in one place instead of being inferred from a 200-package global list.
3. It satisfies the portability requirement: nothing outside the venv is required.

---

## 8. RISKS

| ID | Risk | Severity | Evidence | Mitigation |
|---|---|---|---|---|
| R-01 | **Global `pandas` is broken** (NumPy 2.5.2 ABI mismatch). Any naive `import pandas` in the global env crashes. | High | Reproduced: `ValueError: numpy.dtype size changed` | Project-local `.venv`; never rely on global site-packages |
| R-02 | **`scikit-learn` and `joblib` absent.** Phase 8/13 blocked without install. | High | Import probe: `NOT INSTALLED` | Install into `.venv` in Phase 1; pin versions |
| R-03 | **Working directory is inside `OneDrive`.** Sync can rewrite/lock files mid-test, corrupt generated artifacts, and produce phantom diffs. | High | Path is `...\OneDrive\Desktop\AIII LMS` | Treat `data/`, `artifacts/`, `.venv/` as disposable/regenerable; never store the only copy of an artifact there; keep all committed source in Git so a sync incident is recoverable |
| R-04 | **No Git repository.** No history, no rollback, no reproducibility trail. | High | `git rev-parse` failed | `git init` in Phase 1, `.gitignore` before first commit |
| R-05 | **No real learner data exists.** Any accuracy figure produced in the foundation phase is measured against synthetic labels. | High | No datasets found | Label every artifact `SYNTHETIC`; report metrics as *pipeline* metrics, never as evidence of human behaviour |
| R-06 | **Simulator-generated labels are self-fulfilling.** A model can learn the simulator's generative rules and score highly while learning nothing real. | High | Inherent to synthetic validation | Explicitly document as a known limitation; hold out whole *archetypes*, not just rows; report the "simulator artifact" risk in `FOUNDATION_STATUS.md` |
| R-07 | **Target definition is a modelling choice, not a discovery.** "Declining engagement" must be operationalised from observable proxies, and its validity on humans is unproven. | High | Specification §26 | Publish the label definition and its limitations in `DATA_DICTIONARY.md`; classify labels as `SYNTHETIC_LABEL` / `PROXY_LABEL` / `GROUND_TRUTH` |
| R-08 | **Windows/PowerShell environment.** Shell is PowerShell 5.1 (no `&&`, no bash heredocs). Portability claims about "runs locally" must be validated on the actual platform used. | Medium | Shell confirmed | All scripts tested on this machine; avoid POSIX-only assumptions in tooling |
| R-09 | **Scope inflation.** 24 acceptance criteria invite building all at once, producing a shallow, untested skeleton. | Medium | Specification §49 | Strict phase ordering; each phase gated on its own passing tests |
| R-10 | **Unsupervised claims of novelty or efficacy.** Pressure to describe the system as validated. | Medium | Specification §2, §51, §53 | `RESEARCH_PRINCIPLES.md`; per-artifact status stamps; no efficacy language without an experiment record |
| R-11 | **Model registry drift.** Silent changes to feature calculations would break reproducibility retroactively. | Medium | Specification §32 | Immutable `FEATURE_SET_V*` identifiers; changing a definition mandates a new version |
| R-12 | **Global env has 200+ packages including cloud/LLM SDKs.** Easy to accidentally introduce a prohibited dependency. | Low | `pip list` | Dependency allow-list review; requirements files are minimal and explicit |

---

## 9. FIRST IMPLEMENTATION PHASE (PHASE 1)

**Scope: repository organization, configuration, and schemas. No intelligence yet.**

| Step | Deliverable | Verification |
|---|---|---|
| 1.1 | `git init` + `.gitignore` (venv, caches, generated artifacts, secrets) | `git status` clean of noise |
| 1.2 | Project-local `.venv` via `uv`; install the §7.1 set; pin | `python -c "import pandas, sklearn, joblib, pydantic, fastapi"` succeeds inside venv |
| 1.3 | `pyproject.toml` with `requires-python = ">=3.11"`, tool config for `ruff`/`mypy`/`pytest` | `ruff check` and `mypy` run |
| 1.4 | `src/focus_engine/` package skeleton, one module per layer, clean public interfaces | import smoke test |
| 1.5 | `configuration/` — typed settings objects, all thresholds named and sourced (no magic numbers), env-var overrides | unit tests for defaults + overrides |
| 1.6 | `schemas/` — identifier, timestamp, event envelope, and versioning primitives | unit tests for validation + rejection of malformed input |
| 1.7 | `ARCHITECTURE.md`, `RESEARCH_PRINCIPLES.md`, `README.md`, `ROADMAP.md` reflecting **actual** state only | reviewed against implementation |
| 1.8 | Phase 1 test suite green; `ruff`/`mypy` clean | command output recorded in `FOUNDATION_STATUS.md` |

**Explicitly out of scope for Phase 1:** features, baselines, temporal logic, models,
interventions, outcomes, the simulator, the API, and any ML training.

---

## 10. WHAT WILL DELIBERATELY NOT BE BUILT YET

| Deferred | Reason |
|---|---|
| Any EduFlow integration, import, or dependency | Explicitly prohibited for this phase. |
| Deep learning (LSTM / Transformer / sequence models) | §41. Signal must be proven first. |
| Reinforcement learning for intervention selection | §18. Policies start transparent and deterministic. |
| LLM agents or LLM-in-the-loop components | §41 and §6 portability rules. |
| Hosted-LLM or cloud-provider SDKs | §6. Must run locally with no cloud account. |
| Automated production retraining from live data | §22. Retraining must be controlled, versioned, and approved. |
| Causal-effect claims from observational data | §20, §23. Only controlled experiments can support those, and none exist yet. |
| Identity resolution, name/email/device-fingerprint collection | §36. Pseudonymous IDs only. |
| Database vendor lock-in (Postgres, Mongo, etc.) | §6 portability. File-backed store behind an interface. |
| Personality / cognitive / diagnostic modelling | §45. Out of scope and scientifically unsupportable. |
| Any claim of novelty, priority, or scientific validation | §2, §51. Not established. |

---

## 11. AUDIT INTEGRITY STATEMENT

- Every "found" statement in this document was produced by an executed command in
  this session; the command and its result are quoted in §1.2 and §3.
- Every "not found" statement means the item was searched for and is absent — not
  that it was overlooked.
- No dataset, model, metric, competitor, paper, or patent is referenced anywhere in
  this project unless it is present in the repository. None are.
- No scientific or novelty claim is made in this document.

**End of PHASE 0 audit.**
