# ROADMAP

Phase sequence is fixed. Foundational phases are not skipped or merged, because each
later phase assumes the contracts the earlier one establishes. A phase is complete only
when its exit criteria are met and verified by an executed command.

**Status legend:** `COMPLETE` · `PARTIAL` · `IN PROGRESS` · `NOT STARTED`

`PARTIAL` means a phase delivered a working subset and its broader specification is
recorded as outstanding rather than met. A partial phase is never reported as complete for
the items it did not build.

---

## Current position

| Phase | Name | Status |
|---|---|---|
| 0 | Repository audit | `COMPLETE` |
| 1 | Architecture + schemas | `COMPLETE` |
| 2 | Event system | `COMPLETE` |
| 3 | Context system | `COMPLETE` |
| 4 | Synthetic learner simulator | `COMPLETE` |
| 5 | Feature engine | `COMPLETE` |
| 6 | Personal baseline | `COMPLETE` |
| 7 | Temporal state engine | `COMPLETE` |
| 8 | Baseline ML models | `COMPLETE` |
| 9 | Prediction + uncertainty | `COMPLETE` |
| 10 | Intervention policy | `COMPLETE` |
| 11 | Outcome engine | `COMPLETE` |
| 12 | Feedback architecture | `COMPLETE` |
| 13 | Evaluation framework | `PARTIAL` (core framework complete; split construction and advanced metrics planned) |
| 14 | API | `NOT STARTED` (development-only lab surface exists at `api/lab/` — deterministic, dev-only delivery, no auth, not the production API) |
| 15 | Security / privacy hardening & Student Intelligence | `COMPLETE` (security foundations in Phase 1; deep student intelligence & evidence reasoning completed) |
| 16 | Documentation + technical inventory | `PARTIAL` |
| — | Foundation status report | `COMPLETE` (`FOUNDATION_STATUS.md`) |
| — | Cross-phase clock determinism (defect in Phase 8/9 code) | `COMPLETE` |

---

## Phase 0 — Repository audit `COMPLETE`

Determine the environment, detect what exists, state what is missing, and record risks
without inventing anything.

**Exit criteria**
- `PROJECT_AUDIT.md` written, stating `GREENFIELD REPOSITORY` where applicable.
- Every "found" claim backed by an executed command.
- Every "not found" item actually searched for.
- Dependency plan with a reason per package.
- Risks enumerated with mitigations.

**Result:** See `PROJECT_AUDIT.md`. One blocking defect found and fixed (broken global
`pandas`/NumPy ABI conflict, R-01) by moving to a project-local environment.

---

## Phase 1 — Architecture + schemas `COMPLETE`

Repository organisation, dependency environment, typed configuration, and the shared
vocabulary every later layer depends on.

**Delivered**
- `git init` + `.gitignore` (secrets, caches, generated artifacts, EduFlow).
- Project-local `.venv`; pandas 3.0.6 / numpy 2.5.3 / scikit-learn 1.9.1 / pydantic
  2.13.5 / fastapi 0.141.1.
- `pyproject.toml`: `requires-python >= 3.11`, ruff / mypy strict / pytest config.
- `src/focus_engine/{schemas,configuration,utils}`.
- `ARCHITECTURE.md`, `RESEARCH_PRINCIPLES.md`, `README.md`, `ROADMAP.md`.

**Exit criteria — met**
| Check | Command | Result |
|---|---|---|
| Tests | `python -m pytest` | 223 passed |
| Lint | `python -m ruff check .` | All checks passed |
| Format | `python -m ruff format --check .` | Clean |
| Types | `python -m mypy` | Success, 11 source files |

**Defects found and fixed during this phase** (all by tests, not by inspection):
1. `zip(ladder, ladder[1:], strict=True)` always raised — replaced with `itertools.pairwise`.
2. Nested `BaseSettings` silently ignored `FOCUS_RUNTIME_*` / `FOCUS_SEC_*` environment
   variables, so settings appeared configurable while never changing.
3. Log redaction leaked an `Authorization: Bearer <token>` header: the key/value pattern
   consumed the word `Bearer` first, leaving the token in the record.
4. Log redaction missed quoted JSON keys such as `{"password": "..."}`.
5. Log redaction was not idempotent, so a second filter pass produced `[REDACTED]]`.

**Open item carried forward:** `requires-python` is `>= 3.11` but only 3.12 is verified.
Python 3.11.9 is present on this machine and will be checked when a lockfile is added.

---

## Phase 2 — Event system `COMPLETE`

A strongly typed behavioural event vocabulary. Every event validated; malformed events
rejected at the boundary.

**Delivered**
- Closed `EventType` taxonomy: 18 event types covering session lifecycle, questions,
  video, content, navigation, inactivity, and interventions.
- Per-type frozen payload models with `extra="forbid"`: each event type has a dedicated
  schema, and a mismatched payload is a validation error.
- `EventEnvelope` with pseudonymous IDs, timezone-aware UTC timestamps, type/payload
  matching validation, and required synthetic stamp when `origin=SYNTHETIC`.
- Synthetic-origin invariants: synthetic events cannot claim `provenance=OBSERVED`.
- `validate_event` entry point and `EventIngestor` with batch validation, duplicate
  detection, strict-mode error raising, and batch-size limits.
- `InMemoryEventStore` and `JsonlEventStore` behind an `EventStore` protocol; both are
  append-only with no mutation API.
- `EventQuery` with learner, session, event-type, time-range, origin, and limit filters.
- Session bounds summarisation for temporal analysis.

**Exit criteria — met**
| Check | Command | Result |
|---|---|---|
| Tests | `python -m pytest` | 307 passed |
| Lint | `python -m ruff check .` | 2 minor refactor suggestions (SIM102/SIM103) |
| Types | `python -m mypy` | Success, 15 source files |

**Key design decisions**
- Opaque content identifiers (`question_id`, `video_id`) require 8–128 characters to
  prevent re-identification by enumeration and to avoid ad-hoc short keys.
- Event type vocabulary is closed by design; an unknown type is a validation error, not
  a warning, ensuring downstream switches remain exhaustive.
- Timestamps are validated as timezone-aware on ingest; naive datetimes are rejected.

---

## Phase 3 — Context system `COMPLETE`

A signal is never interpreted in isolation. Content type, task difficulty, session
position, elapsed time, recent performance, prior intervention, cooldown, baseline
maturity, and current trajectory are all available to downstream layers.

**Delivered**
- `context/models.py` — `ContextModel` plus `SessionContext`, `ContentContext`,
  `PerformanceContext`, `InterventionContext`, `ContextVersion`, `ContextKey`.
- `context/engine.py` — `ContextEngine`, deriving all of the above from events alone.
- `ContextKey` names every context field a downstream layer may depend on, so an
  absent value is a checkable statement rather than a `None` inferred from a dict.
- `missing` is **derived** by a model validator, so a caller cannot declare a gap closed
  while the value is still absent.
- `BLOCKING_CONTEXT_KEYS` distinguishes gaps that make interpretation unsound
  (`session`, `elapsed_seconds`, `position_in_session`, `recent_accuracy`,
  `baseline_maturity`) from gaps that merely reduce precision (`difficulty`,
  `trajectory`).
- `origins` records contributing data origins; `is_synthetic_only()` keeps a
  synthetic-derived context from reading as observational.

**Exit criteria — met**
| Check | Command | Result |
|---|---|---|
| Tests | `python -m pytest` | 359 passed |
| Lint | `python -m ruff check .` | All checks passed |
| Types | `python -m mypy` | Success, 18 source files |

**Design decisions worth recording**
- **Replay determinism.** All elapsed times are measured against the timestamp of the
  last observed event, not the wall clock. Replaying a recorded session therefore
  reproduces the context it produced live. The injected clock is a fallback for the
  empty-stream case only. Verified by a test that runs the same stream through two
  engines with clocks ten and five hundred days apart and asserts identical output.
- **A stream may begin mid-session.** When no explicit `session_started` has been seen,
  the session start is inferred from the first event and `start_inferred=True` is set.
  The engine tolerates the gap and reports it rather than refusing to produce context.
- **Outcomes match by identifier, not recency.** With two interventions in flight,
  completing the first must not be recorded against the second. A completion whose
  start was never observed is ignored rather than invented — that gap is a producer
  defect, and silently repairing it would hide it.
- **No threshold is a constant.** `expected_session_seconds` normalises session
  position and is a constructor parameter, not a literal buried in a division. It is an
  engineering starting point, not an empirical finding.
- **The 17-second example.** A 17-second answer is slow on an easy multiple-choice item
  and unremarkable on a hard free-response one. The context layer supplies the
  difficulty, content type, topic, session position, and accuracy needed to compare it,
  and refuses to interpret: with no baseline maturity recorded, `is_interpretable()`
  returns `False`. The comparison belongs to a later layer.

---

## Phase 4 — Synthetic learner simulator `COMPLETE`

Configurable learner archetypes producing **event sequences**, not static rows.

**Delivered**
- `simulator/archetypes.py` — the closed `LearnerArchetype` enum, `LatencyShape`, and
  `ArchetypeProfile` coefficients, plus `pattern_is_measurable()`.
- `simulator/config.py` — `SimulationConfig` (frozen, `extra="forbid"`) and
  `SimulatorProvenance`, with the synthetic warning string defined once.
- `simulator/generator.py` — `generate_session`, `generate_session_events`,
  `session_seed`, and the invariant checks every returned stream must satisfy.
- `simulator/__init__.py` — public surface.

**Exit criteria — met**
| Check | Command | Result |
|---|---|---|
| Tests | `python -m pytest` | 441 passed at Phase 4 close |
| Lint | `python -m ruff check .` | All checks passed |
| Types | `python -m mypy` | Success, 22 source files |
| Cross-process reproducibility | 4 processes, `PYTHONHASHSEED` 0/12345/999/random | Identical BLAKE2b digest |

**Key design decisions**
- **Uniqueness is structural, not probabilistic.** Event and question identifiers are
  `blake2b(session_seed, index)`, so a session cannot emit a duplicate identifier no
  matter what the random stream does. The generator asserts this before returning.
- **The generator never reads a clock.** Time enters only through
  `SimulationConfig.start_time`. A static test asserts the generator source contains no
  `utc_now`, `datetime.now`, `time.time`, or `SystemClock`, because that defect is
  invisible to a behavioural test — it only surfaces as irreproducible output on a
  different day.
- **Payloads are internally consistent.** `question_answered.response_seconds` equals
  the interval between its `question_started` and `question_answered` timestamps, and
  `session_ended.duration_seconds` equals the real session span. A test asserts both to
  `1e-3`, so a consumer can trust the payload instead of recomputing it.
- **Dropout is opt-in.** `enable_dropout=False` by default, so a default run emits exactly
  `question_count` events and structural tests stay exact rather than probabilistic. When
  enabled, completion is drawn from the archetype's `session_completion_rate` and the
  truncated session ends with a distinct reason.
- **A too-short session says so.** `min_questions` is not a generation constraint but a
  consumer-side check, surfaced as `provenance.pattern_measurable`. A three-question
  `RECOVERY` session is generated on request, but its provenance records that the
  archetype's pattern is not separable from noise at that length.
- **Drift must exceed the noise meant to hide it.** Coefficients are chosen so a shaped
  archetype's peak departure is several times its `noise_scale`. An early `RECOVERY` drift
  of 0.045 produced an 18% peak against 8% jitter, which failed its own test at 9/12 seeds;
  raising it to 0.09 fixed the *data*, not the test. A pattern buried under its own noise
  would make distinctness unsatisfiable for reasons unrelated to whatever is being tested.
- **Distinctness is measured, not asserted.** `ArchetypeProfile.signature()` returns the
  coefficient vector and a test asserts all eight are pairwise unique; separate tests assert
  the *output* follows from the coefficients (difficulty-latency correlation above 0.3 for
  `CONTEXT_SENSITIVE`, dispersion at least 1.5× for `NOISY`, a dip that returns to
  baseline for `RECOVERY`, no index trend for `STABLE`).

**What this does not establish**
The coefficients are synthetic test-fixture parameters, not empirical estimates of any
human population. Nothing here is evidence about real learners, and a model fitted on this
output is fitted to a simulator's assumptions. The generated stream is deliberately
question-centric; richer event mixes remain a later concern.

---

## Phase 5 — Feature engine `COMPLETE`

Events to measurable features, with a registry carrying name, definition, data source,
calculation, version, and validity requirements.

**Delivered**
- `features/models.py` — `FeatureName`, `FeatureCategory`, `FeatureValueType`,
  `FeatureAvailability`, `INSUFFICIENT_DATA_MARKER`, the validated `FeatureSpec`, and the
  frozen `TypedFeatureValue` / `FeatureValue` output pair.
- `features/engine.py` — the append-only `FeatureRegistry`, `FEATURE_SPECS_V1`, one
  calculation per feature behind a closed dispatch table, the stateless `FeatureEngine`,
  and `compute_features`.
- `features/__init__.py` — public surface.
- `schemas/versioning.py` — `checked_version()`, so a version string passed to a registry
  is held to the same grammar as one passed through a model.

**Fourteen features across five categories:** session position and elapsed time; content
difficulty and type; recent accuracy, recent latency, and the two windowed trends;
intervention count, cooldown, and time since last; trajectory and baseline maturity as
closed encodings.

**Exit criteria — met**
| Check | Command | Result |
|---|---|---|
| Tests | `python -m pytest` | 495 passed |
| Lint | `python -m ruff check .` | All checks passed |
| Types | `python -m mypy` | Success, 25 source files |
| Versioning | `test_registry_refuses_to_reregister_a_version` | Version re-registration refused |
| Leakage | `test_future_answers_do_not_enter_the_short_window` | Future answers do not move a computed trend |

**Key design decisions**
- **A missing value is not replaced by a plausible constant.** The first draft of this
  engine substituted `0.5` for absent difficulty, `20.0` for absent response time, and
  `"unknown"` for absent identifiers, and clamped out-of-range results into range. Each
  substitution would have been indistinguishable from a measurement. Absent inputs now
  yield `INSUFFICIENT_DATA` with a reason, and an out-of-range result is reported as the
  defect it is instead of being clamped — a test asserts the reason says so.
- **"Never intervened with" is not "intervened with just now".** `seconds_since_last`
  yields `INSUFFICIENT_DATA` rather than `0.0`, so the two states stay distinguishable.
  The reasoning is the same one that keeps a ten-question window from being read as a
  personal baseline.
- **The leakage invariant is enforced, not trusted.** `_answer_windows` drops events later
  than `context.reference_time` before computing either trend. Relying on callers to pass
  only history would make the property hold only in the cases nobody tests.
- **Provenance is not a two-valued flag.** A vector records `origins` as a set, so a
  stream mixing real and synthetic events is labelled `mixed` rather than being forced to
  `real`. `TypedFeatureValue` additionally refuses to report `AVAILABLE` in a vector whose
  origins are purely synthetic.
- **The marker cannot lie.** `TypedFeatureValue` rejects a value whose availability and
  marker disagree, and rejects a computed value carrying an insufficiency reason. A
  marker consumed downstream as a number is the failure this prevents.
- **An unknown version raises.** `UnsupportedFeatureSetError` is raised at engine
  construction, not swallowed into an empty vector — an empty vector would otherwise reach
  a model as a legitimate input row.
- **The registry is append-only.** Re-registering a version is refused, because the
  definitions behind a version are what make an already-computed vector interpretable.
  `test_registry_supports_a_new_version_without_touching_history` adds `FEATURE_SET_V2`
  and asserts `FEATURE_SET_V1` is unchanged.
- **Two state features are deliberately inadequate.** `trajectory_encoded` and
  `baseline_maturity_encoded` return `INSUFFICIENT_DATA` because no engine could yet
  populate them. Publishing `unknown` would present a not-yet-implemented engine as a
  completed measurement. *Superseded at Phase 6 close:* `baseline_maturity_encoded` now
  reports a real value whenever a context carries a maturity, which the baseline engine
  makes reachable; `trajectory_encoded` remains a genuine absence until Phase 7.
  *Superseded at Phase 7 close:* `trajectory_encoded` now reports a real value whenever a
  context carries a trajectory, which the temporal engine makes reachable. Both state
  features still report `INSUFFICIENT_DATA` when the value is genuinely absent, which is
  the correct absence rather than a placeholder.

**What this does not establish**
A feature is a measurement of a recorded interaction, not of a person. Nothing here
establishes that any feature relates to attention, motivation, or comprehension; that is
the claim a later layer would have to earn and the evaluation framework must test. No
feature has been checked for stability across learners or against a label, which is the
measurement that would justify keeping it.

---

## Phase 6 — Personal baseline `COMPLETE`

The learner's own normal behaviour, with a maturity ladder and explicit cold start.

**Exit criteria — met**
| Check | Command | Result |
|---|---|---|
| Maturity ladder | `test_a_learner_walks_the_whole_ladder_in_order` | `NEW` → `EARLY` → `DEVELOPING` → `ESTABLISHED` in order, and never backwards |
| Cold start labelling | `test_cold_start_returns_a_labelled_population_prior` | A prior is returned with `POPULATION_PRIOR` basis and its cohort named |
| Cold start with no prior | `test_cold_start_without_a_prior_is_unavailable_not_zero` | `UNAVAILABLE` with a reason, never a default |
| Robust by default | `test_robust_statistics_are_the_default` | `ROBUST_WINSORISED_MAD` is the default; `MEAN_STDDEV` remains configurable |
| Outliers do not mask change | `test_the_non_robust_estimator_fails_the_same_outlier_test` | One 3600-second value inflates the non-robust spread >50× and the robust one not at all |
| A real change stays visible | `test_a_real_change_after_an_outlier_is_visible_and_not_masked` | A sustained shift is >10 robust z-units and >10× the non-robust reading |
| Tests | `python -m pytest` | 558 passed |
| Lint | `python -m ruff check .` | All checks passed |
| Types | `python -m mypy` | Success, 29 source files |
| Layer boundary | `test_the_baseline_layer_imports_no_later_layer` | No import of a later layer |
| Composition | `test_baseline_maturity_flows_back_into_the_context_and_features` | A computed maturity reaches `baseline_maturity_encoded` and closes the blocking gap |

**Key design decisions**
- **The basis is a type constraint, not a convention.** `BaselineReference` and
  `DeviationResult` both carry a maturity and an `InferenceBasis`, and a validator refuses
  an inconsistent pair. Separation 2 is therefore enforced by the constructor: there is no
  way to build a population estimate that claims to be personal.
- **The engine ships no priors.** A hard-coded "typical" response time is a claim about real
  learners that nothing in this repository can support, and freezing one in would make it
  indistinguishable from a value actually estimated from a cohort. Priors are
  caller-supplied, carry a `source` and a `n_reference`, and a cold start with none is
  `UNAVAILABLE` with a reason rather than a plausible number.
- **Robustness requires both halves of the estimator.** The first draft winsorised only the
  centre. A bounded centre alone is not enough: the spread was still recomputed around a
  displaced centre, so one extreme value could both move the baseline *and* widen the
  tolerance around it, and the two effects masked each other. Bounding the centre in
  dispersion units *and* taking a median absolute deviation about that centre is what makes
  the outlier test pass rather than merely move the number.
- **The MAD is taken about the reported centre, not the sample median.** The published
  centre is a time-weighted mean; measuring dispersion about an unweighted median would
  pair a centre with a spread derived from a different centre, and the standardised
  deviation would then be measured against a reference that was never reported.
- **Zero spread is a reported state, not a division.** Fewer than two retained values, and
  a window in which every value is identical, both yield no standardised deviation with an
  explicit reason. Reporting an infinite z-score would be the one place the engine
  fabricated a number.
- **Retention thins rather than truncates.** `decimate` repeatedly halves an over-limit
  window, which bounds memory while letting the retained sample span a growing interval —
  necessary, because a learner who never truncates reaches `ESTABLISHED` and one who
  truncates at a fixed count never does. Halving an odd-length window initially dropped the
  newest value; a test caught it, and the newest observation is now held back explicitly.
- **An out-of-range value is rejected, not clamped.** Clamping an impossible measurement to
  the boundary replaces a visible defect with an invisible plausible number, and
  `DimensionStatistics` re-validates its own centre for exactly this reason.
- **Absence and defect are different states.** A source feature reporting
  `INSUFFICIENT_DATA` is untouched and uncounted; a value of the wrong type or outside the
  dimension's range is counted as a rejection with a reason. Conflating them would misreport
  data quality in one direction or the other.
- **The context must be revalidated, not copied.** `missing` is derived by an `after`
  validator, and `model_copy(update=...)` skips validators — so a maturity added that way
  would leave the blocking gap recorded as open. The integration test asserts the gap
  actually closes, and the baseline engine revalidates on every update for the same reason.
- **A profile records the settings that shaped it.** Every `BaselineProfile` carries a
  `settings_fingerprint`, so a stored baseline cannot later be read as though it had been
  built under thresholds it never saw.
- **`intervention_count` is deliberately not a dimension.** It is an output of the
  intervention policy, so baselining it creates a loop in which the policy's own actions
  raise the bar it is later measured against, and sustained over-intervention would
  progressively hide itself.

**What this does not establish**
A baseline describes how one learner has behaved so far. It does not establish that a
deviation from that baseline means anything, that the behaviour measured is the behaviour of
interest, or that the maturity thresholds are correct for any real population. The
thresholds in `BaselineSettings` are documented defaults with stated reasoning, not
validated choices; establishing them would need the evaluation framework, whose core
arrived in Phase 13 but which has not been pointed at these thresholds and cannot be
pointed at a threshold without a non-synthetic corpus. No feature used here has been checked for stability
across learners, and a baseline computed from simulator output reflects that simulator's
assumptions and nothing more.

---

## Phase 7 — Temporal state engine `COMPLETE`

State, previous state, duration, direction, rate of change, stability, persistence — and
the ability to distinguish a temporary anomaly from a sustained trajectory change.

**Delivered**
- `temporal/models.py` — `BehavioralEngagementState`, `TrendDirection`, `TrendCharacter`,
  `ChangeKind`, `TemporalObservation`, `DeviationPoint`, `TemporalTrack`, `TrajectoryChange`,
  `StateEvidence`, `StateTransition`, `TemporalState`, `weakest_maturity`.
- `temporal/statistics.py` — `current_sign_run`, `signed_direction`,
  `change_per_observation`, `is_sustained`, `deviation_dispersion`, `mean_standardised`.
- `temporal/engine.py` — `TRACKED_DIMENSIONS`, `TemporalEngine`, `TemporalStateError`.
- `tests/integration/test_temporal_integration.py` — the full traversal, closing the loop
  from the temporal state back into the feature set.
- `trajectory_encoded` now reports a real value from a populated context.

**Exit criteria — met**
| Check | Command | Result |
|---|---|---|
| Trajectory vs anomaly | `test_one_large_excursion_is_an_anomaly_and_not_a_trajectory`, `test_a_run_below_the_persistence_minimum_stays_an_anomaly`, `test_a_sustained_run_produces_exactly_one_trajectory` | 1 point → `ANOMALY`; 3 points → `TRAJECTORY`; `TrajectoryChange` unconstructible below its own run length |
| Clock injected | `test_time_enters_only_through_the_injected_clock` | Two engines whose clocks differ by decades produce equal states from the same stream |
| Tests | `python -m pytest` | 642 passed, of which 78 in `tests/unit/test_temporal.py` and 6 in `tests/integration/test_temporal_integration.py` |
| Lint | `python -m ruff check .` | All checks passed |
| Format | `python -m ruff format --check .` | 51 files already formatted |
| Types | `python -m mypy src` | Success, 33 source files |
| Layer boundary | `test_the_temporal_layer_imports_no_later_layer` | No import of a later layer |
| Composition | `test_a_trajectory_reaches_the_context_and_becomes_a_measurement` | The trajectory travels simulator → context → features → baseline → temporal → context → features |

**Key design decisions**
- **Absence is unconstructible, not merely discouraged.** A learner with no deviation history
  has no state. The engine reports `INSUFFICIENT_DATA` with a reason, and a validator
  refuses to pair that state with any evidence of movement. Substituting `STABLE` because
  nothing has gone wrong is the most seductive failure in behavioural analytics; making the
  state unconstructible is the only answer that survives a code review nobody performs.
- **Direction is read from the run, not the rate window.** A rate-of-change window that
  includes the pre-run zeros preceding a departure reports a *shrinking* departure
  (`[-3, -2, -1]`) as worsening, because the mean rate falls while the deviation still
  deepens. A test caught this on a real recovery sequence, and the run's own sign sequence
  is now the reading, with the window consulted only for a run of one.
- **A run that never started is recorded as such.** `INSUFFICIENT_DATA` always clears
  `run_started_at`, so a warm-up run at value 1 cannot be read as a run in progress.
- **Bounded retention is observable, and its boundary is a real case.** A trajectory's
  start can age out of the window while its end is still present. The answer is layered:
  each track keeps a ledger of the trajectories it emitted, state trajectories are rebuilt
  from those ledgers every ingest so the two cannot drift, the ledger check is the
  always-applicable anti-fabrication guard, and the retained-endpoint check runs only while
  the whole run survives. Once the start has aged out, the run is only partly verifiable and
  the ledger is trusted rather than fabricating a failure.
- **The state is a window, not a tally.** `TemporalState.observations` counts retained
  points; the lifetime counts live on the track as `recorded` and `dropped`. Reporting the
  lifetime total as the state's observation count would make a long-lived learner look
  newly arrived on every fresh state, which is exactly the cold-start signal a downstream
  layer would act on.
- **The weakest link records the maturity.** A state is personal only if every contributing
  dimension is; `weakest_maturity` is applied across the tracks, so a single
  prior-derived dimension demotes the whole state to `POPULATION_PRIOR`. The integration
  test makes this concrete: it drives one dimension to `ESTABLISHED` and the state still
  reports `NEW`, because the other three tracked dimensions have never been measured. A
  state that reported `ESTABLISHED` there would be a personal inference resting on three
  dimensions that do not exist.
- **`change` describes the run in progress; `trajectories` is the ledger of every run that
  ever persisted.** A stream that recorded a trajectory and then crossed zero restarts the
  run, so an `ANOMALY` current change can legitimately sit beside a recorded trajectory.
  Reading the ledger as the current claim would report a learner who has already reversed
  as still sustaining. The integration test asserts the current change against the
  *current* run's length, not against the ledger.
- **The layer consumes deviations, never re-derives them.** The baseline layer already
  decided what a deviation is. Recomputing it from features would create a second,
  divergent definition of the same quantity.

**Process defect found and fixed at Phase 7 close**
`ruff format --check .` was verified at Phase 1 and then not re-run for five phases. When it
was finally run, 20 of 50 source files were unformatted — the lint and type gates were
being reported green while the formatting gate silently rotted. All 51 files are now
formatted, and the check is in the Phase 7 exit table so the next phase inherits a command
that is actually executed. A gate written into a table nobody re-runs is not a gate.

**What this does not establish**
The layer distinguishes a sustained run from a transient excursion using configured run
lengths, an epsilon, and dispersion thresholds. Those thresholds are documented engineering
defaults with stated reasoning, not validated choices — establishing them would need the
evaluation framework (Phase 13). Nothing here shows that any tracked dimension relates to
attention, motivation, or comprehension, or that a `TRAJECTORY` is a condition worth acting
on. The layer classifies movement; whether a movement is meaningful is the uncertainty
engine's question in Phase 9 and the policy's in Phase 10, and collapsing those questions
into this one would be the same error in a new place. A temporal state computed from
simulator output reflects that simulator's assumptions and nothing more.

---

## Phase 8 — Baseline ML models `COMPLETE`

Logistic regression, random forest, gradient boosting. No deep learning.

**Delivered**
- `models/encoding.py` — `FeatureRequirement`, `DesignMatrix`, `RejectedRow`,
  `CATEGORICAL_VOCABULARIES`, `default_features`, `encode_rows`, `requirements_from_schema`.
- `models/artifacts.py` — `ModelArtifact`, `save_artifact`, `load_artifact`,
  `class_one_probabilities`.
- `models/trainers.py` — `Algorithm`, `TrainingConfig`, `TrainingOutcome`, `train_model`.
- `models/predictors.py` — `Prediction`, `PredictionRefusal`, `predict_with_artifact`,
  `score_batch`.
- `models/registry.py` — `ModelRegistry`, `ALLOWED_TRANSITIONS`, `RegistryTransition`.

**Exit criteria — met**
| Check | Command | Result |
|---|---|---|
| Training | `test_each_algorithm_trains`, `test_each_algorithm_learns_a_learnable_signal` | Each algorithm fits and recovers a learnable signal; `test_the_record_names_the_concrete_estimator` pins the algorithm name |
| Reproducibility | `test_the_same_seed_produces_identical_predictions`, `test_a_different_seed_produces_a_different_model` | Same seed → identical probabilities; different seed → a different fit |
| Seed provenance | `test_the_record_keeps_the_run_seed_not_the_derived_one`, `test_different_model_identities_do_not_share_a_seed` | The record keeps the master seed while the estimator receives a derived child seed; two model identities never collide |
| Split ownership | `test_the_holdout_split_is_named`, `test_mismatched_schemas_are_refused` | The caller owns the split, the split is recorded, and a train/holdout schema disagreement is refused |
| Serialisation | `test_the_estimator_round_trips`, `test_the_record_round_trips`, `test_the_schema_round_trips`, `test_every_algorithm_round_trips` | Estimator, record, schema, seed, and timestamp round-trip cleanly for every algorithm |
| Environment | `test_training_records_the_library_versions`, `test_the_library_versions_round_trip`, `test_saving_preserves_the_training_environment`, `test_drift_is_reported_when_a_version_moved` | The library versions that can move a numeric result are recorded, survive a round trip, and are reported as drift on reload |
| Loading refusals | `test_a_tampered_schema_is_refused`, `test_a_model_without_provenance_is_refused`, `test_an_unfitted_or_non_estimator_is_refused`, `test_a_sidecar_that_is_not_metadata_is_refused` | A tampered digest, absent sidecar, forged metadata, and a non-estimator payload are all refused |
| Prediction | `test_a_complete_vector_is_scored`, `test_the_prediction_carries_its_provenance`, `test_the_threshold_is_reported_and_honoured` | A complete vector scores; model, schema, and threshold travel with the result |
| Refusals | `test_absent_feature_is_rejected_not_imputed`, `test_a_vector_with_an_absent_feature_is_refused`, `test_an_unattributed_vector_is_refused`, `test_a_vector_from_another_feature_set_is_refused` | Absent features are named, unattributed vectors are refused, a foreign feature-set version is refused |
| Batch prediction | `test_refusals_are_separated_from_predictions`, `test_batch_predictions_match_single_predictions`, `test_a_batch_mixing_feature_set_versions_is_refused` | Refusals are counted separately; batching never changes a score |
| Registry lifecycle | `test_production_is_only_reachable_from_candidate`, `test_skipped_stages_are_refused`, `test_retired_is_terminal`, `test_every_transition_appends_a_record` | `PRODUCTION` is reachable only from `CANDIDATE`; `RETIRED` has no outgoing edges; history is append-only |
| Synthetic-only guard | `test_synthetic_only_model_cannot_be_promoted`, `test_a_failed_promotion_does_not_change_the_recorded_status` | Promotion of a simulator-derived model is refused and leaves the recorded status untouched |
| Attribution | `test_author_and_rationale_are_mandatory`, `test_the_transition_log_names_every_decision` | Every transition names who decided and why |
| Metrics | `test_a_single_class_holdout_reports_no_roc_auc`, `test_probability_metrics_stay_in_range`, `test_a_well_calibrated_model_scores_a_low_brier` | Undefined rank metrics are `None`; probability metrics stay in range; a well-calibrated model scores a low Brier |
| Invalid input | `test_an_empty_training_set_is_refused`, `test_a_single_class_training_set_is_refused`, `test_a_negative_seed_is_refused`, `test_a_blank_model_id_is_refused` | Degenerate input is refused at the boundary rather than fitted |
| Tests | `python -m pytest` | 800 passed, of which 158 under `tests/unit/models/` |
| Lint | `python -m ruff check .` | All checks passed |
| Format | `python -m ruff format --check .` | 71 files already formatted |
| Types | `python -m mypy src` | Success, 39 source files |
| Layer boundary | `test_no_model_module_imports_a_later_layer`, `test_model_modules_import_only_permitted_layers` | No model module imports a later layer or an undeclared sibling, including via deferred function-level imports |
| Public surface | `test_the_models_package_exports_only_known_names` | Every name in `models.__all__` is bound and listed once, so the export list cannot drift from the module |

**Key design decisions**
- **Imputation does not exist.** Absent features produce a refusal with the feature name
  and the reason. A prediction computed from a subset of a model's inputs is not that
  model's output and should not be presented as one.
- **Unattributed vectors are refused.** `learner_id` is optional in `FeatureValue`, but a
  vector with no learner cannot be sliced per learner, held out per learner, or traced back
  when a result is questioned. The encoding layer produces a sentinel and a reportable
  refusal rather than silently absorbing it into the training set.
- **Categorical vocabularies are declared, not learned.** A vocabulary learned from a
  training set is a vocabulary that can grow at serving time, and a new category would then
  either have no column or be folded into a catch-all the model was never trained against.
  Declaring the sets in code means an unrecognised value is a refusal rather than a silent
  reinterpretation.
- **Integer features must be whole numbers.** Rounding 4.5 to 5 silently changes a count
  the learner actually produced, and the change is invisible in every downstream metric.
- **The schema round-trips with the artifact.** A reloaded model verifies its stored schema
  digest against the schema itself before scoring, and refuses a disagreement. A model
  whose schema has drifted from the data it will receive is a bug that happens to still be
  silent, and failing loudly at load time is cheaper than failing quietly in production.
- **Reproducibility is asserted as predictive equivalence.** The same seed and data must
  produce identical fitted parameters and therefore identical predictions. Byte-for-byte
  pickle identity is not guaranteed, because that would make the contract unmeetable
  across a dependency upgrade even when the model's behaviour is unchanged.
- **The seed reaches the estimator deterministically.** A child seed is derived from the
  master seed, the model identity, and a label via `derive_seed`, then passed as
  `random_state` to the estimator. A test on overlapping data confirms that different seeds
  actually produce different fits; a test on the same seed confirms they agree.
- **The environment is recorded, because reproducibility is a claim about one.** A seed
  and a dataset are not sufficient: two runs sharing both still disagree if scikit-learn
  was upgraded in between, and nothing about the fitted parameters would reveal that the
  difference came from the environment rather than the data. `ModelArtifact` carries the
  resolved versions of the packages that can move a numeric result, and `version_drift()`
  reports them against the current environment on reload. Drift is reported and not
  refused, because a patch-level upgrade frequently does not move a fit, and a loader that
  refused on any version difference would be unusable; a caller who needs the guarantee can
  act on an empty result.
- **Re-saving preserves the environment the model was *fitted* in.** A model relocated
  between machines is re-saved by the new machine. A save that overwrote the recorded
  versions would make the artifact describe an environment that never produced it, which is
  the one claim the field exists to support.
- **Probability extraction lives in `artifacts.py`.** Both training and scoring read the
  positive-class column from `predict_proba`, and the shared helper prevents them from
  ever disagreeing about which column that is.
- **Metrics that are undefined on a single-class split are `None`, not 0.5.** A ROC-AUC of
  0.5 on a split with no positive class would be a number that was never measured, and a
  later reader could not tell the difference between a real 0.5 and an invented one.
- **The training function does not decide the split.** Split policy stays with the caller,
  so that temporal, per-learner, or stratified splits are a decision made outside this
  function and recorded elsewhere. A function that decides its own splits is a function
  that can be reconfigured into leaking.
- **The registry is append-only.** History is extended, never rewritten. Registration is the
  only entry point and always starts at `EXPERIMENTAL`. A model cannot jump to production
  without passing through the declared stages. The author and rationale are mandatory on
  every transition. Retired is terminal. Synthetic-only models cannot be promoted.
- **The digest is over what is registered, not over when.** Timestamps are excluded
  deliberately, so a digest that changed with every run would be useless for comparing two
  registries' contents.
- **`n_jobs=1` for tree ensembles.** Parallel job schedulers across different library
  versions may assign work to threads in a different order, which means a claim of exact
  reproducibility for a fit with `n_jobs>1` cannot be verified. Logistic regression uses
  `lbfgs`, which is already single-threaded and rejects the deprecated `n_jobs` parameter.

**Process defect found and fixed at Phase 8 close**
`.gitignore` carried an unanchored `models/` entry, written to ignore serialised model
output. An unanchored pattern with a trailing slash matches at any depth, so it also matched
`src/focus_engine/models/` and `tests/unit/models/` — the entire layer delivered in this
phase. Three consequences, none of them visible from the gate results alone:

- **The layer was invisible to version control.** Fourteen files of new code and tests
  matched `.gitignore` and would not have been committed.
- **The lint and format gates were green while covering none of it.** Ruff respects
  `.gitignore` by default, so `ruff check .` and `ruff format --check .` silently skipped
  the model layer. The Phase 7 close recorded a similar failure — a format gate that was
  written into a table and not re-run — and this is the same class of defect one phase
  later: a gate reporting success because it examined the wrong file set. The count itself
  was the tell, and the only reason it was chased down is that 57 formatted files did not
  reconcile with 65 Python files on disk.
- **A real defect was hiding behind it.** With the exclusion corrected, the lint gate
  immediately reported `F821 Undefined name FeatureVector` in `tests/unit/models/scenarios.py`:
  a local annotation named a type that does not exist in `focus_engine.features`. It had
  survived because `from __future__ import annotations` defers evaluation, and because
  `mypy` is configured with `files = ["src"]` and so never looked at the tests.

The fix is to anchor the pattern to the repository root, where a scratch directory of
serialised models actually lives; `*.joblib` and `*.pkl` already cover the artifacts
themselves. Two things follow from this that are worth carrying forward. A gate's result is
evidence about the files it examined, so a gate that examined too few files is not a pass —
and the honest way to tell is to compare the reported count against the files on disk. And
a green lint run over a newly added layer is worth nothing until something has confirmed the
new layer was inside the linted set.

**What this does not establish**
The models are fitted on synthetic feature vectors produced by the simulator. A trained
model's metrics reflect the simulator's assumptions and nothing more. No model has been
validated against real learner data; that question is Phase 13's, and the core framework
Phase 13 delivered does not answer it — it scores recorded predictions against observed
evidence under a leakage-safe window, and every corpus it can currently be run on is
synthetic. The hyperparameters are
documented defaults with stated reasoning, not validated choices. The split used for
holdout metrics is caller-supplied, and remains so: the evaluation framework
(Phase 13) validates split *settings* but does not yet construct a split, and its slice
reporting and calibration analysis are outstanding rather than delivered. The registry
models a process, not a decision: a model's presence at `PRODUCTION` means it was promoted
through the declared stages with an attribution, not that it should be serving learners
right now.

---

## Phase 9 — Prediction + uncertainty `COMPLETE`

Calibrated probability of a labelled behavioural state, plus an honest verdict when the
evidence is insufficient.

**Delivered**
- `uncertainty/calibration.py` — `ReliabilityBin`, `reliability_table`, `CalibrationReport`.
- `uncertainty/evidence.py` — `EvidenceVolume`, `maturity_ceiling`.
- `uncertainty/explanation.py` — `SignalContribution`, `PredictionExplanation`,
  `occlusion_contributions`, `DEFAULT_MIN_CONTRIBUTION`.
- `uncertainty/outcomes.py` — `Verdict`, `Remedy`, `REMEDIES`, `Constraint`,
  `ConfidenceAssessment`, `OutcomeIdentity`, `PredictionOutcome`.
- `uncertainty/engine.py` — `UncertaintyEngine`, `band_confidence`.
- `models/encoding.py` — `column_references` (added): one training-split median per
  encoded column, used as the occlusion reference.
- `models/artifacts.py` — `ModelArtifact.feature_reference` (added), persisted in the
  sidecar and restored on load; a pre-Phase-9 sidecar loads with an empty mapping.
- `models/trainers.py` — computes `feature_reference` from the training split only, and
  the corrected `_expected_calibration_error` (see the defect note below).
- `configuration/thresholds.py` — `UncertaintySettings` extended with the confidence
  ladder, evidence gate and saturation, calibration limit, and a validated ascending
  ladder.

**Defect found and corrected in Phase 8 code**
`_expected_calibration_error` compared hard-prediction accuracy against stated confidence,
which yields a value near 0.5 for a perfect model and near 0.5 for a useless one alike. It
now compares mean predicted probability against observed positive frequency within each
reliability bin, and the top bin is closed so a stated probability of `1.0` is counted.
Three known-answer cases pin it: perfectly calibrated `0.0`, inverted `1.0`, and a model
stating `0.3` against 30% positives `0.0` against 90% positives `0.6`. Had this gone
unfixed, every `UNKNOWN` verdict downstream would have been raised for the wrong reason.

**Defect found and corrected in Phase 8 and Phase 9 code: two direct wall-clock reads**

Found at the Phase 13 close, by the same static scrutiny the Phase 4 exit criterion applies
to the generator. `models/predictors.py` stamped `Prediction.computed_at` from `utc_now()`
while every other time-aware layer took a `Clock`, and `uncertainty/engine.py` had no clock
at all. The effect was narrow and real: a replay over identical events produced identical
probabilities but different timestamps, so a recorded decision could not be reproduced
field for field, and a replay could not demonstrate it had replayed the same decision. The
Phase 13 exit table already claimed *"The engine holds no mutable state, takes an injectable
`Clock`, and holds no current-time dependency"* — true of the evaluation engine, and
silently untrue of the two layers feeding it.

The fix introduced no new clock. `predict_with_artifact` is now dated by
`vector.computed_at`, which is what its own two refusal branches already did, so the success
and refusal paths of one function can no longer report different instants for the same
vector depending only on whether the model had an opinion. `UncertaintyEngine` gained the
`clock` field the temporal, policy, and outcome engines already take, defaulting to
`SystemClock` so an unconfigured caller is still dated. Nine regression tests in
`tests/unit/test_clock_determinism.py` cover the property from both sides — the same inputs
plus the same clock must agree, *and* a different clock must produce a different stamp,
because a test asserting only the first half still passes against a layer that stopped
stamping a time altogether.

The remaining `utc_now()` calls in `models/` and `schemas/` are the write-side stamps
(`trained_at`, `metadata_written_at`, `recorded_at`) and the `EventEnvelope` ingest default.
These record when a record was written rather than inferring anything from evidence, so they
are a different class from an inference timestamp and were left as they are. The distinction
is recorded here because "no `utc_now` in `src`" is not a rule this repository holds, and a
reviewer should not apply it blind.

**Process note — a stale gate, for the third time**

The format gate was stale when this work was picked up: the nine new tests had been added and
the source fix applied, but `ruff format --check .` had not been re-run, so it reported 114
files clean while 115 were on disk. This is the same defect class as the Phase 7 close (a
format gate verified once and not re-run for five phases) and the Phase 8 close (a lint gate
green over a file set that excluded the whole new layer). Three occurrences is a pattern
rather than an accident, and the pattern is not that the checks are unreliable — every gate
here is a correct check. It is that a gate is only evidence about the moment it ran, and a
table recording a green result is a claim about the past, not a state of the repository. The
only defence that has worked each time is running the gates rather than reading the table,
and comparing the reported file count against the files on disk, which is what surfaced
both earlier instances.

**Exit criteria - met**
| Check | Command | Result |
|---|---|---|
| `UNKNOWN` and `INSUFFICIENT_DATA` reachable | `test_insufficient_data_is_reached_below_the_evidence_gate`, `test_unknown_is_reached_when_calibration_cannot_be_measured`, `test_unknown_is_reached_when_calibration_is_not_believable` | All four verdicts are reached: below the evidence gate gives `INSUFFICIENT_DATA`; unmeasured calibration and untrustworthy calibration each give `UNKNOWN` |
| `REFUSED` kept distinct | `test_refused_propagates_from_a_missing_feature`, `test_refused_propagates_through_assess_from_a_refusal` | A caller error stays a `REFUSED` with its reason and missing features, and is never folded into a data-collection verdict |
| Confidence derived, never invented | `test_all_four_ceilings_are_recorded`, `test_the_confidence_is_a_derived_not_invented_number`, `test_binding_is_the_ceiling_that_set_the_value`, `test_a_value_that_is_not_the_lowest_ceiling_is_refused` | All four ceilings are recorded, the value equals their minimum, the binding constraint is identified, and an assessment claiming any other value is refused at construction |
| Maturity cap independent of model output | `test_a_confident_model_is_capped_by_maturity`, `test_maturity_bounds_the_output_regardless_of_model_confidence`, `test_the_cap_tightens_on_a_tighter_ladder`, `test_the_ceiling_ignores_anything_a_model_might_say` | A model reporting above `0.9` is capped at the `EARLY` ceiling of `0.50`; the cap reads configuration and is a function of maturity alone |
| Evidence ramp | `test_sufficiency_is_exactly_the_configured_gate`, `test_below_the_gate_there_is_no_defensible_confidence`, `test_the_ramp_is_logarithmic_so_the_gate_is_not_a_cliff`, `test_evidence_stops_constraining_at_the_saturation_point` | Sufficiency is exactly the gate; confidence rises logarithmically to `1.0` at saturation and stops there |
| Remedy ordering | `test_bad_calibration_trumps_sufficient_evidence`, `test_every_verdict_has_exactly_one_remedy`, `test_the_two_negative_verdicts_have_different_remedies`, `test_a_refusal_is_fixed_at_the_input_not_by_waiting` | Calibration is checked before evidence, because a broken model is not repaired by collecting more data; each verdict maps to exactly one remedy and the two negatives differ |
| Explanation shows only contributing signals | `test_only_features_above_the_tolerance_are_reported`, `test_a_feature_the_model_ignores_is_never_attributed`, `test_probability_moves_only_for_columns_the_model_weights`, `test_moving_a_feature_further_from_its_reference_enlarges_its_contribution` | Attribution is verified against the model's own coefficients column by column, a zero-coefficient feature is never credited, and signals below the tolerance are omitted rather than reported small |
| Empty explanation surfaced | `test_a_very_high_tolerance_produces_an_empty_explanation`, `test_an_empty_explanation_still_reports_which_features_were_considered`, `test_an_empty_explanation_is_reported_rather_than_hidden`, `test_an_explanation_available_but_empty_caps_the_band_to_low` | A prediction no feature moved is reported as such, the omitted features are listed, and the band is held at `LOW` while the probability is left exactly as the model produced it |
| No global-importance substitution | `test_the_unavailability_reason_refuses_the_global_importance_shortcut`, `test_an_artifact_without_reference_values_yields_no_explanation` | An artifact with no reference values yields an explicitly unavailable explanation, and the reason names the substitution being refused |
| One-hot features are not multiplied | `test_each_feature_appears_at_most_once`, `test_a_one_hot_feature_is_named_by_its_feature_and_not_its_category` | One-hot expansion is grouped to a single reportable signal per feature |
| Reference values come from the training split | `test_the_reference_covers_all_columns`, `test_the_reference_values_are_medians`, `test_the_reference_is_persisted_and_reloaded`, `test_a_pre_existing_artifact_with_no_reference_is_loaded` | One median per column, computed on train only, round-tripped, and a pre-Phase-9 sidecar still loads |
| Calibration behaviour | `test_a_perfectly_calibrated_model_scores_near_zero`, `test_an_inverted_model_scores_near_one`, `test_an_overconfident_model_is_penalised_proportionally`, `test_an_unmeasured_report_is_never_believable`, `test_ece_uses_observed_frequency_not_accuracy`, `test_a_perfect_model_on_separated_data_reports_near_zero_ece` | Known-answer cases pin the metric; an unmeasured report is never believable, so absence of measurement produces `UNKNOWN` rather than a zero |
| Malformed input | `test_a_probability_may_not_accompany_a_refusal`, `test_an_assessment_may_not_accompany_a_refusal`, `test_a_negative_verdict_must_explain_itself`, `test_a_resolved_outcome_must_be_banded_in_the_numeric_range`, `test_a_vector_from_another_feature_set_is_refused` | An outcome that contradicts itself is refused at construction; no probability or confidence can be read out of a verdict that declined to issue one |
| Provenance travels with the answer | `test_the_outcome_carries_the_models_data_origin`, `test_the_outcome_carrying_the_probability_also_carries_its_definition` | `data_origin` and `target_definition_version` are carried on every outcome, so a number is never separable from what it is a number about |
| Layer boundaries | `test_no_uncertainty_module_imports_a_later_layer`, `test_uncertainty_modules_import_only_permitted_layers`, `test_the_uncertainty_package_exports_only_known_names` | Statically checked: no module reaches into policy, outcome, feedback, evaluation, or API, and every name in `__all__` is bound |
| Clock injection | `test_two_runs_over_one_vector_agree_on_the_timestamp`, `test_the_timestamp_is_the_vectors_own_reference_time`, `test_two_runs_with_one_clock_agree_on_the_timestamp`, `test_a_different_clock_produces_a_different_timestamp` | The prediction is dated by its evidence and the outcome by its injected clock; the stamp tracks the input, and the clock is read rather than ignored |
| Replay agreement | `test_prediction_and_outcome_come_from_one_instant` | One vector scored and assessed under one clock yields one instant across both layers |
| No live-clock regression | `test_the_default_engine_still_reads_a_real_instant` | The unconfigured default remains a working `SystemClock`, not `None` |
| Tests | `python -m pytest` | 1230 passed, of which 140 under `tests/unit/uncertainty/`, 165 under `tests/unit/models/`, and 9 in `tests/unit/test_clock_determinism.py` |

---

## Phase 10 — Intervention policy `COMPLETE`

Whether to act, with restraints. Policies start transparent and deterministic. No
reinforcement learning is claimed.

**Delivered**
- `policy/models.py` — `PolicyDecision`, `PolicyDecisionType`, `NonActionReason`, `Restraint`,
  `RestraintKind`, `InterventionCandidate`, `ACTIONABLE_STATES`, `CONFIDENCE_RANK`, and
  `meets_confidence`.
- `policy/engine.py` — `InterventionPolicy`, `DEFAULT_CATALOGUE`, `WINDOW`, the candidate
  filter, and the composition point.
- `policy/history.py` — `InterventionHistory`, `DeliveredIntervention`, `OutcomeClass`,
  `classify_outcome`, and `history_from_events`.
- `tests/unit/policy/test_decisions.py` — the policy's own decision logic.
- `tests/unit/policy/test_restraints.py` — cooldown, session cap, sliding-window cap,
  repeated-type limit, repeated-failure protection, and the property test that tightening
  any restraint can only ever remove interventions.
- `tests/unit/policy/test_layer_boundaries.py` — static import guards for the policy layer.

**Exit criteria — met**
| Check | Command | Result |
|---|---|---|
| Tests | `python -m pytest tests/unit/policy` | 87 passed |
| Full suite | `python -m pytest tests` | 1034 passed |
| Lint | `python -m ruff check .` | All checks passed |
| Format | `python -m ruff format --check .` | 94 files already formatted |
| Types | `python -m mypy src tests/unit/policy` | Success, 54 source files |
| Layer boundary | `test_no_policy_module_imports_a_later_layer`, `test_policy_modules_import_only_permitted_layers` | No import of a later layer, including via deferred function-level imports |
| Public surface | `test_the_policy_package_exports_only_known_names`, `test_the_package_exports_its_composition_point_and_its_evidence_type` | Every name in `__all__` is bound, and the two entry points are exported |
| Direction | `test_a_tighter_policy_never_acts_where_a_looser_one_declines` | Over a grid of 400 scenario configurations, tightening any restraint can only remove interventions, never add them |

**Key design decisions**
- **Retirement outranks the repeat limit in blame.** When both apply to the same
  candidate, only the first `continue` in the filter loop was ever recorded, which was
  always the repeat limit. With both defaults at two, two declined recaps were reported
  as "delivered twice in a row" — which is what happened — rather than as the actual
  fact: the learner declined both times and should not be offered a third. The filter now
  collects all applicable reasons per candidate before deciding, so an operator auditing
  a suppressed type sees every reason it was excluded, not merely the first one evaluated.
- **`NO_INTERVENTION` is a first-class outcome, not a null.** A decision that chose not
  to act carries its reason (`IN_COOLDOWN`, `SESSION_CAP_REACHED`, etc.), the restraint
  that bound the decision (including limit and observed), and every other restraint the
  policy considered. A null answer would be unauditable; this one names exactly what
  stopped it.
- **The grid test is the load-bearing one.** Every individual restraint passes its own
  threshold test while the policy as a whole could behave perversely — raising a cap
  could in theory make the policy *more* willing to act, and the bug would hide behind
  green individual tests. The property test asserts the direction over a grid of 8
  histories × 10 outcomes × 5 states × 8 restraint pairs simultaneously, catching the
  class of defect where one restraint's boundary shifts the point at which another fires.
- **The sliding window is half-open.** A delivery an hour old exactly has *aged out*; an
  inclusive boundary would quietly raise the effective hourly rate above the configured
  cap.
- **The cooldown detail reports elapsed and limit, and headroom gives the remainder.**
  The `Restraint` dataclass carries a computed `headroom` property (`limit - observed`,
  direction-agnostic) so a caller need not re-derive the remaining cooldown from the
  detail string, which would be brittle and out of scope for this layer.

**What this does not establish**
The restraints are documented engineering defaults with stated reasoning, not validated
choices. The cooldown of ten minutes and the session cap of three are starting points,
not empirically derived limits. Establishing whether a given configuration produces the
right rate and cadence of intervention for a real learner population is the evaluation
framework's question (Phase 13), not this layer's. No restraint has been tested against
real learner behaviour, and a policy computed from simulator output reflects that
simulator's assumptions and nothing more. The deterministic selection is a deliberate
choice: the policy chooses the next gentlest option in catalogue order, which is an
engineering escalation ladder, not a personalisation. Personalisation belongs to the
feedback architecture (Phase 12) and is excluded here by design.

---

## Phase 11 — Outcome engine `COMPLETE`

Standardised before/after measurement following an intervention, recorded as an **observed
outcome** — never as a causal claim.

**Delivered**
- `outcomes/models.py` — `OUTCOME_V1`, `OUTCOME_MEASURES`, `Measurement`, `MeasurementStatus`,
  `OutcomeDirection`, `OutcomeMeasure`, `OutcomeWindow`, `WindowKind`, and `OutcomeRecord`.
- `outcomes/measures.py` — the seven measures, `direction_of`, and the shared
  activity-event set.
- `outcomes/engine.py` — `OutcomeEngine`, `OutcomeError`, window construction, record
  assembly, `measure`, and `measure_many`.
- `tests/unit/outcomes/scenarios.py` — the shared event builders, synthetic by default.
- `tests/unit/outcomes/test_measures.py` — the rate, share, and latency readings and their
  absence paths.
- `tests/unit/outcomes/test_measures_context.py` — task persistence, session continuation,
  and the trajectory reading.
- `tests/unit/outcomes/test_windows.py` — the window geometry and the record/window
  agreement refusals.
- `tests/unit/outcomes/test_models.py` — the model's own refusals.
- `tests/unit/outcomes/test_layer_boundaries.py` — static import guards.
- `tests/integration/test_outcome_integration.py` — six simulated sessions through the real
  context, feature, baseline, temporal, policy, and outcome layers.

**Exit criteria — met**
| Check | Command | Result |
|---|---|---|
| Tests | `python -m pytest tests/unit/outcomes` | 128 passed |
| Integration | `python -m pytest tests/integration/test_outcome_integration.py` | 14 passed |
| Full suite | `python -m pytest tests` | 1176 passed |
| Lint | `python -m ruff check .` | All checks passed |
| Format | `python -m ruff format --check .` | 106 files already formatted |
| Types | `python -m mypy src tests/unit/policy tests/unit/outcomes` | Success, 65 source files |
| Layer boundary | `test_no_outcome_module_imports_a_later_layer`, `test_outcome_modules_import_only_permitted_layers` | No import of a later layer |
| Public surface | `test_the_outcomes_package_exports_only_known_names` | Every name in `__all__` is bound |
| Window geometry | `test_the_windows_meet_at_exactly_one_instant`, `test_the_delivery_instant_belongs_to_the_after_window_only` | The delivery instant belongs to the after window alone |
| Measurement integrity | `test_a_record_missing_a_measure_is_refused`, `test_a_record_hiding_synthetic_measurements_behind_a_real_origin_is_refused` | No holes, no laundering |

**Key design decisions**
- **A missing event is not a fact about a learner.** Every measure that cannot compare says
  so and returns no direction. A rate of zero computed from no observations is
  indistinguishable from a rate of zero computed from a learner who did nothing, and the
  second is a claim about a person. `INSUFFICIENT_DATA` carries a `reason` naming the
  missing evidence; the model refuses a record where an unmeasured reading has none, so
  "no change" can never be mistaken for a failure to look.
- **The measure set is frozen and the settings are fingerprinted.** `OUTCOME_V1` and the
  seven measures are versioned together, and a record carries a fingerprint of the window
  geometry that produced it. A record cannot later be read as though it had been measured
  under different widths, and `test_a_different_window_configuration_changes_the_fingerprint`
  checks the derivation rather than the presence of the field.
- **The temporal verdict is read, not recomputed.** `SUBSEQUENT_TRAJECTORY` is a reader of
  the temporal layer's direction. The outcome layer does no temporal reasoning of its own,
  which is why the integration test drives the real `TemporalEngine` rather than a
  hand-built state — a trajectory rebuilt from raw deviations on this side would be a second
  opinion wearing a reader's clothes.
- **The prompt is not evidence that the learner resumed.** `ACTIVITY_EVENTS` excludes the
  two intervention events, and also `INACTIVITY_STARTED`, which evidences an *absence* of
  interaction. Counting either would move a measure in the direction its own name
  contradicts, and counting the system clicking on the learner's behalf is the one
  fabrication this layer exists to prevent.
- **A measured direction and a causal claim are different fields.** A reading may be
  `MEASURED` with a direction *and* a reason stating that the events do not show the prompt
  is why — the session-continuation case, where the session ended inside the after window.
  The record therefore supplies the direction a consumer needs and withholds the inference
  the evidence cannot support, in the same object, with neither one overwriting the other.
- **`PROVENANCE` is constrained to `OBSERVED`.** The vocabulary contains `PROXY_LABEL` and
  `CAUSAL_CLAIM`, and the type refuses both. Stating the vocabulary and then refusing the
  other members is the only form of the constraint that cannot be bypassed by setting a
  field.

**What this does not establish**
Every reading here is observational, and none of the window widths, the sample floor, or the
change tolerance is an empirically validated choice. They are engineering starting points
with stated reasoning; establishing which geometry produces honest measurements for a real
learner population is the evaluation framework's question (Phase 13), and the core
framework delivered there does not settle it — it can score these records but every corpus
available to it is synthetic. The layer measures
what happened either side of a prompt and does not attribute it, so a measured improvement
is not evidence that the prompt caused one — attributing it, and accumulating that
attribution across interventions, is Phase 12's work and is excluded here by design.
`tail_minutes` is validated but consumed by no measure in `OUTCOME_V1`; its field docstring
says so, rather than leaving a reader to infer it from a measurement that ignored it.
A record computed from simulator output reflects that simulator's assumptions and
establishes nothing about any real learner.

---

## Phase 12 — Feedback architecture `COMPLETE`

Accumulates observed intervention response per learner, intervention type, and context, then
allows response profiles to adjust candidate order only after a minimum evidence threshold.

**Delivered**
- `feedback/models.py` — `FEEDBACK_V1`, immutable intervention response observations,
  response profile synthesis, maturity states, and strict rate/count consistency validation.
- `feedback/engine.py` — `FeedbackEngine`, outcome-to-observation conversion, profile
  accumulation, minimum-observation gating, and active-profile candidate reordering.
- `feedback/__init__.py` — public package surface.
- `schemas/versioning.py` — `FeedbackVersion` added to the shared version vocabulary.
- `tests/unit/feedback/` — profile invariants, accumulation, gating, no-retraining, and
  static import boundary coverage.
- `tests/integration/test_feedback_integration.py` — outcome record integration coverage.

**Exit criteria — met**
- Response profiles are keyed by learner, intervention type, and context.
- Below `min_response_observations`, observations are retained but profiles remain
  `COLLECTING`, expose no effective success rate, and do not alter candidate order.
- `FeedbackEngine.retrain_model_from_feedback()` structurally raises `NotImplementedError`;
  production retraining remains an offline, controlled, versioned, evaluated, and approved
  workflow outside this layer.
- Unit and integration tests cover profile accumulation, maturity transition, gating, derived
  outcome observations, static boundaries, and prohibited implicit retraining.

**What this does not establish**
Response profiles summarize observed post-delivery behaviour; they do not establish that an
intervention caused a change. The current corpus is synthetic, so active profile rates describe
the simulator's generative assumptions rather than real learner behaviour. Feedback can adjust
candidate order only; it cannot select an intervention, bypass policy or authority safeguards,
or update a model artifact.

---

## Phase 13 — Evaluation framework `PARTIAL — core framework COMPLETE, advanced metrics PLANNED`

Scoring recorded predictions against the evidence that actually followed them, under a
window that cannot admit the prediction's own input. The core framework is built and gated.
The split construction and advanced ranking/calibration metrics named in the original
specification are **not** implemented and remain future work; see *Outstanding* below, which
is not a list of this phase's exit criteria.

**Delivered**
- `evaluation/models.py` — `EvaluationVersion`, `EVALUATION_V1`, `PredictionTarget`,
  `PredictionHorizon`, `ObservationWindow`, `PredictionSnapshot`, `GroundTruthStatus`,
  `GroundTruth`, `EvaluationVerdict`, `BinaryVerdict`, `EvaluationRecord`, and
  `InterventionResponse`.
- `evaluation/engine.py` — `EvaluationEngine`, `DEFAULT_EVALUATION_VERSION`, the
  half-open window filter, reading refusals, `ground_truth`, `evaluate`, `evaluate_many`.
- `evaluation/aggregate.py` — `ConfusionCounts`, `EvaluationSummary`, `summarise`,
  `group_by`, `group_by_learner`, `group_by_model_version`, `group_by_target`,
  `group_by_intervention_response`.
- `evaluation/__init__.py` — the package's public surface.
- `tests/unit/evaluation/scenarios.py` — shared builders, synthetic by default.
- `tests/unit/evaluation/test_models.py` — the models' own refusals.
- `tests/integration/test_evaluation_integration.py` — the full chain, boundary, and
  real-state-seam coverage described below.

**Delivered behaviour**
- **Immutable historical prediction snapshots.** `PredictionSnapshot` is frozen and
  carries `model_id`, `model_version`, `feature_set_version`, and the upstream uncertainty
  `verdict`. It is a wrapper around the prediction and cannot revise the claim inside it. A
  validator requires exactly one of `predicted_state`, `predicted_direction`, and
  `predicted_positive`, so a snapshot holding two claims — or none — cannot be built.
- **Deterministic prediction horizons.** `PredictionHorizon` fixes `predicted_at` and
  `duration_seconds` and derives its `ObservationWindow`; the same snapshot always
  describes the same interval.
- **Half-open observation windows.** Admissible evidence is
  `predicted_at <= timestamp < predicted_at + duration`, for the same learner, in the same
  session, restricted to activity events. An event at exactly the prediction instant is
  evidence about what followed; an event at exactly `predicted_at + duration` belongs to
  the next prediction. The start-inclusive/end-exclusive asymmetry is a validated setting
  (`require_horizon_inclusive_start`) and cannot be configured off.
- **Post-prediction observed evidence as the evaluation source.** Ground truth is derived
  only from events inside that window. A reading anchored before the prediction is refused
  by name, because a state computed before the prediction is the prediction's own input and
  scoring against it would grade the model on the evidence it was given.
- **`Provenance.OBSERVED`, and no fabricated ground truth.** `GroundTruth.provenance` is
  constrained to `OBSERVED` unconditionally. The vocabulary contains `GROUND_TRUTH` and the
  type refuses it: events following a prediction are not an external source of truth, and a
  record stamped with it would claim an independence this repository cannot supply.
- **Four-valued ground-truth classification.** `CONFIRMED`, `REFUTED`, `INSUFFICIENT_DATA`,
  `NOT_ASSESSABLE`. The first two are findings; the last two are not, and they are never
  conflated — too little activity in the window is a different problem from activity that
  was never turned into a characterisation.
- **Binary DECLINE scoring.** `BinaryVerdict` carries `TRUE_POSITIVE`, `FALSE_POSITIVE`,
  `TRUE_NEGATIVE`, `FALSE_NEGATIVE`, and `NOT_ASSESSABLE`. The positive class is
  `ACTIONABLE_STATES`, read from the policy layer rather than redefined here.
- **Multi-state STATE/DIRECTION evaluation without a binary matrix.** A `STATE` or
  `DIRECTION` prediction is scored against the observed state or direction and can be
  `CORRECT` or `INCORRECT`, but has no positive class and so produces no confusion cell. The
  summary reports its `assessed` and `not_assessable` counts instead; `assessed` is
  `confirmed + refuted`, so a measured five-way claim counts as assessed while still being
  outside the matrix.
- **Deterministic aggregate metrics.** `precision`, `recall`, `f1`, `accuracy`,
  `false_positive_rate`, `false_negative_rate`, and `coverage` are computed from counts.
  A summary validator refuses a set of counts that does not account for the whole corpus.
- **Zero-denominator metrics are `None`, not zero.** An undefined rate returns `None`, and
  `f1` is `None` when either input is. A metric that could not be computed is never
  reported as a number.
- **Learner and group aggregation without ranking.** `group_by_learner`,
  `group_by_model_version`, `group_by_target`, and `group_by_intervention_response` return
  per-group summaries with keys **sorted** rather than in first-seen or performance order,
  so no reader can mistake the iteration order for a leaderboard, and each group reports
  its own record count and distinct-learner count so a comparison is not made blind to
  sample size.
- **Prediction, model, and version provenance preserved.** Each `EvaluationRecord` carries
  `model_id`, `model_version`, `evaluation_version`, `feature_set_version`, and the target.
  The model version and the evaluator version are separate fields because they answer
  different questions, and collapsing them would make a rule change indistinguishable from a
  model change.
- **Evaluation provenance.** Every ground truth and record carries `data_origin` and
  `provenance`, so a slice computed from simulator output is labelled synthetic at every
  level and cannot be read as a measurement of a person.
- **Observational intervention-response classification only.** `response_from_outcome`
  derives an `InterventionResponse` from an upstream `OutcomeRecord` and only when the
  caller supplies one. Evaluation does not select interventions, does not re-measure a
  delivery, and does not infer a counterfactual: an improvement is recorded as an
  improvement *observed after* the delivery, and nothing in the layer says the delivery
  produced it.
- **Strict temporal leakage prevention.** The window filter, the same-learner and
  same-session restrictions, the refusal of a pre-prediction reading, and the refusal of a
  snapshot's own provenance as evidence are enforced in code and covered by tests.
- **Deterministic behaviour.** The engine holds no mutable state, takes an injectable
  `Clock`, and holds no randomness, network access, or current-time dependency. The same
  inputs produce equal records and byte-identical serialisations.
- **Integration and boundary coverage.** `tests/integration/test_evaluation_integration.py`
  covers the full chain (prediction → horizon → evidence filter → ground truth → record →
  aggregate), the four half-open boundary cases, learner and session isolation, the four
  DECLINE confusion cells, both directions of the `INSUFFICIENT_DATA`/`NOT_ASSESSABLE`
  distinction, frozen snapshot metadata, `GROUND_TRUTH` refusal, determinism under a
  `FixedClock`, and a seam test that scores a genuine `TemporalState` produced by the real
  `TemporalEngine`.

**Architectural position**
The dependency direction is unchanged and preserved:

```
events → features → baseline → temporal state → prediction → policy
       → intervention → outcomes → evaluation
```

Evaluation is the terminal layer and is downstream of everything it scores. No module
outside `evaluation/` imports it, and `policy/`, `models/`, and `uncertainty/` do not
reach for it. Evaluation's own imports are the layers above it plus the supporting layers.
The single reuse worth naming is `ACTIONABLE_STATES` from `policy/models.py`: the positive
class is read from the policy layer rather than reminted, because a layer with its own idea
of "bad" would drift from the one every other layer reads.

**Exit criteria — met (core framework)**
| Check | Command | Result |
|---|---|---|
| Tests | `python -m pytest tests` | 1230 passed |
| Integration | `python -m pytest tests/integration` | 50 passed |
| Evaluation integration | `python -m pytest tests/integration/test_evaluation_integration.py` | 23 passed |
| Evaluation unit | `python -m pytest tests/unit/evaluation` | 14 passed |
| Configuration | `python -m pytest tests/unit/test_configuration_thresholds.py` | 71 passed |
| Lint | `python -m ruff check .` | All checks passed |
| Format | `python -m ruff format --check .` | 115 files already formatted |
| Types | `python -m mypy src` | Success, no issues in 57 source files |
| Window geometry | `test_event_at_prediction_instant_included`, `test_event_at_horizon_end_excluded`, `test_event_one_microsecond_before_end_included`, `test_event_one_microsecond_after_end_excluded` | The start is inclusive and the end exclusive at actual timestamp precision |
| Isolation | `test_learner_isolation_ignores_other_learner`, `test_session_isolation` | Another learner's or session's activity cannot reach the record |
| Non-findings stay distinct | `test_insufficient_data_distinct_from_not_assessable` | Thin evidence stays `INSUFFICIENT_DATA` even with a reading; ample evidence without one is `NOT_ASSESSABLE` |
| Refused readings | `test_contradictory_reading_refused`, `test_reading_anchored_before_prediction_refused`, `test_insufficient_data_reading_refused` | A misdirected, pre-prediction, or self-declared-insufficient reading cannot score a prediction |
| Matrix integrity | `test_all_four_cells_aggregate_correctly` | TP, FP, TN, FN each occur once and aggregate to the expected counts and rates |
| Metadata immutability | `test_historical_prediction_metadata_cannot_be_rewritten` | Recorded prediction attributes cannot be edited after the fact |
| No fabricated ground truth | `test_ground_truth_provenance_cannot_be_asserted`, `test_engine_never_emits_ground_truth_provenance` | `GROUND_TRUTH` is unconstructable, and the engine never emits it |
| Determinism | `test_same_inputs_produce_identical_records`, `test_evaluate_many_is_deterministic` | Repeated runs agree field for field and byte for byte |
| Evidence floor configured | `test_min_evidence_floor_is_respected`, `test_zero_evidence_floor_is_rejected`, `test_exclusive_window_start_may_not_be_configured` | The floor is honoured, zero is refused, and the inclusive start is not switchable off |
| Real-state seam | `test_evaluator_accepts_a_genuine_temporal_state` | A state the real `TemporalEngine` produced is scored without any stand-in |

**Key design decisions**
- **The window is half-open, and the asymmetry is load-bearing.** The prediction instant is
  inside the window and the horizon boundary is outside it. An event at exactly
  `predicted_at` is the first observation *after* the prediction; one at exactly
  `predicted_at + duration` belongs to the next prediction. Flipping either edge would move
  events across the prediction boundary, so the rule is a validated setting named
  explicitly rather than a literal in an expression.
- **Missing evidence cannot become a finding.** `INSUFFICIENT_DATA` and `NOT_ASSESSABLE`
  are separate statuses, both of which leave every `observed_*` field `None`, and the
  verdict is `NOT_ASSESSABLE` rather than a failure. A gap in a log is not a mistake by the
  model, and a summary that subtracted only one of the two non-findings would misstate how
  much of the corpus was actually measured.
- **A multi-state claim is measured without being forced into a binary cell.** A
  five-way `STATE` prediction has no positive class. Collapsing it to a confusion matrix
  would invent one; reporting its `confirmed`/`refuted` counts and leaving the matrix
  `NOT_ASSESSABLE` keeps "measured" and "placed in a cell" as different claims.
- **A rate that cannot be computed is `None`.** `_rate` returns `None` on a zero
  denominator, so a metric is never a fabricated `0` or a flattering `0.5`. This is the
  same discipline the model layer applies to rank metrics on a single-class holdout.
- **Groups are descriptions, not rankings.** Group keys are sorted, per-group record and
  learner counts are reported, and `group_by_model_version` groups on the *version* rather
  than the model id, so pooling two artifacts of one approach cannot make a regression look
  like a stable average.
- **Response is read, not recomputed.** The intervention response is derived from the
  outcome layer's own record rather than re-measured here. Evaluation is a reader of
  Phase 11's measurement, which is why a delivery that improved is filed as an improvement
  observed after the delivery with no counterfactual attached.
- **Both origins are computed, not asserted.** `data_origin` is derived from the
  admissible evidence, so a record over mixed real and synthetic events is synthetic. A
  slice is never laundered into looking like a real measurement.

**What this does not establish**
Nothing here is a claim about predictive performance, and no number produced by this layer
is a measurement of a real learner. Every corpus in the repository is simulator output, so a
`precision` of 0.5 computed over it describes the simulator's archetypes and the evaluator's
arithmetic, not the behaviour of any person. The layer establishes that a missing event
cannot become a finding about a learner, that the window geometry cannot admit a
prediction's own input, and that an undefined rate is reported as undefined — not that any
prediction was right.

No metric here is evidence that an intervention caused a behavioural change. Evaluation
classifies what was *observed after* a delivery; attributing it is Phase 12's work, and is
excluded by design. `min_evidence_for_evaluation` (default four) and the
`require_horizon_inclusive_start` rule are engineering choices with stated reasoning, not
empirically validated ones, for the same reason the outcome layer's window widths are not.

**Outstanding — future work, not this phase's exit criteria**
The original Phase 13 specification was broader than the framework that was built. These
items are **not implemented** and are recorded here so the roadmap is not read as claiming
them:
- **Split construction.** `EvaluationSettings` validates `temporal_split_quantile`,
  `min_learners_per_split`, and `min_positive_rate`, and no code constructs a split. The
  temporal-holdout split itself, and any guard against learner overlap between a fitting
  and a scoring set, is future work. `ModelMetrics.split_name` in the model layer is
  caller-supplied for the same reason: the layer that would produce the split does not
  exist yet.
- **Leakage test matrix.** Future leakage within the window, post-intervention leakage,
  duplicated sessions, target leakage, and feature contamination have partial coverage —
  the window filter, learner/session isolation, and pre-prediction reading refusal are
  tested — but there is no dedicated cross-layer leakage suite.
- **Slice reporting beyond the four group keys.** Cold-start, noisy-behaviour, and
  class-imbalance slices are not implemented. `group_by_learner`, `group_by_model_version`,
  `group_by_target`, and `group_by_intervention_response` are the only groupings.
- **Personalization comparison.** Population-only versus population + personal baseline
  is not implemented. `group_by_model_version` cannot stand in for it, because it compares
  artifacts, not configurations.
- **Ranking and calibration metrics in this layer.** ROC-AUC, PR-AUC, Brier, and ECE are
  **not** computed by the evaluation framework. They exist in the Phase 8 model layer
  (`models/trainers.py`, `schemas/versioning.ModelMetrics`) as holdout metrics for one
  model on one split. Wiring them into a prediction corpus scored across learners and
  versions is future work.
- **Independent prediction-quality and intervention-quality evaluation.** A record may
  carry both a verdict and an observational response, but nothing here compares the two
  populations against each other.

---

## Phase 14 — API `NOT STARTED`

A FastAPI boundary over the **real** engine (`api/lab/` is *not* this: it is a
development-only, deterministic, unauthenticated lab surface whose delivery boundary is
dev-only and sends nothing).

**Planned surface:** `POST /events`, `POST /predict`, `POST /interventions/decide`,
`POST /interventions/{id}/outcome`, `GET /health`.

**Exit criteria**
- Endpoints call the actual core engine.
- Errors are typed and meaningful; malformed payloads rejected.
- Tests: validation, errors, health, full prediction flow.

---

## Phase 15 — Security / privacy hardening & Student Intelligence `COMPLETE`

**Done in Phase 1 (Security Foundations)**
- Schema validation at every boundary; secrets only from environment variables.
- No credential has a default; secrets excluded from reprs, logs, and serialized output.
- Fail-closed production gate: refuses to start without a secret or with auth disabled.
- Prototype in-process rate limiting configured; wildcard CORS origin rejected.
- Pseudonymous identifiers enforced by type; no PII representable.
- Redacting log filter, idempotent, never dropping records.

**Delivered in Deep Student Intelligence & Evidence Reasoning**
- **15 Question Intents:** Grounded query classification and answer generation (`api/lab/intelligence/query.py`) supporting `profile`, `explain`, `explanation`, `missing`, `gaps`, `autonomy`, `authority`, `current_state`, `change`, `comparison`, `subject`, `intervention`, `outcome`, `timeline`, and `evidence`.
- **RBAC Before Retrieval:** Rigid scoping across `Admin`, `Teacher`, and `Student` roles. Student role is restricted to self-lookup, and withheld fields surface as `PRIVILEGE_RESTRICTED` gap records rather than leaking or silently dropping.
- **Deep-Evidence Reasoning:** Deterministic reconstruction of student interaction history (`api/lab/intelligence/reason.py`) supporting subject roll-ups, temporal change-state classification (`declining`, `improving`, `recovering`, `stable`, `volatile`, `insufficient_evidence`), and cross-signal fusion (`consistent`, `divergent`, `insufficient`) without LLM hallucination.
- **Store Separation:** Dedicated demo store (`api/lab/intelligence/builder.py`) with 4 multi-subject test learners (`intel-full-0001`, `intel-thin-0001`, `intel-conflict-0001`, `intel-recovery-0001`), maintaining 100% byte-invariance on canonical 5-scenario learner profiles.
- **API & UI Integration:** Added `/api/intelligence-demo` REST endpoints and extended Lab UI (`api/lab/static/index.html`) with dual-store support, quick query actions, and freeform question interaction.
- **Verification:** 1363 passing tests, full ruff/mypy typecheck compliance, deterministic replay validation.

**Outstanding (Production API Scope — Phase 14)**
- Production OAuth/JWT authentication middleware.
- Edge proxy rate limiting enforcement.
- Immutable write-ahead audit trail across cluster nodes.
- Automated data retention lifecycle policies.

---

## Phase 16 — Documentation + technical inventory `PARTIAL`

**Done in Phase 1:** `README.md`, `PROJECT_AUDIT.md`, `ARCHITECTURE.md`, `ROADMAP.md`,
`RESEARCH_PRINCIPLES.md`.

**Outstanding:** `DATA_DICTIONARY.md`, `MODEL_CARD.md`, `PRIVACY.md`,
`TECHNICAL_INVENTORY.md`, `EXPERIMENTS.md`, `data/README.md`, `FOUNDATION_STATUS.md`.

Documentation must describe the actual implementation. Planned features are never
documented as implemented.

---

## Deferred indefinitely

Not planned work. Each becomes a legitimate question only when the evidence exists.

| Item | Condition for reconsideration |
|---|---|
| Deep learning (LSTM / Transformer) | Demonstrated that remaining error is genuinely sequential |
| Reinforcement learning for policy | Demonstrated that a deterministic policy is insufficient |
| LLM-in-the-loop components | A specific, justified task that cannot be met deterministically |
| XGBoost / LightGBM | scikit-learn gradient boosting shown to be capacity-limited |
| MLflow | Multi-user experiment tracking becomes a real requirement |
| Dedicated time-series libraries | The explicit state-transition logic proves insufficient |
| Database vendor | Deployment requires concurrent writes at scale |
| Any cloud or hosted-LLM dependency | Never, for this project's portability requirement |
| EduFlow integration | Explicitly out of scope for this phase |
