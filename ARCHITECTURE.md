# ARCHITECTURE

**Status of this document:** describes the architecture as it is actually built. Anything
not yet implemented is marked `PLANNED — NOT IMPLEMENTED`. Planned features are never
described here as though they exist.

---

## 1. Design stance

The engine is a pipeline of independent layers. Each layer owns one question, exposes a
narrow typed interface, and cannot reach sideways into its neighbours' internals.

The architectural commitment is not any particular algorithm. Algorithms will be
replaced as evidence accumulates. The commitment is to the **separations** below,
because those are what keep the system's claims honest as its models change.

### 1.1 The three separations that must never be collapsed

**Separation 1 — Prediction ≠ Decision ≠ Intervention.**

| Question | Owner | Output |
|---|---|---|
| What is the estimated behavioural state? | Prediction engine | a calibrated probability plus a state |
| Is intervention justified? | Intervention policy | a decision, with reasons |
| What is delivered? | Intervention engine | a concrete intervention record |

These are three interfaces with three audit records. Collapsing them makes it impossible
to tell, after the fact, whether a bad outcome came from a bad prediction, a bad policy,
or a bad delivery — and it makes it impossible to evaluate intervention quality
independently of prediction quality.

**Separation 2 — Population inference ≠ Personalized inference.**

Every prediction carries an `InferenceBasis` and a `BaselineMaturity`. A cold-start
population estimate can never be silently presented as a personal one. This is
structural, not documentary: the fields travel with the value.

**Separation 3 — Engineering complete ≠ Scientifically validated.**

A passing test suite means the machinery runs correctly. It is not evidence that the
machinery is right about human behaviour. These are tracked as separate statuses
throughout; see `FOUNDATION_STATUS.md`.

---

## 2. Layer pipeline

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
                 BASELINE ML MODELS
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
```

| # | Layer | Owns | Status |
|---|---|---|---|
| 0 | Event intelligence | typed, validated behavioural event vocabulary | **IMPLEMENTED** |
| 1 | Context engine | what situation a signal occurred in | **IMPLEMENTED** |
| 2 | Feature engine | events → measurable features, with a versioned registry | **IMPLEMENTED** |
| 3 | Personal baseline engine | the learner's own normal behaviour, with maturity | **IMPLEMENTED** |
| 4 | Temporal state engine | state, direction, persistence, trajectory | **IMPLEMENTED** |
| 5 | Baseline ML models | fitting, serialising, scoring, and tracking the lifecycle of a model | **IMPLEMENTED** |
| 6 | Prediction engine | calibrated probability of a labelled behavioural state | **IMPLEMENTED** (`models/predictors.py`) |
| 7 | Uncertainty engine | whether the evidence supports any answer at all | **IMPLEMENTED** (`uncertainty/`) |
| 8 | Intervention policy | whether to act, subject to restraints | **IMPLEMENTED** (`policy/`) |
| 9 | Intervention engine | what to deliver | `PLANNED — NOT IMPLEMENTED` (dev-only delivery boundary in `api/lab/execution.py` exists under `DEV_ONLY`; production delivery is not implemented) |
| 10 | Outcome engine | standardised before/after measurement | **IMPLEMENTED** (`outcomes/`) |
| 11 | Feedback architecture | accumulating observed response per learner | `PLANNED — NOT IMPLEMENTED` |
| 12 | Model improvement | controlled, versioned, approved retraining | `PLANNED — NOT IMPLEMENTED` |
| 13 | Evaluation framework | scoring recorded predictions against observed evidence | **PARTIAL** (`evaluation/` — core framework implemented; split construction and advanced metrics planned) |

### Supporting layers

| Component | Purpose | Status |
|---|---|---|
| `schemas/` | shared vocabulary: identifiers, timestamps, provenance, confidence, versions | **IMPLEMENTED** |
| `configuration/` | every threshold with a stated rationale; paths, runtime, secrets | **IMPLEMENTED** |
| `utils/` | injectable clock, seed derivation, content hashing, redacting logging | **IMPLEMENTED** |
| Model registry | append-only model records with lifecycle status | **IMPLEMENTED** (`models/registry.py`, Phase 8) |
| Evaluation framework | leakage-safe prediction scoring, four-valued ground truth, confusion matrix, group aggregation | **PARTIAL** (`evaluation/`, Phase 13 — core implemented; split construction, advanced metrics, personalization comparison planned) |
| Synthetic simulator | configurable learner archetypes emitting event sequences | **IMPLEMENTED** |
| Audit trail | every decision traceable to model, features, data, and outcome | `PLANNED — NOT IMPLEMENTED` |
| API boundary | FastAPI wrapper over the real engine | `PLANNED — NOT IMPLEMENTED` (development-only lab with a Full-API surface exists at `api/lab/`; it is not the production boundary) |

---

## 3. Dependency rules

1. **Data flows one way.** A layer may import from layers above it in the pipeline and
   from the supporting layers. It may not import from a layer below it.
2. **No sideways reach.** A layer talks to its neighbour through that neighbour's public
   interface, never by reaching into its module internals.
3. **Supporting layers import nothing from intelligence layers.** `schemas`,
   `configuration`, and `utils` sit underneath everything. If any of them needed to know
   what a prediction is, the dependency direction would be wrong.
4. **The package root imports no layer.** `import focus_engine` must stay cheap and
   side-effect free, so no layer can be used without declaring its own dependencies.
5. **No layer redefines another's vocabulary.** Identifiers, timestamps, provenance, and
   version types are declared once in `schemas` and imported everywhere.
6. **Evaluation is terminal.** The pipeline direction is
   `events → features → baseline → temporal state → prediction → policy → intervention →
   outcomes → evaluation`. No module outside `evaluation/` imports it. The layer reads
   upstream records and never writes back, never re-measures them, and never feeds a
   verdict into the prediction, policy, or intervention path that produced the record. A
   layer that could be corrected by its own evaluation would be able to grade its homework.

A violation of any of these is an architecture defect, not a style preference. The
rationale is testability and auditability: if the feature engine can reach into the
baseline engine, then a prediction can no longer be explained from a recorded feature
vector.

---

## 4. What is implemented now (Phases 1–13)

### 4.1 `schemas` — the shared vocabulary

| Module | Contents |
|---|---|
| `schemas/primitives.py` | pseudonymous identifier types, timezone-aware `Timestamp`, unit-interval `Probability`, `DataOrigin`, `Provenance`, `ConfidenceLevel`, `InferenceBasis`, `BehavioralEngagementState`, `SyntheticDataStamp` |
| `schemas/versioning.py` | `VersionStamp`, `ReproductionStamp`, `ModelMetrics`, `ModelRecord`, `ModelStatus`, `ExperimentRecord`, `ExperimentConclusion` |

Three invariants are encoded in types rather than in prose:

- **A directly identifying value is not representable.** The identifier character class
  excludes `@`, `.`, `/`, and whitespace, and enforces a minimum length, so an email
  address or a filesystem path cannot be supplied as a learner id.
- **A naive timestamp is not representable.** Naive datetimes are rejected and aware
  values are normalised to UTC.
- **A synthetic-only model cannot be marked validated or production.** `ModelRecord`
  raises on that combination. This is the guard against the single most damaging false
  claim the system could make.

### 4.2 `configuration` — thresholds with stated derivations

Seven frozen settings objects hold every number that can influence a feature, a
prediction, an uncertainty verdict, or an intervention decision: `FeatureSettings`,
`BaselineSettings`, `TemporalSettings`, `UncertaintySettings`, `InterventionSettings`,
`EvaluationSettings`, `SimulationSettings`.

Each field's docstring states why the initial value was chosen and what evidence would
justify changing it. **These are engineering starting points, not empirical findings.**
No threshold in this repository has been validated against data.

`settings.py` adds filesystem paths derived from a single configurable root, a runtime
mode, and secret sourcing that reads only from the environment and is never included in
a repr, a log record, or a serialised artifact.

### 4.3 `utils` — infrastructure guarantees

- `clock.py` — `Clock` protocol with `SystemClock` and `FixedClock`. Every layer that
  stamps an inference timestamp takes one: the context, temporal, uncertainty, policy,
  outcome, evaluation, and registry engines, plus the model registry. State transitions,
  cooldowns, and window boundaries are therefore testable without sleeping and without
  flakiness, and a recorded decision can be replayed to the same instant. The two exceptions
  are deliberate and documented where they occur: a prediction is dated by its input vector
  rather than a clock (§4.10), and the artifact/training write-side stamps record when a
  record was written rather than inferring from evidence.
- `determinism.py` — seed derivation by label path, so adding a component does not shift
  an existing component's random stream; BLAKE2b content hashing for artifact identity.
- `logging.py` — a `RedactingFilter` attached at the handler, so a future contributor
  cannot leak a credential by forgetting to sanitise. Redaction is idempotent and never
  drops a record.

### 4.4 `events` — the validated boundary

| Module | Contents |
|---|---|
| `events/types.py` | `EventType` (18 closed values), one frozen payload model per type, `EventEnvelope` |
| `events/validation.py` | `validate_event`, `IngestError`, `EventIngestor`, `IngestResult` |
| `events/store.py` | `EventStore` protocol, `InMemoryEventStore`, `JsonlEventStore`, `EventQuery`, `SessionBounds` |

Four invariants are encoded in types rather than in prose:

- **A payload cannot contradict its event type.** `EventEnvelope` maps every
  `event_type` to exactly one payload class and raises on a mismatch, so a downstream
  layer never receives a `question_answered` event carrying a video payload.
- **An unrecognised event type does not pass.** The vocabulary is closed. A typo is an
  ingest error, not a silently ignored record, which keeps every downstream switch
  exhaustive.
- **A synthetic event is never an observation.** `origin=SYNTHETIC` requires a
  `synthetic_stamp` and is rejected when `provenance=OBSERVED`. The stamp travels with
  the record, so a synthetic event cannot be read as real after the fact.
- **Stored history cannot be rewritten.** Both stores expose `append` and `extend` only.
  There is no update or delete method to call. A wrong event is corrected by a new event,
  not by editing the log.

### 4.5 Storage, as built

`JsonlEventStore` is the file-backed append-only store: one event per line, UTF-8, opened
per write so data is flushed before `append` returns. Corrupt lines are skipped and
counted rather than raising, because one bad line should not make a session unreadable —
but the count is retained in `skipped_lines` so the corruption stays visible instead of
being swallowed.

No database vendor is assumed, and none is selected. `InMemoryEventStore` satisfies the
same protocol, and both are verified to behave identically for the operations the
protocol defines, so swapping the backing store cannot silently change semantics.

### 4.6 `context` — conditions, not conclusions

| Module | Contents |
|---|---|
| `context/models.py` | `ContextModel`, `SessionContext`, `ContentContext`, `PerformanceContext`, `InterventionContext`, `ContextKey`, `ContextVersion`, `BLOCKING_CONTEXT_KEYS` |
| `context/engine.py` | `ContextEngine` |

The context layer answers one question: under what conditions did this happen? It never
answers "what state is the learner in?". Three properties make that boundary real rather
than documentary:

- **The event stream is the only input.** No content database, no profile store, no
  side-channel. Given a session's events, the context is fully determined.
- **Replay reproduces the live result.** Elapsed times are measured against the last
  observed event's timestamp, not the wall clock. A recorded session replays to the
  context it produced when it was recorded. The injected clock is a fallback for the
  empty case only.
- **Gaps are named, and derived rather than supplied.** `missing` is computed by a model
  validator, so a caller cannot declare a gap closed while the value is absent.
  `BLOCKING_CONTEXT_KEYS` separates gaps that make interpretation unsound from gaps
  that merely reduce precision, and `is_interpretable()` is the mechanical form of the
  rule that a latency signal is never read without context.

This is where Separation 2 is enforced in code. `baseline_maturity` is a blocking key,
so no layer can produce a behavioural reading from a context that does not say whether it
rests on a population prior or an established personal baseline.

### 4.7 `features` — named, versioned, and unable to invent data

| Module | Contents |
|---|---|
| `features/models.py` | `FeatureName`, `FeatureCategory`, `FeatureValueType`, `FeatureAvailability`, `FeatureSpec`, `TypedFeatureValue`, `FeatureValue`, `INSUFFICIENT_DATA_MARKER`, `FEATURE_SET_V1` |
| `features/engine.py` | `FeatureRegistry`, `FEATURE_SPECS_V1`, `FeatureEngine`, `UnsupportedFeatureSetError`, `compute_features` |

Fourteen features across five categories. The layer turns a recorded interaction into a
number, and four properties keep that number honest:

- **A missing value is never replaced.** No plausible constant stands in for absent
  difficulty, absent latency, or an absent identifier, and an out-of-range result is
  reported as a defect instead of being clamped into range. Every gap yields
  `INSUFFICIENT_DATA` with a reason naming what was absent.
- **The marker cannot be misread.** `TypedFeatureValue` rejects a value whose availability
  and marker disagree, and rejects a computed value carrying an insufficiency reason. A
  marker string consumed downstream as a number is the failure this forecloses.
- **The leakage invariant is enforced.** Events later than `context.reference_time` are
  dropped before any window is computed, so "no future information" holds even when a
  caller supplies a longer stream than the reference time permits.
- **Versions are append-only.** `FeatureRegistry` refuses to re-register a version, and
  `UnsupportedFeatureSetError` is raised at engine construction rather than resolved into
  an empty vector. The definitions behind a version are what make an already-computed
  vector interpretable, so they cannot be edited after the fact.

Provenance is carried as a set of origins, not a flag: a stream mixing real and synthetic
events is labelled `mixed` rather than being forced to `real`, and a vector whose origins
are purely synthetic cannot report any feature as `AVAILABLE`.

Both state features are now measurable. `baseline_maturity_encoded` reports a real value as
soon as a context carries a maturity, which Phase 6 made possible, and `trajectory_encoded`
reports one as soon as a context carries a trajectory, which Phase 7 made possible. Both
still return `INSUFFICIENT_DATA` when the value is genuinely absent — publishing `unknown`
because no engine has produced one would present a placeholder as a measurement.

---

### 4.8 `baseline` — what normal looks like for this learner

| Module | Contents |
|---|---|
| `baseline/models.py` | `BaselineMaturity`, `BaselineDimension`, `StatisticsMethod`, `ReferenceSource`, `PopulationPrior`, `PopulationPriorSet`, `DimensionStatistics`, `BaselineProfile`, `BaselineReference`, `DeviationResult`, `basis_for`, `maturity_rank`, `dimension_range`, `MAD_NORMAL_SCALE`, `PERSONAL_BASELINE_V1` |
| `baseline/statistics.py` | `maturity_for`, `decimate`, `robust_dispersion`, `sample_dispersion` |
| `baseline/engine.py` | `DIMENSION_SOURCE_FEATURES`, `BaselineEngine`, `BaselineUpdateError` |

Four dimensions — accuracy, response seconds, session seconds, content difficulty — each fed
by exactly one source feature declared in `DIMENSION_SOURCE_FEATURES`. The layer answers
two questions: what is this learner's normal value, and how far does a measurement sit from
it. Six properties keep the answer from being a plausible guess:

- **The basis travels with the value.** `BaselineReference` and `DeviationResult` both carry
  a `BaselineMaturity` and an `InferenceBasis`, and a validator refuses an inconsistent
  pair. This is Separation 2 as a type constraint rather than a review convention: there is
  no way to construct a population estimate that claims to be personal.
- **Cold start is a labelled fallback.** Below `min_samples_early` the personal centre is
  recorded but not used, and the population prior is returned instead, labelled
  `POPULATION_PRIOR`. The engine ships **no** priors, because a hard-coded "typical" value
  is a claim about real learners that nothing in this repository can support. With no prior
  supplied, the reference is `UNAVAILABLE` with a reason — never zero, never a default.
- **The spread cannot be inflated by the outlier it exists to catch.** Robustness needs
  both halves of the estimator. The centre is a winsorised exponential mean whose step is
  capped at `winsorisation_limit_spreads` current dispersions, and the spread is a scaled
  median absolute deviation (`MAD_NORMAL_SCALE = 1.4826`) about that same centre. Bounding
  only one of them is what lets a single 3600-second value relocate the baseline and then
  re-centre the spread around its new location, so the widening and the masking arrive
  together and neither is individually visible. The non-robust estimator remains available
  as configuration, because a robustness claim only means something against an alternative.
- **Retention thins rather than truncates.** `decimate` repeatedly halves an over-limit
  window, holding back the newest value on odd lengths. Memory stays bounded, the retained
  sample keeps spanning a growing interval, and the most recent observations survive. A
  fixed-count truncation would freeze the span and stop representing anything the learner
  did afterwards.
- **Absence is reported, never substituted.** Fewer than two retained values yields no
  dispersion, and a dispersion of exactly zero means every retained value was identical —
  in both cases a reason is recorded rather than an infinite or floored z-score. An
  out-of-range observation is rejected with a reason and is never clamped into range.
- **Time only moves forward.** An observation at or before the profile's last update is
  refused, and a vector must carry a `learner_id` matching the profile. A baseline that
  could absorb an out-of-order event or an unattributed one would be a function of arrival
  order rather than of behaviour, and would not be reproducible by replay.

`DeviationResult` deliberately carries no state, severity, or classification. This layer
reports distance; deciding whether a distance matters belongs to the temporal and
uncertainty engines, and a deviation that arrived pre-classified would collapse Separation 1
at its first use.

The layer is stateless: a `BaselineProfile` is an immutable value passed in and returned, so
two concurrent callers cannot interleave into a shared accumulator and a baseline is
reproducible from its event stream. Every profile records the `settings_fingerprint` that
shaped it, so a profile cannot later be read as though it had been built under thresholds
it never saw.

---

### 4.9 `temporal` — is this a moment, or is this a change?

| Module | Contents |
|---|---|
| `temporal/models.py` | `BehavioralEngagementState`, `TrendDirection`, `TrendCharacter`, `ChangeKind`, `TemporalObservation`, `DeviationPoint`, `TemporalTrack`, `TrajectoryChange`, `StateEvidence`, `StateTransition`, `TemporalState`, `weakest_maturity`, `TEMPORAL_STATE_V1` |
| `temporal/statistics.py` | `current_sign_run`, `signed_direction`, `change_per_observation`, `is_sustained`, `deviation_dispersion`, `mean_standardised` |
| `temporal/engine.py` | `TRACKED_DIMENSIONS`, `TemporalEngine`, `TemporalStateError` |

The baseline layer reports how far a measurement sits from normal. This layer reports
whether that distance is heading anywhere, and whether it has become sustained. The two
questions are kept apart: the layer consumes `DeviationResult` objects and never re-derives
a deviation, because the baseline layer already decided what a deviation is and a second
definition of the same quantity would diverge from it silently.

Four dimensions — accuracy, response seconds, session seconds, content difficulty — are
tracked independently, each with a bounded window of standardised deviations. The engine
is stateless: a `TemporalState` is an immutable value passed in and returned, so a temporal
state is reproducible from its deviation stream and two callers cannot interleave into a
shared accumulator.

The separation between an anomaly and a trajectory is a construction rule, not a
convention:

- **Absence is reported, never inferred.** A learner with no deviation history has no
  state. The engine reports `INSUFFICIENT_DATA` with a reason, and `StateEvidence` refuses
  to pair that state with any evidence of movement. Substituting `STABLE` because nothing
  has gone wrong is the most seductive failure in behavioural analytics, and it is
  prevented by making the state unconstructible rather than by asking a reviewer to notice.
- **A single anomalous observation can never be a trajectory.** The current run is read
  backwards from the newest retained point, and `TrajectoryChange` refuses to be
  constructed for a run shorter than its own `consecutive` count. The Phase 7 exit
  criterion is enforced by the model, not by the call site.
- **Direction is read from the run, not the window.** A rate-of-change window that
  includes the pre-run zeros preceding a departure would report a *shrinking* departure
  (`[-3, -2, -1]`) as worsening, because the averaged rate falls while the deviation is
  still deepening. The run's own sign sequence is the reading, and the window is consulted
  only when the run holds a single point.
- **A run that never started is recorded as such.** `INSUFFICIENT_DATA` always clears
  `run_started_at`, so a warm-up run cannot be read as a run in progress.
- **The standing of the inference travels with the state.** Every `StateEvidence` carries
  the maturity and basis in force when the state was computed, and the *weakest* maturity
  among the contributing dimensions is the one recorded. A state derived partly from a
  population prior is not a personal inference.

Retention is bounded, and the bound is observable rather than silent. `TemporalTrack`
reports its lifetime `recorded` and `dropped` counts alongside the retained window, so a
state that has silently forgotten its history is distinguishable from one that never had
any. `TemporalState.observations` counts what is *retained*, not what was ever seen —
reporting the lifetime total would make a long-lived learner look newly arrived on every
fresh state.

Bounded retention creates one problem that has to be answered rather than ignored: a
trajectory's start can age out of the window while its end is still present. The engine
answers it in two layers. Each track keeps a **ledger** of the trajectories it actually
emitted, and the state's trajectories are rebuilt from those ledgers on every ingest rather
than accumulated, so the two cannot drift apart. Validation then checks the ledger first —
the anti-fabrication guard, which always applies — and checks the retained endpoint values
only while the whole run is still retained. Once the start has aged out the run is only
partly verifiable, and the ledger entry is trusted instead of inventing a failure.

`trajectory_encoded` in the feature set now reports a real value from this layer. It
remains `INSUFFICIENT_DATA` whenever the context carries no trajectory, which is the
correct absence: the feature measures a computed trajectory, and reporting `unknown`
because the engine has not yet seen one would present a placeholder as a measurement.

The unit suite drives this layer with hand-built `DeviationResult` objects, which proves
the temporal arithmetic but not that it accepts what the baseline engine really emits.
`tests/integration/test_temporal_integration.py` closes that gap with the whole traversal —
simulator to context to features to baseline to deviation to temporal state and back out to
a feature vector — because a temporal engine that computed a correct state nobody read would
satisfy every test in its own file while leaving `trajectory_encoded` permanently absent.
That test also pins two things the arithmetic alone does not show: the state's maturity is
the weakest link across all four tracked dimensions, so driving one to `ESTABLISHED` still
reports `NEW`; and `change` describes the run in progress while `trajectories` is the
ledger of every run that ever persisted, so the current claim must be checked against the
current run's length rather than against the ledger.

---

### 4.10 `models` — a model that knows what it was trained on

| Module | Contents |
|---|---|
| `models/encoding.py` | `FeatureRequirement`, `DesignMatrix`, `RejectedRow`, `CATEGORICAL_VOCABULARIES`, `default_features`, `encode_rows`, `requirements_from_schema` |
| `models/artifacts.py` | `ModelArtifact`, `save_artifact`, `load_artifact`, `class_one_probabilities` |
| `models/trainers.py` | `Algorithm`, `TrainingConfig`, `TrainingOutcome`, `train_model` |
| `models/predictors.py` | `Prediction`, `PredictionRefusal`, `predict_with_artifact`, `score_batch` |
| `models/registry.py` | `ModelRegistry`, `ALLOWED_TRANSITIONS`, `RegistryTransition` |

The feature engine produces a *named, versioned* vector. This layer is where that vector
becomes a number a model can be fitted on, and where the question of what the number means
is kept answerable for as long as the model exists.

The design is shaped by one observation: the failure that matters here is not a model that
crashes, it is a model that runs and produces a confidently wrong number. Every decision in
this section exists to make that failure loud instead of silent.

**Imputation does not exist in this layer.** A row with an absent feature is rejected and
reported, never filled in. An absent measurement carries no information about the learner's
future, and imputing one produces a confident prediction from nothing — a confidence that
is indistinguishable, downstream, from one earned by measurement.

**A vector must be attributable.** `FeatureValue.learner_id` is optional, because a vector
can legitimately be computed for a stream whose session start was never observed. But a
vector with no learner cannot be sliced per learner, held out per learner, or traced back
when an evaluation result is questioned. `encode_rows` refuses it and records the refusal
under the explicit `unattributed` sentinel, rather than absorbing it into a training set
that becomes partly unaccountable.

**Categorical vocabularies are declared, not learned.** A vocabulary learned from a
training set is one that can grow at serving time, and a new category would then either have
no column or be folded into a catch-all the model was never trained against. The four
vocabularies are declared in code, mirroring the closed enums the feature engine already
encodes into, so they cannot drift from the definitions they mirror. An unrecognised value
is a refusal, not a fallback.

**The schema is load-bearing.** One-hot columns are positional, so a model handed the right
values in the wrong order produces a valid number and a wrong meaning. `ModelArtifact`
carries the schema and its digest; `load_artifact` recomputes the digest from the stored
schema and refuses a disagreement; `predict_with_artifact` rebuilds the row from the
artifact's recorded requirements and refuses a vector that does not satisfy them, including
one produced under a different feature-set version — because the column names can match
while the quantities behind them do not.

**Reproducibility is predictive equivalence, not byte identity.** The same seed and data
must produce identical fitted parameters, and therefore identical predictions. Byte-for-byte
pickle identity is not the contract: a pickle embeds library versions, so byte identity
would make the claim unmeetable across a dependency upgrade even when the model's behaviour
is unchanged. The seed reaches the estimator as a `random_state` derived from the run seed
and the model identity, so a change in one experiment's consumption of randomness cannot
shift another's. Tree ensembles are fitted with `n_jobs=1`, because work distribution
across threads is documented as an acceptable difference and this layer claims
reproducibility it must therefore not delegate to a scheduler.

**The environment is part of the claim.** Predictive equivalence holds under a fixed
environment, so `ModelArtifact` records the resolved versions of the packages that can move
a numeric result and `version_drift()` compares them against the current environment on
reload. Without this, a model whose predictions changed after an upgrade would be
indistinguishable from one whose behaviour genuinely changed, and the seed recorded in the
sidecar would be pointing at a run that no longer reproduces anything. Drift is reported
rather than refused: a patch-level upgrade frequently does not move a fit, so a loader that
refused on any difference would be unusable in practice, and a caller who needs the
guarantee can act on an empty result. Re-saving preserves the versions recorded at fit
time, because a model relocated between machines must continue to describe the environment
that produced its parameters rather than the one that last wrote it.

**A training function that picks its own splits is a leak waiting to be discovered.** The
caller supplies the train and holdout matrices, so split policy — temporal, per-learner,
stratified — is decided outside this layer and recorded elsewhere. Phase 13 is where that
decision belongs, and it is not yet made: the delivered evaluation core validates split
*settings* but constructs no split, so `split_name` remains caller-supplied and the
enforcement framework is outstanding rather than in place.

**A prediction is dated by its evidence, not by when it was scored.**
`Prediction.computed_at` is the scored vector's own `computed_at` on all three return paths —
the success path and both refusal paths — rather than a wall-clock read. A replay over
identical events therefore reproduces a decision field for field, including its timestamp,
which is what makes an auditable record replayable. The two refusal paths already worked this
way; the success path did not, so one function could report two different instants for the
same vector depending only on whether the model had an opinion. `tests/unit/test_clock_determinism.py`
pins the property from both directions, since a test asserting only that two runs agree
would still pass against a layer that stopped stamping a time at all.

**Undefined metrics are `None`, not a default.** ROC-AUC, average precision, and expected
calibration error are gated on both classes being present. On a single-class split
scikit-learn warns and returns `nan` or `0.0`; a `0.0` returned without being asked for is
a number this system did not measure, and a later reader could not distinguish it from a
real one. `ModelMetrics` types these as optional for exactly this reason.

**A model is a claim, and the registry is where claims are checked.** `ModelRegistry` is
append-only: registration always enters at `EXPERIMENTAL`, a version is minted once, and
every subsequent status is a new record rather than a rewrite. `ALLOWED_TRANSITIONS`
declares the graph, so `PRODUCTION` is reachable only from `CANDIDATE` and `RETIRED` is
terminal. Every transition requires an author and a rationale, because a status change
without an attribution is a decision nobody can be asked about later. The synthetic-only
guard is enforced by the registry on top of the `ModelRecord` validator: relying on the
caller to have constructed a valid record would put the only defence against the system's
worst false claim in the hands of whoever most wants to bypass it.

**Prediction is not decision.** `predict_with_artifact` returns a probability and the
threshold applied to it, and selects nothing. The threshold is a parameter with a visible
default, so a caller who disagrees with it has to say so. `PredictionRefusal` is a
first-class return rather than an exception, because a pipeline meeting an unusable vector
needs to record and count it, not crash; and `score_batch` reports refusals separately so a
shrinking served population is visible rather than silent.

What this layer does not establish: every model here is fitted on simulator output, so its
metrics reflect the simulator's assumptions. No model has been validated against real
learners. A model's presence at `PRODUCTION` means it was promoted through the declared
stages with an attribution, not that it should be serving anyone.

### 4.11 `uncertainty` - four answers, and only one of them is a number

The modelling layer produces a probability. This layer decides whether that probability is
allowed to become an answer, and it is where a prediction system either earns or loses the
right to be believed. `UncertaintyEngine.assess` accepts a `Prediction` or a
`PredictionRefusal` and returns a `PredictionOutcome`, which is one of four verdicts:
`RESOLVED`, `INSUFFICIENT_DATA`, `UNKNOWN`, or `REFUSED`.

**The three negative verdicts are not interchangeable, and their order is the design.**
`REFUSED` means the input could not be encoded, which is a caller error fixed at the payload.
`UNKNOWN` means the model cannot be trusted, which is fixed by replacing or retraining it.
`INSUFFICIENT_DATA` means the model is fine and the learner's history is too thin, which is
fixed by waiting. Sending all three to one handler means one of them never gets the response
that would have helped, so `Remedy` is derived from the verdict and is a distinct value per
verdict — and `REMEDIES` plus the `PredictionOutcome` constructor enforce that mapping
rather than documenting it.

**Calibration is checked before evidence, deliberately.** The engine reads
`CalibrationReport.from_record` and refuses to issue a probability at any confidence when
calibration was never measured or is not believable. Checking evidence first would produce
`INSUFFICIENT_DATA` for a learner with 600 observations and a model that states 0.9 for every
input — a message that says "wait", when the truthful message is "this model is broken".
Because a single-class split cannot support a calibration measurement at all, the absence of
a measurement is `UNKNOWN` and never a zero.

**Confidence is the minimum of four ceilings, and the record says which one bound.**
`ConfidenceAssessment` holds the model assertion, the calibration ceiling (`1 - ECE`), the
evidence ceiling, and the maturity ceiling. The value is their minimum, and the dataclass
refuses to be constructed with a value that is not the minimum, so the derivation cannot be
asserted rather than computed. `Constraint` names the binding one, because "capped at 0.30
because the baseline is NEW" and "capped at 0.30 because the model is miscalibrated" are
different operational situations and only the first is resolved by waiting.

**The maturity cap does not consult the model.** `maturity_ceiling` is a function of
`BaselineMaturity` and `UncertaintySettings` alone, and `max_confidence_when_early` and its
siblings are validated to ascend. A model reporting 0.99 to a `NEW` learner is capped at
0.30 regardless of what it says, which is the only arrangement in which a confident model
cannot promote itself out of a thin history. The evidence ceiling is a logarithmic ramp
from 0.5 at `min_evidence_for_prediction` to 1.0 at `full_confidence_evidence_units`, so
meeting the gate is not a cliff.

**An explanation is a function of one row, or it is not an explanation.** `occlusion_contributions`
replaces each column's value with the training-split median from
`ModelArtifact.feature_reference` and rescores, so a signal is reported only if replacing it
measurably moved this prediction. Global `feature_importances_` and coefficients are
rejected by design: they are identical for every learner the model ever scores, so using them
would produce exactly the unacceptable outcome of explaining a prediction by citing the
model's overall behaviour. One-hot columns are grouped to their base feature so a single
fact is not reported once per category. `DEFAULT_MIN_CONTRIBUTION` filters weak movement out
of the report rather than showing it as a small number, and an empty explanation is a
reportable finding that caps the band at `LOW` while leaving the probability exactly as the
model produced it.

**Provenance travels with every outcome.** `model_id`, `model_version`,
`feature_set_version`, `target_definition_version`, `data_origin`, and `computed_at` are set
from the artifact on all five return paths via the `OutcomeIdentity` `TypedDict`, so a
number is never separable from what it is a number about. This layer does not map the
binary model target onto the seven `BehavioralEngagementState` values; the temporal engine
owns the state, and `target_definition_version` records which definition the probability
was fitted against.

**The outcome is dated by an injected clock.** `UncertaintyEngine` takes a `clock: Clock`
defaulting to `SystemClock`, the same contract the temporal, policy, and outcome engines
already honour, and stamps `computed_at` from it. This layer previously had no clock at all
and read `utc_now()` directly, so an outcome could not be reproduced on replay even when the
probability and confidence were identical. Pairing the vector-derived prediction time with a
clock-derived outcome time means a caller that wants one decision at one instant constructs
the engine with `FixedClock(vector.computed_at)` — the property
`test_prediction_and_outcome_come_from_one_instant` asserts. The `SystemClock` default is
load-bearing in the other direction: without it, a caller who never configures a clock would
silently stop receiving a timestamp at all.

`models/` and `schemas/` do still call `utc_now()`, for the write-side stamps `trained_at`,
`metadata_written_at`, `recorded_at`, and the `EventEnvelope` ingest default. These record
when a record was written rather than inferring anything from evidence, so "no `utc_now` in
`src`" is not a rule this codebase holds and a reviewer should not apply it blind.

What this layer does not establish: every calibration figure in circulation was measured on
simulator output, so the belief that a model is calibrated is currently the simulator's
belief about itself. The engine's job is to make the reasoning auditable, not to make the
numbers true. No verdict here has been exercised against real learners.

---

### 4.12 `outcomes` — what happened, without a claim about why

| Module | Contents |
|---|---|
| `outcomes/models.py` | `OUTCOME_V1`, `OUTCOME_MEASURES`, `Measurement`, `MeasurementStatus`, `OutcomeDirection`, `OutcomeMeasure`, `OutcomeWindow`, `WindowKind`, `OutcomeRecord` |
| `outcomes/measures.py` | `ACTIVITY_EVENTS`, `direction_of`, and the seven measure functions |
| `outcomes/engine.py` | `OutcomeEngine`, `OutcomeError`, window construction, record assembly, `measure`, `measure_many` |

This layer takes a delivery the policy layer already recorded and measures what the learner
did either side of it, as an **observed outcome**. It is the first layer in the pipeline
that looks backwards, and that is what makes its failure modes different in kind from every
layer above it. A prediction that is wrong is visibly wrong, and a policy that acts wrongly
is visible in a decision log. A measurement built from a gap in the event stream is
indistinguishable from a measurement of a learner who did nothing, and it arrives wearing
exactly the same shape as a real finding.

**The three states of a reading, and the direction field's narrow domain.** Every
`Measurement` is `MEASURED`, `INSUFFICIENT_DATA`, or `NOT_APPLICABLE`, and the validator
refuses a direction on anything but `MEASURED`. This is the single most consequential
refusal in the layer: it is what makes "the log stopped" structurally unable to become "the
learner disengaged". The corollary is that an unmeasured reading must carry a `reason`
naming the missing evidence — the reason string is the only channel through which a reader
learns that a number is absent — while a measured reading is not obliged to carry one, so
a no-change can never be confused with a failure to look.

**Windows are delivery-anchored, half-open, and versioned with the measure set.** The before
and after windows meet at exactly one boundary, the delivery instant, and that instant
belongs to the after window alone. An inclusive end on the before window would put the
prompt inside its own baseline, and the baseline would then contain the thing it is a
comparison for. `OUTCOME_V1` names the seven measures together, and every record carries a
fingerprint of the `OutcomeSettings` that produced it, so a record cannot later be read as
though it had been measured under different widths.

**The prompt is not evidence that the learner resumed.** `ACTIVITY_EVENTS` excludes the two
intervention events, because the learner did not click and the system clicked for them.
It also excludes `INACTIVITY_STARTED`, which is the less obvious exclusion: that event
evidences an *absence* of interaction, so counting it in an activity rate would let one
event move a measure in the direction its own name contradicts.

**A direction and a causal claim are separate fields, and a reading can carry both a
direction and a caveat.** The session-continuation measure is the case the design exists
for. When a session ends inside the after window, the reading is `MEASURED` with direction
`DETERIORATED` — the session genuinely did not continue — and a `reason` stating that
whether the prompt is why is not something the events can answer. Refusing to report the
direction would be as dishonest as reporting it as an effect: the first hides a real
observation, the second fabricates a causal claim. A `PROVENANCE` constrained to `OBSERVED`
at the record level makes the fabrication unreachable by field assignment, since the
vocabulary's `CAUSAL_CLAIM` member cannot be constructed.

**The temporal verdict is read, not recomputed.** `SUBSEQUENT_TRAJECTORY` maps the temporal
layer's `direction` to a reading and reports the observation count alongside it. It does no
temporal reasoning of its own, so the unit suite builds `TemporalState` objects by hand —
a second implementation of the same judgement inside the outcome layer would be a second
opinion wearing a reader's clothes.
`tests/integration/test_outcome_integration.py` closes that seam by driving the real
`temporal.TemporalEngine` from six simulated sessions, then asserting that the reading's
observation count equals the state's own. That is the check the hand-built fixtures cannot
make: a trajectory reconstructed from raw deviations on the outcome side would carry a count
of its own inventing.

**Task persistence tracks the item, not the window.** The tracked item is the last one
opened at or before the delivery instant, which is why the measure takes no before window
and reaches further back than any configured width: a task opened twenty minutes before
delivery is still the task that was open when the prompt arrived, and a window that failed
to see it would report that nothing was interrupted for a learner interrupted mid-item.
Displacement is counted in both directions — a different item *opened* and a different item
*closed* inside the window — because closing one is equally evidence that the window's work
happened elsewhere, and a rule watching only for opens reports a learner finishing an earlier
item as still on the task they were interrupted on.

The unit suite's fixtures are synthetic by default and the integration test's record is
labelled synthetic at every level, because a record derived from a simulator reflects that
simulator's assumptions and a slice that was accidentally real would silently mark a whole
record as a measurement of a person. What this layer does not establish: every window
width, the sample floor of four, and the 5% change tolerance are engineering starting points
with stated reasoning, not validated choices. Nothing here attributes a measured change to
the prompt — accumulating that attribution across interventions is the feedback
architecture's work (Phase 12), and is excluded here by design. `tail_minutes` is validated
but read by no measure in `OUTCOME_V1`; its field docstring says so rather than leaving a
reader to infer it from a measurement that ignored it.

### 4.13 `evaluation` — the layer that grades the record, and is graded by nothing

| Module | Contents |
|---|---|
| `evaluation/models.py` | `EvaluationVersion`, `EVALUATION_V1`, `PredictionTarget`, `PredictionHorizon`, `ObservationWindow`, `PredictionSnapshot`, `GroundTruth`, `GroundTruthStatus`, `EvaluationVerdict`, `BinaryVerdict`, `EvaluationRecord`, `InterventionResponse`, `response_from_outcome` |
| `evaluation/engine.py` | `EvaluationEngine`, `DEFAULT_EVALUATION_VERSION`, the half-open window filter, reading refusals, `ground_truth`, `evaluate`, `evaluate_many` |
| `evaluation/aggregate.py` | `ConfusionCounts`, `EvaluationSummary`, `summarise`, `group_by`, `group_by_learner`, `group_by_model_version`, `group_by_target`, `group_by_intervention_response` |

This layer scores a prediction that was already made against the evidence that actually
followed it. It is the only layer that looks backwards *at the whole pipeline*, and it is
terminal: nothing imports it, so a verdict here cannot reach the prediction, policy, or
intervention path that produced the record. Its own imports are the layers above it plus
the supporting layers, with one deliberate reuse — the positive class is
`ACTIONABLE_STATES` read from `policy/models.py` rather than reminted, because a second
notion of "bad" would drift from the one every other layer reads and a disagreement between
two state enums is indistinguishable from two learners in two different states.

**The window is half-open, and the asymmetry is the whole defence.** Admissible evidence
is `predicted_at <= timestamp < predicted_at + duration`, for the same learner, in the same
session, restricted to activity events. The prediction instant is inside the window; the
horizon boundary is outside it. An event at exactly `predicted_at` is the first observation
after the prediction, and one at exactly `predicted_at + duration` belongs to the next
prediction. Evidence before `predicted_at` is the prediction's own input — grading a model
on its inputs is the failure this window exists to make impossible. The rule is a validated
setting (`require_horizon_inclusive_start`, which refuses `False`) rather than a literal in
an expression, because flipping either edge moves events across the prediction boundary and
that is a versioning matter, not a preference.

**Four ground-truth values, and the two non-findings are not interchangeable.**
`CONFIRMED` and `REFUTED` are findings. `INSUFFICIENT_DATA` (the window held too little
activity) and `NOT_ASSESSABLE` (activity was present but was never turned into a
characterisation) are not, and conflating them would misstate how much of a corpus was
actually measured. Every `observed_*` field is `None` in both non-finding cases, and
`assessed` is defined as `confirmed + refuted` rather than `total - not_assessable` for
exactly that reason. A prediction that was never scored is `EvaluationVerdict.NOT_ASSESSABLE`,
not a failure: a gap in a log is not a mistake by the model.

**Only `DECLINE` has a positive class.** `BinaryVerdict` fills all four cells — true
positive, false positive, true negative, false negative — and the matrix is partitioned
only for that target. A `STATE` or `DIRECTION` prediction is a multi-state claim with no
positive class, so it is scored `CORRECT` or `INCORRECT` against the observed value and
reports no cell; its `not_assessable` count in the confusion partition is not a claim that
it went unmeasured, which is what `assessed` is for. Forcing a five-way claim into a binary
cell would invent the positive class it does not have.

**An undefined rate is `None`, and a group is not a ranking.** `precision`, `recall`, `f1`,
`accuracy`, `false_positive_rate`, `false_negative_rate`, and `coverage` return `None` on a
zero denominator rather than `0` or a flattering `0.5`; a summary validator refuses a count
set that does not account for the whole corpus. `group_by_learner` and friends sort their
keys and report each group's own record and learner counts, so iteration order cannot be
read as a leaderboard and a per-learner comparison is not made blind to sample size.
`group_by_model_version` groups on the version rather than the model id, because pooling two
artifacts of one approach would make a regression look like a stable average.

**Provenance is derived, and `GROUND_TRUTH` is unconstructable.** `GroundTruth.provenance`
is constrained to `OBSERVED` unconditionally. The vocabulary contains `GROUND_TRUTH` and
the type refuses it: events following a prediction are not an external source of truth, and
a record stamped with it would claim an independence this repository cannot supply. Saying
the vocabulary and then refusing the other member is the only form of the constraint that
cannot be bypassed by setting a field. `data_origin` is computed from the admissible
evidence, so a mixed real-and-synthetic window is synthetic and cannot be laundered.

**The response is read, not recomputed, and it is never a cause.** `response_from_outcome`
derives an `InterventionResponse` from an upstream `OutcomeRecord` and only when the caller
supplies one. Evaluation does not select interventions, does not re-measure a delivery, and
does not infer a counterfactual: a delivery that improved is recorded as an improvement
observed *after* the delivery. This layer classifies; it does not attribute, and a
`CONFIRMED` prediction says nothing about whether an intervention caused anything.

**Determinism is a structural property, not a test outcome.** The engine holds no mutable
state, takes an injectable `Clock`, and has no randomness, network, or current-time
dependency, so the same inputs produce equal records and byte-identical serialisations
under a `FixedClock`.

What this layer does not establish: nothing here is a claim about predictive performance,
and no number it produces is a measurement of a real learner. Every corpus in the
repository is simulator output, so a `precision` computed over it describes the simulator's
archetypes and the evaluator's arithmetic. The `min_evidence_for_evaluation` floor of four
and the inclusive-start rule are engineering choices with stated reasoning, not empirically
validated ones, for the same reason the outcome layer's widths are not. Split construction,
the ranking and calibration metrics (ROC-AUC, PR-AUC, Brier, ECE), slice reporting beyond
the four group keys, and the personalization comparison are **not implemented here** — the
rank and calibration metrics that do exist belong to the Phase 8 model layer as single-model
holdout metrics, and no code constructs a temporal split, so `ModelMetrics.split_name`
remains caller-supplied.

---

## 5. Storage posture

The file-backed append-only store is implemented (see §4.5). What remains `PLANNED —
NOT IMPLEMENTED` is the hash-manifest layer for non-event artifacts, and the
learner-overlap-safe, temporally-split evaluation harness that will make swapping the store
safe later. The evaluation core in §4.13 scores recorded predictions against observed
evidence; it does not build the temporal split that harness needs, so this gap is unchanged
by Phase 13.

No database vendor is assumed, and none is selected.

---

## 6. Module evolution path

Interfaces are designed so that the following sequence is reachable without rework.
**The supporting layers, event intelligence, context, the feature engine, the personal
baseline engine, the temporal engine, the synthetic simulator, the model layer, and the
uncertainty layer exist today.** The simulator sits outside the pipeline: it produces event
streams for the layers above it and imports from the event schema, never the reverse. The
model layer sits below the feature engine and above nothing: it consumes named, versioned
vectors and produces artifacts and predictions, and reaches no later layer. The uncertainty
layer is deliberately absent from the V0–V5 table below, because it adds no modelling
capability — V1 is still the ceiling on what this system can fit. What Phase 9 added is the
ability to decline to answer, which is a different axis from being able to model.

| Stage | Content | Status |
|---|---|---|
| V0 | rules + statistical baseline | **IMPLEMENTED** |
| V1 | classical ML | **IMPLEMENTED** (`models/` — Phase 8) |
| V2 | temporal ML | `PLANNED — NOT IMPLEMENTED` |
| V3 | personalized temporal modelling | `PLANNED — NOT IMPLEMENTED` |
| V4 | adaptive intervention policy | `PLANNED — NOT IMPLEMENTED` |
| V5 | advanced sequential / representation learning | **DELIBERATELY NOT ATTEMPTED** |

V5 is not planned work. It becomes a legitimate question only if V1–V4 experiments
demonstrate that the remaining error is genuinely sequential and not a feature,
label, or measurement problem.

---

## 7. Verification

Phase 1–13 exit gate, all executed in this repository:

| Check | Command | Result |
|---|---|---|
| Tests | `python -m pytest tests` | 1230 passed, of which 14 under `tests/unit/evaluation/`, 23 in `tests/integration/test_evaluation_integration.py`, and 9 in `tests/unit/test_clock_determinism.py` |
| Integration | `python -m pytest tests/integration` | 50 passed |
| Lint | `python -m ruff check .` | All checks passed |
| Format | `python -m ruff format --check .` | 115 files already formatted |
| Types | `python -m mypy src` | Success, no issues in 57 source files |
| Cross-process reproducibility | 4 processes, `PYTHONHASHSEED` = 0 / 12345 / 999 / random | Identical BLAKE2b digest over the serialised event stream |
| Feature leakage | `test_future_answers_do_not_enter_the_short_window` | Answers after the reference time do not move a computed trend |
| Cold start labelling | `test_cold_start_without_a_prior_is_unavailable_not_zero` | No prior supplied yields `UNAVAILABLE` with a reason, not a default |
| Outlier containment | `test_the_non_robust_estimator_fails_the_same_outlier_test` | The same 3600-second value inflates the non-robust spread by >50× and the robust one not at all |
| Layer composition | `test_baseline_maturity_flows_back_into_the_context_and_features` | A computed maturity reaches `baseline_maturity_encoded` and closes the blocking context gap |
| Model reproducibility | `test_the_same_seed_produces_identical_predictions`, `test_a_different_seed_produces_a_different_model` | Same seed → identical probabilities; different seed → a different fit, asserted as predictive equivalence rather than pickle bytes |
| Environment recorded | `test_the_library_versions_round_trip`, `test_saving_preserves_the_training_environment` | A reloaded model reports the environment it was fitted in, not the one that last saved it |
| Layer boundary | `test_no_model_module_imports_a_later_layer`, `test_model_modules_import_only_permitted_layers` | No model module imports a later layer or an undeclared sibling, including via deferred function-level imports |
| Model schema binding | `test_a_tampered_schema_is_refused`, `test_an_artifact_without_a_schema_is_refused` | A model whose stored digest disagrees with its schema, or that carries no schema at all, is refused at load |
| Missing-feature honesty | `test_absent_feature_is_rejected_not_imputed`, `test_a_vector_with_an_absent_feature_is_refused` | An absent feature produces a named refusal at both encoding and scoring, never an imputed value |
| Registry lifecycle | `test_skipped_stages_are_refused`, `test_production_is_only_reachable_from_candidate`, `test_retired_is_terminal` | `PRODUCTION` is reachable only from `CANDIDATE`; `RETIRED` has no outgoing edges |
| Synthetic-only guard | `test_synthetic_only_model_cannot_be_promoted`, `test_a_synthetic_model_may_still_be_retired` | Promotion of a simulator-derived model is refused, while retirement stays available |
| Undefined metrics | `test_a_single_class_holdout_reports_no_roc_auc` | Rank metrics are `None` on a single-class holdout, not a fabricated 0.5 |
| Negative verdicts reachable | `test_insufficient_data_is_reached_below_the_evidence_gate`, `test_unknown_is_reached_when_calibration_cannot_be_measured`, `test_unknown_is_reached_when_calibration_is_not_believable` | All four verdicts are exercised, each with its own remedy, and `REFUSED` stays distinct from the two data-shaped negatives |
| Confidence is derived | `test_all_four_ceilings_are_recorded`, `test_the_confidence_is_a_derived_not_invented_number`, `test_a_value_that_is_not_the_lowest_ceiling_is_refused` | The recorded value equals the minimum of the four ceilings, and an assessment claiming any other value cannot be built |
| Maturity cap independence | `test_a_confident_model_is_capped_by_maturity`, `test_maturity_bounds_the_output_regardless_of_model_confidence` | A model reporting above 0.9 is capped at the `EARLY` ceiling of 0.50, and the cap is a function of maturity alone |
| Calibration correctness | `test_a_perfectly_calibrated_model_scores_near_zero`, `test_an_inverted_model_scores_near_one`, `test_ece_uses_observed_frequency_not_accuracy` | The corrected ECE metric matches known answers; the previous implementation scored a perfect model near 0.5 |
| Attribution is per-row | `test_probability_moves_only_for_columns_the_model_weights`, `test_a_feature_the_model_ignores_is_never_attributed`, `test_the_unavailability_reason_refuses_the_global_importance_shortcut` | Attribution is checked column-by-column against the model's own coefficients, a zero-coefficient feature is never credited, and the global-importance substitution is refused by name |
| Layer boundary | `test_no_uncertainty_module_imports_a_later_layer`, `test_uncertainty_modules_import_only_permitted_layers` | No uncertainty module imports policy, outcome, feedback, evaluation, or API |
| Clock injection (prediction) | `test_two_runs_over_one_vector_agree_on_the_timestamp`, `test_the_timestamp_is_the_vectors_own_reference_time`, `test_refusals_and_predictions_agree_on_where_the_time_comes_from` | A prediction is dated by the vector it scored on every return path, so a replay reproduces the instant and one function cannot report two instants for one vector |
| Clock injection (outcome) | `test_two_runs_with_one_clock_agree_on_the_timestamp`, `test_a_different_clock_produces_a_different_timestamp`, `test_the_default_engine_still_reads_a_real_instant` | The outcome is dated by the injected clock, the clock is demonstrably read rather than accepted and discarded, and the unconfigured default is still a working clock |
| One decision, one instant | `test_prediction_and_outcome_come_from_one_instant` | Scoring and assessing the same vector under `FixedClock(vector.computed_at)` yields one instant across both layers, which is what a replayable audit trail depends on |
| Window geometry | `test_the_windows_meet_at_exactly_one_instant`, `test_the_delivery_instant_belongs_to_the_after_window_only` | The delivery instant is evidence after the prompt and never before it, so it cannot sit inside the baseline it is a baseline for |
| Unmeasured is not deterioration | `test_an_absent_reading_cannot_carry_a_direction`, `test_silence_after_the_prompt_is_not_reported_as_deterioration` | A direction on a non-measured reading cannot be constructed, and a learner who left is indistinguishable at the type from a log that has not flushed |
| Reason integrity | `test_an_absent_reading_without_a_reason_is_refused`, `test_a_measured_reading_need_not_carry_a_reason` | A withheld reading must name the missing evidence; a measured one is not obliged to manufacture one |
| Measurement is not causation | `test_a_measured_reading_may_carry_a_caveat`, `test_a_record_claiming_a_provenance_other_than_observed_is_refused` | A direction and a caveat coexist in one reading, and `CAUSAL_CLAIM` cannot be constructed |
| The prompt is not the learner | `test_the_prompt_itself_does_not_count_as_the_learner_resuming`, `test_an_inactivity_event_is_not_counted_as_activity` | Counting the intervention events as activity makes no difference to the rate, and the event that evidences an absence of interaction does not count as one either |
| Traversal composition | `test_a_simulated_session_produces_a_record_over_every_measure`, `test_the_trajectory_reading_carries_the_temporal_layers_own_verdict` | Six simulated sessions through the real context, feature, baseline, temporal, policy, and outcome layers produce a record with all seven measures, and the trajectory's observation count is the temporal engine's own |
| Provenance is derived | `test_every_reading_is_labelled_synthetic`, `test_a_record_hiding_synthetic_measurements_behind_a_real_origin_is_refused` | A simulator-derived record is synthetic at every level, and a record claiming a real origin over synthetic evidence is refused |
| Determinism | `test_the_same_events_produce_the_same_record`, `test_a_different_window_configuration_changes_the_fingerprint` | Two measurements of one delivery agree field for field, and a different geometry produces a different fingerprint |
| Refusals survive composition | `test_events_from_another_learner_are_refused`, `test_a_temporal_state_for_another_learner_is_refused`, `test_a_delivery_that_was_never_made_cannot_be_measured` | Misdirected events, a state describing someone else, and an intervention never delivered all raise rather than yielding a thin record |
| Window geometry (evaluation) | `test_event_at_prediction_instant_included`, `test_event_at_horizon_end_excluded`, `test_event_one_microsecond_before_end_included`, `test_event_one_microsecond_after_end_excluded` | The start is inclusive and the end exclusive at actual timestamp precision, so an event at the horizon boundary belongs to the next prediction |
| Isolation | `test_learner_isolation_ignores_other_learner`, `test_session_isolation` | Another learner's or another session's activity cannot reach the record even inside the window |
| Non-findings stay distinct | `test_insufficient_data_distinct_from_not_assessable` | Thin evidence stays `INSUFFICIENT_DATA` even when a reading is supplied; ample evidence with no reading is `NOT_ASSESSABLE` |
| Refused readings | `test_contradictory_reading_refused`, `test_reading_anchored_before_prediction_refused`, `test_insufficient_data_reading_refused` | A misdirected, a pre-prediction, and a self-declared-insufficient reading each yield `NOT_ASSESSABLE` rather than a score |
| Matrix integrity | `test_all_four_cells_aggregate_correctly` | TP, FP, TN, and FN each occur once and aggregate to the expected counts, with accuracy, precision, and recall at 0.5 |
| Historical metadata is immutable | `test_historical_prediction_metadata_cannot_be_rewritten` | The predicted claim, learner, instant, and model version cannot be edited after the prediction was made |
| No fabricated ground truth | `test_ground_truth_provenance_cannot_be_asserted`, `test_engine_never_emits_ground_truth_provenance` | An engine-produced ground truth re-validated with `GROUND_TRUTH` is refused, and the engine never emits it |
| Determinism | `test_same_inputs_produce_identical_records`, `test_evaluate_many_is_deterministic` | Repeated runs over the same inputs agree field for field and byte for byte under a `FixedClock` |
| Configuration cannot be weakened | `test_min_evidence_floor_is_respected`, `test_zero_evidence_floor_is_rejected`, `test_exclusive_window_start_may_not_be_configured` | The evidence floor is honoured, a floor of zero is refused, and the inclusive window start cannot be switched off |
| Real-state seam | `test_evaluator_accepts_a_genuine_temporal_state` | A `TemporalState` produced by the real `TemporalEngine` is scored without any stand-in, so the two layers agree on a shape |

The last row of the Phase 1–6 group is not redundant with the test suite. The unit tests
compare two generations inside one interpreter, where `PYTHONHASHSEED` is constant; a
hash-order dependence in the identifier or seeding path would pass them and still break
reproducibility across runs.

The Phase 8 rows carry the same intent. A model layer that reports metrics on simulator
output is running correctly and is still not evidence about learners, and the test suite
cannot make it one — the only honest reading of a passing model suite is that the
machinery is sound and the claims are unproven.

The Phase 9 rows carry it once more, and this is the phase where it would be most tempting
to stop. Every calibration figure in the system was measured on simulator output, so a
`RESOLVED` verdict currently means the simulator's model is well calibrated *on data the
simulator produced*, and the `UNKNOWN` verdicts are as much a statement about the simulator
as about the model. What Phase 9 establishes is that the refusal is available, specific,
and correctly ordered — not that the numbers behind it are true.

The Phase 11 rows carry the same warning in its sharpest form, because this is the first
layer that looks backwards at real behaviour. A record built from simulator output is
correctly computed, correctly provenance-labelled, and still establishes nothing about any
learner: the intervention that produced it was delivered by a test rather than by the
system, and the engagement that followed it was generated by an archetype rather than
observed. What Phase 11 establishes is that a missing event cannot become a finding about a
person — the window geometry, the direction validator, and the reason requirement are what
make that structurally true, and the integration test shows the whole traversal agreeing
without any of the layer's inputs being arranged for the test's benefit. Whether these
windows and tolerances are the *right* ones for a real learner population is a question no
test in this repository can answer.

The Phase 13 rows carry the warning at its limit, because this is the layer whose whole
purpose is to produce a number that a reader might mistake for a result. Every corpus
available to it is simulator output, so a `precision` of 0.5 computed here describes the
simulator's archetypes and the evaluator's arithmetic — not a model, not a learner, and not
a system that has been shown to work. What Phase 13 establishes is narrower and structural:
a gap in a log cannot become a finding, the window cannot admit a prediction's own input, an
undefined rate is reported as undefined rather than as zero, and `GROUND_TRUTH` cannot be
constructed even by a caller who wants it. Those are real guarantees about the arithmetic
and the types. They are not evidence of accuracy, and the layer reports no metric that
could be read as such.

The clock-injection rows are the exception to that pattern, and the reason is worth stating
plainly. Every other row in this table is a guarantee about a claim the system makes, and
most of them are guarantees that a number should not be read as more than it is. The clock
rows guarantee something narrower and fully delivered: a decision reached twice from the same
evidence carries the same timestamp, so a recorded decision can be replayed and compared. No
synthetic corpus is required to establish that, because the claim is about arithmetic
identity rather than about behaviour — which is also why it is the one guarantee here that
survives the absence of real data intact.

One caveat applies to the table itself, and it has now bitten three times. A recorded gate
result is a claim about the moment the gate ran, not a state of the repository. The Phase 7
close found a format gate verified once and not re-run for five phases; the Phase 8 close
found a lint gate green over a file set that `.gitignore` had silently excluded, taking the
whole model layer with it; and the clock work above was completed with the format gate
stale, reporting 114 clean files where 115 existed on disk. The checks themselves are sound
in all three cases. The failure is treating a table as a gate, and the only reliable
defence is to run the commands — and to compare the reported file count against the files on
disk, which is what surfaced the second and third of those.
