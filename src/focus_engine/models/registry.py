"""Append-only model registry with an explicit lifecycle.

The registry exists to answer one question that a file of trained models cannot: *how did
this model get to production, and who decided that?* A model is a claim. The registry is
the audit trail for the claim, and the design is shaped by three failures it must prevent.

**No automatic promotion.** There is no code path in this module that raises a model's
status as a side effect of something else happening. A model does not become ``VALIDATED``
because it scored well on a split, and it does not become ``PRODUCTION`` because a newer
version was registered. Every transition is a distinct, explicit call whose arguments
include *who* is authorising it and *why*. A training run that finished successfully is
evidence about a model, not a mandate to serve it.

**Transitions are appended, never overwritten.** The registry is a sequence of
:class:`~focus_engine.schemas.versioning.ModelRecord` entries, not a mutable map. A
model's current status is a *derived* value: the status of its latest record. A model's
history is therefore always fully reconstructable, and a model that was once in production
and was later retired cannot be made to look as though it had never been served. An
overwritable ``status`` field is a field that will be overwritten.

**The path is constrained, not merely documented.** :data:`_ALLOWED_TRANSITIONS` encodes
which moves are legal. ``RETIRED`` is terminal, because a retired model that can be
reinstated has not been retired - it has been paused, and a serving path that may
un-pause a withdrawn model cannot reason about what is currently deployed. There is no
transition *out* of ``EXPERIMENTAL`` except to ``VALIDATED`` or ``RETIRED``: skipping
validation is the failure the lifecycle exists to prevent.

Synthetic-data safety is enforced one level down, by a validator on
:class:`~focus_engine.schemas.versioning.ModelRecord` itself, so it holds for records that
never pass through this registry. The registry's contribution is the transition graph: it
must not offer a caller a way to reach ``PRODUCTION`` that the record would then reject at
an inconvenient moment, and it must record the authorisation that accompanied each move.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from focus_engine.schemas.primitives import Timestamp, non_empty_text
from focus_engine.schemas.versioning import (
    ModelRecord,
    ModelStatus,
    ModelVersion,
    checked_version,
)
from focus_engine.utils.clock import Clock, SystemClock
from focus_engine.utils.determinism import digest_iterable

__all__ = [
    "ALLOWED_TRANSITIONS",
    "ModelRegistry",
    "ModelRegistryError",
    "ModelRegistryNotFoundError",
    "RegistryTransition",
]


class ModelRegistryError(Exception):
    """Raised when a registry operation is not permitted."""


class ModelRegistryNotFoundError(ModelRegistryError):
    """Raised when a requested model or version is absent from the registry."""


#: The legal lifecycle moves. A move absent from this mapping is refused regardless of
#: what the caller asks for. ``RETIRED`` is deliberately absent as a key: retirement is
#: terminal, so no model can be reinstated by a second transition.
ALLOWED_TRANSITIONS: Final[dict[ModelStatus, frozenset[ModelStatus]]] = {
    ModelStatus.EXPERIMENTAL: frozenset({ModelStatus.VALIDATED, ModelStatus.RETIRED}),
    ModelStatus.VALIDATED: frozenset({ModelStatus.CANDIDATE, ModelStatus.RETIRED}),
    ModelStatus.CANDIDATE: frozenset({ModelStatus.PRODUCTION, ModelStatus.RETIRED}),
    ModelStatus.PRODUCTION: frozenset({ModelStatus.RETIRED}),
    ModelStatus.RETIRED: frozenset(),
}

#: Statuses from which no further transition is possible.
TERMINAL_STATUSES: Final[frozenset[ModelStatus]] = frozenset({ModelStatus.RETIRED})


class RegistryTransition(BaseModel):
    """One recorded lifecycle move.

    A transition is a fact about a decision - who made it, and on what grounds - and not
    merely a change of value. ``author`` and ``rationale`` are required rather than
    optional because a status change with no recorded reason is indistinguishable, six
    months later, from a status change that nobody made deliberately.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: non_empty_text
    """The model whose status changed."""

    model_version: ModelVersion
    """The specific version whose status changed."""

    from_status: ModelStatus | None = None
    """The status before the move, or ``None`` for the model's first registration.

    ``None`` rather than a sentinel status because registration is not a move from
    anything: it is the moment a model becomes known to the registry, and recording it as
    a transition out of a fake ``UNREGISTERED`` status would put a value in the vocabulary
    that no model can ever hold.
    """

    to_status: ModelStatus
    """The status after the move."""

    author: non_empty_text = Field(min_length=1)
    """Who authorised the move. Required."""

    rationale: non_empty_text = Field(min_length=1)
    """Why. A move with no stated grounds cannot be reviewed after the fact."""

    recorded_at: Timestamp | None = None
    """When the move was recorded, from the registry's injected clock."""

    @model_validator(mode="after")
    def _reject_a_no_op(self) -> RegistryTransition:
        """Forbid a transition that does not change the status.

        Recording ``EXPERIMENTAL -> EXPERIMENTAL`` would pad the history with entries that
        assert nothing, making a genuine move harder to find in an audit. Registration is
        exempt: its ``from_status`` is ``None``, which is not a status at all.

        Returns:
            ``self``, unchanged.

        Raises:
            ValueError: If both statuses are present and equal.
        """
        if self.from_status is not None and self.from_status is self.to_status:
            raise ValueError(
                f"a transition must change the status; {self.from_status.value!r} was given "
                "for both sides"
            )
        return self

    def describe(self) -> str:
        """Render the transition as a single line.

        Returns:
            A description of the move, its author, and its rationale. A registration reads
            as ``(registered) -> experimental``.
        """
        source = "registered" if self.from_status is None else self.from_status.value
        return (
            f"{self.model_id}:{self.model_version} "
            f"{source} -> {self.to_status.value} "
            f"by {self.author} ({self.rationale})"
        )


class ModelRegistry:
    """An append-only sequence of model records and their lifecycle transitions.

    The registry holds records in registration order and never removes or edits one. A
    model's current status is the status of its most recent record, so the registry holds
    one record per transition rather than one mutable record per model. That is what makes
    a status change auditable: the fact that a model was in production is preserved in the
    history even after it has been retired.
    """

    __slots__ = ("_clock", "_records", "_transitions")

    def __init__(self, clock: Clock | None = None) -> None:
        """Create an empty registry.

        Args:
            clock: Time source for recording transitions. Injected so the registry's
                history is reproducible in tests, exactly as the temporal engine's clock is.
        """
        self._clock = clock if clock is not None else SystemClock()
        self._records: list[ModelRecord] = []
        self._transitions: list[RegistryTransition] = []

    def register(
        self,
        record: ModelRecord,
        *,
        author: str,
        rationale: str,
    ) -> ModelRecord:
        """Register a newly trained model as ``EXPERIMENTAL``.

        Registration is the only way a model enters the registry, and it always enters at
        ``EXPERIMENTAL``. A caller cannot register a model directly at ``PRODUCTION``,
        because a model that has never been through validation has not been validated, and
        the only way to make the registry believe otherwise is to not use the registry.

        Args:
            record: The model record to register. Its status must already be
                ``EXPERIMENTAL``; a record claiming a higher status at registration is a
                caller attempting to skip the lifecycle, and is refused rather than
                corrected, because silently downgrading it would hide the attempt.
            author: Who is registering the model.
            rationale: Why it is being registered.

        Returns:
            The registered record, with status ``EXPERIMENTAL``.

        Raises:
            ModelRegistryError: If the record claims a status other than ``EXPERIMENTAL``,
                or if this version is already registered.
        """
        if record.status is not ModelStatus.EXPERIMENTAL:
            raise ModelRegistryError(
                f"a new model must be registered as {ModelStatus.EXPERIMENTAL.value!r}, not "
                f"{record.status.value!r}. Registration is the start of the lifecycle, not a "
                "way to skip it; use the transition method to advance the status."
            )
        if self._find(record.model_id, record.model_version) is not None:
            raise ModelRegistryError(
                f"model {record.model_id}:{record.model_version} is already registered. The "
                "registry is append-only; a version is minted once and its history is extended, "
                "not replaced."
            )
        checked_version(record.model_version)
        self._records.append(record)
        self._transitions.append(
            RegistryTransition(
                model_id=record.model_id,
                model_version=record.model_version,
                from_status=None,
                to_status=ModelStatus.EXPERIMENTAL,
                author=author,
                rationale=rationale,
                recorded_at=self._clock.now(),
            )
        )
        return record

    def transition(
        self,
        model_id: str,
        model_version: str,
        to_status: ModelStatus,
        *,
        author: str,
        rationale: str,
    ) -> ModelRecord:
        """Advance a model's status, recording who authorised the move and why.

        This is the only path to ``VALIDATED``, ``CANDIDATE`` and ``PRODUCTION``, and it is
        a method rather than an attribute so that the move is always accompanied by an
        attribution. There is no automatic promotion: nothing else in this class, and no
        caller, can change a status without coming through here.

        Args:
            model_id: The model to advance.
            model_version: The version to advance.
            to_status: The status to move to.
            author: Who authorises the move.
            rationale: Why. Required.

        Returns:
            A new record at the new status, appended to the history.

        Raises:
            ModelRegistryNotFoundError: If the model version is not registered.
            ModelRegistryError: If the move is not in the allowed transition graph, or if
                the resulting record would be invalid (e.g. promoting a synthetic-only
                model to ``VALIDATED``).
        """
        current = self._find(model_id, model_version)
        if current is None:
            raise ModelRegistryNotFoundError(
                f"model {model_id}:{model_version} is not registered. A status can only be "
                "changed for a model the registry has seen."
            )
        permitted = ALLOWED_TRANSITIONS.get(current.status, frozenset())
        if to_status not in permitted:
            raise ModelRegistryError(
                f"cannot move {model_id}:{model_version} from "
                f"{current.status.value!r} to {to_status.value!r}. Permitted moves from "
                f"{current.status.value!r} are "
                f"{sorted(s.value for s in permitted) or ['none; this status is terminal']}."
            )
        try:
            moved = current.model_copy(update={"status": to_status})
            moved = ModelRecord.model_validate(moved.model_dump())
        except ValueError as error:
            raise ModelRegistryError(
                f"the transition to {to_status.value!r} is not permitted for {model_id}:"
                f"{model_version}: {error}"
            ) from error
        self._records.append(moved)
        transition = RegistryTransition(
            model_id=model_id,
            model_version=model_version,
            from_status=current.status,
            to_status=to_status,
            author=author,
            rationale=rationale,
            recorded_at=self._clock.now(),
        )
        self._transitions.append(transition)
        return moved

    def history(self, model_id: str, model_version: str) -> tuple[ModelRecord, ...]:
        """Return every record for a model version, oldest first.

        Args:
            model_id: The model to look up.
            model_version: The version to look up.

        Returns:
            The full sequence of records, in registration and transition order.

        Raises:
            ModelRegistryNotFoundError: If the model version is not registered.
        """
        records = tuple(
            record
            for record in self._records
            if record.model_id == model_id and record.model_version == model_version
        )
        if not records:
            raise ModelRegistryNotFoundError(f"model {model_id}:{model_version} is not registered")
        return records

    def status(self, model_id: str, model_version: str) -> ModelStatus:
        """Return a model's current status, derived from its latest record.

        Args:
            model_id: The model to look up.
            model_version: The version to look up.

        Returns:
            The status of the most recent record.

        Raises:
            ModelRegistryNotFoundError: If the model version is not registered.
        """
        return self.history(model_id, model_version)[-1].status

    def is_registered(self, model_id: str, model_version: str) -> bool:
        """Report whether a model version is present.

        Args:
            model_id: The model to look up.
            model_version: The version to look up.

        Returns:
            ``True`` if at least one record exists for that version.
        """
        return self._find(model_id, model_version) is not None

    def production_models(self) -> tuple[ModelRecord, ...]:
        """Return every model currently in production.

        Returns:
            The latest record of each model whose current status is ``PRODUCTION``.
        """
        latest: dict[tuple[str, str], ModelRecord] = {}
        for record in self._records:
            latest[(record.model_id, record.model_version)] = record
        return tuple(
            record for record in latest.values() if record.status is ModelStatus.PRODUCTION
        )

    def transitions(self, model_id: str, model_version: str) -> tuple[RegistryTransition, ...]:
        """Return every recorded lifecycle event for a model version, oldest first.

        Args:
            model_id: The model to look up.
            model_version: The version to look up.

        Returns:
            Registration first (``from_status is None``), then each status move in order.
        """
        return tuple(
            transition
            for transition in self._transitions
            if transition.model_id == model_id and transition.model_version == model_version
        )

    def all_records(self) -> tuple[ModelRecord, ...]:
        """Return every record in the registry, in registration order.

        Returns:
            All records, oldest first.
        """
        return tuple(self._records)

    def digest(self) -> str:
        """Return a stable digest over the registry's content.

        The digest is over model identifiers, versions, and statuses - not over metrics or
        timestamps - so that it identifies *which models are registered and in what state*,
        which is the fact a manifest needs to attest to. Two registries holding the same
        models in the same states have the same digest regardless of when they were built.

        Returns:
            A hex digest.
        """
        latest: dict[tuple[str, str], ModelRecord] = {}
        for record in self._records:
            latest[(record.model_id, record.model_version)] = record
        return digest_iterable(
            f"{model_id}:{version}={record.status.value}"
            for (model_id, version), record in sorted(latest.items())
        )

    def _find(self, model_id: str, model_version: str) -> ModelRecord | None:
        """Return the latest record for a version, or ``None`` if absent.

        Args:
            model_id: The model to look up.
            model_version: The version to look up.

        Returns:
            The most recent record, or ``None``.
        """
        latest: ModelRecord | None = None
        for record in self._records:
            if record.model_id == model_id and record.model_version == model_version:
                latest = record
        return latest


def assert_monotonic_versions(
    records: Iterable[ModelRecord],
) -> Sequence[ModelVersion]:
    """Check that each model family's versions increase numerically.

    This is a helper rather than a registry method because it operates on an arbitrary
    sequence, e.g. a registry's records read from disk. A model version that decreases
    within a family would mean a later artifact claims to predate an earlier one, which
    would break the meaning of "latest".

    Args:
        records: The records to check.

    Returns:
        The versions in the order supplied.

    Raises:
        ModelRegistryError: If any family's version numbers do not increase.
    """
    seen: dict[str, int] = {}
    versions: list[ModelVersion] = []
    for record in records:
        suffix = record.model_version.rsplit("_V", 1)[-1]
        number = int(suffix) if suffix.isdigit() else 0
        if record.model_id in seen and number <= seen[record.model_id]:
            raise ModelRegistryError(
                f"model {record.model_id} has version {record.model_version} after "
                f"{seen[record.model_id]}; versions within a family must increase"
            )
        seen[record.model_id] = number
        versions.append(record.model_version)
    return versions
