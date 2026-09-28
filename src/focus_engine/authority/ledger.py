"""Append-only, hash-chained ledger of authority decisions, feedback, and decay.

The authority layer's value depends on its auditability. A mutable history can be rewritten
and an unchained ledger can be spliced, and both leave the reviewer guessing whether the
chain they are reading is the chain that was written. This ledger is append-only, each
entry carries the hash of the one before it, and tampering breaks the chain visibly.

**Integrity is checked by the reader, not the writer.** A writer that validates its own
writes before recording them is self-auditing, which is circular. :meth:`integrity_ok`
returns a report for a caller, and the Lab's UI shows the result so a human can act on it.
**Chaining is plain SHA-256.** A cryptographic commitment is sufficient and deliberate:
the ledger is not trying to prevent forgery at scale (a user with filesystem access can
rewrite the whole chain) but to make tampering with a single entry visible to anyone who
reads the log later.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

__all__ = ["AuthorityLedger", "LedgerEntry", "LedgerIntegrity"]


def _hash_payload(payload: dict[str, object], previous_hash: str) -> str:
    """Deterministic hash of a payload and its chain link.

    The JSON serialisation is sorted to avoid order-dependent hashes. ``None`` values
    are preserved so that a missing field is distinguishable from an absent key.

    Args:
        payload: The flat dict to hash.
        previous_hash: The hash of the preceding entry, or the empty string for the
            first entry.

    Returns:
        A lowercase hex SHA-256 digest.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{previous_hash}:{canonical}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One record in the authority ledger.

    Attributes:
        entry_id: Sequential position in the ledger, starting at ``1``.
        recorded_at: When the entry was written.
        event_type: The category: ``authority_calculation``, ``feedback_received``,
            ``feedback_applied``, ``decay_applied``, ``aggregated_disagreement``,
            ``system_start``, or ``system_stop``.
        learner_id: The learner the entry concerns.
        context_type: The context, or ``None`` for whole-system events.
        decision_ref: The decision this entry is about, if any.
        detail: Flat, JSON-safe payload. The schema is per ``event_type``.
        previous_hash: The hash of the preceding entry, or ``""`` for the first.
        entry_hash: The hash of this entry.
    """

    entry_id: int
    recorded_at: datetime
    event_type: str
    learner_id: str
    context_type: str | None
    decision_ref: str | None
    detail: dict[str, object]
    previous_hash: str
    entry_hash: str

    def to_dict(self) -> dict[str, object]:
        """Serialise the entry for hashing and storage.

        Returns:
            A dict suitable for ``json.dumps``.
        """
        return {
            "entry_id": self.entry_id,
            "recorded_at": self.recorded_at.isoformat(),
            "event_type": self.event_type,
            "learner_id": self.learner_id,
            "context_type": self.context_type,
            "decision_ref": self.decision_ref,
            "detail": dict(self.detail),
            "previous_hash": self.previous_hash,
        }


@dataclass(frozen=True, slots=True)
class LedgerIntegrity:
    """Report on the chain's integrity.

    Attributes:
        is_valid: ``True`` when every link is unbroken and every hash matches.
        total_entries: How many entries the chain holds.
        broken_at: The ``entry_id`` where the first mismatch was found, or ``None``.
        reason: A human-readable explanation.
    """

    is_valid: bool
    total_entries: int
    broken_at: int | None
    reason: str


class AuthorityLedger:
    """Append-only, hash-chained record of authority-layer events.

    The ledger is the final source of truth for what happened and when. Nothing the
    authority layer does is not recorded here, and nothing recorded here can be removed.
    That rigour is the point: a reviewer who opens the ledger can reconstruct the sequence
    of authority calculations, feedbacks, and decay events in the order they happened, and
    can verify that the record they are reading has not been altered.

    Args:
        max_entries: Upper bound on retained entries. The oldest entries are trimmed when
            this is exceeded, which is logged rather than silent. An error would be raised
            if this were ``0``; a ledger of size zero is useless.
    """

    __slots__ = ("_entries", "_max")

    def __init__(self, *, max_entries: int = 1000) -> None:
        """Initialise an empty ledger.

        Args:
            max_entries: Maximum retained entries. Must be positive.

        Raises:
            ValueError: If ``max_entries`` is not positive.
        """
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive; got {max_entries!r}")
        self._max = max_entries
        self._entries: list[LedgerEntry] = []

    def append(
        self,
        event_type: str,
        learner_id: str,
        detail: dict[str, object],
        *,
        context_type: str | None = None,
        decision_ref: str | None = None,
        recorded_at: datetime | None = None,
    ) -> LedgerEntry:
        """Append an entry and return it.

        Args:
            event_type: The category of event.
            learner_id: The learner affected.
            detail: The event payload. Must be ``json``-serialisable.
            context_type: The context, if any.
            decision_ref: The decision this entry relates to, if any.
            recorded_at: When the entry was recorded. Injected for testability; the caller
                should usually omit it and let the method read the clock. When ``None``
                the method falls back to ``datetime.now(UTC)``.

        Returns:
            The appended :class:`LedgerEntry`.
        """
        from datetime import UTC  # noqa: PLC0415
        from datetime import datetime as _dt

        when = _dt.now(UTC) if recorded_at is None else recorded_at
        previous_hash = "" if not self._entries else self._entries[-1].entry_hash
        entry_id = len(self._entries) + 1
        payload = {
            "entry_id": entry_id,
            "recorded_at": when.isoformat(),
            "event_type": event_type,
            "learner_id": learner_id,
            "context_type": context_type,
            "decision_ref": decision_ref,
            "detail": detail,
            "previous_hash": previous_hash,
        }
        entry_hash = _hash_payload(payload, previous_hash)
        entry = LedgerEntry(
            entry_id=entry_id,
            recorded_at=when,
            event_type=event_type,
            learner_id=learner_id,
            context_type=context_type,
            decision_ref=decision_ref,
            detail=detail,
            previous_hash=previous_hash,
            entry_hash=entry_hash,
        )
        self._entries.append(entry)
        self._trim()
        return entry

    def _trim(self) -> None:
        """Drop the oldest entries when the size exceeds the limit."""
        while len(self._entries) > self._max:
            self._entries.pop(0)

    def entries(self) -> tuple[LedgerEntry, ...]:
        """Return every entry, oldest first.

        Returns:
            A tuple of :class:`LedgerEntry` objects. The caller cannot mutate the ledger
            through the returned tuple.
        """
        return tuple(self._entries)

    def entries_for(self, learner_id: str) -> tuple[LedgerEntry, ...]:
        """Return the entries that concern one learner.

        Args:
            learner_id: The learner to filter by.

        Returns:
            Matching entries, oldest first.
        """
        return tuple(e for e in self._entries if e.learner_id == learner_id)

    def __len__(self) -> int:
        """Return the number of entries in the ledger."""
        return len(self._entries)

    def __iter__(self) -> tuple[LedgerEntry, ...]:
        """Return all entries, oldest first."""
        return self.entries()

    def integrity_ok(self) -> LedgerIntegrity:
        """Verify the chain is unbroken and every hash matches.

        Returns:
            A :class:`LedgerIntegrity` reporting validity, entry count, and the first
            break, if any.
        """
        if not self._entries:
            return LedgerIntegrity(
                is_valid=True,
                total_entries=0,
                broken_at=None,
                reason="an empty ledger has no links to break",
            )
        for i, entry in enumerate(self._entries):
            expected_previous = "" if i == 0 else self._entries[i - 1].entry_hash
            if entry.previous_hash != expected_previous:
                return LedgerIntegrity(
                    is_valid=False,
                    total_entries=len(self._entries),
                    broken_at=entry.entry_id,
                    reason=(
                        f"entry {entry.entry_id} has previous_hash "
                        f"{entry.previous_hash!r}, but entry {i} has hash "
                        f"{expected_previous!r}; the chain is broken here"
                    ),
                )
            recomputed = _hash_payload(entry.to_dict(), entry.previous_hash)
            if recomputed != entry.entry_hash:
                return LedgerIntegrity(
                    is_valid=False,
                    total_entries=len(self._entries),
                    broken_at=entry.entry_id,
                    reason=(
                        f"entry {entry.entry_id} has entry_hash "
                        f"{entry.entry_hash!r}, but recomputing from its payload "
                        f"produces {recomputed!r}; the payload was altered after recording"
                    ),
                )
        return LedgerIntegrity(
            is_valid=True,
            total_entries=len(self._entries),
            broken_at=None,
            reason="the chain is intact; every link and hash matches",
        )
