"""Injectable time source.

Every component that needs the current time takes a :class:`Clock` rather than calling
:func:`datetime.now` directly. This is what makes temporal behaviour testable: a test
can advance a :class:`FixedClock` by eleven minutes and assert on state transitions,
cooldown expiry, and window boundaries, without sleeping and without flakiness.

Direct calls to ``datetime.now`` inside intelligence code are therefore a review defect,
not a style preference — they make temporal logic depend on wall-clock timing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "FixedClock", "SystemClock"]


@runtime_checkable
class Clock(Protocol):
    """A source of the current instant.

    Implementations must return a timezone-aware UTC datetime.
    """

    def now(self) -> datetime:
        """Return the current instant.

        Returns:
            The current instant, timezone-aware and normalised to UTC.
        """
        ...


class SystemClock:
    """The real clock, reading the host system time.

    Used in serving paths. Normalised to UTC on every read so that a host configured to
    another zone cannot inject a non-UTC timestamp into the event stream.
    """

    __slots__ = ()

    def now(self) -> datetime:
        """Return the current host instant in UTC.

        Returns:
            The current instant, timezone-aware and in UTC.
        """
        return datetime.now(UTC)

    def __repr__(self) -> str:
        """Return a debugging representation.

        Returns:
            A short representation identifying this as the system clock.
        """
        return "SystemClock()"


class FixedClock:
    """A manually advanced clock for deterministic tests.

    The clock starts at a supplied instant and moves only when :meth:`advance` or
    :meth:`set_time` is called. Because nothing advances it implicitly, a test that
    depends on elapsed time must say so, and a passing test means what it claims.
    """

    __slots__ = ("_now",)

    def __init__(self, start: datetime) -> None:
        """Initialise the clock.

        Args:
            start: The initial instant. Must be timezone-aware.

        Raises:
            ValueError: If ``start`` is naive.
        """
        if start.tzinfo is None or start.tzinfo.utcoffset(start) is None:
            raise ValueError("FixedClock requires a timezone-aware start instant")
        self._now = start.astimezone(UTC)

    def now(self) -> datetime:
        """Return the current simulated instant.

        Returns:
            The simulated instant, in UTC.
        """
        return self._now

    def advance(self, delta: timedelta) -> datetime:
        """Move the clock forward.

        Args:
            delta: Duration to advance. Must not be negative, so that a test cannot
                silently travel backwards and invalidate a monotonicity assumption.

        Returns:
            The new current instant.

        Raises:
            ValueError: If ``delta`` is negative.
        """
        if delta < timedelta(0):
            raise ValueError(f"cannot advance a FixedClock backwards; got {delta}")
        self._now = self._now + delta
        return self._now

    def set_time(self, moment: datetime) -> datetime:
        """Jump the clock to an absolute instant.

        Args:
            moment: The new instant. Must be timezone-aware.

        Returns:
            The new current instant.

        Raises:
            ValueError: If ``moment`` is naive.
        """
        if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
            raise ValueError("FixedClock.set_time requires a timezone-aware instant")
        self._now = moment.astimezone(UTC)
        return self._now

    def __repr__(self) -> str:
        """Return a debugging representation including the simulated instant.

        Returns:
            A short representation identifying the current simulated time.
        """
        return f"FixedClock({self._now.isoformat()})"
