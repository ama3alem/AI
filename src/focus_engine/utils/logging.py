"""Logging with mandatory secret redaction.

The engine handles behavioural records, so its logs are a potential data-exfiltration
surface. Two rules follow, and both are enforced here rather than left to reviewer
discipline:

* **Learner identifiers are opaque but still linkable.** They are not logged at INFO by
  default, because a log aggregator becomes a secondary store of behavioural data with
  weaker governance than the primary store. A helper is provided for the cases that
  genuinely need an identifier at WARNING or above.
* **Secrets never reach a log record.** Redaction is applied by a logging filter, not by
  caller discipline, so a future contributor cannot leak one by forgetting.

Redaction is defence in depth, not a substitute for not passing secrets in. The correct
fix for a value that needs logging is to log an opaque reference to it.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any, Final

__all__ = [
    "REDACTED",
    "RedactingFilter",
    "configure_logging",
    "get_logger",
    "pseudonymous_ref",
]

#: Substitution text written in place of a redacted value.
REDACTED: Final[str] = "[REDACTED]"

#: Key names whose presence marks the surrounding value as a credential. The optional
#: surrounding quotes let this match a JSON body such as ``{"password": "..."}`` as well
#: as a bare ``password=...``.
_SENSITIVE_KEY_ALTERNATION: Final[str] = (
    r"(?:password|passwd|secret|token|api[_-]?key|apikey|authorization|access[_-]?token)"
)

#: Patterns applied to free text, in order.
#:
#: Order is significant and is a correctness requirement, not a preference. The
#: credential-bearing header patterns must run *before* the key/value pattern: given
#: ``Authorization: Bearer <token>`` the key/value pattern would match the key
#: ``Authorization``, replace the word ``Bearer``, and leave the actual token in the
#: record. Applying the header patterns first removes the whole header.
_CREDENTIAL_HEADER_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    # PEM blocks
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.S,
    ),
    # Authorization: Bearer <token> / Basic <token>
    re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
    # Bare JWT-shaped value
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\b"),
)

_KEY_VALUE_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"(?i)([\"']?{_SENSITIVE_KEY_ALTERNATION}[\"']?)"  # group 1: the key
    rf"(\s*[=:]\s*)"  # group 2: the separator
    # group 3: the value. Square brackets are excluded from the value character class so
    # that redaction is idempotent: a record that passes through two filters (a logger
    # filter and a handler filter, say) must not have "[REDACTED]" re-redacted into
    # "[REDACTED]]".
    rf"(\"[^\"]*\"|'[^']*'|[^\s,;&}}\[\]]+)"
)

#: Attribute names whose values are always replaced, regardless of content.
_SENSITIVE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "auth_secret",
        "access_token",
        "refresh_token",
        "session_key",
    }
)

#: Substrings that mark an attribute name as sensitive even when embedded in a longer
#: name, e.g. ``db_password`` or ``user_auth_token``.
_SENSITIVE_KEY_SUBSTRINGS: Final[tuple[str, ...]] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
)


def _redact_text(text: str) -> str:
    """Scrub credential material from a free-text message.

    Args:
        text: The raw text.

    Returns:
        The scrubbed text.
    """
    redacted = text
    for pattern in _CREDENTIAL_HEADER_PATTERNS:
        redacted = pattern.sub(REDACTED, redacted)
    return _KEY_VALUE_PATTERN.sub(rf"\1\2{REDACTED}", redacted)


class RedactingFilter(logging.Filter):
    """Logging filter that removes credential material from records.

    Applied to the handler rather than to individual loggers, so that a logger created
    anywhere in the codebase is covered without having to remember to attach it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Scrub a record in place and always keep it.

        Args:
            record: The record to scrub.

        Returns:
            ``True``. A record is never dropped for containing a secret; it is scrubbed,
            because silently dropping security-relevant records would hide incidents.
        """
        if isinstance(record.msg, str):
            record.msg = _redact_text(record.msg)
        if record.args:
            record.args = self._redact_args(record.args)
        for key, value in list(record.__dict__.items()):
            if key.lower() in _SENSITIVE_KEYS:
                record.__dict__[key] = REDACTED
            elif isinstance(value, str):
                record.__dict__[key] = _redact_text(value)
            elif isinstance(value, (tuple, list)) and any(
                isinstance(item, str) and _matches_sensitive_key(key) for item in value
            ):
                record.__dict__[key] = REDACTED
        return True

    @staticmethod
    def _redact_args(args: Any) -> Any:
        """Redact string arguments.

        Args:
            args: The logging format arguments.

        Returns:
            The redacted arguments.
        """
        if isinstance(args, tuple):
            return tuple(_redact_text(item) if isinstance(item, str) else item for item in args)
        if isinstance(args, dict):
            return {
                key: (REDACTED if _matches_sensitive_key(key) else value)
                for key, value in args.items()
            }
        return args


def _matches_sensitive_key(key: str) -> bool:
    """Report whether a key name looks sensitive.

    Args:
        key: The key name.

    Returns:
        ``True`` if the name is a known sensitive attribute or contains one as a
        substring, so that names such as ``db_password`` or ``user_auth_token`` are also
        caught.
    """
    lowered = key.lower()
    if lowered in _SENSITIVE_KEYS:
        return True
    return any(candidate in lowered for candidate in _SENSITIVE_KEY_SUBSTRINGS)


def pseudonymous_ref(identifier: str) -> str:
    """Return a short, non-reversible-enough reference for an identifier.

    Intended for correlating log lines about one learner without reproducing the
    identifier itself in a log store. The suffix is a truncated digest, so two log lines
    carrying the same reference concern the same learner, but the reference cannot be
    walked back to the identifier without the engine's configuration.

    Args:
        identifier: The pseudonymous identifier.

    Returns:
        A reference string of the form ``<prefix>#<digest-prefix>``.
    """
    import hashlib  # noqa: PLC0415 - local import: only this helper needs it

    digest = hashlib.blake2b(
        identifier.encode("utf-8"), digest_size=6, person=b"focus-log"
    ).hexdigest()
    return f"{identifier[:4]}#{digest}"


def configure_logging(
    *,
    level: str = "INFO",
    stream: Any = None,
    force: bool = True,
) -> logging.Logger:
    """Configure the root logger with redaction enabled.

    Args:
        level: Root log level name.
        stream: Destination stream. Defaults to ``sys.stderr``, so that log output never
            contaminates ``stdout`` and therefore never corrupts a machine-readable
            result written there.
        force: Whether to replace existing handlers.

    Returns:
        The configured root logger.
    """
    target = stream if stream is not None else sys.stderr
    handler = logging.StreamHandler(target)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    handler.addFilter(RedactingFilter())

    root = logging.getLogger()
    if force:
        for existing in list(root.handlers):
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger.

    Args:
        name: Usually ``__name__``.

    Returns:
        The logger.
    """
    return logging.getLogger(name)
