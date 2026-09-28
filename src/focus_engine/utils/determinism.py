"""Determinism and reproducibility helpers.

Reproducibility is a hard requirement, not an aspiration: the same inputs, seed, and
version coordinates must produce equivalent results within a documented tolerance. Two
mechanisms support that here.

* **Seed derivation.** Randomness is never drawn from an unseeded global generator.
  Each unit of work derives its own seed deterministically from a master seed and a
  label, so that adding a new component does not shift the random stream of an existing
  one. A change in one component's consumption must not silently alter another's data.
* **Stable hashing.** Content digests use BLAKE2b with an explicit encoding, so a digest
  is a property of the content rather than of a process's hash randomisation.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

__all__ = [
    "DIGEST_SIZE_BYTES",
    "canonical_json",
    "content_digest",
    "derive_seed",
    "digest_iterable",
    "library_versions",
    "seed_everything",
    "stable_hash_int",
]

#: Packages whose version can move a numeric result. A reproducibility stamp that omits
#: these is incomplete: the same seed and the same data will not necessarily give the same
#: fit across a scikit-learn upgrade, and the difference would be silent.
_VERSIONED_PACKAGES: Final[tuple[str, ...]] = ("joblib", "numpy", "pandas", "scikit-learn", "scipy")

#: Digest width for content hashing. 16 bytes is ample for artifact identity and keeps
#: manifest files small.
DIGEST_SIZE_BYTES: Final[int] = 16

#: Domain-separation prefix for seed derivation. Prevents a seed intended for one
#: purpose from colliding with a seed intended for another.
_SEED_NAMESPACE: Final[str] = "focus_engine/seed/v1"


def stable_hash_int(*parts: str, bits: int = 64) -> int:
    """Derive a stable integer from string parts.

    Args:
        parts: Ordered, meaningful components. Order is significant.
        bits: Width of the resulting integer.

    Returns:
        A deterministic non-negative integer.

    Raises:
        ValueError: If ``bits`` is not positive.
    """
    if bits <= 0:
        raise ValueError(f"bits must be positive; got {bits}")
    joined = "\x1f".join(parts).encode("utf-8")
    digest = hashlib.blake2b(joined, digest_size=(bits + 7) // 8).digest()
    return int.from_bytes(digest, "big") % (1 << bits)


def derive_seed(master_seed: int, *labels: str) -> int:
    """Derive a child seed from a master seed and a label path.

    Args:
        master_seed: The run's master seed.
        labels: Ordered labels identifying the unit of work, e.g.
            ``("simulator", learner_id, "sessions")``.

    Returns:
        A child seed in ``[0, 2**32)``.

    Raises:
        ValueError: If ``master_seed`` is negative.
    """
    if master_seed < 0:
        raise ValueError(f"master_seed must be non-negative; got {master_seed}")
    return stable_hash_int(_SEED_NAMESPACE, str(master_seed), *labels, bits=32)


def seed_everything(master_seed: int) -> np.random.Generator:
    """Seed the global generators used by the engine and return a private generator.

    Both the :mod:`random` module and NumPy's legacy global generator are seeded,
    because third-party numeric code may draw from either. A dedicated
    :class:`numpy.random.Generator` is returned as well, so callers that care about
    isolation should use the returned object rather than the global state.

    Args:
        master_seed: The run's master seed.

    Returns:
        An independently seeded NumPy generator.

    Raises:
        ValueError: If ``master_seed`` is negative.
    """
    if master_seed < 0:
        raise ValueError(f"master_seed must be non-negative; got {master_seed}")
    random.seed(master_seed)
    np.random.seed(master_seed % (2**32))
    return np.random.default_rng(master_seed)


def library_versions() -> dict[str, str]:
    """Report the installed version of every package that can move a numeric result.

    Reproducibility is a claim about a specific environment, not about a seed in the
    abstract. Two runs sharing a seed and a dataset still disagree if scikit-learn, NumPy,
    or SciPy changed in between, and nothing about the fitted parameters would reveal that
    the difference came from the environment rather than from the data. Recording these
    versions is what makes a reproducibility claim falsifiable.

    A package that is not installed is reported as ``"absent"`` rather than omitted, so
    the absence is visible in a stored record instead of being indistinguishable from a
    package that was simply never considered.

    Returns:
        A mapping of distribution name to installed version, in a stable order.
    """
    from importlib.metadata import PackageNotFoundError, version

    resolved: dict[str, str] = {}
    for distribution in _VERSIONED_PACKAGES:
        try:
            resolved[distribution] = version(distribution)
        except PackageNotFoundError:  # pragma: no cover - depends on the environment
            resolved[distribution] = "absent"
    return resolved


def canonical_json(payload: Any) -> str:
    """Serialise a payload to a canonical, order-independent JSON string.

    Keys are sorted and separators are fixed, so two structurally equal payloads always
    produce byte-identical output. This is what allows a content digest to identify
    content rather than serialisation order.

    Args:
        payload: A JSON-serialisable object.

    Returns:
        The canonical JSON string.

    Raises:
        TypeError: If the payload contains a non-serialisable value.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_fallback_encoder,
    )


def _fallback_encoder(value: Any) -> Any:
    """Encode values that ``json`` cannot handle natively.

    Handles the types that actually appear in this engine's records: paths, enums,
    datetimes, NumPy scalars and arrays, and sets.

    Args:
        value: The unsupported value.

    Returns:
        A JSON-encodable representation.

    Raises:
        TypeError: If the value has no defined encoding.
    """
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"unsupported type for canonical encoding: {type(value)!r}")


def content_digest(payload: Any) -> str:
    """Compute a short, stable content digest for a payload.

    Args:
        payload: A JSON-serialisable object.

    Returns:
        A ``blake2b`` hex digest of ``DIGEST_SIZE_BYTES`` bytes.
    """
    canonical = canonical_json(payload)
    digest = hashlib.blake2b(
        canonical.encode("utf-8"),
        digest_size=DIGEST_SIZE_BYTES,
        person=b"focus-engine",
    )
    return digest.hexdigest()


def digest_iterable(items: Iterable[str]) -> str:
    """Compute a stable digest over an ordered iterable of strings.

    Args:
        items: Ordered items. Order is significant.

    Returns:
        A hex digest.
    """
    hasher = hashlib.blake2b(digest_size=DIGEST_SIZE_BYTES, person=b"focus-engine")
    for item in items:
        hasher.update(item.encode("utf-8"))
        hasher.update(b"\x1e")
    return hasher.hexdigest()
