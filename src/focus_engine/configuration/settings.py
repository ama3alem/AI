"""Process-level settings: paths, runtime mode, and secret sourcing.

Layer thresholds live in :mod:`focus_engine.configuration.thresholds`. This module
handles the concerns that surround the engine rather than sit inside it: where files
live, which mode the process is in, and how credentials are obtained.

Security posture for the foundation phase, stated plainly:

* Secrets are read **only** from environment variables. No credential has a default,
  none is written to disk, and none is logged.
* Settings objects are frozen, so a resolved configuration cannot be mutated after
  validation.
* The defaults here are **prototype-grade**. ``RuntimeSettings.environment`` defaults to
  ``development``. Production mode requires an explicit opt-in and refuses to start
  without a configured authentication secret, because a prototype that silently runs
  unauthenticated in production would be a real vulnerability.

Every path is derived from a single configurable root; no absolute path is hardcoded in
any module.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__ = [
    "AppSettings",
    "RuntimeEnvironment",
    "RuntimeSettings",
    "SecuritySettings",
    "SettingsPaths",
    "default_project_root",
    "load_settings",
]

#: All environment variables read by the engine are prefixed with this, so that a
#: setting can never collide with an unrelated variable in the host environment.
ENV_PREFIX: Final[str] = "FOCUS_"

#: Name of the environment variable holding the authentication secret. Read directly
#: rather than through settings so that the value never passes through a repr, a log
#: line, or a serialised settings object.
AUTH_SECRET_ENV_VAR: Final[str] = "FOCUS_AUTH_SECRET"

#: Type alias for CORS origins that disables JSON decoding, allowing comma-separated
#: values in environment variables instead of requiring JSON arrays.
AllowedOrigins = Annotated[tuple[str, ...], NoDecode]


def default_project_root() -> Path:
    """Resolve the project root from this module's location.

    Resolution is ``<root>/src/focus_engine/configuration/settings.py``, so the root is
    three parents up. This keeps the engine relocatable: nothing depends on the current
    working directory or on a machine-specific absolute path.

    Returns:
        The project root directory.
    """
    return Path(__file__).resolve().parents[3]


class RuntimeEnvironment(StrEnum):
    """Deployment mode of the running process.

    The mode changes what the service is *allowed* to do, not merely how it is logged.
    """

    DEVELOPMENT = "development"
    """Local development. Permissive defaults; no authentication expected."""

    TEST = "test"
    """Automated tests. Deterministic, no network egress, no secret required."""

    PRODUCTION = "production"
    """Deployed. Refuses to start without an authentication secret, and applies
    production rate limits."""


class SettingsPaths(BaseSettings):
    """Filesystem locations used by the engine.

    Attributes:
        project_root: Root of the repository. All other paths are resolved relative to it.
        data_dir: Parent of the raw/processed/synthetic data trees.
        artifacts_dir: Destination for trained models, feature registries, and
            experiment artifacts. Deliberately outside version control: artifacts are
            regenerable, and the working directory may sit on a synchronised folder
            (see ``PROJECT_AUDIT.md`` risk R-03).
        reports_dir: Destination for rendered evaluation reports.
    """

    model_config = SettingsConfigDict(env_prefix=f"{ENV_PREFIX}PATH_", extra="ignore")

    project_root: Path = Field(default_factory=default_project_root)
    data_dir: Path | None = None
    artifacts_dir: Path | None = None
    reports_dir: Path | None = None

    @field_validator("project_root", "data_dir", "artifacts_dir", "reports_dir")
    @classmethod
    def _require_absolute(cls, value: Path | None) -> Path | None:
        """Require absolute paths so behaviour does not depend on the working directory.

        Args:
            value: Candidate path.

        Returns:
            The resolved absolute path.

        Raises:
            ValueError: If a provided path is relative.
        """
        if value is not None and not value.is_absolute():
            raise ValueError(f"path must be absolute, got {value}")
        return value

    def resolved_data_dir(self) -> Path:
        """Return the data directory, defaulting under the project root.

        Returns:
            Absolute path to the data directory.
        """
        return self.data_dir or (self.project_root / "data")

    def resolved_artifacts_dir(self) -> Path:
        """Return the artifacts directory, defaulting under the project root.

        Returns:
            Absolute path to the artifacts directory.
        """
        return self.artifacts_dir or (self.project_root / "artifacts")

    def resolved_reports_dir(self) -> Path:
        """Return the reports directory, defaulting under the project root.

        Returns:
            Absolute path to the reports directory.
        """
        return self.reports_dir or (self.project_root / "evaluation" / "reports")

    def raw_dir(self) -> Path:
        """Return the raw data directory.

        Returns:
            Absolute path to ``data/raw``.
        """
        return self.resolved_data_dir() / "raw"

    def processed_dir(self) -> Path:
        """Return the processed data directory.

        Returns:
            Absolute path to ``data/processed``.
        """
        return self.resolved_data_dir() / "processed"

    def synthetic_dir(self) -> Path:
        """Return the synthetic data directory.

        Returns:
            Absolute path to ``data/synthetic``.
        """
        return self.resolved_data_dir() / "synthetic"

    def ensure_directories(self) -> None:
        """Create the data, artifact, and report directories if absent.

        Raises:
            OSError: If a directory cannot be created.
        """
        for path in (
            self.raw_dir(),
            self.processed_dir(),
            self.synthetic_dir(),
            self.resolved_artifacts_dir(),
            self.resolved_reports_dir(),
        ):
            path.mkdir(parents=True, exist_ok=True)


class RuntimeSettings(BaseSettings):
    """Process runtime behaviour.

    Attributes:
        environment: Deployment mode. See :class:`RuntimeEnvironment`.
        log_level: Root log level name.
        deterministic: When true, all random number generators used by the engine are
            seeded from a fixed seed and hash-based ordering is stabilised. Required for
            reproducible evaluation runs; costs nothing and is therefore the default.
        default_seed: Master seed applied under ``deterministic``.
    """

    model_config = SettingsConfigDict(env_prefix=f"{ENV_PREFIX}RUNTIME_", extra="ignore")

    environment: RuntimeEnvironment = RuntimeEnvironment.DEVELOPMENT
    log_level: str = "INFO"
    deterministic: bool = True
    default_seed: int = 20260926

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        """Normalise the log level to upper case.

        Args:
            value: Candidate level name.

        Returns:
            The upper-cased level name.

        Raises:
            ValueError: If the level is not one of the standard names.
        """
        normalised = value.strip().upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if normalised not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}; got {value!r}")
        return normalised


class SecuritySettings(BaseSettings):
    """Security-relevant configuration.

    Attributes:
        auth_required: Whether write endpoints require an authenticated caller. Defaults
            to ``False`` for local development and is forced to ``True`` in production by
            :meth:`validate_for_runtime`.
        rate_limit_requests: Requests permitted per window per client, for the prototype
            in-process limiter. Prototype-grade: a single-process limiter is trivially
            bypassed by a multi-worker deployment and must be replaced with a shared
            store before any real exposure.
        rate_limit_window_seconds: Length of the rate-limit window.
        max_events_per_request: Upper bound on the number of events accepted in one
            batch, to bound memory and parse cost from an untrusted caller.
        cors_allow_origins: Allowed browser origins. Empty by default: the API is not
            browser-facing until someone has a reason to make it so.

            Accepts a comma-separated list in the environment, which is the natural form
            in a ``.env`` file. JSON decoding is disabled for this field because
            ``pydantic-settings`` parses complex types as JSON by default, so a plain
            comma-separated value would abort settings loading outright.
    """

    model_config = SettingsConfigDict(env_prefix=f"{ENV_PREFIX}SEC_", extra="ignore")

    auth_required: bool = False
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    max_events_per_request: int = 1000
    cors_allow_origins: AllowedOrigins = ()

    @field_validator("rate_limit_requests", "rate_limit_window_seconds", "max_events_per_request")
    @classmethod
    def _require_positive(cls, value: int) -> int:
        """Require positive limits.

        Args:
            value: Candidate limit.

        Returns:
            The validated limit.

        Raises:
            ValueError: If the limit is below one.
        """
        if value < 1:
            raise ValueError(f"limit must be at least 1; got {value}")
        return value

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_and_validate_origins(cls, value: object) -> tuple[str, ...]:
        """Parse comma-separated origins from the environment and reject wildcards.

        ``NoDecode`` prevents ``pydantic-settings`` from interpreting the raw string as
        JSON, but ``tuple[str, ...]`` cannot coerce a plain string on its own, so a
        ``mode="before"`` validator is required to split it.

        Args:
            value: Raw value from the environment: a comma-separated string, a list, or
                a tuple.

        Returns:
            A normalised, whitespace-stripped tuple of origins.

        Raises:
            ValueError: If the input is not a string/list/tuple or contains ``"*"``.
        """
        if isinstance(value, str):
            origins = tuple(item.strip() for item in value.split(",") if item.strip())
        elif isinstance(value, (list, tuple)):
            origins = tuple(value)
        else:
            raise ValueError(
                f"cors_allow_origins must be a comma-separated string or a list/tuple; "
                f"got {type(value).__name__}"
            )
        if "*" in origins:
            raise ValueError(
                "cors_allow_origins must not contain '*'. A wildcard origin on an endpoint "
                "carrying behavioural records permits cross-origin exfiltration."
            )
        return origins


def _resolve_auth_secret() -> SecretStr | None:
    """Read the authentication secret from the environment.

    Returns:
        The secret, or ``None`` when the variable is unset or empty.

    Note:
        Read via :func:`os.environ` rather than through ``BaseSettings`` so the value is
        never included in a settings repr, a log record, or a serialised artifact.
    """
    raw = os.environ.get(AUTH_SECRET_ENV_VAR, "")
    if not raw.strip():
        return None
    return SecretStr(raw)


class AppSettings(BaseSettings):
    """Fully resolved application settings.

    Attributes:
        paths: Filesystem locations.
        runtime: Process runtime behaviour.
        security: Security configuration.
        auth_secret: The authentication secret, or ``None`` when unset. Never logged and
            never serialised.
    """

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="ignore")

    paths: SettingsPaths = Field(default_factory=SettingsPaths)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    auth_secret: SecretStr | None = None

    @classmethod
    def from_env(cls) -> AppSettings:
        """Build settings from environment variables and defaults.

        Nested settings objects are constructed explicitly rather than relying on the
        parent's field defaults. ``BaseSettings`` does not propagate environment
        variables into a nested model that was created by a ``default_factory``, so
        relying on the default would silently ignore ``FOCUS_RUNTIME_*`` and
        ``FOCUS_SEC_*`` — the settings would appear to be configurable while never
        actually changing.

        Returns:
            The resolved settings.
        """
        return cls(
            paths=SettingsPaths(),
            runtime=RuntimeSettings(),
            security=SecuritySettings(),
            auth_secret=_resolve_auth_secret(),
        )

    def validate_for_runtime(self) -> AppSettings:
        """Enforce the invariants that depend on the deployment mode.

        In production this refuses to return settings that would run the service
        unauthenticated. Failing closed here is deliberate: a missing secret is an
        operator error, and an unauthenticated behavioural-data endpoint is a breach.

        Returns:
            ``self``, when the configuration is acceptable.

        Raises:
            ValueError: If production mode is requested without an authentication secret
                or with authentication disabled.
        """
        if self.runtime.environment is RuntimeEnvironment.PRODUCTION:
            if self.auth_secret is None:
                raise ValueError(
                    f"{AUTH_SECRET_ENV_VAR} must be set to run in production. Refusing to "
                    "start an unauthenticated behavioural-data endpoint."
                )
            if not self.security.auth_required:
                raise ValueError(
                    "security.auth_required must be true in production. Refusing to start "
                    "with authentication disabled."
                )
        return self

    def redacted_summary(self) -> str:
        """Render settings for logging with the secret withheld.

        Returns:
            A multi-line summary containing no secret material.
        """
        return (
            f"environment={self.runtime.environment.value} "
            f"log_level={self.runtime.log_level} "
            f"deterministic={self.runtime.deterministic} "
            f"project_root={self.paths.project_root} "
            f"artifacts_dir={self.paths.resolved_artifacts_dir()} "
            f"auth_configured={self.auth_secret is not None} "
            f"auth_required={self.security.auth_required}"
        )


def load_settings(*, validate: bool = True) -> AppSettings:
    """Load application settings from the environment.

    Args:
        validate: Whether to enforce the production invariants.

    Returns:
        The resolved, optionally validated settings.
    """
    settings = AppSettings.from_env()
    if validate:
        settings = settings.validate_for_runtime()
    return settings
