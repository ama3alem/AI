"""Unit tests for :mod:`focus_engine.configuration.settings`.

The security-relevant tests here are
:func:`test_production_refuses_to_start_without_a_secret` and
:func:`test_secret_never_appears_in_repr_or_summary`. They encode the fail-closed
posture: an unauthenticated behavioural-data endpoint must not be reachable by
accident, and a secret must not be recoverable from a log line or a settings dump.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from focus_engine.configuration.settings import (
    AUTH_SECRET_ENV_VAR,
    AppSettings,
    RuntimeEnvironment,
    RuntimeSettings,
    SecuritySettings,
    SettingsPaths,
    default_project_root,
    load_settings,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_auth_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no ambient auth secret leaks into a test.

    The host environment may define the variable; a test asserting on the unset case
    would otherwise be non-deterministic.
    """
    monkeypatch.delenv(AUTH_SECRET_ENV_VAR, raising=False)


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------


def test_project_root_resolves_to_a_directory_containing_pyproject() -> None:
    root = default_project_root()
    assert root.is_dir()
    assert (root / "pyproject.toml").is_file()


def test_paths_default_under_the_project_root() -> None:
    paths = SettingsPaths()
    assert paths.raw_dir() == paths.project_root / "data" / "raw"
    assert paths.processed_dir() == paths.project_root / "data" / "processed"
    assert paths.synthetic_dir() == paths.project_root / "data" / "synthetic"
    assert paths.resolved_reports_dir() == paths.project_root / "evaluation" / "reports"


def test_paths_are_not_dependent_on_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing directory must not change where the engine reads or writes."""
    paths = SettingsPaths()
    expected = paths.resolved_artifacts_dir()
    monkeypatch.chdir(tmp_path)
    assert paths.resolved_artifacts_dir() == expected


def test_explicit_absolute_path_overrides_are_honoured(tmp_path: Path) -> None:
    paths = SettingsPaths(data_dir=tmp_path / "d", artifacts_dir=tmp_path / "a")
    assert paths.raw_dir() == tmp_path / "d" / "raw"
    assert paths.resolved_artifacts_dir() == tmp_path / "a"


def test_relative_paths_are_rejected(tmp_path: Path) -> None:
    """A relative path would make behaviour depend on the working directory."""
    with pytest.raises(ValidationError, match="must be absolute"):
        SettingsPaths(data_dir=Path("relative/path"))


def test_ensure_directories_creates_the_full_tree(tmp_path: Path) -> None:
    paths = SettingsPaths(
        data_dir=tmp_path / "d", artifacts_dir=tmp_path / "a", reports_dir=tmp_path / "r"
    )
    paths.ensure_directories()
    for expected in (
        tmp_path / "d" / "raw",
        tmp_path / "d" / "processed",
        tmp_path / "d" / "synthetic",
        tmp_path / "a",
        tmp_path / "r",
    ):
        assert expected.is_dir()


# --------------------------------------------------------------------------------------
# Runtime settings
# --------------------------------------------------------------------------------------


def test_runtime_defaults_to_development_not_production() -> None:
    assert RuntimeSettings().environment is RuntimeEnvironment.DEVELOPMENT


def test_determinism_is_enabled_by_default() -> None:
    """Reproducibility is cheap, so it is the default rather than an opt-in."""
    assert RuntimeSettings().deterministic is True


@pytest.mark.parametrize("value", ["info", "Info", "INFO", " debug "])
def test_log_level_is_normalised(value: str) -> None:
    assert RuntimeSettings(log_level=value).log_level == value.strip().upper()


def test_invalid_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError, match="log_level must be one of"):
        RuntimeSettings(log_level="VERBOSE")


# --------------------------------------------------------------------------------------
# Security settings
# --------------------------------------------------------------------------------------


def test_auth_is_not_required_in_development() -> None:
    assert SecuritySettings().auth_required is False


def test_rate_limits_are_positive() -> None:
    settings = SecuritySettings()
    assert settings.rate_limit_requests >= 1
    assert settings.rate_limit_window_seconds >= 1
    assert settings.max_events_per_request >= 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rate_limit_requests": 0},
        {"rate_limit_window_seconds": 0},
        {"max_events_per_request": 0},
    ],
)
def test_non_positive_rate_limits_are_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        SecuritySettings(**kwargs)


def test_wildcard_cors_origin_is_rejected() -> None:
    """A wildcard origin on an endpoint carrying behavioural records is a real risk."""
    with pytest.raises(ValidationError, match="must not contain"):
        SecuritySettings(cors_allow_origins=("*",))


def test_explicit_cors_origins_are_accepted() -> None:
    settings = SecuritySettings(cors_allow_origins=("https://app.example.org",))
    assert settings.cors_allow_origins == ("https://app.example.org",)


def test_cors_defaults_to_no_browser_origins() -> None:
    assert SecuritySettings().cors_allow_origins == ()


# --------------------------------------------------------------------------------------
# The production fail-closed gate
# --------------------------------------------------------------------------------------


def test_production_refuses_to_start_without_a_secret() -> None:
    settings = AppSettings(runtime=RuntimeSettings(environment=RuntimeEnvironment.PRODUCTION))
    with pytest.raises(ValueError, match="must be set to run in production"):
        settings.validate_for_runtime()


def test_production_refuses_to_start_with_auth_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A secret alone is not enough: authentication must also be switched on."""
    monkeypatch.setenv(AUTH_SECRET_ENV_VAR, "a-sufficiently-long-secret-value")
    monkeypatch.setenv("FOCUS_RUNTIME_ENVIRONMENT", "production")
    settings = AppSettings.from_env()
    assert settings.runtime.environment is RuntimeEnvironment.PRODUCTION
    settings = settings.model_copy(update={"security": SecuritySettings(auth_required=False)})
    with pytest.raises(ValueError, match="auth_required must be true"):
        settings.validate_for_runtime()


def test_production_starts_when_secret_present_and_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AUTH_SECRET_ENV_VAR, "a-sufficiently-long-secret-value")
    monkeypatch.setenv("FOCUS_RUNTIME_ENVIRONMENT", "production")
    monkeypatch.setenv("FOCUS_SEC_AUTH_REQUIRED", "true")
    settings = AppSettings.from_env()
    assert settings.validate_for_runtime() is settings


def test_development_does_not_require_a_secret() -> None:
    settings = AppSettings()
    assert settings.validate_for_runtime() is settings


def test_blank_secret_is_treated_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace must not satisfy the production gate."""
    monkeypatch.setenv(AUTH_SECRET_ENV_VAR, "   ")
    assert AppSettings.from_env().auth_secret is None


# --------------------------------------------------------------------------------------
# Secret containment
# --------------------------------------------------------------------------------------


def test_secret_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(AUTH_SECRET_ENV_VAR, "super-secret-token-value")
    settings = AppSettings.from_env()
    assert settings.auth_secret is not None
    assert settings.auth_secret.get_secret_value() == "super-secret-token-value"


def test_secret_never_appears_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pydantic masks SecretStr in repr, so a debug dump cannot leak it."""
    monkeypatch.setenv(AUTH_SECRET_ENV_VAR, "super-secret-token-value")
    assert "super-secret-token-value" not in repr(AppSettings.from_env())


def test_secret_never_appears_in_redacted_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(AUTH_SECRET_ENV_VAR, "super-secret-token-value")
    summary = AppSettings.from_env().redacted_summary()
    assert "super-secret-token-value" not in summary
    assert "auth_configured=True" in summary


def test_redacted_summary_reports_absence_distinctly(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = AppSettings.from_env().redacted_summary()
    assert "auth_configured=False" in summary


def test_load_settings_validates_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOCUS_RUNTIME_ENVIRONMENT", "production")
    with pytest.raises(ValueError):
        load_settings()


def test_load_settings_can_skip_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOCUS_RUNTIME_ENVIRONMENT", "production")
    assert load_settings(validate=False).runtime.environment is RuntimeEnvironment.PRODUCTION


def test_settings_read_the_documented_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prefix prevents collision with unrelated host variables."""
    monkeypatch.setenv("FOCUS_RUNTIME_LOG_LEVEL", "DEBUG")
    assert RuntimeSettings().log_level == "DEBUG"


def test_nested_settings_receive_their_own_env_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard.

    ``BaseSettings`` does not propagate environment variables into a nested model built
    by a ``default_factory``. Without explicit construction in ``from_env``, these
    variables would be ignored while appearing to be supported.
    """
    monkeypatch.setenv("FOCUS_RUNTIME_ENVIRONMENT", "production")
    monkeypatch.setenv("FOCUS_SEC_RATE_LIMIT_REQUESTS", "7")
    settings = AppSettings.from_env()
    assert settings.runtime.environment is RuntimeEnvironment.PRODUCTION
    assert settings.security.rate_limit_requests == 7


def test_unprefixed_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert RuntimeSettings().log_level == "INFO"


def test_no_credential_is_written_into_any_settings_field() -> None:
    """Static check: the default settings object must not materialise a secret."""
    settings = AppSettings()
    assert settings.auth_secret is None
    assert os.environ.get(AUTH_SECRET_ENV_VAR) in (None, "")
