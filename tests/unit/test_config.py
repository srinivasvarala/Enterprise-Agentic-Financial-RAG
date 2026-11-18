import pytest

from finrag.core.config import Environment, Settings


def test_settings_read_prefixed_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINRAG_ENV", "prod")
    monkeypatch.setenv("FINRAG_QDRANT_URL", "http://qdrant:6333")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.env is Environment.PROD
    assert settings.qdrant_url == "http://qdrant:6333"


def test_secrets_are_not_leaked_in_repr() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert "finrag-secret" not in repr(settings)
