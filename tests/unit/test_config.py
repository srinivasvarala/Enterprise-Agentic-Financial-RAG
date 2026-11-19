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


def test_an_empty_eval_cost_cap_means_no_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINRAG_EVAL_COST_CAP_USD", "")

    assert Settings(_env_file=None).eval_cost_cap_usd is None  # type: ignore[call-arg]

    monkeypatch.setenv("FINRAG_EVAL_COST_CAP_USD", "12.5")

    assert Settings(_env_file=None).eval_cost_cap_usd == 12.5  # type: ignore[call-arg]
