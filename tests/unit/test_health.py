from fastapi.testclient import TestClient

from finrag import __version__
from finrag.api.app import create_app
from finrag.core.config import Environment, Settings


def test_health_reports_ok_version_and_env() -> None:
    client = TestClient(create_app(Settings(env=Environment.TEST)))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__, "env": "test"}
