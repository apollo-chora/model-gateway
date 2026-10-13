import pytest
from blacksheep.testing import TestClient

from model_gateway import health
from model_gateway.app import create_app


@pytest.fixture
async def client():
    app = create_app()
    await app.start()
    try:
        yield TestClient(app)
    finally:
        await app.stop()


async def test_healthz_returns_ok(client):
    response = await client.get("/healthz")
    assert response.status == 200
    assert await response.text() == "ok"


async def test_readyz_returns_ready(client):
    response = await client.get("/readyz")
    assert response.status == 200
    assert await response.json() == {"status": "ready", "checks": {}}


async def test_readyz_returns_503_when_a_check_fails(client):
    health._checks["failing"] = lambda: False
    try:
        response = await client.get("/readyz")
        assert response.status == 503
        assert await response.json() == {
            "status": "not_ready",
            "checks": {"failing": "fail"},
        }
    finally:
        health._checks.clear()


async def test_readyz_returns_503_when_a_check_raises(client):
    def broken():
        raise RuntimeError("boom")

    health._checks["broken"] = broken
    try:
        response = await client.get("/readyz")
        assert response.status == 503
        assert await response.json() == {
            "status": "not_ready",
            "checks": {"broken": "error"},
        }
    finally:
        health._checks.clear()
