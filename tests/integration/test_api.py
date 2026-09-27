"""Integration tests for FastAPI endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport

from backend.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_root_endpoint(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "ThermaTwin" in response.json()["name"]


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires live PostgreSQL database — run via docker-compose")
async def test_wells_crud(client):
    # Create
    create_resp = await client.post("/api/v1/wells", json={"well_name": "TEST-WELL-001"})
    assert create_resp.status_code == 201
    well_id = create_resp.json()["well_id"]

    # Read
    get_resp = await client.get(f"/api/v1/wells/{well_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["well_name"] == "TEST-WELL-001"

    # Update
    update_resp = await client.put(f"/api/v1/wells/{well_id}", json={"well_name": "TEST-WELL-002"})
    assert update_resp.status_code == 200
    assert update_resp.json()["well_name"] == "TEST-WELL-002"

    # Delete
    delete_resp = await client.delete(f"/api/v1/wells/{well_id}")
    assert delete_resp.status_code == 204

    # Verify deleted
    get_resp = await client.get(f"/api/v1/wells/{well_id}")
    assert get_resp.status_code == 404
