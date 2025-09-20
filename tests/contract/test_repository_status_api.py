"""Contract tests for GET /repositories/{id}/status API endpoint."""

import pytest
from fastapi.testclient import TestClient
from repo_kgraph.api.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_repository_id():
    return "550e8400-e29b-41d4-a716-446655440000"

@pytest.mark.contract
def test_repository_status_endpoint_exists(client, sample_repository_id):
    response = client.get(f"/repositories/{sample_repository_id}/status")
    assert response.status_code != 404

@pytest.mark.contract
def test_repository_status_format(client, sample_repository_id):
    response = client.get(f"/repositories/{sample_repository_id}/status")
    # Will return 404 until implemented, but structure should be defined
    if response.status_code == 200:
        data = response.json()
        assert "repository_id" in data
        assert "status" in data
        assert "statistics" in data