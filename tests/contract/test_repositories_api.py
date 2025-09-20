"""Contract tests for GET /repositories API endpoint."""

import pytest
from fastapi.testclient import TestClient
from repo_kgraph.api.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.contract
def test_repositories_endpoint_exists(client):
    response = client.get("/repositories")
    assert response.status_code != 404

@pytest.mark.contract
def test_repositories_list_format(client):
    response = client.get("/repositories")
    assert response.status_code == 200
    
    data = response.json()
    assert "repositories" in data
    assert "total_count" in data
    assert isinstance(data["repositories"], list)

@pytest.mark.contract
def test_repositories_pagination(client):
    response = client.get("/repositories?limit=5&offset=0")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data["repositories"]) <= 5