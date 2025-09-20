"""Contract tests for GET /health API endpoint."""

import pytest
from fastapi.testclient import TestClient
from repo_kgraph.api.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.contract
def test_health_endpoint_exists(client):
    response = client.get("/health")
    assert response.status_code != 404

@pytest.mark.contract
def test_health_response_format(client):
    response = client.get("/health")
    assert response.status_code in [200, 503]  # Healthy or unhealthy
    
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert data["status"] in ["healthy", "degraded", "unhealthy"]

@pytest.mark.contract
def test_health_components(client):
    response = client.get("/health")
    data = response.json()
    
    if "components" in data:
        components = data["components"]
        # Should have database and vector_store components
        for component in ["database", "vector_store"]:
            if component in components:
                assert "status" in components[component]