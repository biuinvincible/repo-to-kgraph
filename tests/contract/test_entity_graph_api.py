"""
Contract tests for GET /graph/{entity_id} API endpoint.

These tests define the expected behavior of the entity graph exploration API
and MUST FAIL initially (TDD approach).
"""

import pytest
from fastapi.testclient import TestClient

from repo_kgraph.api.main import app


@pytest.fixture
def client():
    """Create test client for API testing."""
    return TestClient(app)


@pytest.fixture
def sample_entity_id():
    """Sample entity ID for testing."""
    return "770e8400-e29b-41d4-a716-446655440002"


@pytest.mark.contract
def test_entity_graph_endpoint_exists(client, sample_entity_id):
    """Test that the entity graph endpoint exists and accepts GET requests."""
    response = client.get(f"/graph/{sample_entity_id}")
    assert response.status_code != 404, f"GET /graph/{sample_entity_id} endpoint should exist"


@pytest.mark.contract
def test_entity_graph_valid_request(client, sample_entity_id):
    """Test retrieving entity graph with valid entity ID."""
    response = client.get(f"/graph/{sample_entity_id}")

    # Should return 200 for successful request
    assert response.status_code == 200

    response_data = response.json()
    assert "entity_id" in response_data
    assert "entity" in response_data
    assert "relationships" in response_data

    # Validate entity_id matches request
    assert response_data["entity_id"] == sample_entity_id

    # Entity should have required fields
    entity = response_data["entity"]
    required_fields = ["id", "entity_type", "name", "file_path", "language"]
    for field in required_fields:
        assert field in entity

    # Relationships should be a list
    assert isinstance(response_data["relationships"], list)


@pytest.mark.contract
def test_entity_graph_with_depth_parameter(client, sample_entity_id):
    """Test entity graph with depth parameter."""
    depth = 3
    response = client.get(f"/graph/{sample_entity_id}?depth={depth}")

    assert response.status_code == 200
    response_data = response.json()

    # Should include graph_metrics with depth info
    if "graph_metrics" in response_data:
        metrics = response_data["graph_metrics"]
        assert "max_depth" in metrics
        assert metrics["max_depth"] <= depth


@pytest.mark.contract
def test_entity_graph_entity_not_found(client):
    """Test error handling for non-existent entity."""
    non_existent_id = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"/graph/{non_existent_id}")

    # Should return 404
    assert response.status_code == 404

    response_data = response.json()
    assert "error" in response_data
    assert "message" in response_data
    assert "not found" in response_data["message"].lower()


@pytest.mark.contract
def test_entity_graph_invalid_uuid(client):
    """Test error handling for invalid UUID format."""
    invalid_id = "not-a-valid-uuid"
    response = client.get(f"/graph/{invalid_id}")

    # Should return 400 for invalid UUID
    assert response.status_code == 400


@pytest.mark.contract
def test_entity_graph_relationship_filtering(client, sample_entity_id):
    """Test filtering relationships by type."""
    relationship_types = ["CALLS", "INHERITS"]
    params = "&".join([f"relationship_types={rt}" for rt in relationship_types])
    response = client.get(f"/graph/{sample_entity_id}?{params}")

    assert response.status_code == 200
    response_data = response.json()

    # If relationships exist, they should match the filter
    relationships = response_data["relationships"]
    for rel in relationships:
        assert rel["relationship_type"] in relationship_types