"""
Contract tests for POST /query API endpoint.

These tests define the expected behavior of the context query API
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
def sample_repository_id():
    """Sample repository ID for testing."""
    return "550e8400-e29b-41d4-a716-446655440000"


@pytest.mark.contract
def test_query_endpoint_exists(client):
    """Test that the query endpoint exists and accepts POST requests."""
    # This will fail until we implement the endpoint
    response = client.post("/query")
    assert response.status_code != 404, "POST /query endpoint should exist"


@pytest.mark.contract
def test_query_valid_request(client, sample_repository_id):
    """Test querying with valid parameters."""
    request_data = {
        "repository_id": sample_repository_id,
        "task_description": "Add authentication middleware to the Express.js API",
        "max_results": 20,
        "include_context": True,
        "confidence_threshold": 0.3
    }

    response = client.post("/query", json=request_data)

    # Should return 200 (OK) for successful query
    assert response.status_code == 200

    response_data = response.json()
    assert "query_id" in response_data
    assert "results" in response_data
    assert "processing_time_ms" in response_data
    assert "total_results" in response_data
    assert "confidence_score" in response_data

    # Validate query_id is UUID format
    query_id = response_data["query_id"]
    assert len(query_id) == 36  # UUID length with hyphens
    assert query_id.count("-") == 4  # UUID has 4 hyphens

    # Results should be an array
    assert isinstance(response_data["results"], list)

    # Processing time should be reasonable
    assert response_data["processing_time_ms"] >= 0
    assert response_data["processing_time_ms"] < 60000  # Less than 60 seconds

    # Confidence score should be between 0 and 1
    confidence = response_data["confidence_score"]
    assert 0.0 <= confidence <= 1.0


@pytest.mark.contract
def test_query_missing_repository_id(client):
    """Test error handling when repository_id is missing."""
    request_data = {
        "task_description": "Fix bug in payment processing"
    }

    response = client.post("/query", json=request_data)

    # Should return 400 (Bad Request)
    assert response.status_code == 400

    response_data = response.json()
    assert "error" in response_data
    assert "message" in response_data
    assert "repository_id" in response_data["message"].lower()


@pytest.mark.contract
def test_query_missing_task_description(client, sample_repository_id):
    """Test error handling when task_description is missing."""
    request_data = {
        "repository_id": sample_repository_id
    }

    response = client.post("/query", json=request_data)

    # Should return 400 (Bad Request)
    assert response.status_code == 400

    response_data = response.json()
    assert "error" in response_data
    assert "message" in response_data
    assert "task_description" in response_data["message"].lower()


@pytest.mark.contract
def test_query_repository_not_found(client):
    """Test error handling for non-existent repository."""
    request_data = {
        "repository_id": "00000000-0000-0000-0000-000000000000",
        "task_description": "Test query"
    }

    response = client.post("/query", json=request_data)

    # Should return 404 (Not Found)
    assert response.status_code == 404

    response_data = response.json()
    assert "error" in response_data
    assert "message" in response_data
    assert "not found" in response_data["message"].lower()


@pytest.mark.contract
def test_query_result_format(client, sample_repository_id):
    """Test that individual results match expected schema."""
    request_data = {
        "repository_id": sample_repository_id,
        "task_description": "Add user authentication",
        "max_results": 5,
        "include_context": True
    }

    response = client.post("/query", json=request_data)

    assert response.status_code == 200
    response_data = response.json()

    # If results exist, check their format
    results = response_data["results"]
    if results:  # Only check if there are results
        result = results[0]

        # Required fields from OpenAPI schema
        required_fields = [
            "entity_id", "entity_type", "name", "file_path", "relevance_score"
        ]
        for field in required_fields:
            assert field in result, f"Required field '{field}' missing from result"

        # Validate entity_id is UUID
        entity_id = result["entity_id"]
        assert len(entity_id) == 36
        assert entity_id.count("-") == 4

        # Validate entity_type enum
        valid_types = ["FILE", "CLASS", "FUNCTION", "VARIABLE", "MODULE", "INTERFACE", "ENUM", "STRUCT"]
        assert result["entity_type"] in valid_types

        # Validate relevance_score is between 0 and 1
        relevance = result["relevance_score"]
        assert 0.0 <= relevance <= 1.0


@pytest.mark.contract
def test_query_with_filters(client, sample_repository_id):
    """Test querying with entity type filters."""
    request_data = {
        "repository_id": sample_repository_id,
        "task_description": "Authentication functions",
        "max_results": 10,
        "filter_types": ["FUNCTION", "CLASS"],
        "confidence_threshold": 0.5
    }

    response = client.post("/query", json=request_data)

    assert response.status_code == 200
    response_data = response.json()
    assert "results" in response_data

    # If results exist, check filtering worked
    results = response_data["results"]
    for result in results:
        assert result["entity_type"] in ["FUNCTION", "CLASS"]
        assert result["relevance_score"] >= 0.5


@pytest.mark.contract
def test_query_max_results_limit(client, sample_repository_id):
    """Test that max_results parameter is respected."""
    max_results = 5
    request_data = {
        "repository_id": sample_repository_id,
        "task_description": "Database operations",
        "max_results": max_results
    }

    response = client.post("/query", json=request_data)

    assert response.status_code == 200
    response_data = response.json()

    results = response_data["results"]
    assert len(results) <= max_results


@pytest.mark.contract
def test_query_empty_description(client, sample_repository_id):
    """Test error handling for empty task description."""
    request_data = {
        "repository_id": sample_repository_id,
        "task_description": ""
    }

    response = client.post("/query", json=request_data)

    # Should return 400 for empty description
    assert response.status_code == 400

    response_data = response.json()
    assert "error" in response_data
    assert "message" in response_data


@pytest.mark.contract
def test_query_invalid_confidence_threshold(client, sample_repository_id):
    """Test error handling for invalid confidence threshold."""
    request_data = {
        "repository_id": sample_repository_id,
        "task_description": "Test query",
        "confidence_threshold": 1.5  # Invalid: > 1.0
    }

    response = client.post("/query", json=request_data)

    # Should return 400 for invalid threshold
    assert response.status_code == 400

    response_data = response.json()
    assert "error" in response_data


@pytest.mark.contract
def test_query_response_performance(client, sample_repository_id):
    """Test that queries complete within reasonable time."""
    request_data = {
        "repository_id": sample_repository_id,
        "task_description": "Complex query for performance testing with multiple terms and context"
    }

    response = client.post("/query", json=request_data)

    assert response.status_code == 200
    response_data = response.json()

    # Query should complete within 5 seconds for typical cases
    processing_time = response_data["processing_time_ms"]
    assert processing_time < 5000, f"Query took {processing_time}ms, expected < 5000ms"