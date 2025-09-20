"""
Contract tests for POST /parse-repo API endpoint.

These tests define the expected behavior of the repository parsing API
and MUST FAIL initially (TDD approach).
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from repo_kgraph.api.main import app


@pytest.fixture
def client():
    """Create test client for API testing."""
    return TestClient(app)


@pytest.mark.contract
def test_parse_repo_endpoint_exists(client):
    """Test that the parse-repo endpoint exists and accepts POST requests."""
    # This will fail until we implement the endpoint
    response = client.post("/parse-repo")
    assert response.status_code != 404, "POST /parse-repo endpoint should exist"


@pytest.mark.contract
def test_parse_repo_valid_request(client, temp_dir):
    """Test parsing a valid repository."""
    request_data = {
        "repository_path": str(temp_dir),
        "incremental": False,
        "languages": ["python", "javascript"],
        "exclude_patterns": ["*.test.py", "__pycache__/*"]
    }

    response = client.post("/parse-repo", json=request_data)

    # Should return 202 (Accepted) for async processing
    assert response.status_code == 202

    response_data = response.json()
    assert "repository_id" in response_data
    assert "status" in response_data
    assert response_data["status"] in ["queued", "processing"]
    assert "message" in response_data

    # Should have UUID format for repository_id
    repository_id = response_data["repository_id"]
    assert len(repository_id) == 36  # UUID length with hyphens
    assert repository_id.count("-") == 4  # UUID has 4 hyphens


@pytest.mark.contract
def test_parse_repo_missing_path(client):
    """Test error handling when repository_path is missing."""
    request_data = {
        "incremental": False
    }

    response = client.post("/parse-repo", json=request_data)

    # Should return 400 (Bad Request)
    assert response.status_code == 400

    response_data = response.json()
    assert "error" in response_data
    assert "message" in response_data
    assert "repository_path" in response_data["message"].lower()


@pytest.mark.contract
def test_parse_repo_invalid_path(client):
    """Test error handling for non-existent repository path."""
    request_data = {
        "repository_path": "/non/existent/path",
        "incremental": False
    }

    response = client.post("/parse-repo", json=request_data)

    # Should return 404 (Not Found)
    assert response.status_code == 404

    response_data = response.json()
    assert "error" in response_data
    assert "message" in response_data
    assert "not found" in response_data["message"].lower()


@pytest.mark.contract
def test_parse_repo_incremental_update(client, temp_dir):
    """Test incremental repository parsing."""
    request_data = {
        "repository_path": str(temp_dir),
        "incremental": True
    }

    response = client.post("/parse-repo", json=request_data)

    assert response.status_code == 202
    response_data = response.json()
    assert "repository_id" in response_data
    assert "status" in response_data


@pytest.mark.contract
def test_parse_repo_response_format(client, temp_dir):
    """Test that response matches expected schema."""
    request_data = {
        "repository_path": str(temp_dir)
    }

    response = client.post("/parse-repo", json=request_data)

    assert response.status_code == 202
    response_data = response.json()

    # Required fields from OpenAPI schema
    required_fields = ["repository_id", "status", "message"]
    for field in required_fields:
        assert field in response_data, f"Required field '{field}' missing from response"

    # Optional fields that might be present
    optional_fields = ["estimated_completion", "progress"]
    for field in optional_fields:
        if field in response_data:
            assert response_data[field] is not None

    # Validate status enum values
    valid_statuses = ["queued", "processing", "completed", "failed"]
    assert response_data["status"] in valid_statuses


@pytest.mark.contract
def test_parse_repo_with_language_filter(client, temp_dir):
    """Test parsing with specific language filters."""
    request_data = {
        "repository_path": str(temp_dir),
        "languages": ["python", "typescript"],
        "exclude_patterns": ["node_modules/*", "*.test.ts"]
    }

    response = client.post("/parse-repo", json=request_data)

    assert response.status_code == 202
    response_data = response.json()
    assert "repository_id" in response_data


@pytest.mark.contract
def test_parse_repo_invalid_json(client):
    """Test handling of malformed JSON requests."""
    response = client.post(
        "/parse-repo",
        data="invalid json",
        headers={"Content-Type": "application/json"}
    )

    # Should return 400 for malformed JSON
    assert response.status_code == 400


@pytest.mark.contract
def test_parse_repo_large_exclude_patterns(client, temp_dir):
    """Test parsing with many exclude patterns."""
    request_data = {
        "repository_path": str(temp_dir),
        "exclude_patterns": [
            "node_modules/*",
            "*.test.js",
            "*.test.py",
            "__pycache__/*",
            "*.pyc",
            ".git/*",
            "build/*",
            "dist/*",
            ".venv/*",
            "coverage/*"
        ]
    }

    response = client.post("/parse-repo", json=request_data)

    assert response.status_code == 202
    response_data = response.json()
    assert "repository_id" in response_data