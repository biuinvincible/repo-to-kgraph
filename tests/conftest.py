"""
Pytest configuration and shared fixtures for repo-kgraph tests.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Generator, AsyncGenerator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from neo4j import GraphDatabase
from chromadb import Client as ChromaClient
from chromadb.config import Settings

from repo_kgraph.models.repository import Repository
from repo_kgraph.models.code_entity import CodeEntity
from repo_kgraph.models.relationship import Relationship


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_repo_path(temp_dir: Path) -> Path:
    """Create a sample repository structure for testing."""
    repo_path = temp_dir / "sample_repo"
    repo_path.mkdir()

    # Create sample Python file
    (repo_path / "main.py").write_text("""
def hello_world(name: str) -> str:
    '''Say hello to someone.'''
    return f"Hello, {name}!"

class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b
""")

    # Create sample JavaScript file
    (repo_path / "utils.js").write_text("""
function formatDate(date) {
    return date.toISOString();
}

class DataProcessor {
    constructor() {
        this.data = [];
    }

    process(input) {
        return input.map(x => x * 2);
    }
}

module.exports = { formatDate, DataProcessor };
""")

    # Create requirements file
    (repo_path / "requirements.txt").write_text("""
requests>=2.25.0
numpy>=1.20.0
""")

    return repo_path


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing without database connection."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = None
    return driver


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client for testing without vector database."""
    client = MagicMock(spec=ChromaClient)
    collection = MagicMock()
    client.create_collection.return_value = collection
    client.get_collection.return_value = collection
    return client


@pytest.fixture
def sample_repository() -> Repository:
    """Create a sample Repository instance for testing."""
    return Repository(
        id="550e8400-e29b-41d4-a716-446655440000",
        path="/test/sample_repo",
        name="sample_repo",
        size_bytes=1024,
        file_count=3,
        language_stats={"python": 15, "javascript": 12},
        last_indexed="2025-09-17T10:30:00Z",
        last_modified="2025-09-17T10:00:00Z",
        git_hash="abc123def456",
        metadata={"branch": "main"}
    )


@pytest.fixture
def sample_code_entities() -> list[CodeEntity]:
    """Create sample CodeEntity instances for testing."""
    return [
        CodeEntity(
            id="entity-1",
            repository_id="550e8400-e29b-41d4-a716-446655440000",
            entity_type="FUNCTION",
            name="hello_world",
            qualified_name="main.hello_world",
            file_path="main.py",
            start_line=2,
            end_line=4,
            start_column=0,
            end_column=25,
            language="python",
            signature="def hello_world(name: str) -> str",
            docstring="Say hello to someone.",
            complexity_score=1.0,
            embedding_vector=[0.1, 0.2, 0.3] * 128,  # 384-dim vector
            metadata={"returns": "str"}
        ),
        CodeEntity(
            id="entity-2",
            repository_id="550e8400-e29b-41d4-a716-446655440000",
            entity_type="CLASS",
            name="Calculator",
            qualified_name="main.Calculator",
            file_path="main.py",
            start_line=6,
            end_line=12,
            start_column=0,
            end_column=25,
            language="python",
            signature="class Calculator",
            docstring="",
            complexity_score=3.0,
            embedding_vector=[0.4, 0.5, 0.6] * 128,
            metadata={"methods": ["add", "multiply"]}
        ),
    ]


@pytest.fixture
def sample_relationships(sample_code_entities: list[CodeEntity]) -> list[Relationship]:
    """Create sample Relationship instances for testing."""
    return [
        Relationship(
            id="rel-1",
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="CONTAINS",
            strength=0.9,
            line_number=6,
            context="class Calculator:",
            metadata={"containment_type": "class_in_file"}
        )
    ]


@pytest.fixture
def test_config() -> dict:
    """Configuration for testing environment."""
    return {
        "database": {
            "neo4j_uri": "neo4j://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "testpass",
        },
        "vector_store": {
            "type": "chroma",
            "persist_directory": ":memory:",
        },
        "embedding": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": 384,
        },
        "parsing": {
            "max_file_size": 1024 * 1024,  # 1MB
            "parallel_jobs": 2,
            "exclude_patterns": ["*.test.py", "__pycache__/*"],
        },
        "api": {
            "host": "localhost",
            "port": 8000,
            "debug": True,
        },
    }


@pytest.fixture
async def test_app():
    """Create test FastAPI application instance."""
    # This will be implemented when we create the main app
    # For now, return a mock
    return MagicMock()


@pytest.fixture
def client(test_app) -> TestClient:
    """Create test client for API testing."""
    return TestClient(test_app)


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Automatically set up test environment variables."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("NEO4J_URI", "neo4j://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "test")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")


# Custom pytest markers for test categorization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.contract = pytest.mark.contract
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow