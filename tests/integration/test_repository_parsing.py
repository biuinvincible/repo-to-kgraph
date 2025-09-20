"""
Integration test for full repository parsing workflow.

This test verifies the complete end-to-end repository analysis process.
MUST FAIL initially (TDD approach).
"""

import pytest
import tempfile
from pathlib import Path

from repo_kgraph.services.repository_manager import RepositoryManager
from repo_kgraph.services.parser import CodeParser
from repo_kgraph.services.graph_builder import GraphBuilder


@pytest.mark.integration
async def test_full_repository_parsing_workflow(sample_repo_path, mock_neo4j_driver, mock_chroma_client):
    """Test complete repository parsing from start to finish."""
    # This will fail until we implement the services

    # Initialize services (these classes don't exist yet)
    parser = CodeParser()
    graph_builder = GraphBuilder(driver=mock_neo4j_driver)
    repo_manager = RepositoryManager(
        parser=parser,
        graph_builder=graph_builder,
        vector_client=mock_chroma_client
    )

    # Parse repository
    repository = await repo_manager.parse_repository(
        repository_path=str(sample_repo_path),
        incremental=False
    )

    # Verify repository was created
    assert repository is not None
    assert repository.path == str(sample_repo_path)
    assert repository.file_count > 0
    assert repository.entity_count > 0

    # Verify entities were extracted
    entities = await repo_manager.get_entities(repository.id)
    assert len(entities) > 0

    # Should have found Python and JavaScript entities
    entity_types = {entity.entity_type for entity in entities}
    assert "FUNCTION" in entity_types
    assert "CLASS" in entity_types

    # Verify relationships were built
    relationships = await repo_manager.get_relationships(repository.id)
    assert len(relationships) > 0


@pytest.mark.integration
async def test_parsing_with_language_filter(sample_repo_path):
    """Test repository parsing with language filtering."""
    repo_manager = RepositoryManager()

    repository = await repo_manager.parse_repository(
        repository_path=str(sample_repo_path),
        languages=["python"]  # Only parse Python files
    )

    entities = await repo_manager.get_entities(repository.id)

    # Should only have Python entities
    for entity in entities:
        if entity.entity_type != "FILE":
            assert entity.language == "python"


@pytest.mark.integration
async def test_parsing_with_exclusions(sample_repo_path):
    """Test repository parsing with file exclusions."""
    # Create a test file that should be excluded
    (sample_repo_path / "test_file.py").write_text("# This should be excluded")
    (sample_repo_path / "__pycache__").mkdir()
    (sample_repo_path / "__pycache__" / "cache.pyc").write_text("cached")

    repo_manager = RepositoryManager()

    repository = await repo_manager.parse_repository(
        repository_path=str(sample_repo_path),
        exclude_patterns=["*.test.*", "__pycache__/*"]
    )

    entities = await repo_manager.get_entities(repository.id)

    # Should not have excluded files
    file_paths = {entity.file_path for entity in entities if entity.entity_type == "FILE"}
    assert "test_file.py" not in file_paths
    assert not any("__pycache__" in path for path in file_paths)


@pytest.mark.integration
@pytest.mark.slow
async def test_large_repository_parsing():
    """Test parsing a larger repository structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_path = Path(tmp_dir)

        # Create a more complex repository structure
        for i in range(10):
            subdir = repo_path / f"module_{i}"
            subdir.mkdir()

            for j in range(5):
                python_file = subdir / f"file_{j}.py"
                python_file.write_text(f"""
class Class{i}{j}:
    def method_{j}(self):
        return {i} + {j}

def function_{i}_{j}():
    obj = Class{i}{j}()
    return obj.method_{j}()
""")

        repo_manager = RepositoryManager()
        repository = await repo_manager.parse_repository(str(repo_path))

        # Should have parsed all files
        assert repository.file_count >= 50  # 10 dirs * 5 files

        entities = await repo_manager.get_entities(repository.id)
        assert len(entities) >= 150  # At least 3 entities per file


@pytest.mark.integration
async def test_parsing_error_handling():
    """Test error handling during parsing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_path = Path(tmp_dir)

        # Create a file with syntax errors
        bad_file = repo_path / "bad_syntax.py"
        bad_file.write_text("""
def incomplete_function(
    # Missing closing parenthesis and implementation
""")

        # Create a very large file that might cause issues
        large_file = repo_path / "large_file.py"
        large_file.write_text("# Comment\n" * 10000)

        repo_manager = RepositoryManager()

        # Should handle errors gracefully and still parse good files
        good_file = repo_path / "good_file.py"
        good_file.write_text("def good_function(): return 'ok'")

        repository = await repo_manager.parse_repository(str(repo_path))

        # Should have successfully parsed at least the good file
        assert repository is not None
        entities = await repo_manager.get_entities(repository.id)

        # Should have at least some entities from the good file
        good_entities = [e for e in entities if "good_file.py" in e.file_path]
        assert len(good_entities) > 0