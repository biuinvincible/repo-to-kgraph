"""Integration test for incremental repository updates."""

import pytest
from repo_kgraph.services.repository_manager import RepositoryManager

@pytest.mark.integration
async def test_incremental_update_workflow(sample_repo_path):
    """Test incremental updates when repository changes."""
    repo_manager = RepositoryManager()
    
    # Initial parse
    repository = await repo_manager.parse_repository(str(sample_repo_path))
    initial_entity_count = repository.entity_count
    
    # Add a new file
    new_file = sample_repo_path / "new_module.py"
    new_file.write_text("def new_function(): pass")
    
    # Incremental update
    updated_repository = await repo_manager.parse_repository(
        str(sample_repo_path),
        incremental=True
    )
    
    # Should have more entities
    assert updated_repository.entity_count > initial_entity_count

@pytest.mark.integration
async def test_file_modification_detection():
    """Test detection of modified files."""
    # Will fail until implemented
    pass