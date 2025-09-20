"""Integration test for multi-language parsing workflow."""

import pytest
from pathlib import Path
from repo_kgraph.services.repository_manager import RepositoryManager

@pytest.mark.integration
async def test_multilanguage_repository_parsing():
    """Test parsing repository with multiple programming languages."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_path = Path(tmp_dir)
        
        # Create Python file
        (repo_path / "main.py").write_text("""
class PythonClass:
    def method(self): pass
""")
        
        # Create JavaScript file
        (repo_path / "app.js").write_text("""
class JavaScriptClass {
    method() { return true; }
}
""")
        
        # Create TypeScript file
        (repo_path / "types.ts").write_text("""
interface TypeScriptInterface {
    property: string;
}
""")
        
        repo_manager = RepositoryManager()
        repository = await repo_manager.parse_repository(str(repo_path))
        
        entities = await repo_manager.get_entities(repository.id)
        
        # Should have entities from all languages
        languages = {entity.language for entity in entities}
        assert "python" in languages
        assert "javascript" in languages
        assert "typescript" in languages

@pytest.mark.integration
async def test_cross_language_relationships():
    """Test relationships between entities in different languages."""
    # Will fail until implemented
    pass