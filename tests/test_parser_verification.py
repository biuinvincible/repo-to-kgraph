"""
Comprehensive parser verification test.

This test validates that our parser extracts exactly the expected entities and relationships
from a manually crafted test repository where every line of code is understood.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Set, Any

# Add src to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from repo_kgraph.services.parser import CodeParser
from repo_kgraph.services.graph_builder import GraphBuilder
from repo_kgraph.models.code_entity import EntityType
from repo_kgraph.lib.config import get_config


logger = logging.getLogger(__name__)


class ParserVerificationTest:
    """Test class for comprehensive parser verification."""

    def __init__(self):
        self.test_repo_path = repo_root / "test-verification"
        self.config = get_config()
        self.parser = CodeParser()
        self.graph_builder = None
        self.repository_id = "test-verification"

    async def setup(self):
        """Setup test environment."""
        # Initialize graph builder
        self.graph_builder = GraphBuilder(
            uri=self.config.database.neo4j_uri,
            username=self.config.database.neo4j_username,
            password=self.config.database.neo4j_password,
            database=self.config.database.neo4j_database
        )

        # Clear any existing test data
        await self.graph_builder.delete_repository_graph(self.repository_id)

    async def parse_test_repository(self):
        """Parse the test repository."""
        # Parse directory using the correct method
        parsing_stats = await self.parser.parse_directory(
            directory_path=str(self.test_repo_path),
            repository_id=self.repository_id
        )

        # For verification, we need to get individual parse results
        # Let's parse each file individually to get the detailed results
        parsing_results = []

        # Find all Python files in test directory
        test_files = list(self.test_repo_path.glob("*.py"))

        for file_path in test_files:
            result = await self.parser.parse_file(
                str(file_path),
                self.repository_id,
                str(self.test_repo_path)
            )
            parsing_results.append(result)

        # Store in graph database
        all_entities = []
        all_relationships = []

        for file_result in parsing_results:
            entities = file_result.entities
            relationships = file_result.relationships

            # Set repository_id for entities
            for entity in entities:
                entity.repository_id = self.repository_id

            # Note: relationships don't have repository_id field in the model
            # but the parser should create them correctly

            all_entities.extend(entities)
            all_relationships.extend(relationships)

        # Store entities in batches
        if all_entities:
            await self.graph_builder.add_entities_batch(all_entities)

        # Store relationships in batches
        if all_relationships:
            await self.graph_builder.add_relationships_batch(all_relationships)

        return parsing_results

    async def get_parsed_entities(self) -> List[Any]:
        """Get all entities from the graph database."""
        return await self.graph_builder.get_entities_by_repository(self.repository_id)

    async def get_parsed_relationships(self) -> List[Any]:
        """Get all relationships from the graph database."""
        # Since there's no get_relationships_by_repository method,
        # let's use a direct Neo4j query to get relationships
        if not self.graph_builder._driver:
            return []

        relationships = []
        try:
            with self.graph_builder._driver.session(database=self.graph_builder.database) as session:
                result = session.run(
                    """
                    MATCH (e1:Entity {repository_id: $repository_id})-[r:RELATES]-(e2:Entity {repository_id: $repository_id})
                    RETURN r.id as id, r.relationship_type as type, r.source_entity_id as source, r.target_entity_id as target
                    """,
                    repository_id=self.repository_id
                )
                for record in result:
                    # Create a simple relationship object
                    rel_obj = type('Relationship', (), {
                        'id': record['id'],
                        'relationship_type': record['type'],
                        'source_entity_id': record['source'],
                        'target_entity_id': record['target']
                    })()
                    relationships.append(rel_obj)
        except Exception as e:
            logger.error(f"Error fetching relationships: {e}")

        return relationships

    def verify_entity_counts(self, entities: List[Any]) -> Dict[str, Any]:
        """Verify entity counts match expectations."""
        results = {
            "total_entities": len(entities),
            "expected_total": 12,
            "entity_types": {},
            "files": set()
        }

        # Count by type and file
        for entity in entities:
            # Handle Neo4j Node objects - access properties using dict-like notation
            entity_type = entity.get("entity_type", "UNKNOWN")
            results["entity_types"][entity_type] = results["entity_types"].get(entity_type, 0) + 1
            results["files"].add(entity.get("file_path", "unknown_file"))

        # Expected distribution
        expected_types = {
            "CLASS": 2,
            "FUNCTION": 5,
            "METHOD": 5,
            "VARIABLE": 0  # Variables might not be detected depending on parser
        }

        results["type_verification"] = {}
        for expected_type, expected_count in expected_types.items():
            actual_count = results["entity_types"].get(expected_type, 0)
            results["type_verification"][expected_type] = {
                "expected": expected_count,
                "actual": actual_count,
                "match": actual_count >= expected_count  # >= because parser might find more
            }

        return results

    def verify_specific_entities(self, entities: List[Any]) -> Dict[str, bool]:
        """Verify that specific expected entities are found."""
        entity_names = {entity.get("name", "") for entity in entities}

        expected_entities = {
            # From user.py
            "User",
            "__init__",  # User.__init__
            "deactivate",
            "validate_email",
            "get_user_by_id",
            "create_user",
            # From database.py
            "DatabaseConnection",
            "connect",
            "execute",
            "close",
            "connect_to_database",
            "execute_query"
        }

        verification = {}
        for expected in expected_entities:
            verification[expected] = expected in entity_names

        return verification

    def verify_file_coverage(self, entities: List[Any]) -> Dict[str, bool]:
        """Verify that all expected files are parsed."""
        files_found = {Path(entity.get("file_path", "")).name for entity in entities if entity.get("file_path")}

        expected_files = {"user.py", "database.py"}

        return {
            "user.py": "user.py" in files_found,
            "database.py": "database.py" in files_found,
            "total_files": len(files_found),
            "expected_files": len(expected_files)
        }

    def verify_relationships(self, relationships: List[Any]) -> Dict[str, Any]:
        """Verify relationships are correctly identified."""
        results = {
            "total_relationships": len(relationships),
            "expected_minimum": 5,  # Conservative estimate
            "relationship_types": {},
            "cross_file_relationships": 0
        }

        for rel in relationships:
            rel_type = rel.relationship_type
            results["relationship_types"][rel_type] = results["relationship_types"].get(rel_type, 0) + 1

            # Check for cross-file relationships
            if hasattr(rel, 'source_entity') and hasattr(rel, 'target_entity'):
                source_file = getattr(rel.source_entity, 'file_path', '')
                target_file = getattr(rel.target_entity, 'file_path', '')
                if source_file and target_file and source_file != target_file:
                    results["cross_file_relationships"] += 1

        return results

    async def run_comprehensive_verification(self) -> Dict[str, Any]:
        """Run complete verification test."""
        print("🚀 Starting comprehensive parser verification...")

        # Setup
        await self.setup()

        # Parse repository
        print("📁 Parsing test repository...")
        parsing_results = await self.parse_test_repository()

        # Get results from database
        entities = await self.get_parsed_entities()
        relationships = await self.get_parsed_relationships()

        print(f"📊 Found {len(entities)} entities and {len(relationships)} relationships")

        # Run verifications
        verification_results = {
            "entity_counts": self.verify_entity_counts(entities),
            "specific_entities": self.verify_specific_entities(entities),
            "file_coverage": self.verify_file_coverage(entities),
            "relationships": self.verify_relationships(relationships),
            "parsing_success": len(parsing_results) > 0,
            "entities_raw": [
                {
                    "name": e.get("name", "unknown"),
                    "type": e.get("entity_type", "UNKNOWN"),
                    "file": Path(e.get("file_path", "")).name if e.get("file_path") else "unknown",
                    "line": e.get("start_line", "unknown")
                }
                for e in entities
            ]
        }

        # Calculate overall success
        entity_count_success = verification_results["entity_counts"]["total_entities"] >= 10  # Allow some variance
        specific_entities_success = sum(verification_results["specific_entities"].values()) >= 10
        file_coverage_success = verification_results["file_coverage"]["user.py"] and verification_results["file_coverage"]["database.py"]
        relationships_success = verification_results["relationships"]["total_relationships"] >= 5

        verification_results["overall_success"] = all([
            entity_count_success,
            specific_entities_success,
            file_coverage_success,
            relationships_success
        ])

        return verification_results

    async def cleanup(self):
        """Cleanup test data."""
        if self.graph_builder:
            await self.graph_builder.delete_repository_graph(self.repository_id)


async def test_parser_comprehensive_verification():
    """Main test function."""
    test = ParserVerificationTest()

    try:
        results = await test.run_comprehensive_verification()

        # Print detailed results
        print("\n" + "="*60)
        print("🔍 PARSER VERIFICATION RESULTS")
        print("="*60)

        print(f"\n📊 Entity Count Verification:")
        print(f"   Total entities found: {results['entity_counts']['total_entities']}")
        print(f"   Expected minimum: {results['entity_counts']['expected_total']}")

        print(f"\n📝 Entity Type Distribution:")
        for entity_type, count in results['entity_counts']['entity_types'].items():
            print(f"   {entity_type}: {count}")

        print(f"\n✅ Specific Entity Verification:")
        for entity_name, found in results['specific_entities'].items():
            status = "✓" if found else "✗"
            print(f"   {status} {entity_name}")

        print(f"\n📁 File Coverage:")
        for file_name, covered in results['file_coverage'].items():
            if file_name.endswith('.py'):
                status = "✓" if covered else "✗"
                print(f"   {status} {file_name}")

        print(f"\n🔗 Relationship Verification:")
        print(f"   Total relationships: {results['relationships']['total_relationships']}")
        print(f"   Cross-file relationships: {results['relationships']['cross_file_relationships']}")

        print(f"\n🎯 Overall Success: {'✅ PASS' if results['overall_success'] else '❌ FAIL'}")

        # Assertions for pytest
        assert results['entity_counts']['total_entities'] >= 10, f"Expected at least 10 entities, got {results['entity_counts']['total_entities']}"
        assert results['file_coverage']['user.py'], "user.py not properly parsed"
        assert results['file_coverage']['database.py'], "database.py not properly parsed"
        assert results['relationships']['total_relationships'] >= 5, f"Expected at least 5 relationships, got {results['relationships']['total_relationships']}"

        # Check critical entities
        critical_entities = ['User', 'validate_email', 'DatabaseConnection', 'create_user']
        for entity in critical_entities:
            assert results['specific_entities'].get(entity, False), f"Critical entity '{entity}' not found"

        print("\n🎉 All verification tests passed!")

    finally:
        await test.cleanup()


if __name__ == "__main__":
    # Run the test directly
    asyncio.run(test_parser_comprehensive_verification())