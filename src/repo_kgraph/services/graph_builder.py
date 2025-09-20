"""
Graph builder service for Neo4j graph construction.

Handles creation and management of the knowledge graph in Neo4j
with entity and relationship persistence, queries, and optimization.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

try:
    from neo4j import GraphDatabase, Driver, Session, Result
    from neo4j.exceptions import ServiceUnavailable, AuthError
except ImportError:
    GraphDatabase = None
    Driver = None
    Session = None
    Result = None

from repo_kgraph.models.repository import Repository
from repo_kgraph.models.code_entity import CodeEntity, EntityType
from repo_kgraph.models.relationship import Relationship, RelationshipType
from repo_kgraph.models.knowledge_graph import KnowledgeGraph, GraphStatus


logger = logging.getLogger(__name__)


class GraphConnectionError(Exception):
    """Raised when unable to connect to Neo4j database."""
    pass


class GraphBuilder:
    """
    Neo4j graph database builder and manager.

    Handles creation, updates, and queries of the knowledge graph
    with optimized batch operations and relationship management.
    """

    def __init__(
        self,
        uri: str = "neo4j://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        max_connection_lifetime: int = 30,
        max_connection_pool_size: int = 50,
        driver: Optional[Driver] = None
    ):
        """
        Initialize the graph builder.

        Args:
            uri: Neo4j connection URI
            username: Database username
            password: Database password
            database: Database name
            max_connection_lifetime: Connection lifetime in seconds
            max_connection_pool_size: Maximum connections in pool
            driver: Optional pre-configured driver for testing
        """
        self.uri = uri
        self.username = username
        self.database = database
        self._driver = driver

        if not self._driver and GraphDatabase:
            try:
                self._driver = GraphDatabase.driver(
                    uri,
                    auth=(username, password),
                    max_connection_lifetime=max_connection_lifetime,
                    max_connection_pool_size=max_connection_pool_size
                )
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j driver: {e}")
                raise GraphConnectionError(f"Cannot connect to Neo4j: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._driver:
            self._driver.close()

    def close(self) -> None:
        """Close the database connection."""
        if self._driver:
            self._driver.close()

    async def verify_connection(self) -> bool:
        """
        Verify database connection is working.

        Returns:
            True if connection is successful, False otherwise
        """
        if not self._driver:
            return False

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            logger.error(f"Database connection verification failed: {e}")
            return False

    async def initialize_schema(self) -> None:
        """Initialize database schema with indexes and constraints."""
        if not self._driver:
            raise GraphConnectionError("No database connection available")

        schema_commands = [
            # Constraints for unique IDs
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT repository_id_unique IF NOT EXISTS FOR (r:Repository) REQUIRE r.id IS UNIQUE",

            # Indexes for performance
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX entity_language_idx IF NOT EXISTS FOR (e:Entity) ON (e.language)",
            "CREATE INDEX entity_file_path_idx IF NOT EXISTS FOR (e:Entity) ON (e.file_path)",
            "CREATE INDEX relationship_type_idx IF NOT EXISTS FOR ()-[r:RELATES]-() ON (r.relationship_type)",

            # Composite indexes
            "CREATE INDEX entity_repo_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.repository_id, e.entity_type)",
            "CREATE INDEX entity_repo_file_idx IF NOT EXISTS FOR (e:Entity) ON (e.repository_id, e.file_path)",
        ]

        # Use sync session inside async method
        with self._driver.session(database=self.database) as session:
            for command in schema_commands:
                try:
                    session.run(command)
                    logger.debug(f"Executed schema command: {command}")
                except Exception as e:
                    logger.warning(f"Schema command failed (may already exist): {command} - {e}")

    async def create_repository_graph(self, repository: Repository) -> KnowledgeGraph:
        """
        Create a new knowledge graph for a repository.

        Args:
            repository: Repository to create graph for

        Returns:
            KnowledgeGraph instance
        """
        if not self._driver:
            raise GraphConnectionError("No database connection available")

        # Create repository node
        with self._driver.session(database=self.database) as session:
            session.run(
                """
                CREATE (r:Repository {
                    id: $id,
                    name: $name,
                    path: $path,
                    created_at: $created_at
                })
                """,
                id=repository.id,
                name=repository.name,
                path=repository.path,
                created_at=repository.created_at.isoformat()
            )

        # Create and return knowledge graph
        knowledge_graph = KnowledgeGraph(
            repository_id=repository.id,
            status=GraphStatus.BUILDING,
            created_at=datetime.utcnow()
        )

        return knowledge_graph

    async def add_entities_batch(
        self,
        entities: List[CodeEntity],
        batch_size: int = 1000
    ) -> int:
        """
        Add entities to the graph in batches.

        Args:
            entities: List of entities to add
            batch_size: Number of entities per batch

        Returns:
            Number of entities added successfully
        """
        if not self._driver or not entities:
            return 0

        added_count = 0

        # Process in batches
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            batch_params = []

            for entity in batch:
                entity_data = {
                    "id": entity.id,
                    "repository_id": entity.repository_id,
                    "entity_type": entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type),
                    "name": entity.name,
                    "qualified_name": entity.qualified_name,
                    "file_path": entity.file_path,
                    "start_line": entity.start_line,
                    "end_line": entity.end_line,
                    "start_column": entity.start_column,
                    "end_column": entity.end_column,
                    "language": entity.language,
                    "signature": entity.signature or "",
                    "docstring": entity.docstring or "",
                    "complexity_score": entity.complexity_score or 0.0,
                    "line_count": entity.line_count,
                    "created_at": entity.created_at.isoformat()
                }
                batch_params.append(entity_data)

            try:
                with self._driver.session(database=self.database) as session:
                    result = session.run(
                        """
                        UNWIND $entities as entity
                        CREATE (e:Entity {
                            id: entity.id,
                            repository_id: entity.repository_id,
                            entity_type: entity.entity_type,
                            name: entity.name,
                            qualified_name: entity.qualified_name,
                            file_path: entity.file_path,
                            start_line: entity.start_line,
                            end_line: entity.end_line,
                            start_column: entity.start_column,
                            end_column: entity.end_column,
                            language: entity.language,
                            signature: entity.signature,
                            docstring: entity.docstring,
                            complexity_score: entity.complexity_score,
                            line_count: entity.line_count,
                            created_at: entity.created_at
                        })
                        """,
                        entities=batch_params
                    )
                    # Consume result to get accurate statistics
                    summary = result.consume()
                    batch_added = summary.counters.nodes_created
                    added_count += batch_added
                    logger.debug(f"Added batch: {batch_added} entities created out of {len(batch)} attempted")

            except Exception as e:
                logger.error(f"Failed to add entity batch: {e}", exc_info=True)
                # Don't increment added_count if batch failed

        return added_count

    async def add_relationships_batch(
        self,
        relationships: List[Relationship],
        batch_size: int = 1000
    ) -> int:
        """
        Add relationships to the graph in batches.

        Args:
            relationships: List of relationships to add
            batch_size: Number of relationships per batch

        Returns:
            Number of relationships added successfully
        """
        if not self._driver or not relationships:
            return 0

        added_count = 0

        # Process in batches
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i + batch_size]
            batch_params = []

            for rel in batch:
                rel_data = {
                    "id": rel.id,
                    "source_id": rel.source_entity_id,
                    "target_id": rel.target_entity_id,
                    "relationship_type": rel.relationship_type.value if hasattr(rel.relationship_type, 'value') else str(rel.relationship_type),
                    "strength": rel.strength,
                    "line_number": rel.line_number or 0,
                    "context": rel.context or "",
                    "confidence": rel.confidence,
                    "frequency": rel.frequency,
                    "created_at": rel.created_at.isoformat()
                }
                batch_params.append(rel_data)

            try:
                with self._driver.session(database=self.database) as session:
                    result = session.run(
                        """
                        UNWIND $relationships as rel
                        MATCH (source:Entity {id: rel.source_id})
                        MATCH (target:Entity {id: rel.target_id})
                        CREATE (source)-[r:RELATES {
                            id: rel.id,
                            relationship_type: rel.relationship_type,
                            strength: rel.strength,
                            line_number: rel.line_number,
                            context: rel.context,
                            confidence: rel.confidence,
                            frequency: rel.frequency,
                            created_at: rel.created_at
                        }]->(target)
                        """,
                        relationships=batch_params
                    )
                    # Consume result to get accurate statistics
                    summary = result.consume()
                    batch_added = summary.counters.relationships_created
                    added_count += batch_added
                    logger.debug(f"Added batch: {batch_added} relationships created out of {len(batch)} attempted")

            except Exception as e:
                logger.error(f"Failed to add relationship batch: {e}")

        return added_count

    async def get_entities_by_repository(
        self,
        repository_id: str,
        entity_types: Optional[List[EntityType]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get entities for a repository.

        Args:
            repository_id: Repository identifier
            entity_types: Optional filter by entity types
            limit: Optional limit on number of results

        Returns:
            List of entity dictionaries
        """
        if not self._driver:
            return []

        query = "MATCH (e:Entity {repository_id: $repository_id})"
        params = {"repository_id": repository_id}

        if entity_types:
            type_values = [et.value for et in entity_types]
            query += " WHERE e.entity_type IN $entity_types"
            params["entity_types"] = type_values

        query += " RETURN e"

        if limit:
            query += f" LIMIT {limit}"

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(query, params)
                return [record["e"] for record in result]
        except Exception as e:
            logger.error(f"Failed to get entities: {e}")
            return []

    async def get_relationships_by_entity(
        self,
        entity_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        direction: str = "both"  # "outgoing", "incoming", "both"
    ) -> List[Dict[str, Any]]:
        """
        Get relationships for an entity.

        Args:
            entity_id: Entity identifier
            relationship_types: Optional filter by relationship types
            direction: Relationship direction to consider

        Returns:
            List of relationship dictionaries
        """
        if not self._driver:
            return []

        if direction == "outgoing":
            pattern = "(e:Entity {id: $entity_id})-[r:RELATES]->()"
        elif direction == "incoming":
            pattern = "()-[r:RELATES]->(e:Entity {id: $entity_id})"
        else:  # both
            pattern = "(e:Entity {id: $entity_id})-[r:RELATES]-()"

        query = f"MATCH {pattern}"
        params = {"entity_id": entity_id}

        if relationship_types:
            type_values = [rt.value for rt in relationship_types]
            query += " WHERE r.relationship_type IN $relationship_types"
            params["relationship_types"] = type_values

        query += " RETURN r"

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(query, params)
                return [record["r"] for record in result]
        except Exception as e:
            logger.error(f"Failed to get relationships: {e}")
            return []

    async def find_path_between_entities(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find shortest path between two entities.

        Args:
            source_id: Source entity identifier
            target_id: Target entity identifier
            max_depth: Maximum path depth

        Returns:
            List of path dictionaries
        """
        if not self._driver:
            return []

        query = """
        MATCH path = shortestPath(
            (source:Entity {id: $source_id})-[*1..%d]-(target:Entity {id: $target_id})
        )
        RETURN path
        """ % max_depth

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(query, source_id=source_id, target_id=target_id)
                return [record["path"] for record in result]
        except Exception as e:
            logger.error(f"Failed to find path: {e}")
            return []

    async def calculate_graph_metrics(self, repository_id: str) -> Dict[str, Any]:
        """
        Calculate graph metrics for a repository.

        Args:
            repository_id: Repository identifier

        Returns:
            Dictionary of graph metrics
        """
        if not self._driver:
            return {}

        metrics = {}

        try:
            with self._driver.session(database=self.database) as session:
                # Entity count by type
                result = session.run(
                    """
                    MATCH (e:Entity {repository_id: $repository_id})
                    RETURN e.entity_type as type, count(e) as count
                    """,
                    repository_id=repository_id
                )
                entity_counts = {record["type"]: record["count"] for record in result}
                metrics["entity_counts"] = entity_counts

                # Relationship count by type
                result = session.run(
                    """
                    MATCH (e1:Entity {repository_id: $repository_id})-[r:RELATES]-(e2:Entity {repository_id: $repository_id})
                    RETURN r.relationship_type as type, count(r) as count
                    """,
                    repository_id=repository_id
                )
                relationship_counts = {record["type"]: record["count"] for record in result}
                metrics["relationship_counts"] = relationship_counts

                # Total counts
                metrics["total_entities"] = sum(entity_counts.values())
                metrics["total_relationships"] = sum(relationship_counts.values())

                # Average degree (simplified)
                if metrics["total_entities"] > 0:
                    metrics["average_degree"] = (2 * metrics["total_relationships"]) / metrics["total_entities"]
                else:
                    metrics["average_degree"] = 0.0

        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")

        return metrics

    async def delete_repository_graph(self, repository_id: str) -> bool:
        """
        Delete all data for a repository.

        Args:
            repository_id: Repository identifier

        Returns:
            True if deletion successful, False otherwise
        """
        if not self._driver:
            return False

        try:
            with self._driver.session(database=self.database) as session:
                # Delete entities and their relationships
                session.run(
                    """
                    MATCH (e:Entity {repository_id: $repository_id})
                    DETACH DELETE e
                    """,
                    repository_id=repository_id
                )

                # Delete repository node
                session.run(
                    """
                    MATCH (r:Repository {id: $repository_id})
                    DELETE r
                    """,
                    repository_id=repository_id
                )

                logger.info(f"Deleted graph data for repository {repository_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete repository graph: {e}")
            return False

    async def get_graph_statistics(self, repository_id: str) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        metrics = await self.calculate_graph_metrics(repository_id)

        if not self._driver:
            return metrics

        try:
            with self._driver.session(database=self.database) as session:
                # Language distribution
                result = session.run(
                    """
                    MATCH (e:Entity {repository_id: $repository_id})
                    RETURN e.language as language, count(e) as count
                    """,
                    repository_id=repository_id
                )
                language_dist = {record["language"]: record["count"] for record in result}
                metrics["language_distribution"] = language_dist

                # File count
                result = session.run(
                    """
                    MATCH (e:Entity {repository_id: $repository_id, entity_type: 'FILE'})
                    RETURN count(e) as file_count
                    """,
                    repository_id=repository_id
                )
                record = result.single()
                metrics["file_count"] = record["file_count"] if record else 0

        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")

        return metrics

    async def list_repositories(self) -> List[Dict[str, Any]]:
        """List all repositories in the graph database."""
        if not self._driver:
            return []

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (r:Repository)
                    OPTIONAL MATCH (e:Entity {repository_id: r.id})
                    OPTIONAL MATCH (e)-[rel:RELATES]-()
                    RETURN r.id as repository_id,
                           r.name as repository_name,
                           r.path as repository_path,
                           r.created_at as created_at,
                           r.updated_at as updated_at,
                           r.status as status,
                           count(DISTINCT e) as entity_count,
                           count(DISTINCT rel) as relationship_count
                    ORDER BY r.updated_at DESC
                    """
                )

                repositories = []
                for record in result:
                    repositories.append({
                        "repository_id": record["repository_id"],
                        "repository_name": record["repository_name"],
                        "repository_path": record["repository_path"],
                        "created_at": record["created_at"],
                        "updated_at": record["updated_at"],
                        "status": record["status"],
                        "entity_count": record["entity_count"] or 0,
                        "relationship_count": record["relationship_count"] or 0
                    })

                return repositories

        except Exception as e:
            logger.error(f"Failed to list repositories: {e}")
            return []