"""
Repository manager service coordinating parsing workflow.

Orchestrates the complete process of repository analysis, entity extraction,
graph construction, and embedding generation.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from uuid import uuid4

from repo_kgraph.models.repository import Repository, RepositoryStatus
from repo_kgraph.models.knowledge_graph import KnowledgeGraph, GraphStatus
from repo_kgraph.models.code_entity import CodeEntity
from repo_kgraph.models.relationship import Relationship
from repo_kgraph.services.parser import CodeParser, ParsingStats
from repo_kgraph.services.graph_builder import GraphBuilder
from repo_kgraph.services.embedding import EmbeddingService
from repo_kgraph.lib.source_filter import SourceCodeFilter, create_source_only_filter


logger = logging.getLogger(__name__)


class RepositoryManagerError(Exception):
    """Raised when repository management operations fail."""
    pass


class RepositoryManager:
    """
    Service coordinating the complete repository processing workflow.

    Manages the end-to-end process from repository parsing through
    graph construction to embedding generation and indexing.
    """

    def __init__(
        self,
        parser: CodeParser,
        graph_builder: GraphBuilder,
        embedding_service: EmbeddingService,
        max_concurrent_files: int = 10,
        batch_size: int = 1000
    ):
        """
        Initialize the repository manager.

        Args:
            parser: Code parsing service
            graph_builder: Graph database service
            embedding_service: Embedding generation service
            max_concurrent_files: Maximum files to parse concurrently
            batch_size: Batch size for database operations
        """
        self.parser = parser
        self.graph_builder = graph_builder
        self.embedding_service = embedding_service
        self.max_concurrent_files = max_concurrent_files
        self.batch_size = batch_size

        # Track ongoing operations
        self._active_operations: Dict[str, Dict[str, Any]] = {}
        self._parsing_results: Dict[str, ParsingStats] = {}

    async def process_repository(
        self,
        repository_path: str,
        repository_id: Optional[str] = None,
        incremental: bool = False,
        languages: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ) -> Tuple[Repository, KnowledgeGraph]:
        """
        Process a repository completely from parsing to indexing.

        Args:
            repository_path: Path to repository directory
            repository_id: Optional repository ID (generates if not provided)
            incremental: Whether to perform incremental processing
            languages: List of languages to include
            exclude_patterns: File patterns to exclude
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of Repository and KnowledgeGraph objects

        Raises:
            RepositoryManagerError: If processing fails
        """
        start_time = time.time()
        repository_id = repository_id or str(uuid4())

        try:
            # Validate repository path
            repo_path = Path(repository_path)
            if not repo_path.exists() or not repo_path.is_dir():
                raise RepositoryManagerError(f"Repository path does not exist or is not a directory: {repository_path}")

            # Calculate repository statistics
            total_size = 0
            file_count = 0
            for file_path in repo_path.rglob('*'):
                if file_path.is_file():
                    file_count += 1
                    try:
                        total_size += file_path.stat().st_size
                    except (OSError, IOError):
                        pass  # Skip files we can't read

            # Create repository object
            repository = Repository(
                id=repository_id,
                name=repo_path.name,
                path=str(repo_path.absolute()),
                size_bytes=total_size,
                file_count=file_count,
                status=RepositoryStatus.INDEXING,
                created_at=datetime.utcnow()
            )

            # Track operation
            self._active_operations[repository_id] = {
                "status": "starting",
                "progress": 0.0,
                "start_time": start_time,
                "current_phase": "initialization",
                "repository_path": repository_path
            }

            logger.info(f"Starting repository processing: {repository.name} ({repository_id})")

            # Phase 1: Initialize database schema
            await self._update_progress(repository_id, 5.0, "database_initialization", progress_callback)
            await self.graph_builder.initialize_schema()

            # Phase 2: Create repository in graph
            await self._update_progress(repository_id, 10.0, "graph_creation", progress_callback)
            knowledge_graph = await self.graph_builder.create_repository_graph(repository)

            # Phase 3: Parse repository (traditional approach - collect all first)
            await self._update_progress(repository_id, 15.0, "parsing", progress_callback)
            
            # Convert language filters
            include_patterns = None
            if languages:
                # Create patterns for specified languages
                extensions = {
                    "python": ["*.py"],
                    "javascript": ["*.js", "*.jsx"],
                    "typescript": ["*.ts", "*.tsx"],
                    "java": ["*.java"],
                    "c": ["*.c", "*.h"],
                    "cpp": ["*.cpp", "*.cxx", "*.cc", "*.hpp"],
                    "csharp": ["*.cs"],
                    "go": ["*.go"],
                    "rust": ["*.rs"],
                    "ruby": ["*.rb"],
                    "php": ["*.php"],
                    "swift": ["*.swift"],
                    "kotlin": ["*.kt"],
                    "scala": ["*.scala"]
                }
                include_patterns = []
                for lang in languages:
                    include_patterns.extend(extensions.get(lang.lower(), []))

            # Parse directory and collect all entities and relationships (stable approach)
            parsing_stats = await self.parser.parse_directory(
                directory_path=repository_path,
                repository_id=repository_id,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                max_workers=self.max_concurrent_files
            )

            # Store parsing results (statistics only)
            self._parsing_results[repository_id] = parsing_stats

            # Update progress during parsing
            await self._update_progress(repository_id, 45.0, "parsing_complete", progress_callback)

            logger.info(f"Parsing completed: {parsing_stats.parsed_files} files, {parsing_stats.total_entities} entities, {parsing_stats.total_relationships} relationships")

            # Phase 4: Add entities and relationships to graph in batches
            await self._update_progress(repository_id, 50.0, "graph_insertion", progress_callback)
            
            # For the stable baseline, we'll collect all entities and relationships first
            # and then add them to the graph in batches
            
            # Collect all entities and relationships by re-parsing with streaming
            # but collecting them instead of processing immediately
            all_entities = []
            all_relationships = []
            
            async def collect_entity_callback(entity):
                all_entities.append(entity)
                
            async def collect_relationship_callback(relationship):
                all_relationships.append(relationship)
            
            # Re-parse with collection callbacks
            await self.parser.stream_directory_entities(
                directory_path=repository_path,
                repository_id=repository_id,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                entity_callback=collect_entity_callback,
                relationship_callback=collect_relationship_callback
            )

            # Add entities to graph in batches
            entities_added = 0
            for i in range(0, len(all_entities), self.batch_size):
                batch = all_entities[i:i + self.batch_size]
                batch_added = await self.graph_builder.add_entities_batch(batch)
                entities_added += batch_added
                logger.debug(f"Added entity batch: {batch_added} entities")

            # Add relationships to graph in batches
            relationships_added = 0
            for i in range(0, len(all_relationships), self.batch_size):
                batch = all_relationships[i:i + self.batch_size]
                batch_added = await self.graph_builder.add_relationships_batch(batch)
                relationships_added += batch_added
                logger.debug(f"Added relationship batch: {batch_added} relationships")

            logger.info(f"Added {entities_added} entities to graph")
            logger.info(f"Added {relationships_added} relationships to graph")

            # Phase 5: Generate embeddings with unlimited timeout for development
            await self._update_progress(repository_id, 70.0, "embedding_generation", progress_callback)
            logger.info(f"Starting embedding generation for {len(all_entities)} entities (may take time on resource-constrained systems)")

            try:
                embeddings_created = await self._generate_embeddings(all_entities, repository_id, progress_callback)
                logger.info(f"Generated {embeddings_created} embeddings successfully")
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}", exc_info=True)
                embeddings_created = 0

            # Phase 6: Store embeddings with unlimited timeout
            await self._update_progress(repository_id, 90.0, "embedding_storage", progress_callback)

            if embeddings_created > 0:
                try:
                    collection_name = f"repo_{repository_id}"
                    logger.info(f"Storing embeddings in ChromaDB collection: {collection_name}")
                    embeddings_stored = await self.embedding_service.store_embeddings_in_chroma(
                        all_entities, collection_name
                    )
                    logger.info(f"Stored {embeddings_stored} embeddings successfully")
                except Exception as e:
                    logger.error(f"Embedding storage failed: {e}", exc_info=True)
                    embeddings_stored = 0
            else:
                embeddings_stored = 0
                logger.info("No embeddings to store")

            # Phase 7: Calculate final metrics
            await self._update_progress(repository_id, 95.0, "metrics_calculation", progress_callback)
            graph_metrics = await self.graph_builder.calculate_graph_metrics(repository_id)

            # Update repository status
            end_time = time.time()
            processing_time = end_time - start_time

            repository.status = RepositoryStatus.INDEXED
            repository.entity_count = entities_added
            repository.relationship_count = relationships_added
            repository.file_count = parsing_stats.parsed_files
            repository.language_stats = self._calculate_language_stats(all_entities)
            repository.last_indexed = datetime.utcnow()
            repository.indexing_time_ms = int(processing_time * 1000)
            repository.parsing_errors = parsing_stats.failed_files

            # Update knowledge graph with calculated metrics
            knowledge_graph.status = GraphStatus.READY
            knowledge_graph.entity_count = entities_added
            knowledge_graph.relationship_count = relationships_added
            knowledge_graph.construction_time = processing_time
            knowledge_graph.calculate_density()
            knowledge_graph.calculate_average_degree()
            knowledge_graph.updated_at = datetime.utcnow()

            # Final progress update
            await self._update_progress(repository_id, 100.0, "completed", progress_callback)

            # Clean up operation tracking
            del self._active_operations[repository_id]

            logger.info(f"Repository processing completed in {processing_time:.2f}s: {repository.name}")

            return repository, knowledge_graph

        except Exception as e:
            # Update status on error
            if repository_id in self._active_operations:
                self._active_operations[repository_id]["status"] = "failed"
                self._active_operations[repository_id]["error"] = str(e)

            logger.error(f"Repository processing failed: {e}")
            raise RepositoryManagerError(f"Failed to process repository: {e}")

    async def _parse_repository_streaming(
        self,
        repository_path: str,
        repository_id: str,
        languages: Optional[List[str]],
        exclude_patterns: Optional[List[str]],
        entity_callback: callable,
        relationship_callback: callable,
        progress_callback: Optional[callable]
    ) -> ParsingStats:
        """Parse repository and collect all entities and relationships."""
        try:
            # Convert language filters
            include_patterns = None
            if languages:
                # Create patterns for specified languages
                extensions = {
                    "python": ["*.py"],
                    "javascript": ["*.js", "*.jsx"],
                    "typescript": ["*.ts", "*.tsx"],
                    "java": ["*.java"],
                    "c": ["*.c", "*.h"],
                    "cpp": ["*.cpp", "*.cxx", "*.cc", "*.hpp"],
                    "csharp": ["*.cs"],
                    "go": ["*.go"],
                    "rust": ["*.rs"],
                    "ruby": ["*.rb"],
                    "php": ["*.php"],
                    "swift": ["*.swift"],
                    "kotlin": ["*.kt"],
                    "scala": ["*.scala"]
                }
                include_patterns = []
                for lang in languages:
                    include_patterns.extend(extensions.get(lang.lower(), []))

            # Parse directory and collect all entities and relationships
            parsing_stats = await self.parser.parse_directory(
                directory_path=repository_path,
                repository_id=repository_id,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                max_workers=self.max_concurrent_files
            )

            # Store parsing results (statistics only)
            self._parsing_results[repository_id] = parsing_stats

            # Update progress during parsing
            await self._update_progress(repository_id, 45.0, "parsing_complete", progress_callback)

            return parsing_stats

        except Exception as e:
            logger.error(f"Repository parsing failed: {e}")
            raise

    # Note: _collect_parsing_results method is no longer needed
    # Entities and relationships are now processed directly via streaming callbacks

    async def _process_entity_batch(self, entities: List[CodeEntity], relationships: List[Relationship], repository_id: str):
        """Process a batch of entities immediately to save memory."""
        try:
            # Store entities in graph database
            if entities:
                await self.graph_builder.add_entities_batch(entities, len(entities))

            # Store relationships in graph database
            if relationships:
                await self.graph_builder.add_relationships_batch(relationships, len(relationships))

            # Generate and store embeddings for this batch
            if entities:
                await self._generate_embeddings_batch(entities, repository_id)

        except Exception as e:
            logger.error(f"Failed to process entity batch: {e}")

    async def _generate_embeddings_batch(self, entities: List[CodeEntity], repository_id: str):
        """Generate embeddings for a batch of entities."""
        try:
            embeddings_created = await self.embedding_service.embed_code_entities_batch(entities)
            logger.debug(f"Generated {embeddings_created} embeddings for batch")
        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch: {e}")

    async def _generate_embeddings(
        self,
        entities: List[CodeEntity],
        repository_id: str,
        progress_callback: Optional[callable]
    ) -> int:
        """Generate embeddings for all entities."""
        try:
            embeddings_created = await self.embedding_service.embed_code_entities_batch(entities)

            # Update progress
            await self._update_progress(repository_id, 85.0, "embeddings_generated", progress_callback)

            return embeddings_created

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def _calculate_language_stats(self, entities: List[CodeEntity]) -> Dict[str, int]:
        """Calculate language distribution statistics."""
        language_counts = {}
        for entity in entities:
            language = entity.language
            language_counts[language] = language_counts.get(language, 0) + 1
        return language_counts

    async def _update_progress(
        self,
        repository_id: str,
        progress: float,
        phase: str,
        progress_callback: Optional[callable]
    ) -> None:
        """Update processing progress."""
        if repository_id in self._active_operations:
            self._active_operations[repository_id]["progress"] = progress
            self._active_operations[repository_id]["current_phase"] = phase

        if progress_callback:
            try:
                await progress_callback(repository_id, progress, phase)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    async def get_processing_status(self, repository_id: str) -> Optional[Dict[str, Any]]:
        """Get current processing status for a repository."""
        return self._active_operations.get(repository_id)

    async def update_repository_incremental(
        self,
        repository_id: str,
        changed_files: List[str],
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Perform incremental update for changed files.

        Args:
            repository_id: Repository identifier
            changed_files: List of changed file paths
            progress_callback: Optional progress callback

        Returns:
            True if update successful, False otherwise
        """
        try:
            start_time = time.time()

            # Track incremental operation
            operation_id = f"{repository_id}_incremental_{int(start_time)}"
            self._active_operations[operation_id] = {
                "status": "incremental_update",
                "progress": 0.0,
                "start_time": start_time,
                "current_phase": "starting",
                "changed_files": len(changed_files)
            }

            logger.info(f"Starting incremental update for {len(changed_files)} files")

            # Phase 1: Parse changed files
            await self._update_progress(operation_id, 10.0, "parsing_changed_files", progress_callback)

            parsed_entities = []
            parsed_relationships = []

            for i, file_path in enumerate(changed_files):
                try:
                    # Parse individual file
                    parse_result = await self.parser.parse_file(file_path, repository_id)

                    if parse_result.success:
                        parsed_entities.extend(parse_result.entities)
                        parsed_relationships.extend(parse_result.relationships)

                    # Update progress
                    file_progress = 10.0 + (40.0 * (i + 1) / len(changed_files))
                    await self._update_progress(operation_id, file_progress, "parsing_files", progress_callback)

                except Exception as e:
                    logger.warning(f"Failed to parse file {file_path}: {e}")

            # Phase 2: Update graph
            await self._update_progress(operation_id, 60.0, "updating_graph", progress_callback)

            # Remove old entities for these files (simplified approach)
            # In practice, would need more sophisticated change detection

            # Add new entities
            entities_added = await self.graph_builder.add_entities_batch(parsed_entities, self.batch_size)

            # Add new relationships
            relationships_added = await self.graph_builder.add_relationships_batch(parsed_relationships, self.batch_size)

            # Phase 3: Update embeddings
            await self._update_progress(operation_id, 80.0, "updating_embeddings", progress_callback)

            embeddings_created = await self.embedding_service.embed_code_entities_batch(parsed_entities)

            collection_name = f"repo_{repository_id}"
            await self.embedding_service.store_embeddings_in_chroma(parsed_entities, collection_name)

            # Phase 4: Update metrics
            await self._update_progress(operation_id, 95.0, "updating_metrics", progress_callback)

            graph_metrics = await self.graph_builder.calculate_graph_metrics(repository_id)

            # Complete
            await self._update_progress(operation_id, 100.0, "completed", progress_callback)

            # Clean up
            del self._active_operations[operation_id]

            end_time = time.time()
            logger.info(f"Incremental update completed in {end_time - start_time:.2f}s")

            return True

        except Exception as e:
            logger.error(f"Incremental update failed: {e}")
            return False

    async def delete_repository(self, repository_id: str) -> bool:
        """
        Delete all data for a repository.

        Args:
            repository_id: Repository identifier

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            # Delete from graph database
            graph_deleted = await self.graph_builder.delete_repository_graph(repository_id)

            # Delete from vector database
            if self.embedding_service._chroma_client:
                try:
                    collection_name = f"repo_{repository_id}"
                    self.embedding_service._chroma_client.delete_collection(collection_name)
                    logger.info(f"Deleted vector collection {collection_name}")
                except Exception as e:
                    logger.warning(f"Failed to delete vector collection: {e}")

            # Remove from active operations if present
            if repository_id in self._active_operations:
                del self._active_operations[repository_id]

            logger.info(f"Repository {repository_id} deleted successfully")
            return graph_deleted

        except Exception as e:
            logger.error(f"Repository deletion failed: {e}")
            return False

    async def get_repository_statistics(self, repository_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a repository."""
        try:
            graph_stats = await self.graph_builder.get_graph_statistics(repository_id)

            # Add embedding statistics
            embedding_stats = {
                "embedding_model": self.embedding_service.model_name,
                "embedding_provider": self.embedding_service.embedding_provider,
                "embedding_dimension": self.embedding_service.get_embedding_dimension(),
                "embeddings_available": self.embedding_service.is_available()
            }

            # Combine statistics
            stats = {
                **graph_stats,
                "embedding_info": embedding_stats,
                "last_updated": datetime.utcnow().isoformat()
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get repository statistics: {e}")
            return {}

    def list_active_operations(self) -> Dict[str, Dict[str, Any]]:
        """List all currently active operations."""
        return dict(self._active_operations)

    async def list_repositories(self) -> List[Dict[str, Any]]:
        """List all indexed repositories."""
        try:
            repositories = await self.graph_builder.list_repositories()
            logger.info(f"Retrieved {len(repositories)} repositories")
            return repositories
        except Exception as e:
            logger.error(f"Failed to list repositories: {e}")
            raise RepositoryManagerError(f"Repository listing failed: {e}")

    async def cleanup(self) -> None:
        """Cleanup resources and connections."""
        try:
            # Close graph builder
            self.graph_builder.close()

            # Cleanup embedding service
            await self.embedding_service.cleanup()

            # Clear active operations
            self._active_operations.clear()

            logger.info("Repository manager cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()