"""
Context retrieval service for semantic code search.

Implements hybrid search combining vector similarity and graph traversal
to find relevant code context for coding tasks.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
import heapq

from repo_kgraph.models.code_entity import CodeEntity, EntityType
from repo_kgraph.models.relationship import Relationship, RelationshipType
from repo_kgraph.models.query import Query
from repo_kgraph.models.context_result import ContextResult, RetrievalReason
from repo_kgraph.services.embedding import EmbeddingService
from repo_kgraph.services.graph_builder import GraphBuilder
from repo_kgraph.services.multi_hop_retriever import MultiHopRetriever
from repo_kgraph.services.task_context_adapter import TaskContextAdapter


logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Raised when context retrieval fails."""
    pass


class ContextRetriever:
    """
    Service for retrieving relevant code context using hybrid search.

    Combines semantic similarity search with graph traversal to find
    the most relevant code entities and relationships for a given task.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        graph_builder: GraphBuilder,
        default_max_results: int = 20,
        default_confidence_threshold: float = 0.3,
        graph_traversal_depth: int = 2,
        similarity_weight: float = 0.7,
        graph_weight: float = 0.3
    ):
        """
        Initialize the context retriever.

        Args:
            embedding_service: Service for embedding generation and similarity search
            graph_builder: Service for graph database operations
            default_max_results: Default maximum number of results to return
            default_confidence_threshold: Default confidence threshold for filtering
            graph_traversal_depth: Maximum depth for graph traversal
            similarity_weight: Weight for similarity scores in hybrid scoring
            graph_weight: Weight for graph scores in hybrid scoring
        """
        self.embedding_service = embedding_service
        self.graph_builder = graph_builder
        self.default_max_results = default_max_results
        self.default_confidence_threshold = default_confidence_threshold
        self.graph_traversal_depth = graph_traversal_depth
        self.similarity_weight = similarity_weight
        self.graph_weight = graph_weight

        # Initialize multi-hop retriever
        self.multi_hop_retriever = MultiHopRetriever(
            embedding_service=embedding_service,
            graph_builder=graph_builder,
            max_hops=graph_traversal_depth
        )

        # Initialize task context adapter
        self.task_context_adapter = TaskContextAdapter()

    async def retrieve_context(
        self,
        query: Query,
        max_results: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        include_related: bool = True
    ) -> List[ContextResult]:
        """
        Retrieve relevant code context for a query.

        Args:
            query: Query object with task description and parameters
            max_results: Maximum number of results to return
            confidence_threshold: Minimum confidence threshold for results
            include_related: Whether to include related entities via graph traversal

        Returns:
            List of context results ranked by relevance
        """
        max_results = max_results or self.default_max_results
        confidence_threshold = confidence_threshold or self.default_confidence_threshold

        try:
            # Step 1: Generate query embedding
            query_embedded = await self.embedding_service.embed_query(query)
            if not query_embedded:
                logger.warning("Failed to generate query embedding")
                return []

            # Step 2: Semantic similarity search
            collection_name = f"repo_{query.repository_id}"
            similarity_results = await self.embedding_service.similarity_search(
                query_text=query.query_text,
                collection_name=collection_name,
                top_k=max_results * 2,  # Get more candidates for filtering
                filters=self._build_search_filters(query)
            )

            # Step 3: Convert similarity results to context results
            context_candidates = await self._convert_similarity_results(
                similarity_results, query, confidence_threshold
            )
            logger.info(f"After similarity conversion: {len(context_candidates)} context candidates")

            # Step 4: Multi-hop graph expansion if enabled
            if include_related and context_candidates:
                # Convert context candidates back to entities for multi-hop
                logger.info(f"Original candidate scores before multi-hop: {[f'{c.entity_id[:8]}:{c.relevance_score:.3f}' for c in context_candidates[:3]]}")

                initial_entities = []
                for candidate in context_candidates:
                    node = await self.graph_builder.get_entity_by_id(candidate.entity_id)
                    if node:
                        # Convert Neo4j Node to CodeEntity
                        entity = self._node_to_code_entity(node)
                        if entity:
                            initial_entities.append(entity)
                        else:
                            logger.warning(f"Failed to convert node to entity for {candidate.entity_id}")
                    else:
                        logger.warning(f"No Neo4j node found for entity_id: {candidate.entity_id}")
                logger.info(f"Multi-hop: {len(initial_entities)} initial entities from {len(context_candidates)} candidates")

                # Perform multi-hop expansion
                context_graph = await self.multi_hop_retriever.retrieve_with_multi_hop_context(
                    query.query_text,
                    query.repository_id,
                    initial_entities,
                    max_results * 2  # Get more candidates for better filtering
                )

                # Convert context graph back to context results
                multi_hop_results = self.multi_hop_retriever.convert_to_context_results(
                    context_graph, max_results, query.id
                )
                logger.info(f"Multi-hop results: {len(multi_hop_results)} final results")

                # CRITICAL FIX: Preserve original similarity scores from context_candidates
                # Create a mapping of entity_id -> original_score
                original_scores = {c.entity_id: c.relevance_score for c in context_candidates}

                # Restore original scores for entities that had them
                for result in multi_hop_results:
                    if result.entity_id in original_scores:
                        result.relevance_score = original_scores[result.entity_id]
                        logger.debug(f"Restored score for {result.entity_id[:8]}: {result.relevance_score:.3f}")

                for r in multi_hop_results[:3]:  # Log first 3 for debugging
                    logger.info(f"Multi-hop result {r.entity_id}: score={r.relevance_score:.6f}")

                # Replace candidates with multi-hop enhanced results
                context_candidates = multi_hop_results

            # Step 5: Re-rank and filter results
            final_results = await self._rerank_and_filter_results(
                context_candidates, query, max_results, confidence_threshold
            )
            logger.info(f"After re-ranking: {len(final_results)} results")

            # Step 6: Apply task-specific context adaptation
            adapted_context = self.task_context_adapter.adapt_context_for_task(
                task_description=query.query_text,
                context_results=final_results
            )

            # Combine primary and supporting context
            adapted_results = adapted_context.primary_context + adapted_context.supporting_context

            # Step 7: Add metadata and return
            for result in adapted_results:
                result.metadata.update({
                    "retrieval_method": "hybrid_search_with_task_adaptation",
                    "similarity_weight": self.similarity_weight,
                    "graph_weight": self.graph_weight,
                    "graph_traversal_depth": self.graph_traversal_depth,
                    "total_candidates": len(context_candidates),
                    "task_type": adapted_context.task_specific_metadata.get("task_type"),
                    "context_summary": adapted_context.context_summary,
                    "recommended_actions": adapted_context.recommended_actions
                })

            return adapted_results

        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            raise RetrievalError(f"Failed to retrieve context: {e}")

    def _build_search_filters(self, query: Query) -> Optional[Dict[str, Any]]:
        """Build metadata filters for similarity search."""
        # For now, only use repository_id filter to avoid ChromaDB issues
        # Future enhancement: Implement proper multi-condition filtering
        return {"repository_id": query.repository_id}

    async def _convert_similarity_results(
        self,
        similarity_results: List[Dict[str, Any]],
        query: Query,
        confidence_threshold: float
    ) -> List[ContextResult]:
        """Convert similarity search results to ContextResult objects."""
        context_results = []

        for result in similarity_results:
            try:
                # Extract metadata
                metadata = result.get("metadata", {})
                similarity_score = result.get("similarity_score", 0.0)

                # Apply confidence threshold
                logger.debug(f"Similarity score: {similarity_score}, threshold: {confidence_threshold}")
                if similarity_score < confidence_threshold:
                    logger.debug(f"Filtering out result with similarity {similarity_score} < {confidence_threshold}")
                    continue

                # Create context result
                entity_type_value = metadata.get("entity_type", "FILE")
                if isinstance(entity_type_value, EntityType):
                    entity_type = entity_type_value
                else:
                    try:
                        entity_type = EntityType[entity_type_value]
                    except KeyError:
                        # Fallback to FILE if the entity type is not recognized
                        entity_type = EntityType.FILE
                
                context_result = ContextResult(
                    query_id=query.id,
                    entity_id=result["entity_id"],
                    entity_name=metadata.get("name", "unknown"),
                    entity_type=entity_type.value if hasattr(entity_type, 'value') else str(entity_type),
                    file_path=metadata.get("file_path", ""),
                    start_line=metadata.get("start_line", 1),
                    end_line=metadata.get("end_line", 1),
                    relevance_score=similarity_score,
                    rank_position=1,  # This will be set properly when ranking
                    retrieval_reason=RetrievalReason.SEMANTIC_SIMILARITY,
                    context_snippet=self._extract_context_snippet(result.get("document", ""))
                )

                context_results.append(context_result)

            except Exception as e:
                logger.warning(f"Failed to convert similarity result: {e}")
                continue

        return context_results

    async def _expand_with_graph_traversal(
        self,
        initial_results: List[ContextResult],
        query: Query,
        max_results: int
    ) -> List[ContextResult]:
        """Expand results using graph traversal to find related entities."""
        expanded_results = []
        visited_entities = {result.entity_id for result in initial_results}

        for initial_result in initial_results[:max_results // 2]:  # Limit expansion seed
            try:
                # Get relationships for this entity
                relationships = await self.graph_builder.get_relationships_by_entity(
                    entity_id=initial_result.entity_id,
                    direction="both"
                )

                # Traverse relationships
                for rel_data in relationships:
                    related_entity_id = self._get_related_entity_id(
                        rel_data, initial_result.entity_id
                    )

                    if related_entity_id and related_entity_id not in visited_entities:
                        # Get entity data
                        entity_data = await self._get_entity_data(
                            related_entity_id, query.repository_id
                        )

                        if entity_data:
                            # Calculate graph-based relevance score
                            graph_score = self._calculate_graph_relevance_score(
                                rel_data, initial_result.relevance_score
                            )

                            # Create context result for related entity
                            related_result = self._create_context_result_from_entity(
                                entity_data, query, graph_score, "graph_traversal"
                            )

                            if related_result and related_result.confidence_score >= query.confidence_threshold:
                                expanded_results.append(related_result)
                                visited_entities.add(related_entity_id)

            except Exception as e:
                logger.warning(f"Graph traversal failed for entity {initial_result.entity_id}: {e}")
                continue

        return expanded_results

    def _get_related_entity_id(self, rel_data: Dict[str, Any], source_entity_id: str) -> Optional[str]:
        """Extract the related entity ID from relationship data."""
        # This would depend on the exact structure returned by the graph database
        # Assuming the relationship data contains source and target entity IDs
        source_id = rel_data.get("source_entity_id")
        target_id = rel_data.get("target_entity_id")

        if source_id == source_entity_id:
            return target_id
        elif target_id == source_entity_id:
            return source_id
        else:
            return None

    async def _get_entity_data(
        self,
        entity_id: str,
        repository_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get entity data from the graph database."""
        try:
            return await self.graph_builder.get_entity_by_id(entity_id)
        except Exception as e:
            logger.warning(f"Failed to get entity data for {entity_id}: {e}")
            return None

    def _calculate_graph_relevance_score(
        self,
        rel_data: Dict[str, Any],
        source_relevance: float
    ) -> float:
        """Calculate relevance score for graph-traversed entity."""
        # Base score from source entity
        base_score = source_relevance

        # Relationship strength
        rel_strength = rel_data.get("strength", 0.5)

        # Relationship type importance
        rel_type = rel_data.get("relationship_type", "")
        type_weights = {
            "CALLS": 0.9,
            "IMPORTS": 0.8,
            "INHERITS": 0.8,
            "CONTAINS": 0.7,
            "REFERENCES": 0.6,
            "DEPENDS_ON": 0.6
        }
        type_weight = type_weights.get(rel_type, 0.5)

        # Decay factor for traversal depth
        decay_factor = 0.7  # Reduce score for indirect relationships

        return base_score * rel_strength * type_weight * decay_factor

    def _create_context_result_from_entity(
        self,
        entity_data: Dict[str, Any],
        query: Query,
        relevance_score: float,
        match_reason: str
    ) -> Optional[ContextResult]:
        """Create ContextResult from entity data."""
        try:
            entity_type_value = entity_data.get("entity_type", "FILE")
            if isinstance(entity_type_value, EntityType):
                entity_type = entity_type_value
            else:
                try:
                    entity_type = EntityType[entity_type_value]
                except KeyError:
                    # Fallback to FILE if the entity type is not recognized
                    entity_type = EntityType.FILE
            
            return ContextResult(
                query_id=query.id,
                entity_id=entity_data.get("id", ""),
                entity_name=entity_data.get("name", "unknown"),
                entity_type=entity_type.value if hasattr(entity_type, 'value') else str(entity_type),
                file_path=entity_data.get("file_path", ""),
                start_line=entity_data.get("start_line", 1),
                end_line=entity_data.get("end_line", 1),
                relevance_score=relevance_score,
                rank_position=1,  # This should be set properly when ranking
                retrieval_reason=RetrievalReason(match_reason) if match_reason in RetrievalReason.__members__ else RetrievalReason.DIRECT_MATCH,
                context_snippet=self._extract_context_snippet(entity_data.get("content", ""))
            )
        except Exception as e:
            logger.warning(f"Failed to create context result: {e}")
            return None

    async def _rerank_and_filter_results(
        self,
        candidates: List[ContextResult],
        query: Query,
        max_results: int,
        confidence_threshold: float
    ) -> List[ContextResult]:
        """Re-rank and filter results using hybrid scoring."""
        # Calculate hybrid scores
        for candidate in candidates:
            candidate.relevance_score = self._calculate_hybrid_score(candidate, query)

        # Filter by confidence threshold
        logger.info(f"Before confidence filter: {len(candidates)} candidates")
        for c in candidates[:3]:  # Log first 3 for debugging
            logger.info(f"Candidate {c.entity_id}: score={c.relevance_score:.6f}, reason={c.retrieval_reason}, threshold={confidence_threshold}")

        filtered_candidates = [
            c for c in candidates
            if c.relevance_score >= confidence_threshold
        ]
        logger.info(f"After confidence filter: {len(filtered_candidates)} candidates")

        # Sort by relevance score
        filtered_candidates.sort(key=lambda x: x.relevance_score, reverse=True)

        # Set rank positions
        for i, candidate in enumerate(filtered_candidates):
            candidate.rank_position = i + 1

        # Apply diversification to avoid too many results from same file
        diversified_results = self._diversify_results(filtered_candidates, max_results)

        # Update rank positions after diversification
        for i, result in enumerate(diversified_results):
            result.rank_position = i + 1

        return diversified_results[:max_results]

    def _calculate_hybrid_score(self, result: ContextResult, query: Query) -> float:
        """Calculate hybrid score combining similarity and graph scores."""
        # Always use the base relevance score as a starting point
        # Don't zero out scores based on retrieval reason - that loses information
        base_score = result.relevance_score

        # Apply entity type preferences
        type_bonus = self._get_entity_type_bonus(result.entity_type, query)

        # Apply recency bonus if applicable
        recency_bonus = 0.0  # Could add based on entity.updated_at

        # Use base score as primary factor, with bonuses
        hybrid_score = base_score + type_bonus + recency_bonus

        return min(hybrid_score, 1.0)  # Cap at 1.0

    def _get_entity_type_bonus(self, entity_type: EntityType, query: Query) -> float:
        """Get bonus score based on entity type preferences."""
        # Prefer certain entity types based on query context
        if query.query_text:
            task_lower = query.query_text.lower()

            if "function" in task_lower or "method" in task_lower:
                if entity_type in [EntityType.FUNCTION, EntityType.METHOD]:
                    return 0.1
            elif "class" in task_lower:
                if entity_type == EntityType.CLASS:
                    return 0.1
            elif "variable" in task_lower:
                if entity_type in [EntityType.VARIABLE, EntityType.CONSTANT]:
                    return 0.1

        return 0.0

    def _diversify_results(
        self,
        results: List[ContextResult],
        max_results: int
    ) -> List[ContextResult]:
        """Diversify results to avoid too many from the same file."""
        diversified = []
        file_counts = {}
        max_per_file = max(1, max_results // 5)  # At most 20% from same file

        for result in results:
            file_path = result.file_path
            current_count = file_counts.get(file_path, 0)

            if current_count < max_per_file or len(diversified) < max_results // 2:
                diversified.append(result)
                file_counts[file_path] = current_count + 1

            if len(diversified) >= max_results:
                break

        return diversified

    def _extract_context_snippet(self, content: str, max_length: int = 200) -> str:
        """Extract a context snippet from content."""
        if not content:
            return ""

        # Take first few lines or characters
        lines = content.split('\n')
        snippet_lines = []
        total_length = 0

        for line in lines:
            if total_length + len(line) > max_length:
                break
            snippet_lines.append(line)
            total_length += len(line)

        snippet = '\n'.join(snippet_lines)
        if len(content) > len(snippet):
            snippet += "..."

        return snippet

    async def search_by_entity_name(
        self,
        repository_id: str,
        entity_name: str,
        entity_types: Optional[List[EntityType]] = None,
        exact_match: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for entities by name.

        Args:
            repository_id: Repository identifier
            entity_name: Name to search for
            entity_types: Optional filter by entity types
            exact_match: Whether to use exact matching

        Returns:
            List of matching entities
        """
        try:
            # This would use the graph database to search by name
            # Implementation depends on specific graph query capabilities
            entities = await self.graph_builder.get_entities_by_repository(
                repository_id=repository_id,
                entity_types=entity_types
            )

            # Filter by name (simplified implementation)
            if exact_match:
                matching_entities = [
                    e for e in entities
                    if e.get("name", "").lower() == entity_name.lower()
                ]
            else:
                matching_entities = [
                    e for e in entities
                    if entity_name.lower() in e.get("name", "").lower()
                ]

            return matching_entities

        except Exception as e:
            logger.error(f"Entity name search failed: {e}")
            return []

    async def get_entity_context(
        self,
        entity_id: str,
        repository_id: str,
        depth: int = 2,
        relationship_types: Optional[List[RelationshipType]] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive context for a specific entity.

        Args:
            entity_id: Entity identifier
            repository_id: Repository identifier
            depth: Traversal depth for related entities
            relationship_types: Optional filter for relationship types

        Returns:
            Dictionary with entity and related context
        """
        try:
            # Get entity data
            entity_data = await self._get_entity_data(entity_id, repository_id)
            if not entity_data:
                return {}

            # Get relationships
            relationships = await self.graph_builder.get_relationships_by_entity(
                entity_id=entity_id,
                relationship_types=relationship_types,
                direction="both"
            )

            # Get related entities (limited depth)
            related_entities = []
            if depth > 0:
                for rel_data in relationships:
                    related_id = self._get_related_entity_id(rel_data, entity_id)
                    if related_id:
                        related_entity = await self._get_entity_data(related_id, repository_id)
                        if related_entity:
                            related_entities.append({
                                "entity": related_entity,
                                "relationship": rel_data
                            })

            return {
                "entity": entity_data,
                "relationships": relationships,
                "related_entities": related_entities,
                "context_depth": depth
            }

        except Exception as e:
            logger.error(f"Failed to get entity context: {e}")
            return {}

    def _node_to_code_entity(self, node) -> Optional[CodeEntity]:
        """
        Convert Neo4j Node to CodeEntity object.

        Args:
            node: Neo4j Node object from database

        Returns:
            CodeEntity object or None if conversion fails
        """
        try:
            # Extract properties from Neo4j node
            props = dict(node.items())

            # Required fields
            entity_id = props.get('id')
            repository_id = props.get('repository_id')
            entity_type = props.get('entity_type')
            name = props.get('name')

            if not all([entity_id, repository_id, entity_type, name]):
                logger.warning(f"Missing required fields in node: {props}")
                return None

            # Create CodeEntity with available properties, providing defaults for required fields
            embedding_vector = props.get('embedding_vector', [])
            # If embedding_vector is empty, provide a dummy vector to satisfy validation
            if not embedding_vector:
                embedding_vector = [0.0] * 768  # SweRankEmbed dimension

            entity = CodeEntity(
                id=entity_id,
                repository_id=repository_id,
                entity_type=EntityType(entity_type),
                name=name,
                qualified_name=props.get('qualified_name', name),
                file_path=props.get('file_path', ''),
                start_line=props.get('start_line', 0),
                end_line=props.get('end_line', 0),
                start_column=props.get('start_column', 0),  # Add missing required field
                end_column=props.get('end_column', 0),      # Add missing required field
                language=props.get('language', 'unknown'),
                content=props.get('content', ''),
                docstring=props.get('docstring'),
                signature=props.get('signature'),
                complexity=props.get('complexity', 1),
                dependencies=props.get('dependencies', []),
                side_effects=props.get('side_effects', []),
                is_public=props.get('is_public', True),
                is_async=props.get('is_async', False),
                return_type=props.get('return_type'),
                parameters=props.get('parameters', []),
                semantic_info=props.get('semantic_info', {}),
                embedding_vector=embedding_vector
            )

            return entity

        except Exception as e:
            logger.error(f"Failed to convert node to CodeEntity: {e}")
            return None