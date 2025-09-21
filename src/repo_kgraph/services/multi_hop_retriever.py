"""
Multi-hop context retrieval service for finding related code across relationships.

Implements multi-hop reasoning to find contextually relevant code entities
by following relationships and building comprehensive context graphs.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque

from repo_kgraph.models.code_entity import CodeEntity, EntityType
from repo_kgraph.models.relationship import Relationship, RelationshipType
from repo_kgraph.models.context_result import ContextResult, RetrievalReason
from repo_kgraph.services.embedding import EmbeddingService
from repo_kgraph.services.graph_builder import GraphBuilder


logger = logging.getLogger(__name__)


@dataclass
class ContextNode:
    """A node in the context graph with relationship information."""
    entity: CodeEntity
    hop_distance: int
    relationship_path: List[str]
    relevance_score: float
    retrieval_reason: str


@dataclass
class ContextGraph:
    """A graph of related entities with their relationships."""
    nodes: Dict[str, ContextNode]
    edges: List[Tuple[str, str, RelationshipType]]
    primary_entities: List[str]
    context_summary: Dict[str, Any]


class MultiHopRetriever:
    """
    Service for multi-hop context retrieval following entity relationships.

    Builds context graphs by traversing relationships to find all relevant
    code entities for a given query, similar to how Cursor analyzes codebases.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        graph_builder: GraphBuilder,
        max_hops: int = 3,
        max_entities_per_hop: int = 10,
        relevance_threshold: float = 0.3
    ):
        """
        Initialize the multi-hop retriever.

        Args:
            embedding_service: Service for semantic similarity
            graph_builder: Service for graph traversal
            max_hops: Maximum number of relationship hops
            max_entities_per_hop: Maximum entities to explore per hop
            relevance_threshold: Minimum relevance score to include
        """
        self.embedding_service = embedding_service
        self.graph_builder = graph_builder
        self.max_hops = max_hops
        self.max_entities_per_hop = max_entities_per_hop
        self.relevance_threshold = relevance_threshold

    async def retrieve_with_multi_hop_context(
        self,
        query: str,
        repository_id: str,
        initial_entities: List[CodeEntity],
        max_results: int = 20
    ) -> ContextGraph:
        """
        Perform multi-hop retrieval to build comprehensive context.

        Args:
            query: Original query text
            repository_id: Repository identifier
            initial_entities: Starting entities from similarity search
            max_results: Maximum entities to return

        Returns:
            ContextGraph with related entities and relationships
        """
        try:
            # Generate query embedding for relevance scoring
            query_embedding = await self.embedding_service.generate_query_embedding(query)

            # Initialize context graph
            context_graph = ContextGraph(
                nodes={},
                edges=[],
                primary_entities=[entity.id for entity in initial_entities],
                context_summary={}
            )

            # Add initial entities as seed nodes
            for entity in initial_entities:
                relevance_score = await self._calculate_entity_relevance(
                    entity, query_embedding
                )

                context_node = ContextNode(
                    entity=entity,
                    hop_distance=0,
                    relationship_path=[],
                    relevance_score=relevance_score,
                    retrieval_reason="initial_similarity_match"
                )
                context_graph.nodes[entity.id] = context_node

            # Perform multi-hop expansion
            await self._expand_context_graph(
                context_graph, query_embedding, repository_id
            )

            # Build context summary
            context_graph.context_summary = self._build_context_summary(context_graph)

            return context_graph

        except Exception as e:
            logger.error(f"Multi-hop retrieval failed: {e}")
            raise

    async def _expand_context_graph(
        self,
        context_graph: ContextGraph,
        query_embedding: List[float],
        repository_id: str
    ) -> None:
        """Expand context graph through relationship traversal."""

        # Track visited entities to avoid cycles
        visited_entities = set(context_graph.nodes.keys())

        # Queue for breadth-first traversal
        expansion_queue = deque([
            (entity_id, 0) for entity_id in context_graph.primary_entities
        ])

        while expansion_queue and len(context_graph.nodes) < self.max_entities_per_hop * self.max_hops:
            current_entity_id, current_hop = expansion_queue.popleft()

            if current_hop >= self.max_hops:
                continue

            # Get related entities through different relationship types
            related_entities = await self._find_related_entities(
                current_entity_id, repository_id
            )

            # Process each type of relationship
            for relationship_type, entities in related_entities.items():
                await self._process_relationship_hop(
                    context_graph,
                    current_entity_id,
                    relationship_type,
                    entities,
                    current_hop + 1,
                    query_embedding,
                    visited_entities,
                    expansion_queue
                )

    async def _find_related_entities(
        self,
        entity_id: str,
        repository_id: str
    ) -> Dict[RelationshipType, List[CodeEntity]]:
        """Find entities related through different relationship types."""

        related_entities = defaultdict(list)

        try:
            # Get all relationships for this entity
            outgoing_relationships = await self.graph_builder.get_entity_relationships(
                entity_id, direction="outgoing"
            )
            incoming_relationships = await self.graph_builder.get_entity_relationships(
                entity_id, direction="incoming"
            )

            # Process outgoing relationships
            for relationship in outgoing_relationships:
                target_entity = await self.graph_builder.get_entity_by_id(
                    relationship.target_entity_id
                )
                if target_entity:
                    related_entities[relationship.relationship_type].append(target_entity)

            # Process incoming relationships
            for relationship in incoming_relationships:
                source_entity = await self.graph_builder.get_entity_by_id(
                    relationship.source_entity_id
                )
                if source_entity:
                    related_entities[relationship.relationship_type].append(source_entity)

        except Exception as e:
            logger.warning(f"Failed to find related entities for {entity_id}: {e}")

        return related_entities

    async def _process_relationship_hop(
        self,
        context_graph: ContextGraph,
        source_entity_id: str,
        relationship_type: RelationshipType,
        related_entities: List[CodeEntity],
        hop_distance: int,
        query_embedding: List[float],
        visited_entities: Set[str],
        expansion_queue: deque
    ) -> None:
        """Process entities found through a specific relationship type."""

        # Calculate relationship relevance weights
        relationship_weights = {
            RelationshipType.CALLS: 0.9,
            RelationshipType.CONTAINS: 0.8,
            RelationshipType.IMPORTS: 0.7,
            RelationshipType.INHERITS: 0.8,
            RelationshipType.REFERENCES: 0.6,
            RelationshipType.DEPENDS_ON: 0.7
        }

        relationship_weight = relationship_weights.get(relationship_type, 0.5)

        for entity in related_entities:
            if entity.id in visited_entities:
                continue

            # Calculate relevance score
            entity_relevance = await self._calculate_entity_relevance(
                entity, query_embedding
            )

            # Apply relationship weight and hop decay
            hop_decay = 0.8 ** hop_distance
            final_relevance = entity_relevance * relationship_weight * hop_decay

            if final_relevance >= self.relevance_threshold:
                # Build relationship path
                source_node = context_graph.nodes.get(source_entity_id)
                relationship_path = (source_node.relationship_path + [relationship_type.value]
                                   if source_node else [relationship_type.value])

                # Add to context graph
                context_node = ContextNode(
                    entity=entity,
                    hop_distance=hop_distance,
                    relationship_path=relationship_path,
                    relevance_score=final_relevance,
                    retrieval_reason=f"hop_{hop_distance}_{relationship_type.value}"
                )

                context_graph.nodes[entity.id] = context_node
                context_graph.edges.append((source_entity_id, entity.id, relationship_type))

                visited_entities.add(entity.id)
                expansion_queue.append((entity.id, hop_distance))

    async def _calculate_entity_relevance(
        self,
        entity: CodeEntity,
        query_embedding: List[float]
    ) -> float:
        """Calculate relevance score between entity and query."""

        if not query_embedding or not entity.embedding_vector:
            return 0.5  # Default relevance if embeddings not available

        try:
            similarity = self.embedding_service.calculate_similarity(
                query_embedding, entity.embedding_vector
            )
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        except Exception as e:
            logger.warning(f"Failed to calculate relevance for {entity.id}: {e}")
            return 0.0

    def _build_context_summary(self, context_graph: ContextGraph) -> Dict[str, Any]:
        """Build summary information about the context graph."""

        summary = {
            "total_entities": len(context_graph.nodes),
            "total_relationships": len(context_graph.edges),
            "entities_by_type": defaultdict(int),
            "entities_by_hop": defaultdict(int),
            "relationship_types": defaultdict(int),
            "high_relevance_entities": [],
            "context_coverage": {}
        }

        # Analyze entities
        for node in context_graph.nodes.values():
            entity_type = str(node.entity.entity_type)
            summary["entities_by_type"][entity_type] += 1
            summary["entities_by_hop"][node.hop_distance] += 1

            if node.relevance_score > 0.7:
                summary["high_relevance_entities"].append({
                    "entity_id": node.entity.id,
                    "name": node.entity.name,
                    "type": entity_type,
                    "relevance": node.relevance_score,
                    "hop_distance": node.hop_distance
                })

        # Analyze relationships
        for _, _, relationship_type in context_graph.edges:
            summary["relationship_types"][relationship_type.value] += 1

        # Calculate context coverage
        file_coverage = set()
        for node in context_graph.nodes.values():
            file_coverage.add(node.entity.file_path)

        summary["context_coverage"] = {
            "files_covered": len(file_coverage),
            "file_paths": list(file_coverage)
        }

        return dict(summary)

    def convert_to_context_results(
        self,
        context_graph: ContextGraph,
        max_results: int = 20
    ) -> List[ContextResult]:
        """Convert context graph to standard ContextResult format."""

        results = []

        # Sort nodes by relevance score
        sorted_nodes = sorted(
            context_graph.nodes.values(),
            key=lambda x: x.relevance_score,
            reverse=True
        )

        for i, node in enumerate(sorted_nodes[:max_results]):
            # Map retrieval reason
            reason_mapping = {
                "initial_similarity_match": RetrievalReason.SEMANTIC_SIMILARITY,
                "hop_1_CALLS": RetrievalReason.DIRECT_DEPENDENCY,
                "hop_1_CONTAINS": RetrievalReason.STRUCTURAL_RELATIONSHIP,
                "hop_2_CALLS": RetrievalReason.INDIRECT_DEPENDENCY,
                "hop_2_CONTAINS": RetrievalReason.STRUCTURAL_RELATIONSHIP,
                "hop_3_CALLS": RetrievalReason.INDIRECT_DEPENDENCY
            }

            retrieval_reason = reason_mapping.get(
                node.retrieval_reason,
                RetrievalReason.GRAPH_TRAVERSAL
            )

            # Build context result
            context_result = ContextResult(
                entity_id=node.entity.id,
                entity_type=node.entity.entity_type,
                entity_name=node.entity.name,
                file_path=node.entity.file_path,
                start_line=node.entity.start_line,
                end_line=node.entity.end_line,
                content=node.entity.content or "",
                relevance_score=node.relevance_score,
                retrieval_reason=retrieval_reason,
                context_snippet=self._build_context_snippet(node, context_graph),
                metadata={
                    "hop_distance": node.hop_distance,
                    "relationship_path": node.relationship_path,
                    "dependencies": node.entity.dependencies,
                    "control_flow": node.entity.control_flow_info,
                    "semantic_analysis": {
                        "side_effects": node.entity.side_effects,
                        "error_handling": node.entity.error_handling
                    }
                }
            )

            results.append(context_result)

        return results

    def _build_context_snippet(
        self,
        node: ContextNode,
        context_graph: ContextGraph
    ) -> str:
        """Build context snippet showing entity relationships."""

        snippets = []

        # Add entity signature/summary
        if node.entity.signature:
            snippets.append(f"Signature: {node.entity.signature}")

        # Add relationship context
        if node.hop_distance > 0:
            path_desc = " → ".join(node.relationship_path)
            snippets.append(f"Found via: {path_desc}")

        # Add dependencies
        if node.entity.dependencies:
            deps = ", ".join(node.entity.dependencies[:3])
            snippets.append(f"Dependencies: {deps}")

        # Add related entities
        related_entities = [
            edge[1] for edge in context_graph.edges
            if edge[0] == node.entity.id
        ][:3]

        if related_entities:
            related_names = [
                context_graph.nodes[eid].entity.name
                for eid in related_entities
                if eid in context_graph.nodes
            ]
            snippets.append(f"Related: {', '.join(related_names)}")

        return " | ".join(snippets)