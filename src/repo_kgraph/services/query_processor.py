"""
Query processor service for handling natural language queries.

Processes user queries, coordinates context retrieval, and formats results
for consumption by coding agents and developers.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4

from repo_kgraph.models.query import Query, QueryStatus
from repo_kgraph.models.context_result import ContextResult, RetrievalReason
from repo_kgraph.models.code_entity import EntityType
from repo_kgraph.services.retriever import ContextRetriever
from repo_kgraph.services.embedding import EmbeddingService


logger = logging.getLogger(__name__)


class QueryProcessorError(Exception):
    """Raised when query processing fails."""
    pass


class QueryProcessor:
    """
    Service for processing natural language queries and retrieving context.

    Handles query parsing, context retrieval coordination, result ranking,
    and response formatting for coding agents.
    """

    def __init__(
        self,
        context_retriever: ContextRetriever,
        embedding_service: EmbeddingService,
        default_max_results: int = 20,
        default_confidence_threshold: float = 0.3,
        query_timeout_seconds: int = 30
    ):
        """
        Initialize the query processor.

        Args:
            context_retriever: Service for context retrieval
            embedding_service: Service for embedding operations
            default_max_results: Default maximum results to return
            default_confidence_threshold: Default confidence threshold
            query_timeout_seconds: Timeout for query processing
        """
        self.context_retriever = context_retriever
        self.embedding_service = embedding_service
        self.default_max_results = default_max_results
        self.default_confidence_threshold = default_confidence_threshold
        self.query_timeout_seconds = query_timeout_seconds

        # Query cache and statistics
        self._query_cache: Dict[str, Dict[str, Any]] = {}
        self._query_stats: Dict[str, Any] = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0
        }

    async def process_query(
        self,
        repository_id: str,
        task_description: str,
        max_results: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        language_filter: Optional[List[str]] = None,
        entity_types: Optional[List[EntityType]] = None,
        file_path_filter: Optional[str] = None,
        include_related: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Process a natural language query and return relevant context.

        Args:
            repository_id: Repository to search in
            task_description: Natural language description of the task
            max_results: Maximum number of results to return
            confidence_threshold: Minimum confidence for results
            language_filter: Optional programming language filter
            entity_types: Optional entity type filter
            file_path_filter: Optional file path filter
            include_related: Whether to include related entities
            use_cache: Whether to use query caching

        Returns:
            Dictionary with query results and metadata

        Raises:
            QueryProcessorError: If query processing fails
        """
        start_time = time.time()
        query_id = str(uuid4())
        query = None

        try:
            # Create query object
            query = Query(
                id=query_id,
                repository_id=repository_id,
                query_text=task_description,
                max_results=max_results or self.default_max_results,
                confidence_threshold=confidence_threshold or self.default_confidence_threshold,
                filter_languages=language_filter or [],
                filter_entity_types=entity_types or [],
                filter_file_patterns=[file_path_filter] if file_path_filter else [],
                created_at=datetime.utcnow(),
                status=QueryStatus.PROCESSING
            )

            logger.info(f"Processing query {query_id}: {task_description[:100]}...")

            # Check cache if enabled
            if use_cache:
                cached_result = self._check_query_cache(query)
                if cached_result:
                    self._query_stats["cache_hits"] += 1
                    logger.info(f"Cache hit for query {query_id}")
                    return cached_result

            # Set timeout for query processing
            try:
                result = await asyncio.wait_for(
                    self._execute_query(query, include_related),
                    timeout=self.query_timeout_seconds
                )
            except asyncio.TimeoutError:
                raise QueryProcessorError(f"Query processing timed out after {self.query_timeout_seconds}s")

            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time

            # Format final response
            response = {
                "query_id": query_id,
                "results": result["context_results"],
                "processing_time_ms": int(processing_time * 1000),
                "total_results": len(result["context_results"]),
                "confidence_score": self._calculate_overall_confidence(result["context_results"]),
                "message": "Query processed successfully",
                "query_metadata": {
                    "repository_id": repository_id,
                    "task_description": task_description,
                    "filters_applied": {
                        "language_filter": language_filter,
                        "entity_types": [et.value if hasattr(et, 'value') else str(et) for et in (entity_types or [])],
                        "file_path_filter": file_path_filter
                    },
                    "retrieval_params": {
                        "max_results": query.max_results,
                        "confidence_threshold": query.confidence_threshold,
                        "include_related": include_related
                    }
                }
            }

            # Cache result if appropriate
            if use_cache and len(result["context_results"]) > 0:
                self._cache_query_result(query, response)

            # Update statistics
            self._update_query_statistics(processing_time, True)

            # Update query status
            query.status = QueryStatus.COMPLETED
            query.completed_at = datetime.utcnow()

            logger.info(f"Query {query_id} completed in {processing_time:.2f}s with {len(result['context_results'])} results")

            return response

        except Exception as e:
            # Update statistics
            self._update_query_statistics(time.time() - start_time, False)

            # Update query status if query was created
            if query is not None:
                query.status = QueryStatus.FAILED
                query.error_message = str(e)

            logger.error(f"Query processing failed for {query_id}: {e}")
            raise QueryProcessorError(f"Failed to process query: {e}")

    async def _execute_query(self, query: Query, include_related: bool) -> Dict[str, Any]:
        """Execute the actual query processing."""
        try:
            # Step 1: Enhance query with context analysis
            enhanced_query = await self._enhance_query_context(query)

            # Step 2: Retrieve context using the retriever service
            context_results = await self.context_retriever.retrieve_context(
                query=enhanced_query,
                max_results=query.max_results,
                confidence_threshold=query.confidence_threshold,
                include_related=include_related
            )

            # Step 3: Post-process and rank results
            processed_results = await self._post_process_results(context_results, query)

            # Step 4: Generate query insights
            insights = await self._generate_query_insights(processed_results, query)

            return {
                "context_results": processed_results,
                "query_insights": insights,
                "enhanced_query": enhanced_query
            }

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    async def _enhance_query_context(self, query: Query) -> Query:
        """Enhance query with additional context and intent analysis."""
        try:
            # Extract intent and entities from the task description
            intent_info = self._analyze_query_intent(query.query_text)

            # Update query with intent-based filters
            if intent_info["suggested_entity_types"] and not query.filter_entity_types:
                query.filter_entity_types = intent_info["suggested_entity_types"]

            # Add semantic keywords for better matching
            # Store in metadata since semantic_keywords is not a direct field
            if not query.metadata:
                query.metadata = {}
            query.metadata["semantic_keywords"] = intent_info["keywords"]

            # Generate query embedding
            await self.embedding_service.embed_query(query)

            return query

        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}")
            return query

    def _analyze_query_intent(self, task_description: str) -> Dict[str, Any]:
        """Analyze query intent and extract relevant information."""
        task_lower = task_description.lower()
        intent_info = {
            "suggested_entity_types": [],
            "keywords": [],
            "intent_category": "general"
        }

        # Intent patterns
        if any(word in task_lower for word in ["function", "method", "def", "implement"]):
            intent_info["suggested_entity_types"].append(EntityType.FUNCTION)
            intent_info["intent_category"] = "function_related"

        if any(word in task_lower for word in ["class", "object", "inheritance"]):
            intent_info["suggested_entity_types"].append(EntityType.CLASS)
            intent_info["intent_category"] = "class_related"

        if any(word in task_lower for word in ["variable", "constant", "property"]):
            intent_info["suggested_entity_types"].extend([EntityType.VARIABLE, EntityType.CONSTANT])
            intent_info["intent_category"] = "variable_related"

        if any(word in task_lower for word in ["import", "module", "package"]):
            intent_info["suggested_entity_types"].append(EntityType.MODULE)
            intent_info["intent_category"] = "import_related"

        if any(word in task_lower for word in ["api", "endpoint", "route"]):
            intent_info["intent_category"] = "api_related"

        if any(word in task_lower for word in ["test", "testing", "unittest"]):
            intent_info["intent_category"] = "test_related"

        if any(word in task_lower for word in ["bug", "fix", "error", "issue"]):
            intent_info["intent_category"] = "debugging"

        # Extract potential keywords
        # Simple keyword extraction - could be enhanced with NLP
        keywords = []
        words = task_description.split()
        for word in words:
            if len(word) > 3 and word.isalnum():
                keywords.append(word.lower())

        intent_info["keywords"] = keywords[:10]  # Limit to top 10

        return intent_info

    async def _post_process_results(
        self,
        context_results: List[ContextResult],
        query: Query
    ) -> List[Dict[str, Any]]:
        """Post-process and format context results."""
        processed_results = []

        for result in context_results:
            try:
                # Convert to dictionary format
                result_dict = {
                    "id": result.id,
                    "entity_id": result.entity_id,
                    "name": result.entity_name,
                    "entity_type": result.entity_type.value if hasattr(result.entity_type, 'value') else str(result.entity_type),
                    "file_path": result.file_path,
                    "start_line": result.start_line,
                    "end_line": result.end_line,
                    "content": result.context_snippet,
                    "relevance_score": round(result.relevance_score, 3),
                    "confidence_score": round(result.relevance_score, 3),
                    "context": result.context_snippet,
                    "language": result.metadata.get("language", "unknown"),
                    "match_reason": result.retrieval_reason.value if hasattr(result.retrieval_reason, 'value') else str(result.retrieval_reason)
                }

                # Add additional metadata
                result_dict["line_count"] = result.end_line - result.start_line + 1
                result_dict["location"] = f"{result.file_path}:{result.start_line}-{result.end_line}"

                # Add relevance explanation
                result_dict["relevance_explanation"] = self._generate_relevance_explanation(result, query)

                processed_results.append(result_dict)

            except Exception as e:
                logger.warning(f"Failed to process result {result.id}: {e}")
                continue

        return processed_results

    def _generate_relevance_explanation(self, result: ContextResult, query: Query) -> str:
        """Generate human-readable explanation for why a result is relevant."""
        explanations = []

        if result.retrieval_reason == RetrievalReason.SEMANTIC_SIMILARITY:
            explanations.append(f"Semantically similar to query (score: {result.relevance_score:.2f})")

        if result.retrieval_reason == RetrievalReason.GRAPH_TRAVERSAL:
            explanations.append(f"Related through code relationships (score: {result.relevance_score:.2f})")

        if result.entity_type in (query.filter_entity_types or []):
            explanations.append(f"Matches requested entity type: {result.entity_type}")

        if query.filter_languages and result.metadata.get("language", "unknown") in query.filter_languages:
            explanations.append(f"Matches language filter: {result.metadata.get('language', 'unknown')}")

        if not explanations:
            explanations.append("General relevance match")

        return "; ".join(explanations)

    async def _generate_query_insights(
        self,
        results: List[Dict[str, Any]],
        query: Query
    ) -> Dict[str, Any]:
        """Generate insights about the query results."""
        if not results:
            return {"message": "No relevant results found"}

        insights = {}

        # Entity type distribution
        entity_types = {}
        for result in results:
            entity_type = result["entity_type"]
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        insights["entity_type_distribution"] = entity_types

        # Language distribution
        languages = {}
        for result in results:
            language = result["language"]
            languages[language] = languages.get(language, 0) + 1
        insights["language_distribution"] = languages

        # File distribution
        files = {}
        for result in results:
            file_path = result["file_path"]
            files[file_path] = files.get(file_path, 0) + 1
        insights["file_distribution"] = dict(list(files.items())[:10])  # Top 10 files

        # Confidence statistics
        confidence_scores = [result["confidence_score"] for result in results]
        insights["confidence_stats"] = {
            "min": min(confidence_scores) if confidence_scores else 0,
            "max": max(confidence_scores) if confidence_scores else 0,
            "avg": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        }

        # Suggestions
        suggestions = []
        if len(results) < query.max_results // 2:
            suggestions.append("Consider lowering the confidence threshold for more results")

        if len(set(result["file_path"] for result in results)) < 3:
            suggestions.append("Results are concentrated in few files - consider broader search")

        insights["suggestions"] = suggestions

        return insights

    def _calculate_overall_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for the query results."""
        if not results:
            return 0.0

        # Weighted average with higher weight for top results
        total_weighted_score = 0.0
        total_weight = 0.0

        for i, result in enumerate(results):
            weight = 1.0 / (i + 1)  # Decreasing weight for lower-ranked results
            score = result["confidence_score"]
            total_weighted_score += score * weight
            total_weight += weight

        return round(total_weighted_score / total_weight, 3) if total_weight > 0 else 0.0

    def _check_query_cache(self, query: Query) -> Optional[Dict[str, Any]]:
        """Check if query result exists in cache."""
        cache_key = self._generate_cache_key(query)
        cached_entry = self._query_cache.get(cache_key)

        if cached_entry:
            # Check if cache entry is still valid (e.g., not too old)
            cache_age = time.time() - cached_entry["timestamp"]
            if cache_age < 3600:  # 1 hour cache validity
                return cached_entry["result"]

        return None

    def _cache_query_result(self, query: Query, result: Dict[str, Any]) -> None:
        """Cache query result for future use."""
        cache_key = self._generate_cache_key(query)
        self._query_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }

        # Limit cache size
        if len(self._query_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self._query_cache.keys(),
                key=lambda k: self._query_cache[k]["timestamp"]
            )[:100]
            for key in oldest_keys:
                del self._query_cache[key]

    def _generate_cache_key(self, query: Query) -> str:
        """Generate cache key for query."""
        key_parts = [
            query.repository_id,
            query.query_text,
            str(query.max_results),
            str(query.confidence_threshold),
            str(sorted(query.filter_languages or [])),
            str(sorted([et.value if hasattr(et, 'value') else str(et) for et in (query.filter_entity_types or [])])),
            str(query.filter_file_patterns or "")
        ]
        return "|".join(key_parts)

    def _update_query_statistics(self, processing_time: float, success: bool) -> None:
        """Update query processing statistics."""
        self._query_stats["total_queries"] += 1

        if success:
            self._query_stats["successful_queries"] += 1
        else:
            self._query_stats["failed_queries"] += 1

        # Update average processing time
        total_queries = self._query_stats["total_queries"]
        current_avg = self._query_stats["average_processing_time"]
        new_avg = ((current_avg * (total_queries - 1)) + processing_time) / total_queries
        self._query_stats["average_processing_time"] = new_avg

    async def search_by_entity_name(
        self,
        repository_id: str,
        entity_name: str,
        entity_types: Optional[List[EntityType]] = None,
        exact_match: bool = False
    ) -> Dict[str, Any]:
        """Search for entities by name."""
        try:
            start_time = time.time()

            results = await self.context_retriever.search_by_entity_name(
                repository_id=repository_id,
                entity_name=entity_name,
                entity_types=entity_types,
                exact_match=exact_match
            )

            processing_time = time.time() - start_time

            return {
                "query_type": "entity_name_search",
                "entity_name": entity_name,
                "exact_match": exact_match,
                "results": results,
                "processing_time_ms": int(processing_time * 1000),
                "total_results": len(results)
            }

        except Exception as e:
            logger.error(f"Entity name search failed: {e}")
            raise QueryProcessorError(f"Entity name search failed: {e}")

    async def get_entity_context(
        self,
        repository_id: str,
        entity_id: str,
        depth: int = 2,
        relationship_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get comprehensive context for a specific entity."""
        try:
            start_time = time.time()

            context = await self.context_retriever.get_entity_context(
                entity_id=entity_id,
                repository_id=repository_id,
                depth=depth,
                relationship_types=relationship_types
            )

            processing_time = time.time() - start_time

            return {
                "query_type": "entity_context",
                "entity_id": entity_id,
                "depth": depth,
                "context": context,
                "processing_time_ms": int(processing_time * 1000)
            }

        except Exception as e:
            logger.error(f"Entity context retrieval failed: {e}")
            raise QueryProcessorError(f"Entity context retrieval failed: {e}")

    def get_query_statistics(self) -> Dict[str, Any]:
        """Get query processing statistics."""
        stats = dict(self._query_stats)
        stats["cache_size"] = len(self._query_cache)
        stats["success_rate"] = (
            stats["successful_queries"] / stats["total_queries"]
            if stats["total_queries"] > 0 else 0.0
        )
        return stats

    def clear_query_cache(self) -> None:
        """Clear the query cache."""
        self._query_cache.clear()
        logger.info("Query cache cleared")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._query_cache.clear()
        logger.info("Query processor cleanup completed")