"""
Query model for natural language context retrieval requests.

Handles user queries for context retrieval with processing metadata,
embeddings, and result tracking.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator
from pydantic.config import ConfigDict


class QueryType(str, Enum):
    """Types of queries supported by the system."""
    CONTEXT_RETRIEVAL = "context_retrieval"    # General context for coding tasks
    ENTITY_SEARCH = "entity_search"           # Search for specific entities
    RELATIONSHIP_ANALYSIS = "relationship_analysis"  # Analyze relationships
    CODE_NAVIGATION = "code_navigation"       # Navigate through codebase
    SEMANTIC_SEARCH = "semantic_search"       # Semantic similarity search
    PATTERN_MATCHING = "pattern_matching"     # Pattern-based search


class QueryStatus(str, Enum):
    """Query processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


class Query(BaseModel):
    """
    Natural language query for context retrieval.

    Represents user requests for code context with processing metadata,
    embeddings, and result tracking for optimization.
    """

    # Core identification
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique query identifier")
    repository_id: str = Field(..., description="Target repository identifier")

    # Query content
    query_text: str = Field(..., min_length=1, description="Natural language task description")
    query_type: QueryType = Field(default=QueryType.CONTEXT_RETRIEVAL, description="Type of query")

    # Query parameters
    max_results: int = Field(default=20, ge=1, le=1000, description="Maximum number of results")
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")
    include_context: bool = Field(default=True, description="Whether to include code snippets")

    # Filtering options
    filter_entity_types: List[str] = Field(default_factory=list, description="Filter by entity types")
    filter_languages: List[str] = Field(default_factory=list, description="Filter by programming languages")
    filter_file_patterns: List[str] = Field(default_factory=list, description="Filter by file patterns")

    # Semantic information
    query_embedding: Optional[List[float]] = Field(None, description="Vector embedding of query text")
    embedding_model: Optional[str] = Field(None, description="Model used for embedding generation")
    normalized_query: Optional[str] = Field(None, description="Preprocessed/normalized query text")

    # Processing metadata
    processing_time_ms: int = Field(default=0, ge=0, description="Query execution time in milliseconds")
    result_count: int = Field(default=0, ge=0, description="Number of results returned")
    cache_hit: bool = Field(default=False, description="Whether result was served from cache")

    # Quality metrics
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall confidence in results")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Average relevance of results")
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Coverage of query intent")

    # User context
    user_id: Optional[str] = Field(None, description="User identifier if available")
    session_id: Optional[str] = Field(None, description="Session identifier for query grouping")
    context_history: List[str] = Field(default_factory=list, description="Previous queries in session")

    # Status and timestamps
    status: QueryStatus = Field(default=QueryStatus.PENDING, description="Query processing status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Query creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Processing start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Processing completion timestamp")

    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if query failed")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")

    # Performance tracking
    stages_timing: Dict[str, float] = Field(
        default_factory=dict,
        description="Timing for different processing stages"
    )

    # Metadata for extensions
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional query metadata")

    # Pydantic configuration
    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "id": "bb0e8400-e29b-41d4-a716-446655440006",
                "repository_id": "550e8400-e29b-41d4-a716-446655440000",
                "query_text": "Add authentication middleware to Express API",
                "query_type": "context_retrieval",
                "max_results": 15,
                "confidence_threshold": 0.4,
                "include_context": True,
                "filter_entity_types": ["FUNCTION", "CLASS"],
                "processing_time_ms": 127,
                "result_count": 8,
                "confidence_score": 0.85,
                "relevance_score": 0.78,
                "status": "completed",
                "stages_timing": {
                    "embedding": 25.0,
                    "similarity_search": 45.0,
                    "graph_traversal": 32.0,
                    "ranking": 18.0,
                    "formatting": 7.0
                }
            }
        }
    )

    @field_validator("query_text")
    @classmethod
    def validate_query_text(cls, v: str) -> str:
        """Validate query text content."""
        if not v or not v.strip():
            raise ValueError("Query text cannot be empty or only whitespace")

        # Check for reasonable length
        if len(v) > 10000:  # 10k character limit
            raise ValueError("Query text too long (max 10,000 characters)")

        return v.strip()

    @field_validator("filter_entity_types")
    @classmethod
    def validate_entity_types(cls, v: List[str]) -> List[str]:
        """Validate entity type filters."""
        valid_types = {
            "FILE", "CLASS", "FUNCTION", "METHOD", "VARIABLE",
            "CONSTANT", "MODULE", "INTERFACE", "ENUM", "STRUCT",
            "NAMESPACE", "PROPERTY", "FIELD"
        }

        for entity_type in v:
            if entity_type not in valid_types:
                raise ValueError(f"Invalid entity type: {entity_type}")

        return v

    @field_validator("query_embedding")
    @classmethod
    def validate_embedding(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate query embedding vector."""
        if v is None:
            return v

        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("Query embedding must be non-empty list of floats")

        # Check for common embedding dimensions
        valid_dimensions = [128, 256, 384, 512, 768, 1024, 1536]
        if len(v) not in valid_dimensions:
            # Allow other dimensions but could log warning
            pass

        # Validate numeric values
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Embedding value at index {i} must be numeric")

        return v

    def start_processing(self) -> None:
        """Mark query as started and record timestamp."""
        self.status = QueryStatus.PROCESSING
        self.started_at = datetime.utcnow()

    def complete_processing(self, result_count: int, confidence_score: float) -> None:
        """Mark query as completed with results."""
        self.status = QueryStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.result_count = result_count
        self.confidence_score = confidence_score

        # Calculate processing time
        if self.started_at:
            time_delta = self.completed_at - self.started_at
            self.processing_time_ms = int(time_delta.total_seconds() * 1000)

    def fail_processing(self, error_message: str) -> None:
        """Mark query as failed with error message."""
        self.status = QueryStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message

        # Calculate processing time even for failures
        if self.started_at:
            time_delta = self.completed_at - self.started_at
            self.processing_time_ms = int(time_delta.total_seconds() * 1000)

    def add_stage_timing(self, stage: str, duration_ms: float) -> None:
        """Add timing information for a processing stage."""
        self.stages_timing[stage] = duration_ms

    def get_total_stage_time(self) -> float:
        """Get total time from all recorded stages."""
        return sum(self.stages_timing.values())

    def is_similar_to(self, other: "Query", threshold: float = 0.85) -> bool:
        """Check if this query is similar to another query."""
        if not self.query_embedding or not other.query_embedding:
            # Fall back to text similarity
            return self.query_text.lower() == other.query_text.lower()

        if len(self.query_embedding) != len(other.query_embedding):
            return False

        # Calculate cosine similarity
        import math

        dot_product = sum(a * b for a, b in zip(self.query_embedding, other.query_embedding))
        magnitude_a = math.sqrt(sum(a * a for a in self.query_embedding))
        magnitude_b = math.sqrt(sum(b * b for b in other.query_embedding))

        if magnitude_a == 0 or magnitude_b == 0:
            return False

        similarity = dot_product / (magnitude_a * magnitude_b)
        return similarity >= threshold

    def get_cache_key(self) -> str:
        """Generate cache key for query results."""
        # Include repository, query text, and key parameters
        key_components = [
            self.repository_id,
            self.query_text.lower().strip(),
            str(self.max_results),
            str(self.confidence_threshold),
            "|".join(sorted(self.filter_entity_types)),
            "|".join(sorted(self.filter_languages)),
        ]

        import hashlib
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def update_quality_metrics(self, relevance_scores: List[float]) -> None:
        """Update quality metrics based on result relevance scores."""
        if relevance_scores:
            self.relevance_score = sum(relevance_scores) / len(relevance_scores)

            # Calculate coverage based on score distribution
            high_relevance_count = sum(1 for score in relevance_scores if score > 0.7)
            self.coverage_score = min(1.0, high_relevance_count / min(10, self.max_results))

    def add_context_history(self, previous_query: str) -> None:
        """Add previous query to context history."""
        self.context_history.append(previous_query)

        # Keep only recent history (last 5 queries)
        if len(self.context_history) > 5:
            self.context_history = self.context_history[-5:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring."""
        return {
            "query_id": self.id,
            "processing_time_ms": self.processing_time_ms,
            "result_count": self.result_count,
            "confidence_score": self.confidence_score,
            "cache_hit": self.cache_hit,
            "stages_timing": self.stages_timing,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(by_alias=True, exclude_unset=False)

    def __str__(self) -> str:
        """String representation."""
        return f"Query({self.query_type.value}, '{self.query_text[:50]}...', {self.status.value})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"Query(id={self.id}, repository={self.repository_id}, "
            f"type={self.query_type.value}, status={self.status.value}, "
            f"results={self.result_count}, time={self.processing_time_ms}ms)"
        )