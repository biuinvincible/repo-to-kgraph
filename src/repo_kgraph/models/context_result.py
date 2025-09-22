"""ContextResult model for query response items."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class RetrievalReason(str, Enum):
    """Reasons why an entity was included in results."""
    DIRECT_MATCH = "direct_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    GRAPH_TRAVERSAL = "graph_traversal"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    PATTERN_MATCH = "pattern_match"
    CONTEXT_EXPANSION = "context_expansion"
    DIRECT_DEPENDENCY = "direct_dependency"
    INDIRECT_DEPENDENCY = "indirect_dependency"
    STRUCTURAL_RELATIONSHIP = "structural_relationship"


class ContextResult(BaseModel):
    """Individual result item from context retrieval query."""
    
    # Core identification
    id: str = Field(default_factory=lambda: str(uuid4()))
    query_id: str = Field(..., description="Parent query identifier")
    entity_id: str = Field(..., description="Referenced code entity identifier")
    
    # Ranking and relevance
    relevance_score: float = Field(ge=0.0, le=1.0, description="Similarity/relevance score")
    rank_position: int = Field(ge=1, description="Result ranking position")
    retrieval_reason: RetrievalReason = Field(..., description="Reason for inclusion")
    
    # Content
    context_snippet: Optional[str] = Field(None, description="Relevant code snippet")
    highlighted_terms: List[str] = Field(default_factory=list, description="Highlighted search terms")
    
    # Entity summary (denormalized for performance)
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    file_path: str = Field(..., description="File path")
    start_line: int = Field(ge=1, description="Starting line number")
    end_line: int = Field(ge=1, description="Ending line number")
    
    # Additional context
    surrounding_context: Optional[str] = Field(None, description="Broader code context")
    related_entities: List[str] = Field(default_factory=list, description="Related entity IDs")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Pydantic configuration
    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "id": "cc0e8400-e29b-41d4-a716-446655440007",
                "query_id": "bb0e8400-e29b-41d4-a716-446655440006",
                "entity_id": "770e8400-e29b-41d4-a716-446655440002",
                "relevance_score": 0.85,
                "rank_position": 1,
                "retrieval_reason": "direct_match",
                "context_snippet": "def calculate_total(items: List[Item], tax_rate: float) -> float:",
                "highlighted_terms": ["calculate", "total", "tax_rate"],
                "entity_name": "calculate_total",
                "entity_type": "FUNCTION",
                "file_path": "src/billing/utils.py",
                "start_line": 15,
                "end_line": 32,
                "surrounding_context": "# Utility functions for billing calculations...",
                "related_entities": ["990e8400-e29b-41d4-a716-446655440004"],
                "metadata": {
                    "language": "python",
                    "complexity_score": 3.2,
                    "line_count": 18,
                    "visibility": "public"
                }
            }
        }
    )
    
    def __str__(self) -> str:
        return f"ContextResult({self.entity_name}, rank={self.rank_position}, score={self.relevance_score:.2f})"
    
    def __repr__(self) -> str:
        return (
            f"ContextResult(id={self.id}, query={self.query_id}, "
            f"entity={self.entity_id}, rank={self.rank_position}, "
            f"score={self.relevance_score:.2f}, reason={self.retrieval_reason.value})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(by_alias=True, exclude_unset=False)
    
    def is_highly_relevant(self, threshold: float = 0.7) -> bool:
        """Check if result is highly relevant."""
        return self.relevance_score >= threshold
    
    def get_location_string(self) -> str:
        """Get human-readable location string."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"
    
    def add_highlighted_term(self, term: str) -> None:
        """Add highlighted search term."""
        if term not in self.highlighted_terms:
            self.highlighted_terms.append(term)
    
    def add_related_entity(self, entity_id: str) -> None:
        """Add related entity ID."""
        if entity_id not in self.related_entities:
            self.related_entities.append(entity_id)
    
    def update_relevance_score(self, new_score: float) -> None:
        """Update relevance score with validation."""
        if not 0.0 <= new_score <= 1.0:
            raise ValueError("Relevance score must be between 0.0 and 1.0")
        self.relevance_score = new_score
    
    def get_context_summary(self) -> str:
        """Get summary of context snippet."""
        if not self.context_snippet:
            return ""
        
        # Return first few lines of context
        lines = self.context_snippet.split("\n")
        return "\n".join(lines[:3]) + ("..." if len(lines) > 3 else "")
    
    def is_in_same_file(self, other: "ContextResult") -> bool:
        """Check if this result is in the same file as another."""
        return self.file_path == other.file_path
    
    def overlaps_with(self, other: "ContextResult") -> bool:
        """Check if this result overlaps with another in the same file."""
        if not self.is_in_same_file(other):
            return False
        
        return not (self.end_line < other.start_line or
                   other.end_line < self.start_line)