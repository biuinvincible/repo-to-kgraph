"""Index model for optimized search structures."""

from datetime import datetime
from enum import Enum
from typing import Dict, Any
from uuid import uuid4
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class IndexType(str, Enum):
    """Types of indexes for different search strategies."""
    VECTOR_INDEX = "vector_index"
    GRAPH_INDEX = "graph_index"
    TEXT_INDEX = "text_index"
    METADATA_INDEX = "metadata_index"


class IndexStatus(str, Enum):
    """Index construction and maintenance status."""
    BUILDING = "building"
    READY = "ready"
    STALE = "stale"
    ERROR = "error"


class Index(BaseModel):
    """Processed search index for fast retrieval."""
    
    # Core identification
    id: str = Field(default_factory=lambda: str(uuid4()))
    repository_id: str = Field(..., description="Parent repository identifier")
    
    # Index classification
    index_type: IndexType = Field(..., description="Type of search index")
    index_name: str = Field(..., description="Human-readable index identifier")
    
    # Size and performance metrics
    index_size_bytes: int = Field(default=0, ge=0, description="Storage size of index")
    entry_count: int = Field(default=0, ge=0, description="Number of indexed entries")
    build_time_ms: int = Field(default=0, ge=0, description="Time to construct index")
    
    # Configuration
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Index-specific settings")
    
    # Status and timestamps
    status: IndexStatus = Field(default=IndexStatus.BUILDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Pydantic configuration
    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "id": "dd0e8400-e29b-41d4-a716-446655440008",
                "repository_id": "550e8400-e29b-41d4-a716-446655440000",
                "index_type": "vector_index",
                "index_name": "Python Functions Vector Index",
                "index_size_bytes": 104857600,  # 100MB
                "entry_count": 8500,
                "build_time_ms": 15470,
                "configuration": {
                    "embedding_model": "nomic-ai/nomic-embed-code",
                    "dimensions": 384,
                    "similarity_metric": "cosine",
                    "batch_size": 1000
                },
                "status": "ready",
                "created_at": "2025-09-17T10:30:00Z",
                "last_updated": "2025-09-17T10:30:00Z",
                "metadata": {
                    "language": "python",
                    "entity_types": ["FUNCTION", "CLASS"],
                    "avg_processing_time_per_entry": 1.82,
                    "peak_memory_mb": 512
                }
            }
        }
    )
    
    def update_status(self, new_status: IndexStatus) -> None:
        """Update index status with timestamp."""
        self.status = new_status
        self.last_updated = datetime.utcnow()
    
    def __str__(self) -> str:
        return f"Index({self.index_type.value}, {self.index_name}, {self.status.value})"
    
    def __repr__(self) -> str:
        return (
            f"Index(id={self.id}, repository={self.repository_id}, "
            f"type={self.index_type.value}, name={self.index_name}, "
            f"status={self.status.value}, entries={self.entry_count}, "
            f"size={self.index_size_bytes} bytes)"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(by_alias=True, exclude_unset=False)
    
    def is_ready(self) -> bool:
        """Check if index is ready for queries."""
        return self.status == IndexStatus.READY
    
    def is_stale(self) -> bool:
        """Check if index needs rebuilding."""
        return self.status == IndexStatus.STALE
    
    def has_error(self) -> bool:
        """Check if index has an error."""
        return self.status == IndexStatus.ERROR
    
    def get_build_performance(self) -> Dict[str, Any]:
        """Get build performance metrics."""
        return {
            "build_time_ms": self.build_time_ms,
            "entries_per_second": self.entry_count / max(1, self.build_time_ms / 1000.0),
            "bytes_per_entry": self.index_size_bytes / max(1, self.entry_count),
            "status": self.status.value
        }
    
    def update_build_metrics(self, build_time_ms: int, index_size_bytes: int, entry_count: int) -> None:
        """Update build metrics and mark as ready."""
        self.build_time_ms = build_time_ms
        self.index_size_bytes = index_size_bytes
        self.entry_count = entry_count
        self.update_status(IndexStatus.READY)
        self.last_updated = datetime.utcnow()
    
    def mark_as_stale(self) -> None:
        """Mark index as stale (needs rebuilding)."""
        self.update_status(IndexStatus.STALE)
        self.last_updated = datetime.utcnow()
    
    def mark_as_error(self, error_message: str) -> None:
        """Mark index as error with message."""
        self.update_status(IndexStatus.ERROR)
        self.metadata["last_error"] = {
            "message": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.last_updated = datetime.utcnow()
    
    def get_storage_efficiency(self) -> float:
        """Calculate storage efficiency (bytes per entry)."""
        if self.entry_count == 0:
            return 0.0
        return self.index_size_bytes / self.entry_count
    
    def get_construction_speed(self) -> float:
        """Calculate construction speed (entries per second)."""
        if self.build_time_ms == 0:
            return 0.0
        return self.entry_count / (self.build_time_ms / 1000.0)
    
    def add_configuration(self, key: str, value: Any) -> None:
        """Add configuration setting."""
        self.configuration[key] = value
        self.last_updated = datetime.utcnow()
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata."""
        self.metadata[key] = value
        self.last_updated = datetime.utcnow()
    
    def get_age_hours(self) -> float:
        """Get index age in hours."""
        now = datetime.utcnow()
        age_delta = now - self.created_at
        return age_delta.total_seconds() / 3600.0
    
    def needs_rebuild(self, max_age_hours: float = 24.0) -> bool:
        """Check if index needs rebuild based on age."""
        return self.get_age_hours() > max_age_hours