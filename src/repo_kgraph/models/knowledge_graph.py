"""
KnowledgeGraph model for representing complete graph structure.

Container entity representing the complete graph structure for a repository
with metrics, statistics, and construction metadata.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator
from pydantic.config import ConfigDict


class GraphStatus(str, Enum):
    """Knowledge graph construction status."""
    BUILDING = "building"
    READY = "ready"
    UPDATING = "updating"
    ERROR = "error"
    STALE = "stale"


class KnowledgeGraph(BaseModel):
    """
    Complete knowledge graph representation for a repository.

    Container for all entities and relationships with graph-level
    metrics, statistics, and construction metadata.
    """

    # Core identification
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique graph identifier")
    repository_id: str = Field(..., description="Parent repository identifier")

    # Graph composition
    entity_count: int = Field(default=0, ge=0, description="Total number of entities in graph")
    relationship_count: int = Field(default=0, ge=0, description="Total number of relationships")
    file_count: int = Field(default=0, ge=0, description="Number of files represented")

    # Graph structure metrics
    max_depth: int = Field(default=0, ge=0, description="Maximum relationship depth in graph")
    average_degree: float = Field(default=0.0, ge=0.0, description="Average node degree")
    density: float = Field(default=0.0, ge=0.0, le=1.0, description="Graph density")
    clustering_coefficient: float = Field(default=0.0, ge=0.0, le=1.0, description="Global clustering coefficient")

    # Component analysis
    connected_components: int = Field(default=1, ge=1, description="Number of connected components")
    largest_component_size: int = Field(default=0, ge=0, description="Size of largest connected component")
    isolated_nodes: int = Field(default=0, ge=0, description="Number of isolated nodes")

    # Language distribution
    language_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of entities by programming language"
    )

    # Entity type distribution
    entity_type_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of entities by type"
    )

    # Relationship type distribution
    relationship_type_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of relationships by type"
    )

    # Construction metadata
    construction_time: float = Field(default=0.0, ge=0.0, description="Time taken to build graph (seconds)")
    graph_version: str = Field(default="1.0", description="Schema version for compatibility")
    algorithm_version: str = Field(default="1.0", description="Graph construction algorithm version")

    # Quality metrics
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Graph completeness score")
    consistency_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Graph consistency score")
    coverage_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Repository coverage percentage")

    # Index information
    indexed_paths: List[str] = Field(default_factory=list, description="File paths included in graph")
    excluded_paths: List[str] = Field(default_factory=list, description="File paths excluded from graph")
    error_paths: List[str] = Field(default_factory=list, description="File paths that failed parsing")

    # Performance statistics
    parsing_errors: int = Field(default=0, ge=0, description="Number of parsing errors encountered")
    memory_usage_mb: float = Field(default=0.0, ge=0.0, description="Peak memory usage during construction")

    # Status and timestamps
    status: GraphStatus = Field(default=GraphStatus.BUILDING, description="Current graph status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Graph creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Graph update timestamp")
    last_validated: Optional[datetime] = Field(None, description="Last validation timestamp")

    # Advanced metrics (optional)
    centrality_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Centrality metrics (betweenness, closeness, etc.)"
    )

    # Metadata for extensions
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional graph metadata")

    # Pydantic configuration
    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "id": "aa0e8400-e29b-41d4-a716-446655440005",
                "repository_id": "550e8400-e29b-41d4-a716-446655440000",
                "entity_count": 15893,
                "relationship_count": 43521,
                "file_count": 1247,
                "max_depth": 8,
                "average_degree": 5.48,
                "density": 0.00034,
                "clustering_coefficient": 0.23,
                "connected_components": 3,
                "largest_component_size": 15800,
                "construction_time": 154.7,
                "completeness_score": 0.95,
                "consistency_score": 0.98,
                "coverage_percentage": 92.3,
                "language_distribution": {
                    "python": 12500,
                    "javascript": 2800,
                    "typescript": 593
                },
                "entity_type_distribution": {
                    "FUNCTION": 8500,
                    "CLASS": 2100,
                    "VARIABLE": 3800,
                    "FILE": 1247,
                    "MODULE": 246
                }
            }
        }
    )

    @field_validator("density", "clustering_coefficient", "completeness_score", "consistency_score")
    @classmethod
    def validate_probability_fields(cls, v: float) -> float:
        """Validate fields that should be between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Value must be between 0.0 and 1.0")
        return v

    @field_validator("coverage_percentage")
    @classmethod
    def validate_percentage(cls, v: float) -> float:
        """Validate percentage field."""
        if not 0.0 <= v <= 100.0:
            raise ValueError("Percentage must be between 0.0 and 100.0")
        return v

    def calculate_density(self) -> float:
        """Calculate and update graph density."""
        if self.entity_count <= 1:
            self.density = 0.0
            return self.density

        max_possible_edges = self.entity_count * (self.entity_count - 1)
        if max_possible_edges == 0:
            self.density = 0.0
        else:
            self.density = min(1.0, (2.0 * self.relationship_count) / max_possible_edges)

        return self.density

    def calculate_average_degree(self) -> float:
        """Calculate and update average node degree."""
        if self.entity_count == 0:
            self.average_degree = 0.0
            return self.average_degree

        self.average_degree = (2.0 * self.relationship_count) / self.entity_count
        return self.average_degree

    def update_entity_count(self, new_count: int) -> None:
        """Update entity count and recalculate derived metrics."""
        self.entity_count = new_count
        self.calculate_density()
        self.calculate_average_degree()
        self.updated_at = datetime.utcnow()

    def update_relationship_count(self, new_count: int) -> None:
        """Update relationship count and recalculate derived metrics."""
        self.relationship_count = new_count
        self.calculate_density()
        self.calculate_average_degree()
        self.updated_at = datetime.utcnow()

    def add_language_entities(self, language: str, count: int) -> None:
        """Add entities for a specific language."""
        if language in self.language_distribution:
            self.language_distribution[language] += count
        else:
            self.language_distribution[language] = count
        self.updated_at = datetime.utcnow()

    def add_entity_type_count(self, entity_type: str, count: int) -> None:
        """Add entities of a specific type."""
        if entity_type in self.entity_type_distribution:
            self.entity_type_distribution[entity_type] += count
        else:
            self.entity_type_distribution[entity_type] = count
        self.updated_at = datetime.utcnow()

    def add_relationship_type_count(self, relationship_type: str, count: int) -> None:
        """Add relationships of a specific type."""
        if relationship_type in self.relationship_type_distribution:
            self.relationship_type_distribution[relationship_type] += count
        else:
            self.relationship_type_distribution[relationship_type] = count
        self.updated_at = datetime.utcnow()

    def get_dominant_language(self) -> Optional[str]:
        """Get the dominant programming language by entity count."""
        if not self.language_distribution:
            return None
        return max(self.language_distribution.items(), key=lambda x: x[1])[0]

    def get_dominant_entity_type(self) -> Optional[str]:
        """Get the most common entity type."""
        if not self.entity_type_distribution:
            return None
        return max(self.entity_type_distribution.items(), key=lambda x: x[1])[0]

    def calculate_coverage_percentage(self, total_files_in_repo: int) -> float:
        """Calculate and update coverage percentage."""
        if total_files_in_repo == 0:
            self.coverage_percentage = 0.0
        else:
            self.coverage_percentage = min(100.0, (self.file_count / total_files_in_repo) * 100.0)
        return self.coverage_percentage

    def is_well_connected(self) -> bool:
        """Check if graph is well-connected (few isolated components)."""
        if self.entity_count == 0:
            return True

        # Consider well-connected if >80% of nodes are in largest component
        return (self.largest_component_size / self.entity_count) > 0.8

    def get_complexity_score(self) -> float:
        """Calculate overall graph complexity score."""
        if self.entity_count == 0:
            return 0.0

        # Combine multiple factors
        size_factor = min(1.0, self.entity_count / 10000.0)  # Normalize by 10k entities
        depth_factor = min(1.0, self.max_depth / 20.0)       # Normalize by depth 20
        density_factor = self.density

        return (size_factor * 0.4 + depth_factor * 0.3 + density_factor * 0.3)

    def update_status(self, new_status: GraphStatus, error_message: Optional[str] = None) -> None:
        """Update graph status with timestamp."""
        self.status = new_status
        self.updated_at = datetime.utcnow()

        if error_message:
            self.metadata["last_error"] = {
                "message": error_message,
                "timestamp": datetime.utcnow().isoformat()
            }

        if new_status == GraphStatus.READY:
            self.last_validated = datetime.utcnow()

    def add_parsing_error(self, file_path: str, error_message: str) -> None:
        """Record a parsing error."""
        self.parsing_errors += 1
        if file_path not in self.error_paths:
            self.error_paths.append(file_path)

        if "parsing_errors" not in self.metadata:
            self.metadata["parsing_errors"] = []

        self.metadata["parsing_errors"].append({
            "file_path": file_path,
            "error": error_message,
            "timestamp": datetime.utcnow().isoformat()
        })

        self.updated_at = datetime.utcnow()

    def get_health_score(self) -> float:
        """Calculate overall graph health score."""
        # Combine completeness, consistency, and error rate
        error_rate = self.parsing_errors / max(1, self.file_count)
        error_penalty = max(0.0, 1.0 - error_rate)

        return (self.completeness_score * 0.4 +
                self.consistency_score * 0.4 +
                error_penalty * 0.2)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Get summary dictionary with key metrics."""
        return {
            "id": self.id,
            "repository_id": self.repository_id,
            "entity_count": self.entity_count,
            "relationship_count": self.relationship_count,
            "file_count": self.file_count,
            "status": self.status,
            "construction_time": self.construction_time,
            "coverage_percentage": self.coverage_percentage,
            "health_score": self.get_health_score(),
            "complexity_score": self.get_complexity_score(),
            "dominant_language": self.get_dominant_language(),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(by_alias=True, exclude_unset=False)

    def __str__(self) -> str:
        """String representation."""
        return f"KnowledgeGraph({self.entity_count} entities, {self.relationship_count} relationships, {self.status.value})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"KnowledgeGraph(id={self.id}, repository={self.repository_id}, "
            f"entities={self.entity_count}, relationships={self.relationship_count}, "
            f"status={self.status.value}, health={self.get_health_score():.2f})"
        )