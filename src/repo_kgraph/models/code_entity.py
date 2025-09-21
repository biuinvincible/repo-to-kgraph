"""
CodeEntity model for representing individual code components.

Handles files, classes, functions, variables, and other code elements
with their semantic metadata and embeddings.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.config import ConfigDict


class EntityType(str, Enum):
    """Types of code entities that can be extracted."""
    FILE = "FILE"
    CLASS = "CLASS"
    FUNCTION = "FUNCTION"
    METHOD = "METHOD"
    VARIABLE = "VARIABLE"
    CONSTANT = "CONSTANT"
    MODULE = "MODULE"
    INTERFACE = "INTERFACE"
    ENUM = "ENUM"
    STRUCT = "STRUCT"
    NAMESPACE = "NAMESPACE"
    PROPERTY = "PROPERTY"
    FIELD = "FIELD"


class CodeEntity(BaseModel):
    """
    Individual code component with semantic metadata.

    Represents any extractable element from source code including
    files, classes, functions, variables with their context and embeddings.
    """

    # Core identification
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique entity identifier")
    repository_id: str = Field(..., description="Parent repository identifier")

    # Entity classification
    entity_type: EntityType = Field(..., description="Type of code entity")
    name: str = Field(..., description="Entity name or identifier")
    qualified_name: str = Field(..., description="Fully qualified name with namespace")

    # Location information
    file_path: str = Field(..., description="Relative path from repository root")
    start_line: int = Field(ge=1, description="Starting line number in file")
    end_line: int = Field(ge=1, description="Ending line number in file")
    start_column: int = Field(ge=0, description="Starting column position")
    end_column: int = Field(ge=0, description="Ending column position")

    # Language and context
    language: str = Field(..., description="Programming language")
    signature: Optional[str] = Field(None, description="Function/method signature or type declaration")
    docstring: Optional[str] = Field(None, description="Documentation string or comments")
    content: Optional[str] = Field(None, description="Source code content")

    # Semantic analysis fields (for coding agent context)
    dependencies: List[str] = Field(default_factory=list, description="Function/variable dependencies")
    side_effects: List[str] = Field(default_factory=list, description="Side effects and external interactions")
    error_handling: Dict[str, Any] = Field(default_factory=dict, description="Error handling patterns")
    control_flow_info: Dict[str, Any] = Field(default_factory=dict, description="Control flow analysis")
    data_flow_info: Dict[str, Any] = Field(default_factory=dict, description="Data flow analysis")
    usage_patterns: List[str] = Field(default_factory=list, description="Common usage patterns")

    # Enhanced flow analysis (Phase 2)
    call_patterns: Dict[str, Any] = Field(default_factory=dict, description="Function call patterns and dependencies")
    complexity_metrics: Dict[str, Any] = Field(default_factory=dict, description="Various complexity measurements")

    # Analysis metrics
    complexity_score: Optional[float] = Field(None, ge=0.0, description="Cyclomatic complexity score")
    line_count: int = Field(default=0, ge=0, description="Number of lines in entity")

    # Semantic information
    embedding_vector: Optional[List[float]] = Field(None, description="Semantic embedding vector")
    embedding_model: Optional[str] = Field(None, description="Model used for embedding generation")

    # Relationships context
    parent_entity_id: Optional[str] = Field(None, description="Parent entity (e.g., class for method)")
    child_entity_ids: List[str] = Field(default_factory=list, description="Direct child entities")

    # Additional metadata
    visibility: Optional[str] = Field(None, description="Visibility modifier (public, private, etc.)")
    is_abstract: bool = Field(default=False, description="Whether entity is abstract")
    is_static: bool = Field(default=False, description="Whether entity is static")
    is_async: bool = Field(default=False, description="Whether function is async")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Entity creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Entity update timestamp")

    # Flexible metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Language-specific metadata")

    # Pydantic configuration
    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "id": "770e8400-e29b-41d4-a716-446655440002",
                "repository_id": "550e8400-e29b-41d4-a716-446655440000",
                "entity_type": "FUNCTION",
                "name": "calculate_total",
                "qualified_name": "billing.utils.calculate_total",
                "file_path": "src/billing/utils.py",
                "start_line": 15,
                "end_line": 32,
                "start_column": 0,
                "end_column": 23,
                "language": "python",
                "signature": "def calculate_total(items: List[Item], tax_rate: float) -> float",
                "docstring": "Calculate total price including tax for a list of items.",
                "complexity_score": 3.2,
                "line_count": 18,
                "visibility": "public",
                "metadata": {
                    "returns": "float",
                    "parameters": ["items", "tax_rate"],
                    "decorators": ["@cache"]
                }
            }
        }
    )

    @field_validator("qualified_name")
    @classmethod
    def generate_qualified_name(cls, v: Optional[str], info) -> str:
        """Generate qualified name if not provided."""
        if v:
            return v

        # Access values from info.data in Pydantic V2
        if hasattr(info, 'data'):
            name = info.data.get("name", "unknown")
            file_path = info.data.get("file_path", "")

            if file_path:
                # Remove file extension and convert path to module-like name
                module_parts = file_path.replace("/", ".").replace("\\", ".")
                if "." in module_parts:
                    module_parts = ".".join(module_parts.split(".")[:-1])  # Remove extension
                return f"{module_parts}.{name}"

        return v or "unknown"

    @field_validator("end_line")
    @classmethod
    def validate_line_order(cls, v: int, info) -> int:
        """Validate that end_line >= start_line."""
        if hasattr(info, 'data'):
            start_line = info.data.get("start_line")
            if start_line and v < start_line:
                raise ValueError("end_line must be >= start_line")
        return v

    @model_validator(mode='after')
    def validate_column_positions(self) -> 'CodeEntity':
        """Validate column positions are consistent."""
        # If on same line, end_column must be >= start_column
        if (self.start_line == self.end_line and
            self.start_column is not None and self.end_column is not None and
            self.end_column < self.start_column):
            raise ValueError("end_column must be >= start_column when on same line")

        return self

    @field_validator("embedding_vector")
    @classmethod
    def validate_embedding_vector(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate embedding vector dimensions and values."""
        if v is None:
            return v

        if not isinstance(v, list):
            raise ValueError("embedding_vector must be a list of floats")

        if len(v) == 0:
            raise ValueError("embedding_vector cannot be empty")

        # Common embedding dimensions
        valid_dimensions = [128, 256, 384, 512, 768, 1024, 1536]
        if len(v) not in valid_dimensions:
            # Allow other dimensions but warn about common ones
            pass

        # Validate all values are numeric
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"embedding_vector[{i}] must be numeric, got {type(val)}")
            if abs(val) > 100:  # Reasonable bounds check
                raise ValueError(f"embedding_vector[{i}] value {val} seems out of reasonable bounds")

        return v

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate file path format."""
        if not v:
            raise ValueError("file_path cannot be empty")

        # Should be relative path
        if v.startswith("/") or (len(v) > 1 and v[1] == ":"):
            raise ValueError("file_path should be relative to repository root")

        return v.replace("\\", "/")  # Normalize path separators

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate and normalize language name."""
        if not v:
            raise ValueError("language cannot be empty")

        # Normalize common language names
        language_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "jsx": "javascript",
            "tsx": "typescript",
            "cpp": "c++",
            "cxx": "c++",
            "cc": "c++",
            "c": "c",
            "java": "java",
            "kt": "kotlin",
            "rb": "ruby",
            "go": "go",
            "rs": "rust",
            "php": "php",
            "cs": "c#",
            "fs": "f#",
            "vb": "vb.net",
            "swift": "swift",
            "scala": "scala",
            "clj": "clojure",
            "hs": "haskell",
        }

        normalized = language_map.get(v.lower(), v.lower())
        return normalized

    def calculate_line_count(self) -> int:
        """Calculate and update line count from position."""
        self.line_count = max(1, self.end_line - self.start_line + 1)
        return self.line_count

    def get_location_string(self) -> str:
        """Get human-readable location string."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def is_in_same_file(self, other: "CodeEntity") -> bool:
        """Check if this entity is in the same file as another."""
        return (self.repository_id == other.repository_id and
                self.file_path == other.file_path)

    def contains_line(self, line_number: int) -> bool:
        """Check if a line number is within this entity."""
        return self.start_line <= line_number <= self.end_line

    def overlaps_with(self, other: "CodeEntity") -> bool:
        """Check if this entity overlaps with another in the same file."""
        if not self.is_in_same_file(other):
            return False

        return not (self.end_line < other.start_line or
                   other.end_line < self.start_line)

    def get_embedding_similarity(self, other: "CodeEntity") -> Optional[float]:
        """Calculate cosine similarity with another entity's embedding."""
        if not self.embedding_vector or not other.embedding_vector:
            return None

        if len(self.embedding_vector) != len(other.embedding_vector):
            return None

        # Simple cosine similarity calculation
        import math

        dot_product = sum(a * b for a, b in zip(self.embedding_vector, other.embedding_vector))
        magnitude_a = math.sqrt(sum(a * a for a in self.embedding_vector))
        magnitude_b = math.sqrt(sum(b * b for b in other.embedding_vector))

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def update_content(self, new_content: str, new_signature: Optional[str] = None) -> None:
        """Update entity content and derived fields."""
        self.content = new_content
        if new_signature:
            self.signature = new_signature
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(by_alias=True, exclude_unset=False)

    def __str__(self) -> str:
        """String representation."""
        return f"CodeEntity({self.entity_type.value}, {self.qualified_name})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"CodeEntity(id={self.id}, type={self.entity_type.value}, "
            f"name={self.qualified_name}, location={self.get_location_string()})"
        )