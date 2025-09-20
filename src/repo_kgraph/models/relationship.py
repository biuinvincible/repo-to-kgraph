"""
Relationship model for connections between code entities.

Handles semantic and structural dependencies between code components
with relationship types and strength scoring.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.config import ConfigDict


class RelationshipType(str, Enum):
    """Types of relationships between code entities."""
    IMPORTS = "IMPORTS"              # Module/package import statements
    CALLS = "CALLS"                  # Function/method invocations
    INHERITS = "INHERITS"            # Class inheritance relationships
    IMPLEMENTS = "IMPLEMENTS"        # Interface implementation
    CONTAINS = "CONTAINS"            # Parent-child containment
    REFERENCES = "REFERENCES"        # Variable/type references
    DEPENDS_ON = "DEPENDS_ON"        # Broader dependency relationships
    EXTENDS = "EXTENDS"              # Extension relationships
    OVERRIDES = "OVERRIDES"          # Method overriding
    INSTANTIATES = "INSTANTIATES"    # Object creation
    ANNOTATES = "ANNOTATES"          # Type annotations
    DECORATES = "DECORATES"          # Decorator applications
    ACCESSES = "ACCESSES"            # Property/field access


class Relationship(BaseModel):
    """
    Connection between code entities with semantic meaning.

    Represents any type of relationship between code components including
    imports, calls, inheritance, and dependencies with context and scoring.
    """

    # Core identification
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique relationship identifier")
    source_entity_id: str = Field(..., description="Source entity identifier")
    target_entity_id: str = Field(..., description="Target entity identifier")

    # Relationship classification
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship strength score")

    # Location context
    line_number: Optional[int] = Field(None, ge=1, description="Source line where relationship occurs")
    column_number: Optional[int] = Field(None, ge=0, description="Source column where relationship occurs")
    context: Optional[str] = Field(None, description="Surrounding code context")

    # Relationship properties
    is_direct: bool = Field(default=True, description="Whether relationship is direct or inferred")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in relationship detection")
    frequency: int = Field(default=1, ge=1, description="Number of times relationship occurs")

    # Conditional properties
    is_conditional: bool = Field(default=False, description="Whether relationship is conditional")
    condition_context: Optional[str] = Field(None, description="Condition under which relationship exists")

    # Semantic information
    semantic_role: Optional[str] = Field(None, description="Semantic role in the relationship")
    direction: str = Field(default="outgoing", description="Relationship direction from source perspective")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Relationship creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Relationship update timestamp")

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Relationship-specific metadata")

    # Pydantic configuration
    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "id": "880e8400-e29b-41d4-a716-446655440003",
                "source_entity_id": "770e8400-e29b-41d4-a716-446655440002",
                "target_entity_id": "990e8400-e29b-41d4-a716-446655440004",
                "relationship_type": "CALLS",
                "strength": 0.85,
                "line_number": 42,
                "context": "total = calculate_tax(amount, rate)",
                "is_direct": True,
                "confidence": 0.95,
                "frequency": 3,
                "semantic_role": "function_call",
                "metadata": {
                    "parameters": ["amount", "rate"],
                    "return_used": True,
                    "async_call": False
                }
            }
        }
    )

    @field_validator("source_entity_id", "target_entity_id")
    @classmethod
    def validate_entity_ids(cls, v: str) -> str:
        """Validate entity ID format."""
        if not v or not isinstance(v, str):
            raise ValueError("Entity ID must be non-empty string")
        # Could add UUID format validation here if needed
        return v

    @model_validator(mode='after')
    def validate_no_self_relationship(self) -> 'Relationship':
        """Validate that source and target entities are different."""
        if self.source_entity_id and self.target_entity_id and self.source_entity_id == self.target_entity_id:
            raise ValueError("Relationship cannot have the same source and target entity")
        return self

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        """Validate relationship direction."""
        valid_directions = ["outgoing", "incoming", "bidirectional"]
        if v not in valid_directions:
            raise ValueError(f"Direction must be one of {valid_directions}")
        return v

    @model_validator(mode='after')
    def validate_conditional_context(self) -> 'Relationship':
        """Validate conditional relationship properties."""
        if self.is_conditional and not self.condition_context:
            raise ValueError("condition_context is required when is_conditional is True")
        return self

    @field_validator("context")
    @classmethod
    def validate_context_length(cls, v: Optional[str]) -> Optional[str]:
        """Validate context string length."""
        if v and len(v) > 1000:  # Reasonable limit
            raise ValueError("Context string too long (max 1000 characters)")
        return v

    def calculate_composite_score(self) -> float:
        """Calculate composite relationship score combining strength and confidence."""
        return (self.strength * self.confidence +
               min(self.frequency / 10.0, 1.0) * 0.1)  # Frequency bonus capped at 0.1

    def is_structural_relationship(self) -> bool:
        """Check if relationship is structural (not behavioral)."""
        structural_types = {
            RelationshipType.INHERITS,
            RelationshipType.IMPLEMENTS,
            RelationshipType.CONTAINS,
            RelationshipType.EXTENDS
        }
        return self.relationship_type in structural_types

    def is_behavioral_relationship(self) -> bool:
        """Check if relationship is behavioral (runtime)."""
        behavioral_types = {
            RelationshipType.CALLS,
            RelationshipType.ACCESSES,
            RelationshipType.INSTANTIATES
        }
        return self.relationship_type in behavioral_types

    def is_dependency_relationship(self) -> bool:
        """Check if relationship indicates dependency."""
        dependency_types = {
            RelationshipType.IMPORTS,
            RelationshipType.DEPENDS_ON,
            RelationshipType.REFERENCES,
            RelationshipType.CALLS
        }
        return self.relationship_type in dependency_types

    def get_reverse_type(self) -> Optional[RelationshipType]:
        """Get the reverse relationship type if applicable."""
        reverse_map = {
            RelationshipType.INHERITS: RelationshipType.EXTENDS,
            RelationshipType.EXTENDS: RelationshipType.INHERITS,
            RelationshipType.CONTAINS: None,  # No clear reverse
            RelationshipType.CALLS: None,     # Asymmetric
            RelationshipType.IMPORTS: None,   # Asymmetric
        }
        return reverse_map.get(self.relationship_type)

    def update_frequency(self, increment: int = 1) -> None:
        """Update relationship frequency and timestamp."""
        self.frequency += increment
        self.updated_at = datetime.utcnow()

    def update_strength(self, new_strength: float, reason: Optional[str] = None) -> None:
        """Update relationship strength with optional reason."""
        if not 0.0 <= new_strength <= 1.0:
            raise ValueError("Strength must be between 0.0 and 1.0")

        self.strength = new_strength
        self.updated_at = datetime.utcnow()

        if reason:
            if "strength_updates" not in self.metadata:
                self.metadata["strength_updates"] = []
            self.metadata["strength_updates"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "old_strength": self.strength,
                "new_strength": new_strength,
                "reason": reason
            })

    def add_context_information(self, key: str, value: Any) -> None:
        """Add contextual information to metadata."""
        self.metadata[key] = value
        self.updated_at = datetime.utcnow()

    def get_relationship_description(self) -> str:
        """Get human-readable relationship description."""
        type_descriptions = {
            RelationshipType.IMPORTS: "imports",
            RelationshipType.CALLS: "calls",
            RelationshipType.INHERITS: "inherits from",
            RelationshipType.IMPLEMENTS: "implements",
            RelationshipType.CONTAINS: "contains",
            RelationshipType.REFERENCES: "references",
            RelationshipType.DEPENDS_ON: "depends on",
            RelationshipType.EXTENDS: "extends",
            RelationshipType.OVERRIDES: "overrides",
            RelationshipType.INSTANTIATES: "creates instance of",
            RelationshipType.ANNOTATES: "is annotated with",
            RelationshipType.DECORATES: "decorates",
            RelationshipType.ACCESSES: "accesses",
        }

        description = type_descriptions.get(self.relationship_type, str(self.relationship_type).lower())

        if self.is_conditional:
            description = f"conditionally {description}"

        if not self.is_direct:
            description = f"indirectly {description}"

        return description

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(by_alias=True, exclude_unset=False)

    def __str__(self) -> str:
        """String representation."""
        return f"Relationship({self.relationship_type.value}, strength={self.strength:.2f})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"Relationship(id={self.id}, type={self.relationship_type.value}, "
            f"source={self.source_entity_id}, target={self.target_entity_id}, "
            f"strength={self.strength:.2f}, confidence={self.confidence:.2f})"
        )