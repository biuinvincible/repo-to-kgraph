"""
Repository entity model for the Knowledge Graph system.

Represents a source code repository with metadata and validation.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.config import ConfigDict


class RepositoryStatus(str, Enum):
    """Repository indexing status."""
    UNINDEXED = "unindexed"
    INDEXING = "indexing"
    INDEXED = "indexed"
    STALE = "stale"
    ERROR = "error"


class Repository(BaseModel):
    """
    Repository entity representing a source code repository.

    This model handles the root container for all code analysis data,
    including metadata, statistics, and state management.
    """

    # Core identification
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique repository identifier")
    path: str = Field(..., description="Absolute file system path to repository root")
    name: str = Field(..., description="Repository name derived from directory")

    # Statistics
    size_bytes: int = Field(ge=0, description="Total repository size in bytes")
    file_count: int = Field(ge=0, description="Number of tracked files")
    entity_count: int = Field(default=0, ge=0, description="Number of extracted code entities")
    relationship_count: int = Field(default=0, ge=0, description="Number of identified relationships")
    indexing_time_ms: int = Field(default=0, ge=0, description="Time taken to index repository in milliseconds")

    # Language distribution
    language_stats: Dict[str, int] = Field(
        default_factory=dict,
        description="Map of programming language to line count"
    )

    # Timestamps
    last_indexed: Optional[datetime] = Field(None, description="Timestamp of last complete indexing")
    last_modified: Optional[datetime] = Field(None, description="Repository last modification timestamp")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Record creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Record update timestamp")

    # Version control
    git_hash: Optional[str] = Field(None, description="Current git commit hash if git repository")
    branch: Optional[str] = Field(None, description="Current git branch")

    # Status and metadata
    status: RepositoryStatus = Field(default=RepositoryStatus.UNINDEXED, description="Current indexing status")
    error_message: Optional[str] = Field(None, description="Error message if status is ERROR")
    parsing_errors: list[str] = Field(default_factory=list, description="List of files that failed to parse")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional repository metadata")

    # Configuration
    exclude_patterns: list[str] = Field(
        default_factory=list,
        description="File patterns to exclude from parsing"
    )
    include_languages: list[str] = Field(
        default_factory=list,
        description="Specific languages to include (empty = all)"
    )

    # Pydantic configuration
    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "path": "/home/user/my-project",
                "name": "my-project",
                "size_bytes": 1048576,
                "file_count": 247,
                "entity_count": 1893,
                "relationship_count": 4521,
                "language_stats": {
                    "python": 15000,
                    "javascript": 8500,
                    "typescript": 3200
                },
                "last_indexed": "2025-09-17T10:30:00Z",
                "last_modified": "2025-09-17T09:15:00Z",
                "git_hash": "abc123def456",
                "branch": "main",
                "status": "indexed",
                "exclude_patterns": ["node_modules/*", "*.test.js"],
                "include_languages": ["python", "javascript"]
            }
        }
    )

    @field_validator("path")
    @classmethod
    def validate_path_exists(cls, v: str) -> str:
        """Validate that repository path exists and is accessible."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Repository path does not exist: {v}")
        if not path.is_dir():
            raise ValueError(f"Repository path is not a directory: {v}")
        if not path.resolve().is_absolute():
            raise ValueError(f"Repository path must be absolute: {v}")
        return str(path.resolve())

    @field_validator("name")
    @classmethod
    def derive_name_from_path(cls, v: Optional[str], info) -> str:
        """Derive repository name from path if not provided."""
        if v:
            return v
        # Access values from info.data in Pydantic V2
        if hasattr(info, 'data') and "path" in info.data:
            return Path(info.data["path"]).name
        return "unknown"

    @field_validator("git_hash")
    @classmethod
    def validate_git_hash(cls, v: Optional[str]) -> Optional[str]:
        """Validate git hash format if provided."""
        if v is None:
            return v
        if len(v) < 7 or len(v) > 40:
            raise ValueError("Git hash must be 7-40 characters long")
        if not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("Git hash must contain only hexadecimal characters")
        return v.lower()

    @model_validator(mode='after')
    def validate_timestamps(self) -> 'Repository':
        """Validate timestamp relationships."""
        now = datetime.utcnow()

        # Future timestamp validation
        for field_name, timestamp in [
            ("last_indexed", self.last_indexed),
            ("last_modified", self.last_modified),
            ("created_at", self.created_at),
            ("updated_at", self.updated_at),
        ]:
            if timestamp and timestamp > now:
                raise ValueError(f"{field_name} cannot be in the future")

        # Relationship validation
        if self.created_at and self.updated_at and self.created_at > self.updated_at:
            raise ValueError("created_at cannot be after updated_at")

        return self

    @field_validator("language_stats")
    @classmethod
    def validate_language_stats(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Validate language statistics."""
        for language, line_count in v.items():
            if not isinstance(language, str) or not language:
                raise ValueError("Language name must be non-empty string")
            if not isinstance(line_count, int) or line_count < 0:
                raise ValueError(f"Line count for {language} must be non-negative integer")
        return v

    def update_status(self, new_status: RepositoryStatus, error_message: Optional[str] = None) -> None:
        """Update repository status with timestamp."""
        self.status = new_status
        self.error_message = error_message
        self.updated_at = datetime.utcnow()

        if new_status == RepositoryStatus.INDEXED:
            self.last_indexed = datetime.utcnow()
            self.error_message = None  # Clear any previous errors

    def is_stale(self) -> bool:
        """Check if repository needs re-indexing based on modification time."""
        if not self.last_indexed or not self.last_modified:
            return True
        return self.last_modified > self.last_indexed

    def get_primary_language(self) -> Optional[str]:
        """Get the primary programming language by line count."""
        if not self.language_stats:
            return None
        return max(self.language_stats.items(), key=lambda x: x[1])[0]

    def get_total_lines(self) -> int:
        """Get total lines of code across all languages."""
        return sum(self.language_stats.values())

    def add_language_stats(self, language: str, lines: int) -> None:
        """Add or update language statistics."""
        if language in self.language_stats:
            self.language_stats[language] += lines
        else:
            self.language_stats[language] = lines
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(by_alias=True, exclude_unset=False)

    def __str__(self) -> str:
        """String representation."""
        return f"Repository({self.name}, {self.status.value}, {self.entity_count} entities)"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"Repository(id={self.id}, name={self.name}, path={self.path}, "
            f"status={self.status.value}, entities={self.entity_count}, "
            f"files={self.file_count})"
        )