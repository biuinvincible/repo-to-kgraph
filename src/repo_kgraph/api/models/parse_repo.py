"""API models for repository parsing requests and responses."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ParseRepoRequest(BaseModel):
    """Request model for parsing a repository."""
    
    repository_path: str = Field(
        ..., 
        description="Absolute path to the repository directory to parse"
    )
    
    incremental: bool = Field(
        default=False,
        description="Whether to perform incremental parsing (only changed files)"
    )
    
    languages: Optional[List[str]] = Field(
        default=None,
        description="List of programming languages to include (None = all)"
    )
    
    exclude_patterns: Optional[List[str]] = Field(
        default=None,
        description="List of file patterns to exclude from parsing"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "repository_path": "/home/user/my-project",
                "incremental": False,
                "languages": ["python", "javascript"],
                "exclude_patterns": ["node_modules/*", "*.test.js", "__pycache__/*"]
            }
        }


class ParseRepoResponse(BaseModel):
    """Response model for repository parsing requests."""
    
    repository_id: str = Field(
        ...,
        description="Unique identifier for the repository"
    )
    
    status: str = Field(
        ...,
        description="Current status of the parsing operation",
        enum=["queued", "processing", "completed", "failed"]
    )
    
    message: str = Field(
        ...,
        description="Human-readable status message"
    )
    
    estimated_completion: Optional[str] = Field(
        default=None,
        description="Estimated completion time (ISO 8601 format)"
    )
    
    progress: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Progress percentage (0-100)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "repository_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "queued",
                "message": "Repository parsing job queued",
                "estimated_completion": "2025-09-17T15:30:00Z",
                "progress": 0.0
            }
        }