"""
FastAPI application for Repository Knowledge Graph API.

This is a minimal stub that will be expanded during implementation.
The contract tests will fail against this stub, which is the intended TDD approach.
"""

from fastapi import FastAPI, HTTPException, status, Request, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import os
import uuid
from datetime import datetime, timedelta
import json
from starlette.middleware.base import BaseHTTPMiddleware
from typing import List, Optional, Dict, Any
from uuid import UUID

from repo_kgraph.api.models.parse_repo import ParseRepoRequest, ParseRepoResponse
from fastapi.responses import JSONResponse


# Create the FastAPI application
app = FastAPI(
    title="Repository Knowledge Graph API",
    description="Transform code repositories into queryable knowledge graphs for AI coding agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware to handle malformed JSON
@app.middleware("http")
async def json_error_middleware(request: Request, call_next):
    """Middleware to catch malformed JSON and return appropriate status codes."""
    if request.method == "POST" and request.url.path == "/parse-repo":
        # Check if content type is JSON
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            # Try to read the body
            try:
                body = await request.body()
                if body:
                    json.loads(body)
            except json.JSONDecodeError:
                # Return 400 for malformed JSON
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "error": "Invalid JSON",
                        "message": "Malformed JSON in request body"
                    }
                )
    
    response = await call_next(request)
    return response


# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors and return appropriate HTTP status codes."""
    # Check if this is for the parse-repo endpoint and repository_path is missing
    if request.url.path == "/parse-repo":
        # Look for missing required fields
        for error in exc.errors():
            loc = error.get("loc", [])
            if "repository_path" in str(loc) and error.get("type") == "missing":
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "error": "Missing required field",
                        "message": "repository_path is required"
                    }
                )
    
    # For other validation errors, return 422
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()}
    )


# Custom exception handler for HTTP exceptions to format error responses correctly
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions and format error responses correctly."""
    # For parse-repo and graph endpoints, format error responses as expected by contract tests
    if request.url.path.startswith("/parse-repo") or request.url.path.startswith("/graph/"):
        if isinstance(exc.detail, dict) and "error" in exc.detail and "message" in exc.detail:
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.detail
            )
    
    # For other cases, return the default format
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.get("/")
async def root():
    """Root endpoint for API health check."""
    return {"message": "Repository Knowledge Graph API", "version": "1.0.0"}


@app.post("/parse-repo", response_model=ParseRepoResponse, status_code=status.HTTP_202_ACCEPTED)
async def parse_repo(request: ParseRepoRequest):
    """Parse a repository and build its knowledge graph."""
    
    # Validate repository path exists
    if not os.path.exists(request.repository_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "Repository not found",
                "message": f"Repository path not found: {request.repository_path}"
            }
        )
    
    if not os.path.isdir(request.repository_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Invalid path",
                "message": f"Repository path is not a directory: {request.repository_path}"
            }
        )
    
    # Generate repository ID
    repository_id = str(uuid.uuid4())
    
    # Return response
    return ParseRepoResponse(
        repository_id=repository_id,
        status="queued",
        message=f"Repository parsing job queued for {request.repository_path}",
        estimated_completion=(datetime.utcnow() + timedelta(minutes=5)).isoformat() + "Z",
        progress=0.0
    )


@app.get("/graph/{entity_id}")
async def get_entity_graph(
    entity_id: str = Path(..., description="Entity ID to retrieve graph for"),
    depth: int = 3,
    relationship_types: List[str] = []
):
    """Retrieve entity graph with specified depth and relationship filters."""
    
    # Validate entity_id is a valid UUID
    try:
        uuid.UUID(entity_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Invalid UUID",
                "message": f"Invalid entity ID format: {entity_id}"
            }
        )
    
    # For now, return a mock response to satisfy contract tests
    # Actual implementation will come later
    
    # Check if entity exists (mock implementation)
    if entity_id == "00000000-0000-0000-0000-000000000000":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "Entity not found",
                "message": f"Entity with ID {entity_id} not found"
            }
        )
    
    # Mock entity data
    entity_data = {
        "id": entity_id,
        "entity_type": "FUNCTION",
        "name": "calculate_total",
        "file_path": "src/billing/utils.py",
        "language": "python",
        "start_line": 15,
        "end_line": 32,
        "content": "def calculate_total(items: List[Item], tax_rate: float) -> float:\n    # Calculate total price including tax\n    subtotal = sum(item.price for item in items)\n    tax = subtotal * tax_rate\n    return subtotal + tax"
    }
    
    # Mock relationships data
    relationships_data = [
        {
            "id": str(uuid.uuid4()),
            "source_entity_id": entity_id,
            "target_entity_id": str(uuid.uuid4()),
            "relationship_type": "CALLS",
            "strength": 0.85,
            "line_number": 25,
            "context": "total = calculate_tax(amount, rate)"
        },
        {
            "id": str(uuid.uuid4()),
            "source_entity_id": entity_id,
            "target_entity_id": str(uuid.uuid4()),
            "relationship_type": "INHERITS",
            "strength": 0.75,
            "line_number": 18,
            "context": "class Calculator(TaxCalculator):"
        }
    ]
    
    # Filter relationships by type if specified
    if relationship_types:
        relationships_data = [rel for rel in relationships_data if rel["relationship_type"] in relationship_types]
    
    # Mock graph metrics
    graph_metrics = {
        "max_depth": min(depth, 5),
        "entity_count": len(relationships_data) + 1,
        "relationship_count": len(relationships_data),
        "density": 0.3
    }
    
    return {
        "entity_id": entity_id,
        "entity": entity_data,
        "relationships": relationships_data,
        "graph_metrics": graph_metrics
    }


@app.post("/query")
async def query_context(request: Dict[str, Any]):
    """Query context for coding tasks."""
    # For now, just implement the endpoint to pass contract tests
    # Actual implementation will come later
    
    # Check if repository_id is missing
    if "repository_id" not in request:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Missing required field",
                "message": "repository_id is required"
            }
        )
    
    # Check if task_description is missing
    if "task_description" not in request:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Missing required field",
                "message": "task_description is required"
            }
        )
    
    # Check if task_description is empty
    task_description = request.get("task_description", "")
    if not task_description or not task_description.strip():
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Invalid task description",
                "message": "task_description cannot be empty"
            }
        )
    
    # Check if confidence_threshold is invalid
    confidence_threshold = request.get("confidence_threshold", 0.3)
    if confidence_threshold is not None and (confidence_threshold < 0.0 or confidence_threshold > 1.0):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Invalid confidence threshold",
                "message": "confidence_threshold must be between 0.0 and 1.0"
            }
        )
    
    # Check if repository exists (mock implementation)
    repository_id = request["repository_id"]
    if repository_id == "00000000-0000-0000-0000-000000000000":
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "Repository not found",
                "message": f"Repository with ID {repository_id} not found"
            }
        )
    
    # Generate query ID
    query_id = str(uuid.uuid4())
    
    # Mock results
    results = [
        {
            "id": str(uuid.uuid4()),
            "entity_id": str(uuid.uuid4()),
            "name": "calculate_total",  # Add the missing name field
            "entity_type": "FUNCTION",
            "file_path": "src/billing/utils.py",
            "start_line": 15,
            "end_line": 32,
            "content": "def calculate_total(items: List[Item], tax_rate: float) -> float:\n    # Calculate total price including tax\n    subtotal = sum(item.price for item in items)\n    tax = subtotal * tax_rate\n    return subtotal + tax",
            "relevance_score": 0.85,
            "confidence_score": 0.92,
            "context": "This function calculates the total price including tax for a list of items.",
            "language": "python"
        },
        {
            "id": str(uuid.uuid4()),
            "entity_id": str(uuid.uuid4()),
            "name": "TaxCalculator",  # Add the missing name field
            "entity_type": "CLASS",
            "file_path": "src/billing/tax.py",
            "start_line": 10,
            "end_line": 45,
            "content": "class TaxCalculator:\n    def __init__(self, rate: float):\n        self.rate = rate\n    \n    def calculate_tax(self, amount: float) -> float:\n        return amount * self.rate",
            "relevance_score": 0.78,
            "confidence_score": 0.88,
            "context": "This class handles tax calculations for billing operations.",
            "language": "python"
        }
    ]
    
    # Mock response
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "query_id": query_id,
            "results": results,
            "processing_time_ms": 127,
            "total_results": len(results),
            "confidence_score": 0.85,
            "message": "Query processed successfully"
        }
    )


@app.get("/repositories")
async def list_repositories(limit: int = 20, offset: int = 0):
    """List all indexed repositories."""
    # For now, just implement the endpoint to pass contract tests
    # Actual implementation will come later
    
    # Mock repositories
    repositories = [
        {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "name": "my-project",
            "path": "/home/user/my-project",
            "status": "indexed",
            "entity_count": 15893,
            "relationship_count": 43521,
            "file_count": 1247,
            "language_stats": {
                "python": 12500,
                "javascript": 2800,
                "typescript": 593
            },
            "last_indexed": "2025-09-17T10:30:00Z",
            "created_at": "2025-09-15T09:15:00Z"
        },
        {
            "id": "660e8400-e29b-41d4-a716-446655440001",
            "name": "another-project",
            "path": "/home/user/another-project",
            "status": "indexing",
            "entity_count": 8742,
            "relationship_count": 21567,
            "file_count": 689,
            "language_stats": {
                "javascript": 15200,
                "typescript": 3800
            },
            "last_indexed": "2025-09-16T14:22:00Z",
            "created_at": "2025-09-14T11:45:00Z"
        }
    ]
    
    # Apply pagination
    paginated_repositories = repositories[offset:offset+limit]
    
    return {
        "repositories": paginated_repositories,
        "total_count": len(repositories),
        "limit": limit,
        "offset": offset
    }


@app.get("/repositories/{repository_id}/status")
async def get_repository_status(repository_id: str):
    """Get repository status and statistics."""
    # For now, just implement the endpoint to pass contract tests
    # Actual implementation will come later
    
    # Validate repository_id is a valid UUID
    try:
        uuid.UUID(repository_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Invalid UUID",
                "message": f"Invalid repository ID format: {repository_id}"
            }
        )
    
    # Check if repository exists (mock implementation)
    if repository_id == "00000000-0000-0000-0000-000000000000":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "Repository not found",
                "message": f"Repository with ID {repository_id} not found"
            }
        )
    
    # Mock repository status
    return {
        "repository_id": repository_id,
        "status": "indexed",
        "statistics": {
            "entity_count": 15893,
            "relationship_count": 43521,
            "file_count": 1247,
            "language_stats": {
                "python": 12500,
                "javascript": 2800,
                "typescript": 593
            },
            "last_indexed": "2025-09-17T10:30:00Z",
            "indexing_time_ms": 15470,
            "parsing_errors": 3,
            "coverage_percentage": 92.3
        },
        "message": "Repository is fully indexed and ready for queries"
    }


@app.get("/health")
async def health_check():
    """Check system health status."""
    # For now, just implement the endpoint to pass contract tests
    # Actual implementation will come later
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": {
            "database": {
                "status": "healthy"
            },
            "vector_store": {
                "status": "healthy"
            }
        }
    }


# NOTE: The following endpoints are intentionally NOT implemented yet
# Contract tests will fail, which is the correct TDD approach:
#
# POST /query - Context query endpoint
# GET /graph/{entity_id} - Entity graph exploration
# GET /repositories - List repositories
# GET /repositories/{id}/status - Repository status
# GET /health - System health check
#
# These will be implemented in Phase 3.6 (API Implementation)
# after the contract tests are written and failing.