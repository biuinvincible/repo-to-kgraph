"""
Contextual chunking service for maintaining context during code embedding.

Implements Anthropic's contextual retrieval approach for code, ensuring that
each chunk maintains sufficient context for accurate semantic understanding.
"""

import ast
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from repo_kgraph.models.code_entity import CodeEntity, EntityType


logger = logging.getLogger(__name__)


class ContextualChunk:
    """A chunk of code with its contextual information."""

    def __init__(
        self,
        content: str,
        context_prefix: str,
        entity: CodeEntity,
        chunk_type: str,
        metadata: Dict[str, Any] = None
    ):
        self.content = content
        self.context_prefix = context_prefix
        self.entity = entity
        self.chunk_type = chunk_type
        self.metadata = metadata or {}

    @property
    def full_content_with_context(self) -> str:
        """Get the complete content with contextual prefix."""
        return f"{self.context_prefix}\n\n{self.content}"

    @property
    def embedding_text(self) -> str:
        """Get the text that should be used for embedding generation."""
        return self.full_content_with_context


class ContextualChunker:
    """
    Service for creating contextually-aware code chunks.

    Based on Anthropic's contextual retrieval approach, this service ensures
    that each code chunk maintains sufficient context for accurate retrieval
    and understanding by coding agents.
    """

    def __init__(
        self,
        max_chunk_size: int = 2048,
        context_window: int = 512,
        overlap_size: int = 256
    ):
        """
        Initialize the contextual chunker.

        Args:
            max_chunk_size: Maximum size of each chunk in characters
            context_window: Size of context to include with each chunk
            overlap_size: Overlap between adjacent chunks
        """
        self.max_chunk_size = max_chunk_size
        self.context_window = context_window
        self.overlap_size = overlap_size

    def create_contextual_chunks(
        self,
        entities: List[CodeEntity],
        file_content: str,
        file_path: str
    ) -> List[ContextualChunk]:
        """
        Create contextual chunks from code entities.

        Args:
            entities: List of code entities from the file
            file_content: Complete file content
            file_path: Path to the source file

        Returns:
            List of contextual chunks with context information
        """
        chunks = []
        file_context = self._build_file_context(file_content, file_path)

        for entity in entities:
            if entity.entity_type in [EntityType.FUNCTION, EntityType.CLASS]:
                entity_chunks = self._chunk_entity_with_context(
                    entity, file_context, file_content
                )
                chunks.extend(entity_chunks)

        return chunks

    def _build_file_context(self, file_content: str, file_path: str) -> Dict[str, Any]:
        """Build contextual information about the file."""
        try:
            parsed_ast = ast.parse(file_content)

            # Extract imports
            imports = []
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"from {module} import {alias.name}")

            # Extract top-level classes and functions for reference
            top_level_entities = []
            for node in parsed_ast.body:
                if isinstance(node, ast.FunctionDef):
                    top_level_entities.append(f"function: {node.name}")
                elif isinstance(node, ast.ClassDef):
                    top_level_entities.append(f"class: {node.name}")

            return {
                "file_path": file_path,
                "imports": imports,
                "top_level_entities": top_level_entities,
                "module_docstring": ast.get_docstring(parsed_ast)
            }
        except Exception as e:
            logger.warning(f"Failed to parse file context for {file_path}: {e}")
            return {
                "file_path": file_path,
                "imports": [],
                "top_level_entities": [],
                "module_docstring": None
            }

    def _chunk_entity_with_context(
        self,
        entity: CodeEntity,
        file_context: Dict[str, Any],
        file_content: str
    ) -> List[ContextualChunk]:
        """Create contextual chunks for a single entity."""
        chunks = []

        if not entity.content:
            return chunks

        # For smaller entities, create a single chunk
        if len(entity.content) <= self.max_chunk_size:
            context_prefix = self._build_context_prefix(entity, file_context)
            chunk = ContextualChunk(
                content=entity.content,
                context_prefix=context_prefix,
                entity=entity,
                chunk_type="complete_entity"
            )
            chunks.append(chunk)
        else:
            # For larger entities, split into multiple chunks with context
            entity_chunks = self._split_large_entity(entity, file_context)
            chunks.extend(entity_chunks)

        return chunks

    def _build_context_prefix(
        self,
        entity: CodeEntity,
        file_context: Dict[str, Any]
    ) -> str:
        """Build contextual prefix for an entity chunk."""
        context_parts = []

        # File information
        context_parts.append(f"File: {file_context['file_path']}")

        # Module docstring if available
        if file_context.get("module_docstring"):
            context_parts.append(f"Module: {file_context['module_docstring'][:200]}...")

        # Relevant imports
        if file_context.get("imports"):
            context_parts.append("Imports:")
            for imp in file_context["imports"][:5]:  # Limit to first 5 imports
                context_parts.append(f"  {imp}")

        # Entity type and location
        entity_type_str = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
        context_parts.append(f"Entity: {entity_type_str} '{entity.name}' (lines {entity.start_line}-{entity.end_line})")

        # Parent context if available
        if entity.parent_entity_id:
            context_parts.append(f"Parent: {entity.parent_entity_id}")

        # Dependencies
        if entity.dependencies:
            context_parts.append(f"Dependencies: {', '.join(entity.dependencies[:5])}")

        # Control flow complexity
        if entity.control_flow_info:
            complexity = entity.control_flow_info.get("complexity_indicators", [])
            if complexity:
                context_parts.append(f"Complexity: {', '.join(complexity[:3])}")

        return "\n".join(context_parts)

    def _split_large_entity(
        self,
        entity: CodeEntity,
        file_context: Dict[str, Any]
    ) -> List[ContextualChunk]:
        """Split large entities into multiple contextual chunks."""
        chunks = []
        content = entity.content

        # Try to split at logical boundaries (methods, blocks)
        split_points = self._find_logical_split_points(content)

        if not split_points:
            # Fallback to character-based splitting
            split_points = self._create_character_splits(content)

        for i, (start, end) in enumerate(split_points):
            chunk_content = content[start:end]

            # Build context for this specific chunk
            context_prefix = self._build_chunk_context_prefix(
                entity, file_context, i, len(split_points), chunk_content
            )

            chunk = ContextualChunk(
                content=chunk_content,
                context_prefix=context_prefix,
                entity=entity,
                chunk_type="entity_fragment",
                metadata={
                    "chunk_index": i,
                    "total_chunks": len(split_points),
                    "start_offset": start,
                    "end_offset": end
                }
            )
            chunks.append(chunk)

        return chunks

    def _find_logical_split_points(self, content: str) -> List[Tuple[int, int]]:
        """Find logical boundaries for splitting large code entities."""
        try:
            parsed_ast = ast.parse(content)
            split_points = []

            # Split at method boundaries for classes
            methods = [node for node in ast.walk(parsed_ast)
                      if isinstance(node, ast.FunctionDef)]

            if methods:
                current_pos = 0
                for method in methods:
                    # Calculate method boundaries in source
                    method_source = ast.unparse(method)
                    method_start = content.find(method_source, current_pos)
                    if method_start != -1:
                        method_end = method_start + len(method_source)
                        split_points.append((method_start, method_end))
                        current_pos = method_end

            return split_points
        except Exception:
            return []

    def _create_character_splits(self, content: str) -> List[Tuple[int, int]]:
        """Create character-based splits with overlap."""
        splits = []
        content_length = len(content)

        start = 0
        while start < content_length:
            end = min(start + self.max_chunk_size, content_length)

            # Try to break at line boundaries
            if end < content_length:
                last_newline = content.rfind('\n', start, end)
                if last_newline > start:
                    end = last_newline

            splits.append((start, end))
            start = end - self.overlap_size

            if start >= content_length:
                break

        return splits

    def _build_chunk_context_prefix(
        self,
        entity: CodeEntity,
        file_context: Dict[str, Any],
        chunk_index: int,
        total_chunks: int,
        chunk_content: str
    ) -> str:
        """Build context prefix for a fragment chunk."""
        base_context = self._build_context_prefix(entity, file_context)

        # Add chunk-specific information
        chunk_info = f"Chunk {chunk_index + 1} of {total_chunks} for {entity.entity_type.value} '{entity.name}'"

        # Add a brief description of what this chunk contains
        chunk_description = self._describe_chunk_content(chunk_content)

        return f"{base_context}\n{chunk_info}\n{chunk_description}"

    def _describe_chunk_content(self, content: str) -> str:
        """Generate a brief description of chunk content."""
        try:
            parsed_ast = ast.parse(content)

            descriptions = []
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.FunctionDef):
                    descriptions.append(f"method: {node.name}")
                elif isinstance(node, ast.If):
                    descriptions.append("conditional logic")
                elif isinstance(node, (ast.For, ast.While)):
                    descriptions.append("loop")
                elif isinstance(node, ast.Try):
                    descriptions.append("error handling")

            if descriptions:
                return f"Contains: {', '.join(descriptions[:3])}"
            else:
                return "Contains: implementation details"
        except Exception:
            # Fallback to simple line counting
            lines = len(content.split('\n'))
            return f"Contains: {lines} lines of code"