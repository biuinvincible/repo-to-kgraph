"""
Call graph analysis service for understanding function dependencies and call patterns.

Builds comprehensive call graphs across the entire repository to help coding agents
understand function interactions, dependencies, and execution patterns.
"""

import ast
import logging
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque

from repo_kgraph.models.code_entity import CodeEntity, EntityType
from repo_kgraph.models.relationship import Relationship, RelationshipType

logger = logging.getLogger(__name__)


@dataclass
class CallNode:
    """A node in the call graph representing a function."""
    entity_id: str
    entity_name: str
    file_path: str
    calls_to: Set[str]  # Entity IDs this function calls
    called_by: Set[str]  # Entity IDs that call this function
    call_count: int  # Total number of calls made by this function
    is_recursive: bool
    is_entry_point: bool  # Has no callers (potential entry point)
    depth_level: int  # Distance from entry points


@dataclass
class CallPath:
    """A path through the call graph."""
    path: List[str]  # List of entity IDs
    total_calls: int
    path_complexity: float
    has_cycles: bool


@dataclass
class CallGraphMetrics:
    """Metrics about the call graph structure."""
    total_functions: int
    total_calls: int
    max_depth: int
    cyclic_dependencies: List[List[str]]
    entry_points: List[str]
    most_called_functions: List[Tuple[str, int]]
    complexity_hotspots: List[str]


class CallGraphAnalyzer:
    """
    Service for analyzing function call patterns and dependencies across the repository.

    Builds call graphs to help coding agents understand:
    - Function dependencies and call hierarchies
    - Potential entry points and critical paths
    - Circular dependencies and complexity hotspots
    - Impact analysis for code changes
    """

    def __init__(self):
        self.call_nodes: Dict[str, CallNode] = {}
        self.call_relationships: List[Relationship] = []
        self.entity_map: Dict[str, CodeEntity] = {}

    def build_repository_call_graph(
        self,
        entities: List[CodeEntity],
        relationships: List[Relationship]
    ) -> Dict[str, Any]:
        """
        Build comprehensive call graph for the entire repository.

        Args:
            entities: All code entities in the repository
            relationships: All relationships between entities

        Returns:
            Dictionary containing call graph analysis results
        """
        try:
            # Initialize entity mapping
            self.entity_map = {entity.id: entity for entity in entities}
            self.call_nodes = {}
            self.call_relationships = []

            # Build call nodes for functions
            self._build_call_nodes(entities)

            # Process call relationships
            self._process_call_relationships(relationships)

            # Extract additional call information from entity content
            self._extract_call_patterns_from_content()

            # Analyze call graph structure
            metrics = self._analyze_call_graph_metrics()
            paths = self._analyze_critical_paths()
            cycles = self._detect_circular_dependencies()
            impact_analysis = self._perform_impact_analysis()

            return {
                "call_graph_metrics": metrics,
                "critical_paths": paths,
                "circular_dependencies": cycles,
                "impact_analysis": impact_analysis,
                "call_nodes": {node_id: self._call_node_to_dict(node)
                              for node_id, node in self.call_nodes.items()},
                "analysis_summary": {
                    "total_functions": len(self.call_nodes),
                    "total_call_relationships": len(self.call_relationships),
                    "complexity_score": self._calculate_overall_complexity()
                }
            }

        except Exception as e:
            logger.error(f"Call graph analysis failed: {e}")
            return {}

    def _build_call_nodes(self, entities: List[CodeEntity]) -> None:
        """Build initial call nodes from function entities."""
        for entity in entities:
            if entity.entity_type == EntityType.FUNCTION:
                call_node = CallNode(
                    entity_id=entity.id,
                    entity_name=entity.name,
                    file_path=entity.file_path,
                    calls_to=set(),
                    called_by=set(),
                    call_count=0,
                    is_recursive=False,
                    is_entry_point=False,
                    depth_level=0
                )
                self.call_nodes[entity.id] = call_node

    def _process_call_relationships(self, relationships: List[Relationship]) -> None:
        """Process existing CALLS relationships."""
        for relationship in relationships:
            if relationship.relationship_type == RelationshipType.CALLS:
                source_id = relationship.source_entity_id
                target_id = relationship.target_entity_id

                # Update call nodes
                if source_id in self.call_nodes:
                    self.call_nodes[source_id].calls_to.add(target_id)
                    self.call_nodes[source_id].call_count += 1

                if target_id in self.call_nodes:
                    self.call_nodes[target_id].called_by.add(source_id)

                # Check for recursion
                if source_id == target_id and source_id in self.call_nodes:
                    self.call_nodes[source_id].is_recursive = True

                self.call_relationships.append(relationship)

    def _extract_call_patterns_from_content(self) -> None:
        """Extract additional call patterns from entity content."""
        for entity_id, entity in self.entity_map.items():
            if entity.entity_type != EntityType.FUNCTION or not entity.content:
                continue

            try:
                # Parse entity content to find function calls
                parsed_ast = ast.parse(entity.content)
                call_extractor = CallExtractor(entity_id, self.entity_map)
                additional_calls = call_extractor.extract_calls(parsed_ast)

                # Update call nodes with additional information
                if entity_id in self.call_nodes:
                    for called_function in additional_calls:
                        # Try to match with existing entities
                        matched_entity = self._find_matching_entity(called_function)
                        if matched_entity:
                            self.call_nodes[entity_id].calls_to.add(matched_entity.id)
                            if matched_entity.id in self.call_nodes:
                                self.call_nodes[matched_entity.id].called_by.add(entity_id)

            except Exception as e:
                logger.debug(f"Failed to extract calls from {entity.name}: {e}")

    def _find_matching_entity(self, function_name: str) -> Optional[CodeEntity]:
        """Find entity that matches a function call by name."""
        for entity in self.entity_map.values():
            if entity.entity_type == EntityType.FUNCTION:
                # Simple name matching
                if entity.name == function_name:
                    return entity
                # Qualified name matching
                if entity.qualified_name and entity.qualified_name.endswith(function_name):
                    return entity
        return None

    def _analyze_call_graph_metrics(self) -> CallGraphMetrics:
        """Analyze overall call graph structure and metrics."""
        # Identify entry points (functions with no callers)
        entry_points = [
            node.entity_id for node in self.call_nodes.values()
            if not node.called_by and node.calls_to
        ]

        # Calculate depth levels from entry points
        self._calculate_depth_levels(entry_points)

        # Find most called functions
        call_counts = [(node.entity_id, len(node.called_by))
                      for node in self.call_nodes.values()]
        most_called = sorted(call_counts, key=lambda x: x[1], reverse=True)[:10]

        # Identify complexity hotspots (functions with many calls)
        complexity_hotspots = [
            node.entity_id for node in self.call_nodes.values()
            if node.call_count > 5 or len(node.called_by) > 5
        ]

        # Calculate maximum depth
        max_depth = max((node.depth_level for node in self.call_nodes.values()), default=0)

        return CallGraphMetrics(
            total_functions=len(self.call_nodes),
            total_calls=len(self.call_relationships),
            max_depth=max_depth,
            cyclic_dependencies=[],  # Will be populated by cycle detection
            entry_points=entry_points,
            most_called_functions=most_called,
            complexity_hotspots=complexity_hotspots
        )

    def _calculate_depth_levels(self, entry_points: List[str]) -> None:
        """Calculate depth levels for all functions from entry points."""
        visited = set()
        queue = deque([(ep, 0) for ep in entry_points])

        while queue:
            current_id, depth = queue.popleft()

            if current_id in visited or current_id not in self.call_nodes:
                continue

            visited.add(current_id)
            self.call_nodes[current_id].depth_level = depth
            self.call_nodes[current_id].is_entry_point = (depth == 0)

            # Add called functions to queue
            for called_id in self.call_nodes[current_id].calls_to:
                if called_id not in visited:
                    queue.append((called_id, depth + 1))

    def _analyze_critical_paths(self) -> List[CallPath]:
        """Identify critical execution paths through the call graph."""
        critical_paths = []

        # Find paths from entry points to deeply nested functions
        entry_points = [node.entity_id for node in self.call_nodes.values()
                       if node.is_entry_point]

        for entry_point in entry_points:
            paths = self._find_longest_paths_from(entry_point, max_depth=10)
            for path in paths:
                call_path = CallPath(
                    path=path,
                    total_calls=len(path) - 1,
                    path_complexity=self._calculate_path_complexity(path),
                    has_cycles=self._path_has_cycles(path)
                )
                critical_paths.append(call_path)

        # Sort by complexity and return top paths
        critical_paths.sort(key=lambda p: p.path_complexity, reverse=True)
        return critical_paths[:20]

    def _find_longest_paths_from(self, start_id: str, max_depth: int = 10) -> List[List[str]]:
        """Find longest paths from a starting function."""
        paths = []

        def dfs(current_id: str, current_path: List[str], depth: int) -> None:
            if depth >= max_depth or current_id in current_path:  # Avoid infinite recursion
                return

            current_path.append(current_id)

            if current_id in self.call_nodes:
                calls_to = self.call_nodes[current_id].calls_to
                if not calls_to:  # Leaf node
                    paths.append(current_path.copy())
                else:
                    for called_id in calls_to:
                        dfs(called_id, current_path.copy(), depth + 1)

        dfs(start_id, [], 0)
        return paths

    def _calculate_path_complexity(self, path: List[str]) -> float:
        """Calculate complexity score for a call path."""
        complexity = 0.0

        for entity_id in path:
            if entity_id in self.entity_map:
                entity = self.entity_map[entity_id]
                # Add complexity based on entity characteristics
                if entity.complexity_metrics:
                    complexity += entity.complexity_metrics.get("cyclomatic_complexity", 1)
                else:
                    complexity += 1.0

        # Bonus for path length
        complexity += len(path) * 0.5

        return complexity

    def _path_has_cycles(self, path: List[str]) -> bool:
        """Check if a path contains cycles."""
        return len(path) != len(set(path))

    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the call graph."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs_cycle_detection(node_id: str, path: List[str]) -> None:
            if node_id in rec_stack:
                # Found a cycle
                cycle_start = path.index(node_id)
                cycle = path[cycle_start:]
                if len(cycle) > 1:  # Ignore self-loops for now
                    cycles.append(cycle)
                return

            if node_id in visited or node_id not in self.call_nodes:
                return

            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            for called_id in self.call_nodes[node_id].calls_to:
                dfs_cycle_detection(called_id, path.copy())

            rec_stack.remove(node_id)

        for node_id in self.call_nodes:
            if node_id not in visited:
                dfs_cycle_detection(node_id, [])

        return cycles

    def _perform_impact_analysis(self) -> Dict[str, Any]:
        """Analyze potential impact of changes to functions."""
        impact_scores = {}

        for node_id, node in self.call_nodes.items():
            # Calculate impact based on number of dependent functions
            direct_dependents = len(node.called_by)

            # Calculate transitive dependents (functions that depend indirectly)
            transitive_dependents = self._calculate_transitive_dependents(node_id)

            impact_scores[node_id] = {
                "direct_dependents": direct_dependents,
                "transitive_dependents": transitive_dependents,
                "total_impact_score": direct_dependents + (transitive_dependents * 0.5),
                "is_critical": direct_dependents > 3 or transitive_dependents > 10
            }

        # Sort by impact score
        sorted_impacts = sorted(
            impact_scores.items(),
            key=lambda x: x[1]["total_impact_score"],
            reverse=True
        )

        return {
            "function_impacts": dict(sorted_impacts[:20]),  # Top 20 most impactful
            "critical_functions": [
                fid for fid, info in impact_scores.items()
                if info["is_critical"]
            ]
        }

    def _calculate_transitive_dependents(self, node_id: str) -> int:
        """Calculate number of functions that transitively depend on this function."""
        dependents = set()
        visited = set()

        def dfs_dependents(current_id: str) -> None:
            if current_id in visited or current_id not in self.call_nodes:
                return

            visited.add(current_id)
            dependents.add(current_id)

            for dependent_id in self.call_nodes[current_id].called_by:
                dfs_dependents(dependent_id)

        # Start from direct dependents
        for dependent_id in self.call_nodes[node_id].called_by:
            dfs_dependents(dependent_id)

        return len(dependents) - 1  # Exclude the starting function

    def _calculate_overall_complexity(self) -> float:
        """Calculate overall call graph complexity score."""
        if not self.call_nodes:
            return 0.0

        # Factors contributing to complexity
        total_functions = len(self.call_nodes)
        total_calls = len(self.call_relationships)
        avg_calls_per_function = total_calls / total_functions if total_functions > 0 else 0

        # Complexity from cyclic dependencies
        cycles = self._detect_circular_dependencies()
        cycle_penalty = len(cycles) * 2

        # Complexity from deep call chains
        max_depth = max((node.depth_level for node in self.call_nodes.values()), default=0)
        depth_penalty = max_depth * 1.5

        return avg_calls_per_function + cycle_penalty + depth_penalty

    def _call_node_to_dict(self, node: CallNode) -> Dict[str, Any]:
        """Convert CallNode to dictionary for serialization."""
        return {
            "entity_name": node.entity_name,
            "file_path": node.file_path,
            "calls_to_count": len(node.calls_to),
            "called_by_count": len(node.called_by),
            "call_count": node.call_count,
            "is_recursive": node.is_recursive,
            "is_entry_point": node.is_entry_point,
            "depth_level": node.depth_level
        }


class CallExtractor(ast.NodeVisitor):
    """AST visitor for extracting function calls from code."""

    def __init__(self, source_entity_id: str, entity_map: Dict[str, CodeEntity]):
        self.source_entity_id = source_entity_id
        self.entity_map = entity_map
        self.function_calls = []

    def extract_calls(self, node: ast.AST) -> List[str]:
        """Extract function calls from AST."""
        self.function_calls = []
        self.visit(node)
        return self.function_calls

    def visit_Call(self, node):
        """Handle function calls."""
        # Extract function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        else:
            func_name = None

        if func_name:
            self.function_calls.append(func_name)

        self.generic_visit(node)