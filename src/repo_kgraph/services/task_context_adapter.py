"""
Task-specific context adaptation service for coding agents.

Adapts and tailors context delivery based on the specific type of coding task
being performed, following patterns used by modern coding agents like Cursor.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from repo_kgraph.models.code_entity import CodeEntity, EntityType
from repo_kgraph.models.context_result import ContextResult, RetrievalReason

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of coding tasks that require different context strategies."""
    DEBUG = "debug"
    REFACTOR = "refactor"
    IMPLEMENT_FEATURE = "implement_feature"
    OPTIMIZE = "optimize"
    TEST_WRITE = "test_write"
    DOCUMENTATION = "documentation"
    API_INTEGRATION = "api_integration"
    BUG_FIX = "bug_fix"
    CODE_REVIEW = "code_review"
    MIGRATE = "migrate"


@dataclass
class TaskContext:
    """Context configuration for a specific task type."""
    task_type: TaskType
    priority_entity_types: List[EntityType]
    priority_relationships: List[str]
    context_depth: int
    max_results: int
    relevance_threshold: float
    include_usage_patterns: bool
    include_side_effects: bool
    include_dependencies: bool
    include_call_graph: bool
    context_window_lines: int


@dataclass
class AdaptedContext:
    """Context results adapted for a specific task."""
    primary_context: List[ContextResult]
    supporting_context: List[ContextResult]
    task_specific_metadata: Dict[str, Any]
    context_summary: str
    recommended_actions: List[str]


class TaskContextAdapter:
    """
    Service for adapting context delivery based on coding task type.

    Analyzes task descriptions to determine task type and provides
    contextually relevant information optimized for that specific task.
    """

    def __init__(self):
        self.task_configurations = self._initialize_task_configurations()
        self.task_patterns = self._initialize_task_patterns()

    def _initialize_task_configurations(self) -> Dict[TaskType, TaskContext]:
        """Initialize context configurations for different task types."""
        return {
            TaskType.DEBUG: TaskContext(
                task_type=TaskType.DEBUG,
                priority_entity_types=[EntityType.FUNCTION, EntityType.CLASS],
                priority_relationships=["calls", "uses", "depends_on"],
                context_depth=3,
                max_results=15,
                relevance_threshold=0.4,
                include_usage_patterns=True,
                include_side_effects=True,
                include_dependencies=True,
                include_call_graph=True,
                context_window_lines=50
            ),
            TaskType.REFACTOR: TaskContext(
                task_type=TaskType.REFACTOR,
                priority_entity_types=[EntityType.FUNCTION, EntityType.CLASS, EntityType.MODULE],
                priority_relationships=["contains", "calls", "inherits"],
                context_depth=2,
                max_results=20,
                relevance_threshold=0.3,
                include_usage_patterns=True,
                include_side_effects=False,
                include_dependencies=True,
                include_call_graph=True,
                context_window_lines=100
            ),
            TaskType.IMPLEMENT_FEATURE: TaskContext(
                task_type=TaskType.IMPLEMENT_FEATURE,
                priority_entity_types=[EntityType.CLASS, EntityType.FUNCTION, EntityType.MODULE],
                priority_relationships=["contains", "inherits", "imports"],
                context_depth=2,
                max_results=25,
                relevance_threshold=0.25,
                include_usage_patterns=True,
                include_side_effects=False,
                include_dependencies=True,
                include_call_graph=False,
                context_window_lines=80
            ),
            TaskType.OPTIMIZE: TaskContext(
                task_type=TaskType.OPTIMIZE,
                priority_entity_types=[EntityType.FUNCTION, EntityType.CLASS],
                priority_relationships=["calls", "uses"],
                context_depth=3,
                max_results=12,
                relevance_threshold=0.5,
                include_usage_patterns=True,
                include_side_effects=True,
                include_dependencies=True,
                include_call_graph=True,
                context_window_lines=60
            ),
            TaskType.TEST_WRITE: TaskContext(
                task_type=TaskType.TEST_WRITE,
                priority_entity_types=[EntityType.FUNCTION, EntityType.CLASS],
                priority_relationships=["calls", "uses", "depends_on"],
                context_depth=2,
                max_results=10,
                relevance_threshold=0.6,
                include_usage_patterns=True,
                include_side_effects=True,
                include_dependencies=True,
                include_call_graph=False,
                context_window_lines=40
            ),
            TaskType.BUG_FIX: TaskContext(
                task_type=TaskType.BUG_FIX,
                priority_entity_types=[EntityType.FUNCTION, EntityType.CLASS],
                priority_relationships=["calls", "uses", "depends_on"],
                context_depth=4,
                max_results=18,
                relevance_threshold=0.3,
                include_usage_patterns=True,
                include_side_effects=True,
                include_dependencies=True,
                include_call_graph=True,
                context_window_lines=70
            ),
            TaskType.API_INTEGRATION: TaskContext(
                task_type=TaskType.API_INTEGRATION,
                priority_entity_types=[EntityType.FUNCTION, EntityType.CLASS, EntityType.MODULE],
                priority_relationships=["imports", "calls", "uses"],
                context_depth=2,
                max_results=15,
                relevance_threshold=0.4,
                include_usage_patterns=True,
                include_side_effects=True,
                include_dependencies=True,
                include_call_graph=False,
                context_window_lines=60
            ),
            TaskType.CODE_REVIEW: TaskContext(
                task_type=TaskType.CODE_REVIEW,
                priority_entity_types=[EntityType.FUNCTION, EntityType.CLASS],
                priority_relationships=["contains", "calls", "inherits"],
                context_depth=2,
                max_results=30,
                relevance_threshold=0.2,
                include_usage_patterns=True,
                include_side_effects=True,
                include_dependencies=True,
                include_call_graph=True,
                context_window_lines=120
            )
        }

    def _initialize_task_patterns(self) -> Dict[TaskType, List[str]]:
        """Initialize keyword patterns for task type detection."""
        return {
            TaskType.DEBUG: [
                "debug", "error", "exception", "bug", "issue", "problem", "crash",
                "not working", "failing", "broken", "trace", "stack trace"
            ],
            TaskType.REFACTOR: [
                "refactor", "restructure", "reorganize", "clean up", "improve structure",
                "extract", "rename", "move", "simplify", "redesign"
            ],
            TaskType.IMPLEMENT_FEATURE: [
                "implement", "add", "create", "build", "new feature", "functionality",
                "capability", "enhancement", "feature request"
            ],
            TaskType.OPTIMIZE: [
                "optimize", "performance", "speed up", "efficiency", "faster",
                "memory", "cpu", "bottleneck", "profiling", "benchmark"
            ],
            TaskType.TEST_WRITE: [
                "test", "testing", "unit test", "integration test", "coverage",
                "test case", "assertion", "mock", "spec"
            ],
            TaskType.BUG_FIX: [
                "fix", "resolve", "solve", "correct", "patch", "repair",
                "address issue", "handle error"
            ],
            TaskType.API_INTEGRATION: [
                "api", "integrate", "service", "endpoint", "request", "response",
                "http", "rest", "graphql", "webhook"
            ],
            TaskType.CODE_REVIEW: [
                "review", "check", "examine", "audit", "quality", "standards",
                "best practices", "security", "maintainability"
            ]
        }

    def detect_task_type(self, task_description: str) -> TaskType:
        """
        Detect the type of coding task from the description.

        Args:
            task_description: Natural language description of the task

        Returns:
            Detected task type
        """
        task_description_lower = task_description.lower()

        # Score each task type based on keyword matches
        task_scores = {}
        for task_type, patterns in self.task_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in task_description_lower:
                    score += 1
                    # Bonus for exact matches at word boundaries
                    if f" {pattern} " in f" {task_description_lower} ":
                        score += 1

            task_scores[task_type] = score

        # Return the task type with the highest score
        best_task_type = max(task_scores, key=task_scores.get)

        # Default to IMPLEMENT_FEATURE if no clear pattern is detected
        if task_scores[best_task_type] == 0:
            return TaskType.IMPLEMENT_FEATURE

        logger.debug(f"Detected task type: {best_task_type} (score: {task_scores[best_task_type]})")
        return best_task_type

    def adapt_context_for_task(
        self,
        task_description: str,
        context_results: List[ContextResult],
        task_type: Optional[TaskType] = None
    ) -> AdaptedContext:
        """
        Adapt context results for a specific task type.

        Args:
            task_description: Description of the coding task
            context_results: Raw context results from retrieval
            task_type: Optional explicit task type (will be detected if not provided)

        Returns:
            Adapted context optimized for the task
        """
        # Detect task type if not provided
        if task_type is None:
            task_type = self.detect_task_type(task_description)

        task_config = self.task_configurations.get(task_type, self.task_configurations[TaskType.IMPLEMENT_FEATURE])

        # Filter and prioritize context based on task configuration
        filtered_results = self._filter_context_for_task(context_results, task_config)

        # Separate primary and supporting context
        primary_context = filtered_results[:task_config.max_results // 2]
        supporting_context = filtered_results[task_config.max_results // 2:task_config.max_results]

        # Generate task-specific metadata
        task_metadata = self._generate_task_metadata(task_description, filtered_results, task_config)

        # Create context summary
        context_summary = self._create_context_summary(primary_context, task_type)

        # Generate recommended actions
        recommended_actions = self._generate_recommended_actions(task_type, primary_context)

        return AdaptedContext(
            primary_context=primary_context,
            supporting_context=supporting_context,
            task_specific_metadata=task_metadata,
            context_summary=context_summary,
            recommended_actions=recommended_actions
        )

    def _filter_context_for_task(
        self,
        context_results: List[ContextResult],
        task_config: TaskContext
    ) -> List[ContextResult]:
        """Filter and prioritize context based on task configuration."""
        filtered_results = []

        for result in context_results:
            # Check relevance threshold
            if result.relevance_score < task_config.relevance_threshold:
                continue

            # Priority boost for preferred entity types
            priority_boost = 0.0
            if result.entity_type in task_config.priority_entity_types:
                priority_boost += 0.2

            # Adjust relevance score
            adjusted_score = result.relevance_score + priority_boost

            # Create new result with adjusted score
            adjusted_result = ContextResult(
                query_id=result.query_id,
                entity_id=result.entity_id,
                rank_position=result.rank_position,
                entity_type=result.entity_type,
                entity_name=result.entity_name,
                file_path=result.file_path,
                start_line=result.start_line,
                end_line=result.end_line,
                relevance_score=adjusted_score,
                retrieval_reason=result.retrieval_reason,
                context_snippet=result.context_snippet,
                metadata=result.metadata
            )

            filtered_results.append(adjusted_result)

        # Sort by adjusted relevance score
        filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)

        return filtered_results

    def _generate_task_metadata(
        self,
        task_description: str,
        context_results: List[ContextResult],
        task_config: TaskContext
    ) -> Dict[str, Any]:
        """Generate task-specific metadata."""
        metadata = {
            "task_type": task_config.task_type.value,
            "context_strategy": {
                "max_results": task_config.max_results,
                "context_depth": task_config.context_depth,
                "relevance_threshold": task_config.relevance_threshold
            },
            "entity_distribution": self._analyze_entity_distribution(context_results),
            "complexity_indicators": self._extract_complexity_indicators(context_results),
            "file_coverage": list(set(result.file_path for result in context_results))
        }

        # Add task-specific analysis
        if task_config.include_side_effects:
            metadata["side_effects_analysis"] = self._analyze_side_effects(context_results)

        if task_config.include_dependencies:
            metadata["dependency_analysis"] = self._analyze_dependencies(context_results)

        return metadata

    def _analyze_entity_distribution(self, context_results: List[ContextResult]) -> Dict[str, int]:
        """Analyze distribution of entity types in context."""
        distribution = {}
        for result in context_results:
            entity_type = result.entity_type.value if hasattr(result.entity_type, 'value') else str(result.entity_type)
            distribution[entity_type] = distribution.get(entity_type, 0) + 1
        return distribution

    def _extract_complexity_indicators(self, context_results: List[ContextResult]) -> List[str]:
        """Extract complexity indicators from context."""
        indicators = []
        for result in context_results:
            if result.metadata:
                complexity_metrics = result.metadata.get("complexity_metrics", {})
                if complexity_metrics.get("cyclomatic_complexity", 0) > 10:
                    indicators.append(f"High complexity in {result.entity_name}")

                control_flow = result.metadata.get("control_flow_info", {})
                if "deeply_nested_conditions" in control_flow.get("patterns", []):
                    indicators.append(f"Deep nesting in {result.entity_name}")

        return indicators

    def _analyze_side_effects(self, context_results: List[ContextResult]) -> Dict[str, List[str]]:
        """Analyze side effects across context results."""
        side_effects = {"io_operations": [], "state_modifications": [], "external_calls": []}

        for result in context_results:
            if result.metadata:
                entity_side_effects = result.metadata.get("semantic_analysis", {}).get("side_effects", [])
                for effect in entity_side_effects:
                    if "io" in effect.lower():
                        side_effects["io_operations"].append(f"{result.entity_name}: {effect}")
                    elif "state" in effect.lower():
                        side_effects["state_modifications"].append(f"{result.entity_name}: {effect}")
                    else:
                        side_effects["external_calls"].append(f"{result.entity_name}: {effect}")

        return side_effects

    def _analyze_dependencies(self, context_results: List[ContextResult]) -> Dict[str, Any]:
        """Analyze dependencies across context results."""
        all_dependencies = set()
        dependency_chains = []

        for result in context_results:
            if result.metadata:
                dependencies = result.metadata.get("dependencies", [])
                all_dependencies.update(dependencies)

                if len(dependencies) > 5:
                    dependency_chains.append({
                        "entity": result.entity_name,
                        "dependency_count": len(dependencies),
                        "top_dependencies": dependencies[:5]
                    })

        return {
            "total_unique_dependencies": len(all_dependencies),
            "high_dependency_entities": dependency_chains,
            "common_dependencies": list(all_dependencies)[:10]  # Top 10 most common
        }

    def _create_context_summary(self, primary_context: List[ContextResult], task_type: TaskType) -> str:
        """Create a human-readable context summary."""
        if not primary_context:
            return "No relevant context found."

        entity_types = [result.entity_type.value if hasattr(result.entity_type, 'value') else str(result.entity_type)
                       for result in primary_context]
        entity_counts = {}
        for et in entity_types:
            entity_counts[et] = entity_counts.get(et, 0) + 1

        files = set(result.file_path for result in primary_context)

        summary_parts = [
            f"Found {len(primary_context)} relevant entities for {task_type.value} task",
            f"Spanning {len(files)} files"
        ]

        if entity_counts:
            type_summary = ", ".join([f"{count} {type_name}{'s' if count > 1 else ''}"
                                    for type_name, count in entity_counts.items()])
            summary_parts.append(f"Including: {type_summary}")

        return ". ".join(summary_parts) + "."

    def _generate_recommended_actions(
        self,
        task_type: TaskType,
        primary_context: List[ContextResult]
    ) -> List[str]:
        """Generate recommended actions based on task type and context."""
        actions = []

        if task_type == TaskType.DEBUG:
            actions.extend([
                "Check error handling patterns in related functions",
                "Trace execution flow through call graph",
                "Examine side effects and state modifications",
                "Look for similar error patterns in the codebase"
            ])
        elif task_type == TaskType.REFACTOR:
            actions.extend([
                "Identify functions that can be extracted or combined",
                "Check for code duplication across related entities",
                "Analyze dependency relationships for reorganization opportunities",
                "Consider interface consistency across similar components"
            ])
        elif task_type == TaskType.IMPLEMENT_FEATURE:
            actions.extend([
                "Study existing patterns in similar functionality",
                "Identify interfaces and contracts to follow",
                "Check for reusable components in the codebase",
                "Plan integration points with existing systems"
            ])
        elif task_type == TaskType.BUG_FIX:
            actions.extend([
                "Examine control flow and edge cases",
                "Check input validation and error handling",
                "Trace data flow through the problematic code",
                "Look for related fixes in version history"
            ])

        # Add context-specific actions
        if primary_context:
            high_complexity_entities = [
                result for result in primary_context
                if result.metadata and
                result.metadata.get("complexity_metrics", {}).get("cyclomatic_complexity", 0) > 10
            ]

            if high_complexity_entities:
                actions.append(f"Pay special attention to complex functions: {', '.join([e.entity_name for e in high_complexity_entities[:3]])}")

        return actions[:5]  # Limit to top 5 recommendations