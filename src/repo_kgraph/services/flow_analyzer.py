"""
Advanced control and data flow analysis for code entities.

Provides comprehensive analysis of execution paths, variable dependencies,
and data transformations to enhance coding agent context understanding.
"""

import ast
import logging
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque

from repo_kgraph.models.code_entity import CodeEntity, EntityType

logger = logging.getLogger(__name__)


@dataclass
class ControlFlowNode:
    """A node in the control flow graph."""
    node_id: str
    node_type: str  # 'statement', 'condition', 'loop', 'exception'
    ast_node: ast.AST
    line_number: int
    predecessors: Set[str]
    successors: Set[str]
    conditions: List[str]  # Conditions that lead to this node


@dataclass
class DataFlowInfo:
    """Information about data flow for a variable."""
    variable_name: str
    definitions: List[int]  # Line numbers where variable is defined
    uses: List[int]  # Line numbers where variable is used
    dependencies: Set[str]  # Variables this depends on
    influences: Set[str]  # Variables this influences
    flow_type: str  # 'local', 'parameter', 'global', 'closure'


@dataclass
class CallInfo:
    """Information about function calls."""
    function_name: str
    line_number: int
    arguments: List[str]
    is_method: bool
    receiver: Optional[str]  # For method calls
    side_effects: List[str]


class FlowAnalyzer:
    """
    Advanced flow analysis for understanding code execution and data patterns.

    Provides control flow graphs, data dependency analysis, and call patterns
    to help coding agents understand code behavior and relationships.
    """

    def __init__(self):
        self.control_flow_nodes = {}
        self.data_flow_info = {}
        self.call_info = []

    def analyze_entity_flows(self, entity: CodeEntity) -> Dict[str, Any]:
        """
        Perform comprehensive flow analysis on a code entity.

        Args:
            entity: Code entity to analyze

        Returns:
            Dictionary containing flow analysis results
        """
        if not entity.content or entity.entity_type not in [EntityType.FUNCTION, EntityType.CLASS]:
            return {}

        try:
            # Parse the entity content
            parsed_ast = ast.parse(entity.content)

            # Perform different types of analysis
            control_flow = self._analyze_control_flow(parsed_ast, entity)
            data_flow = self._analyze_data_flow(parsed_ast, entity)
            call_patterns = self._analyze_call_patterns(parsed_ast, entity)
            complexity_metrics = self._calculate_complexity_metrics(parsed_ast, entity)

            return {
                "control_flow": control_flow,
                "data_flow": data_flow,
                "call_patterns": call_patterns,
                "complexity_metrics": complexity_metrics,
                "analysis_metadata": {
                    "total_nodes": len(self.control_flow_nodes),
                    "total_variables": len(self.data_flow_info),
                    "total_calls": len(self.call_info)
                }
            }

        except Exception as e:
            logger.warning(f"Flow analysis failed for {entity.name}: {e}")
            return {}

    def _analyze_control_flow(self, parsed_ast: ast.AST, entity: CodeEntity) -> Dict[str, Any]:
        """Analyze control flow patterns and execution paths."""
        self.control_flow_nodes = {}

        # Build control flow graph
        self._build_control_flow_graph(parsed_ast)

        # Analyze patterns
        patterns = self._identify_control_patterns()
        execution_paths = self._analyze_execution_paths()
        complexity_indicators = self._identify_complexity_indicators()

        return {
            "patterns": patterns,
            "execution_paths": execution_paths,
            "complexity_indicators": complexity_indicators,
            "total_branches": len([n for n in self.control_flow_nodes.values()
                                  if n.node_type == 'condition']),
            "total_loops": len([n for n in self.control_flow_nodes.values()
                               if n.node_type == 'loop']),
            "exception_handlers": len([n for n in self.control_flow_nodes.values()
                                      if n.node_type == 'exception'])
        }

    def _build_control_flow_graph(self, node: ast.AST, parent_id: str = None) -> None:
        """Build control flow graph from AST."""
        node_id = f"{type(node).__name__}_{id(node)}"

        # Determine node type and characteristics
        if isinstance(node, (ast.If, ast.IfExp)):
            node_type = 'condition'
            conditions = [ast.unparse(node.test)] if hasattr(ast, 'unparse') else ['condition']
        elif isinstance(node, (ast.For, ast.While)):
            node_type = 'loop'
            conditions = []
        elif isinstance(node, (ast.Try, ast.ExceptHandler)):
            node_type = 'exception'
            conditions = []
        else:
            node_type = 'statement'
            conditions = []

        # Create control flow node
        cf_node = ControlFlowNode(
            node_id=node_id,
            node_type=node_type,
            ast_node=node,
            line_number=getattr(node, 'lineno', 0),
            predecessors=set([parent_id] if parent_id else []),
            successors=set(),
            conditions=conditions
        )

        self.control_flow_nodes[node_id] = cf_node

        # Update parent's successors
        if parent_id and parent_id in self.control_flow_nodes:
            self.control_flow_nodes[parent_id].successors.add(node_id)

        # Recursively process children
        for child in ast.iter_child_nodes(node):
            self._build_control_flow_graph(child, node_id)

    def _identify_control_patterns(self) -> List[str]:
        """Identify common control flow patterns."""
        patterns = []

        # Check for nested conditions
        nested_conditions = 0
        for node in self.control_flow_nodes.values():
            if node.node_type == 'condition':
                # Count nested conditions
                condition_children = [s for s in node.successors
                                     if self.control_flow_nodes.get(s, {}).node_type == 'condition']
                if condition_children:
                    nested_conditions += 1

        if nested_conditions > 2:
            patterns.append("deeply_nested_conditions")

        # Check for early returns
        return_statements = [n for n in self.control_flow_nodes.values()
                           if isinstance(n.ast_node, ast.Return)]
        if len(return_statements) > 1:
            patterns.append("multiple_exit_points")

        # Check for exception handling patterns
        try_blocks = [n for n in self.control_flow_nodes.values()
                     if isinstance(n.ast_node, ast.Try)]
        if try_blocks:
            patterns.append("exception_handling")

        # Check for loop patterns
        loops = [n for n in self.control_flow_nodes.values() if n.node_type == 'loop']
        if len(loops) > 1:
            patterns.append("multiple_loops")

        return patterns

    def _analyze_execution_paths(self) -> Dict[str, Any]:
        """Analyze possible execution paths through the code."""
        paths = []

        # Find entry points (nodes with no predecessors)
        entry_points = [n for n in self.control_flow_nodes.values()
                       if not n.predecessors]

        # Find exit points (nodes with no successors)
        exit_points = [n for n in self.control_flow_nodes.values()
                      if not n.successors]

        # Calculate approximate path complexity
        total_paths = 1
        for node in self.control_flow_nodes.values():
            if node.node_type == 'condition':
                total_paths *= 2  # Each condition doubles paths
            elif node.node_type == 'loop':
                total_paths *= 3  # Approximate loop iteration impact

        return {
            "entry_points": len(entry_points),
            "exit_points": len(exit_points),
            "estimated_paths": min(total_paths, 1000),  # Cap for sanity
            "cyclomatic_complexity": self._calculate_cyclomatic_complexity()
        }

    def _calculate_cyclomatic_complexity(self) -> int:
        """Calculate cyclomatic complexity."""
        edges = sum(len(node.successors) for node in self.control_flow_nodes.values())
        nodes = len(self.control_flow_nodes)
        connected_components = 1  # Assuming single connected component

        # McCabe's formula: M = E - N + 2P
        return max(1, edges - nodes + 2 * connected_components)

    def _identify_complexity_indicators(self) -> List[str]:
        """Identify indicators of code complexity."""
        indicators = []

        # High cyclomatic complexity
        if self._calculate_cyclomatic_complexity() > 10:
            indicators.append("high_cyclomatic_complexity")

        # Deep nesting
        max_depth = self._calculate_nesting_depth()
        if max_depth > 4:
            indicators.append("deep_nesting")

        # Many conditional branches
        conditions = [n for n in self.control_flow_nodes.values()
                     if n.node_type == 'condition']
        if len(conditions) > 5:
            indicators.append("many_conditions")

        return indicators

    def _calculate_nesting_depth(self) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0

        def calculate_depth(node_id: str, current_depth: int) -> int:
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)

            node = self.control_flow_nodes.get(node_id)
            if not node:
                return current_depth

            # Increase depth for nested structures
            depth_increment = 1 if node.node_type in ['condition', 'loop', 'exception'] else 0

            for successor_id in node.successors:
                calculate_depth(successor_id, current_depth + depth_increment)

            return max_depth

        # Start from entry points
        entry_points = [n.node_id for n in self.control_flow_nodes.values()
                       if not n.predecessors]

        for entry_id in entry_points:
            calculate_depth(entry_id, 0)

        return max_depth

    def _analyze_data_flow(self, parsed_ast: ast.AST, entity: CodeEntity) -> Dict[str, Any]:
        """Analyze data flow and variable dependencies."""
        self.data_flow_info = {}

        # Extract variable information
        variable_analyzer = VariableAnalyzer()
        variable_info = variable_analyzer.analyze(parsed_ast)

        # Build data flow graph
        data_dependencies = self._build_data_dependencies(variable_info)

        return {
            "variables": variable_info,
            "dependencies": data_dependencies,
            "data_complexity": len(variable_info),
            "dependency_depth": self._calculate_dependency_depth(data_dependencies)
        }

    def _build_data_dependencies(self, variable_info: Dict[str, DataFlowInfo]) -> Dict[str, List[str]]:
        """Build data dependency graph."""
        dependencies = {}

        for var_name, var_info in variable_info.items():
            dependencies[var_name] = list(var_info.dependencies)

        return dependencies

    def _calculate_dependency_depth(self, dependencies: Dict[str, List[str]]) -> int:
        """Calculate maximum dependency chain depth."""
        def get_depth(var: str, visited: Set[str]) -> int:
            if var in visited:
                return 0  # Avoid cycles

            visited.add(var)
            max_child_depth = 0

            for dep in dependencies.get(var, []):
                max_child_depth = max(max_child_depth, get_depth(dep, visited.copy()))

            return max_child_depth + 1

        max_depth = 0
        for var in dependencies:
            depth = get_depth(var, set())
            max_depth = max(max_depth, depth)

        return max_depth

    def _analyze_call_patterns(self, parsed_ast: ast.AST, entity: CodeEntity) -> Dict[str, Any]:
        """Analyze function call patterns and dependencies."""
        self.call_info = []

        call_analyzer = CallAnalyzer()
        calls = call_analyzer.extract_calls(parsed_ast)

        # Categorize calls
        external_calls = [c for c in calls if self._is_external_call(c.function_name)]
        method_calls = [c for c in calls if c.is_method]
        recursive_calls = [c for c in calls if c.function_name == entity.name]

        return {
            "total_calls": len(calls),
            "external_calls": len(external_calls),
            "method_calls": len(method_calls),
            "recursive_calls": len(recursive_calls),
            "call_details": [self._call_to_dict(c) for c in calls[:10]],  # Limit details
            "side_effect_calls": len([c for c in calls if c.side_effects])
        }

    def _is_external_call(self, function_name: str) -> bool:
        """Determine if a call is to an external function."""
        # Simple heuristic: if it contains a dot, it's likely external
        return '.' in function_name or function_name in ['print', 'len', 'str', 'int', 'list', 'dict']

    def _call_to_dict(self, call: CallInfo) -> Dict[str, Any]:
        """Convert CallInfo to dictionary."""
        return {
            "function": call.function_name,
            "line": call.line_number,
            "args_count": len(call.arguments),
            "is_method": call.is_method,
            "has_side_effects": bool(call.side_effects)
        }

    def _calculate_complexity_metrics(self, parsed_ast: ast.AST, entity: CodeEntity) -> Dict[str, int]:
        """Calculate various complexity metrics."""
        return {
            "lines_of_code": len(entity.content.split('\n')) if entity.content else 0,
            "cyclomatic_complexity": self._calculate_cyclomatic_complexity(),
            "nesting_depth": self._calculate_nesting_depth(),
            "number_of_variables": len(self.data_flow_info),
            "number_of_calls": len(self.call_info)
        }


class VariableAnalyzer(ast.NodeVisitor):
    """AST visitor for analyzing variable usage patterns."""

    def __init__(self):
        self.variables = {}
        self.current_line = 0

    def analyze(self, node: ast.AST) -> Dict[str, DataFlowInfo]:
        """Analyze variable usage in AST."""
        self.variables = {}
        self.visit(node)
        return self.variables

    def visit(self, node):
        """Visit AST node and track line numbers."""
        if hasattr(node, 'lineno'):
            self.current_line = node.lineno
        super().visit(node)

    def visit_Name(self, node):
        """Handle variable names."""
        var_name = node.id

        if var_name not in self.variables:
            self.variables[var_name] = DataFlowInfo(
                variable_name=var_name,
                definitions=[],
                uses=[],
                dependencies=set(),
                influences=set(),
                flow_type='local'
            )

        if isinstance(node.ctx, ast.Store):
            self.variables[var_name].definitions.append(self.current_line)
        elif isinstance(node.ctx, ast.Load):
            self.variables[var_name].uses.append(self.current_line)

        self.generic_visit(node)

    def visit_Assign(self, node):
        """Handle assignments to track dependencies."""
        # Extract target variables
        targets = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                targets.append(target.id)

        # Extract source variables
        source_vars = set()
        for name_node in ast.walk(node.value):
            if isinstance(name_node, ast.Name) and isinstance(name_node.ctx, ast.Load):
                source_vars.add(name_node.id)

        # Update dependencies
        for target in targets:
            if target in self.variables:
                self.variables[target].dependencies.update(source_vars)

        self.generic_visit(node)


class CallAnalyzer(ast.NodeVisitor):
    """AST visitor for analyzing function calls."""

    def __init__(self):
        self.calls = []
        self.current_line = 0

    def extract_calls(self, node: ast.AST) -> List[CallInfo]:
        """Extract function calls from AST."""
        self.calls = []
        self.visit(node)
        return self.calls

    def visit(self, node):
        """Visit AST node and track line numbers."""
        if hasattr(node, 'lineno'):
            self.current_line = node.lineno
        super().visit(node)

    def visit_Call(self, node):
        """Handle function calls."""
        # Extract function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            is_method = False
            receiver = None
        elif isinstance(node.func, ast.Attribute):
            func_name = f"{ast.unparse(node.func.value) if hasattr(ast, 'unparse') else 'obj'}.{node.func.attr}"
            is_method = True
            receiver = ast.unparse(node.func.value) if hasattr(ast, 'unparse') else 'obj'
        else:
            func_name = 'unknown_call'
            is_method = False
            receiver = None

        # Extract arguments
        arguments = []
        for arg in node.args:
            if isinstance(arg, ast.Name):
                arguments.append(arg.id)
            else:
                arguments.append('expression')

        # Detect potential side effects
        side_effects = []
        if any(keyword in func_name.lower() for keyword in ['print', 'write', 'save', 'delete', 'update']):
            side_effects.append('io_operation')
        if 'global' in func_name.lower() or receiver == 'self':
            side_effects.append('state_modification')

        call_info = CallInfo(
            function_name=func_name,
            line_number=self.current_line,
            arguments=arguments,
            is_method=is_method,
            receiver=receiver,
            side_effects=side_effects
        )

        self.calls.append(call_info)
        self.generic_visit(node)