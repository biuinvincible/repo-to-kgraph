#!/usr/bin/env python3
"""
Fix the nested function call relationship detection.
The issue: nested function calls aren't being detected properly because
the call resolution doesn't handle nested scopes correctly.
"""

import asyncio
import tempfile
import time
import os
import sys
import re
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

sys.path.insert(0, str(Path(__file__).parent / "src"))

from implement_fixes import TwoPassParser, EnhancedParseResult, DiagnosticLevel, Diagnostic, EnhancedPathResolver
from repo_kgraph.models.code_entity import CodeEntity, EntityType
from repo_kgraph.models.relationship import Relationship, RelationshipType


class FixedTwoPassParser(TwoPassParser):
    """
    Final fixed parser that properly handles nested function call relationships.
    """

    def __init__(self):
        super().__init__()
        self.entity_stack = []
        self.scope_stack = []  # Track scope hierarchy for proper call resolution

    async def parse_file_enhanced(self, file_path: str, repository_id: str, repository_path: str) -> EnhancedParseResult:
        """Enhanced parsing with proper nested function and async detection."""

        import time
        start_time = time.time()

        # Step 1: Path normalization
        abs_path, rel_path, path_diagnostics = EnhancedPathResolver.normalize_paths(file_path, repository_path)

        result = EnhancedParseResult(
            file_path=rel_path,
            entities=[],
            forward_references=[],
            diagnostics=path_diagnostics,
            parse_time_ms=0,
            language=self._detect_language(file_path)
        )

        if any(d.level == DiagnosticLevel.ERROR for d in path_diagnostics):
            result.parse_time_ms = (time.time() - start_time) * 1000
            return result

        try:
            content = Path(abs_path).read_text(encoding='utf-8')
            result.add_diagnostic(DiagnosticLevel.INFO, f"File read successfully ({len(content)} chars)")

        except Exception as e:
            result.add_diagnostic(DiagnosticLevel.ERROR, f"Failed to read file: {e}", code="READ_ERROR")
            result.parse_time_ms = (time.time() - start_time) * 1000
            return result

        # Enhanced AST-based parsing for Python
        if result.language == 'python':
            await self._parse_python_with_fixed_ast(content, abs_path, rel_path, repository_id, result)
        else:
            await self._pass1_extract_entities(content, abs_path, rel_path, repository_id, result)
            await self._pass1_extract_forward_references(content, abs_path, rel_path, result)

        result.parse_time_ms = (time.time() - start_time) * 1000
        result.add_diagnostic(DiagnosticLevel.INFO, f"Fixed parsing completed in {result.parse_time_ms:.2f}ms")

        return result

    async def _parse_python_with_fixed_ast(self, content: str, abs_path: str, rel_path: str, repository_id: str, result: EnhancedParseResult):
        """Parse Python with fixed nested scope handling."""

        try:
            tree = ast.parse(content)
            lines = content.split('\n')

            # Reset state
            self.entity_stack = []
            self.scope_stack = []

            # Walk AST to extract entities and calls with proper scope tracking
            self._extract_entities_and_calls_with_scope(tree, lines, rel_path, repository_id, result)

            result.add_diagnostic(DiagnosticLevel.INFO, f"Fixed AST parsing: {len(result.entities)} entities, {len(result.forward_references)} refs")

        except SyntaxError as e:
            result.add_diagnostic(DiagnosticLevel.WARNING, f"Syntax error: {e}", e.lineno or 0, code="SYNTAX_ERROR")
            await self._pass1_extract_entities(content, abs_path, rel_path, repository_id, result)
            await self._pass1_extract_forward_references(content, abs_path, rel_path, result)

        except Exception as e:
            result.add_diagnostic(DiagnosticLevel.ERROR, f"AST parsing failed: {e}", code="AST_ERROR")
            await self._pass1_extract_entities(content, abs_path, rel_path, repository_id, result)
            await self._pass1_extract_forward_references(content, abs_path, rel_path, result)

    def _extract_entities_and_calls_with_scope(self, tree: ast.AST, lines: List[str], rel_path: str, repository_id: str, result: EnhancedParseResult):
        """Extract entities and calls with proper scope tracking."""

        class ScopeAwareExtractor(ast.NodeVisitor):
            def __init__(self, parser_instance):
                self.parser = parser_instance
                self.lines = lines
                self.rel_path = rel_path
                self.repository_id = repository_id
                self.result = result

            def visit_FunctionDef(self, node):
                self._process_function(node, is_async=False)

            def visit_AsyncFunctionDef(self, node):
                self._process_function(node, is_async=True)

            def visit_ClassDef(self, node):
                self._process_class(node)

            def _process_function(self, node, is_async=False):
                """Process function with scope tracking."""
                func_name = node.name

                # Build qualified name with proper scope
                scope_path = self._get_current_scope_path()
                if scope_path:
                    qualified_name = f"{scope_path}.{func_name}"
                else:
                    qualified_name = f"{self.rel_path}::{func_name}"

                # Create entity
                entity = self._create_function_entity(node, func_name, qualified_name, is_async)
                self.result.entities.append(entity)

                # Push to scope stack
                scope_info = {
                    'type': 'function',
                    'name': func_name,
                    'qualified_name': qualified_name,
                    'entity': entity,
                    'node': node
                }
                self.parser.scope_stack.append(scope_info)

                # Process function body for calls and nested definitions
                self._process_function_body(node, entity)

                # Pop from scope stack
                self.parser.scope_stack.pop()

            def _process_class(self, node):
                """Process class with scope tracking."""
                class_name = node.name

                scope_path = self._get_current_scope_path()
                if scope_path:
                    qualified_name = f"{scope_path}.{class_name}"
                else:
                    qualified_name = f"{self.rel_path}::{class_name}"

                entity = self._create_class_entity(node, class_name, qualified_name)
                self.result.entities.append(entity)

                # Push to scope stack
                scope_info = {
                    'type': 'class',
                    'name': class_name,
                    'qualified_name': qualified_name,
                    'entity': entity,
                    'node': node
                }
                self.parser.scope_stack.append(scope_info)

                # Process class body
                self.generic_visit(node)

                # Pop from scope stack
                self.parser.scope_stack.pop()

            def _process_function_body(self, func_node, func_entity):
                """Process function body to find calls and nested functions."""

                # Find all calls in this function
                class CallFinder(ast.NodeVisitor):
                    def __init__(self, function_entity, target_function_node):
                        self.function_entity = function_entity
                        self.target_function_node = target_function_node
                        self.calls = []
                        self.depth = 0

                    def visit_Call(self, node):
                        # Only record calls at depth 0 (direct calls in the target function)
                        if self.depth == 0:
                            call_name = self._get_call_name(node)
                            if call_name:
                                self.calls.append({
                                    'source_entity_id': self.function_entity.id,
                                    'called_name': call_name,
                                    'line_number': node.lineno,
                                    'call_type': 'function',
                                    'node': node
                                })
                        self.generic_visit(node)

                    def visit_FunctionDef(self, node):
                        if node != self.target_function_node:
                            # Entering a nested function - increase depth
                            self.depth += 1
                            self.generic_visit(node)
                            self.depth -= 1
                        else:
                            # This is our target function - visit its body
                            self.generic_visit(node)

                    def visit_AsyncFunctionDef(self, node):
                        if node != self.target_function_node:
                            # Entering a nested async function - increase depth
                            self.depth += 1
                            self.generic_visit(node)
                            self.depth -= 1
                        else:
                            # This is our target function - visit its body
                            self.generic_visit(node)

                    def visit_ClassDef(self, node):
                        # Entering a nested class - increase depth
                        self.depth += 1
                        self.generic_visit(node)
                        self.depth -= 1

                    def _get_call_name(self, node: ast.Call) -> Optional[str]:
                        if isinstance(node.func, ast.Name):
                            return node.func.id
                        elif isinstance(node.func, ast.Attribute):
                            return node.func.attr
                        return None

                # Find calls
                call_finder = CallFinder(func_entity, func_node)
                call_finder.visit(func_node)

                # Add calls to forward references
                self.result.forward_references.extend(call_finder.calls)

                # Now visit for nested definitions (this will handle nested functions)
                for child in ast.iter_child_nodes(func_node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        self.visit(child)

            def _get_current_scope_path(self) -> str:
                """Get the current scope path for qualified names."""
                if not self.parser.scope_stack:
                    return f"{self.rel_path}::"

                # Build path from scope stack
                parts = [self.rel_path]
                for scope in self.parser.scope_stack:
                    parts.append(scope['name'])

                return "::".join(parts)

            def _create_function_entity(self, node, func_name, qualified_name, is_async):
                """Create function entity."""
                start_line = node.lineno
                end_line = node.end_lineno or start_line

                func_lines = self.lines[start_line-1:end_line] if end_line <= len(self.lines) else [self.lines[start_line-1]]
                content = '\n'.join(func_lines)

                entity = CodeEntity(
                    repository_id=self.repository_id,
                    entity_type=EntityType.FUNCTION,
                    name=func_name,
                    qualified_name=qualified_name,
                    file_path=self.rel_path,
                    start_line=start_line,
                    end_line=end_line,
                    start_column=node.col_offset,
                    end_column=getattr(node, 'end_col_offset', node.col_offset + len(func_name)),
                    language='python',
                    content=content
                )

                if is_async:
                    self.result.add_diagnostic(DiagnosticLevel.INFO, f"Detected async function: {func_name}", start_line)

                return entity

            def _create_class_entity(self, node, class_name, qualified_name):
                """Create class entity."""
                start_line = node.lineno
                end_line = node.end_lineno or start_line

                class_lines = self.lines[start_line-1:end_line] if end_line <= len(self.lines) else [self.lines[start_line-1]]
                content = '\n'.join(class_lines)

                return CodeEntity(
                    repository_id=self.repository_id,
                    entity_type=EntityType.CLASS,
                    name=class_name,
                    qualified_name=qualified_name,
                    file_path=self.rel_path,
                    start_line=start_line,
                    end_line=end_line,
                    start_column=node.col_offset,
                    end_column=getattr(node, 'end_col_offset', node.col_offset + len(class_name)),
                    language='python',
                    content=content
                )

        # Run the extractor
        extractor = ScopeAwareExtractor(self)
        extractor.visit(tree)

    async def resolve_relationships(self, parse_results: List[EnhancedParseResult]) -> List[Relationship]:
        """Enhanced relationship resolution with proper nested scope handling."""

        relationships = []

        # Build comprehensive entity lookup
        entity_lookup = {}  # name -> entity
        qualified_lookup = {}  # qualified_name -> entity
        all_entities = []

        for result in parse_results:
            for entity in result.entities:
                all_entities.append(entity)

                # Index by simple name
                if entity.name not in entity_lookup:
                    entity_lookup[entity.name] = []
                entity_lookup[entity.name].append(entity)

                # Index by qualified name
                qualified_lookup[entity.qualified_name] = entity

                # Also index by last part of qualified name for nested lookups
                parts = entity.qualified_name.split("::")
                if len(parts) > 1:
                    last_part = parts[-1]
                    if last_part not in entity_lookup:
                        entity_lookup[last_part] = []
                    entity_lookup[last_part].append(entity)

        # Resolve forward references with scope awareness
        for result in parse_results:
            for ref in result.forward_references:
                target_entities = self._resolve_call_with_scope(ref, entity_lookup, qualified_lookup, result)

                for target_entity in target_entities:
                    if target_entity.id != ref['source_entity_id']:  # Avoid self-calls
                        relationship = Relationship(
                            repository_id=target_entity.repository_id,
                            source_entity_id=ref['source_entity_id'],
                            target_entity_id=target_entity.id,
                            relationship_type=RelationshipType.CALLS,
                            strength=1.0,
                            metadata={
                                "call_type": ref.get('call_type', 'unknown'),
                                "line_number": ref.get('line_number', 0),
                                "resolved_name": ref.get('called_name', '')
                            }
                        )
                        relationships.append(relationship)

        # Add diagnostics
        for result in parse_results:
            file_refs = len(result.forward_references)
            file_resolved = sum(1 for rel in relationships
                              if any(ref['source_entity_id'] == rel.source_entity_id
                                   for ref in result.forward_references))

            if file_refs > 0:
                result.add_diagnostic(
                    DiagnosticLevel.INFO,
                    f"Resolved {file_resolved}/{file_refs} references with scope awareness",
                    code="SCOPE_RESOLUTION"
                )

        return relationships

    def _resolve_call_with_scope(self, ref: Dict[str, Any], entity_lookup: Dict[str, List[CodeEntity]],
                                qualified_lookup: Dict[str, CodeEntity], result: EnhancedParseResult) -> List[CodeEntity]:
        """Resolve a call with scope awareness."""

        called_name = ref.get('called_name', '')
        source_entity_id = ref.get('source_entity_id', '')

        # Find the source entity to understand its scope
        source_entity = None
        for entity in result.entities:
            if entity.id == source_entity_id:
                source_entity = entity
                break

        if not source_entity:
            return []

        targets = []

        # Strategy 1: Look for exact matches in current scope
        if source_entity:
            # Get the scope prefix of the source entity
            scope_parts = source_entity.qualified_name.split("::")

            # Try different scope levels
            for i in range(len(scope_parts), 0, -1):
                scope_prefix = "::".join(scope_parts[:i])
                potential_qualified_name = f"{scope_prefix}::{called_name}"

                if potential_qualified_name in qualified_lookup:
                    targets.append(qualified_lookup[potential_qualified_name])
                    break

                # Also try with just the scope part (for nested functions)
                if i > 1:  # Don't do this for the top level
                    parent_scope = "::".join(scope_parts[:i-1])
                    potential_qualified_name = f"{parent_scope}::{called_name}"
                    if potential_qualified_name in qualified_lookup:
                        targets.append(qualified_lookup[potential_qualified_name])
                        break

        # Strategy 2: Look for simple name matches if no scoped match found
        if not targets and called_name in entity_lookup:
            # Filter by same file first, then global
            same_file_entities = [e for e in entity_lookup[called_name] if e.file_path == source_entity.file_path]
            if same_file_entities:
                targets.extend(same_file_entities)
            else:
                targets.extend(entity_lookup[called_name])

        return targets[:1]  # Return at most one target to avoid duplicates


async def test_fixed_nested_calls():
    """Test the fixed nested function call detection."""

    print("🔧 TESTING FIXED NESTED FUNCTION CALLS")
    print("=" * 50)

    parser = FixedTwoPassParser()

    test_code = '''
def outer():
    def inner():
        return helper()
    return inner()

def helper():
    return 1

def complex_nested():
    def level1():
        def level2():
            def level3():
                return deep_helper()
            return level3()
        return level2()
    return level1()

def deep_helper():
    return 42

class TestClass:
    def method(self):
        def nested_in_method():
            return self.other_method()
        return nested_in_method()

    def other_method(self):
        return 100
'''

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            temp_file = f.name

        result = await parser.parse_file_enhanced(temp_file, "test_repo", tempfile.gettempdir())

        if not result.success:
            print("   ❌ Parse failed")
            return False

        # Check entities
        functions = [e for e in result.entities if e.entity_type == EntityType.FUNCTION]
        classes = [e for e in result.entities if e.entity_type == EntityType.CLASS]

        print(f"   📊 Found {len(functions)} functions, {len(classes)} classes")

        # Print all entities with their qualified names
        print("   📋 Entities found:")
        for entity in result.entities:
            print(f"      {entity.entity_type}: {entity.name} ({entity.qualified_name})")

        # Check relationships
        relationships = await parser.resolve_relationships([result])
        print(f"   📊 Found {len(relationships)} relationships")

        # Print all relationships
        entity_names = {e.id: (e.name, e.qualified_name) for e in result.entities}
        print("   🔗 Relationships found:")

        critical_relationships = []
        for rel in relationships:
            source_info = entity_names.get(rel.source_entity_id, ('unknown', 'unknown'))
            target_info = entity_names.get(rel.target_entity_id, ('unknown', 'unknown'))
            source_name = source_info[0]
            target_name = target_info[0]

            print(f"      {source_name} → {target_name}")
            critical_relationships.append((source_name, target_name))

        # Check for specific critical relationships
        expected_critical = [
            ('outer', 'inner'),
            ('inner', 'helper'),
            ('complex_nested', 'level1'),
            ('level1', 'level2'),
            ('level2', 'level3'),
            ('level3', 'deep_helper')
        ]

        found_critical = 0
        missing_critical = []

        for expected in expected_critical:
            if expected in critical_relationships:
                print(f"      ✅ Critical: {expected[0]} → {expected[1]}")
                found_critical += 1
            else:
                print(f"      ❌ Missing: {expected[0]} → {expected[1]}")
                missing_critical.append(expected)

        success_rate = found_critical / len(expected_critical) * 100

        print(f"\n   📊 Critical relationships: {found_critical}/{len(expected_critical)} ({success_rate:.0f}%)")

        if success_rate >= 80:  # 80% or better
            print("   🎉 NESTED FUNCTION CALLS: FIXED!")
            return True
        else:
            print("   ⚠️  Still needs improvement")
            return False

    except Exception as e:
        print(f"   💥 Exception: {e}")
        return False

    finally:
        if 'temp_file' in locals():
            os.unlink(temp_file)


if __name__ == "__main__":
    success = asyncio.run(test_fixed_nested_calls())

    if success:
        print(f"\n🎉 NESTED FUNCTION CALL DETECTION FIXED!")
    else:
        print(f"\n⚠️  Still working on nested function call detection.")