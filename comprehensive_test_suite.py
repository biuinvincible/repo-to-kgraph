#!/usr/bin/env python3
"""
Comprehensive test suite based on the testing checklist.
Focus on the most critical areas that could break in production.
"""

import asyncio
import tempfile
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fix_nested_calls import FixedTwoPassParser, EnhancedParseResult, DiagnosticLevel


class ComprehensiveTestSuite:
    """Test suite covering critical functionality from the checklist."""

    def __init__(self):
        self.parser = FixedTwoPassParser()
        self.test_results = []
        self.failures = []

    async def run_critical_tests(self):
        """Run the most critical tests that could break in production."""

        print("🧪 COMPREHENSIVE TEST SUITE")
        print("=" * 60)

        # Critical Tests (highest priority)
        await self._test_relationship_detection()
        await self._test_cross_file_relationships()
        await self._test_language_specific_features()
        await self._test_error_handling_edge_cases()
        await self._test_performance_scalability()

        # Summary
        self._print_test_summary()

    async def _test_relationship_detection(self):
        """Test 2: Critical relationship detection accuracy."""

        print("\n1️⃣ RELATIONSHIP DETECTION TESTS")
        print("-" * 40)

        test_cases = [
            {
                'name': 'Function Call Detection',
                'code': '''
def caller():
    return callee1() + callee2()

def callee1():
    return 1

def callee2():
    return callee1() * 2
''',
                'expected_calls': [('caller', 'callee1'), ('caller', 'callee2'), ('callee2', 'callee1')]
            },
            {
                'name': 'Method Call Detection',
                'code': '''
class MyClass:
    def method_a(self):
        return self.method_b() + external_func()

    def method_b(self):
        return 42

def external_func():
    return 1
''',
                'expected_calls': [('method_a', 'method_b'), ('method_a', 'external_func')]
            },
            {
                'name': 'Nested Function Calls',
                'code': '''
def outer():
    def inner():
        return helper()
    return inner()

def helper():
    return 1
''',
                'expected_calls': [('outer', 'inner'), ('inner', 'helper')]
            },
            {
                'name': 'Constructor Calls',
                'code': '''
class Parent:
    pass

class Child:
    def __init__(self):
        super().__init__()
        self.obj = Parent()

def create_objects():
    return Child()
''',
                'expected_calls': [('create_objects', 'Child'), ('__init__', 'Parent')]
            }
        ]

        for test_case in test_cases:
            success = await self._run_relationship_test(test_case)
            self.test_results.append({
                'category': 'Relationship Detection',
                'test': test_case['name'],
                'success': success
            })

    async def _run_relationship_test(self, test_case: dict) -> bool:
        """Run a single relationship detection test."""

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_case['code'])
                temp_file = f.name

            # Parse the file
            result = await self.parser.parse_file_enhanced(temp_file, "test_repo", tempfile.gettempdir())

            if not result.success:
                print(f"   ❌ {test_case['name']}: Parse failed")
                self.failures.append(f"{test_case['name']}: Parse failed")
                return False

            # Resolve relationships
            relationships = await self.parser.resolve_relationships([result])

            # Extract actual call patterns
            entity_names = {e.id: e.name for e in result.entities}
            actual_calls = set()

            for rel in relationships:
                source_name = entity_names.get(rel.source_entity_id, 'unknown')
                target_name = entity_names.get(rel.target_entity_id, 'unknown')
                actual_calls.add((source_name, target_name))

            # Check expected patterns
            expected_calls = set(test_case['expected_calls'])
            missing_calls = expected_calls - actual_calls
            extra_calls = actual_calls - expected_calls

            if missing_calls or extra_calls:
                print(f"   ❌ {test_case['name']}: Call pattern mismatch")
                if missing_calls:
                    print(f"      Missing: {missing_calls}")
                if extra_calls:
                    print(f"      Extra: {extra_calls}")
                self.failures.append(f"{test_case['name']}: Expected {expected_calls}, got {actual_calls}")
                return False
            else:
                print(f"   ✅ {test_case['name']}: All {len(expected_calls)} calls detected correctly")
                return True

        except Exception as e:
            print(f"   💥 {test_case['name']}: Exception - {e}")
            self.failures.append(f"{test_case['name']}: Exception - {e}")
            return False

        finally:
            if 'temp_file' in locals():
                os.unlink(temp_file)

    async def _test_cross_file_relationships(self):
        """Test 3: Cross-file relationship resolution."""

        print("\n2️⃣ CROSS-FILE RELATIONSHIP TESTS")
        print("-" * 40)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Create file A
                file_a = temp_path / "module_a.py"
                file_a.write_text('''
from module_b import function_b
from module_c import ClassC

def function_a():
    return function_b() + ClassC().method()

class ClassA:
    def method(self):
        return function_b()
''')

                # Create file B
                file_b = temp_path / "module_b.py"
                file_b.write_text('''
def function_b():
    return 42

def unused_function():
    return 0
''')

                # Create file C
                file_c = temp_path / "module_c.py"
                file_c.write_text('''
class ClassC:
    def method(self):
        return 1

    def other_method(self):
        return self.method()
''')

                # Parse all files
                files = [file_a, file_b, file_c]
                results = []

                for file_path in files:
                    result = await self.parser.parse_file_enhanced(str(file_path), "test_repo", str(temp_path))
                    results.append(result)

                # Check individual file parsing
                successful_parses = sum(1 for r in results if r.success)
                if successful_parses != len(files):
                    print(f"   ❌ Cross-file parsing: Only {successful_parses}/{len(files)} files parsed")
                    self.failures.append(f"Cross-file parsing: Only {successful_parses}/{len(files)} files succeeded")
                    return False

                # Resolve cross-file relationships
                relationships = await self.parser.resolve_relationships(results)

                # Expected cross-file relationships
                expected_cross_file = [
                    ('function_a', 'function_b'),  # A -> B
                    ('function_a', 'ClassC'),      # A -> C
                    ('method', 'function_b'),      # A.ClassA.method -> B
                ]

                # Build entity lookup
                all_entities = []
                for result in results:
                    all_entities.extend(result.entities)

                entity_lookup = {}
                for entity in all_entities:
                    entity_lookup[entity.id] = entity

                # Check relationships
                found_cross_file = []
                for rel in relationships:
                    source = entity_lookup.get(rel.source_entity_id)
                    target = entity_lookup.get(rel.target_entity_id)

                    if source and target and source.file_path != target.file_path:
                        found_cross_file.append((source.name, target.name))

                print(f"   📊 Found {len(found_cross_file)} cross-file relationships")
                print(f"   📊 Expected {len(expected_cross_file)} cross-file relationships")

                success = len(found_cross_file) >= len(expected_cross_file) // 2  # At least half
                if success:
                    print(f"   ✅ Cross-file relationships: Acceptable coverage")
                else:
                    print(f"   ❌ Cross-file relationships: Poor coverage")
                    self.failures.append(f"Cross-file relationships: Only {len(found_cross_file)} found")

                self.test_results.append({
                    'category': 'Cross-file Relationships',
                    'test': 'Multi-file dependency resolution',
                    'success': success
                })

                return success

        except Exception as e:
            print(f"   💥 Cross-file test exception: {e}")
            self.failures.append(f"Cross-file test exception: {e}")
            return False

    async def _test_language_specific_features(self):
        """Test 4: Language-specific parsing features."""

        print("\n3️⃣ LANGUAGE-SPECIFIC FEATURE TESTS")
        print("-" * 40)

        test_cases = [
            {
                'name': 'Python Decorators',
                'code': '''
@property
def getter(self):
    return self._value

@staticmethod
def static_func():
    return 1

@classmethod
def class_func(cls):
    return cls()
''',
                'expected_entities': 3
            },
            {
                'name': 'Python Async/Await',
                'code': '''
async def async_function():
    result = await other_async()
    return result

async def other_async():
    return 42
''',
                'expected_entities': 2,
                'expected_calls': [('async_function', 'other_async')]
            },
            {
                'name': 'Python Context Managers',
                'code': '''
class MyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def use_context():
    with MyContext() as ctx:
        return ctx
''',
                'expected_entities': 4  # class + 3 methods
            },
            {
                'name': 'Python Generators',
                'code': '''
def generator_func():
    for i in range(10):
        yield i

def list_comprehension():
    return [x for x in generator_func()]
''',
                'expected_entities': 2,
                'expected_calls': [('list_comprehension', 'generator_func')]
            }
        ]

        for test_case in test_cases:
            success = await self._run_language_feature_test(test_case)
            self.test_results.append({
                'category': 'Language Features',
                'test': test_case['name'],
                'success': success
            })

    async def _run_language_feature_test(self, test_case: dict) -> bool:
        """Run a language-specific feature test."""

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_case['code'])
                temp_file = f.name

            result = await self.parser.parse_file_enhanced(temp_file, "test_repo", tempfile.gettempdir())

            if not result.success:
                print(f"   ❌ {test_case['name']}: Parse failed")
                return False

            # Check entity count
            if 'expected_entities' in test_case:
                if len(result.entities) < test_case['expected_entities']:
                    print(f"   ❌ {test_case['name']}: Expected {test_case['expected_entities']} entities, got {len(result.entities)}")
                    return False

            # Check calls if specified
            if 'expected_calls' in test_case:
                relationships = await self.parser.resolve_relationships([result])
                entity_names = {e.id: e.name for e in result.entities}
                actual_calls = {(entity_names.get(r.source_entity_id), entity_names.get(r.target_entity_id))
                               for r in relationships}

                for expected_call in test_case['expected_calls']:
                    if expected_call not in actual_calls:
                        print(f"   ❌ {test_case['name']}: Missing call {expected_call}")
                        return False

            print(f"   ✅ {test_case['name']}: Feature detected correctly")
            return True

        except Exception as e:
            print(f"   💥 {test_case['name']}: Exception - {e}")
            return False

        finally:
            if 'temp_file' in locals():
                os.unlink(temp_file)

    async def _test_error_handling_edge_cases(self):
        """Test 7-8: Error handling and edge cases."""

        print("\n4️⃣ ERROR HANDLING & EDGE CASE TESTS")
        print("-" * 40)

        edge_cases = [
            {
                'name': 'Syntax Errors',
                'code': '''
def broken_function(
    # Missing closing parenthesis
    pass

class BrokenClass
    # Missing colon
    def method(self):
        return
''',
                'should_succeed': True,  # Should handle gracefully
                'min_diagnostics': 1
            },
            {
                'name': 'Unicode Content',
                'code': '''
def функция():
    """Функция на русском языке"""
    return "Привет мир! 🌍"

class Café:
    def méthode(self):
        return "résultat"
''',
                'should_succeed': True,
                'min_entities': 3
            },
            {
                'name': 'Very Long Lines',
                'code': 'def very_long_function(' + ', '.join(f'param{i}' for i in range(100)) + '):\n    return sum([' + ', '.join(f'param{i}' for i in range(100)) + '])',
                'should_succeed': True,
                'min_entities': 1
            },
            {
                'name': 'Empty File',
                'code': '',
                'should_succeed': True,
                'expected_entities': 0
            },
            {
                'name': 'Only Comments',
                'code': '''
# This is a comment
# Another comment
"""
Multi-line comment
"""
''',
                'should_succeed': True,
                'expected_entities': 0
            }
        ]

        for case in edge_cases:
            success = await self._run_edge_case_test(case)
            self.test_results.append({
                'category': 'Error Handling',
                'test': case['name'],
                'success': success
            })

    async def _run_edge_case_test(self, case: dict) -> bool:
        """Run an edge case test."""

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(case['code'])
                temp_file = f.name

            result = await self.parser.parse_file_enhanced(temp_file, "test_repo", tempfile.gettempdir())

            # Check if it should succeed
            if case['should_succeed'] and not result.success:
                print(f"   ❌ {case['name']}: Should succeed but failed")
                return False

            if not case['should_succeed'] and result.success:
                print(f"   ❌ {case['name']}: Should fail but succeeded")
                return False

            # Check specific requirements
            if 'min_entities' in case and len(result.entities) < case['min_entities']:
                print(f"   ❌ {case['name']}: Expected at least {case['min_entities']} entities, got {len(result.entities)}")
                return False

            if 'expected_entities' in case and len(result.entities) != case['expected_entities']:
                print(f"   ❌ {case['name']}: Expected {case['expected_entities']} entities, got {len(result.entities)}")
                return False

            if 'min_diagnostics' in case and len(result.diagnostics) < case['min_diagnostics']:
                print(f"   ❌ {case['name']}: Expected at least {case['min_diagnostics']} diagnostics, got {len(result.diagnostics)}")
                return False

            print(f"   ✅ {case['name']}: Handled correctly")
            return True

        except Exception as e:
            if case['should_succeed']:
                print(f"   ❌ {case['name']}: Unexpected exception - {e}")
                return False
            else:
                print(f"   ✅ {case['name']}: Expected exception - {e}")
                return True

        finally:
            if 'temp_file' in locals():
                os.unlink(temp_file)

    async def _test_performance_scalability(self):
        """Test 10: Performance and scalability."""

        print("\n5️⃣ PERFORMANCE & SCALABILITY TESTS")
        print("-" * 40)

        # Test with increasingly large files
        size_tests = [
            {'name': 'Small File (100 lines)', 'lines': 100},
            {'name': 'Medium File (1000 lines)', 'lines': 1000},
            {'name': 'Large File (5000 lines)', 'lines': 5000},
        ]

        for size_test in size_tests:
            success = await self._run_performance_test(size_test)
            self.test_results.append({
                'category': 'Performance',
                'test': size_test['name'],
                'success': success
            })

    async def _run_performance_test(self, size_test: dict) -> bool:
        """Run a performance test with generated code."""

        try:
            # Generate code of specified size
            lines = []
            lines.append("# Generated test file")

            for i in range(size_test['lines'] // 10):
                lines.append(f"def function_{i}():")
                lines.append(f"    '''Function {i} for testing'''")
                lines.append(f"    x = {i}")
                lines.append(f"    y = x * 2")
                lines.append(f"    return function_{(i+1) % (size_test['lines'] // 10)}() if x > 5 else y")
                lines.append("")

            code = '\n'.join(lines)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Time the parsing
            start_time = time.time()
            result = await self.parser.parse_file_enhanced(temp_file, "test_repo", tempfile.gettempdir())
            parse_time = time.time() - start_time

            # Performance thresholds (adjust as needed)
            max_time_seconds = size_test['lines'] / 1000  # 1 second per 1000 lines
            max_time_seconds = max(0.1, max_time_seconds)  # At least 100ms

            if not result.success:
                print(f"   ❌ {size_test['name']}: Parse failed")
                return False

            if parse_time > max_time_seconds:
                print(f"   ❌ {size_test['name']}: Too slow ({parse_time:.2f}s > {max_time_seconds:.2f}s)")
                return False

            entities_per_second = len(result.entities) / parse_time if parse_time > 0 else 0

            print(f"   ✅ {size_test['name']}: {parse_time:.2f}s, {len(result.entities)} entities ({entities_per_second:.0f} entities/sec)")
            return True

        except Exception as e:
            print(f"   💥 {size_test['name']}: Exception - {e}")
            return False

        finally:
            if 'temp_file' in locals():
                os.unlink(temp_file)

    def _print_test_summary(self):
        """Print comprehensive test summary."""

        print(f"\n🎯 TEST SUMMARY")
        print("=" * 50)

        # Group by category
        categories = {}
        for result in self.test_results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = {'total': 0, 'passed': 0}
            categories[cat]['total'] += 1
            if result['success']:
                categories[cat]['passed'] += 1

        # Print category summaries
        total_tests = len(self.test_results)
        total_passed = sum(1 for r in self.test_results if r['success'])

        for category, stats in categories.items():
            passed = stats['passed']
            total = stats['total']
            percentage = (passed / total * 100) if total > 0 else 0
            status = "✅" if passed == total else "⚠️" if passed > total // 2 else "❌"
            print(f"   {status} {category}: {passed}/{total} ({percentage:.0f}%)")

        print(f"\n📊 OVERALL: {total_passed}/{total_tests} ({total_passed/total_tests*100:.0f}%) tests passed")

        # Show failures
        if self.failures:
            print(f"\n❌ FAILURES ({len(self.failures)}):")
            for failure in self.failures[:10]:  # Show first 10
                print(f"   • {failure}")
            if len(self.failures) > 10:
                print(f"   ... and {len(self.failures) - 10} more")

        # Assessment
        if total_passed == total_tests:
            print(f"\n🎉 EXCELLENT: All tests passed! System is robust.")
            assessment = "EXCELLENT"
        elif total_passed >= total_tests * 0.8:
            print(f"\n✅ GOOD: Most tests passed. Minor issues to address.")
            assessment = "GOOD"
        elif total_passed >= total_tests * 0.6:
            print(f"\n⚠️  FAIR: Several issues found. Needs improvement.")
            assessment = "FAIR"
        else:
            print(f"\n❌ POOR: Major issues found. System needs significant work.")
            assessment = "POOR"

        return {
            'total_tests': total_tests,
            'passed_tests': total_passed,
            'failed_tests': total_tests - total_passed,
            'pass_rate': total_passed / total_tests * 100,
            'assessment': assessment,
            'categories': categories
        }


async def main():
    """Run the comprehensive test suite."""

    suite = ComprehensiveTestSuite()
    await suite.run_critical_tests()


if __name__ == "__main__":
    asyncio.run(main())