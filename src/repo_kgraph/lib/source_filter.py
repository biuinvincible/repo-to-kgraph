"""
Smart source code filtering for knowledge graph extraction.

This module implements industry best practices for identifying and including
only actual source code while excluding dependencies, virtual environments,
tests, build artifacts, and other non-source files.
"""

import os
import re
from pathlib import Path
from typing import List, Set, Tuple, Optional, Callable
import fnmatch

try:
    from gitignore_parser import parse_gitignore
    GITIGNORE_AVAILABLE = True
except ImportError:
    GITIGNORE_AVAILABLE = False
    parse_gitignore = None


class SourceCodeFilter:
    """
    Filter that identifies actual source code vs dependencies, tests, etc.

    Based on industry best practices from gitignore patterns, code analysis tools,
    and static analysis exclusion patterns.
    """

    # Virtual environments and package management
    DEPENDENCY_PATTERNS = {
        # Python
        'venv/', '.venv/', 'env/', 'ENV/', '__pycache__/', '*.egg-info/',
        'site-packages/', 'pip-log.txt', 'pip-delete-this-directory.txt',

        # Node.js
        'node_modules/', 'npm-debug.log*', 'yarn-debug.log*', 'yarn-error.log*',
        '.npm/', '.yarn/', 'bower_components/',

        # Ruby
        'vendor/', '.bundle/', 'Gemfile.lock',

        # PHP
        'vendor/', 'composer.lock',

        # Java/Scala
        'target/', '.m2/', '.gradle/', 'gradle/',

        # C/C++
        'build/', 'cmake-build-*/', '.cmake/',

        # .NET
        'bin/', 'obj/', 'packages/', '.nuget/',

        # Go
        'vendor/', 'go.sum',

        # Rust
        'target/', 'Cargo.lock',

        # General package managers
        '.pnpm-store/', 'jspm_packages/', 'typings/',
    }

    # Test directories and files
    TEST_PATTERNS = {
        'test/', 'tests/', 'spec/', 'specs/', '__tests__/', 'testing/',
        '*test*/', '*spec*/', '*mock*/', 'fixtures/', 'mocks/',
        '*.test.py', '*.spec.py', '*.test.js', '*.spec.js', '*.test.ts', '*.spec.ts',
        'test_*.py', 'spec_*.py', '*_test.py', '*_spec.py',
        '*Test.java', '*Spec.java', '*Tests.java',
        'TestCase*.php', '*Test.php', '*TestCase.php',
        'test*.go', '*_test.go',
        'test*.c', 'test*.cpp', '*_test.c', '*_test.cpp',
    }

    # Build artifacts and generated files
    BUILD_PATTERNS = {
        'dist/', 'build/', 'out/', 'target/', 'bin/', 'obj/',
        '.next/', '.nuxt/', '.output/', 'public/build/',
        'coverage/', 'htmlcov/', '.coverage', '.nyc_output/',
        '.cache/', '.parcel-cache/', '.webpack-cache/',
        '*.min.js', '*.min.css', '*.bundle.js', '*.chunk.js',
        '*.map', '*.d.ts', '*.tsbuildinfo',
    }

    # Documentation and config
    DOC_CONFIG_PATTERNS = {
        'docs/', 'doc/', 'documentation/', 'man/', 'manual/',
        'examples/', 'samples/', 'demo/', 'demos/', 'tutorial/', 'tutorials/',
        '.github/', '.gitlab/', '.vscode/', '.idea/', '.vs/',
        '*.md', '*.txt', '*.rst', '*.adoc', 'README*', 'CHANGELOG*', 'LICENSE*',
        'Dockerfile*', 'docker-compose*.yml', '.dockerignore',
        'Makefile*', '*.mk', 'CMakeLists.txt', 'configure*',
        '*.yml', '*.yaml', '*.toml', '*.ini', '*.cfg', '*.conf', '*.config',
        'package.json', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
        'requirements*.txt', 'Pipfile*', 'poetry.lock', 'pyproject.toml',
        'setup.py', 'setup.cfg', 'MANIFEST.in', 'tox.ini',
        '.gitignore', '.gitattributes', '.editorconfig', '.pre-commit*',
    }

    # Version control and system files
    SYSTEM_PATTERNS = {
        '.git/', '.svn/', '.hg/', '.bzr/',
        '.DS_Store', 'Thumbs.db', 'desktop.ini',
        '*.tmp', '*.temp', '*.log', '*.bak', '*.swp', '*.swo',
        '.env*', '.local', '*.pid', '*.lock',
    }

    # Source code extensions by language
    SOURCE_EXTENSIONS = {
        'python': {'.py', '.pyx', '.pyi'},
        'javascript': {'.js', '.mjs', '.jsx'},
        'typescript': {'.ts', '.tsx'},
        'java': {'.java'},
        'kotlin': {'.kt', '.kts'},
        'scala': {'.scala'},
        'csharp': {'.cs'},
        'cpp': {'.cpp', '.cc', '.cxx', '.c++', '.hpp', '.hh', '.hxx', '.h++'},
        'c': {'.c', '.h'},
        'go': {'.go'},
        'rust': {'.rs'},
        'php': {'.php', '.php3', '.php4', '.php5', '.phtml'},
        'ruby': {'.rb', '.rbw'},
        'swift': {'.swift'},
        'objective-c': {'.m', '.mm'},
        'sql': {'.sql'},
        'shell': {'.sh', '.bash', '.zsh', '.fish'},
        'powershell': {'.ps1', '.psm1'},
        'lua': {'.lua'},
        'perl': {'.pl', '.pm'},
        'r': {'.r', '.R'},
        'matlab': {'.m'},
        'dart': {'.dart'},
        'elixir': {'.ex', '.exs'},
        'erlang': {'.erl', '.hrl'},
        'haskell': {'.hs', '.lhs'},
        'clojure': {'.clj', '.cljs', '.cljc'},
        'vue': {'.vue'},
        'angular': {'.component.ts', '.service.ts', '.module.ts'},
    }

    def __init__(self,
                 include_tests: bool = False,
                 include_docs: bool = False,
                 custom_excludes: Optional[List[str]] = None,
                 only_languages: Optional[List[str]] = None,
                 use_gitignore: bool = True):
        """
        Initialize source code filter.

        Args:
            include_tests: Whether to include test files
            include_docs: Whether to include documentation
            custom_excludes: Additional patterns to exclude
            only_languages: Only include these programming languages
            use_gitignore: Whether to use .gitignore files from target repo
        """
        self.include_tests = include_tests
        self.include_docs = include_docs
        self.custom_excludes = custom_excludes or []
        self.only_languages = only_languages
        self.use_gitignore = use_gitignore and GITIGNORE_AVAILABLE
        self.gitignore_matchers = []

        # Build exclude patterns
        self.exclude_patterns = set()
        self.exclude_patterns.update(self.DEPENDENCY_PATTERNS)
        self.exclude_patterns.update(self.BUILD_PATTERNS)
        self.exclude_patterns.update(self.SYSTEM_PATTERNS)

        if not include_tests:
            self.exclude_patterns.update(self.TEST_PATTERNS)

        if not include_docs:
            self.exclude_patterns.update(self.DOC_CONFIG_PATTERNS)

        self.exclude_patterns.update(self.custom_excludes)

        # Build allowed extensions
        self.allowed_extensions = set()
        if only_languages:
            for lang in only_languages:
                if lang.lower() in self.SOURCE_EXTENSIONS:
                    self.allowed_extensions.update(self.SOURCE_EXTENSIONS[lang.lower()])
        else:
            for extensions in self.SOURCE_EXTENSIONS.values():
                self.allowed_extensions.update(extensions)

    def should_exclude_path(self, file_path: str, repo_root: str) -> Tuple[bool, str]:
        """
        Check if a path should be excluded from analysis.

        Args:
            file_path: Full path to the file
            repo_root: Root directory of the repository

        Returns:
            Tuple of (should_exclude, reason)
        """
        try:
            # Convert to relative path from repo root
            rel_path = os.path.relpath(file_path, repo_root)
            if rel_path.startswith('..'):
                return True, "outside_repo"

            # Check gitignore patterns first (if available)
            if self.gitignore_matchers:
                gitignore_matched = False
                for gitignore_matcher in self.gitignore_matchers:
                    if gitignore_matcher(file_path):
                        gitignore_matched = True
                        break

                if gitignore_matched:
                    # Double-check that we're not excluding actual source code due to overly broad patterns
                    file_ext = Path(file_path).suffix.lower()
                    filename = os.path.basename(file_path)

                    # If it's a source file, be more conservative about gitignore exclusion
                    if file_ext in self.allowed_extensions:
                        # Allow source files unless they're clearly build artifacts
                        if any(filename.endswith(ext) for ext in ['.pyc', '.pyo', '.pyd', '.class', '.o', '.obj', '.min.js', '.min.css']):
                            return True, "gitignore_build_artifact"
                        # Otherwise include the source file despite gitignore match
                        # (gitignore might be too broad or meant for other tools)
                    else:
                        # Non-source files - respect gitignore
                        return True, "gitignore"

            # Check against our built-in exclude patterns
            for pattern in self.exclude_patterns:
                if self._matches_pattern(rel_path, pattern):
                    return True, f"matches_pattern:{pattern}"

            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext and file_ext not in self.allowed_extensions:
                return False, "not_source_extension"  # Don't exclude, just not prioritized

            return False, "include"

        except Exception as e:
            return True, f"error:{e}"

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches exclude pattern."""
        # Normalize paths
        path = path.replace('\\', '/')
        pattern = pattern.replace('\\', '/')

        # Directory patterns (ending with /)
        if pattern.endswith('/'):
            dir_pattern = pattern[:-1]
            path_parts = path.split('/')
            for i, part in enumerate(path_parts):
                if fnmatch.fnmatch(part, dir_pattern):
                    return True
            return False

        # File patterns
        if '/' in pattern:
            # Full path pattern
            return fnmatch.fnmatch(path, pattern)
        else:
            # Filename pattern
            filename = os.path.basename(path)
            return fnmatch.fnmatch(filename, pattern)

    def filter_files(self, repo_path: str) -> Tuple[List[str], dict]:
        """
        Filter repository files to source code only.

        Args:
            repo_path: Path to repository root

        Returns:
            Tuple of (included_files, stats)
        """
        repo_path = Path(repo_path).resolve()

        # Load gitignore files if enabled
        if self.use_gitignore:
            self._load_gitignore_files(str(repo_path))

        included_files = []
        stats = {
            'total_files': 0,
            'included_files': 0,
            'excluded_by_pattern': 0,
            'excluded_by_extension': 0,
            'excluded_by_gitignore': 0,
            'exclude_reasons': {},
            'languages_found': set(),
            'gitignore_files_found': len(self.gitignore_matchers),
        }

        # Walk through all files
        for root, dirs, files in os.walk(repo_path):
            # Skip excluded directories early for performance
            dirs[:] = [d for d in dirs if not self._should_exclude_dir(os.path.join(root, d), str(repo_path))]

            for file in files:
                file_path = os.path.join(root, file)
                stats['total_files'] += 1

                should_exclude, reason = self.should_exclude_path(file_path, str(repo_path))

                if should_exclude:
                    if reason == 'gitignore':
                        stats['excluded_by_gitignore'] += 1
                    elif reason.startswith('matches_pattern:'):
                        stats['excluded_by_pattern'] += 1
                    else:
                        stats['excluded_by_extension'] += 1

                    # Track exclude reasons
                    stats['exclude_reasons'][reason] = stats['exclude_reasons'].get(reason, 0) + 1
                else:
                    included_files.append(file_path)
                    stats['included_files'] += 1

                    # Track language
                    ext = Path(file_path).suffix.lower()
                    for lang, extensions in self.SOURCE_EXTENSIONS.items():
                        if ext in extensions:
                            stats['languages_found'].add(lang)
                            break

        return included_files, stats

    def _should_exclude_dir(self, dir_path: str, repo_root: str) -> bool:
        """Quick check if entire directory should be excluded."""
        rel_path = os.path.relpath(dir_path, repo_root)
        dir_name = os.path.basename(rel_path)

        # Check common directory exclusions
        exclude_dirs = {
            'node_modules', 'venv', '.venv', 'env', 'ENV', '__pycache__',
            'vendor', '.git', '.svn', 'build', 'dist', 'target',
            'coverage', 'htmlcov', '.pytest_cache', '.mypy_cache',
            'tests', 'test', 'spec', 'specs', '__tests__',
            'docs', 'doc', 'documentation', 'examples', 'demo',
            '.vscode', '.idea', '.vs', '.github', '.gitlab'
        }

        return dir_name in exclude_dirs

    def _load_gitignore_files(self, repo_root: str):
        """Load and parse .gitignore files from repository."""
        if not GITIGNORE_AVAILABLE:
            return

        self.gitignore_matchers = []
        gitignore_files_found = []

        # Only load the main .gitignore from repo root to avoid path conflicts
        main_gitignore = os.path.join(repo_root, '.gitignore')
        if os.path.exists(main_gitignore):
            try:
                matcher = parse_gitignore(main_gitignore, base_dir=repo_root)
                self.gitignore_matchers.append(matcher)
                gitignore_files_found.append('.gitignore')
            except Exception as e:
                print(f"Warning: Failed to parse {main_gitignore}: {e}")

        if gitignore_files_found:
            print(f"📋 Found .gitignore files: {', '.join(gitignore_files_found)}")
        else:
            print("📋 No .gitignore files found in repository")

    def get_filter_summary(self) -> str:
        """Get human-readable summary of filter configuration."""
        summary = ["Source Code Filter Configuration:"]
        summary.append(f"  Include tests: {self.include_tests}")
        summary.append(f"  Include docs: {self.include_docs}")
        summary.append(f"  Use .gitignore: {self.use_gitignore}")

        if self.only_languages:
            summary.append(f"  Languages: {', '.join(self.only_languages)}")
        else:
            summary.append(f"  Languages: All supported ({len(self.SOURCE_EXTENSIONS)})")

        summary.append(f"  Exclude patterns: {len(self.exclude_patterns)}")
        summary.append(f"  Custom excludes: {len(self.custom_excludes)}")

        if hasattr(self, 'gitignore_matchers'):
            summary.append(f"  Gitignore files: {len(self.gitignore_matchers)}")

        return '\n'.join(summary)


def create_source_only_filter(languages: Optional[List[str]] = None) -> SourceCodeFilter:
    """Create a filter configured for source code only (no tests, no docs)."""
    return SourceCodeFilter(
        include_tests=False,
        include_docs=False,
        only_languages=languages
    )


def create_comprehensive_filter(languages: Optional[List[str]] = None) -> SourceCodeFilter:
    """Create a filter that includes tests and minimal docs."""
    return SourceCodeFilter(
        include_tests=True,
        include_docs=False,
        only_languages=languages
    )